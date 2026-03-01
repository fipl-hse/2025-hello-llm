"""
Laboratory work.

Fine-tuning Large Language Models for a downstream task.
"""

# pylint: disable=too-few-public-methods, undefined-variable, duplicate-code, unused-argument, too-many-arguments

from pathlib import Path
from typing import Callable, cast, Iterable, Sequence

import evaluate
import pandas as pd
import torch
from datasets import load_dataset
from pandas import DataFrame
from peft import get_peft_model, LoraConfig, PeftConfig
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

from core_utils.llm.llm_pipeline import AbstractLLMPipeline
from core_utils.llm.metrics import Metrics
from core_utils.llm.raw_data_importer import AbstractRawDataImporter
from core_utils.llm.raw_data_preprocessor import AbstractRawDataPreprocessor, ColumnNames
from core_utils.llm.sft_pipeline import AbstractSFTPipeline
from core_utils.llm.task_evaluator import AbstractTaskEvaluator
from core_utils.llm.time_decorator import report_time
from core_utils.project.lab_settings import SFTParams


class RawDataImporter(AbstractRawDataImporter):
    """
    Custom implementation of data importer.
    """

    @report_time
    def obtain(self) -> None:
        """
        Import dataset.
        """
        self._raw_data = load_dataset(self._hf_name, split="test").to_pandas()

        if not isinstance(self._raw_data, pd.DataFrame):
            raise TypeError("Downloaded dataset is not pd.DataFrame")


class RawDataPreprocessor(AbstractRawDataPreprocessor):
    """
    Custom implementation of data preprocessor.
    """

    def analyze(self) -> dict:
        """
        Analyze preprocessed dataset.

        Returns:
            dict: dataset key properties.
        """
        data = self._raw_data.dropna()[["report"]]
        lengths = data.apply(lambda x: x.str.len())

        return {
            "dataset_number_of_samples": len(self._raw_data),
            "dataset_columns": len(self._raw_data.columns),
            "dataset_duplicates": int(self._raw_data.duplicated().sum()),
            "dataset_empty_rows": int(self._raw_data.isna().any(axis=1).sum()),
            "dataset_sample_min_len": int(lengths.min().min()),
            "dataset_sample_max_len": int(lengths.max().max()),
        }

    @report_time
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.
        """
        self._data = self._raw_data.rename(
            columns={"report": ColumnNames.SOURCE.value, "summary": ColumnNames.TARGET.value}
        ).reset_index(drop=True)


class TaskDataset(Dataset):
    """
    A class that converts pd.DataFrame to Dataset and works with it.
    """

    def __init__(self, data: pd.DataFrame) -> None:
        """
        Initialize an instance of TaskDataset.

        Args:
            data (pandas.DataFrame): Original data
        """
        self._data = data

    def __len__(self) -> int:
        """
        Return the number of items in the dataset.

        Returns:
            int: The number of items in the dataset
        """
        return len(self._data)

    def __getitem__(self, index: int) -> tuple[str, ...]:
        """
        Retrieve an item from the dataset by index.

        Args:
            index (int): Index of sample in dataset

        Returns:
            tuple[str, ...]: The item to be received
        """
        return (self._data[str(ColumnNames.SOURCE.value)][index],)

    @property
    def data(self) -> DataFrame:
        """
        Property with access to preprocessed DataFrame.

        Returns:
            pandas.DataFrame: Preprocessed DataFrame
        """
        return self._data


def tokenize_sample(
    sample: pd.Series, tokenizer: AutoTokenizer, max_length: int
) -> dict[str, torch.Tensor]:
    """
    Tokenize sample.

    Args:
        sample (pandas.Series): sample from a dataset
        tokenizer (transformers.models.auto.tokenization_auto.AutoTokenizer): Tokenizer to tokenize
            original data
        max_length (int): max length of sequence

    Returns:
        dict[str, torch.Tensor]: Tokenized sample
    """
    source_tokens = tokenizer(
        sample[ColumnNames.SOURCE.value],
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )

    target_tokens = tokenizer(
        sample[ColumnNames.TARGET.value],
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )

    return {
        "input_ids": source_tokens["input_ids"],
        "attention_mask": source_tokens["attention_mask"],
        "labels": target_tokens["input_ids"],
    }


class TokenizedTaskDataset(Dataset):
    """
    A class that converts pd.DataFrame to Dataset and works with it.
    """

    def __init__(self, data: pd.DataFrame, tokenizer: AutoTokenizer, max_length: int) -> None:
        """
        Initialize an instance of TaskDataset.

        Args:
            data (pandas.DataFrame): Original data
            tokenizer (transformers.models.auto.tokenization_auto.AutoTokenizer): Tokenizer to
                tokenize the dataset
            max_length (int): max length of a sequence
        """
        self._data = list(
            data.apply(lambda sample: tokenize_sample(sample, tokenizer, max_length), axis=1)
        )

    def __len__(self) -> int:
        """
        Return the number of items in the dataset.

        Returns:
            int: The number of items in the dataset
        """
        return len(self._data)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """
        Retrieve an item from the dataset by index.

        Args:
            index (int): Index of sample in dataset

        Returns:
            dict[str, torch.Tensor]: An element from the dataset
        """
        return dict(self._data[index])


class LLMPipeline(AbstractLLMPipeline):
    """
    A class that initializes a model, analyzes its properties and infers it.
    """

    def __init__(
        self, model_name: str, dataset: TaskDataset, max_length: int, batch_size: int, device: str
    ) -> None:
        """
        Initialize an instance.

        Args:
            model_name (str): The name of the pre-trained model.
            dataset (TaskDataset): The dataset to be used for translation.
            max_length (int): The maximum length of generated sequence.
            batch_size (int): The size of the batch inside DataLoader.
            device (str): The device for inference.
        """
        super().__init__(model_name, dataset, max_length, batch_size, device)
        self._model = T5ForConditionalGeneration.from_pretrained(model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

    def analyze_model(self) -> dict:
        """
        Analyze model computing properties.

        Returns:
            dict: Properties of a model
        """
        if not isinstance(self._model, torch.nn.Module):
            return {}
        config = self._model.config
        ids = torch.ones(1, cast(int, config.d_model), dtype=torch.long)
        result = summary(
            self._model,
            input_data={"input_ids": ids, "decoder_input_ids": ids},
            device=self._device,
            verbose=0,
        )
        return {
            "input_shape": [1, config.d_model],
            "embedding_size": config.d_model,
            "output_shape": result.summary_list[-1].output_size,
            "num_trainable_params": result.trainable_params,
            "vocab_size": config.vocab_size,
            "size": result.total_param_bytes,
            "max_context_length": config.max_length,
        }

    @report_time
    def infer_sample(self, sample: tuple[str, ...]) -> str | None:
        """
        Infer model on a single sample.

        Args:
            sample (tuple[str, ...]): The given sample for inference with model

        Returns:
            str | None: A prediction
        """
        return self._infer_batch([sample])[0]

    @report_time
    def infer_dataset(self) -> pd.DataFrame:
        """
        Infer model on a whole dataset.

        Returns:
            pd.DataFrame: Data with predictions
        """
        predictions = []
        dataloader = DataLoader(dataset=self._dataset, batch_size=self._batch_size)

        for batch in dataloader:
            prediction = self._infer_batch(batch)
            predictions.extend(prediction)

        result_data = pd.DataFrame(
            {
                ColumnNames.PREDICTION.value: predictions,
                ColumnNames.TARGET.value: self._dataset.data[ColumnNames.TARGET.value],
            }
        )
        return result_data

    @torch.no_grad()
    def _infer_batch(self, sample_batch: Sequence[tuple[str, ...]]) -> list[str]:
        """
        Infer single batch.

        Args:
            sample_batch (Sequence[tuple[str, ...]]): batch to infer the model

        Returns:
            list[str]: model predictions as strings
        """
        if self._model is None:
            return []

        tokens = self._tokenizer(
            sample_batch[0],
            max_length=self._max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self._device)

        outputs = self._model.generate(**tokens, max_length=self._max_length)
        decoded_output = self._tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return [str(text) for text in decoded_output]


class TaskEvaluator(AbstractTaskEvaluator):
    """
    A class that compares prediction quality using the specified metric.
    """

    def __init__(self, data_path: Path, metrics: Iterable[Metrics]) -> None:
        """
        Initialize an instance of Evaluator.

        Args:
            data_path (pathlib.Path): Path to predictions
            metrics (Iterable[Metrics]): List of metrics to check
        """
        super().__init__(data_path, metrics)

        self._data_path = data_path
        self._metrics = metrics

    def run(self) -> dict:
        """
        Evaluate the predictions against the references using the specified metric.

        Returns:
            dict: A dictionary containing information about the calculated metric
        """
        data = pd.read_csv(self._data_path)

        predictions = data[ColumnNames.PREDICTION.value].tolist()
        targets = data[ColumnNames.TARGET.value].tolist()

        results = {}

        for metric in self._metrics:
            if metric == Metrics.BLEU:
                bleu_metric = evaluate.load("bleu")
                targets_for_bleu = [[target] for target in targets]
                bleu_value = bleu_metric.compute(
                    predictions=predictions, references=targets_for_bleu
                )
                results["bleu"] = float(bleu_value["bleu"])

            elif metric == Metrics.ROUGE:
                rouge_metric = evaluate.load("rouge", seed=77)
                rouge_value = rouge_metric.compute(predictions=predictions, references=targets)
                results["rouge"] = float(rouge_value["rougeL"])

        return results


class SFTPipeline(AbstractSFTPipeline):
    """
    A class that initializes a model, fine-tuning.
    """

    def __init__(
        self,
        model_name: str,
        dataset: Dataset,
        sft_params: SFTParams,
        data_collator: Callable[[AutoTokenizer], torch.Tensor] | None = None,
    ) -> None:
        """
        Initialize an instance of ClassificationSFTPipeline.

        Args:
            model_name (str): The name of the pre-trained model.
            dataset (torch.utils.data.dataset.Dataset): The dataset used.
            sft_params (SFTParams): Fine-Tuning parameters.
            data_collator (Callable[[AutoTokenizer], torch.Tensor] | None, optional): processing
                                                                    batch. Defaults to None.
        """
        super().__init__(model_name, dataset, data_collator)

        self._sft_params = sft_params
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._lora_config = LoraConfig(
            r=sft_params.rank,
            lora_alpha=sft_params.alpha,
            lora_dropout=0.1,
            target_modules=sft_params.target_modules,
        )

    def run(self) -> None:
        """
        Fine-tune model.
        """
        model = get_peft_model(self._model, cast(PeftConfig, self._lora_config))
        finetuned_model_path = str(self._sft_params.finetuned_model_path)
        training_args = TrainingArguments(
            output_dir=finetuned_model_path,
            per_device_train_batch_size=self._sft_params.batch_size,
            max_steps=self._sft_params.max_fine_tuning_steps,
            learning_rate=self._sft_params.learning_rate,
            use_cpu=True,
            save_strategy="no",
            load_best_model_at_end=False,
        )

        trainer = Trainer(model=model, args=training_args, train_dataset=self._dataset)
        trainer.train()

        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(finetuned_model_path)

        ft_tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        ft_tokenizer.save_pretrained(finetuned_model_path)
