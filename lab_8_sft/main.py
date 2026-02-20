"""
Laboratory work.

Fine-tuning Large Language Models for a downstream task.
"""
from pathlib import Path

# pylint: disable=too-few-public-methods, undefined-variable, duplicate-code, unused-argument, too-many-arguments
from typing import Callable, Iterable, Sequence

import evaluate
import pandas as pd
import torch
from datasets import load_dataset
from pandas import DataFrame
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

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
        self._raw_data = load_dataset(
            self._hf_name,
            name="1.0.0",
            split="test"
        ).to_pandas()

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

        return {
            "dataset_number_of_samples": len(self._raw_data),
            "dataset_columns": len(self._raw_data.columns),
            "dataset_duplicates": self._raw_data.duplicated().sum(),
            "dataset_empty_rows": self._raw_data.isna().any(axis=1).sum(),
            "dataset_sample_min_len": min(len(str(row)) for row in self._raw_data['article']),
            "dataset_sample_max_len": max(len(str(row)) for row in self._raw_data['article'])
        }

    @report_time
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.
        """
        self._raw_data.drop(["id"], axis=1)
        self._raw_data.rename(columns={'highlights': ColumnNames.TARGET.value,
                                       'article': ColumnNames.SOURCE.value
                                       }, inplace=True)
        self._raw_data.drop_duplicates()
        self._raw_data[
            ColumnNames.SOURCE.value
        ] = self._raw_data[
            ColumnNames.SOURCE.value
        ].str.replace(
            "(CNN)",
            ""
        )
        self._raw_data.reset_index(inplace=True, drop=True)

        self._data = self._raw_data


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
        max_length=120
    )

    target_tokens = tokenizer(
        sample[ColumnNames.TARGET.value],
        padding="max_length",
        truncation=True,
        max_length=120
    )
    return {
        "input_ids": source_tokens["input_ids"],
        "attention_mask": source_tokens["attention_mask"],
        "labels": target_tokens["input_ids"]
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
        self._data = data
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self._max_length = max_length

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
        self._model_name = model_name
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._dataset = dataset
        self._max_length = max_length
        self._batch_size = batch_size
        self._device = device

    def analyze_model(self) -> dict:
        """
        Analyze model computing properties.

        Returns:
            dict: Properties of a model
        """
        if self._model is None:
            raise ValueError("The model is not initialized")

        max_input_length = self._model.config.n_positions

        input_ids = torch.ones(1, max_input_length, dtype=torch.long)
        attention_mask = torch.ones(1, max_input_length, dtype=torch.long)

        decoder_input_ids = torch.ones(1, max_input_length, dtype=torch.long)

        tokens = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "use_cache": False
        }

        model_stats = summary(
            self._model,
            input_data=tokens,
            verbose=0,
            device=self._device,
        )

        return {
            'input_shape': [1, max_input_length],
            'embedding_size': self._model.config.hidden_size,
            'output_shape': model_stats.summary_list[-1].output_size,
            'num_trainable_params': model_stats.trainable_params,
            'vocab_size': self._model.config.vocab_size,
            'size': model_stats.total_param_bytes,
            'max_context_length': self._model.config.max_length,
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
        dataloader = DataLoader(dataset=self._dataset, batch_size=self._batch_size)

        predictions = []

        for batch in dataloader:
            prediction = self._infer_batch(batch)
            predictions.extend(prediction)

        return pd.DataFrame(
            {
                str(ColumnNames.TARGET): self._dataset.data[str(ColumnNames.TARGET)],
                str(ColumnNames.PREDICTION): predictions
            }
        )

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
            raise ValueError("The model is not initialized")

        self._model.eval()

        tokens = self._tokenizer(
            list(sample_batch[0]),
            max_length=self._max_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
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
        self._data_path = data_path
        self._metrics = metrics

    def run(self) -> dict:
        """
        Evaluate the predictions against the references using the specified metric.

        Returns:
            dict: A dictionary containing information about the calculated metric
        """
        df = pd.read_csv(self._data_path)

        predictions = df[ColumnNames.PREDICTION.value].tolist()
        targets = df[ColumnNames.TARGET.value].tolist()

        results = {}

        for metric in self._metrics:
            if metric == Metrics.BLEU:
                bleu_metric = evaluate.load("bleu")
                targets_for_bleu = [[target] for target in targets]
                bleu_value = bleu_metric.compute(
                    predictions=predictions,
                    references=targets_for_bleu
                )
                results["bleu"] = float(bleu_value["bleu"])

            elif metric == Metrics.ROUGE:
                rouge_metric = evaluate.load("rouge", seed=77)
                rouge_value= rouge_metric.compute(
                    predictions=predictions,
                    references=targets
                )
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


    def run(self) -> None:
        """
        Fine-tune model.
        """
