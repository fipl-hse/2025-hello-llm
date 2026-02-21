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
from peft import get_peft_model, LoraConfig, PeftConfig
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
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
        self._raw_data = load_dataset(self._hf_name, split='train').to_pandas()

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
        text_lengths = self._raw_data.content.dropna().str.len()

        return {
            'dataset_number_of_samples': len(self._raw_data),
            'dataset_columns': len(self._raw_data.columns),
            'dataset_duplicates': int(self._raw_data.duplicated().sum()),
            'dataset_empty_rows': self._raw_data.isna().any(axis=1).sum(),
            'dataset_sample_min_len': int(text_lengths.min()),
            'dataset_sample_max_len': int(text_lengths.max())
        }

    @report_time
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.
        """
        self._data = self._raw_data.copy().rename(columns={'content': ColumnNames.SOURCE.value,
                                                           'grade3': ColumnNames.TARGET.value})
        self._data = self._data.dropna()
        self._data[ColumnNames.TARGET.value] = self._data[ColumnNames.TARGET.value].map(
                                                    {'Neutral': 2, 'Good': 1, 'Bad': 0})
        self._data = self._data.reset_index(drop=True)


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
        row = self._data.iloc[index]
        return row[ColumnNames.SOURCE.value], row[ColumnNames.TARGET.value]

    @property
    def data(self) -> pd.DataFrame:
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
    tokens = tokenizer(sample[ColumnNames.SOURCE.value], return_tensors='pt', padding='max_length',
                       truncation=True, max_length=max_length)

    return {
        'input_ids': tokens['input_ids'][0],
        'attention_mask': tokens['attention_mask'][0],
        'labels': sample[ColumnNames.TARGET.value]
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
        self._data = [tokenize_sample(row, tokenizer, max_length) for _, row in data.iterrows()]

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
        return self._data[index]


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
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model.eval()

    def analyze_model(self) -> dict:
        """
        Analyze model computing properties.

        Returns:
            dict: Properties of a model
        """
        if not isinstance(self._model, torch.nn.Module):
            return {}

        config = self._model.config

        ids = torch.ones((1, getattr(config, 'max_position_embeddings')),
                         dtype=torch.long, device=self._device)
        tokens = {'input_ids': ids, 'attention_mask': ids}

        result = summary(self._model, input_data=tokens, device=self._device, verbose=0)
        return({
            'input_shape': {k: list(v) for k, v in result.input_size.items()},
            'embedding_size': config.max_position_embeddings,
            'output_shape': result.summary_list[-1].output_size,
            'num_trainable_params': result.trainable_params,
            'vocab_size': config.vocab_size,
            'size': result.total_param_bytes,
            'max_context_length': config.max_length
        })

    @report_time
    def infer_sample(self, sample: tuple[str, ...]) -> str | None:
        """
        Infer model on a single sample.

        Args:
            sample (tuple[str, ...]): The given sample for inference with model

        Returns:
            str | None: A prediction
        """
        if not self._model or not self._tokenizer:
            return None

        return self._infer_batch([sample])[0]

    @report_time
    def infer_dataset(self) -> pd.DataFrame:
        """
        Infer model on a whole dataset.

        Returns:
            pd.DataFrame: Data with predictions
        """
        predictions, references = [], []

        for batch in DataLoader(self._dataset, self._batch_size):
            predictions.extend(self._infer_batch(batch))
            references.extend(batch[1].tolist())

        return pd.DataFrame({ColumnNames.TARGET.value: references,
                             ColumnNames.PREDICTION.value: predictions})

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

        inputs = self._tokenizer(sample_batch[0], return_tensors='pt', padding=True,
                                 truncation=True, max_length=self._max_length).to(self._device)

        with torch.no_grad():
            output = self._model(**inputs)

        predictions = [str(torch.argmax(prediction).item()) for prediction in output.logits]

        return ['0' if label == '2' else '2' if label == '0' else label for label in predictions]


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

        return {
            metric.value: evaluate.load(metric.value).compute(
                predictions=data[ColumnNames.PREDICTION.value],
                references=data[ColumnNames.TARGET.value], average='micro')[metric.value]
            for metric in self._metrics
        }


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
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._lora_config = LoraConfig(r=sft_params.rank, lora_alpha=sft_params.alpha,
                                       lora_dropout=0.1, target_modules=sft_params.target_modules)

    def run(self) -> None:
        """
        Fine-tune model.
        """
        model = get_peft_model(self._model, cast(PeftConfig, self._lora_config))

        finetuned_model_path = str(self._sft_params.finetuned_model_path)

        training_args = TrainingArguments(
            output_dir=finetuned_model_path,
            max_steps=self._sft_params.max_fine_tuning_steps,
            per_device_train_batch_size=self._sft_params.batch_size,
            learning_rate=self._sft_params.learning_rate,
            save_strategy='no',
            use_cpu=True,
            load_best_model_at_end=False,
            remove_unused_columns=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=self._dataset,
        )
        trainer.train()

        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(finetuned_model_path)
        self._tokenizer.save_pretrained(finetuned_model_path)
