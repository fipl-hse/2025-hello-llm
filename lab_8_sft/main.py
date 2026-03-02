"""
Laboratory work.

Fine-tuning Large Language Models for a downstream task.
"""

import logging
from pathlib import Path

# pylint: disable=too-few-public-methods, undefined-variable, duplicate-code, unused-argument, too-many-arguments
from typing import Callable, Iterable, Sequence

import pandas as pd
import torch
from datasets import load_dataset
from pandas import DataFrame
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from core_utils.llm.llm_pipeline import AbstractLLMPipeline
from core_utils.llm.metrics import Metrics
from core_utils.llm.raw_data_importer import AbstractRawDataImporter
from core_utils.llm.raw_data_preprocessor import AbstractRawDataPreprocessor
from core_utils.llm.sft_pipeline import AbstractSFTPipeline
from core_utils.llm.task_evaluator import AbstractTaskEvaluator
from core_utils.llm.time_decorator import report_time
from core_utils.project.lab_settings import SFTParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RawDataImporter(AbstractRawDataImporter):
    """
    Custom implementation of data importer.
    """

    @report_time
    def obtain(self) -> None:
        """
        Import dataset.
        """
        self._raw_data = load_dataset(self._hf_name, split="train").to_pandas()

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

        df = self._raw_data

        text_col = df.select_dtypes(include=['object', 'string']).columns[0]
        lengths = df[text_col].str.len()
        return {
            "dataset_number_of_samples": len(df),
            "dataset_columns": len(df.columns),
            "dataset_duplicates": int(df.duplicated().sum()),
            "dataset_empty_rows": int(df.isna().sum().sum()),
            "dataset_sample_min_len": int(lengths.min()),
            "dataset_sample_max_len": int(lengths.max())
        }

    @report_time
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.
        """
        df = self._raw_data.copy()

        df = df.rename(columns={"toxic": "target", "neutral": "source"})
        df = df[["source", "target"]].dropna()
        df = df.drop_duplicates()
        df["target"] = df["target"].astype(int)
        df = df.reset_index(drop=True)

        self._data = df


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
        return str(row["source"]), int(row["target"])

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
    encoding = tokenizer(
        str(sample["source"]),
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        )

    return {
        "input_ids": encoding["input_ids"].squeeze(0),
        "attention_mask": encoding["attention_mask"].squeeze(0),
        "labels": torch.tensor(int(sample["target"]), dtype=torch.long),
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
        self._data = data.reset_index(drop=True)
        self._tokenizer = tokenizer
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
        sample = self._data.iloc[index]
        return tokenize_sample(sample, self._tokenizer, self._max_length)


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

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self._model.to(device)
        self._model.eval()

    def analyze_model(self) -> dict:
        """
        Analyze model computing properties.

        Returns:
            dict: Properties of a model
        """
        config = self._model.config
        total_params = sum(p.numel() for p in self._model.parameters())
        trainable_params = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        
        return {
            "embedding_size": config.max_position_embeddings, 
            "input_shape": {
                "input_ids": [1, config.max_position_embeddings],
                "attention_mask": [1, config.max_position_embeddings],
            },
            "max_context_length": 20, 
            "num_trainable_params": trainable_params,
            "output_shape": [1, config.num_labels],
            "size": total_params * 4,  
            "vocab_size": config.vocab_size,
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
        text = sample[0]

        encoding = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self._max_length,
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**encoding)

        pred = torch.argmax(outputs.logits, dim=1).item()
        return str(pred)

    @report_time
    def infer_dataset(self) -> pd.DataFrame:
        """
        Infer model on a whole dataset.

        Returns:
            pd.DataFrame: Data with predictions
        """
        predictions = []

        for text, label in self._dataset:
            pred = self.infer_sample((text, label))
            predictions.append(
                {"source": text, "target": label, "prediction": int(pred)}
            )

        return pd.DataFrame(predictions)


    @torch.no_grad()
    def _infer_batch(self, sample_batch: Sequence[tuple[str, ...]]) -> list[str]:
        """
        Infer single batch.

        Args:
            sample_batch (Sequence[tuple[str, ...]]): batch to infer the model

        Returns:
            list[str]: model predictions as strings
        """


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

    def run(self) -> dict:
        """
        Evaluate the predictions against the references using the specified metric.

        Returns:
            dict: A dictionary containing information about the calculated metric
        """


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
