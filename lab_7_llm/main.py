"""
Laboratory work.

Working with Large Language Models.
"""

# pylint: disable=too-few-public-methods, undefined-variable, too-many-arguments, super-init-not-called
from typing import Iterable, Sequence
import pandas as pd
import torch
from pathlib import Path
from datasets import load_dataset
from torch.utils.data import Dataset
from torchinfo import summary
from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

from core_utils.llm.llm_pipeline import AbstractLLMPipeline
from core_utils.llm.metrics import Metrics
from core_utils.llm.raw_data_importer import AbstractRawDataImporter
from core_utils.llm.raw_data_preprocessor import AbstractRawDataPreprocessor, ColumnNames
from core_utils.llm.task_evaluator import AbstractTaskEvaluator
from core_utils.llm.time_decorator import report_time

class RawDataImporter(AbstractRawDataImporter):
    """
    A class that imports the HuggingFace dataset.
    """

    @report_time
    def obtain(self) -> None:
        """
        Download a dataset.

        Raises:
            TypeError: In case of downloaded dataset is not pd.DataFrame
        """
        ds = load_dataset(self._hf_name, split='validation')
        self._raw_data = ds.to_pandas()
        if not isinstance(self._raw_data, pd.DataFrame):
            raise TypeError("Downloaded dataset is not pd.DataFrame")


class RawDataPreprocessor(AbstractRawDataPreprocessor):
    """
    A class that analyzes and preprocesses a dataset.
    """

    def analyze(self) -> dict:
        """
        Analyze a dataset.

        Returns:
            dict: Dataset key properties
        """
        df = self._raw_data
        result = {'dataset_number_of_samples': len(df),
                  'dataset_columns': len(df.columns),
                  'dataset_duplicates': df.duplicated().sum(),
                  'dataset_empty_rows': df.isna().any(axis=1).sum()}
        source = df['text'].dropna()
        result['dataset_sample_min_len'] = min(source.apply(len))
        result['dataset_sample_max_len'] = max(source.apply(len))
        return result

    @report_time
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.
        """
        transformed_df = self._raw_data
        classes ={
                "tat": "0",
                "rus": "1",
                "kir": "2",
                "krc": "3",
                "bak": "4",
                "sah": "5",
                "kaz": "6",
                "tyv": "7",
                "chv": "8"
            }
        transformed_df = transformed_df.rename(columns={'label': ColumnNames.TARGET, 'text': ColumnNames.SOURCE})
        transformed_df[ColumnNames.TARGET] = transformed_df[ColumnNames.TARGET].apply(lambda x: classes[x])
        self._data = transformed_df


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
        return tuple(self._data.iloc[index])

    @property
    def data(self) -> pd.DataFrame:
        """
        Property with access to preprocessed DataFrame.

        Returns:
            pandas.DataFrame: Preprocessed DataFrame
        """
        return self._data


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
            model_name (str): The name of the pre-trained model
            dataset (TaskDataset): The dataset used
            max_length (int): The maximum length of generated sequence
            batch_size (int): The size of the batch inside DataLoader
            device (str): The device for inference
        """
        self._model_name = model_name
        self._dataset = dataset
        self._max_length = max_length
        self._batch_size = batch_size
        self._device = device
        self._tokenizer = AutoTokenizer.from_pretrained("tatiana-merz/turkic-cyrillic-classifier")
        self._model = AutoModelForSequenceClassification.from_pretrained("tatiana-merz/turkic-cyrillic-classifier")

    def analyze_model(self) -> dict:
        """
        Analyze model computing properties.

        Returns:
            dict: Properties of a model
        """
        config = self._model.config
        ids = torch.ones(1, config.max_position_embeddings, dtype=torch.long)
        result = summary(self._model, input_data={"input_ids": ids, "attention_mask": ids}, device="cpu", verbose=0)

        analisys = {'input_shape': result.input_size,
                    'embedding_size': config.max_position_embeddings,
                    'output_shape': result.summary_list[-1].output_size,
                    'num_trainable_params': result.trainable_params,
                    'vocab_size': config.vocab_size,
                    'size': result.total_param_bytes,
                    'max_context_length': 20}

    @report_time
    def infer_sample(self, sample: tuple[str, ...]) -> str | None:
        """
        Infer model on a single sample.

        Args:
            sample (tuple[str, ...]): The given sample for inference with model

        Returns:
            str | None: A prediction
        """
        tokens = self._tokenizer(sample[0], return_tensors="pt")
        self._model.eval()
        with torch.no_grad():
            output = self._model(**tokens)
        predictions = torch.argmax(output.logits).item()
        return str(predictions)

    @report_time
    def infer_dataset(self) -> pd.DataFrame:
        """
        Infer model on a whole dataset.

        Returns:
            pd.DataFrame: Data with predictions
        """

    @torch.no_grad()
    def _infer_batch(self, sample_batch: Sequence[tuple[str, ...]]) -> list[str]:
        """
        Infer model on a single batch.

        Args:
            sample_batch (Sequence[tuple[str, ...]]): Batch to infer the model

        Returns:
            list[str]: Model predictions as strings
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
