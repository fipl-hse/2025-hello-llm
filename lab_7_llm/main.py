"""
Laboratory work.

Working with Large Language Models.
"""
from pathlib import Path
from typing import Iterable, Sequence
import pandas as pd
import torch
from datasets import load_dataset
from pandas import DataFrame
from torch.utils.data import Dataset
from torchinfo import summary
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from core_utils.llm.llm_pipeline import AbstractLLMPipeline
from core_utils.llm.metrics import Metrics
from core_utils.llm.raw_data_importer import AbstractRawDataImporter
from core_utils.llm.raw_data_preprocessor import AbstractRawDataPreprocessor, ColumnNames
from core_utils.llm.task_evaluator import AbstractTaskEvaluator
from core_utils.llm.time_decorator import report_time

# pylint: disable=too-few-public-methods, undefined-variable, too-many-arguments, super-init-not-called


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
        dataset = load_dataset(
                self._hf_name,
                split="train"
        )

        self._raw_data = dataset.to_pandas()

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
        if self._raw_data is None or self._raw_data.empty:
            return {}

        dataset_number_of_samples = len(self._raw_data)
        dataset_columns = len(self._raw_data.columns)

        dataset_duplicates = int(self._raw_data.duplicated().sum())

        dataset_empty_rows = int(self._raw_data.isnull().any(axis=1).sum())

        text_column = 'article_content'

        if text_column in self._raw_data.columns:
            non_empty_df = self._raw_data.dropna(subset=[text_column])

            text_lengths = non_empty_df[text_column].astype(str).str.len()

            if len(text_lengths) > 0:
                dataset_sample_min_len = int(text_lengths.min())
                dataset_sample_max_len = int(text_lengths.max())
            else:
                dataset_sample_min_len = 0
                dataset_sample_max_len = 0
        else:
            dataset_sample_min_len = 0
            dataset_sample_max_len = 0

        return {
            "dataset_number_of_samples": dataset_number_of_samples,
            "dataset_columns": dataset_columns,
            "dataset_duplicates": dataset_duplicates,
            "dataset_empty_rows": dataset_empty_rows,
            "dataset_sample_min_len": dataset_sample_min_len,
            "dataset_sample_max_len": dataset_sample_max_len
        }

    @report_time
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.
        """
        if self._raw_data is None:
            return

        processed_data = self._raw_data.copy()

        columns_to_drop = ['title', 'date', 'url']

        existing_columns = [col for col in columns_to_drop if col in processed_data.columns]
        if existing_columns:
            processed_data = processed_data.drop(columns=existing_columns)

        if 'article_content' in processed_data.columns:
            processed_data = processed_data.rename(
                columns={'article_content': ColumnNames.SOURCE}
            )

        if 'summary' in processed_data.columns:
            processed_data = processed_data.rename(
                columns={'summary': ColumnNames.TARGET}
            )

        processed_data = processed_data.reset_index(drop=True)

        if ColumnNames.SOURCE in processed_data.columns:
            processed_data = processed_data.dropna(subset=[ColumnNames.SOURCE])

        if ColumnNames.TARGET in processed_data.columns:
            processed_data = processed_data.dropna(subset=[ColumnNames.TARGET])

        self._data = processed_data


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
    def data(self) -> DataFrame:
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
        config = self._model.config

        input_ids = torch.ones((1, self._max_length), dtype=torch.long)
        attention_mask = torch.ones((1, self._max_length), dtype=torch.long)

        tokens = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

        stats = summary(
            self._model,
            input_data=tokens,
            device=self._device,
            verbose=0
        )

        input_shape_dict = {}
        for key, value in stats.input_size.items():
            input_shape_dict[key] = list(value)

        return {
            "input_shape": input_shape_dict,
            "embedding_size": config.max_position_embeddings,
            "output_shape": stats.summary_list[-1].output_size,
            "num_trainable_params": stats.trainable_params,
            "vocab_size": config.vocab_size,
            "size": stats.total_param_bytes,
            "max_context_length": config.max_length
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
        return str(self._infer_batch([sample])[0])

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
