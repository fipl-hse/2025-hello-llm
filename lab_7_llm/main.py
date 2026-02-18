"""
Laboratory work.

Working with Large Language Models.
"""

# pylint: disable=too-few-public-methods, undefined-variable, too-many-arguments, super-init-not-called
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd
import torch
from datasets import load_dataset
from evaluate import load
from pandas import DataFrame
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
from transformers import AlbertForSequenceClassification, AutoTokenizer

from core_utils.llm.llm_pipeline import AbstractLLMPipeline
from core_utils.llm.metrics import Metrics
from core_utils.llm.raw_data_importer import AbstractRawDataImporter
from core_utils.llm.raw_data_preprocessor import (
    AbstractRawDataPreprocessor,
    ColumnNames,
)
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
        self._raw_data: pd.DataFrame = load_dataset(self._hf_name, split="test").to_pandas()

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
        len_text = [len(row) for row in self._raw_data['text'].dropna()]

        return {
            'dataset_number_of_samples': len(self._raw_data),
            'dataset_columns': len(self._raw_data.columns),
            'dataset_duplicates': int(self._raw_data.duplicated().sum()),
            'dataset_empty_rows': int(self._raw_data.isna().sum().sum()),
            'dataset_sample_min_len': min(len_text),
            'dataset_sample_max_len': max(len_text),
        }

    @report_time
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.
        """
        self._data = self._raw_data.rename(columns={'text': ColumnNames.SOURCE,
                                                    'label': ColumnNames.TARGET})
        self._data = self._data.drop_duplicates()
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
        return tuple(self._data.loc[index])

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
        self, model_name: str, dataset: TaskDataset,
        max_length: int, batch_size: int, device: str
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
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AlbertForSequenceClassification.from_pretrained(model_name)

    def analyze_model(self) -> dict:
        """
        Analyze model computing properties.

        Returns:
            dict: Properties of a model
        """

        embeddings_length = self._model.config.max_position_embeddings
        ids = torch.ones(1, embeddings_length, dtype=torch.long)
        tokens = {"input_ids": ids, "attention_mask": ids}

        if not isinstance(self._model, Module):
            raise ValueError("The model has incompatible type")
        model_summary = summary(self._model, input_data=tokens,
                                device=self._device, verbose=0)

        input_shape = {}
        for key, value in model_summary.input_size.items():
            input_shape[key] = list(value)

        return {
             "input_shape" : input_shape,
             "embedding_size" : embeddings_length,
             "output_shape" : model_summary.summary_list[-1].output_size,
             "num_trainable_params" : model_summary.trainable_params,
             "vocab_size" : self._model.config.vocab_size,
             "size" : model_summary.total_param_bytes,
             "max_context_length" : self._model.config.max_length

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
        predictions = self._infer_batch([sample])

        return predictions[0]

    @report_time
    def infer_dataset(self) -> pd.DataFrame:
        """
        Infer model on a whole dataset.

        Returns:
            pd.DataFrame: Data with predictions
        """
        dataset_loader = DataLoader(self._dataset, self._batch_size)
        all_predictions = []
        all_targets = []

        for batch in dataset_loader:
            predictions = self._infer_batch(batch[0])
            all_predictions.extend(predictions)
            all_targets.extend(batch[1].tolist())

        return pd.DataFrame({'target': all_targets, 'predictions': all_predictions})

    @torch.no_grad()
    def _infer_batch(self, sample_batch: Sequence[tuple[str, ...]]) -> list[str]:
        """
        Infer model on a single batch.

        Args:
            sample_batch (Sequence[tuple[str, ...]]): Batch to infer the model

        Returns:
            list[str]: Model predictions as strings
        """
        if self._model is None:
            raise ValueError("The model is not initialized")

        samples = [sample[0] for sample in sample_batch]
        tokens = self._tokenizer(samples, return_tensors="pt",
                                 padding=True, truncation=True)

        self._model.eval()
        with torch.no_grad():
            output = self._model(**tokens)

        return [str(torch.argmax(pred).item()) for pred in output.logits]


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
        self._metrics = metrics
        self._data_path = data_path


    def run(self) -> dict:
        """
        Evaluate the predictions against the references using the specified metric.

        Returns:
            dict: A dictionary containing information about the calculated metric
        """

        data_pred = pd.read_csv(self._data_path)

        for metric in self._metrics:
            hf_metric = load(str(metric))
            results: dict[Any, Any]  = hf_metric.compute(
                                                predictions=list(data_pred['predictions']),
                                                references=list(data_pred['target']),
                                                average='weighted',
                                                )

        return results
