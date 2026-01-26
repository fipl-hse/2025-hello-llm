"""
Laboratory work.

Working with Large Language Models.
"""
# pylint: disable=too-few-public-methods, undefined-variable, too-many-arguments, super-init-not-called
import re
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import torch
from datasets import load_dataset
from pandas import DataFrame
from torch.utils.data import Dataset
from torchinfo import summary
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from core_utils.llm.llm_pipeline import AbstractLLMPipeline
from core_utils.llm.metrics import Metrics
from core_utils.llm.raw_data_importer import AbstractRawDataImporter
from core_utils.llm.raw_data_preprocessor import AbstractRawDataPreprocessor
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
        self._raw_data = load_dataset(
            self._hf_name,
            name="simplified",
            split="validation"
        ).to_pandas()

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
        df_copy = self._raw_data.copy()
        df_copy["labels"] = df_copy["labels"].apply(tuple)

        return {
           "dataset_number_of_samples": len(self._raw_data),
            "dataset_columns": len(self._raw_data.columns),
            "dataset_duplicates": df_copy.duplicated().sum(),
            "dataset_empty_rows": self._raw_data.isna().any(axis=1).sum(),
            "dataset_sample_min_len": min(len(str(row)) for row in self._raw_data['ru_text']),
            "dataset_sample_max_len": max(len(str(row)) for row in self._raw_data['ru_text'])
        }

    @report_time
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.
        """
        self._raw_data['labels'] = self._raw_data['labels'].apply(tuple)
        self._raw_data.drop(['id', 'text'], axis=1, inplace=True)

        exclude_values = {0, 4, 5, 6, 7, 8, 10, 12, 15, 18, 21, 22, 23}

        mapping_dict = {
            (1, 13, 17, 20): 1,
            (9, 16, 24, 25): 2,
            (14, 19): 3,
            (2, 3): 4,
            27: 7,
            26: 6
        }

        self._raw_data.loc[:, 'labels'] = (
            self._raw_data['labels']
            .apply(lambda x: next((v for v in x if v not in exclude_values), None))
            .map(lambda x: mapping_dict.get(x, 8))
            .astype(int)
        )

        self._raw_data.rename(columns={'labels': 'target', 'ru_text': 'source'}, inplace=True)
        self._raw_data = self._raw_data.query("target != 8")

        mapping_ordered = {
            1: 0,
            2: 1,
            3: 2,
            4: 3,
            6: 4,
            7: 5}
        self._raw_data['target'] = self._raw_data['target'].map(mapping_ordered)

        self._raw_data['source'] = self._raw_data['source'].apply(lambda x: re.sub(
            r'[^\w\s]', '', x.strip())
                                                                  )

        self._raw_data.reset_index(drop=True, inplace=True)

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
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
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
        ids = torch.ones(self._batch_size, self._max_length, dtype=torch.long)
        tokens = {"input_ids": ids, "attention_mask": ids}

        stats = summary(
            self._model,
            input_data=tokens,
            device=self._device,
            verbose=0
        )

        return {
            "input_shape": list(stats.input_size["input_ids"]),
            "embedding_size": config.emb_size,
            "output_shape": stats.summary_list[-1].output_size,
            "num_trainable_params": stats.trainable_params,
            "vocab_size": config.vocab_size,
            "size": stats.total_param_bytes,
            "max_context_length": config.max_position_embeddings
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
        if self._model is None:
            return None

        tokens = self._tokenizer(sample[0], return_tensors="pt")

        self._model.eval()

        with torch.no_grad():
            output = self._model(**tokens)

        predictions = torch.argmax(output.logits).item()

        labels = self._model.config.id2label

        return labels[predictions]

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
