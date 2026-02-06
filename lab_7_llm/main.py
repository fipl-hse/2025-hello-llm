"""
Laboratory work.

Working with Large Language Models.
"""

from pathlib import Path

# pylint: disable=too-few-public-methods, undefined-variable, too-many-arguments, super-init-not-called
from typing import Iterable, Sequence

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
        self._raw_data = load_dataset(self._hf_name, split='test').to_pandas()

        if not isinstance(self._raw_data, pd.DataFrame):
            raise TypeError("The downloaded dataset is not pd.DataFrame")


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
        df_copy = self._raw_data["EN"].dropna()
        df_len = df_copy.str.len()

        return {
            'dataset_number_of_samples': len(self._raw_data),
            'dataset_columns': len(self._raw_data.columns),
            'dataset_duplicates': self._raw_data.duplicated().sum(),
            'dataset_empty_rows': self._raw_data.isna().sum().sum(),
            'dataset_sample_min_len': df_len.min(),
            'dataset_sample_max_len': df_len.max(),
        }

    @report_time
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.
        """
        self._data = self._raw_data
        self._data = self._data.rename(columns={'EN': ColumnNames.SOURCE, 'DE': ColumnNames.TARGET})
        self._data = self._data.drop_duplicates(subset=[ColumnNames.SOURCE])
        self._data[ColumnNames.SOURCE] = ("Translate from English to German: " +
                                          self._data[ColumnNames.SOURCE].astype(str))
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
        return (str(self._data[ColumnNames.SOURCE].iloc[index]),
                str(self._data[ColumnNames.TARGET].iloc[index]))

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
        super().__init__(model_name, dataset, max_length, batch_size, device)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self._model.to(self._device)

    def analyze_model(self) -> dict:
        """
        Analyze model computing properties.

        Returns:
            dict: Properties of a model
        """
        if not isinstance(self._model, torch.nn.Module):
            return {}

        config = self._model.config
        batch = 1

        x = torch.zeros((self._batch_size, self._max_length), dtype=torch.long, device=self._device)
        stats = summary(
            self._model,
            input_data={"input_ids": x, "decoder_input_ids": x},
            verbose=0
        )

        return {
            "input_shape": [batch, config.d_model],
            "embedding_size": config.d_model,
            "output_shape": [batch, config.d_model, config.vocab_size],
            "num_trainable_params": int(stats.trainable_params),
            "vocab_size": config.vocab_size,
            "size": int(stats.total_param_bytes),
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
        if self._model is None:
            return None

        return self._infer_batch([sample])[0]

    @report_time
    def infer_dataset(self) -> pd.DataFrame:
        """
        Infer model on a whole dataset.

        Returns:
            pd.DataFrame: Data with predictions
        """
        preds = []
        refs = []

        for sources, targets in DataLoader(self._dataset, batch_size=self._batch_size):
            preds.extend(self._infer_batch(sources))
            refs.extend(targets)

        return pd.DataFrame({ColumnNames.TARGET.value: refs,
                             ColumnNames.PREDICTION.value: preds})

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
            return []

        first = sample_batch[0]
        if isinstance(first, str):
            text = list(sample_batch)
        else:
            text = [x[0] for x in sample_batch]

        tokens = self._tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._max_length,
        )

        tokens = {k: v.to(self._device) for k, v in tokens.items()}

        self._model.eval()
        out_ids = self._model.generate(**tokens, max_length=self._max_length)

        decoded = self._tokenizer.batch_decode(out_ids, skip_special_tokens=True)
        return [str(s) for s in decoded]


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
        data = pd.read_csv(self._data_path)

        return {
            metric.value: evaluate.load(metric.value).compute(
                predictions=data[ColumnNames.PREDICTION.value],
                references=data[ColumnNames.TARGET.value])["bleu"]
            for metric in self._metrics
        }
