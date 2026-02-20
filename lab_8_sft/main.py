"""
Laboratory work.

Fine-tuning Large Language Models for a downstream task.
"""
from pathlib import Path

# pylint: disable=too-few-public-methods, undefined-variable, duplicate-code, unused-argument, too-many-arguments
from typing import Callable, Iterable, Sequence, cast

import pandas as pd
import torch
from datasets import load_dataset
from pandas import DataFrame
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

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
        if self._raw_data is None or self._raw_data.empty:
            return {}

        dataset_number_of_samples = len(self._raw_data)
        dataset_columns = len(self._raw_data.columns)
        dataset_duplicates = int(self._raw_data.duplicated().sum())
        dataset_empty_rows = int(self._raw_data.isna().any(axis=1).sum())

        return {
            'dataset_number_of_samples': dataset_number_of_samples,
            'dataset_columns': dataset_columns,
            'dataset_duplicates': dataset_duplicates,
            'dataset_empty_rows': dataset_empty_rows,
            'dataset_sample_min_len': min(self._raw_data['text'].dropna().apply(len)),
            'dataset_sample_max_len': max(self._raw_data['text'].dropna().apply(len)),
        }

    @report_time
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.
        """
        self._data = self._raw_data.drop(["title", "date", "url"], axis=1)
        self._data = self._data.rename(
            columns={"text": ColumnNames.SOURCE, "summary": ColumnNames.TARGET}
        )

        self._data = self._data.reset_index()


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

    def __len__(self) -> int:
        """
        Return the number of items in the dataset.

        Returns:
            int: The number of items in the dataset
        """

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """
        Retrieve an item from the dataset by index.

        Args:
            index (int): Index of sample in dataset

        Returns:
            dict[str, torch.Tensor]: An element from the dataset
        """


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
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self._dataset = dataset
        self._device = device
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._batch_size = batch_size
        self._max_length = max_length

    def analyze_model(self) -> dict:
        """
        Analyze model computing properties.

        Returns:
            dict: Properties of a model
        """
        if self._model is None:
            raise ValueError("The model is not initialized")

        config = self._model.config
        ids = torch.ones(1, config.max_length, dtype=torch.long)

        decoder_input_ids = torch.ones(1, config.max_length, dtype=torch.long)

        tokens = {
            "input_ids": ids,
            "attention_mask": ids,
            "decoder_input_ids": decoder_input_ids,
            "use_cache": False
        }
        if not isinstance(self._model, Module):
            raise ValueError("The model has incompatible type")

        model_summary = summary(
            self._model,
            input_data=tokens,
            device=self._device,
            verbose=0
        )

        return {
            "input_shape": [1, config.hidden_size],
            "embedding_size": config.hidden_size,
            "output_shape": [1, config.hidden_size, config.vocab_size],
            "num_trainable_params": int(model_summary.trainable_params),
            "vocab_size": config.vocab_size,
            "size": int(model_summary.total_param_bytes),
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
        return self._infer_batch([sample])[0]

    @report_time
    def infer_dataset(self) -> pd.DataFrame:
        """
        Infer model on a whole dataset.

        Returns:
            pd.DataFrame: Data with predictions
        """
        if self._model is None:
            return pd.DataFrame()

        dataloader = DataLoader(self._dataset, self._batch_size)
        predictions = []
        targets = []

        for _, texts, labels in dataloader:

            texts = [str(x) for x in texts]
            labels = [int(x.item()) for x in labels]

            batch_predictions = self._infer_batch([(t,) for t in texts])

            predictions.extend(batch_predictions)
            targets.extend(labels)

        return pd.DataFrame({ColumnNames.TARGET: targets, ColumnNames.PREDICTION: predictions})

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

        source_texts = [str(sample[0]) for sample in sample_batch]

        tokens = self._tokenizer(source_texts, return_tensors="pt",
                                 padding=True, truncation=True, max_length=self._max_length)

        tokens = {k: v.to(self._device) for k, v in tokens.items()}

        output_ids = self._model.generate(
            **tokens,
            max_length=self._max_length,
        )

        predictions = self._tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=True
        )

        return cast(list[str], predictions)


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
