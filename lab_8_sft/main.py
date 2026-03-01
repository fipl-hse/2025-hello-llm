"""
Laboratory work.

Fine-tuning Large Language Models for a downstream task.
"""
from pathlib import Path
# pylint: disable=too-few-public-methods, undefined-variable, duplicate-code, unused-argument, too-many-arguments
from typing import Callable, Iterable, Sequence
import pandas as pd
import torch
from datasets import load_dataset
from pandas import DataFrame
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torchinfo import summary
from peft import LoraConfig

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
        dataset = load_dataset(self._hf_name, split="if_test")
        self._raw_data = dataset.to_pandas()

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
        source_column = "ru"
        text_data = self._raw_data[source_column].dropna()

        return {
            "dataset_number_of_samples": len(self._raw_data),
            "dataset_columns": len(self._raw_data.columns),
            "dataset_duplicates": int(self._raw_data.duplicated().sum()),
            "dataset_empty_rows": int(self._raw_data.isnull().any(axis=1).sum()),
            "dataset_sample_min_len": int(text_data.str.len().min()),
            "dataset_sample_max_len": int(text_data.str.len().max())
        }

    @report_time
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.
        """
        df = self._raw_data.copy()
        dropped_df = df.drop(columns=["ru_annotated", "styles"])

        renamed_df = dropped_df.rename(
            columns={"en": ColumnNames.TARGET.value, "ru": ColumnNames.SOURCE.value}
        )

        self._data = renamed_df.reset_index(drop=True)


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
        source = str(row[ColumnNames.SOURCE.value])
        target = str(row[ColumnNames.TARGET.value])
        return source, target

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
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.data = data

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
        self._device = device
        self._batch_size = batch_size
        self._max_length = max_length
        self._dataset = dataset
        self._model_name = model_name

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        self._model.to(self._device)
        self._model.eval()

    def analyze_model(self) -> dict:
        """
        Analyze model computing properties.

        Returns:
            dict: Properties of a model
        """
        config = self._model.config

        dummy_input = self._tokenizer("Артём Михайлович", return_tensors="pt").to(self._device)

        if not isinstance(self._model, torch.nn.Module):
            raise ValueError("Model must have type torch.nn.Module")

        model_stats = summary(self._model, input_data=dummy_input, verbose=0, device=self._device)

        return {
            "input_shape": model_stats.input_size,
            "embedding_size": config.d_model,
            "output_shape": model_stats.summary_list[-1].output_size,
            "num_trainable_params": model_stats.trainable_params,
            "vocab_size": config.vocab_size,
            "size": model_stats.total_param_bytes,
            "max_context_length": config.max_position_embeddings,
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

        text = sample[0]

        inputs = self._tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=self._max_length
        ).to(self._device)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self._model.generate(**inputs, max_length=self._max_length)

        decoded = self._tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        return decoded

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
        self.metrics = metrics
        self.data_path = data_path

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
        self.data_collator = data_collator
        self.sft_params = sft_params
        self.dataset = dataset
        self.model_name = model_name

    def run(self) -> None:
        """
        Fine-tune model.
        """
