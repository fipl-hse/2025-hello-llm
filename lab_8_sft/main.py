"""
Laboratory work.

Fine-tuning Large Language Models for a downstream task.
"""

# pylint: disable=too-few-public-methods, undefined-variable, duplicate-code, unused-argument, too-many-arguments
from pathlib import Path
from typing import Callable, Iterable, Sequence

import pandas as pd
import torch
from datasets import load_dataset
from evaluate import load
from pandas import DataFrame
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
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
            raise TypeError('The downloaded dataset is not pd.DataFrame')


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
        if self._raw_data is None:
            raise ValueError('The data is empty')

        text_lengths = self._raw_data.dropna(how='any')['text'].str.len()

        return {
            'dataset_number_of_samples': self._raw_data.shape[0],
            'dataset_columns': len(self._raw_data.columns),
            'dataset_duplicates': self._raw_data.duplicated(keep=False).sum(),
            'dataset_empty_rows': self._raw_data.isna().any(axis=1).sum(),
            'dataset_sample_min_len': text_lengths.min(),
            'dataset_sample_max_len': text_lengths.max(),
        }

    @report_time
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.
        """
        self._data = self._raw_data.rename(columns={'text': ColumnNames.SOURCE, 'toxic': ColumnNames.TARGET})


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
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
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
        max_input_length = self._model.config.max_position_embeddings

        input_ids = torch.ones(1, max_input_length, dtype=torch.long)
        attention_mask = torch.ones(1, max_input_length, dtype=torch.long)

        tokens = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        model_stats = summary(
            self._model,
            input_data=tokens,
            verbose=0,
            device=self._device,
        )

        return {
            'input_shape': {
                'attention_mask': [1, max_input_length],
                'input_ids': [1, max_input_length],
            },
            'embedding_size': self._model.config.max_position_embeddings,
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
        predictions = self._infer_batch([sample])

        if not predictions:
            return None
        return predictions[0]

    @report_time
    def infer_dataset(self) -> pd.DataFrame:
        """
        Infer model on a whole dataset.

        Returns:
            pd.DataFrame: Data with predictions
        """
        if self._model is None:
            return pd.DataFrame()

        dataloader = DataLoader(batch_size=self._batch_size, dataset=self._dataset)

        predictions = []
        targets = []

        for batch in dataloader:
            texts = list(batch[0])
            samples = [(text,) for text in texts]
            preds = self._infer_batch(samples)
            targets.extend([int(target) for target in batch[1]])
            predictions.extend(preds)

        return pd.DataFrame({"target": targets, "predictions": predictions})

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

        first_values = [sample[0] for sample in sample_batch]
        second_values = None
        if sample_batch and all(len(sample) > 1 and isinstance(sample[1], str) for sample in sample_batch):
            second_values = [sample[1] for sample in sample_batch]

        inputs = self._tokenizer(
            text=first_values,
            text_pair=second_values,
            truncation=True,
            padding=True,
            max_length=self._max_length,
            return_tensors='pt'
        ).to(self._device)

        self._model.eval()

        with torch.no_grad():
            logits = self._model(**inputs).logits
            predictions = torch.argmax(logits, dim=1).tolist()
            
        return [str(prediction) for prediction in predictions]


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
        if not self._data_path.exists():
            raise FileNotFoundError(f"No data found at {self._data_path}")

        predictions_df = pd.read_csv(self._data_path)

        predictions = predictions_df['predictions'].tolist()
        targets = predictions_df['target'].tolist()

        predictions = list(map(int, predictions))
        cleaned_targets = list(map(int, targets))

        result = {}

        for metric in self._metrics:
            metric_evaluate = load(str(metric))
            score = metric_evaluate.compute(
                predictions=predictions,
                references=cleaned_targets,
                average="micro"
            )

            result.update(score)
        return result


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
