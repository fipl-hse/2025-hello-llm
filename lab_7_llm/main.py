"""
Laboratory work.

Working with Large Language Models.
"""

from pathlib import Path

# pylint: disable=too-few-public-methods, undefined-variable, too-many-arguments, super-init-not-called
from typing import Iterable, Sequence

import evaluate
import pandas as pd

# import pandas import DataFrame
import torch
from datasets import load_dataset
from pandas import DataFrame
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
from transformers import AutoModelForSequenceClassification, AutoTokenizer

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
        dataset = load_dataset(self._hf_name, split="train")
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
        source_column = "neutral"
        text_data = self._raw_data[source_column].dropna()

        return {
            "dataset_number_of_samples": len(self._raw_data),
            "dataset_columns": len(self._raw_data.columns),
            "dataset_duplicates": int(self._raw_data.duplicated().sum()),
            "dataset_empty_rows": int(self._raw_data.isnull().any(axis=1).sum()),
            "dataset_sample_min_len": int(text_data.str.len().min()),
            "dataset_sample_max_len": int(text_data.str.len().max()),
        }

    @report_time
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.
        """
        raw_data_df = self._raw_data.rename(
            columns={"toxic": ColumnNames.TARGET.value, "neutral": ColumnNames.SOURCE.value}
        )

        cleaned_df = raw_data_df.drop_duplicates()
        target_col = ColumnNames.TARGET.value
        cleaned_df[target_col] = cleaned_df[target_col].map({False: 0, True: 1})

        self._data = cleaned_df.reset_index(drop=True)


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
        target_val = row[ColumnNames.TARGET.value]
        target = str(int(float(target_val)))
        return source, target

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
        self._device = device
        self._batch_size = batch_size
        self._max_length = max_length
        self._dataset = dataset
        self._model_name = model_name

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def analyze_model(self) -> dict:
        """
        Analyze model computing properties.

        Returns:
            dict: Properties of a model
        """
        config = self._model.config

        input_ids = torch.ones(1, config.max_position_embeddings, dtype=torch.long).to(self._device)

        inputs = {"input_ids": input_ids, "attention_mask": input_ids}

        if not isinstance(self._model, torch.nn.Module):
            raise ValueError("Model must have type torch.nn.Module")

        model_stats = summary(self._model, input_data=inputs, verbose=0, device=self._device)

        return {
            "input_shape": {k: list(v.shape) for k, v in inputs.items()},
            "embedding_size": config.max_position_embeddings,
            "output_shape": model_stats.summary_list[-1].output_size,
            "num_trainable_params": model_stats.trainable_params,
            "vocab_size": config.vocab_size,
            "size": model_stats.total_param_bytes,
            "max_context_length": config.max_length,
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
        if not sample:
            return None

        text = sample[0]
        inputs = self._tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=self._max_length
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        if self._model is None:
            return None

        with torch.no_grad():
            outputs = self._model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=-1).item()

        return str(prediction)

    @report_time
    def infer_dataset(self) -> pd.DataFrame:
        """
        Infer model on a whole dataset.

        Returns:
            pd.DataFrame: Data with predictions
        """
        dataloader = DataLoader(self._dataset, batch_size=self._batch_size, shuffle=False)

        targets = []
        predictions = []

        for batch in dataloader:
            batch_sources, batch_targets = batch
            batch_texts = list(batch_sources)

            inputs = self._tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self._max_length,
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)
                preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

            targets.extend(batch_targets)
            predictions.extend(map(str, preds))

        return pd.DataFrame({"target": targets, "predictions": predictions})

    @torch.no_grad()
    def _infer_batch(self, sample_batch: Sequence[tuple[str, ...]]) -> list[str]:
        """
        Infer model on a single batch.

        Args:
            sample_batch (Sequence[tuple[str, ...]]): Batch to infer the model

        Returns:
            list[str]: Model predictions as strings
        """
        texts = [sample[0] for sample in sample_batch]

        inputs = self._tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=self._max_length
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        if self._model is None:
            return []

        outputs = self._model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

        return [str(pred) for pred in predictions]


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
        self.data_path = data_path
        self.metrics = metrics

    def run(self) -> dict:
        """
        Evaluate the predictions against the references using the specified metric.

        Returns:
            dict: A dictionary containing information about the calculated metric
        """
        predictions_df = pd.read_csv(self.data_path)

        target2pred = {
            ColumnNames.TARGET.value: predictions_df["target"].tolist(),
            ColumnNames.PREDICTION.value: predictions_df["predictions"].tolist(),
        }

        results = {}

        for metric in self.metrics:
            result = evaluate.load(str(metric)).compute(
                predictions=target2pred[ColumnNames.PREDICTION.value],
                references=target2pred[ColumnNames.TARGET.value],
                average="micro",
            )
            results.update(result)

        return results
