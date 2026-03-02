"""
Laboratory work.

Working with Large Language Models.
"""

# pylint: disable=too-few-public-methods, undefined-variable, too-many-arguments, super-init-not-called
from pathlib import Path
from typing import cast, Iterable, Sequence

import evaluate
import pandas as pd
from datasets import load_dataset
from pandas import DataFrame
from torch.utils.data import DataLoader, Dataset

from core_utils.llm.llm_pipeline import AbstractLLMPipeline
from core_utils.llm.metrics import Metrics
from core_utils.llm.raw_data_importer import AbstractRawDataImporter
from core_utils.llm.raw_data_preprocessor import AbstractRawDataPreprocessor, ColumnNames
from core_utils.llm.task_evaluator import AbstractTaskEvaluator
from core_utils.llm.time_decorator import report_time

try:
    from transformers import AutoConfig, AutoTokenizer, XLMRobertaForSequenceClassification
except ImportError:
    print('Library "transformers" not installed. Failed to import.')

try:
    from torchinfo import summary  # type: ignore
except ImportError:
    print('Library "torchinfo" not installed. Failed to import.')

try:
    import torch
except ImportError:
    print('Library "torch" not installed. Failed to import.')


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
        self._raw_data = load_dataset(self._hf_name, split="validation").to_pandas()
        if not isinstance(self._raw_data, pd.DataFrame):
            raise TypeError("Downloaded dataset is not a pandas DataFrame")


class RawDataPreprocessor(AbstractRawDataPreprocessor):
    """
    A class that analyzes and preprocesses a dataset.
    """

    def __init__(self, raw_data: DataFrame, model_name: str | None = None) -> None:
        """
        Initialize an instance of RawDataPreprocessor.

        Args:
            raw_data (pandas.DataFrame): Original dataset
            model_name (str | None): Model name for label alignment; if set,
                targets are mapped to model's label indices
        """
        super().__init__(raw_data)
        self._model_name = model_name

    def analyze(self) -> dict:
        """
        Analyze a dataset.

        Returns:
            dict: Dataset key properties
        """
        df_non_empty = self._raw_data["text"].dropna()
        lengths = df_non_empty.str.len()

        return {
            "dataset_number_of_samples": len(self._raw_data),
            "dataset_columns": len(self._raw_data.columns),
            "dataset_duplicates": self._raw_data.duplicated().sum(),
            "dataset_empty_rows": self._raw_data.isnull().all(axis=1).sum(),
            "dataset_sample_min_len": lengths.min(),
            "dataset_sample_max_len": lengths.max(),
        }

    @report_time
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.
        """
        self._data = self._raw_data.rename(
            columns={
                "text": ColumnNames.SOURCE.value,
                "labels": ColumnNames.TARGET.value,
            }
        )

        if self._model_name is not None:
            config = AutoConfig.from_pretrained(self._model_name)
            if hasattr(config, "label2id") and config.label2id:
                label2id = {str(k): int(v) for k, v in config.label2id.items()}
            else:
                id2label = config.id2label
                label2id = {str(v): int(k) for k, v in id2label.items()}
            target_col = ColumnNames.TARGET.value
            self._data[target_col] = self._data[target_col].map(label2id)
        else:
            target_col = ColumnNames.TARGET.value
            unique_labels = sorted(self._data[target_col].unique())
            label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
            self._data[target_col] = self._data[target_col].map(label_mapping)

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
        row = self._data.iloc[index]
        return (
            row[ColumnNames.SOURCE.value],
            row[ColumnNames.TARGET.value],
        )

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
        self._model = XLMRobertaForSequenceClassification.from_pretrained(
            model_name)

        self._model.to(self._device)

    def analyze_model(self) -> dict:
        """
        Analyze model computing properties.

        Returns:
            dict: Properties of a model
        """
        assert self._model is not None
        config = self._model.config
        stats = summary(
            cast(torch.nn.Module, self._model),
            input_size=(self._batch_size, self._max_length),
            device=self._device,
            verbose=0,
            dtypes=[torch.long],
        )

        return {
            "input_shape": {
            "attention_mask": [1, config.max_position_embeddings],
            "input_ids": [1, config.max_position_embeddings],
        },
            "embedding_size": int(config.max_position_embeddings),
            "output_shape": [1, int(config.num_labels)],
            "num_trainable_params": int(stats.trainable_params),
            "vocab_size": int(config.vocab_size),
            "size": int(stats.total_params * 4),
            "max_context_length": int(config.num_labels),
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
        predictions = self._infer_batch([sample])
        return predictions[0] if predictions else None

    @report_time
    def infer_dataset(self) -> pd.DataFrame:
        """
        Infer model on a whole dataset.

        Returns:
            pd.DataFrame: Data with predictions
        """
        data_loader = DataLoader(
            self._dataset,
            batch_size=self._batch_size,
        )
        all_targets: list[int] = []
        all_predictions: list[str] = []
        for batch in data_loader:
            sources, targets = batch
            sample_batch = list(zip(sources, targets.tolist()))
            batch_predictions = self._infer_batch(sample_batch)
            all_targets.extend(targets.tolist())
            all_predictions.extend(batch_predictions)
        return pd.DataFrame({
            ColumnNames.TARGET: all_targets,
            ColumnNames.PREDICTION: all_predictions,
        })

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
        encoded = self._tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self._max_length,
        )
        encoded = {k: v.to(self._device) for k, v in encoded.items()}
        assert self._model is not None
        outputs = self._model(**encoded)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        return [str(p.item()) for p in predictions]


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
        references = data[ColumnNames.TARGET.value].tolist()
        predictions = data[ColumnNames.PREDICTION.value].astype(int).tolist()

        result: dict[str, float] = {}
        for metric in self._metrics:
            metric_fn = evaluate.load(str(metric))
            compute_kwargs: dict = {
                "references": references,
                "predictions": predictions,
            }
            if metric in (Metrics.F1, Metrics.PRECISION, Metrics.RECALL):
                compute_kwargs["average"] = "macro"
            computed = metric_fn.compute(**compute_kwargs)
            result[str(metric)] = float(computed[str(metric)])
        return result
