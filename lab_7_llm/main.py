"""
Laboratory work.

Working with Large Language Models.
"""
# pylint: disable=too-few-public-methods, undefined-variable, too-many-arguments, super-init-not-called

from pathlib import Path
from typing import Iterable, Sequence

import datasets
import pandas as pd
import torch
from evaluate import load
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
from transformers import AutoTokenizer, BertForSequenceClassification

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
        self._raw_data = datasets.load_dataset(self._hf_name, split="validation",
                                               revision="refs/convert/parquet").to_pandas()

        if not isinstance(self._raw_data, pd.DataFrame):
            raise TypeError('The downloaded dataset is not a pandas.DataFrame.')


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
        analysis = {'dataset_number_of_samples': len(df),
                    'dataset_columns': 	len(df.columns),
                    'dataset_duplicates': len(df[df.duplicated()]),
                    'dataset_empty_rows': len(df[df.isna().any(axis=1)])}

        column = df['content'].dropna()
        analysis['dataset_sample_min_len'] = min(column.apply(len))
        analysis['dataset_sample_max_len'] = max(column.apply(len))
        return analysis

    @report_time
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.
        """
        preprocessed = self._raw_data[['content', 'sentiment']]
        preprocessed = preprocessed.rename(columns={"sentiment": ColumnNames.TARGET,
                                                    "content": ColumnNames.SOURCE})
        preprocessed[ColumnNames.TARGET] = (preprocessed[ColumnNames.TARGET].
                                            apply(lambda x: "1" if x == 'positive' else "2"))
        self._data = preprocessed


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
        super().__init__(model_name, dataset, max_length, batch_size, device)
        self._model = BertForSequenceClassification.from_pretrained(model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

    def analyze_model(self) -> dict:
        """
        Analyze model computing properties.

        Returns:
            dict: Properties of a model
        """
        if not isinstance(self._model, torch.nn.Module):
            return {}

        config = self._model.config
        ids = torch.ones(1, getattr(config, 'max_position_embeddings'), dtype=torch.long)
        tokens = {"input_ids": ids, "attention_mask": ids}
        stats = summary(self._model, input_data=tokens, device=self._device, verbose=0)
        analysis = {'input_shape': {k: list(v) for k, v in stats.input_size.items()},
                    'embedding_size': config.max_position_embeddings,
                    'output_shape': stats.summary_list[-1].output_size,
                    'num_trainable_params': stats.trainable_params,
                    'vocab_size': config.vocab_size,
                    'size': stats.total_param_bytes,
                    'max_context_length': self._model.config.max_length}
        return analysis

    @report_time
    def infer_sample(self, sample: tuple[str, ...]) -> str | None:
        """
        Infer model on a single sample.

        Args:
            sample (tuple[str, ...]): The given sample for inference with model

        Returns:
            str | None: A prediction
        """
        if not self._model or not isinstance(sample, tuple):
            return None
        return self._infer_batch([sample])[0]

    @report_time
    def infer_dataset(self) -> pd.DataFrame:
        """
        Infer model on a whole dataset.

        Returns:
            pd.DataFrame: Data with predictions
        """
        loader = DataLoader(self._dataset, self._batch_size)
        predictions = []
        references = []

        for texts, targets in loader:
            texts = [(text, ) for text in texts]
            predictions.extend(self._infer_batch(texts))
            references.extend([int(x) for x in targets])
        return pd.DataFrame({ColumnNames.TARGET: references,
                            ColumnNames.PREDICTION: predictions})

    @torch.no_grad()
    def _infer_batch(self, sample_batch: Sequence[tuple[str, ...]]) -> list[str]:
        """
        Infer model on a single batch.

        Args:
            sample_batch (Sequence[tuple[str, ...]]): Batch to infer the model

        Returns:
            list[str]: Model predictions as strings
        """
        if not self._model:
            return ['']

        inputs = self._tokenizer([sample[0] for sample in sample_batch],
                                 return_tensors="pt",
                                 padding=True,
                                 truncation=True,
                                 max_length=self._max_length)

        self._model.eval()
        output = [str(torch.argmax(prediction).item())
                  for prediction in self._model(**inputs).logits]
        return ["2" if label == "0" else label for label in output]


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
            metric.value: load(metric.value).compute(references=data[ColumnNames.TARGET.value],
                                                     predictions=data[ColumnNames.PREDICTION.value],
                                                     average='micro')[metric.value]
            for metric in self._metrics
        }
