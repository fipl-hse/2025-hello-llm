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
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from core_utils.llm.llm_pipeline import AbstractLLMPipeline
from core_utils.llm.raw_data_importer import AbstractRawDataImporter
from core_utils.llm.raw_data_preprocessor import AbstractRawDataPreprocessor, ColumnNames
from core_utils.llm.task_evaluator import AbstractTaskEvaluator, Metrics
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

        dataset = load_dataset(self._hf_name, split="test")
        self._raw_data = dataset.to_pandas()

        if not isinstance(self._raw_data, pd.DataFrame):
            raise TypeError('The downloaded dataset is not pd.DataFrame')


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
        initial_len = len(self._raw_data)
        df_clean = self._raw_data.dropna(subset=["text"])
        self._raw_data = df_clean 
        empty_rows = initial_len - len(df_clean)

        lengths = df_clean["text"].str.len()
        return {
            "dataset_number_of_samples": len(df_clean),
            "dataset_columns": len(df_clean.columns),
            "dataset_duplicates": int(df_clean.duplicated().sum()),
            "dataset_empty_rows": empty_rows,
            "dataset_sample_min_len": int(lengths.min()),
            "dataset_sample_max_len": int(lengths.max())
        }

    @report_time
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.
        """
        self._data = self._raw_data.rename(columns={
            "label": ColumnNames.TARGET,
            "text": ColumnNames.SOURCE
        })[[ColumnNames.SOURCE, ColumnNames.TARGET]].copy()
        self._data.reset_index(drop=True, inplace=True)

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
        return (str(row[ColumnNames.SOURCE]),)

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
        self._dataset = dataset
        self._max_length = max_length
        self._batch_size = batch_size
        self._device = device if device else 'cpu'

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self._device)
        self._model.eval()


    def analyze_model(self) -> dict:
        """
        Analyze model computing properties.

        Returns:
            dict: Properties of a model
        """
        config = self._model.config
        total_params = self._model.num_parameters()
        trainable_params = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        return {
            "input_shape": [1, config.max_position_embeddings],
            "embedding_size": config.hidden_size,
            "output_shape": [1, config.num_labels],
            "num_trainable_params": trainable_params,
            "vocab_size": config.vocab_size,
            "size": total_params,
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
        batch = ([sample[0]],)
        return self._infer_batch(batch)[0]  

    @report_time
    def infer_dataset(self) -> pd.DataFrame:
        """
        Infer model on a whole dataset.

        Returns:
            pd.DataFrame: Data with predictions
        """
        
        dataloader = DataLoader(self._dataset, batch_size=self._batch_size, shuffle=False)
        predictions = []
        for batch in dataloader:
            batch_preds = self._infer_batch(batch)
            predictions.extend(batch_preds)
        
        targets = self._dataset.data[ColumnNames.TARGET].values[:len(predictions)]
        return pd.DataFrame({
            "target": targets,
            "prediction": predictions
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

        if isinstance(sample_batch, tuple):
            texts = sample_batch[0]
        else:
            texts = [item for item in sample_batch[0]]

        inputs = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors='pt'
        ).to(self._device)

        outputs = self._model(**inputs)
        pred_indices = torch.argmax(outputs.logits, dim=-1).cpu().tolist()
        return pred_indices


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
        df = pd.read_csv(self._data_path)

        references = df['target'].tolist()
        predictions = df['prediction'].tolist()

        results = {}
        for metric_enum in self._metrics:
            metric_name = metric_enum.value
            metric = evaluate.load(metric_name)
            score_dict = metric.compute(predictions=predictions, references=references)
            
            results[metric_name] = score_dict[metric_name]
        return results

