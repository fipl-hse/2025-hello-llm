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
from transformers import AutoTokenizer, DebertaV2ForQuestionAnswering

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
        self._raw_data = load_dataset(self._hf_name, split="test").to_pandas()

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
        df = self._raw_data.dropna()[["instruction", "context"]]
        lengths = df.apply(lambda x: x.str.len())

        return {
            "dataset_number_of_samples": len(self._raw_data),
            "dataset_columns": len(self._raw_data.columns),
            "dataset_duplicates": int(self._raw_data.duplicated().sum()),
            "dataset_empty_rows": int(self._raw_data.isna().any(axis=1).sum()),
            "dataset_sample_min_len": int(lengths.min().min()),
            "dataset_sample_max_len": int(lengths.max().max()),
        }

    @report_time
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.
        """
        self._data = self._raw_data[["instruction", "context", "response"]]
        self._data = self._data.rename(
            columns={"instruction": ColumnNames.QUESTION, "response": ColumnNames.TARGET}
        ).reset_index(drop=True)


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
        self._model = DebertaV2ForQuestionAnswering.from_pretrained(model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._dataset = dataset
        self._max_length = max_length
        self._batch_size = batch_size
        self._device = device
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
        emb_length = int(config.max_position_embeddings)
        result = summary(
            self._model,
            input_data={
                "input_ids": torch.ones(
                    (1,
                    emb_length),
                    dtype=torch.long),
                "attention_mask": torch.ones(
                    (1,
                    emb_length),
                    dtype=torch.long)
            },
            device=self._device
        )
        return {
            "input_shape": {
                "input_ids": [1, config.max_position_embeddings],
                "attention_mask": [1, config.max_position_embeddings],
            },
            "embedding_size": config.max_position_embeddings,
            "output_shape": result.summary_list[-1].output_size,
            "num_trainable_params": result.trainable_params,
            "vocab_size": config.vocab_size,
            "size": result.total_param_bytes,
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
        if not self._model:
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
        targets = []

        for el in DataLoader(dataset=self._dataset, batch_size=self._batch_size):
            ques_cont = list(zip(el[0], el[1]))
            targ = el[2]

            preds.extend(self._infer_batch(ques_cont))
            targets.extend(targ)

        return pd.DataFrame({"target": targets, "predictions": preds})

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
        predictions = []
        inputs = list(zip(*sample_batch))

        ids = self._tokenizer(
            inputs[0], inputs[1],
            max_length = self._max_length,
            padding=True, truncation=True,
            return_tensors="pt"
        )
        ids = {k: v.to(self._device) for k, v in ids.items()}
        output = self._model(**ids)
        start_ids = torch.argmax(output.start_logits, dim=1)
        end_ids = torch.argmax(output.end_logits, dim=1)

        for i, (s, e) in enumerate(zip(start_ids, end_ids)):
            sample = ids["input_ids"][i]
            answer_ids = sample[s : e + 1]
            decoded_text = self._tokenizer.decode(answer_ids, skip_special_tokens=True)
            predictions.append(decoded_text.strip())
        return predictions


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
        result = {}

        for metric in self._metrics:
            metric_evaluate = evaluate.load(str(metric))
            preds = [
                {"prediction_text": str(el), "id": str(i)}
                for i, el in enumerate(data[ColumnNames.PREDICTION.value])
            ]
            refs = [
                {"answers": {"answer_start": [0], "text": [str(el)]}, "id": str(i)}
                for i, el in enumerate(data[ColumnNames.TARGET.value])
            ]
            result[str(metric)] = metric_evaluate.compute(predictions=preds, references=refs)['f1']
        return result
