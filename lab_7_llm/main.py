"""
Laboratory work.

Working with Large Language Models.
"""

# pylint: disable=too-few-public-methods, undefined-variable, too-many-arguments, super-init-not-called

from pathlib import Path
from typing import cast, Iterable, List, Sequence

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from pandas import DataFrame
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
from transformers import AutoTokenizer, BatchEncoding, BertForTokenClassification

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
        self._raw_data = load_dataset(self._hf_name, split="val_en").to_pandas()

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
        df = self._raw_data.copy()
        df["tokens_tuple"] = df["tokens"].apply(tuple)

        df_clean = df.dropna()
        if len(df_clean) > 0:
            lengths = df_clean["tokens"].apply(len)
        else:
            lengths = pd.Series([0])

        return {
            "dataset_number_of_samples": len(self._raw_data),
            "dataset_columns": len(self._raw_data.columns),
            "dataset_duplicates": int(df.duplicated(subset=["tokens_tuple"]).sum()),
            "dataset_empty_rows": int(self._raw_data.isna().any(axis=1).sum()),
            "dataset_sample_min_len": int(lengths.min()) if len(lengths) > 0 else 0,
            "dataset_sample_max_len": int(lengths.max()) if len(lengths) > 0 else 0,
        }

    @report_time
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.
        """
        self._data = self._raw_data.rename(
            columns={"tokens": ColumnNames.SOURCE.value, "ner_tags": ColumnNames.TARGET.value}
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
        row = self._data.iloc[index]
        source = row[ColumnNames.SOURCE.value]
        target = row[ColumnNames.TARGET.value]

        if isinstance(source, (list, tuple, np.ndarray)):
            source = list(source)
        else:
            source = str(source)

        return source, str(target)

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

        self._model_name = model_name
        self._model = BertForTokenClassification.from_pretrained(model_name)
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
        ids = torch.ones(1, cast(int, config.max_position_embeddings), dtype=torch.long)
        result = summary(
            self._model,
            input_data={"input_ids": ids, "attention_mask": ids},
            device=self._device,
            verbose=0,
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
            sample (tuple[str, ...]): The given sample for inference

        Returns:
            str | None: A prediction
        """
        if not self._model:
            return None

        result = self._infer_batch([sample])
        return result[0] if result else None

    @report_time
    def infer_dataset(self) -> pd.DataFrame:
        """
        Infer model on a whole dataset.

        Returns:
            pd.DataFrame: Data with predictions
        """
        predictions = []
        targets = []

        dataloader = DataLoader(
            self._dataset, batch_size=self._batch_size, collate_fn=lambda batch: list(zip(*batch))
        )

        for batch in dataloader:
            inputs_for_infer = [(s,) for s in batch[0]]
            batch_predictions = self._infer_batch(inputs_for_infer)

            predictions.extend(batch_predictions)
            targets.extend(batch[1])

        result_df = pd.DataFrame(
            {ColumnNames.TARGET.value: targets, ColumnNames.PREDICTION.value: predictions}
        )

        return result_df

    @torch.no_grad()
    def _infer_batch(self, sample_batch: Sequence[tuple[str, ...]]) -> list[str]:
        """
        Infer model on a single batch.

        Args:
            sample_batch (Sequence[tuple[str, ...]]): Batch to infer

        Returns:
            list[str]: Model predictions as strings
        """
        if self._model is None:
            return []

        inputs = self._prepare_inputs_for_batch(sample_batch)

        ids = self._tokenizer(
            inputs,
            is_split_into_words=True,
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        ).to(self._device)

        outputs = self._model(**ids)
        predictions = torch.argmax(outputs.logits, dim=-1)

        return self._process_predictions(predictions, ids)

    def _prepare_inputs_for_batch(self, sample_batch: Sequence[tuple[str, ...]]) -> List[List[str]]:
        """
        Prepare inputs for the batch.

        Args:
            sample_batch: Batch to process

        Returns:
            List of token lists
        """
        inputs = []
        for sample in sample_batch:
            if isinstance(sample, (tuple, list)):
                token_data = sample[0]
            else:
                token_data = sample

            if isinstance(token_data, str):
                inputs.append(token_data.split())
            elif hasattr(token_data, "tolist"):
                inputs.append(token_data.tolist())
            elif isinstance(token_data, (list, tuple)):
                inputs.append([str(t) for t in token_data])
            else:
                inputs.append([str(token_data)])
        return inputs

    def _process_predictions(self, predictions: torch.Tensor, ids: BatchEncoding) -> List[str]:
        """
        Process model predictions.

        Args:
            predictions: Model predictions tensor
            ids: Tokenized inputs

        Returns:
            List of processed predictions as strings
        """
        final_predictions = []
        for i, prediction in enumerate(predictions):
            word_ids = ids.word_ids(batch_index=i)

            word_labels = []
            previous_word_idx = None

            for idx, word_idx in enumerate(word_ids):
                if word_idx is None:
                    continue

                if word_idx != previous_word_idx:
                    word_labels.append(prediction[idx].item())
                    previous_word_idx = word_idx

            final_predictions.append(str(word_labels))

        return final_predictions


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
        super().__init__(data_path, metrics)

        self._data_path = data_path
        self._metrics = metrics

    def run(self) -> dict[str, float]:
        """
        Evaluate the predictions against the references.

        Returns:
            dict: A dictionary containing information about the calculated
                  metric
        """
        data = pd.read_csv(self._data_path)

        all_predictions = []
        all_targets = []

        for _, row in data.iterrows():
            p_str = str(row[ColumnNames.PREDICTION.value])
            t_str = str(row[ColumnNames.TARGET.value])

            all_predictions.append([int(el) for el in p_str.strip("[]").replace(",", " ").split()])
            all_targets.append([int(el) for el in t_str.strip("[]").replace(",", " ").split()])

        result = {}

        for metric in self._metrics:
            predictions = []
            targets = []

            for prediction, target in zip(all_predictions, all_targets):
                min_len = min(len(prediction), len(target))
                predictions.extend(prediction[:min_len])
                targets.extend(target[:min_len])
            score_dict = evaluate.load(str(metric)).compute(
                predictions=predictions,
                references=targets
            )
            result[str(metric)] = score_dict[str(metric)]

        return result
