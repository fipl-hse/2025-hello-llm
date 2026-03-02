"""
Laboratory work.

Fine-tuning Large Language Models for a downstream task.
"""

from pathlib import Path
from typing import Callable, Iterable, Sequence

# pylint: disable=too-few-public-methods, undefined-variable, duplicate-code, unused-argument, too-many-arguments
import datasets
import pandas as pd
import torch
from evaluate import load
from pandas import DataFrame
from peft import LoraConfig
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

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
        self._raw_data = datasets.load_dataset(self._hf_name, split="train").to_pandas()

        if not isinstance(self._raw_data, pd.DataFrame):
            raise TypeError('The downloaded dataset is not a pandas.DataFrame.')


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
        df = self._raw_data
        analysis = {'dataset_number_of_samples': len(df),
                    'dataset_columns': len(df.columns),
                    'dataset_duplicates': len(df[df.duplicated()]),
                    'dataset_empty_rows': len(df[df.isna().any(axis=1)])}

        column = df['article_content'].dropna()
        analysis['dataset_sample_min_len'] = min(column.apply(len))
        analysis['dataset_sample_max_len'] = max(column.apply(len))
        return analysis

    @report_time
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.
        """
        preprocessed = self._raw_data[["article_content", "summary"]]
        preprocessed = preprocessed.rename(columns={"summary": ColumnNames.TARGET,
                                                    "article_content": ColumnNames.SOURCE})
        preprocessed.dropna(inplace=True)
        self._data = preprocessed.reset_index()


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
        return (self._data[ColumnNames.SOURCE].iloc[index],)

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
        a dictionary with the input_ids, attention_mask and labels for current sample as keys
    """
    source = tokenizer(sample[ColumnNames.SOURCE],
                       padding="max_length",
                       truncation=True,
                       max_length=max_length)
    target = tokenizer(sample[ColumnNames.TARGET],
                       padding="max_length",
                       truncation=True,
                       max_length=max_length)
    return {
            "input_ids": source["input_ids"],
            "attention_mask": source["attention_mask"],
            "labels": target["input_ids"]
    }


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
        self._data = [tokenize_sample(row, tokenizer, max_length) for _, row in data.iterrows()]
        self._tokenizer = tokenizer
        self._max_length = max_length

    def __len__(self) -> int:
        """
        Return the number of items in the dataset.

        Returns:
            int: The number of items in the dataset
        """
        return len(self._data)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """
        Retrieve an item from the dataset by index.

        Args:
            index (int): Index of sample in dataset

        Returns:
            dict[str, torch.Tensor]: An element from the dataset
        """
        return self._data[index]


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
        super().__init__(model_name, dataset, max_length, batch_size, device)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

    def analyze_model(self) -> dict:
        """
        Analyze model computing properties.

        Returns:
            dict: Properties of a model
        """
        config = self._model.config
        embeddings_length = config.d_model

        ids = torch.ones(1, embeddings_length, dtype=torch.long, device=self._device)
        stats = summary(self._model, input_data={"input_ids": ids, "decoder_input_ids": ids},
                        device=self._device, verbose=0)
        analysis = {'input_shape': list(stats.input_size["input_ids"]),
                    'embedding_size': embeddings_length,
                    'output_shape': stats.summary_list[-1].output_size,
                    'num_trainable_params': stats.trainable_params,
                    'vocab_size': config.vocab_size,
                    'size': stats.total_param_bytes,
                    'max_context_length': config.max_length}
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
        if not isinstance(sample, tuple):
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

        for batch in loader:
            predictions.extend(self._infer_batch(batch))
            references.extend(batch[0])
        return pd.DataFrame({ColumnNames.TARGET: references,
                             ColumnNames.PREDICTION: predictions})

    @torch.no_grad()
    def _infer_batch(self, sample_batch: Sequence[tuple[str, ...]]) -> list[str]:
        """
        Infer single batch.

        Args:
            sample_batch (Sequence[tuple[str, ...]]): batch to infer the model

        Returns:
            list[str]: model predictions as strings
        """
        inputs = self._tokenizer(sample_batch[0],
                                 return_tensors="pt",
                                 padding=True,
                                 truncation=True,
                                 max_length=self._max_length).to(self._device)

        self._model.eval()
        output = self._model.generate(**inputs, max_length=self._max_length)
        return self._tokenizer.batch_decode(output, skip_special_tokens=True)


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

    def run(self) -> dict:
        """
        Evaluate the predictions against the references using the specified metric.

        Returns:
            dict: A dictionary containing information about the calculated metric
        """
        data = pd.read_csv(self._data_path)

        predictions = data['predictions'].astype(str).tolist()
        references = data['target'].astype(str).tolist()

        results = {}

        for metric in self._metrics:
            if metric == Metrics.BLEU:
                bleu_metric = load("bleu")

                bleu_result = bleu_metric.compute(
                    predictions=predictions,
                    references=[[ref] for ref in references]
                )
                results["bleu"] = round(float(bleu_result["bleu"]), 6)
            elif metric == Metrics.ROUGE:
                rouge_metric = load("rouge", seed=77)

                rouge_result = rouge_metric.compute(
                    predictions=predictions,
                    references=references,
                    )

                results["rouge"] = round(float(rouge_result["rougeL"]), 6)

        return results


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
        super().__init__(model_name, dataset, data_collator)

    def run(self) -> None:
        """
        Fine-tune model.
        """
