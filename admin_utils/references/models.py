"""
Models for references comparison tool
"""

# pylint: disable=too-few-public-methods
from decimal import Decimal
from enum import StrEnum
from pathlib import Path

import simplejson as json
from pydantic import BaseModel, ConfigDict, Field, field_validator, RootModel


class MSGStorage(StrEnum):
    """
    Storage for messages
    """

    MSG_DEGRADATION = "DEGRADED"
    MSG_NOT_COVERED = "CURRENT REFERENCE NOT COVERED"
    MSG_NO_DEGRADATION = "NO DEGRADATION"


class OutputSchema(BaseModel):
    """
    Schema that stores output information to be loaded
    """

    message: str = Field(default=MSGStorage.MSG_DEGRADATION.value)
    model: str
    dataset: str
    degraded_metrics: list[str] = Field(default=[MSGStorage.MSG_NO_DEGRADATION.value])
    current_values: dict[str, float] = Field(default_factory=dict)
    reference_values: dict[str, float] = Field(default_factory=dict)
    model_config = ConfigDict(extra="forbid", str_min_length=1)


class JSONSchema(BaseModel):
    """
    Schema that contains info about model, its dataset and score
    """

    model: str
    dataset: str
    score: dict[str, float]
    model_config = ConfigDict(extra="forbid", str_min_length=1)

    @classmethod
    @field_validator("score")
    def validate_score(cls, v: dict) -> dict:
        """
        Validator of score field

        Args:
            v (dict): Field of score.

        Returns:
            dict: Field itself.
        """
        if not v:
            raise ValueError("Score must be filled")
        for value in v.values():
            if not isinstance(value, (int, float)):
                raise ValueError("Score must be number")
            if value < 0:
                raise ValueError("Score must be positive number")
        return v


class JSONLoader(RootModel[dict[str, dict[str, dict[str, float]]]]):
    """
    Loader of JSON files via pydantic
    """

    @classmethod
    def from_file(cls, filepath: Path) -> "JSONLoader":
        """
        Method that loads file for further comparison

        Args:
            filepath (Path): Path to file to be loaded.

        Returns:
            JSONLoader: Object for further converting to schema.
        """
        with open(filepath, "r", encoding="utf-8") as file:
            return cls.model_validate_json(file.read())

    def to_schemas(self) -> list[JSONSchema]:
        """
        Method that converts file info into schemas

        Returns:
            list[JSONSchema]: Schemas of model-dataset-score info.
        """
        return [
            JSONSchema(model=model_name, dataset=dataset_name, score=score)
            for model_name, further_info in self.root.items()
            for dataset_name, score in further_info.items()
        ]

    @classmethod
    def load(cls, filepath: Path) -> list[JSONSchema]:
        """
        Method that loads and parses json file in one step

        Args:
            filepath (Path): Path to file to be loaded.

        Returns:
            list[JSONSchema]: Schemas of model-dataset-score info.
        """
        loader = cls.from_file(filepath)
        return loader.to_schemas()


class JSONSerializableMixin:
    """
    Mixin for serializable pydantic models.
    """

    def dump(self, json_path: Path) -> None:
        """
        Save model to JSON.

        Args:
            json_path (Path): Path to the file
        """
        with json_path.open("w", encoding="utf-8") as f:
            data = self.model_dump()
            json.dump(data, f, use_decimal=True, indent=4, ensure_ascii=False, sort_keys=True)
            f.write("\n")


class DatasetReferenceDTO(BaseModel):
    """
    Data transfer object for a single dataset's analytics.
    """

    dataset_number_of_samples: int
    dataset_columns: int
    dataset_duplicates: int
    dataset_empty_rows: int
    dataset_sample_min_len: int
    dataset_sample_max_len: int


class MetricScoresDTO(RootModel[dict[str, Decimal]]):
    """
    Data transfer object for metric name and score.
    """

    root: dict[str, Decimal] = {}


class DatasetScoresDTO(RootModel[dict[str, MetricScoresDTO]]):
    """
    Data transfer object for dataset name and metric scores mapping.
    """

    root: dict[str, MetricScoresDTO] = {}


class ModelAnalyticsDTO(BaseModel):
    """
    Data transfer object for a single model analytics
    """

    embedding_size: int
    input_shape: dict[str, list[int]] | list[int]
    max_context_length: int
    num_trainable_params: int
    output_shape: list[int]
    size: int
    vocab_size: int


class ModelAnalyticsModel(RootModel[dict[str, ModelAnalyticsDTO]], JSONSerializableMixin):
    """
    Model for storing multiple analytics
    """

    root: dict[str, ModelAnalyticsDTO] = {}

    def add(self, model_name: str, analytics: dict) -> None:
        """
        Add model to storage

        Args:
            model_name (str): Name of model
            analytics (dict): Model analytics
        """
        self.root[model_name] = ModelAnalyticsDTO(**analytics)


class InferenceReferencesModel(RootModel[dict[str, dict[str, str]]], JSONSerializableMixin):
    """
    Model for storing multiple inferences
    """

    root: dict[str, dict[str, str]] = {}

    def add(self, model_name: str, predictions: dict[str, str]) -> None:
        """
        Add model to storage

        Args:
            model_name (str): Name of model
            predictions (dict[str, str]): Predicted answers
        """
        self.root[model_name] = predictions


class DatasetReferencesModel(RootModel[dict[str, DatasetReferenceDTO]], JSONSerializableMixin):
    """
    Model for storing multiple dataset references.
    """

    root: dict[str, DatasetReferenceDTO] = {}

    def add(self, dataset_name: str, analytics: DatasetReferenceDTO) -> None:
        """
        Add dataset to storage

        Args:
            dataset_name (str): Name of dataset
            analytics (DatasetReferenceDTO): Dataset analytics
        """
        self.root[dataset_name] = analytics


class ReferenceScoresModel(RootModel[dict[str, DatasetScoresDTO]], JSONSerializableMixin):
    """
    Model for storing multiple reference scores of model, dataset and metric.
    """

    root: dict[str, DatasetScoresDTO] = {}

    def add(self, model_name: str, dataset_name: str, metric: str, score: Decimal) -> None:
        """
        Add a single metric score.

        Args:
            model_name (str): Name of the model
            dataset_name (str): Name of the dataset
            metric (str): Metric name
            score (Decimal): Metric score
        """
        if model_name not in self.root:
            self.root[model_name] = DatasetScoresDTO()
        if dataset_name not in self.root[model_name].root:
            self.root[model_name].root[dataset_name] = MetricScoresDTO()
        self.root[model_name].root[dataset_name].root[metric] = score


class EvaluationReferencesModel(BaseModel):
    """
    Model for loading evaluation references from JSON.
    """

    references: dict

    @classmethod
    def from_json(cls, json_path: Path) -> "EvaluationReferencesModel":
        """
        Load references from JSON file.

        Args:
            json_path (Path): Path to the reference file

        Returns:
            EvaluationReferencesModel: Loaded references
        """
        with json_path.open(encoding="utf-8") as f:
            data = json.load(f, parse_float=Decimal)
        return cls(references=data)

    def get_datasets(self) -> list[str]:
        """
        Extract unique dataset names from references.

        Returns:
            list[str]: Sorted list of unique dataset names
        """
        datasets_raw = []
        for _, dataset_pack in self.references.items():
            for dataset_name in dataset_pack.keys():
                datasets_raw.append(dataset_name)
        return sorted(set(datasets_raw))
