"""
Module with description of abstract raw data preprocessor.
"""

# pylint: disable=duplicate-code
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

try:
    from pandas import DataFrame
except ImportError:
    print('Library "pandas" not installed. Failed to import.')
    DataFrame = dict  # type: ignore


class ColumnNames(Enum):
    """
    Column names for preprocessed DataFrame.
    """

    SOURCE = "source"
    TARGET = "target"
    PREDICTION = "predictions"
    QUESTION = "question"
    CONTEXT = "context"
    PREMISE = "premise"
    HYPOTHESIS = "hypothesis"

    def __str__(self) -> Any:
        """
        String representation of column name.

        Returns:
            Any: Name of column name
        """
        return self.value


class AbstractRawDataPreprocessor(ABC):
    """
    Abstract Raw Data Preprocessor.
    """

    #: Preprocessed dataset in a table format
    _data: DataFrame | None

    def __init__(self, raw_data: DataFrame) -> None:
        """
        Initialize an instance of AbstractRawDataPreprocessor.

        Args:
            raw_data (pandas.DataFrame): Original dataset in a table format
        """
        self._raw_data = raw_data
        self._data = None

    @abstractmethod
    def analyze(self) -> dict:
        """
        Analyze a dataset.

        Returns:
            dict: Dataset key properties
        """

    @abstractmethod
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.
        """

    @property
    def data(self) -> DataFrame | None:
        """
        Property for preprocessed dataset.

        Returns:
            pandas.DataFrame | None: Preprocessed dataset in a table format
        """
        return self._data
