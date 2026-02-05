"""
Module with description of abstract data importer.
"""

# pylint: disable=duplicate-code
from abc import ABC, abstractmethod
from pathlib import Path

try:
    from pandas import DataFrame
except ImportError:
    print('Library "pandas" not installed. Failed to import.')
    DataFrame = dict  # type: ignore


class AbstractRawDataImporter(ABC):
    """
    Abstract Raw Data Importer.
    """

    #: A path to dataset
    _local_path: Path | None

    #: A dataset in a table format
    _raw_data: DataFrame | None

    def __init__(self, hf_name: str | None) -> None:
        """
        Initialize an instance of AbstractRawDataImporter.

        Args:
             hf_name (str | None): Name of the HuggingFace dataset
        """
        self._hf_name = hf_name
        self._raw_data = None

    @abstractmethod
    def obtain(self) -> None:
        """
        Download a dataset.
        """

    @property
    def raw_data(self) -> DataFrame | None:
        """
        Property for original dataset in a table format.

        Returns:
            pandas.DataFrame | None: A dataset in a table format
        """
        return self._raw_data
