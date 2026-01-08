"""
Module with description of abstract LLM pipeline.
"""

# pylint: disable=too-few-public-methods, too-many-arguments, duplicate-code, invalid-name
from abc import ABC, abstractmethod
from typing import Any, Protocol

try:
    from pandas import DataFrame
except ImportError:
    print('Library "pandas" not installed. Failed to import.')
    DataFrame = dict  # type: ignore

try:
    from torch.utils.data.dataset import Dataset
except ImportError:
    print('Library "torch" not installed. Failed to import.')
    Dataset = None  # type: ignore
    Tensor = None  # type: ignore


class HFModelLike(Protocol):
    """
    Protocol definition of HF models.
    """

    def __call__(self, *args: tuple, return_dict: bool = False, **kwargs: dict) -> Any:
        """
        Placeholder to claim HF models are callable.

        Args:
            args (tuple): Arbitrary positional arguments
            return_dict (bool, optional): Special argument for QA models. Defaults to False.
            kwargs (dict): Arbitrary named arguments

        Returns:
            Any: Custom value
        """

    def to(self, device: str) -> Any:
        """
        Placeholder to claim HF models are transferred to a specific device.

        Args:
            device (str): Device name

        Returns:
            Any: Custom value
        """


class AbstractLLMPipeline(ABC):
    """
    Abstract LLM Pipeline.
    """

    #: Model
    _model: HFModelLike | None

    def __init__(
        self,
        model_name: str,
        dataset: Dataset,
        max_length: int,
        batch_size: int,
        device: str = "cpu",
    ) -> None:
        """
        Initialize an instance of AbstractLLMPipeline.

        Args:
            model_name (str): The name of the pre-trained model.
            dataset (torch.utils.data.dataset.Dataset): The dataset used.
            max_length (int): The maximum length of generated sequence.
            batch_size (int): The size of the batch inside DataLoader.
            device (str, optional): The device for inference. Defaults to "cpu".
        """
        self._model_name = model_name
        self._model = None
        self._dataset = dataset
        self._max_length = max_length
        self._batch_size = batch_size
        self._device = device

    @abstractmethod
    def infer_sample(self, sample: tuple[str, ...]) -> str | None:
        """
        Infer model on a single sample.

        Args:
            sample (tuple[str, ...]): The given sample for inference with model

        Returns:
            str | None: A prediction
        """

    @abstractmethod
    def infer_dataset(self) -> DataFrame:
        """
        Infer model on a whole dataset.

        Returns:
            pandas.DataFrame: Data with predictions.
        """

    @abstractmethod
    def analyze_model(self) -> dict:
        """
        Analyze model computing properties.

        Returns:
            dict: Properties of a model
        """
