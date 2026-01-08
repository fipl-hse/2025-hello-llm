"""
Module with description of abstract LLM pipeline.
"""

# pylint: disable=too-few-public-methods, too-many-arguments, duplicate-code, invalid-name
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

try:
    from peft import LoraConfig
except ImportError:
    print('Library "peft" not installed. Failed to import.')

try:
    import torch
    from torch.utils.data.dataset import Dataset
except ImportError:
    print('Library "torch" not installed. Failed to import.')
    Dataset = None  # type: ignore

try:
    from transformers import AutoTokenizer
except ImportError:
    print('Library "transformers" not installed. Failed to import.')

from core_utils.llm.llm_pipeline import HFModelLike


class AbstractSFTPipeline(ABC):
    """
    Abstract Fine-Tuning LLM Pipeline.
    """

    #: Model
    _model: HFModelLike | None
    _dataset: Dataset
    _batch_size: int | None
    _lora_config: LoraConfig | None
    _max_length: int | None
    _max_sft_steps: int | None
    _device: str | None
    _finetuned_model_path: Path | None
    _learning_rate: float | None

    def __init__(
        self,
        model_name: str,
        dataset: Dataset,
        data_collator: Callable[[AutoTokenizer], torch.Tensor] | None = None,
    ) -> None:
        """
        Initialize an instance of AbstractLLMPipeline.

        Args:
            model_name (str): The name of the pre-trained model
            dataset (torch.utils.data.dataset.Dataset): The dataset used
            data_collator (Callable[[AutoTokenizer], torch.Tensor] | None, optional): processing
                                                                    batch. Defaults to None.
        """
        self._model_name = model_name
        self._model = None
        self._dataset = dataset
        self._data_collator = data_collator

    @abstractmethod
    def run(self) -> None:
        """
        Fine-tune model.
        """
