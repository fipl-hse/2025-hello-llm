"""
Starter for demonstration of laboratory work.
"""

import json
from pathlib import Path
from random import sample
from types import SimpleNamespace

from huggingface_hub import model_info
from tomlkit import key, value

from core_utils.project.lab_settings import LabSettings

# pylint: disable=too-many-locals, undefined-variable, unused-import
from lab_7_llm.main import (
    LLMPipeline,
    RawDataImporter,
    RawDataPreprocessor,
    report_time,
    TaskDataset,
)

try:
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        XLMRobertaForSequenceClassification,
    )
except ImportError:
    print('Library "transformers" not installed. Failed to import.')

try:
    from torchinfo import summary  # type: ignore
except ImportError:
    print('Library "torchinfo" not installed. Failed to import.')

try:
    import torch
except ImportError:
    print('Library "torch" not installed. Failed to import.')


SETTINGS_PATH = Path(__file__).resolve().with_name("settings.json")

@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    settings = LabSettings(SETTINGS_PATH)
    importer = RawDataImporter(settings.parameters.dataset)
    importer.obtain()

    preprocessor = RawDataPreprocessor(importer.raw_data)
    analysis = preprocessor.analyze()
    preprocessor.transform()

    print("Dataset analysis:")
    for key, value in analysis.items():
        print(f"{key}: {value}")

    dataset = TaskDataset(preprocessor.data.head(100))

    pipeline = LLMPipeline(
        model_name=settings.parameters.model,
        dataset=dataset,
        batch_size=1,
        max_length=120,
        device="cpu",
    )

    model_info = pipeline.analyze_model()

    print("\nModel properties:")
    for key, value in model_info.items():
        print(f"{key}: {value}")

    sample = dataset[0]
    prediction = pipeline.infer_sample(sample)

    print("\nSample inference:")
    print("Text:", sample[0])
    print("Prediction:", prediction)

    result = prediction
    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
