"""
Fine-tuning starter.
"""

# pylint: disable=too-many-locals, undefined-variable, unused-import, too-many-branches, too-many-statements
import json
from pathlib import Path

from core_utils.llm.metrics import Metrics
from core_utils.llm.time_decorator import report_time
from lab_8_sft.main import (
    LLMPipeline,
    RawDataImporter,
    RawDataPreprocessor,
    TaskDataset,
    TaskEvaluator,
)


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    root_path = Path(__file__).parent
    with open(root_path / 'settings.json', 'r', encoding='utf-8') as file:
        settings = json.load(file)

    importer = RawDataImporter(settings['parameters']['dataset'])
    importer.obtain()

    if importer.raw_data is None:
        raise ValueError("Raw data is None")

    # Выведем колонки, чтобы понять структуру датасета
    print("Dataset Columns:", importer.raw_data.columns)

    preprocessor = RawDataPreprocessor(importer.raw_data)
    preprocessor.transform()

    # Берем 10 примеров для теста
    dataset = TaskDataset(preprocessor.data.head(10))

    pipeline = LLMPipeline(
        model_name=settings['parameters']['model'],
        dataset=dataset,
        max_length=120,
        batch_size=1,
        device='cpu'
    )

    model_analysis = pipeline.analyze_model()
    print("Model Analysis:", model_analysis)

    sample = dataset[0]
    print(f"Text: {sample[0]}")
    print(f"True Label: {sample[1]}")

    prediction = pipeline.infer_sample(sample)
    print(f"Prediction (Class ID): {prediction}")

    result = prediction
    assert result is not None, "Fine-tuning does not work correctly"


if __name__ == "__main__":
    main()
