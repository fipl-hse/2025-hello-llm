"""
Fine-tuning starter.
"""

from pathlib import Path

# pylint: disable=too-many-locals, undefined-variable, unused-import
from core_utils.llm.metrics import Metrics
from core_utils.project.lab_settings import LabSettings
from lab_8_sft.main import (
    LLMPipeline,
    RawDataImporter,
    RawDataPreprocessor,
    report_time,
    TaskDataset,
    TaskEvaluator,
)


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    current_path = Path(__file__).parent
    settings = LabSettings(current_path / "settings.json")

    importer = RawDataImporter(settings.parameters.dataset)
    importer.obtain()

    if importer.raw_data is None:
        return

    preprocessor = RawDataPreprocessor(importer.raw_data)
    preprocessor.transform()

    dataset = TaskDataset(preprocessor.data.head(100))
    pipeline = LLMPipeline(
        model_name=settings.parameters.model,
        dataset=dataset,
        batch_size=64,
        max_length=120,
        device="cpu"
    )

    model_info = pipeline.analyze_model()

    print("\nModel properties analysis:")
    for key, value in model_info.items():
        print(f"{key}: {value}")

    sample = dataset[0]
    prediction = pipeline.infer_sample(sample)

    print("\nSample inference:")
    print(sample[0])
    print("Class:", sample[1])
    print("Prediction:", prediction)

    predictions_path = Path(__file__).parent / "dist" / "predictions.csv"
    predictions_path.parent.mkdir(parents=True, exist_ok=True)

    if not predictions_path.exists():
        pipeline.infer_dataset().to_csv(predictions_path)

    predictions_df = pipeline.infer_dataset()
    predictions_df.to_csv(predictions_path)

    metrics = [Metrics(metric) for metric in settings.parameters.metrics]
    evaluator = TaskEvaluator(predictions_path, metrics)

    print('\nEvaluation:')
    for key, value in evaluator.run().items():
        print(f'{key}: {value}')

    result = evaluator
    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
