"""
Fine-tuning starter.
"""

from pathlib import Path

from core_utils.project.lab_settings import LabSettings
from lab_8_sft.main import (
    LLMPipeline,
    RawDataImporter,
    RawDataPreprocessor,
    report_time,
    TaskDataset,
    TaskEvaluator,
)

# pylint: disable=too-many-locals, undefined-variable, unused-import, too-many-branches, too-many-statements
SETTINGS_PATH = Path(__file__).resolve().with_name("settings.json")

@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    settings = LabSettings(SETTINGS_PATH)
    importer = RawDataImporter(settings.parameters.dataset)
    importer.obtain()
    raw_data = importer.raw_data
    assert raw_data is not None, "Dataset was not loaded"

    preprocessor = RawDataPreprocessor(raw_data)
    analysis = preprocessor.analyze()
    preprocessor.transform()

    print("Dataset analysis:")
    for name, val in analysis.items():
        print(f"{name}: {val}")

    data = preprocessor.data.iloc[:100]
    assert data is not None, "Preprocessed data is missing"
    dataset = TaskDataset(data)

    pipeline = LLMPipeline(
        model_name=settings.parameters.model,
        dataset=dataset,
        batch_size=1,
        max_length=120,
        device="cpu",
    )

    model_analysis = pipeline.analyze_model()

    print("\nModel properties:")
    for name, val in model_analysis.items():
        print(f"{name}: {val}")

    first_sample = dataset[0]
    prediction = pipeline.infer_sample(first_sample)

    print("\nSample inference:")
    print("Text:", first_sample[0])
    print("Prediction:", prediction)

    predictions_df = pipeline.infer_dataset()
    dist_path = Path(__file__).resolve().parent / "dist"
    dist_path.mkdir(parents=True, exist_ok=True)
    predictions_path = dist_path / "predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)

    evaluator = TaskEvaluator(predictions_path, settings.parameters.metrics)
    metrics_result = evaluator.run()
    print("\nModel performance evaluation:")
    for metric_name, score in metrics_result.items():
        print(f"{metric_name}: {score}")

    result = metrics_result
    assert result is not None, "Fine-tuning does not work correctly"


if __name__ == "__main__":
    main()
