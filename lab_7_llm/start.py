"""
Starter for demonstration of laboratory work.
"""

from pathlib import Path

from core_utils.llm.metrics import Metrics
from core_utils.project.lab_settings import LabSettings
from lab_7_llm.main import (
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
    settings = LabSettings(Path(__file__).parent / "settings.json")

    importer = RawDataImporter(settings.parameters.dataset)
    importer.obtain()

    if importer.raw_data is None:
        raise ValueError("DataFrame is expected")

    preprocessor = RawDataPreprocessor(importer.raw_data)
    preprocessor.transform()
    dataset_analysis = preprocessor.analyze()

    print(dataset_analysis)

    dataset = TaskDataset(preprocessor.data.head(101))
    pipeline = LLMPipeline(
        model_name=settings.parameters.model,
        dataset=dataset,
        max_length=120,
        batch_size=64,
        device="cpu",
    )
    model_analysis = pipeline.analyze_model()
    print(model_analysis)

    print(pipeline.infer_sample(dataset[0]))

    predictions_path = Path(__file__).parent / "dist" / "predictions.csv"
    predictions_path.parent.mkdir(exist_ok=True)

    predictions = pipeline.infer_dataset()
    predictions.to_csv(predictions_path)

    evaluator = TaskEvaluator(
        predictions_path, [Metrics(metric) for metric in settings.parameters.metrics]
    )
    result = evaluator.run()
    print(result)

    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
