"""
Starter for demonstration of laboratory work.
"""

import pathlib

from core_utils.llm.time_decorator import report_time
from core_utils.project.lab_settings import LabSettings

# pylint: disable=too-many-locals, undefined-variable, unused-import
from lab_7_llm.main import (
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
    path_to_settings = pathlib.Path(__file__).parent / "settings.json"
    settings = LabSettings(path_to_settings)

    importer = RawDataImporter(settings.parameters.dataset)
    importer.obtain()

    preprocessor = RawDataPreprocessor(importer.raw_data)
    preprocessor.transform()
    analysis_result = preprocessor.analyze()

    dataset = TaskDataset(preprocessor.data.iloc[:100])

    pipeline = LLMPipeline(
        model_name=settings.parameters.model,
        dataset=dataset,
        max_length=120,
        batch_size=64,
        device="cpu",
    )

    model_analysis = pipeline.analyze_model()

    sample = dataset[0]
    sample_prediction = pipeline.infer_sample(sample)

    dataset_predictions = pipeline.infer_dataset()

    dist_dir = pathlib.Path(__file__).parent / "dist"
    dist_dir.mkdir(exist_ok=True)
    predictions_path = dist_dir / "predictions.csv"
    dataset_predictions.to_csv(predictions_path, index=False)

    evaluator = TaskEvaluator(predictions_path, settings.parameters.metrics)
    metrics_results = evaluator.run()

    result = {
        "dataset_analysis": analysis_result,
        "model_analysis": model_analysis,
        "sample_prediction": sample_prediction,
        "metrics": metrics_results,
    }
    print(metrics_results)

    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
