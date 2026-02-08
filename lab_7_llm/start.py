"""
Starter for demonstration of laboratory work.
"""

# pylint: disable=too-many-locals, undefined-variable, unused-import
import json
import types
from pathlib import Path

from core_utils.llm.time_decorator import report_time
from core_utils.project.lab_settings import LabSettings
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
    settings = LabSettings(Path(__file__).parent / "settings.json")

    importer = RawDataImporter(settings.parameters.dataset)
    importer.obtain()

    if importer.raw_data is None:
        return 

    preprocessor = RawDataPreprocessor(importer.raw_data) #_raw_data
    dataset_stats = preprocessor.analyze()
    print(dataset_stats)

    preprocessor.transform()
    dataset = TaskDataset(preprocessor.data.head(100))

    batch_size = 1
    max_length = 120
    device = 'cpu'

    pipeline = LLMPipeline(settings.parameters.model, dataset,
                           max_length, batch_size, device)
    model_stats = pipeline.analyze_model()
    print(model_stats)

    sample = dataset[0]
    sample_infer = pipeline.infer_sample(sample)

    # Dataset inference
    batch_size=64

    dataset_pipeline = LLMPipeline(settings.parameters.model, dataset,
                           max_length, batch_size, device)

    dataset_infer = dataset_pipeline.infer_dataset()
    print(dataset_infer)

    predictions_path = Path(__file__).parent / 'dist'/ 'predictions.csv'
    dataset_infer.to_csv(predictions_path)

    evaluator = TaskEvaluator(data_path=predictions_path,
                              metrics=settings.parameters.metrics)

    result = evaluator.run()
    print(result)

    # assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
