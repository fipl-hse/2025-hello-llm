"""
Starter for demonstration of laboratory work.
"""
import json
from pathlib import Path

from core_utils.llm.metrics import Metrics
from lab_7_llm.main import (
    LLMPipeline,
    RawDataImporter,
    RawDataPreprocessor,
    report_time,
    TaskDataset,
    TaskEvaluator,
)

# pylint: disable=too-many-locals, undefined-variable, unused-import


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    settings_path = Path(__file__).parent / 'settings.json'

    with open(settings_path, 'r', encoding='utf-8') as file:
        settings = json.load(file)

    name = settings['parameters']['dataset']

    importer = RawDataImporter(name)
    importer.obtain()

    if importer.raw_data is None:
        raise ValueError("Failed to obtain raw data")

    preprocessor = RawDataPreprocessor(importer.raw_data)
    result = preprocessor.analyze()

    for key, value in result.items():
        print(f'{key} : {value}')

    preprocessor.transform()
    dataset = TaskDataset(preprocessor.data.head(100))

    pipeline = LLMPipeline(settings['parameters']['model'], dataset, 120, 1, 'cpu')

    for key, value in pipeline.analyze_model().items():
        print(f'{key}: {value}')

    print(pipeline.infer_sample(dataset[1]))

    predictions_df = pipeline.infer_dataset()

    predictions_file = Path(__file__).parent / "dist" / "predictions.csv"

    predictions_file.parent.mkdir(parents=True, exist_ok=True)

    predictions_df.to_csv(predictions_file, index=False)

    metric_names = settings['parameters']['metrics']
    metrics = [Metrics[metric.upper()] for metric in metric_names]

    evaluator = TaskEvaluator(predictions_file, metrics)
    result = evaluator.run()

    assert result is not None, "Demo does not work correctly"

if __name__ == "__main__":
    main()


