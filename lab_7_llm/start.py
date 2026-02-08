"""
Starter for demonstration of laboratory work.
"""
import json
from pathlib import Path

from core_utils.llm.metrics import Metrics
from core_utils.llm.time_decorator import report_time
from lab_7_llm.main import (
    LLMPipeline,
    RawDataImporter,
    RawDataPreprocessor,
    TaskDataset,
    TaskEvaluator,
)

BASE_PATH = Path(__file__).parent


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    with open(BASE_PATH / 'settings.json', 'r', encoding='utf-8') as f:
        settings = json.load(f)

    importer = RawDataImporter(settings['parameters']['dataset'])
    importer.obtain()
    if importer.raw_data is None:
        return

    preprocessor = RawDataPreprocessor(importer.raw_data)

    result = preprocessor.analyze()
    print('Dataset analysis:')
    for key, value in result.items():
        print(f'{key}: {value}')

    preprocessor.transform()
    dataset = TaskDataset(preprocessor.data.head(100))

    pipeline = LLMPipeline(settings['parameters']['model'], dataset, 120, 1, 'cpu')

    predictions_path = BASE_PATH / 'dist' / 'predictions.csv'
    predictions_path.parent.mkdir(parents=True, exist_ok=True)

    if not predictions_path.exists():
        pipeline.infer_dataset().to_csv(predictions_path)

    print('\nModel analysis:')
    for key, value in pipeline.analyze_model().items():
        print(f'{key}: {value}')

    source, reference = dataset[0]

    print("\nModel inference:")
    print(f"Source: {source}")
    print(f"Reference: {reference}")
    print(f"Predicted: {pipeline.infer_sample(dataset[0])}")

    metrics = [Metrics(metric) for metric in settings['parameters']['metrics']]
    evaluator = TaskEvaluator(BASE_PATH / 'dist' / 'predictions.csv', metrics)

    print('\nEvaluation:')
    for key, value in evaluator.run().items():
        print(f'{key}: {value}')

    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
