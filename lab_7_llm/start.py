"""
Starter for demonstration of laboratory work.
"""

# pylint: disable=too-many-locals, undefined-variable, unused-import
import json
from pathlib import Path

from core_utils.llm.time_decorator import report_time
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
    settings_path = Path(__file__).parent / 'settings.json'
    with open(settings_path, 'r', encoding='utf-8') as file:
        settings = json.load(file)

    name = settings['parameters']['dataset']
    dataset_importer = RawDataImporter(name)
    dataset_importer.obtain()
    if dataset_importer.raw_data is None:
        return

    dataset_preprocessor = RawDataPreprocessor(dataset_importer.raw_data)
    for feature in dataset_preprocessor.analyze().items():
        print(f'{feature[0]}: {feature[1]}')
    dataset_preprocessor.transform()

    dataset = TaskDataset(dataset_preprocessor.data.head(100))

    pipeline = LLMPipeline(settings['parameters']['model'], dataset, 120, 64, 'cpu')
    for key, value in pipeline.analyze_model().items():
        print(f'{key} : {value}')
    print(pipeline.infer_sample(dataset[0]))

    predictions_df = pipeline.infer_dataset()

    predictions_file = Path(__file__).parent / "dist" / "predictions.csv"
    predictions_file.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(predictions_file)

    evaluator = TaskEvaluator(predictions_file, settings['parameters']['metrics'])
    result = evaluator.run()
    print("Evaluation results:", result)
    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
