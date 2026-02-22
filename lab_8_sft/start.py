"""
Fine-tuning starter.
"""
import json
from pathlib import Path

from lab_8_sft.main import (
    LLMPipeline,
    RawDataImporter,
    RawDataPreprocessor,
    report_time,
    TaskDataset,
    TaskEvaluator,
)

# pylint: disable=too-many-locals, undefined-variable, unused-import, too-many-branches, too-many-statements


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    result = None
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

    predictions_df = pipeline.infer_dataset()

    predictions_file = Path(__file__).parent / "dist" / "predictions.csv"

    predictions_file.parent.mkdir(parents=True, exist_ok=True)

    predictions_df.to_csv(predictions_file, index=False)

    # metrics = settings['parameters']['metrics']
    # evaluator = TaskEvaluator(predictions_file, metrics)
    # result = evaluator.run()
    # print("Evaluation results:", result)

    assert result is not None, "Fine-tuning does not work correctly"

if __name__ == "__main__":
    main()
