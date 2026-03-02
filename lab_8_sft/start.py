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
    SFTPipeline,
    TaskDataset,
    TaskEvaluator,
    TokenizedTaskDataset,
)


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    settings_file = Path(__file__).parent / "settings.json"
    with open(settings_file, 'r', encoding='utf-8') as file:
        settings = json.load(file)

    importer = RawDataImporter(settings['parameters']['dataset'])
    importer.obtain()

    if importer.raw_data is None:
        return
    preprocessor = RawDataPreprocessor(importer.raw_data)
    print('Dataset overview')
    for k, v in preprocessor.analyze().items():
        print(f'{k}: {v}')

    preprocessor.transform()
    dataset = TaskDataset(preprocessor.data.head(100))

    pipeline = LLMPipeline(settings['parameters']['model'], dataset, 120, 64, 'cpu')
    print('\nModel overview')
    for k, v in pipeline.analyze_model().items():
        print(f'{k}: {v}')

    print('\nSample inference: ', pipeline.infer_sample(dataset[0]))
    inference = pipeline.infer_dataset()
    print('\nDataset inference: ', inference)

    predictions_path = Path(__file__).parent / 'dist' / 'predictions.csv'
    predictions_path.parent.mkdir(exist_ok=True)
    inference.to_csv(predictions_path)

    evaluator = TaskEvaluator(predictions_path,
                              [Metrics(metric) for metric in settings['parameters']['metrics']])
    for k, v in evaluator.run().items():
        print(f'{k}: {v}')
    result = TaskEvaluator
    assert result is not None, "Fine-tuning does not work correctly"


if __name__ == "__main__":
    main()
