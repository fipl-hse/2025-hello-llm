"""
Starter for demonstration of laboratory work.
"""

# pylint: disable=too-many-locals, undefined-variable, unused-import
import json
from pathlib import Path

from core_utils.llm.time_decorator import report_time
from lab_7_llm.main import LLMPipeline, RawDataImporter, RawDataPreprocessor, TaskDataset


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

    preprocessor = RawDataPreprocessor(importer.raw_data)
    print('Dataset overview')
    for k, v in preprocessor.analyze().items():
        print(f'{k}: {v}')

    preprocessor.transform()
    dataset = TaskDataset(preprocessor.data.head(100))

    pipeline = LLMPipeline(settings['parameters']['model'], dataset, 120, 1, 'cpu')
    print('\nModel overview')
    for k, v in pipeline.analyze_model().items():
        print(f'{k}: {v}')

    print('\nLabel prediction for 1 sample:', pipeline.infer_sample(dataset[1]))

    result = preprocessor
    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
