"""
Starter for demonstration of laboratory work.
"""

import json

# pylint: disable=too-many-locals, undefined-variable, unused-import
from pathlib import Path

from core_utils.llm.time_decorator import report_time
from lab_7_llm.main import LLMPipeline, RawDataImporter, RawDataPreprocessor, TaskDataset


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    with open(Path(__file__).parent / 'settings.json', 'r', encoding='utf-8') as f:
        settings = json.load(f)

    importer = RawDataImporter(settings['parameters']['dataset'])
    importer.obtain()

    preprocessor = RawDataPreprocessor(importer.raw_data)

    result = preprocessor.analyze()

    for key, value in result.items():
        print(f"{key}: {value}")

    preprocessor.transform()
    dataset = TaskDataset(preprocessor.data.head(100))

    pipeline = LLMPipeline(settings['parameters']['model'], dataset, 120, 1, 'cpu')

    for key, value in pipeline.analyze_model().items():
        print(f'{key}: {value}')

    print(pipeline.infer_sample(dataset[1]))

    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
