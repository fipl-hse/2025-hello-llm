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
    settings_path = Path(__file__).parent / 'settings.json'

    with open(settings_path, 'r', encoding='utf-8') as file:
        settings = json.load(file)

    name = settings['parameters']['dataset']

    data_importer = RawDataImporter(name)
    data_importer.obtain()

    data_preprocessor = RawDataPreprocessor(data_importer.raw_data)
    result = data_preprocessor.analyze()

    for key, value in result.items():
        print(f'{key} : {value}')

    data_preprocessor.transform()
    dataset = TaskDataset(data_preprocessor.data.head(100))

    pipeline = LLMPipeline(settings['parameters']['model'], dataset,120, 1, 'cpu')

    for key, value in pipeline.analyze_model().items():
        print(f'{key} : {value}')

    print(pipeline.infer_sample(dataset[0]))
    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
