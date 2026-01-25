"""
Starter for demonstration of laboratory work.
"""

# pylint: disable=too-many-locals, undefined-variable, unused-import
import json
from pathlib import Path

from core_utils.llm.time_decorator import report_time
from main import RawDataImporter
from main import RawDataPreprocessor


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

    dataset_preprocessor = RawDataPreprocessor(dataset_importer._raw_data)
    for feature in dataset_preprocessor.analyze().items():
        print(f'{feature[0]}: {feature[1]}')

    result = dataset_preprocessor
    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
