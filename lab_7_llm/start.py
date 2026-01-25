"""
Starter for demonstration of laboratory work.
"""
import json
from pathlib import Path
from types import SimpleNamespace

# pylint: disable=too-many-locals, undefined-variable, unused-import

from lab_7_llm.main import RawDataImporter, RawDataPreprocessor, report_time

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

    preprocessor = RawDataPreprocessor(importer.raw_data)
    result = preprocessor.analyze()

    for key, value in result.items():
        print(f'{key} : {value}')

    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
