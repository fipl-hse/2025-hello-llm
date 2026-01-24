"""
Starter for demonstration of laboratory work.
"""
import json

# pylint: disable=too-many-locals, undefined-variable, unused-import

from core_utils.llm.time_decorator import report_time
from lab_7_llm.main import RawDataImporter, RawDataPreprocessor


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    result = None
    with open ('settings.json', 'r', encoding='utf-8') as file:
        settings = json.load(file)

    name = settings['parameters']['dataset']

    data_importer = RawDataImporter(name)
    data_importer.obtain()

    data_preprocessor = RawDataPreprocessor(data_importer._raw_data)
    result = data_preprocessor.analyze()

    for key, value in result.items():
        print(f'{key} : {value}')
    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
