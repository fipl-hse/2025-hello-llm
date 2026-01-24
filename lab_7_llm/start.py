"""
Starter for demonstration of laboratory work.
"""
import json

from core_utils.llm.time_decorator import report_time
# pylint: disable=too-many-locals, undefined-variable, unused-import


from lab_7_llm.main import RawDataImporter, RawDataPreprocessor


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    with open('settings.json', 'r', encoding='utf-8') as file:
        settings = json.load(file)

    name = settings['parameters']['dataset']

    dataset_importer = RawDataImporter(name)
    dataset_importer.obtain()

    preprocessed_dataset = RawDataPreprocessor(dataset_importer.raw_data)
    # preprocessed_dataset.analyze()
    # preprocessed_dataset.transform()

    result = preprocessed_dataset.analyze()
    print(result)
    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
