"""
Starter for demonstration of laboratory work.
"""

# pylint: disable=too-many-locals, undefined-variable, unused-import
import json
from pathlib import Path
from core_utils.llm.time_decorator import report_time
from lab_7_llm.main import RawDataImporter, RawDataPreprocessor


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    root_path = Path(__file__).parent
    with open(root_path / 'settings.json', 'r', encoding='utf-8') as file:
        settings = json.load(file)

    dataset_name = settings['parameters']['dataset']
    importer = RawDataImporter(dataset_name)
    importer.obtain()

    if importer.raw_data is None:
        raise ValueError("Raw Data is None after obtaining")

    preprocessor = RawDataPreprocessor(importer.raw_data)

    analysis = preprocessor.analyze()
    print("Dataset Analysis:", analysis)

    preprocessor.transform()

    result = preprocessor.data

    print("First 5 rows of preprocessed data:")
    print(result.head())

    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
