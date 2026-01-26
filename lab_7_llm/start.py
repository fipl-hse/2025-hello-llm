"""
Starter for demonstration of laboratory work.
"""

# pylint: disable=too-many-locals, undefined-variable, unused-import
from lab_7_llm.main import RawDataImporter

@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    dataset_importer = RawDataImporter("s-nlp/ru_paradetox_toxicity")
    dataset_importer.obtain()
    processed_dataset = RawDataImporter.process(dataset_importer.raw_data)
    result = processed_dataset

    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
