"""
Starter for demonstration of laboratory work.
"""

# pylint: disable=too-many-locals, undefined-variable, unused-import


from lab_7_llm.main import RawDataImporter, RawDataPreprocessor

#@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    dataset_importer = RawDataImporter("blinoff/kinopoisk")
    dataset_importer.obtain()

    preprocessed_dataset = RawDataPreprocessor( dataset_importer.raw_data)
    preprocessed_dataset.transform()

    result = preprocessed_dataset.data
    print(result.head())
    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
