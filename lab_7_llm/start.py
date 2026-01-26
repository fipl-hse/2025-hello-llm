"""
Starter for demonstration of laboratory work.
"""
from pathlib import Path

from core_utils.llm.time_decorator import report_time
from core_utils.project.lab_settings import LabSettings
# pylint: disable=too-many-locals, undefined-variable, unused-import


from lab_7_llm.main import RawDataImporter, RawDataPreprocessor


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    current_path = Path(__file__).parent
    settings = LabSettings(current_path / "settings.json")

    dataset_importer = RawDataImporter(settings.parameters.dataset)
    dataset_importer.obtain()

    preprocessed_dataset = RawDataPreprocessor(dataset_importer.raw_data)
    # preprocessed_dataset.analyze()
    # preprocessed_dataset.transform()

    result = preprocessed_dataset.analyze()
    print(result)
    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
