"""
Fine-tuning starter.
"""

from pathlib import Path

from core_utils.llm.metrics import Metrics
from core_utils.llm.time_decorator import report_time
from core_utils.project.lab_settings import LabSettings
from lab_8_sft.main import (
    RawDataImporter,
    RawDataPreprocessor,
    TaskDataset
)

# pylint: disable=too-many-locals, undefined-variable, unused-import, too-many-branches, too-many-statements


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    current_path = Path(__file__).parent
    settings = LabSettings(current_path / "settings.json")

    dataset_importer = RawDataImporter(settings.parameters.dataset)
    dataset_importer.obtain()

    if dataset_importer.raw_data is None:
        return
    dataset_processor = RawDataPreprocessor(dataset_importer.raw_data)
    print(dataset_processor.analyze())

    dataset_processor.transform()

    dataset = TaskDataset(dataset_processor.data.head(100))
    print(dataset)

    result = dataset
    assert result is not None, "Fine-tuning does not work correctly"


if __name__ == "__main__":
    main()
