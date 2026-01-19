"""
Starter for demonstration of laboratory work.
"""

# pylint: disable=too-many-locals, undefined-variable, unused-import
import json
from pathlib import Path

from config.lab_settings import LabSettings
from core_utils.llm.time_decorator import report_time
from lab_7_llm.main import RawDataImporter


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    settings = LabSettings(Path(__file__).parent / "settings.json")
    importer = RawDataImporter(settings.parameters.dataset)
    importer.obtain()

    result = importer
    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
