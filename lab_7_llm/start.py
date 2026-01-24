"""
Starter for demonstration of laboratory work.
"""

import json
from types import SimpleNamespace
from pathlib import Path

# pylint: disable=too-many-locals, undefined-variable, unused-import
from lab_7_llm.main import RawDataImporter, RawDataPreprocessor, report_time

SETTINGS_PATH = Path(__file__).resolve().with_name("settings.json")

@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    with SETTINGS_PATH.open("r", encoding="utf-8") as f:
        settings = json.load(f, object_hook=lambda d: SimpleNamespace(**d))

    importer = RawDataImporter(settings.parameters.dataset)
    importer.obtain()

    preprocessor = RawDataPreprocessor(importer.raw_data)
    analyzed_dataset = preprocessor.analyze()

    result = analyzed_dataset
    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
