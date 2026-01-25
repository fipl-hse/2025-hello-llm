"""
Starter for demonstration of laboratory work.
"""

# pylint: disable=too-many-locals, undefined-variable, unused-import
import json
from pathlib import Path

from core_utils.llm.time_decorator import report_time
from main import RawDataImporter


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    settings_path = Path(__file__).parent / 'settings.json'
    with open(settings_path, 'r', encoding='utf-8') as file:
        settings = json.load(file)

    name = settings['parameters']['dataset']
    dataset_importer = RawDataImporter(name)
    dataset_importer.obtain()

    result = dataset_importer
    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
