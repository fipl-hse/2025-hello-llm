"""
Starter for demonstration of laboratory work.
"""

# pylint: disable=too-many-locals, undefined-variable, unused-import
import json

from core_utils.llm.time_decorator import report_time
from main import RawDataImporter

@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    with open("settings.json", "r") as file:
        settings = json.load(file)

    importer = RawDataImporter(settings.parameters.dataset)
    result = None
    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
