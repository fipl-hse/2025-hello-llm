"""
Starter for demonstration of laboratory work.
"""
import pathlib

from core_utils.llm.time_decorator import report_time
# pylint: disable=too-many-locals, undefined-variable, unused-import
from lab_7_llm.main import RawDataImporter, RawDataPreprocessor
from core_utils.project.lab_settings import LabSettings


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    path_to_settings = pathlib.Path(__file__).parent / 'settings.json'
    settings = LabSettings(path_to_settings)

    importer = RawDataImporter(settings.parameters.dataset)
    importer.obtain()

    preprocessor = RawDataPreprocessor(importer.raw_data)
    analysis_result = preprocessor.analyze()
    result = analysis_result

    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
