"""
Starter for demonstration of laboratory work.
"""
from pathlib import Path

from core_utils.llm.time_decorator import report_time
from core_utils.project.lab_settings import LabSettings
from lab_7_llm.main import (LLMPipeline, RawDataImporter, RawDataPreprocessor, TaskDataset)
# pylint: disable=too-many-locals, undefined-variable, unused-import


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    settings_path = Path(__file__).parent / 'settings.json'

    with open(settings_path, 'r', encoding='utf-8') as file:
        settings = json.load(file)

    importer = RawDataImporter(settings["parameters"]["dataset"])
    importer.obtain()

    preprocessor = RawDataPreprocessor(raw_data=importer.raw_data)
    preprocessor.transform()

    analyzed_data = preprocessor.analyze()

    for k, v in analyzed_data.items():
        print(f'{k} : {v}')

    dataset = TaskDataset(preprocessor.data.head(100))

    pipeline = LLMPipeline(settings["parameters"]["model"], dataset, 120, 1, "cpu")
    
    analysis_result = pipeline.analyze_model()
    print(analysis_result)

    print(pipeline.infer_sample(dataset[1]))


if __name__ == "__main__":
    main()

