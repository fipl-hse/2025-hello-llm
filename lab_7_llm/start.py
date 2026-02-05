"""
Starter for demonstration of laboratory work.
"""

# pylint: disable=too-many-locals, undefined-variable, unused-import
import json
from pathlib import Path

from core_utils.llm.time_decorator import report_time
from lab_7_llm.main import LLMPipeline, RawDataImporter, RawDataPreprocessor, TaskDataset


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    result = None
    settings_path = Path(__file__).parent / 'settings.json'

    with open(settings_path, 'r', encoding='utf-8') as file:
        settings = json.load(file)
    
    name = settings['parameters']['dataset']
    version = settings['parameters']['version']
    importer = RawDataImporter(hf_name=name, hf_version=version)
    importer.obtain()

    preprocessor = RawDataPreprocessor(raw_data=importer.raw_data)
    result = preprocessor.analyze()

    for parameter, value in result.items():
        print(f'{parameter} : {value}')
    
    preprocessor.transform()
    dataset = TaskDataset(preprocessor.data.head(100))
    pipeline = LLMPipeline(
        model_name=settings['parameters']['model'],
        dataset=TaskDataset,
        max_length=120,
        batch_size=1,
        device='cpu'
    )
    model_analysis = pipeline.analyze_model()
    for parameter, value in model_analysis.items():
        print(f'{parameter} : {value}')
    
    print(pipeline.infer_sample(dataset[0]))

    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
