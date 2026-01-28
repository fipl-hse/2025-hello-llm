"""
Starter for demonstration of laboratory work.
"""
import json
from pathlib import Path
from types import SimpleNamespace

from lab_7_llm.main import (
    LLMPipeline,
    RawDataImporter,
    RawDataPreprocessor,
    report_time,
    TaskDataset,
)

# pylint: disable=too-many-locals, undefined-variable, unused-import


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    settings_path = Path(__file__).parent / 'settings.json'

    with open(settings_path, 'r', encoding='utf-8') as file:
        settings = json.load(file)

    name = settings['parameters']['dataset']

    importer = RawDataImporter(name)
    importer.obtain()

    preprocessor = RawDataPreprocessor(importer.raw_data)
    result = preprocessor.analyze()

    for key, value in result.items():
        print(f'{key} : {value}')

    assert result is not None, "Demo does not work correctly"

    dataset = TaskDataset(importer.raw_data)
    pipeline = LLMPipeline(
        model_name=settings['parameters']['model'],
        dataset=dataset,
        max_length=120,
        batch_size=1,
        device='cpu'
    )

    model_properties = pipeline.analyze_model()

    for key, value in model_properties.items():
        print(f"{key}: {value}")

    sample = dataset[0]
    input_text = sample[0]

    prediction = pipeline.infer_sample(sample)

    print(f"Generated summary:\n{prediction}\n")

    if len(sample) > 1:
        print(f"Reference summary:\n{sample[1]}")




if __name__ == "__main__":
    main()


