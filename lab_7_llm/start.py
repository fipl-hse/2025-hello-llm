"""
Starter for demonstration of laboratory work.
"""
from pathlib import Path

from core_utils.llm.time_decorator import report_time
from core_utils.project.lab_settings import LabSettings
# pylint: disable=too-many-locals, undefined-variable, unused-import


from lab_7_llm.main import LLMPipeline, RawDataImporter, RawDataPreprocessor, TaskDataset


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    current_path = Path(__file__).parent
    settings = LabSettings(current_path / "settings.json")

    dataset_importer = RawDataImporter(settings.parameters.dataset)
    dataset_importer.obtain()

    dataset_processor = RawDataPreprocessor(dataset_importer.raw_data)
    print(dataset_processor.analyze())

    dataset_processor.transform()

    dataset = TaskDataset(dataset_processor.data.head(100))
    print(dataset)

    pipeline = LLMPipeline(
        model_name=settings.parameters.model,
        dataset=dataset,
        max_length=120,
        batch_size=1,
        device='cpu'
    )

    model_properties = pipeline.analyze_model()
    for key, value in model_properties.items():
        print(f'{key}: {value}')

    sample = dataset[0]
    print(sample[0][:100])

    result = pipeline.infer_sample(sample)
    print(result)
    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
