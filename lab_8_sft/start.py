"""
Fine-tuning starter.
"""

# pylint: disable=too-many-locals, undefined-variable, unused-import, too-many-branches, too-many-statements


from pathlib import Path
from core_utils.llm.time_decorator import report_time
from core_utils.project.lab_settings import LabSettings
from lab_8_sft.main import LLMPipeline, RawDataImporter, RawDataPreprocessor, TaskDataset


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    current_path = Path(__file__).parent
    settings = LabSettings(current_path / "settings.json")
    
    importer = RawDataImporter(settings.parameters.dataset)
    importer.obtain()

    preprocessor = RawDataPreprocessor(importer.raw_data)
    preprocessor.transform()

    dataset = TaskDataset(preprocessor.data.head(100))
    pipeline = LLMPipeline(
        model_name=settings.parameters.model,
        dataset=dataset,
        batch_size=1,
        max_length=120,
        device="cpu"
    )

    model_info = pipeline.analyze_model()
    for key, value in model_info.items():
        print(f"{key}: {value}")

    sample = dataset[0]
    prediction = pipeline.infer_sample(sample)

    print("\nSingle sample inference:")
    print("Source:", sample[0])
    print("Target:", sample[1])
    print("Prediction:", prediction)

    result = prediction
    assert result is not None, "Fine-tuning does not work correctly"


if __name__ == "__main__":
    main()
