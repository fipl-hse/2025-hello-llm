"""
Starter for demonstration of laboratory work.
"""

from pathlib import Path

from core_utils.llm.time_decorator import report_time
from core_utils.project.lab_settings import LabSettings
from lab_7_llm.main import LLMPipeline, RawDataImporter, RawDataPreprocessor, TaskDataset

# pylint: disable=too-many-locals, undefined-variable, unused-import

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

    print("Source (ru):", sample[0])
    print("Target (es):", sample[1])
    print("Prediction:", prediction)

    result = prediction
    assert result is not None, "Demo does not work correctly"
    

if __name__ == "__main__":
    main()
