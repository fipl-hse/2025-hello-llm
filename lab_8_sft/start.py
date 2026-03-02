"""
Fine-tuning starter.
"""

# pylint: disable=too-many-locals, undefined-variable, unused-import, too-many-branches, too-many-statements
from pathlib import Path

from core_utils.llm.time_decorator import report_time
from core_utils.project.lab_settings import LabSettings
from lab_8_sft.main import (
    LLMPipeline,
    RawDataImporter,
    RawDataPreprocessor,
    SFTPipeline,
    TaskDataset,
    TaskEvaluator,
    TokenizedTaskDataset,
)


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    result = None
    settings_path = Path(__file__).parent / "settings.json"
    settings = LabSettings(settings_path)

    importer = RawDataImporter(settings.parameters.dataset)
    importer.obtain()
    print("Dataset downloaded successfully.")
    preprocessor = RawDataPreprocessor(importer.raw_data)
    analysis = preprocessor.analyze()
    result = analysis
    print("Dataset analysis:")
    for key, value in analysis.items():
        print(f"  {key}: {value}")

    if settings.target_score >= 6:
        preprocessor.transform()
        print("Data preprocessing completed.")

        inference_num_samples = 100
        dataset = TaskDataset(preprocessor.data.head(inference_num_samples))
        print(f"Dataset size (first {inference_num_samples} samples): {len(dataset)}")

        pipeline = LLMPipeline(
            model_name=settings.parameters.model,
            dataset=dataset,
            max_length=120,
            batch_size=1,
            device="cpu",
        )

        model_props = pipeline.analyze_model()
        print("Model properties:")
        for key, value in model_props.items():
            print(f"  {key}: {value}")
        sample = dataset[0]
        prediction = pipeline.infer_sample(sample)
        print(f"Sample inference: {prediction}")

    assert result is not None, "Fine-tuning does not work correctly"


if __name__ == "__main__":
    main()
