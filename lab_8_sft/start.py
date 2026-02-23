"""
Fine-tuning starter.
"""

from pathlib import Path

from core_utils.llm.metrics import Metrics
from core_utils.project.lab_settings import LabSettings
from lab_8_sft.main import (
    LLMPipeline,
    RawDataImporter,
    RawDataPreprocessor,
    report_time,
    TaskDataset,
    TaskEvaluator,
)

# pylint: disable=too-many-locals, undefined-variable, unused-import, too-many-branches, too-many-statements


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    current_path = Path(__file__).parent
    settings = LabSettings(current_path / "settings.json")

    importer = RawDataImporter(settings.parameters.dataset)
    importer.obtain()

    if importer.raw_data is None:
        raise ValueError(
            '''"RawDataPreprocessor" has incompatible type
                          "DataFrame | None"; expected "DataFrame"'''
        )

    preprocessor = RawDataPreprocessor(importer.raw_data)
    preprocessor.transform()

    dataset = TaskDataset(preprocessor.data.head(100))
    pipeline = LLMPipeline(
        model_name=settings.parameters.model,
        dataset=dataset,
        max_length=120,
        batch_size=64,
        device="cpu",
    )

    model_properties = pipeline.analyze_model()
    for k, v in model_properties.items():
        print(f"{k}: {v}")

    print(pipeline.infer_sample(dataset[0]))

    predictions_path = Path(__file__).parent / "dist" / "predictions.csv"
    predictions_path.parent.mkdir(exist_ok=True)

    predictions = pipeline.infer_dataset()
    predictions.to_csv(predictions_path)

    evaluator = TaskEvaluator(
        predictions_path, [Metrics(metric) for metric in settings.parameters.metrics]
    )
    result = evaluator.run()
    print(result)

    # num_samples = 100
    # fine_tune_samples = batch * fine_tuning_steps
    # dataset = TokenizedTaskDataset(preprocessor.data.loc[
    #         num_samples : num_samples + fine_tune_samples
    #     ])

    # pipeline = SFTPipeline(settings.parameters.model, dataset, sft_params)

    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
