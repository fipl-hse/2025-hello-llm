"""
Fine-tuning starter.
"""

# pylint: disable=too-many-locals, undefined-variable, unused-import, too-many-branches, too-many-statements

from pathlib import Path

from transformers import AutoTokenizer

from core_utils.llm.metrics import Metrics
from core_utils.llm.time_decorator import report_time
from core_utils.project.lab_settings import LabSettings, SFTParams
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
    settings = LabSettings(Path(__file__).parent / "settings.json")

    importer = RawDataImporter(settings.parameters.dataset)
    importer.obtain()

    if importer.raw_data is None:
        raise ValueError("DataFrame is expected")

    preprocessor = RawDataPreprocessor(importer.raw_data)
    dataset_analysis = preprocessor.analyze()
    preprocessor.transform()

    print(dataset_analysis)

    dataset = TaskDataset(preprocessor.data.head(100))
    pipeline = LLMPipeline(
        model_name=settings.parameters.model,
        dataset=dataset,
        max_length=120,
        batch_size=64,
        device="cpu",
    )
    model_analysis = pipeline.analyze_model()
    print(model_analysis)

    print(pipeline.infer_sample(dataset[0]))

    predictions_path = Path(__file__).parent / "dist" / "predictions.csv"
    predictions_path.parent.mkdir(exist_ok=True)

    predictions = pipeline.infer_dataset()
    predictions.to_csv(predictions_path)

    evaluator = TaskEvaluator(
        predictions_path, [Metrics(metric) for metric in settings.parameters.metrics]
    )
    result = evaluator.run()
    print(f"evaluation: {result}")

    print("fine-tuning: ")
    num_samples = 100
    finetuned_model_path = Path(__file__).parent / "dist" / settings.parameters.model
    sft_params = SFTParams(
        batch_size=3,
        max_length=120,
        max_fine_tuning_steps=50,
        learning_rate=1e-3,
        device="cpu",
        finetuned_model_path=finetuned_model_path,
        rank=8,
        alpha=8,
    )

    fine_tune_samples = sft_params.batch_size * sft_params.max_fine_tuning_steps
    tokenized_dataset = TokenizedTaskDataset(
        preprocessor.data.iloc[num_samples : num_samples + fine_tune_samples],
        tokenizer=AutoTokenizer.from_pretrained(settings.parameters.model),
        max_length=sft_params.max_length,
    )
    sft_pipeline = SFTPipeline(
        model_name=settings.parameters.model, dataset=tokenized_dataset, sft_params=sft_params
    )
    sft_pipeline.run()
    finetuned_pipeline = LLMPipeline(str(sft_params.finetuned_model_path), dataset, 120, 64, "cpu")
    for key, value in finetuned_pipeline.analyze_model().items():
        print(f"{key} : {value}")

    print(finetuned_pipeline.infer_sample(dataset[0]))
    finetuned_predictions = finetuned_pipeline.infer_dataset()
    finetuned_predictions_file = Path(__file__).parent / "dist" / "finetuned_predictions.csv"

    finetuned_predictions_file.parent.mkdir(parents=True, exist_ok=True)
    finetuned_predictions.to_csv(finetuned_predictions_file)
    evaluator = TaskEvaluator(finetuned_predictions_file, settings.parameters.metrics)
    result = evaluator.run()
    print("After fine-tuning:", result)

    assert result is not None, "Fine-tuning does not work correctly"


if __name__ == "__main__":
    main()
