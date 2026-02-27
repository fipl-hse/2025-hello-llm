"""
Fine-tuning starter.
"""

# pylint: disable=too-many-locals, undefined-variable, unused-import
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
    current_path = Path(__file__).parent
    settings = LabSettings(current_path / "settings.json")

    importer = RawDataImporter(settings.parameters.dataset)
    importer.obtain()

    if importer.raw_data is None:
        return

    preprocessor = RawDataPreprocessor(importer.raw_data)
    preprocessor.transform()

    dataset = TaskDataset(preprocessor.data.head(101))
    pipeline = LLMPipeline(
        model_name=settings.parameters.model,
        dataset=dataset,
        batch_size=64,
        max_length=120,
        device="cpu"
    )

    predictions_path = Path(__file__).parent / "dist" / "predictions.csv"
    predictions_path.parent.mkdir(parents=True, exist_ok=True)

    if not predictions_path.exists():
        pipeline.infer_dataset().to_csv(predictions_path, index=False)
    else:
        predictions_df = pipeline.infer_dataset()
        predictions_df.to_csv(predictions_path)

    metrics = [Metrics(metric) for metric in settings.parameters.metrics]
    evaluator = TaskEvaluator(predictions_path, metrics)

    print('\nEvaluation:')
    for key, value in evaluator.run().items():
        print(f'{key}: {value}')

    num_samples = 100

    sft_params = SFTParams(batch_size=3, max_length=120, max_fine_tuning_steps=50,
                           learning_rate=1e-3,
                           finetuned_model_path=Path(__file__).parent / 'dist' / settings.parameters.model,
                           device='cpu', rank=8, alpha=8)

    fine_tune_samples = sft_params.batch_size * sft_params.max_fine_tuning_steps
    tokenized_dataset = TokenizedTaskDataset(preprocessor.data.loc[
                                       num_samples: num_samples + fine_tune_samples
                                   ], tokenizer=AutoTokenizer.from_pretrained(settings.parameters.model),
        max_length=sft_params.max_length)

    sft_pipeline = SFTPipeline(settings.parameters.model, tokenized_dataset, sft_params)

    print('\nFine-tuning of the model:')
    sft_pipeline.run()

    finetuned_pipeline = LLMPipeline(str(sft_params.finetuned_model_path),
                                     dataset, 120, 64, 'cpu')

    print('\nAnalysis of the fine-tuned model:')
    for key, value in finetuned_pipeline.analyze_model().items():
        print(f'{key}: {value}')

    print('\nInference of the fine-tuned model:')
    sample = dataset[0]

    print(f"Text: {sample[0]}")
    print(f"Label: {sample[1]}")
    print(f"Predicted label: {finetuned_pipeline.infer_sample(sample)}")

    finetuned_preds = finetuned_pipeline.infer_dataset()
    finetuned_preds.to_csv(Path(__file__).parent / 'dist' / 'predictions.csv')

    result = TaskEvaluator(Path(__file__).parent / 'dist' / 'predictions.csv', metrics)

    print('\nEvaluation of the quality of the fine-tuned model:')
    for key, value in result.run().items():
        print(f'{key}: {value}')

    result = evaluator
    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
