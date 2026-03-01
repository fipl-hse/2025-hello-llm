"""
Fine-tuning starter.
"""
from pathlib import Path

from transformers import AutoTokenizer

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

# pylint: disable=too-many-locals, undefined-variable, unused-import, too-many-branches, too-many-statements


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    result = None

    settings = LabSettings(Path(__file__).parent / "settings.json")
    model_name = settings.parameters.model

    importer = RawDataImporter(settings.parameters.dataset)
    importer.obtain()

    if importer.raw_data is None:
        return

    preprocessor = RawDataPreprocessor(importer.raw_data)
    dataset_stats = preprocessor.analyze()
    print(dataset_stats)

    preprocessor.transform()
    df = preprocessor.data
    num_samples = 100
    dataset = TaskDataset(df.head(num_samples))

    batch_size = 64
    max_length = 120
    device = 'cpu'

    pipeline = LLMPipeline(settings.parameters.model, dataset,
                            max_length, batch_size, device)

    model_stats = pipeline.analyze_model()
    print("Model analysis before sft:", model_stats)

    # Sample inference

    sample = dataset[0]
    sample_infer = pipeline.infer_sample(sample)
    print("Single-sample inference before SFT:", sample_infer)

    # Dataset inference

    dataset_preds = pipeline.infer_dataset()

    predictions_path = Path(__file__).parent / 'dist'/'predictions.csv'
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_preds.to_csv(predictions_path)

    evaluator = TaskEvaluator(data_path=predictions_path,
                              metrics=settings.parameters.metrics)

    result = evaluator.run()
    print("before:",result)

#   FINE-TUNING
    print("Fine-tuning:")
    num_samples = 10
    finetuned_model_path = Path(__file__).parent / 'dist' / f'{model_name}_finetuned'
    sft_params = SFTParams(
         batch_size=3,
         max_length=120,
         max_fine_tuning_steps=150,
         learning_rate=1e-3,
         finetuned_model_path=finetuned_model_path,
         device="cpu",
         rank=24,
         alpha=36,
         target_modules=["query", "key", "value", "dense"]
    )

    fine_tune_samples = sft_params.batch_size * sft_params.max_fine_tuning_steps
    train_df = df.iloc[num_samples: num_samples + fine_tune_samples]

    tokenizer = AutoTokenizer.from_pretrained(settings.parameters.model)
    tokenized_dataset = TokenizedTaskDataset(
        train_df, tokenizer, sft_params.max_length
    )

    sft_pipeline = SFTPipeline(
        model_name=model_name,
        dataset=tokenized_dataset,
        sft_params=sft_params,
        data_collator=None,
    )
    sft_pipeline.run()

    tokenizer.save_pretrained(finetuned_model_path)

    pipeline_for_sft = LLMPipeline(
        model_name=str(finetuned_model_path),
        dataset=dataset,
        max_length=120,
        batch_size=64,
        device="cpu"
    )

    tuned_model_stats = pipeline_for_sft.analyze_model()
    print("Model analysis after sft:", tuned_model_stats)

    # Sample inference
    print(pipeline_for_sft.infer_sample(dataset[0]))
    print("Single-sample inference after SFT:", sample_infer)

    # Dataset inference
    finetuned_predictions = pipeline_for_sft.infer_dataset()

    finetuned_predictions_file = Path(__file__).parent / 'dist' / 'prediction.csv'
    finetuned_predictions_file.parent.mkdir(parents=True, exist_ok=True)
    finetuned_predictions.to_csv(finetuned_predictions_file)

    evaluator = TaskEvaluator(finetuned_predictions_file, settings.parameters.metrics)
    result = evaluator.run()
    print("after:", result)

    assert result is not None, "Fine-tuning does not work correctly"


if __name__ == "__main__":
    main()
