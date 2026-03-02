"""
Starter for demonstration of laboratory work.
"""

from pathlib import Path

from core_utils.project.lab_settings import LabSettings
from lab_7_llm.main import (
    LLMPipeline,
    RawDataImporter,
    RawDataPreprocessor,
    TaskDataset,
    TaskEvaluator,
)


def main() -> None:
    """
    Main function that runs the lab pipeline according to the target score.
    """
    settings_path = Path(__file__).parent / "settings.json"
    settings = LabSettings(settings_path)

    importer = RawDataImporter(settings.parameters.dataset)
    importer.obtain()
    print("Dataset downloaded successfully.")

    preprocessor = RawDataPreprocessor(importer.raw_data)
    analysis = preprocessor.analyze()
    print("Dataset analysis:")
    for key, value in analysis.items():
        print(f"  {key}: {value}")

    if settings.target_score < 6:
        return
    
    preprocessor.transform()
    print("Data preprocessing completed.")

    dataset = TaskDataset(preprocessor.data.head(100))
    print(f"Dataset size (first 100 samples): {len(dataset)}")

    # Initialize pipeline
    pipeline = LLMPipeline(
        model_name=settings.parameters.model,
        dataset=dataset,
        max_length=120,
        batch_size=1,
        device='cpu'
    )

    model_props = pipeline.analyze_model()
    print("Model properties:")
    for key, value in model_props.items():
        print(f"  {key}: {value}")

    sample = dataset[0]
    prediction = pipeline.infer_sample(sample)
    print(f"Sample inference: {prediction}")

    if settings.target_score >= 8:
        
        full_dataset = TaskDataset(preprocessor.data) 
        pipeline_full = LLMPipeline(
            model_name=settings.parameters.model,
            dataset=full_dataset,
            max_length=120,
            batch_size=64,
            device='cpu'
        )

        predictions_df = pipeline_full.infer_dataset()
        print(f"Predictions shape: {predictions_df.shape}")

        output_dir = Path(__file__).parent / "dist"
        output_dir.mkdir(exist_ok=True)
        predictions_path = output_dir / "predictions.csv"
        predictions_df.to_csv(predictions_path, index=False)
        print(f"Predictions saved to {predictions_path}")

        evaluator = TaskEvaluator(predictions_path, settings.parameters.metrics)
        metrics_results = evaluator.run()
        print("Evaluation results:")
        for metric, value in metrics_results.items():
            print(f"  {metric}: {value}")

    assert analysis is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
    