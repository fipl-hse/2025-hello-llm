"""
Fine-tuning starter.
"""

# pylint: disable=too-many-locals, undefined-variable, unused-import, too-many-branches, too-many-statements
import json
from pathlib import Path

from core_utils.llm.metrics import Metrics
from core_utils.llm.time_decorator import report_time
from lab_8_sft.main import (
    LLMPipeline,
    RawDataImporter,
    RawDataPreprocessor,
    TaskDataset,
    TaskEvaluator,
)


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    root_path = Path(__file__).parent
    with open(root_path / 'settings.json', 'r', encoding='utf-8') as file:
        settings = json.load(file)

    importer = RawDataImporter(settings['parameters']['dataset'])
    importer.obtain()

    if importer.raw_data is None:
        raise ValueError("Raw data is None")

    print("Dataset Columns:", importer.raw_data.columns)

    preprocessor = RawDataPreprocessor(importer.raw_data)
    preprocessor.transform()

    # берем 100 примеров для инференса
    dataset = TaskDataset(preprocessor.data.head(100))

    # инициализируем пайплайн с нужными параметрами
    pipeline = LLMPipeline(
        model_name=settings['parameters']['model'],
        dataset=dataset,
        max_length=120,
        batch_size=64,
        device='cpu'
    )

    model_analysis = pipeline.analyze_model()
    print("Model Analysis:", model_analysis)

    # демонстрация работы на одном сэмпле
    sample = dataset[0]
    print(f"Text: {sample[0]}")
    print(f"True Label: {sample[1]}")
    prediction = pipeline.infer_sample(sample)
    print(f"Prediction (Class ID): {prediction}")

    # инференс всего датасета
    predictions_df = pipeline.infer_dataset()

    # сохраняем предсказания в файл
    dist_dir = root_path / 'dist'
    dist_dir.mkdir(exist_ok=True)
    predictions_path = dist_dir / 'predictions.csv'
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Predictions saved to {predictions_path}")

    # оценка качества
    metrics = [Metrics[metric.upper()] for metric in settings['parameters']['metrics']]
    evaluator = TaskEvaluator(data_path=predictions_path, metrics=metrics)
    evaluation_results = evaluator.run()

    print("Evaluation Results:", evaluation_results)

    result = evaluation_results
    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
