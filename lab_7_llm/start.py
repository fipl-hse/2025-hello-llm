"""
Starter for demonstration of laboratory work.
"""

# pylint: disable=too-many-locals, undefined-variable, unused-import
import json
from pathlib import Path

from core_utils.llm.metrics import Metrics
from core_utils.llm.time_decorator import report_time
from lab_7_llm.main import (
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
    # 1. Загрузка настроек
    root_path = Path(__file__).parent
    with open(root_path / 'settings.json', 'r', encoding='utf-8') as file:
        settings = json.load(file)

    # 2. Загрузка данных
    importer = RawDataImporter(settings['parameters']['dataset'])
    importer.obtain()

    # 3. Препроцессинг
    preprocessor = RawDataPreprocessor(importer.raw_data)
    preprocessor.transform()

    # 4. Создание датасета (берем 50 примеров для теста скорости)
    dataset = TaskDataset(preprocessor.data.head(50))

    # 5. Инициализация пайплайна
    pipeline = LLMPipeline(
        model_name=settings['parameters']['model'],
        dataset=dataset,
        max_length=120,
        batch_size=10,
        device='cpu'
    )

    # 6. Анализ модели
    model_analysis = pipeline.analyze_model()
    print("Model Analysis:", model_analysis)

    # Инференс на всем датасете (на наших 50 примерах)
    predictions_df = pipeline.infer_dataset()

    # Сохраняем результаты в CSV
    predictions_path = root_path / 'assets' / 'predictions.csv'
    predictions_path.parent.mkdir(parents=True, exist_ok=True)

    predictions_df.to_csv(predictions_path, index=False)
    print(f"Predictions saved to {predictions_path}")

    # Оценка качества (ROUGE)
    evaluator = TaskEvaluator(
        data_path=predictions_path,
        metrics=[Metrics[metric.upper()] for metric in settings['parameters']['metrics']]
    )

    # Проверка для assert
    results = evaluator.run()
    print("Evaluation Results:", results)

    result = results
    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()