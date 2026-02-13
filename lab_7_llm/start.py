"""
Starter for demonstration of laboratory work.
"""

# pylint: disable=too-many-locals, undefined-variable, unused-import
import json
from pathlib import Path

from core_utils.llm.time_decorator import report_time
from lab_7_llm.main import LLMPipeline, RawDataImporter, RawDataPreprocessor, TaskDataset


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

    # 4. Создание датасета (берем 10 примеров для теста скорости)
    dataset = TaskDataset(preprocessor.data.head(10))

    # 5. Инициализация пайплайна
    pipeline = LLMPipeline(
        model_name=settings['parameters']['model'],
        dataset=dataset,
        max_length=120,
        batch_size=1,
        device='cpu'
    )

    # 6. Анализ модели
    model_analysis = pipeline.analyze_model()
    print("Model Analysis:", model_analysis)

    # 7. Инференс одного примера
    sample = dataset[0]
    print(f"\nSource Text: {sample[0][:200]}...")  # Показываем начало текста
    print(f"Target Summary: {sample[1]}")

    prediction = pipeline.infer_sample(sample)
    print(f"Model Prediction: {prediction}")

    # Проверка для assert
    result = prediction
    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()