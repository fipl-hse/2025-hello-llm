"""
Starter for demonstration of laboratory work.
"""
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

# pylint: disable=too-many-locals, undefined-variable, unused-import


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    result = None
    settings_path = Path(__file__).parent / 'settings.json'

    with open(settings_path, 'r', encoding='utf-8') as file:
        settings = json.load(file)

    name = settings['parameters']['dataset']

    data_importer = RawDataImporter(name)
    data_importer.obtain()

    if data_importer.raw_data is None:
        return

    data_preprocessor = RawDataPreprocessor(data_importer.raw_data)
    result = data_preprocessor.analyze()

    data_preprocessor.transform()

    dataset = TaskDataset(data_preprocessor.data.head(100))
    print(dataset)

    pipeline = LLMPipeline(
        model_name="dmitry-vorobiev/rubert_ria_headlines",
        dataset=dataset,
        max_length=120,
        batch_size=64,
        device='cpu'
    )

    model_properties = pipeline.analyze_model()
    for key, value in model_properties.items():
        print(f'{key}: {value}')

    sample = dataset[0]
    print(sample[0][:100])
    print(pipeline.infer_sample(sample))

    predictions_df = pipeline.infer_dataset()

    predictions_file = Path(__file__).parent / "dist" / "predictions.csv"

    predictions_file.parent.mkdir(parents=True, exist_ok=True)

    predictions_df.to_csv(predictions_file, index=False)

    metric_names = settings['parameters']['metrics']
    metrics = [Metrics[metric.upper()] for metric in metric_names]

    evaluator = TaskEvaluator(predictions_file, metrics)
    result = evaluator.run()

    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
