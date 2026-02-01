"""
Starter for demonstration of laboratory work.
"""

# pylint: disable=too-many-locals, undefined-variable, unused-import
import json
import types
from pathlib import Path
from lab_7_llm.main import LLMPipeline, RawDataImporter, RawDataPreprocessor,TaskEvaluator, TaskDataset, report_time

@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    with open(Path(__file__).parent / 'settings.json', 'r', encoding='utf-8') as f:
        settings = json.load(f, object_hook=lambda d: types.SimpleNamespace(**d))

    importer = RawDataImporter(settings.parameters.dataset)
    importer.obtain()

    preprocessor = RawDataPreprocessor(importer.raw_data) #_raw_data
    dataset_stats = preprocessor.analyze()
    print(dataset_stats)

    preprocessor.transform()
    dataset = TaskDataset(preprocessor.data.head(100))

    batch_size = 1
    max_length = 120
    device = 'cpu'

    pipeline = LLMPipeline(settings.parameters.model, dataset, max_length, batch_size, device) 
    model_stats = pipeline.analyze_model()
    print(model_stats)

    sample = dataset[0]
    text = sample[0]
    sample_infer = pipeline.infer_sample(text)
    print(sample_infer)

    result = pipeline.infer_dataset()
    print(result)

    predictions_path = Path(__file__).parent / 'predictions.csv'
    result.to_csv(predictions_path)

    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
