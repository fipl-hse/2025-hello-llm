"""
Starter for demonstration of laboratory work.
"""

# pylint: disable=too-many-locals, undefined-variable, unused-import
import json
import types
from pathlib import Path
from lab_7_llm.main import LLMPipeline, RawDataImporter, RawDataPreprocessor, TaskDataset, report_time


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
    analyzed_dataset = preprocessor.analyze()
    preprocessor.transform()
    print(analyzed_dataset)

    dataset = TaskDataset(preprocessor.data.head(100))
    dataset_length = dataset.__len__()
    item = dataset.__getitem__(0)
    preprocessed_data = dataset.data

    result = preprocessed_data
    print(result)

    # pipeline = LLMPipeline(settings.parameters.model, dataset, max_length, batch_size, device)
    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
