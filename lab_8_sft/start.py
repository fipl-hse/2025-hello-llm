"""
Fine-tuning starter.
"""
import json
from pathlib import Path

from core_utils.llm.metrics import Metrics
from lab_8_sft.main import (
    LLMPipeline,
    RawDataImporter,
    RawDataPreprocessor,
    report_time,
    TaskDataset,
    TaskEvaluator,
)

# pylint: disable=too-many-locals, undefined-variable, unused-import, too-many-branches, too-many-statements


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    settings_path = Path(__file__).parent / 'settings.json'

    with settings_path.open('r', encoding='utf-8') as file:
        settings = json.load(file)
    
    name = settings['parameters']['dataset']
    importer = RawDataImporter(hf_name=name)
    importer.obtain()

    preprocessor = RawDataPreprocessor(raw_data=importer.raw_data)
    result = preprocessor.analyze()

    for parameter, value in result.items():
        print(f'{parameter} : {value}')
    
    preprocessor.transform()
    dataset = TaskDataset(preprocessor.data.head(100))
    pipeline = LLMPipeline(
        model_name=settings['parameters']['model'],
        dataset=dataset,
        max_length=120,
        batch_size=1,
        device='cpu'
    )
    model_analysis = pipeline.analyze_model()
    for parameter, value in model_analysis.items():
        print(f'{parameter} : {value}')
    
    print(pipeline.infer_sample(dataset[0]))

    predictions_df = pipeline.infer_dataset()

    predictions_file = Path(__file__).parent / "dist" / "predictions.csv"

    predictions_file.parent.mkdir(parents=True, exist_ok=True)

    predictions_df.to_csv(predictions_file, index=False)

    metric_names = settings['parameters']['metrics']
    metrics = [Metrics[metric.upper()] for metric in metric_names]

    evaluator = TaskEvaluator(predictions_file, metrics)
    result = evaluator.run()

    assert result is not None, "Fine-tuning does not work correctly"


if __name__ == "__main__":
    main()
