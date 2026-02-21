"""
Fine-tuning starter.
"""

# pylint: disable=too-many-locals, undefined-variable, unused-import, too-many-branches, too-many-statements
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

BASE_PATH = Path(__file__).parent


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    settings = LabSettings(BASE_PATH / 'settings.json')

    importer = RawDataImporter(settings.parameters.dataset)
    importer.obtain()
    if importer.raw_data is None:
        return

    preprocessor = RawDataPreprocessor(importer.raw_data)

    print('Dataset analysis:')
    for key, value in preprocessor.analyze().items():
        print(f'{key}: {value}')

    preprocessor.transform()
    dataset = TaskDataset(preprocessor.data.head(100))

    pipeline = LLMPipeline(settings.parameters.model, dataset, 120, 64, 'cpu')

    print('\nModel analysis:')
    for key, value in pipeline.analyze_model().items():
        print(f'{key}: {value}')

    predictions_path = BASE_PATH / 'dist' / 'predictions.csv'
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    pipeline.infer_dataset().to_csv(predictions_path)

    text, label = dataset[0]

    print('\nModel inference:')
    print(f'Text: {text}')
    print(f'Label: {label}')
    print(f'Predicted: {pipeline.infer_sample((text,))}')

    metrics = [Metrics(metric) for metric in settings.parameters.metrics]
    evaluator = TaskEvaluator(BASE_PATH / 'dist' / 'predictions.csv', metrics)

    print('\nEvaluation:')
    for key, value in evaluator.run().items():
        print(f'{key}: {value}')

    sft_params = SFTParams(batch_size=3, max_length=120, max_fine_tuning_steps=50,
                           learning_rate=1e-3,
                           finetuned_model_path=BASE_PATH / 'dist' / settings.parameters.model,
                           device='cpu', rank=8, alpha=8)

    num_samples = 10
    fine_tune_samples = sft_params.batch_size * sft_params.max_fine_tuning_steps
    tokenized_data = TokenizedTaskDataset(
        preprocessor.data.loc[num_samples: num_samples + fine_tune_samples],
        tokenizer=AutoTokenizer.from_pretrained(settings.parameters.model),
        max_length=sft_params.max_length
    )

    sft_pipeline = SFTPipeline(settings.parameters.model, tokenized_data, sft_params)

    print('\nFine-tuning model:')
    sft_pipeline.run()

    finetuned_pipeline = LLMPipeline(str(sft_params.finetuned_model_path),
                                     dataset, 120, 64, 'cpu')

    print('\nFine-tuned model analysis:')
    for key, value in finetuned_pipeline.analyze_model().items():
        print(f'{key}: {value}')

    print('\nFine-tuned model inference:')
    print(f'Text: {text}')
    print(f'Label: {label}')
    print(f'Predicted: {finetuned_pipeline.infer_sample((text,))}')

    finetuned_preds = finetuned_pipeline.infer_dataset()
    finetuned_preds.to_csv(BASE_PATH / 'dist' / 'predictions.csv')

    result = TaskEvaluator(BASE_PATH / 'dist' / 'predictions.csv', metrics)

    print('\nFine-tuned model evaluation:')
    for key, value in result.run().items():
        print(f'{key}: {value}')

    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
