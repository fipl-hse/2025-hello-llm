"""
Starter for demonstration of laboratory work.
"""

import json
from pathlib import Path
from types import SimpleNamespace

# pylint: disable=too-many-locals, undefined-variable, unused-import
from lab_7_llm.main import RawDataImporter, RawDataPreprocessor, report_time

try:
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        XLMRobertaForSequenceClassification
    )
except ImportError:
    print('Library "transformers" not installed. Failed to import.')

try:
    from torchinfo import summary  # type: ignore
except ImportError:
    print('Library "torchinfo" not installed. Failed to import.')

try:
    import torch
except ImportError:
    print('Library "torch" not installed. Failed to import.')


SETTINGS_PATH = Path(__file__).resolve().with_name("settings.json")

@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    with SETTINGS_PATH.open("r", encoding="utf-8") as f:
        settings = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
    importer = RawDataImporter(settings.parameters.dataset)
    importer.obtain()

    preprocessor = RawDataPreprocessor(importer.raw_data)
    analysis = preprocessor.analyze()

    tokenizer = AutoTokenizer.from_pretrained("papluca/xlm-roberta-base-language-detection")

    text = "KFC заработал в Нижнем под новым брендом"
    tokens = tokenizer(text, return_tensors="pt")

    print(tokens.keys())

    raw_tokens = tokenizer.convert_ids_to_tokens(tokens["input_ids"].tolist()[0])
    print(raw_tokens)

    print(tokens["input_ids"].tolist()[0])

    model = AutoModelForSequenceClassification.from_pretrained("papluca/xlm-roberta-base-language-detection")

    print(model)

    model.eval()

    with torch.no_grad():
        output = model(**tokens)

    print(output.logits)
    print(output.logits.shape)

    predictions = torch.argmax(output.logits).item()

    print(predictions)

    labels = model.config.id2label
    print(labels[predictions])


    model = XLMRobertaForSequenceClassification.from_pretrained("papluca/xlm-roberta-base-language-detection")
    print(type(model))

    config = model.config

    print(config)

    embeddings_length = config.max_position_embeddings

    ids = torch.ones(1, embeddings_length, dtype=torch.long)

    tokens = {"input_ids": ids, "attention_mask": ids}

    result = summary(model, input_data=tokens, device="cpu", verbose=0)

    print(result)

    inp_shape = result.input_size
    print(f"input_shape:\n{inp_shape}\n")

    n_params = result.total_params
    print(f"num_trainable_params:\n{n_params}\n")

    summary_list = result.summary_list
    print(f"summary_list:\n{summary_list}\n")

    total_p_bytes = result.total_param_bytes
    print(f"total_param_bytes:\n{total_p_bytes}\n")


    result = analysis
    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
