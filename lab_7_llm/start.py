"""
Starter for demonstration of laboratory work.
"""
import json

# pylint: disable=too-many-locals, undefined-variable, unused-import

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, EncoderDecoderModel
from core_utils.llm.time_decorator import report_time
from lab_7_llm.main import RawDataImporter, RawDataPreprocessor


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    tokenizer = AutoTokenizer.from_pretrained("dmitry-vorobiev/rubert_ria_headlines")

    text = "Нити Аайог исследует использование блокчейна в образовании, здравоохранении и сельском хозяйстве"
    tokens = tokenizer(text, return_tensors="pt")
    print(tokens)
    print(tokens.keys())

    raw_tokens = tokenizer.convert_ids_to_tokens(tokens["input_ids"].tolist()[0])
    print(raw_tokens)

    model = AutoModelForSeq2SeqLM.from_pretrained("dmitry-vorobiev/rubert_ria_headlines")

    model.to('cpu')

    model.eval()

    # with torch.no_grad():
    #     output = model(**tokens)

    # print(output.logits)
    # print(output.logits.shape)

    result = None
    with open ('settings.json', 'r', encoding='utf-8') as file:
        settings = json.load(file)

    name = settings['parameters']['dataset']

    data_importer = RawDataImporter(name)
    data_importer.obtain()

    data_preprocessor = RawDataPreprocessor(data_importer._raw_data)
    result = data_preprocessor.analyze()

    for key, value in result.items():
        print(f'{key} : {value}')
    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
