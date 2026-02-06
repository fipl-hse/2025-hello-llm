"""
Starter for demonstration of laboratory work.
"""
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchinfo import summary
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification, \
    AutoModelForCausalLM, GenerationConfig, BertForSequenceClassification

from core_utils.llm.time_decorator import report_time


# pylint: disable=too-many-locals, undefined-variable, unused-import
import os

from lab_7_llm.main import TaskDataset, RawDataImporter

os.environ["TOKENIZERS_PARALLELISM"] = "true"

@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    tokenizer = AutoTokenizer.from_pretrained("Babelscape/wikineural-multilingual-ner")




    # 2. Convert text to tokens
    text = "KFC заработал в Нижнем под новым брендом"
    tokens = tokenizer(text, return_tensors="pt")

    # 3. Print tokens keys
    print(tokens.keys())

    raw_tokens = tokenizer.convert_ids_to_tokens(tokens["input_ids"].tolist()[0])
    print(raw_tokens)

    # line numbers with these IDs in vocab.txt (-1 because of zero indexing)
    print(tokens["input_ids"].tolist()[0])

    # 4. Import model
    model = AutoModelForTokenClassification.from_pretrained("Babelscape/wikineural-multilingual-ner")
    #
    # 5 Print model
    print(model)

    model.eval()

    # 6. Classify text
    with torch.no_grad():
        output = model(**tokens)

    # 7. Print prediction
    print(output.logits)
    print({output.logits.shape})

    # 8. Print label
    predictions = torch.argmax(output.logits, dim=-1)

    # 9. Преобразуем в список для удобства просмотра
    predictions_list = predictions.squeeze().tolist()
    print("Predictions for each token:", predictions_list)

    # 10. Map predictions with labels
    labels = model.config.id2label
    print("\nPredictions with labels:")
    for token, pred_id in zip(raw_tokens, predictions_list):
        print(f"{token:15} -> {pred_id:2} ({labels[pred_id]})")

    # 1.1 Import model - exact class name from config
    # model = BertForSequenceClassification.from_pretrained("s-nlp/russian_toxicity_classifier")
    print(type(model))

    # 1.2 Import model - auto class for particular task
    # model = AutoModelForSequenceClassification.from_pretrained("s-nlp/russian_toxicity_classifier")
    # print(type(model))

    # 2. Get model config
    config = model.config

    # 3. Model configuration (reflects config.json on HuggingFace website)
    print(config)

    # 4. Model's max context size
    # Pay attention to where it is defined for your model.
    # For some models it is called d_model, for some (generative ones) in decoder section
    embeddings_length = config.max_position_embeddings

    # 5. Imitating input data - fill full input of the model
    ids = torch.ones(1, embeddings_length, dtype=torch.long)

    # 6. Prepare data based on args of forward method of the corresponding model class
    tokens = {"input_ids": ids, "attention_mask": ids}

    # 7. Call summary method from torchinfo library
    result = summary(model, input_data=tokens, device="cpu", verbose=0)

    # 8. Resulting summary
    print(result)

    # 9. Get output shape
    shape = result.summary_list[-1].output_size
    print(shape)

    # ########################
    # Generation scenario
    ########################

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    print(type(model))
    result = summary(
        model,
        input_data={
            "input_ids": torch.ones(1, model.config.max_position_embeddings, dtype=torch.long)
        },
    )
    shape = result.summary_list[-1].output_size
    print(shape)

    seq = [
  "Tarek",
  "Mitri",
  ",",
  "Leiter",
  "der",
  "United",
  "Nations",
  "Support",
  "Mission",
  "in",
  "Libya",
  ",",
  "versuchte",
  "die",
  "Gespräche",
  "zwischen",
  "beiden",
  "Lagern",
  "wieder",
  "in",
  "Gang",
  "zu",
  "bringen",
  "."
]
    ner_tags = [
  1,
  2,
  0,
  0,
  0,
  3,
  4,
  4,
  4,
  4,
  4,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0
]
    tokens = tokenizer(seq, return_tensors="pt", truncation=True, padding=True, is_split_into_words=True)
    print(tokens)
    model.eval()

    # Predict next token
    with torch.no_grad():
        output = model(**tokens).logits[0]
    last_token_predictions = output[-1]
    next_token_id = torch.argmax(last_token_predictions).item()

    # Shock content: GPT-2 from 2018 predicts continuation from 2024!
    print(next_token_id)
    print(tokenizer.decode((next_token_id,)))#через словарь
#     ######

    import pandas as pd

    splits = {'test_de': 'data/test_de-00000-of-00001.parquet', 'test_en': 'data/test_en-00000-of-00001.parquet',
              'test_es': 'data/test_es-00000-of-00001.parquet', 'test_fr': 'data/test_fr-00000-of-00001.parquet',
              'test_it': 'data/test_it-00000-of-00001.parquet', 'test_nl': 'data/test_nl-00000-of-00001.parquet',
              'test_pl': 'data/test_pl-00000-of-00001.parquet', 'test_pt': 'data/test_pt-00000-of-00001.parquet',
              'test_ru': 'data/test_ru-00000-of-00001.parquet', 'train_de': 'data/train_de-00000-of-00001.parquet',
              'train_en': 'data/train_en-00000-of-00001.parquet', 'train_es': 'data/train_es-00000-of-00001.parquet',
              'train_fr': 'data/train_fr-00000-of-00001.parquet', 'train_it': 'data/train_it-00000-of-00001.parquet',
              'train_nl': 'data/train_nl-00000-of-00001.parquet', 'train_pl': 'data/train_pl-00000-of-00001.parquet',
              'train_pt': 'data/train_pt-00000-of-00001.parquet', 'train_ru': 'data/train_ru-00000-of-00001.parquet',
              'val_de': 'data/val_de-00000-of-00001.parquet', 'val_en': 'data/val_en-00000-of-00001.parquet',
              'val_es': 'data/val_es-00000-of-00001.parquet', 'val_fr': 'data/val_fr-00000-of-00001.parquet',
              'val_it': 'data/val_it-00000-of-00001.parquet', 'val_nl': 'data/val_nl-00000-of-00001.parquet',
              'val_pl': 'data/val_pl-00000-of-00001.parquet', 'val_pt': 'data/val_pt-00000-of-00001.parquet',
              'val_ru': 'data/val_ru-00000-of-00001.parquet'}

    data = pd.read_parquet("hf://datasets/Babelscape/wikineural/" + splits["val_en"])

    imp = RawDataImporter().obtain()
    prep = RawDataPreprocessor()

    # data = load_dataset("s-nlp/ru_paradetox_toxicity", split="train").to_pandas()
    dataset = TaskDataset(data.loc[:100])

    # 2. Get data loader with batch 1
    dataset_loader = DataLoader(dataset)
    print(len(dataset_loader))

    # 3. Print result
    print(next(iter(dataset_loader)))
    print(len(next(iter(dataset_loader))[0]))

    # 4. Get data loader with batch 4
    dataset_loader = DataLoader(dataset, batch_size=4)
    print(len(dataset_loader))

    # 5.
    print(next(iter(dataset_loader)))
    print(len(next(iter(dataset_loader))[0]))

    result = True
    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
