"""
Starter for demonstration of laboratory work.
"""
from core_utils.llm.time_decorator import report_time
from datasets import load_dataset
import pandas
from pandas import DataFrame
from torchinfo import summary
import torch
from evaluate import load
from transformers import (
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        GenerationConfig,
        DebertaV2ForQuestionAnswering,
    )


# pylint: disable=too-many-locals, undefined-variable, unused-import


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    current_path = Path(__file__).parent
    settings = LabSettings(current_path / "settings.json")

    importer = RawDataImporter(settings.parameters.dataset)
    preprocessor = RawDataPreprocessor(importer.raw_data)



    result = True
    assert result is not None, "Demo does not work correctly"

    dataset = load_dataset("lionelchg/dolly_closed_qa", split='test')
    data = dataset.to_pandas()
    print(len(dataset))
    print(data.head(3))

    df_test = data[["instruction", "context", "response"]]
    print(df_test.head(3))
    df_test = df_test.rename(
        columns={"instruction": "question", "response": "target"}
    )
    df_test = df_test.reset_index(drop=True)
    print(df_test.head(3))


    tokenizer = AutoTokenizer.from_pretrained("timpal0l/mdeberta-v3-base-squad2")
# 2. Convert text to tokens
    text = (
        "Ron DeSantis’ fraught presidential campaign ended Sunday following a months-long downward"
    )
    tokens = tokenizer(text, return_tensors="pt")

    # 3. Print tokens keys
    print(tokens.keys())
    raw_tokens = tokenizer.convert_ids_to_tokens(tokens["input_ids"].tolist()[0])
    print(raw_tokens)


    # model = AutoModelForSequenceClassification.from_pretrained("timpal0l/mdeberta-v3-base-squad2")

    model2 = DebertaV2ForQuestionAnswering.from_pretrained("timpal0l/mdeberta-v3-base-squad2")
    # print(model)
    print(type(model2))
    model2.eval()

    


    # 2. Get model config
    # config = model.config
    config2 = model2.config

    # 3. Model configuration (reflects config.json on HuggingFace website)
    # print(config2)

    # 4. Model's max context size

    result = summary(
        model2,
        input_data={
            "input_ids": torch.ones(1, model2.config.max_position_embeddings, dtype=torch.long)
        },
    )
    shape = result.summary_list[-1].output_size
    # print(shape)
    input_shape = result.summary_list[0].input_size
    # print(input_shape)

    # print(num_params)
    # print(result)


    # # Predict next token
    # with torch.no_grad():
    #     output = model(**tokens).logits[0]

    # # 14. next token is stored in last row
    # last_token_predictions = output[-1]
    # next_token_id = torch.argmax(last_token_predictions).item()

    # # Shock content: GPT-2 from 2018 predicts continuation from 2024!
    # print(next_token_id)
    # print(tokenizer.decode((next_token_id,)))


    
    


if __name__ == "__main__":
    main()
