"""
HuggingFace model listing.
"""

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )
except ImportError:
    print('Library "transformers" not installed. Failed to import.')

try:
    import torch
except ImportError:
    print('Library "torch" not installed. Failed to import.')


def main() -> None:
    """
    Entrypoint for the listing.
    """

    #########################
    # Classification scenario
    #########################

    # 1. Classification
    tokenizer = AutoTokenizer.from_pretrained("s-nlp/russian_toxicity_classifier")

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
    model = AutoModelForSequenceClassification.from_pretrained("s-nlp/russian_toxicity_classifier")

    # 5 Print model
    print(model)

    model.eval()

    # 6. Classify text
    with torch.no_grad():
        output = model(**tokens)

    # 7. Print prediction
    print(output.logits)
    print(output.logits.shape)

    # 8. Print label
    predictions = torch.argmax(output.logits).item()

    # 9. Print predictions
    print(predictions)

    # 10. Map with labels
    labels = model.config.id2label
    print(labels[predictions])

    #########################
    # Generation scenario
    #########################

    # 11. Import tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # 12. Convert text to tokens
    text = "Generate any text"
    tokens = tokenizer(text, return_tensors="pt")

    # 13. Print tokens keys
    print(tokens.keys())

    # 14. Load model
    model = AutoModelForCausalLM.from_pretrained("gpt2")

    # 15. Print model
    print(model)

    # 16. Generate text
    output = model.generate(**tokens)
    results = tokenizer.batch_decode(output, skip_special_tokens=True)
    print(results[0])


if __name__ == "__main__":
    main()
