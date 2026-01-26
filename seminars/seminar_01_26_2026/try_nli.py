"""
HuggingFace model listing.
"""

# pylint: disable=duplicate-code
try:
    import torch
except ImportError:
    print('Library "torch" not installed. Failed to import.')

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError:
    print('Library "transformers" not installed. Failed to import.')


def main() -> None:
    """
    Entrypoint for the listing.
    """

    # 1. Import tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    )

    # 2. Convert text to tokens
    premise = "Сократ - человек, а все люди смертны."
    hypothesis = "Сократ никогда не умрёт."
    tokens = tokenizer(premise, hypothesis, return_tensors="pt")

    # 3. Print tokens keys
    print(tokens.keys())

    # 4. Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    )

    # 5. Print model
    print(model)

    # 6. Inference model
    model.eval()
    with torch.no_grad():
        output = model(**tokens)

    # 7. Print logits
    print(output.logits)

    # 8. Get predict
    predicted_class = torch.argmax(torch.softmax(output.logits, -1), dim=1)
    print(f"Predicted class: {predicted_class.item()}")
    print(model.config.id2label)


if __name__ == "__main__":
    main()
