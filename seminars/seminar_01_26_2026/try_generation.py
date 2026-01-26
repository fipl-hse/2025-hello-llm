"""
HuggingFace model listing.
"""

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print('Library "transformers" not installed. Failed to import.')


def main() -> None:
    """
    Entrypoint for the listing.
    """
    # 1. Import tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # 2. Convert text to tokens
    text = "Generate any text"
    tokens = tokenizer(text, return_tensors="pt")

    # 3. Print tokens keys
    print(tokens.keys())

    # 4. Load model
    model = AutoModelForCausalLM.from_pretrained("gpt2")

    # 5. Print model
    print(model)

    # 6. Generate text
    output = model.generate(**tokens)
    results = tokenizer.batch_decode(output, skip_special_tokens=True)
    print(results[0])


if __name__ == "__main__":
    main()
