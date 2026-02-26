"""
HuggingFace model listing.
"""

try:
    import torch
except ImportError:
    print('Library "torch" not installed. Failed to import.')

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
except ImportError:
    print('Library "transformers" not installed. Failed to import.')


def main() -> None:
    """
    Entrypoint for the listing.
    """
    # 1. Import tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # 2. Convert text to tokens
    text = (
        "Ron DeSantis’ fraught presidential campaign ended Sunday following a months-long downward"
    )
    tokens = tokenizer(text, return_tensors="pt")

    raw_tokens = tokenizer.convert_ids_to_tokens(tokens["input_ids"].tolist()[0])
    print(raw_tokens)

    # 3. Load model
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.eval()

    # Predict next token
    with torch.no_grad():
        output = model(**tokens).logits[0]

    # 4. next token is stored in last row
    last_token_predictions = output[-1]
    next_token_id = torch.argmax(last_token_predictions).item()

    # Shock content: GPT-2 from 2018 predicts continuation from 2024!
    print(next_token_id)
    print(tokenizer.decode((next_token_id,)))

    # 5. Generate text of given length
    with torch.no_grad():
        output = model.generate(
            **tokens,
            generation_config=GenerationConfig(
                max_new_tokens=10,
            ),
        )
    results = tokenizer.batch_decode(output, skip_special_tokens=True)
    print(results[0])

    # 6. Use temperature
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=120)
    print("Temperature: 0")
    with torch.no_grad():
        output = model.generate(
            **tokens,
            generation_config=GenerationConfig(
                do_sample=True,
                temperature=0.1,
            ),
            pad_token_id=tokenizer.eos_token_id,
        )
    results = tokenizer.batch_decode(output, skip_special_tokens=True)
    print(results[0])
    print()

    print("Temperature: 0.7")
    with torch.no_grad():
        output = model.generate(
            **tokens,
            generation_config=GenerationConfig(
                do_sample=True,
                temperature=0.7,
            ),
            pad_token_id=tokenizer.eos_token_id,
        )
    results = tokenizer.batch_decode(output, skip_special_tokens=True)
    print(results[0])
    print()

    print("Temperature: 1")
    with torch.no_grad():
        output = model.generate(
            **tokens,
            generation_config=GenerationConfig(
                do_sample=True,
                temperature=1,
            ),
            pad_token_id=tokenizer.eos_token_id,
        )
    results = tokenizer.batch_decode(output, skip_special_tokens=True)
    print(results[0])
    print()


if __name__ == "__main__":
    main()
