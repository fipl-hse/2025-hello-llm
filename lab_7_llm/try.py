from datasets import load_dataset
from transformers import (
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        GenerationConfig,
    )

dataset = load_dataset("dolly_closed_qa")

# 4. Import model
model = AutoModelForSequenceClassification.from_pretrained("timpal0l/mdeberta-v3-base-squad2")
tokenizer = AutoTokenizer.from_pretrained("timpal0l/mdeberta-v3-base-squad2")