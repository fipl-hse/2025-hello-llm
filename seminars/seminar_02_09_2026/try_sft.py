"""
Parameter Efficient Fine-Tuning listing.
"""

# pylint: disable=duplicate-code,too-many-locals


try:
    import torch
    from torch import argmax
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    print('Library "torch" not installed. Failed to import.')
    DataLoader = None  # type: ignore
    Dataset = None  # type: ignore

try:
    from pandas import DataFrame
except ImportError:
    print('Library "pandas" not installed. Failed to import.')
    DataFrame = None  # type: ignore

try:
    from datasets import load_dataset
except ImportError:
    print('Library "datasets" not installed. Failed to import.')
    load_dataset = None  # type: ignore

try:
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        set_seed,
        Trainer,
        TrainingArguments,
    )
except ImportError:
    print('Library "transformers" not installed. Failed to import.')

try:
    from peft import get_peft_model, LoraConfig
except ImportError:
    print('Library "peft" not installed. Failed to import.')

try:
    from evaluate import load
except ImportError:
    print('Library "evaluate" not installed. Failed to import.')


class TaskDataset(Dataset):  # type: ignore
    """
    Dataset with translation data.
    """

    def __init__(self, data: DataFrame) -> None:
        """
        Initialize an instance of TaskDataset.

        Args:
            data (pandas.DataFrame): original data.
        """
        self._data = data

    def __len__(self) -> int:
        """
        Return the number of items in the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        return len(self._data)

    def __getitem__(self, index: int) -> str:
        """
        Retrieve an item from the dataset by index.

        Args:
            index (int): Index of sample in dataset

        Returns:
            str: The item to be received
        """
        return str(self._data["comment"].iloc[index])


def main() -> None:
    """
    Entrypoint for the listing.
    """
    set_seed(42)
    # 1. Load dataset
    data = load_dataset("s-nlp/en_paradetox_toxicity", split="train").to_pandas()
    data = data.rename(columns={"text": "source"})
    eval_dataset = TaskDataset(data.iloc[:50])
    references = data["toxic"].iloc[:50]

    # 2. Get data loader with batch 4
    dataset_loader = DataLoader(eval_dataset, batch_size=4)
    print(len(dataset_loader))

    # 3. Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("IlyaGusev/rubertconv_toxic_clf")
    model = AutoModelForSequenceClassification.from_pretrained("IlyaGusev/rubertconv_toxic_clf")

    # 4. Evaluate pre-trained model
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch_data in dataset_loader:
            ids = tokenizer(batch_data, padding=True, truncation=True, return_tensors="pt")
            output = model(**ids).logits
            predictions.extend(argmax(output, dim=1).tolist())

    # 5. Print predictions
    print("Predictions:", predictions)
    print("References:", references)

    # 6. Load metric
    accuracy_metric = load("accuracy")
    print("Metric name:", accuracy_metric.name)

    # 7. Compute accuracy
    print(accuracy_metric.compute(references=references, predictions=predictions))

    # 8. Prepare model
    lora_config = LoraConfig(r=8, lora_alpha=8, lora_dropout=0.1, target_modules=None)

    # 9. Prepare dataset
    data_to_fine_tune = data.iloc[100:400]
    data_to_fine_tune.reset_index(drop=True)

    tokenized_data_to_fine_tune = []
    for _, sample in data_to_fine_tune.iterrows():
        tokenized_sample = tokenizer(
            sample["comment"],
            padding="max_length",
            truncation=True,
            max_length=120,
            return_tensors="pt",
        )
        tokenized_data_to_fine_tune.append(
            {
                "input_ids": tokenized_sample["input_ids"][0],
                "attention_mask": tokenized_sample["attention_mask"][0],
                "labels": int(sample["toxic"]),
            }
        )

    # 10. Create fine-tuning arguments
    finetuned_model_path = "./dist"
    training_args = TrainingArguments(
        output_dir=finetuned_model_path,
        max_steps=50,
        per_device_train_batch_size=3,
        learning_rate=1e-4,
        save_strategy="no",
        use_cpu=True,
        load_best_model_at_end=True,
    )

    # 11. Fine-tune
    trainer = Trainer(
        model=get_peft_model(model, lora_config),
        args=training_args,
        train_dataset=tokenized_data_to_fine_tune,
    )
    trainer.train()

    # 12. Save model
    trainer.model.merge_and_unload()
    trainer.model.base_model.save_pretrained(finetuned_model_path)

    # 13. Load fine-tuned model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(finetuned_model_path)

    # 14. Evaluate fine-tuned model
    predictions = []
    model.eval()
    with torch.no_grad():
        for batch_data in dataset_loader:
            ids = tokenizer(batch_data, padding=True, truncation=True, return_tensors="pt")
            output = trainer.model(**ids).logits
            predictions.extend(argmax(output, dim=1).tolist())

    # 15. Print predictions
    print("Predictions:", predictions)
    print("References:", references)

    # 16. Load metric
    accuracy_metric = load("accuracy")
    print("Metric name:", accuracy_metric.name)

    # 17. Compute accuracy
    print(accuracy_metric.compute(references=references, predictions=predictions))


if __name__ == "__main__":
    main()
