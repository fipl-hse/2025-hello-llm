"""
HuggingFace evaluate listing.
"""

# pylint: disable=duplicate-code
from transformers import (
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        GenerationConfig,
        DebertaV2ForQuestionAnswering,
    )

try:
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
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError:
    print('Library "transformers" not installed. Failed to import.')

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
        return str(self._data["question"].iloc[index]), str(self._data['context'].iloc[index])


def main() -> None:
    """
    Entrypoint for the listing.
    """

    # 1. Load dataset
    data = load_dataset("lionelchg/dolly_closed_qa", split='test').to_pandas()
    data = data[["instruction", "context", "response"]]
    data = data.rename(
        columns={"instruction": "question", "response": "target"}
    )
    dataset = TaskDataset(data.head(10))
    references = data["target"].head(10)

    # 2. Get data loader with batch 4
    dataset_loader = DataLoader(dataset, batch_size=3)
    print(len(dataset_loader))

    # 3. Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("timpal0l/mdeberta-v3-base-squad2")
    model = DebertaV2ForQuestionAnswering.from_pretrained("timpal0l/mdeberta-v3-base-squad2")

    # 4. Inference dataset
    predictions = []
    for batch_data in dataset_loader:
        ids = tokenizer((batch_data[0][0], batch_data[1][0]), padding=True, truncation=True, return_tensors="pt")
        start_ids, end_ids = model(return_dict = False, **ids)
        output = []
        for i, el in enumerate(ids.input_ids):
            start = argmax(start_ids[i])
            end = argmax(end_ids[i])
            output.extend(list(el[start:end+1]))

        predictions.extend(output)

    # 5. Print predictions
    print("Predictions:", predictions)
    print("References:", references)

    # 6. Load metric
    accuracy_metric = load("squad")
    print("Metric name:", accuracy_metric.name)

    # 7. Compute accuracy
    print(accuracy_metric.compute(references=references, predictions=predictions))


if __name__ == "__main__":
    main()
