"""
Python protocols listing.
"""

# pylint: disable=too-few-public-methods, protected-access, not-an-iterable
from typing import Sequence


class DoNotKnowLenDataset:
    """
    Just a class with some state.
    """

    def __init__(self, data: Sequence[tuple[str, str, int, int]]) -> None:
        """
        Initialize an instance of DoNotKnowLenDataset.

        Args:
            data (Sequence[tuple[str, str, int, int]]): Raw data
        """
        self._data = data


class HaveLengthDataset(DoNotKnowLenDataset):
    """
    A class with some state that knows its length.
    """

    def __len__(self) -> int:
        """
        Returns number of samples in the dataset.


        Returns:
            int: Length of the dataset
        """
        return len(self._data)


class IterableDataset(DoNotKnowLenDataset):
    """
    A class with some state that knows its length and can be a part of for loop.
    """

    def __getitem__(self, item: int) -> tuple[str, str, int, int]:
        """
        Returns an sample from dataset by index.

        Args:
            item (int): index of a sample

        Returns:
            tuple[str, str, int, int]: Sample from the dataset
        """
        return self._data[item]


def main() -> None:
    """
    Entrypoint for the listing.
    """
    # 1. Data is taken from assets/glue_mnli_example.csv
    raw_data = (
        (
            "Есть ли вода на марсе?",
            "Гидросфера Марса — это совокупность водных запасов планеты Марс",
            0,
            1,
        ),
        (
            "Состоит ли англия в евросоюзе?",
            "В полночь с 31 января на 1 февраля 2020 года по центральноевропейскому времени "
            "после 47 лет членства Великобритания официально покинула Евросоюз.",
            1,
            0,
        ),
        (
            "Действительно ли в ссср не было адвокатов?",
            "Семён Львович Ария  — советский и российский юрист, один из "
            "крупнейших советских адвокатов",
            2,
            0,
        ),
    )

    # 2. Instantiate dataset
    plain_dataset = DoNotKnowLenDataset(raw_data)

    ###############################################################################################

    # Problems:
    #       - dataset does not know its length
    #       - dataset does not know how to be iterable for "for" loop

    ###############################################################################################

    # Problem 1. Automatic length collection via len() does not work
    try:
        print(f"Number of samples in CannotIterateDataset dataset: {len(plain_dataset)}") # type: ignore
    except TypeError:
        print("len() method is not supported for class CannotIterateDataset")

    # Workaround for Problem 1. To get length we need to know internal structure of the object:
    # how it stores data, access via protected attribute, which violates public API
    print(
        "Number of samples in CannotIterateDataset dataset "
        f"(WORKAROUND): {len(plain_dataset._data)}"
    )

    # Solution for Problem 1. To get length we use built-in len() that works
    # on top of implemented __len__ protocol. Public API is not violated

    len_dataset = HaveLengthDataset(raw_data)

    print(f"Number of samples in HaveLengthDataset dataset (LEN): {len(len_dataset)}")

    ###############################################################################################

    # Problem 2. Iterate over dataset samples
    try:
        for question, context, sample_id, answer in len_dataset:
            print(question, context, sample_id, answer)
    except TypeError:
        print("HaveLengthDataset is not iterable!")

    # Workaround for Problem 2. To iterate we need to know internal structure of the object:
    # how it stores data, access via protected attribute, which violates public API
    print("Iterating over HaveLengthDataset (WORKAROUND)")
    for question, context, sample_id, answer in len_dataset._data:
        print("\t", question, context, sample_id, answer)

    # Solution for Problem 2. To loop over dataset with "for" loop
    # need to implement __getitem__ protocol. Public API is not violated

    iter_dataset = IterableDataset(raw_data)

    print("Iterating over HaveLengthDataset (GET_ITEM)")
    for question, context, sample_id, answer in iter_dataset:
        print("\t", question, context, sample_id, answer)


if __name__ == "__main__":
    main()
