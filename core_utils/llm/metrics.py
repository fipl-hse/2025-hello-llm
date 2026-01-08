"""
Metrics enum.
"""

import enum


class Metrics(enum.Enum):
    """
    Metrics enum.
    """

    BLEU = "bleu"
    ROUGE = "rouge"
    SQUAD = "squad"
    F1 = "f1"
    PRECISION = "precision"
    RECALL = "recall"
    ACCURACY = "accuracy"

    def __str__(self) -> str:
        """
        String representation of a metric.

        Returns:
             str: Name of a metric
        """
        return self.value
