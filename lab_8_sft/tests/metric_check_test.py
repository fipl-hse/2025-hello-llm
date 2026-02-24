"""
Checks that E2E scenario allows to get desired metrics values
"""

import unittest
from pathlib import Path

import pytest

from lab_7_llm.tests.metric_check_test import run_metrics_check
from lab_8_sft.main import TaskEvaluator
from lab_8_sft.start import main


class MetricCheckTest(unittest.TestCase):
    """
    Tests e2e scenario
    """

    @pytest.mark.lab_8_sft
    @pytest.mark.mark6
    @pytest.mark.mark8
    def test_e2e_ideal(self) -> None:
        """
        Ideal metrics check scenario
        """
        self.assertTrue(
            run_metrics_check(Path(__file__).parent.parent, main, task_evaluator=TaskEvaluator)
        )
