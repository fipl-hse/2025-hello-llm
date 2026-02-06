"""
Checks that the dataset is being analyzed correctly
"""

# pylint: disable=duplicate-code, assignment-from-no-return
import unittest
from pathlib import Path

import pytest

from admin_utils.references.reference_scores import ReferenceAnalysisScores
from core_utils.project.lab_settings import LabSettings
from lab_8_sft.main import RawDataImporter, RawDataPreprocessor
from lab_8_sft.start import main


class DatasetWorkingTest(unittest.TestCase):
    """
    Tests analyse function
    """

    @pytest.mark.lab_8_sft
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    def test_analyze_ideal(self) -> None:
        """
        Ideal analyze scenario
        """
        main()

        settings = LabSettings(Path(__file__).parent.parent / "settings.json")

        importer = RawDataImporter(settings.parameters.dataset)
        importer.obtain()

        if importer.raw_data is None:
            raise ValueError('"importer.raw_data" can not be None!')
        preprocessor = RawDataPreprocessor(importer.raw_data)

        dataset_analysis = preprocessor.analyze()

        references = ReferenceAnalysisScores()

        self.assertEqual(references.get(settings.parameters.dataset), dataset_analysis)

        self.assertEqual(6, len(dataset_analysis))
