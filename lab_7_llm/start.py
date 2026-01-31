"""
Starter for demonstration of laboratory work.
"""

import json
from pathlib import Path

from core_utils.llm.raw_data_preprocessor import AbstractRawDataPreprocessor, ColumnNames
from core_utils.project.lab_settings import LabSettings

# pylint: disable=too-many-locals, undefined-variable, unused-import
from lab_7_llm.main import (
    LLMPipeline,
    RawDataImporter,
    RawDataPreprocessor,
    report_time,
    TaskDataset,
)


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    from admin_utils.references.reference_scores import ReferenceAnalysisScores, ReferenceAnalysisScoresType

    print("INFERENCE keys:", list(ReferenceAnalysisScores(ReferenceAnalysisScoresType.INFERENCE)._dto.keys()))
    print("MODEL keys:", list(ReferenceAnalysisScores(ReferenceAnalysisScoresType.MODEL)._dto.keys()))


if __name__ == "__main__":
    main()
