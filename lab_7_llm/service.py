"""
Web service for model inference.
"""

# pylint: disable=too-few-public-methods, undefined-variable, unused-import, assignment-from-no-return, duplicate-code
from pathlib import Path


def init_application() -> tuple[FastAPI, AbstractLLMPipeline]:
    """
    Initialize core application.

    Returns:
        tuple[FastAPI, AbstractLLMPipeline]: instance of server and pipeline
    """


app, pipeline = (None, None)
