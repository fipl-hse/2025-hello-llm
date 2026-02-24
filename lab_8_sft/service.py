"""
Web service for model inference.
"""

# pylint: disable=too-few-public-methods


def init_application() -> tuple:
    """
    Initialize core application.

    Returns:
        tuple: tuple of three objects, instance of FastAPI server, LLMPipeline and SFTPipeline.
    """


app, pre_trained_pipeline, fine_tuned_pipeline = (None, None, None)
