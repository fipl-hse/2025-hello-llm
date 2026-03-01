"""
Web service for model inference.
"""

from pathlib import Path

# pylint: disable=too-few-public-methods
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from core_utils.project.lab_settings import LabSettings
from lab_7_llm.main import ColumnNames, LLMPipeline, TaskDataset


class Query(BaseModel):
    """
    Query model for the service.
    """
    question: str


def init_application() -> tuple:
    """
    Initialize core application.

    Returns:
        tuple: tuple of two objects, instance of FastAPI server and LLMPipeline pipeline.
    """

    current_path = Path(__file__).parent
    settings = LabSettings(current_path / "settings.json")

    dummy_df = pd.DataFrame(columns=[ColumnNames.SOURCE])
    dataset = TaskDataset(dummy_df)
    pipeline = LLMPipeline(
        model_name=settings.parameters.model,
        dataset=dataset,
        max_length=120,
        batch_size=1,
        device='cpu'
    )
    app = FastAPI()

    app.mount("/assets", StaticFiles(directory="assets"), name="assets")

    templates = Jinja2Templates(directory="assets")

    app.state.pipeline = pipeline
    app.state.templates = templates

    @app.get("/")
    async def root(request: Request):
        """Root endpoint returning the HTML page."""
        return app.state.templates.TemplateResponse("index.html", {"request": request})

    @app.post("/infer")
    async def infer(query: Query):
        """Main endpoint for model inference."""
        sample = (query.question,)
        prediction = app.state.pipeline.infer_sample(sample)
        return {"infer": prediction}

    return app, pipeline


app, pipeline = (None, None)
