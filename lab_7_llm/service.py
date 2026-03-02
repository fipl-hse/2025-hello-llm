"""
Web service for model inference.
"""

# pylint: disable=too-few-public-methods
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic.dataclasses import dataclass

from core_utils.project.lab_settings import LabSettings
from lab_7_llm.main import LLMPipeline, TaskDataset

current_path = Path(__file__).parent
ASSETS_FOLDER = current_path / "assets"

@dataclass
class Query:
    """
    Abstraction class to contain text of the query.
    """
    question: str
    context: str


def init_application() -> tuple:
    """
    Initialize core application.

    Returns:
        tuple: tuple of two objects, instance of FastAPI server and LLMPipeline pipeline.
    """
    lab_app = FastAPI()

    settings = LabSettings(current_path / "settings.json")
    dataset = TaskDataset(pd.DataFrame())
    lab_pipeline = LLMPipeline(
        model_name=settings.parameters.model,
        dataset=dataset,
        max_length=120,
        batch_size=1,
        device="cpu",
    )
    return lab_app, lab_pipeline

app, pipeline = init_application()
app.mount("/assets", StaticFiles(directory=ASSETS_FOLDER), name="assets")
templates = Jinja2Templates(directory=ASSETS_FOLDER)

@app.get("/")
async def root(request: Request) -> HTMLResponse:
    """
    Root endpoint.
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/infer")
async def infer(query:Query) -> dict:
    """
    Inference model with input query.
    """
    result = pipeline.infer_sample((query.question, query.context))
    return{"infer": result}
