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
from lab_8_sft.main import LLMPipeline, TaskDataset

BASE_PATH = Path(__file__).parent
ASSETS_DIR = BASE_PATH / "assets"


@dataclass
class Query:
    """Request schema for inference."""
    question: str
    is_base: bool = False


def init_application() -> tuple:
    """
    Initialize core application.

    Returns:
        tuple: tuple of three objects, instance of FastAPI server, LLMPipeline and SFTPipeline.
    """
    dataset = TaskDataset(pd.DataFrame())
    model_name = LabSettings(BASE_PATH / 'settings.json').parameters.model
    sft_path = BASE_PATH / 'dist' / model_name

    llm = LLMPipeline(model_name, dataset, 120, 1, 'cpu')
    fastapi_app = FastAPI()

    if sft_path.exists():
        sft = LLMPipeline(str(sft_path), dataset, 120, 1, 'cpu')
    else:
        sft = llm

    return fastapi_app, llm, sft


app, pre_trained_pipeline, fine_tuned_pipeline = init_application()

templates = Jinja2Templates(directory=str(ASSETS_DIR))
app.mount("/static", StaticFiles(directory=str(ASSETS_DIR)), name="static")

ID2LABEL = {
    "0": "sadness",
    "1": "joy",
    "2": "love",
    "3": "anger",
    "4": "fear",
    "5": "surprise"
}


@app.get("/", response_class=HTMLResponse)
def root(request: Request) -> HTMLResponse:
    """
    Root endpoint: returns start HTML page with UI.
    """
    return templates.TemplateResponse(
        "index.html",
        {"request": request},
    )


@app.post("/infer")
def infer(query: Query) -> dict:
    """
    Main endpoint: runs LLM pipeline and returns JSON with result.
    """
    pipeline = pre_trained_pipeline if query.is_base else fine_tuned_pipeline
    result = pipeline.infer_sample((query.question,))
    result = ID2LABEL.get(result, str(result))
    return {"infer": result}
