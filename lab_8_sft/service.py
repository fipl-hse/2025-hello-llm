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
from pydantic import BaseModel

from core_utils.project.lab_settings import LabSettings
from lab_8_sft.main import LLMPipeline, TaskDataset

BASE_PATH = Path(__file__).parent


class Query(BaseModel):
    """
    Query abstraction for user input.
    """
    question: str
    use_base: bool = True


def init_application() -> tuple:
    """
    Initialize core application.

    Returns:
        tuple: tuple of three objects, instance of FastAPI server, LLMPipeline and SFTPipeline.
    """
    dataset = TaskDataset(pd.DataFrame())
    model_name = LabSettings(BASE_PATH / 'settings.json').parameters.model

    base = LLMPipeline(model_name, dataset, 120, 1, 'cpu')
    ft = base

    ft_path = BASE_PATH / 'dist' / model_name
    if ft_path.exists():
        ft = LLMPipeline(str(ft_path), dataset, 120, 1, 'cpu')

    return FastAPI(), base, ft


app, base_pipeline, ft_pipeline = init_application()
app.mount("/static", StaticFiles(directory=BASE_PATH / 'assets'), name="static")


@app.get("/", response_class=HTMLResponse)
def root(request: Request) -> HTMLResponse:
    """
    Root endpoint returning HTML page.
    """
    templates = Jinja2Templates(directory=BASE_PATH / 'assets')
    return templates.TemplateResponse('index.html', {"request": request})


@app.post("/infer")
def infer(query: Query) -> dict:
    """
    Main endpoint for model inference.
    """
    pipeline = base_pipeline if query.use_base else ft_pipeline
    result = pipeline.infer_sample((query.question,))
    return {"infer": result}
