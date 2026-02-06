"""
Web service for model inference.
"""
import json
from pathlib import Path

# pylint: disable=too-few-public-methods
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from lab_7_llm.main import LLMPipeline, TaskDataset

BASE_PATH = Path(__file__).parent


class Query(BaseModel):
    """
    Query abstraction for user input.
    """
    question: str


def init_application() -> tuple:
    """
    Initialize core application.

    Returns:
        tuple: tuple of two objects, instance of FastAPI server and LLMPipeline pipeline.
    """
    with open(BASE_PATH / 'settings.json', 'r', encoding='utf-8') as f:
        settings = json.load(f)

    return FastAPI(), LLMPipeline(settings['parameters']['model'],
                                  TaskDataset(pd.DataFrame()), 120, 1, 'cpu')


app, pipeline = init_application()
app.mount("/static", StaticFiles(directory=BASE_PATH / 'assets'), name="static")


@app.get("/", response_class=HTMLResponse)
def root(request: Request) -> HTMLResponse:
    """
    Root endpoint returning HTML page.
    """
    templates = Jinja2Templates(directory=BASE_PATH / 'assets')
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/infer")
def infer(query: Query) -> dict:
    """
    Main endpoint for model inference.
    """
    result = pipeline.infer_sample((query.question,))
    return {"infer": result}
