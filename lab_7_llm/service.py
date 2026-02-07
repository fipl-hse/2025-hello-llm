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
from lab_7_llm.main import LLMPipeline, TaskDataset

BASE_PATH = Path(__file__).parent
ASSETS_PATH = BASE_PATH / "assets"

class Query(BaseModel):
    """
    data model for user's input
    """
    question: str

def init_application() -> tuple:
    """
    Initialize core application.

    Returns:
        tuple: tuple of two objects, instance of FastAPI server and LLMPipeline pipeline.
    """
    model_app = FastAPI()

    settings = LabSettings(BASE_PATH / "settings.json")
    dataset = TaskDataset(pd.DataFrame())
    model_pipeline = LLMPipeline(
        model_name=settings.parameters.model,
        dataset=dataset,
        max_length=120,
        batch_size=1,
        device="cpu"
    )

    return model_app, model_pipeline


app, pipeline = init_application()
app.mount("/assets", StaticFiles(directory=ASSETS_PATH), name="assets")
templates = Jinja2Templates(directory=str(ASSETS_PATH))

@app.get("/", response_class=HTMLResponse)
def get_root(request: Request) -> HTMLResponse:
    """
    root endpoint for the main HTML page
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/infer")
def infer(query: Query) -> dict:
    result = int(pipeline.infer_sample(query.question))

    emotions = ["Радость", "Грусть", "Страх", "Злость", "Нейтрально", "Другое"]

    if 0 <= result < len(emotions):
        return {"infer": emotions[result]}
    else:
        return {"infer": "Неизвестное состояние"}
