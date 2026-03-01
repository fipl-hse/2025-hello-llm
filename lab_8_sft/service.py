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
ASSETS_PATH = BASE_PATH / "assets"

class Query(BaseModel):
    """
    data model for user's input
    """
    question: str
    is_base_model: bool = False

def init_application() -> tuple:
    """
    Initialize core application.

    Returns:
        tuple: tuple of three objects, instance of FastAPI server, LLMPipeline and SFTPipeline.
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

    finetuned_model_path = BASE_PATH / "dist" / settings.parameters.model
    if finetuned_model_path.exists():
        finetuned_pipeline = LLMPipeline(
            str(finetuned_model_path),
            dataset=dataset,
            max_length=120,
            batch_size=1,
            device="cpu"
        )
    else:
        finetuned_pipeline = model_pipeline

    return model_app, model_pipeline, finetuned_pipeline

app, pre_trained_pipeline, fine_tuned_pipeline = init_application()
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
    """
    the main endpoint of inference
    """
    if query.is_base_model:
        response_text = pre_trained_pipeline.infer_sample((query.question,))
    else:
        response_text = fine_tuned_pipeline.infer_sample((query.question,))
    print(f"Model response: {response_text}")
    return {"infer": response_text}
