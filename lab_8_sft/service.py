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

current_path = Path(__file__).parent
ASSETS_FOLDER = current_path / "assets"


@dataclass
class Query:
    """
    Abstraction class to contain text of the query.
    """

    question: str
    use_base_model: bool = True


def init_application() -> tuple:
    """
    Initialize core application.

    Returns:
        tuple: tuple of three objects, instance of FastAPI server, LLMPipeline and SFTPipeline.
    """
    sft_app = FastAPI()

    settings = LabSettings(current_path / "settings.json")
    dataset = TaskDataset(pd.DataFrame())
    base_pipeline = LLMPipeline(
        model_name=settings.parameters.model,
        dataset=dataset,
        max_length=120,
        batch_size=1,
        device="cpu",
    )

    finetuned_path = current_path / "dist" / f"{settings.parameters.model}_finetuned"
    if finetuned_path.exists():
        finetuned_pipeline = LLMPipeline(
            model_name=str(finetuned_path),
            dataset=dataset,
            max_length=120,
            batch_size=1,
            device="cpu",
        )
    else:
        finetuned_pipeline = base_pipeline

    return sft_app, base_pipeline, finetuned_pipeline


app, pre_trained_pipeline, fine_tuned_pipeline = init_application()
app.mount("/assets", StaticFiles(directory=ASSETS_FOLDER), name="assets")
templates = Jinja2Templates(directory=ASSETS_FOLDER)


@app.get("/")
async def root(request: Request) -> HTMLResponse:
    """
    Root endpoint.
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/infer")
async def infer(query: Query) -> dict:
    """
    Inference model with input query.
    """
    if query.use_base_model:
        pipeline = pre_trained_pipeline
    else:
        pipeline = fine_tuned_pipeline

    sample = (query.question,)
    result = pipeline.infer_sample(sample)

    return {"infer": result}
