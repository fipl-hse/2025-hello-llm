"""
Web service for model inference.
"""
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from core_utils.project.lab_settings import LabSettings
from lab_8_sft.main import LLMPipeline, TaskDataset

# pylint: disable=too-few-public-methods


MAIN_PATH = Path(__file__).parent


class Query(BaseModel):
    """
    Request payload schema for inference endpoint.

    """

    question: str
    use_base_model: bool = True


def init_application() -> tuple:
    """
    Initialize core application.

    Returns:
        tuple: tuple of three objects, instance of FastAPI server, LLMPipeline and SFTPipeline.
    """
    fast_app = FastAPI()

    settings = LabSettings(Path(__file__).parent / "settings.json")
    model_name = settings.parameters.model
    dataset = TaskDataset(pd.DataFrame())
    pipeline = LLMPipeline(model_name, dataset, batch_size=1, max_length=120, device="cpu")

    finetuned_model_path = Path(__file__).parent / "dist" / model_name
    if finetuned_model_path.exists():
        sft_pipeline = LLMPipeline(
            str(finetuned_model_path), dataset=dataset, max_length=120, batch_size=1, device="cpu"
        )
    else:
        sft_pipeline = pipeline
    return fast_app, pipeline, sft_pipeline


(
    app,
    base_pipeline,
    ft_pipeline,
) = init_application()
app.mount("/assets", StaticFiles(directory=f"{MAIN_PATH}/assets"), name="assets")


@app.get("/", response_class=HTMLResponse)
def root(request: Request) -> HTMLResponse:
    """
    Root endpoint.

    """
    templates = Jinja2Templates(directory=f"{MAIN_PATH}/assets")
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/infer")
def infer(query: Query) -> dict[str, str]:
    """
    Process user query with LLM pipeline.

    """
    if query.use_base_model:
        result = base_pipeline.infer_sample(query.question)
        output = f"base result: {result}"
    else:
        result = ft_pipeline.infer_sample(query.question)
        output = f"sft result: {result}"

    return {"infer": output}
