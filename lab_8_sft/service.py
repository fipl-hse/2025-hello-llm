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

MAIN_PATH = Path(__file__).parent

class Query(BaseModel):
    """
    Request payload schema for inference endpoint.
    
    """
    question: str
    use_base_model: bool=True

def init_application() -> tuple:
    """
    Initialize core application.

    Returns:
        tuple: tuple of three objects, instance of FastAPI server, LLMPipeline and SFTPipeline.
    """
    app = FastAPI()

    settings = LabSettings(Path(__file__).parent / "settings.json")
    model_name = settings.parameters.model
    dataset = TaskDataset(pd.DataFrame())
    basic_pipeline = LLMPipeline(model_name, dataset,
                           batch_size=1, max_length=120, device='cpu')
    
    finetuned_model_path = Path(__file__).parent / 'dist' / f'{model_name}_finetuned'
    if finetuned_model_path.exists():
        sft_pipeline = LLMPipeline(
            str(finetuned_model_path),
            dataset=dataset,
            max_length=120,
            batch_size=1,
            device="cpu"
        )
    else:
        sft_pipeline = basic_pipeline
    return app, basic_pipeline, sft_pipeline

app, basic_pipeline, sft_pipeline,  = init_application()
app.mount("/assets", StaticFiles(directory=f"{MAIN_PATH}/assets"), name="assets")

@app.get("/", response_class=HTMLResponse)
def root(request: Request) -> HTMLResponse:
    """
    Root endpoint of application.

    """
    templates = Jinja2Templates(directory=f"{MAIN_PATH}/assets")
    return templates.TemplateResponse("index.html",  {"request": request})

@app.post("/infer")
def infer(query: Query) -> dict[str, str]:
    """
    Process user query through LLM pipeline.

    """
    if query.use_base_model:
        result = basic_pipeline.infer_sample(query.question)
        return {"infer": f'base result: {result}'}
    else:
        result = sft_pipeline.infer_sample(query.question)
        return {"infer": f'sft result: {result}'}

#  uvicorn lab_8_sft.service:app --reload

