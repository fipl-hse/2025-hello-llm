"""
Web service for model inference.
"""

# pylint: disable=too-few-public-methods
from pathlib import Path

import pandas as pd
from pydantic.dataclasses import dataclass  
from lab_7_llm.main import LLMPipeline
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from core_utils.project.lab_settings import LabSettings
from lab_7_llm.main import LLMPipeline, TaskDataset

APP_FOLDER = "lab_7_llm"

@dataclass
class Query:
    """
    Request payload schema for inference endpoint.
    
    """
    question: str

def init_application() -> tuple:
    """
    Initialize core application.

    Returns:
        tuple: tuple of two objects, instance of FastAPI server and LLMPipeline pipeline.
    """
    app = FastAPI()

    settings = LabSettings(Path(__file__).parent / "settings.json")
    pipeline = LLMPipeline(model_name=settings.parameters.model,
                           dataset=TaskDataset(pd.DataFrame()),
                           batch_size=1, max_length=120, device='cpu')
    return app, pipeline

app, pipeline = init_application()
app.mount("/assets", StaticFiles(directory=f"{APP_FOLDER}/assets/templates"), name="assets")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """
    Root endpoint of application.

    """
    templates = Jinja2Templates(directory=f"{APP_FOLDER}/assets/templates")
    return templates.TemplateResponse("index.html",  {"request": request})

@app.post("/infer")
async def infer(query: Query) -> dict[str, str]:
    """
    Process user query through LLM pipeline.

    """
    result = pipeline.infer_sample(query.question)
    if result == 0:
        output = 'Negative sentiment'
    else:
        output = 'Positive sentiment'

    return {"infer": output}