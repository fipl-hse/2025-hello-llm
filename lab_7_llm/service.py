"""
Web service for model inference.
"""

# pylint: disable=too-few-public-methods
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import torch
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic.dataclasses import dataclass as pydantic_dataclass

from lab_7_llm.main import LLMPipeline, TaskDataset

NER_TAG_MAPPING = {
    0: "O",
    1: "B-PER",
    2: "I-PER",
    3: "B-ORG",
    4: "I-ORG",
    5: "B-LOC",
    6: "I-LOC",
    7: "B-MISC",
    8: "I-MISC",
}


def init_application() -> Tuple[FastAPI, LLMPipeline]:
    """
    Initialize core application.

    Returns:
        tuple: tuple of two objects, instance of FastAPI server and
               LLMPipeline pipeline.
    """
    model_name = "Babelscape/wikineural-multilingual-ner"
    max_length = 120
    batch_size = 64
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dummy_data = pd.DataFrame({"source": [["dummy"]], "target": [[0]]})
    dataset = TaskDataset(dummy_data)

    pipeline_instance = LLMPipeline(
        model_name=model_name,
        dataset=dataset,
        max_length=max_length,
        batch_size=batch_size,
        device=device,
    )

    app_instance = FastAPI(title="NER Service")

    assets_path_instance = Path(__file__).parent / "assets"
    assets_path_instance.mkdir(exist_ok=True)

    app_instance.mount("/assets", StaticFiles(directory=str(assets_path_instance)), name="assets")

    return app_instance, pipeline_instance


app, pipeline = init_application()
assets_path = Path(__file__).parent / "assets"
templates = Jinja2Templates(directory=str(assets_path))


@pydantic_dataclass
class Query:
    """Query model for inference."""

    question: str


@app.get("/", response_class=HTMLResponse)
async def root(request: Request) -> HTMLResponse:
    """
    Root endpoint serving the main page.
    """
    return templates.TemplateResponse("index.html", {"request": request, "title": "NER Service"})


@app.post("/infer")
async def infer(query: Query) -> Dict[str, str]:
    """
    Main endpoint for model inference.
    """
    result = pipeline.infer_sample((query.question,))
    numbers = [int(num.strip()) for num in result.strip("[]").split(",") if num.strip()]
    mapped_tags = [NER_TAG_MAPPING.get(num, "O") for num in numbers]

    return {"infer": str(mapped_tags)}
