"""
Web service for model inference.
"""

# pylint: disable=too-few-public-methods
import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pandas import DataFrame
from pydantic import BaseModel

from lab_7_llm.main import LLMPipeline, TaskDataset


def init_application() -> tuple:
    """
    Initialize core application.

    Returns:
        tuple: tuple of two objects, instance of FastAPI server and LLMPipeline pipeline.
    """
    server = FastAPI()

    project_root = Path(__file__).parent.parent
    assets_path = project_root / "lab_7_llm" / "assets"
    assets_path.mkdir(parents=True, exist_ok=True)
    server.mount("/assets", StaticFiles(directory=assets_path), name="assets")

    settings_path = Path(__file__).parent / "settings.json"
    with open(settings_path, "r", encoding="utf-8") as file:
        settings = json.load(file)

    empty_data = DataFrame({"source": [], "target": []})
    dataset = TaskDataset(empty_data)

    llm_pipeline = LLMPipeline(
        model_name=settings["parameters"]["model"],
        dataset=dataset,
        max_length=120,
        batch_size=1,
        device="cpu"
    )

    return server, llm_pipeline

app, pipeline = (None, None)


class Query(BaseModel):
    """
    A class that represents the query model.
    """
    question: str


@app.post("/infer")
async def infer(query: Query) -> dict:
    """
    Infer the model.

    Args:
        query (Query): The query

    Returns:
        dict: The prediction
    """
    text = query.question

    prediction = pipeline.infer_sample((text,))

    return {"infer": prediction}