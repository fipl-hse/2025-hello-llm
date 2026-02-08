"""
Web service for model inference.
"""

# pylint: disable=too-few-public-methods
from lab_7_llm.main import LLMPipeline
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates



def init_application() -> tuple:
    """
    Initialize core application.

    Returns:
        tuple: tuple of two objects, instance of FastAPI server and LLMPipeline pipeline.
    """
    app = FastAPI()
    pipeline = LLMPipeline(batch_size=1, max_length=120, device='cpu')
    return tuple(app, pipeline)

# app, pipeline = init_application()

APP_FOLDER = "lab_7_llm"
# app.mount("/assets", StaticFiles(directory=f"{APP_FOLDER}/assets"), name="assets")

app = FastAPI()
app.mount("/static", StaticFiles(directory=f"{APP_FOLDER}/assets/templates"), name="static")


@app.get("/")
async def root(request: Request):
    """
    Root endpoint of application.

    """
    templates = Jinja2Templates(directory=f"{APP_FOLDER}/assets/templates")
    return templates.TemplateResponse("index.html",  {"request": request})

