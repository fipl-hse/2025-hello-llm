"""
Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

# pylint: disable=invalid-name,redefined-builtin

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))

project = "Лабораторный Практикум и Курс Лекций"
copyright = "2025, Демидовский А.В. и другие"
author = "Демидовский А.В. и другие"

extensions = [
    "sphinx_design",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
]

root_doc = "admin_utils/index"

intersphinx_mapping = {
    "torch": ("https://pytorch.org/docs/stable", "admin_utils/intersphinx/pytorch.inv"),
    "python": ("https://docs.python.org/3", None),
    "pandas": (
        "http://pandas.pydata.org/pandas-docs/stable/",
        "admin_utils/intersphinx/pandas.inv",
    ),
    "pydantic": ("https://docs.pydantic.dev/latest/", "admin_utils/intersphinx/pydantic.inv"),
    "fastapi": ("https://fastapi.tiangolo.com/", "admin_utils/intersphinx/fastapi.inv"),
}

exclude_patterns = ["venv/*", "docs/private/*"]

nitpick_ignore = [
    ("py:class", "transformers.models.auto.tokenization_auto.AutoTokenizer"),
    ("py:class", "peft.tuners.lora.config.LoraConfig"),
    ("py:class", "optional"),
    ("py:class", "FastAPI"),
    ("py:class", "LLMPipeline"),
    ("py:class", "AutoTokenizer"),
]

language = "en"

html_theme = "sphinx_rtd_theme"
