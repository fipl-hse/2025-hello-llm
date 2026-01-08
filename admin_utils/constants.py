"""
Constants for references collection.
"""

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Selected {DEVICE} for all reference collection tasks")

USE_VENV = False
