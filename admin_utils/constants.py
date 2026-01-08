"""
Constants for references collection.
"""
try:
    import torch
except ImportError:
    print('Library "torch" not installed. Failed to import.')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Selected {DEVICE} for all reference collection tasks")

USE_VENV = False
