"""
Constants for references collection.
"""

# pylint: disable=invalid-name,too-few-public-methods
try:
    import torch
except ImportError:
    print('Library "torch" not installed. Failed to import.')

    class torch:
        """
        Mock class for public repo that does not have requirements.
        """

        class cuda:
            """
            Mock class for torch.cuda.
            """

            @staticmethod
            def is_available() -> bool:
                """
                Mock method for torch.cuda.is_available.
                """
                return False


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Selected {DEVICE} for all reference collection tasks")

USE_VENV = True
