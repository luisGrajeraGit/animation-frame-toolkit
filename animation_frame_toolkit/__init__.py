"""animation_frame_toolkit package

Exponen funciones principales de `scripts.cartoon_frame_cleaner` para uso
desde Python.
"""

__version__ = "0.1.0"

from scripts.cartoon_frame_cleaner import process_one, main  # noqa: E402,F401

__all__ = ["process_one", "main"]
