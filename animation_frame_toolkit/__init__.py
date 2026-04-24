"""animation_frame_toolkit
~~~~~~~~~~~~~~~~~~~~~~~~~~
Paquete Python para extracción de personajes 2D con fondo blanco.

API pública principal::

    from animation_frame_toolkit import ProcessConfig, process_frame, process_batch
"""

__version__ = "0.2.0"

from .config import ProcessConfig
from .pipeline import iter_inputs, process_frame
from .batch import process_batch

__all__ = [
    "ProcessConfig",
    "process_frame",
    "iter_inputs",
    "process_batch",
]
