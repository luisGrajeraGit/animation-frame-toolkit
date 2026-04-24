"""
animation_frame_toolkit.debug
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Guardado de fotogramas intermedios para diagnóstico y ajuste de parámetros.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import cv2
import numpy as np


def save_debug_frames(
    debug_dir: "str | Path",
    stem: str,
    frames: Dict[str, np.ndarray],
) -> None:
    """Guarda los fotogramas intermedios en *debug_dir*.

    Args:
        debug_dir: Carpeta de destino (se crea si no existe).
        stem: Nombre base del fotograma (sin extensión).
        frames: Diccionario {sufijo: imagen}, p.ej. {"01_gray": gray_array}.
    """
    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    for name, img in frames.items():
        cv2.imwrite(str(debug_dir / f"{stem}_{name}.png"), img)
