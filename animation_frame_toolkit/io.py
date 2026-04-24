"""
animation_frame_toolkit.io
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Lectura y escritura de imágenes. Centraliza el acceso a disco.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_image(path: "str | Path") -> np.ndarray:
    """Lee una imagen desde disco. Lanza RuntimeError si no puede leerla."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"No se pudo leer la imagen: {path}")
    return img


def write_image(path: "str | Path", img: np.ndarray) -> None:
    """Escribe una imagen en disco, creando directorios intermedios si es necesario."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
