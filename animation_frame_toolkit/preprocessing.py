"""
animation_frame_toolkit.preprocessing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Normalización de imagen antes de la extracción del personaje.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def ensure_gray(img: np.ndarray) -> np.ndarray:
    """Convierte cualquier imagen (BGR, BGRA, gray) a escala de grises."""
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def estimate_background(gray: np.ndarray) -> Tuple[float, float, float]:
    """Estima percentiles del fondo muestreando el borde de la imagen.

    Returns:
        (p50, p90, p99) del borde de la imagen en escala de grises.
    """
    h, w = gray.shape
    b = max(8, min(h, w) // 40)
    border = np.concatenate(
        [
            gray[:b, :].ravel(),
            gray[-b:, :].ravel(),
            gray[:, :b].ravel(),
            gray[:, -b:].ravel(),
        ]
    )
    return (
        float(np.percentile(border, 50)),
        float(np.percentile(border, 90)),
        float(np.percentile(border, 99)),
    )


def normalize_background(gray: np.ndarray) -> np.ndarray:
    """Escala la imagen para que el fondo quede en 255.

    Compensa fondos ligeramente grisáceos o amarillentos.
    """
    _, _, bg99 = estimate_background(gray)
    scale = 255.0 / max(bg99, 1.0)
    return np.clip(gray.astype(np.float32) * scale, 0, 255).astype(np.uint8)
