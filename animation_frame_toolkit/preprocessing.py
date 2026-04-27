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
    """Convierte cualquier imagen (BGR, BGRA, gray) a escala de grises.

    Maneja imágenes de 16 bits (uint16) y PNGs con canal alpha pre-keyed:
    - uint16 → uint8 (divide por 257).
    - BGRA: compuesta sobre fondo blanco antes de escalar a gris,
      para que los píxeles transparentes queden en 255 (blanco de fondo).
    """
    # --- Normalizar profundidad de bits ---
    if img.dtype == np.uint16:
        img = (img.astype(np.float32) / 257.0).clip(0, 255).astype(np.uint8)

    if img.ndim == 2:
        return img

    if img.shape[2] == 4:
        # Compositar sobre fondo blanco para que la transparencia quede blanca.
        # Así las imágenes pre-keyed se tratan igual que las de fondo blanco.
        b, g, r, a = cv2.split(img.astype(np.float32))
        alpha_f = a / 255.0
        comp_b = (b * alpha_f + 255.0 * (1.0 - alpha_f)).clip(0, 255).astype(np.uint8)
        comp_g = (g * alpha_f + 255.0 * (1.0 - alpha_f)).clip(0, 255).astype(np.uint8)
        comp_r = (r * alpha_f + 255.0 * (1.0 - alpha_f)).clip(0, 255).astype(np.uint8)
        return cv2.cvtColor(cv2.merge([comp_b, comp_g, comp_r]), cv2.COLOR_BGR2GRAY)

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
