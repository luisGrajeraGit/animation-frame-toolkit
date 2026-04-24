"""
animation_frame_toolkit.fill
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Limpieza y cuantización de los rellenos interiores del personaje.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def clean_fills(
    gray: np.ndarray,
    line_mask: np.ndarray,
    alpha_mask: np.ndarray,
    body_smooth: int = 2,
) -> np.ndarray:
    """Limpia los rellenos del interior del personaje preservando las líneas de tinta."""
    protected = cv2.dilate(
        line_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    base = cv2.inpaint(gray, protected, 3, cv2.INPAINT_TELEA)
    base = cv2.bilateralFilter(base, d=0, sigmaColor=18, sigmaSpace=4)
    base = cv2.medianBlur(base, 3)
    if body_smooth > 0:
        k = 2 * body_smooth + 1
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        base = cv2.morphologyEx(base, cv2.MORPH_CLOSE, se, iterations=1)
    clean = gray.copy()
    non_line_inside = np.logical_and(alpha_mask > 0, protected == 0)
    clean[non_line_inside] = base[non_line_inside]
    return clean


def quantize_fills(
    clean_gray: np.ndarray,
    alpha_mask: np.ndarray,
    dark_gray: int = 72,
) -> Tuple[np.ndarray, int]:
    """Cuantiza el interior a dos tonos (oscuro / blanco) via Otsu.

    Returns:
        (fill_map, threshold)
    """
    out = np.full_like(clean_gray, 255)
    vals = clean_gray[alpha_mask > 0]
    if vals.size == 0:
        return out, 127
    thr, _ = cv2.threshold(
        vals.reshape(-1, 1).astype(np.uint8),
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    out[np.logical_and(alpha_mask > 0, clean_gray <= thr)] = dark_gray
    out[np.logical_and(alpha_mask > 0, clean_gray > thr)] = 255
    return out, int(thr)
