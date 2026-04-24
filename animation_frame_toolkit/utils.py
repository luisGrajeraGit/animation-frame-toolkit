"""
animation_frame_toolkit.utils
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Utilidades morfológicas de bajo nivel compartidas por varios módulos.
"""
from __future__ import annotations

import cv2
import numpy as np


def area_filter(
    mask_u8: np.ndarray,
    min_area: int = 16,
    keep_larger: bool = True,
) -> np.ndarray:
    """Filtra componentes conectadas por área mínima."""
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask_u8 > 0).astype(np.uint8), 8
    )
    out = np.zeros_like(mask_u8)
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area if keep_larger else area < min_area:
            out[labels == i] = 255
    return out


def fill_holes(mask_u8: np.ndarray) -> np.ndarray:
    """Rellena los huecos interiores de una máscara binaria."""
    h, w = mask_u8.shape
    flood = mask_u8.copy()
    ffmask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, ffmask, (0, 0), 255)
    return cv2.bitwise_or(mask_u8, cv2.bitwise_not(flood))
