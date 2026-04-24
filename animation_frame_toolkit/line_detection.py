"""
animation_frame_toolkit.line_detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Detección de líneas de tinta en fotogramas de animación cartoon.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def build_line_score(
    gray: np.ndarray,
    local_radius: int = 9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calcula un mapa de «oscuridad local» combinando varias respuestas.

    Returns:
        (score, local_dark, blackhat_3x3, blackhat_5x5)
    """
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=max(local_radius / 2.0, 1.0))
    local_dark = np.clip(blur.astype(np.int16) - gray.astype(np.int16), 0, 255).astype(np.uint8)
    bh1 = cv2.morphologyEx(
        gray,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    bh2 = cv2.morphologyEx(
        gray,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    score = cv2.addWeighted(local_dark, 0.65, bh1, 0.20, 0)
    score = cv2.addWeighted(score, 1.00, bh2, 0.15, 0)
    return score, local_dark, bh1, bh2


def initial_line_mask(
    gray: np.ndarray,
    max_line_width: int = 5,
    line_thresh: int = 16,
    abs_black: int = 24,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Genera la máscara inicial de líneas de tinta.

    Returns:
        (mask0, score, local_dark, bh1, bh2)
    """
    score, local_dark, bh1, bh2 = build_line_score(gray, local_radius=max(5, max_line_width + 2))
    vals = score[gray < 245]
    thr = max(line_thresh, int(np.percentile(vals, 80))) if vals.size else line_thresh
    mask_local = (score >= thr).astype(np.uint8) * 255
    mask_abs = ((gray <= abs_black) & (local_dark >= max(4, line_thresh // 2))).astype(np.uint8) * 255
    mask0 = cv2.bitwise_or(mask_local, mask_abs)
    mask0 = cv2.morphologyEx(
        mask0,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    return mask0, score, local_dark, bh1, bh2
