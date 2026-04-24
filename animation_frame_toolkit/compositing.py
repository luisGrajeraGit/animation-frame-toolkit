"""
animation_frame_toolkit.compositing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Composición final: tritono + alpha → RGBA listo para exportar.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .fill import quantize_fills


def build_tritone(
    clean_gray: np.ndarray,
    alpha_mask: np.ndarray,
    ink_mask: np.ndarray,
    dark_gray: int = 72,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Genera el mapa tricolor (negro / gris / blanco) con las líneas de tinta.

    Returns:
        (tritone, fill_map, threshold)
    """
    fill_map, thr = quantize_fills(clean_gray, alpha_mask, dark_gray=dark_gray)
    out = np.full_like(clean_gray, 255)
    out[alpha_mask > 0] = fill_map[alpha_mask > 0]
    out[np.logical_and(alpha_mask > 0, ink_mask > 0)] = 0
    return out, fill_map, thr


def to_rgba(tritone: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Compone un PNG RGBA a partir del tritono y el canal alpha."""
    rgba = np.zeros((*tritone.shape, 4), dtype=np.uint8)
    rgba[..., :3] = tritone[..., np.newaxis]
    rgba[..., 3] = alpha
    return rgba
