"""
animation_frame_toolkit.ink
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Refuerzo de la máscara de tinta y generación del contorno de silueta.
"""
from __future__ import annotations

import cv2
import numpy as np


def reinforce_line_mask(
    mask0: np.ndarray,
    gray: np.ndarray,
    local_dark: np.ndarray,
    alpha_mask: np.ndarray,
    fill_map: np.ndarray,
    max_line_width: int = 5,
    abs_black: int = 24,
) -> np.ndarray:
    """Filtra componentes de mask0 con reglas semánticas:

    - Conserva líneas finas oscuras que tocan el personaje.
    - Conserva componentes oscuros rodeados de blanco (ojos, nariz, etc.).
    - Descarta solo ruido pequeño lejos del personaje.
    """
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask0 > 0).astype(np.uint8), 8
    )
    out = np.zeros_like(mask0)
    alpha_dil = cv2.dilate(
        alpha_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=1,
    )

    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        comp = (labels == i).astype(np.uint8)
        if area < 1:
            continue

        ys, xs = np.where(comp > 0)
        h_c = ys.max() - ys.min() + 1
        w_c = xs.max() - xs.min() + 1
        aspect = max(w_c, h_c) / max(1, min(w_c, h_c))

        max_half_width = float(cv2.distanceTransform(comp, cv2.DIST_L2, 3).max())
        inside_mean = float(gray[comp > 0].mean()) if area else 255.0
        ld_mean = float(local_dark[comp > 0].mean()) if area else 0.0
        touches_alpha = bool(np.any((comp > 0) & (alpha_dil > 0)))

        ring = cv2.dilate(
            comp,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
            iterations=1,
        )
        ring = np.logical_and(
            np.clip(ring - comp, 0, 1).astype(bool),
            alpha_dil > 0,
        )
        ring_vals = fill_map[ring]
        white_ratio = float(np.mean(ring_vals == 255)) if ring_vals.size else 0.0

        keep = False
        if touches_alpha and inside_mean <= 70:
            keep = True
        if (
            touches_alpha
            and max_half_width <= (max_line_width / 2.0 + 0.8)
            and (aspect >= 1.8 or ld_mean >= 6)
        ):
            keep = True
        if touches_alpha and white_ratio >= 0.60 and inside_mean <= 80:
            keep = True
        if (not touches_alpha) and area <= 8:
            keep = False
        if area <= 3 and inside_mean > abs_black and ld_mean < 5:
            keep = False

        if keep:
            out[labels == i] = 255

    return out


def silhouette_outline(
    alpha_mask: np.ndarray,
    outline_thickness: int = 2,
) -> np.ndarray:
    """Genera un contorno exterior erosionando el alpha."""
    if outline_thickness <= 0:
        return np.zeros_like(alpha_mask)
    se = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * outline_thickness + 1, 2 * outline_thickness + 1),
    )
    return cv2.subtract(alpha_mask, cv2.erode(alpha_mask, se, iterations=1))


def remove_white_specks(
    tritone: np.ndarray,
    alpha_mask: np.ndarray,
    dark_gray: int = 72,
    min_area: int = 8,
) -> np.ndarray:
    """Rellena pequeñas manchas blancas dentro del personaje."""
    inside_white = np.logical_and(alpha_mask > 0, tritone == 255).astype(np.uint8) * 255
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        (inside_white > 0).astype(np.uint8), 8
    )
    out = tritone.copy()
    for i in range(1, num):
        if int(stats[i, cv2.CC_STAT_AREA]) < min_area:
            out[labels == i] = dark_gray
    return out


def remove_black_specks(
    tritone: np.ndarray,
    alpha_mask: np.ndarray,
    dark_gray: int = 72,
    min_area: int = 3,
) -> np.ndarray:
    """Elimina pequeñas manchas negras sueltas dentro del personaje."""
    black = np.logical_and(alpha_mask > 0, tritone == 0).astype(np.uint8) * 255
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        (black > 0).astype(np.uint8), 8
    )
    out = tritone.copy()
    for i in range(1, num):
        if int(stats[i, cv2.CC_STAT_AREA]) < min_area:
            out[labels == i] = dark_gray
    return out
