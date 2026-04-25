"""
animation_frame_toolkit.alpha
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Cálculo del canal alpha del personaje a partir de la imagen normalizada.

Estrategia v8 (dual-path):
  1. Flood-fill desde el borde para detectar el fondo definitivo.
  2. Silueta basada en píxeles oscuros con cierre morfológico grande,
     para recuperar zonas de patas/contorno no cerrado.
  Ambas vías se combinan con OR restringido a la zona del personaje.
"""

from __future__ import annotations

import cv2
import numpy as np

from .utils import area_filter, fill_holes

# ------------------------------------------------------------------ #
# Funciones internas                                                   #
# ------------------------------------------------------------------ #


def _flood_fill_alpha(
    gray: np.ndarray,
    barrier: np.ndarray,
    bg_lo: int,
    bg_hi: int,
) -> np.ndarray:
    """Flood-fill desde el borde para detectar el fondo definitivo."""
    h, w = gray.shape
    sure_bg = (gray >= bg_hi).astype(np.uint8) * 255
    maybe_bg = (gray >= bg_lo).astype(np.uint8) * 255
    walkable = cv2.bitwise_and(maybe_bg, cv2.bitwise_not(barrier))

    ff = np.where(walkable > 0, 0, 255).astype(np.uint8)
    ffmask = np.zeros((h + 2, w + 2), np.uint8)

    border_seed = np.zeros_like(sure_bg)
    border_seed[0, :] = sure_bg[0, :]
    border_seed[-1, :] = sure_bg[-1, :]
    border_seed[:, 0] = np.maximum(border_seed[:, 0], sure_bg[:, 0])
    border_seed[:, -1] = np.maximum(border_seed[:, -1], sure_bg[:, -1])

    ys, xs = np.where(border_seed > 0)
    for y, x in zip(ys, xs):
        if ff[y, x] == 0:
            cv2.floodFill(ff, ffmask, (int(x), int(y)), 128)

    fg = cv2.bitwise_not((ff == 128).astype(np.uint8) * 255)
    fg = area_filter(fg, min_area=64, keep_larger=True)
    return fill_holes(fg)


def _dark_silhouette_alpha(
    gray: np.ndarray,
    alpha_close: int,
    dark_thresh: int = 180,
) -> np.ndarray:
    """Silueta basada en píxeles oscuros + cierre morfológico grande.

    Cierra grietas entre líneas de patas y rellena el interior.
    """
    dark = (gray < dark_thresh).astype(np.uint8) * 255
    if alpha_close > 0:
        k = 2 * alpha_close + 1
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, se, iterations=1)
    dark_sil = area_filter(dark, min_area=256, keep_larger=True)
    return fill_holes(dark_sil)


def _remove_white_border_leakage(
    fg: np.ndarray,
    gray: np.ndarray,
    bg_lo: int = 245,
) -> np.ndarray:
    """Elimina del alpha los píxeles claros conectados al fondo exterior.

    Preserva las áreas blancas interiores cerradas (cara, ojos, etc.).
    Semillas dobles:
      1. Píxeles brillantes dentro del alpha adyacentes al exterior del alpha.
      2. Píxeles brillantes dentro del alpha en el BORDE DE LA IMAGEN
         (cubre casos en que el personaje sale por el encuadre, p.ej. patas).
    """
    bright_inside = np.logical_and(fg > 0, gray >= bg_lo).astype(np.uint8) * 255

    h, w = gray.shape
    flood = bright_inside.copy()
    ffmask = np.zeros((h + 2, w + 2), np.uint8)

    # --- Semillas desde exterior del alpha (comportamiento original) ---
    exterior = (fg == 0).astype(np.uint8) * 255
    ext_dil = cv2.dilate(
        exterior,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    seeds_alpha = cv2.bitwise_and(bright_inside, ext_dil)

    # --- Semillas desde el BORDE DE LA IMAGEN (nuevo) ---
    # Cualquier píxel brillante del alpha que toque el frame edge = background.
    border_mask = np.zeros((h, w), np.uint8)
    border_mask[0, :] = 255
    border_mask[-1, :] = 255
    border_mask[:, 0] = 255
    border_mask[:, -1] = 255
    seeds_border = cv2.bitwise_and(bright_inside, border_mask)

    all_seeds = cv2.bitwise_or(seeds_alpha, seeds_border)
    ys, xs = np.where(all_seeds > 0)
    for y, x in zip(ys, xs):
        if flood[y, x] == 255:
            cv2.floodFill(flood, ffmask, (int(x), int(y)), 128)

    leaked = (flood == 128).astype(np.uint8) * 255
    return cv2.bitwise_and(fg, cv2.bitwise_not(leaked))


def _defringe_boundary(
    fg: np.ndarray,
    gray: np.ndarray,
    ring_width: int = 2,
    bright_thresh: int = 220,
) -> np.ndarray:
    """Elimina del alpha los píxeles muy brillantes en el ring exterior del alpha.

    Corrige el halo blanco que aparece al componer sobre fondos de color:
    el alpha binario se extiende 1-2 px más allá de la línea de tinta y esos
    píxeles frontera tienen color de fondo blanco en el tritono.

    Solo actúa sobre la corona exterior (ring_width px) del alpha, por lo que
    las áreas blancas interiores cerradas (cara, ojos) quedan intactas.
    """
    if ring_width <= 0:
        return fg
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring_width + 1, 2 * ring_width + 1))
    eroded = cv2.erode(fg, se, iterations=1)
    # ring = píxeles del alpha que desaparecen al erosionar
    boundary_ring = cv2.bitwise_and(fg, cv2.bitwise_not(eroded))
    # dentro del ring, solo los muy brillantes son fringe de fondo
    bright = (gray >= bright_thresh).astype(np.uint8) * 255
    bright_fringe = cv2.bitwise_and(boundary_ring, bright)
    return cv2.bitwise_and(fg, cv2.bitwise_not(bright_fringe))


# ------------------------------------------------------------------ #
# API pública                                                          #
# ------------------------------------------------------------------ #


def compute_alpha(
    gray: np.ndarray,
    line_barrier: np.ndarray,
    bg_lo: int = 245,
    bg_hi: int = 251,
    shrink: int = 1,
    alpha_close: int = 25,
    dark_thresh: int = 180,
    defringe_width: int = 2,
    defringe_thresh: int = 220,
) -> np.ndarray:
    """Calcula el canal alpha del personaje (dual-path v8).

    Combina flood-fill desde borde + silueta oscura con cierre morfológico.
    """
    barrier = cv2.dilate(
        line_barrier,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )

    # Vía 1: flood-fill
    fg = _flood_fill_alpha(gray, barrier, bg_lo, bg_hi)

    # Vía 2: silueta oscura (solo si alpha_close > 0)
    if alpha_close > 0:
        dark_sil = _dark_silhouette_alpha(gray, alpha_close, dark_thresh)
        margin = alpha_close + 20
        se_margin = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))
        fg_expanded = cv2.dilate(fg, se_margin, iterations=1)
        dark_sil = cv2.bitwise_and(dark_sil, fg_expanded)
        fg = cv2.bitwise_or(fg, dark_sil)
        fg = fill_holes(fg)
        fg = area_filter(fg, min_area=64, keep_larger=True)
        fg = fill_holes(fg)

    # Limpieza de fondo blanco fugado
    fg = _remove_white_border_leakage(fg, gray, bg_lo=bg_lo)
    fg = area_filter(fg, min_area=64, keep_larger=True)
    fg = fill_holes(fg)

    # Erosión suave para afinar bordes
    if shrink > 0:
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * shrink + 1, 2 * shrink + 1))
        fg = cv2.erode(fg, se, iterations=1)
        fg = fill_holes(fg)

    # Defringe: eliminar píxeles muy brillantes en la corona exterior (anti-halo)
    fg = _defringe_boundary(fg, gray, ring_width=defringe_width, bright_thresh=defringe_thresh)

    return fg
