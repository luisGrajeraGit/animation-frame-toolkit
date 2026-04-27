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
    num, labels, stats, _ = cv2.connectedComponentsWithStats((mask0 > 0).astype(np.uint8), 8)
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
        if touches_alpha and max_half_width <= (max_line_width / 2.0 + 0.8) and (aspect >= 1.8 or ld_mean >= 6):
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
    num, labels, stats, _ = cv2.connectedComponentsWithStats((inside_white > 0).astype(np.uint8), 8)
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
    num, labels, stats, _ = cv2.connectedComponentsWithStats((black > 0).astype(np.uint8), 8)
    out = tritone.copy()
    for i in range(1, num):
        if int(stats[i, cv2.CC_STAT_AREA]) < min_area:
            out[labels == i] = dark_gray
    return out


def remove_contextual_white_components(
    tritone: np.ndarray,
    alpha_mask: np.ndarray,
    ink_mask: np.ndarray,
    outer_outline: np.ndarray,
    dark_gray: int = 72,
    min_area: int = 6,
    max_area: int = 20000,
) -> np.ndarray:
    """Hace TRANSPARENTES (alpha_mask=0) las islas blancas dentro del alpha
    que probablemente sean artefactos (p.ej. el hueco entre las patas):

    - no tocan la tinta (ink_mask)
    - su centroide Y está en el 25 % inferior del bbox vertical del alpha
      (más robusto que usar el midpoint, que depende de la posición del
      personaje en el encuadre)
    - no tocan el borde exterior del alpha dilatado (protege uñas y bordes
      de diseño que coinciden con el límite de la silueta)
    - area entre ``min_area`` y ``max_area``

    Las áreas blancas legítimas (cara, ojos, dientes, uñas) no resultan
    afectadas.
    """
    inside_white = np.logical_and(alpha_mask > 0, tritone == 255).astype(np.uint8) * 255
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(inside_white, 8)

    ys, xs = np.where(alpha_mask > 0)
    if ys.size == 0:
        return tritone

    alpha_y_min = float(ys.min())
    alpha_y_max = float(ys.max())
    # Solo el 25 % inferior del bbox del personaje
    artifact_y_threshold = alpha_y_min + 0.75 * (alpha_y_max - alpha_y_min)

    # Zona alrededor del borde exterior del alpha → protege uñas/puntas
    se_border = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    boundary_zone = cv2.dilate(outer_outline, se_border, iterations=1)

    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        cy = float(centroids[i][1])
        # Solo candidatos en la zona inferior del personaje
        if cy <= artifact_y_threshold:
            continue
        comp_mask = labels == i
        # Toca tinta → diseño legítimo
        if np.any(np.logical_and(comp_mask, ink_mask > 0)):
            continue
        # Toca el borde exterior del alpha → elemento de borde (uñas, punta de cola…)
        if np.any(comp_mask & (boundary_zone > 0)):
            continue
        # Interior blanco en zona inferior sin tinta → artefacto → transparente
        alpha_mask[comp_mask] = 0

    return tritone


def remove_isolated_white_components(
    tritone: np.ndarray,
    alpha_mask: np.ndarray,
    min_cluster_area: int = 150,
    max_remove_area: int = 300,
    isolation_gap: int = 15,
) -> np.ndarray:
    """Hace TRANSPARENTES (alpha_mask=0) los blancos pequeños que estén lejos
    del cluster principal de blancos (cara/ojos/dientes).

    Algoritmo:
      1. Detecta todos los componentes blancos dentro del alpha.
      2. Los grandes (area >= ``min_cluster_area``) forman el "cluster de cara".
      3. Dilata ese cluster ``isolation_gap`` píxeles.
      4. Componentes pequeños (area < ``max_remove_area``) que NO se solapen
         con el cluster dilatado → probables artefactos junto a bigotes o
         cuerpo → se vuelven transparentes.
    """
    inside_white = np.logical_and(alpha_mask > 0, tritone == 255).astype(np.uint8) * 255
    num, labels, stats, _ = cv2.connectedComponentsWithStats(inside_white, 8)
    if num <= 2:
        return tritone

    main_mask = np.zeros(tritone.shape, dtype=np.uint8)
    has_main = False
    for i in range(1, num):
        if int(stats[i, cv2.CC_STAT_AREA]) >= min_cluster_area:
            main_mask[labels == i] = 255
            has_main = True
    if not has_main:
        return tritone

    k = isolation_gap * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    main_dilated = cv2.dilate(main_mask, kernel)

    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= min_cluster_area or area >= max_remove_area:
            continue
        comp_mask = labels == i
        if np.any(comp_mask & (main_dilated > 0)):
            continue
        alpha_mask[comp_mask] = 0

    return tritone
