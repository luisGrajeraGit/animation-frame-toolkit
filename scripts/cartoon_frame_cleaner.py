#!/usr/bin/env python3
"""
cartoon_frame_cleaner_v8.py

Mejoras sobre v7:

PROBLEMA PRINCIPAL CORREGIDO:
  El flood-fill desde el borde que calcula el alpha se "colaba" por las
  pequeñas grietas entre las líneas de las patas del gato (que no forman
  un contorno perfectamente cerrado en el borde inferior del encuadre).
  Esto hacía que los espacios blancos entre las patas aparecieran como
  transparentes en lugar de estar dentro del personaje.

SOLUCIÓN (dos vías de alpha combinadas):
  1. Alpha por flood-fill desde el borde (igual que v7) — detecta bien el
     fondo lejano y los grandes espacios vacíos.
  2. Alpha por silueta de píxeles oscuros: se toman todos los píxeles
     oscuros de la imagen normalizada, se cierra la máscara con un kernel
     grande (alpha_close, defecto 25 px) para rellenar las grietas entre
     las líneas de las patas, y se rellenan los huecos interiores. El
     resultado es una silueta compacta del personaje.
  Las dos vías se combinan con OR restringido: la silueta oscura solo se
  acepta dentro de una zona generosa alrededor del alpha original (evita
  incluir ruido lejano).

CORRECCIÓN ADICIONAL:
  Paso de "limpieza de borde blanco": los píxeles blancos dentro del alpha
  que están conectados al fondo exterior (alpha=0) a través de caminos de
  píxeles blancos son eliminados del alpha. Esto soluciona los artefactos
  blancos que a veces aparecían fuera del personaje cerca de los bigotes
  u otras líneas finas.

Salida: PNG RGBA
"""

import argparse
from pathlib import Path
import cv2
import numpy as np


def ensure_gray(img):
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def area_filter(mask_u8, min_area=16, keep_larger=True):
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask_u8 > 0).astype(np.uint8), 8
    )
    out = np.zeros_like(mask_u8)
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        ok = area >= min_area if keep_larger else area < min_area
        if ok:
            out[labels == i] = 255
    return out


def fill_holes(mask_u8):
    """Rellena los huecos interiores de una máscara binaria."""
    h, w = mask_u8.shape
    flood = mask_u8.copy()
    ffmask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, ffmask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mask_u8, holes)


def estimate_background(gray):
    h, w = gray.shape
    b = max(8, min(h, w) // 40)
    border = np.concatenate([
        gray[:b, :].ravel(),
        gray[-b:, :].ravel(),
        gray[:, :b].ravel(),
        gray[:, -b:].ravel(),
    ])
    return (
        float(np.percentile(border, 50)),
        float(np.percentile(border, 90)),
        float(np.percentile(border, 99)),
    )


def normalize_background(gray):
    g = gray.astype(np.float32)
    _, _, bg99 = estimate_background(gray)
    scale = 255.0 / max(bg99, 1.0)
    g = np.clip(g * scale, 0, 255)
    return g.astype(np.uint8)


def build_line_score(gray, local_radius=9):
    blur = cv2.GaussianBlur(
        gray, (0, 0), sigmaX=max(local_radius / 2.0, 1.0)
    )
    local_dark = np.clip(
        blur.astype(np.int16) - gray.astype(np.int16), 0, 255
    ).astype(np.uint8)
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


def initial_line_mask0(gray, max_line_width=5, line_thresh=16, abs_black=24):
    score, local_dark, bh1, bh2 = build_line_score(
        gray, local_radius=max(5, max_line_width + 2)
    )
    vals = score[gray < 245]
    thr = (
        max(line_thresh, int(np.percentile(vals, 80)))
        if vals.size
        else line_thresh
    )

    mask_local = (score >= thr).astype(np.uint8) * 255
    mask_abs = (
        (gray <= abs_black) & (local_dark >= max(4, line_thresh // 2))
    ).astype(np.uint8) * 255
    mask0 = cv2.bitwise_or(mask_local, mask_abs)
    mask0 = cv2.morphologyEx(
        mask0,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )

    return mask0, score, local_dark, bh1, bh2


# ---------------------------------------------------------------------------
# Alpha computation — v8: flood-fill + dark-silhouette recovery
# ---------------------------------------------------------------------------

def _flood_fill_alpha(gray, barrier, bg_lo, bg_hi):
    """Flood-fill desde el borde para detectar fondo definitivo."""
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

    bg = (ff == 128).astype(np.uint8) * 255
    fg = cv2.bitwise_not(bg)
    fg = area_filter(fg, min_area=64, keep_larger=True)
    fg = fill_holes(fg)
    return fg


def _dark_silhouette_alpha(gray, alpha_close, dark_thresh=180):
    """
    Silueta basada en píxeles oscuros + cierre morfológico grande.
    Cierra las grietas entre líneas de patas y rellena el interior.
    """
    dark = (gray < dark_thresh).astype(np.uint8) * 255
    if alpha_close > 0:
        k = 2 * alpha_close + 1
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        dark_closed = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, se, iterations=1)
    else:
        dark_closed = dark.copy()
    dark_sil = area_filter(dark_closed, min_area=256, keep_larger=True)
    dark_sil = fill_holes(dark_sil)
    return dark_sil


def _remove_white_border_leakage(fg, gray, bg_lo=245):
    """
    Elimina del alpha los píxeles blancos (fondo claro) que están conectados
    al exterior (alpha=0) a través de un camino de píxeles claros.
    Estos son píxeles de fondo que se colaron dentro del contorno alpha.
    Se mantienen intactas las áreas blancas encerradas (cara, ojos).
    """
    # Píxeles potencialmente problemáticos: dentro del alpha Y claros (fondo)
    bright_inside = np.logical_and(fg > 0, gray >= bg_lo).astype(np.uint8) * 255

    # Los píxeles oscuros forman una barrera; el fondo puede caminar solo
    # por los claros
    h, w = gray.shape
    flood = bright_inside.copy()
    ffmask = np.zeros((h + 2, w + 2), np.uint8)

    # Semillas del exterior: borde de la imagen donde el alpha es 0
    # (fondo real)
    ext_seed_rows = []
    ext_seed_cols = []
    # Borde superior
    for x in range(w):
        if fg[0, x] == 0 and bright_inside[0, x] == 0:
            pass  # ya es exterior, no necesita semilla en bright_inside
        # Buscamos píxeles bright_inside adyacentes al exterior
    # Mejor: dilatamos el exterior y cruzamos con bright_inside
    exterior = (fg == 0).astype(np.uint8) * 255
    ext_dil = cv2.dilate(
        exterior,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    seeds = cv2.bitwise_and(bright_inside, ext_dil)
    ys, xs = np.where(seeds > 0)

    for y, x in zip(ys, xs):
        if flood[y, x] == 255:
            cv2.floodFill(flood, ffmask, (int(x), int(y)), 128)

    leaked = (flood == 128).astype(np.uint8) * 255
    # Quitar del alpha los píxeles blancos fugados
    fg_clean = cv2.bitwise_and(fg, cv2.bitwise_not(leaked))
    return fg_clean


def compute_alpha_from_gray(
    gray,
    line_barrier,
    bg_lo=245,
    bg_hi=251,
    shrink=1,
    alpha_close=25,
    dark_thresh=180,
):
    """
    Calcula el canal alpha del personaje.

    v8: combina flood-fill desde el borde (detecta fondo) con silueta
    basada en píxeles oscuros cerrada morfológicamente (recupera patas).
    """
    h, w = gray.shape
    barrier = cv2.dilate(
        line_barrier,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )

    # --- Vía 1: flood-fill desde el borde ---
    fg_flood = _flood_fill_alpha(gray, barrier, bg_lo, bg_hi)

    # --- Vía 2: silueta oscura con cierre grande ---
    if alpha_close > 0:
        dark_sil = _dark_silhouette_alpha(gray, alpha_close, dark_thresh)

        # Restringir la silueta oscura a una zona próxima al alpha del
        # flood-fill (evita incluir ruido lejos del personaje)
        margin = alpha_close + 20
        se_margin = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1)
        )
        fg_expanded = cv2.dilate(fg_flood, se_margin, iterations=1)
        dark_sil_restricted = cv2.bitwise_and(dark_sil, fg_expanded)

        # Combinar: unión de ambas vías
        fg = cv2.bitwise_or(fg_flood, dark_sil_restricted)
        fg = fill_holes(fg)
        fg = area_filter(fg, min_area=64, keep_larger=True)
        fg = fill_holes(fg)
    else:
        fg = fg_flood

    # --- Limpieza de píxeles blancos fugados del fondo ---
    fg = _remove_white_border_leakage(fg, gray, bg_lo=bg_lo)
    fg = area_filter(fg, min_area=64, keep_larger=True)
    fg = fill_holes(fg)

    # --- Erosión suave para afinar el borde ---
    if shrink > 0:
        fg = cv2.erode(
            fg,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * shrink + 1, 2 * shrink + 1)
            ),
            iterations=1,
        )
        fg = fill_holes(fg)

    return fg


def clean_fills(gray, line_mask, alpha_mask, body_smooth=2):
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


def provisional_fill_quantization(clean_gray, alpha_mask, dark_gray=72):
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


def reinforce_mask0(
    mask0, gray, local_dark, alpha_mask, fill_map, max_line_width=5, abs_black=24
):
    """
    Usa mask0 casi completo, pero con reglas suaves:
    - conservar todo lo que toque o roce el personaje
    - conservar líneas finas oscuras
    - conservar componentes oscuros rodeados de blanco (ojos, nariz, boca)
    - descartar solo basura muy pequeña lejos del personaje
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
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        h_c = y1 - y0 + 1
        w_c = x1 - x0 + 1
        aspect = max(w_c, h_c) / max(1, min(w_c, h_c))

        dt = cv2.distanceTransform(comp, cv2.DIST_L2, 3)
        max_half_width = float(dt.max())

        inside_vals = gray[comp > 0]
        ld_vals = local_dark[comp > 0]
        inside_mean = float(inside_vals.mean()) if inside_vals.size else 255.0
        ld_mean = float(ld_vals.mean()) if ld_vals.size else 0.0

        touches_alpha = np.any((comp > 0) & (alpha_dil > 0))

        ring = cv2.dilate(
            comp,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
            iterations=1,
        )
        ring = np.clip(ring - comp, 0, 1).astype(np.uint8)
        ring = np.logical_and(ring > 0, alpha_dil > 0)
        ring_fill_vals = fill_map[ring]
        white_ratio = (
            float(np.mean(ring_fill_vals == 255)) if ring_fill_vals.size else 0.0
        )

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


def silhouette_outline(alpha_mask, outline_thickness=2):
    if outline_thickness <= 0:
        return np.zeros_like(alpha_mask)
    se = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * outline_thickness + 1, 2 * outline_thickness + 1),
    )
    er = cv2.erode(alpha_mask, se, iterations=1)
    return cv2.subtract(alpha_mask, er)


def remove_tiny_white_specks(tritone, alpha_mask, dark_gray=72, min_area=8):
    inside_white = np.logical_and(
        alpha_mask > 0, tritone == 255
    ).astype(np.uint8) * 255
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        (inside_white > 0).astype(np.uint8), 8
    )
    out = tritone.copy()
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            out[labels == i] = dark_gray
    return out


def remove_tiny_black_specks(tritone, alpha_mask, dark_gray=72, min_area=3):
    black = np.logical_and(alpha_mask > 0, tritone == 0).astype(np.uint8) * 255
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        (black > 0).astype(np.uint8), 8
    )
    out = tritone.copy()
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            out[labels == i] = dark_gray
    return out


def quantize_with_ink(clean_gray, alpha_mask, ink_mask, dark_gray=72):
    fill_map, thr = provisional_fill_quantization(
        clean_gray, alpha_mask, dark_gray=dark_gray
    )
    out = np.full_like(clean_gray, 255)
    out[alpha_mask > 0] = fill_map[alpha_mask > 0]
    out[np.logical_and(alpha_mask > 0, ink_mask > 0)] = 0
    return out, fill_map, thr


def rgba_from_quantized(tritone, alpha):
    rgba = np.zeros((tritone.shape[0], tritone.shape[1], 4), dtype=np.uint8)
    rgba[..., 0] = tritone
    rgba[..., 1] = tritone
    rgba[..., 2] = tritone
    rgba[..., 3] = alpha
    return rgba


def process_one(
    input_path,
    output_path,
    debug_dir=None,
    dark_gray=25,
    max_line_width=5,
    line_thresh=16,
    abs_black=24,
    body_smooth=2,
    bg_lo=245,
    bg_hi=251,
    alpha_shrink=1,
    alpha_close=25,
    dark_thresh=180,
    outline_thickness=2,
    white_speck_area=8,
    black_speck_area=3,
):
    img = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Could not read {input_path}")
    gray = ensure_gray(img)

    norm = normalize_background(gray)
    mask0, score, local_dark, bh1, bh2 = initial_line_mask0(
        norm,
        max_line_width=max_line_width,
        line_thresh=line_thresh,
        abs_black=abs_black,
    )

    # Alpha: flood-fill + recuperación por silueta oscura
    alpha = compute_alpha_from_gray(
        norm,
        mask0,
        bg_lo=bg_lo,
        bg_hi=bg_hi,
        shrink=alpha_shrink,
        alpha_close=alpha_close,
        dark_thresh=dark_thresh,
    )

    clean = clean_fills(norm, mask0, alpha, body_smooth=body_smooth)

    fill_map, thr = provisional_fill_quantization(clean, alpha, dark_gray=dark_gray)

    ink_core = reinforce_mask0(
        mask0,
        norm,
        local_dark,
        alpha,
        fill_map,
        max_line_width=max_line_width,
        abs_black=abs_black,
    )

    outer_outline = silhouette_outline(alpha, outline_thickness=outline_thickness)

    ink_final = cv2.bitwise_or(ink_core, outer_outline)
    ink_final = cv2.bitwise_and(ink_final, alpha)

    tritone, _, _ = quantize_with_ink(clean, alpha, ink_final, dark_gray=dark_gray)
    tritone = remove_tiny_white_specks(
        tritone, alpha, dark_gray=dark_gray, min_area=white_speck_area
    )
    tritone = remove_tiny_black_specks(
        tritone, alpha, dark_gray=dark_gray, min_area=black_speck_area
    )

    rgba = rgba_from_quantized(tritone, alpha)
    cv2.imwrite(str(output_path), rgba)

    if debug_dir:
        d = Path(debug_dir)
        d.mkdir(parents=True, exist_ok=True)
        stem = Path(input_path).stem
        cv2.imwrite(str(d / f"{stem}_01_gray.png"), gray)
        cv2.imwrite(str(d / f"{stem}_02_norm.png"), norm)
        cv2.imwrite(str(d / f"{stem}_03_score.png"), score)
        cv2.imwrite(str(d / f"{stem}_04_local_dark.png"), local_dark)
        cv2.imwrite(str(d / f"{stem}_05_bh1.png"), bh1)
        cv2.imwrite(str(d / f"{stem}_06_bh2.png"), bh2)
        cv2.imwrite(str(d / f"{stem}_07_line_mask0.png"), mask0)
        cv2.imwrite(str(d / f"{stem}_08_alpha.png"), alpha)
        cv2.imwrite(str(d / f"{stem}_09_clean.png"), clean)
        cv2.imwrite(str(d / f"{stem}_10_fill_map.png"), fill_map)
        cv2.imwrite(str(d / f"{stem}_11_ink_core.png"), ink_core)
        cv2.imwrite(str(d / f"{stem}_12_outer_outline.png"), outer_outline)
        cv2.imwrite(str(d / f"{stem}_13_ink_final.png"), ink_final)
        cv2.imwrite(str(d / f"{stem}_14_tritone.png"), tritone)
        cv2.imwrite(str(d / f"{stem}_15_rgba.png"), rgba)


def iter_inputs(input_path, glob_pat):
    p = Path(input_path)
    if p.is_dir():
        return sorted(p.glob(glob_pat))
    return [p]


def main():
    ap = argparse.ArgumentParser(
        description="Limpia fotogramas de animación 2D (gato B/N). v8."
    )
    ap.add_argument("input", help="Imagen de entrada o carpeta")
    ap.add_argument("output", help="Imagen de salida o carpeta")
    ap.add_argument("--glob", default="*.png")
    ap.add_argument("--debug-dir", default=None)
    ap.add_argument("--dark-gray", type=int, default=25,
                    help="Valor de gris oscuro para el cuerpo del gato")
    ap.add_argument("--max-line-width", type=int, default=5)
    ap.add_argument("--line-thresh", type=int, default=16)
    ap.add_argument("--abs-black", type=int, default=24)
    ap.add_argument("--body-smooth", type=int, default=2)
    ap.add_argument("--bg-lo", type=int, default=245)
    ap.add_argument("--bg-hi", type=int, default=251)
    ap.add_argument("--alpha-shrink", type=int, default=1)
    ap.add_argument(
        "--alpha-close",
        type=int,
        default=25,
        help="Radio (px) del cierre morfológico para recuperar patas (0=desactivar)",
    )
    ap.add_argument(
        "--dark-thresh",
        type=int,
        default=180,
        help="Umbral de luminosidad para considerar un píxel 'oscuro' "
             "en la silueta de recuperación",
    )
    ap.add_argument("--outline-thickness", type=int, default=2)
    ap.add_argument("--white-speck-area", type=int, default=8)
    ap.add_argument("--black-speck-area", type=int, default=3)
    args = ap.parse_args()

    inputs = iter_inputs(args.input, args.glob)
    out_path = Path(args.output)

    if len(inputs) == 1 and not Path(args.input).is_dir():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        process_one(
            inputs[0],
            out_path,
            debug_dir=args.debug_dir,
            dark_gray=args.dark_gray,
            max_line_width=args.max_line_width,
            line_thresh=args.line_thresh,
            abs_black=args.abs_black,
            body_smooth=args.body_smooth,
            bg_lo=args.bg_lo,
            bg_hi=args.bg_hi,
            alpha_shrink=args.alpha_shrink,
            alpha_close=args.alpha_close,
            dark_thresh=args.dark_thresh,
            outline_thickness=args.outline_thickness,
            white_speck_area=args.white_speck_area,
            black_speck_area=args.black_speck_area,
        )
    else:
        out_path.mkdir(parents=True, exist_ok=True)
        for inp in inputs:
            process_one(
                inp,
                out_path / (inp.stem + ".png"),
                debug_dir=args.debug_dir,
                dark_gray=args.dark_gray,
                max_line_width=args.max_line_width,
                line_thresh=args.line_thresh,
                abs_black=args.abs_black,
                body_smooth=args.body_smooth,
                bg_lo=args.bg_lo,
                bg_hi=args.bg_hi,
                alpha_shrink=args.alpha_shrink,
                alpha_close=args.alpha_close,
                dark_thresh=args.dark_thresh,
                outline_thickness=args.outline_thickness,
                white_speck_area=args.white_speck_area,
                black_speck_area=args.black_speck_area,
            )


if __name__ == "__main__":
    main()
