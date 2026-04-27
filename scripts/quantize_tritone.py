"""
scripts/quantize_tritone.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Uniformiza imágenes RGBA (ya pre-keyed) reduciéndolas a exactamente 3 colores:
  · negro   (0)   → tinta / líneas
  · gris oscuro   → relleno de cuerpo
  · blanco  (255) → zonas claras (cara, dientes…)

El canal alpha se conserva intacto.
Los umbrales se calculan de forma adaptiva por imagen con dos pasadas de Otsu:
  1. Otsu sobre píxeles opacos → separa cluster oscuro / cluster claro.
  2. Otsu sobre el cluster oscuro → separa tinta / relleno cuerpo.

Uso:
    python3 scripts/quantize_tritone.py <input_dir> <output_dir> [opciones]

Opciones:
    --dark-gray N     Valor de gris para relleno de cuerpo (defecto: 25)
    --workers N       Procesos paralelos (defecto: 4)
    --overwrite       Sobreescribir la carpeta de salida si existe
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import sys
from pathlib import Path

import cv2
import numpy as np


# ------------------------------------------------------------------ #
# Lógica de cuantización de un frame                                  #
# ------------------------------------------------------------------ #

def _otsu_threshold(values: np.ndarray) -> int:
    """Calcula el umbral de Otsu sobre un array 1D de uint8."""
    if values.size == 0:
        return 128
    dummy = np.zeros((values.size, 1), dtype=np.uint8)
    dummy[:, 0] = values.clip(0, 255).astype(np.uint8)
    thr, _ = cv2.threshold(dummy, 0, 255, cv2.THRESH_OTSU)
    return int(thr)


def _remove_white_leakage(alpha: np.ndarray, gray: np.ndarray, white_thresh: int = 200) -> np.ndarray:
    """Hace transparentes los píxeles blancos dentro del alpha que tocan el exterior transparente.

    Los blancos interiores cerrados (cara, ojos, dientes) quedan intactos porque
    están rodeados de tinta negra y no son alcanzables por flood-fill desde el borde.

    Args:
        alpha:        Canal alpha 8-bit (0 = transparente, 255 = opaco).
        gray:         Imagen en escala de grises 8-bit con los mismos valores que el tritono.
        white_thresh: Valor mínimo de gris para considerar un píxel "blanco".

    Returns:
        Canal alpha corregido (ndarray uint8).
    """
    h, w = alpha.shape
    white_inside = ((alpha > 128) & (gray >= white_thresh)).astype(np.uint8) * 255

    # Imagen de flood: 0 donde hay blanco interior (caminable), 255 en el resto
    flood = np.where(white_inside > 0, 0, 255).astype(np.uint8)
    ffmask = np.zeros((h + 2, w + 2), np.uint8)

    # Semilla 1: blancos interiores adyacentes al exterior del alpha
    exterior = (alpha == 0).astype(np.uint8) * 255
    ext_dil = cv2.dilate(
        exterior,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    seeds_alpha = cv2.bitwise_and(white_inside, ext_dil)

    # Semilla 2: blancos en el borde del frame (personaje que sale por el encuadre)
    border_mask = np.zeros((h, w), np.uint8)
    border_mask[0, :] = 255
    border_mask[-1, :] = 255
    border_mask[:, 0] = 255
    border_mask[:, -1] = 255
    seeds_border = cv2.bitwise_and(white_inside, border_mask)

    all_seeds = cv2.bitwise_or(seeds_alpha, seeds_border)
    ys, xs = np.where(all_seeds > 0)
    for y, x in zip(ys, xs):
        if flood[y, x] == 0:
            cv2.floodFill(flood, ffmask, (int(x), int(y)), 128)

    leaked = (flood == 128).astype(np.uint8)
    alpha_out = alpha.copy()
    alpha_out[leaked > 0] = 0
    return alpha_out


def _remove_small_isolated_whites(
    alpha: np.ndarray,
    gray: np.ndarray,
    white_thresh: int = 200,
    min_cluster_area: int = 150,
    max_remove_area: int = 400,
    isolation_gap: int = 12,
) -> np.ndarray:
    """Elimina blancos pequeños que no están conectados al cluster blanco principal.

    Lógica:
      1. Encuentra todos los componentes blancos conectados dentro del alpha.
      2. Identifica el cluster principal (el más grande, típicamente cara/panza).
      3. Dilata ese cluster isolation_gap px para definir su "zona de influencia".
      4. Cualquier componente blanco con área < max_remove_area que NO toca esa zona
         se hace transparente (son huecos entre dedos/uñas).

    Los blancos grandes (cara, ojos) y los adyacentes al cluster principal
    nunca se tocan.

    Args:
        alpha:            Canal alpha uint8.
        gray:             Imagen gris uint8 (tritono ya cuantizado).
        white_thresh:     Umbral para considerar píxel "blanco".
        min_cluster_area: Tamaño mínimo del cluster principal (evita activar en frames vacíos).
        max_remove_area:  Componentes blancos con área <= esto se eliminan si están aislados.
        isolation_gap:    Radio de dilatación (px) para definir "próximo al cluster principal".

    Returns:
        Canal alpha corregido (ndarray uint8).
    """
    white_inside = ((alpha > 128) & (gray >= white_thresh)).astype(np.uint8) * 255

    n, labels, stats, _ = cv2.connectedComponentsWithStats(white_inside, 8)
    if n <= 1:
        return alpha

    # Encontrar el cluster más grande (componente principal: cara/panza)
    areas = stats[1:, cv2.CC_STAT_AREA]  # ignorar fondo (comp 0)
    largest_idx = int(np.argmax(areas)) + 1  # +1 porque saltamos el fondo

    if stats[largest_idx, cv2.CC_STAT_AREA] < min_cluster_area:
        return alpha  # frame demasiado vacío, no actuar

    # Dilatación del cluster principal para definir zona de influencia
    main_mask = (labels == largest_idx).astype(np.uint8) * 255
    se = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * isolation_gap + 1, 2 * isolation_gap + 1)
    )
    main_dilated = cv2.dilate(main_mask, se, iterations=1)

    alpha_out = alpha.copy()
    for i in range(1, n):
        if i == largest_idx:
            continue
        area = stats[i, cv2.CC_STAT_AREA]
        if area > max_remove_area:
            continue  # blanco grande (ojo, diente largo…): conservar
        comp_mask = (labels == i).astype(np.uint8) * 255
        if cv2.bitwise_and(comp_mask, main_dilated).any():
            continue  # toca el cluster principal: conservar (uñas conectadas)
        # Blanco pequeño aislado → transparente
        alpha_out[comp_mask > 0] = 0

    return alpha_out


def quantize_frame(
    input_path: "str | Path",
    output_path: "str | Path",
    dark_gray: int = 25,
) -> str:
    """Cuantiza un frame RGBA a 3 valores: 0 / dark_gray / 255.

    Returns:
        Nombre del archivo procesado.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    img = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"No se pudo leer: {input_path}")

    # Normalizar a uint8 BGRA
    if img.dtype == np.uint16:
        img = (img.astype(np.float32) / 257.0).clip(0, 255).astype(np.uint8)
    if img.ndim == 2:
        # Imagen en escala de grises → añadir alpha opaco
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        img[:, :, 3] = 255
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        img[:, :, 3] = 255

    alpha = img[:, :, 3]
    opaque = alpha > 128

    # Luminancia (solo canales RGB)
    gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)

    opaque_vals = gray[opaque]

    if opaque_vals.size < 10:
        # Frame prácticamente vacío → copiar tal cual
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), img)
        return input_path.name

    # --- Umbral 1: Otsu sobre todos los píxeles opacos (oscuro vs claro) ---
    thr_dark_light = _otsu_threshold(opaque_vals)

    dark_vals = opaque_vals[opaque_vals < thr_dark_light]

    # --- Umbral 2: Otsu dentro del cluster oscuro (tinta vs relleno) ---
    if dark_vals.size > 10:
        thr_ink_fill = _otsu_threshold(dark_vals)
        # Evitar umbral demasiado alto dentro del cluster oscuro:
        # si otsu interno supera 60 % del umbral oscuro/claro, lo recortamos
        thr_ink_fill = min(thr_ink_fill, max(15, thr_dark_light // 3))
    else:
        thr_ink_fill = 15  # fallback razonable

    # --- Construir imagen cuantizada ---
    out_gray = np.full_like(gray, 255, dtype=np.uint8)  # todo blanco de inicio
    mask_dark = opaque & (gray < thr_dark_light)
    # Tinta (muy oscuro) → 0
    out_gray[mask_dark & (gray < thr_ink_fill)] = 0
    # Relleno cuerpo → dark_gray
    out_gray[mask_dark & (gray >= thr_ink_fill)] = dark_gray
    # Píxeles transparentes no importan (su RGB no se verá)

    out_bgr = cv2.cvtColor(out_gray, cv2.COLOR_GRAY2BGR)
    out_rgba = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2BGRA)
    out_rgba[:, :, 3] = alpha  # conservar alpha original

    # Eliminar blancos que filtran hacia el exterior transparente (flood-fill)
    out_rgba[:, :, 3] = _remove_white_leakage(out_rgba[:, :, 3], out_gray)

    # Eliminar blancos pequeños aislados del cluster principal (huecos entre dedos/uñas)
    out_rgba[:, :, 3] = _remove_small_isolated_whites(out_rgba[:, :, 3], out_gray)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), out_rgba)
    return input_path.name


# ------------------------------------------------------------------ #
# CLI                                                                  #
# ------------------------------------------------------------------ #

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Cuantiza carpeta de PNGs RGBA a 3 colores (tinta / cuerpo / blanco).",
    )
    p.add_argument("input_dir", help="Carpeta de entrada con PNGs RGBA.")
    p.add_argument("output_dir", help="Carpeta de destino.")
    p.add_argument("--dark-gray", type=int, default=25, metavar="N",
                   help="Valor gris para relleno de cuerpo (defecto: 25).")
    p.add_argument("--workers", type=int, default=4, metavar="N",
                   help="Procesos paralelos (defecto: 4).")
    p.add_argument("--overwrite", action="store_true",
                   help="Sobreescribir si la carpeta de salida ya existe.")
    return p.parse_args(argv)


def _job(args_tuple):
    """Función de nivel de módulo para que ProcessPoolExecutor pueda serializarla."""
    src, dst, dark_gray = args_tuple
    return quantize_frame(src, dst, dark_gray=dark_gray)


def main(argv=None):
    args = _parse_args(argv)

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)

    if not in_dir.is_dir():
        sys.exit(f"Error: la carpeta de entrada no existe: {in_dir}")

    frames = sorted(in_dir.glob("*.png"))
    if not frames:
        sys.exit(f"No se encontraron PNGs en: {in_dir}")

    if out_dir.exists() and not args.overwrite:
        pass
    out_dir.mkdir(parents=True, exist_ok=True)

    total = len(frames)
    print(f"Cuantizando {total} frame(s) → {out_dir}  [workers={args.workers}]")

    jobs = [(f, out_dir / f.name, args.dark_gray) for f in frames]

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_job, j): j for j in jobs}
        for done_i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            name = fut.result()
            print(f"  [{done_i:4d}/{total}] {name}")

    print(f"\n✓ {total} frame(s) → {out_dir}/")


if __name__ == "__main__":
    main()
