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


def _remove_exterior_whites(
    alpha: np.ndarray,
    gray: np.ndarray,
    white_thresh: int = 200,
) -> np.ndarray:
    """Elimina blancos exteriores haciendo flood-fill desde las esquinas de la imagen.

    Lógica:
      - Los píxeles transparentes (alpha=0) se tratan como blancos en el canvas
        de flood, conectando el exterior transparente con cualquier hueco abierto
        en el contorno de tinta (p.ej. entre los dedos/garras).
      - Se hace flood-fill desde todos los píxeles del borde del frame a través
        de los píxeles blancos (>=white_thresh).
      - Todo lo alcanzado = blanco exterior → alpha=0.
      - Los blancos completamente rodeados de tinta (cara, ojos, bigotes) no son
        alcanzables desde el borde → se conservan intactos.

    Args:
        alpha:       Canal alpha uint8 (0=transparente, 255=opaco).
        gray:        Imagen gris uint8 (tritono cuantizado: 0 / dark_gray / 255).
        white_thresh: Umbral mínimo para considerar un píxel "blanco".

    Returns:
        Canal alpha corregido (ndarray uint8).
    """
    h, w = alpha.shape

    # Canvas de flood: los transparentes se tratan como blanco para que los
    # huecos abiertos en el contorno queden conectados al exterior.
    canvas = gray.copy()
    canvas[alpha == 0] = 255

    # Imagen de flood: 0 = blanco (navegable), 255 = barrera (tinta / cuerpo)
    flood = np.where(canvas >= white_thresh, 0, 255).astype(np.uint8)
    ffmask = np.zeros((h + 2, w + 2), np.uint8)

    # Sembrar desde todos los píxeles del borde del frame que sean blancos
    border_seeds = []
    for y in range(h):
        if flood[y, 0] == 0:
            border_seeds.append((0, y))
        if flood[y, w - 1] == 0:
            border_seeds.append((w - 1, y))
    for x in range(w):
        if flood[0, x] == 0:
            border_seeds.append((x, 0))
        if flood[h - 1, x] == 0:
            border_seeds.append((x, h - 1))

    for x, y in border_seeds:
        if flood[y, x] == 0:
            cv2.floodFill(flood, ffmask, (x, y), 128)

    exterior = flood == 128
    alpha_out = alpha.copy()
    alpha_out[exterior] = 0
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

    # Flood-fill desde las esquinas para eliminar blancos exteriores
    out_rgba[:, :, 3] = _remove_exterior_whites(out_rgba[:, :, 3], out_gray)

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
