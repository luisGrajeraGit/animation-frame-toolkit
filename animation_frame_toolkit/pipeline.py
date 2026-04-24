"""
animation_frame_toolkit.pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Orquestador de un fotograma: llama a cada módulo en orden y produce el RGBA final.
Este es el único lugar donde los módulos se conocen entre sí.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import cv2

from .alpha import compute_alpha
from .compositing import build_tritone, to_rgba
from .config import ProcessConfig
from .debug import save_debug_frames
from .fill import clean_fills, quantize_fills
from .ink import reinforce_line_mask, remove_black_specks, remove_white_specks, silhouette_outline
from .io import read_image, write_image
from .line_detection import initial_line_mask
from .preprocessing import ensure_gray, normalize_background


def process_frame(
    input_path: "str | Path",
    output_path: "str | Path",
    config: Optional[ProcessConfig] = None,
    debug_dir: Optional["str | Path"] = None,
) -> None:
    """Procesa un único fotograma de animación.

    Lee *input_path*, extrae el personaje del fondo blanco y escribe
    un PNG RGBA en *output_path*.

    Args:
        input_path:  Ruta a la imagen de entrada (PNG/JPG con fondo blanco).
        output_path: Ruta de destino para el PNG RGBA resultante.
        config:      Parámetros de procesado. Si es None usa los defaults.
        debug_dir:   Si se indica, guarda las ~15 imágenes intermedias aquí.
    """
    if config is None:
        config = ProcessConfig()

    # 1. Lectura y preparación
    img = read_image(Path(input_path))
    gray = ensure_gray(img)
    norm = normalize_background(gray)

    # 2. Detección de líneas de tinta
    mask0, score, local_dark, bh1, bh2 = initial_line_mask(
        norm,
        max_line_width=config.max_line_width,
        line_thresh=config.line_thresh,
        abs_black=config.abs_black,
    )

    # 3. Cálculo del canal alpha
    alpha = compute_alpha(
        norm,
        mask0,
        bg_lo=config.bg_lo,
        bg_hi=config.bg_hi,
        shrink=config.alpha_shrink,
        alpha_close=config.alpha_close,
        dark_thresh=config.dark_thresh,
    )

    # 4. Limpieza de rellenos + cuantización provisional (necesaria para refuerzo de tinta)
    clean = clean_fills(norm, mask0, alpha, body_smooth=config.body_smooth)
    fill_map, _ = quantize_fills(clean, alpha, dark_gray=config.dark_gray)

    # 5. Refuerzo de máscara de tinta
    ink_core = reinforce_line_mask(
        mask0,
        norm,
        local_dark,
        alpha,
        fill_map,
        max_line_width=config.max_line_width,
        abs_black=config.abs_black,
    )

    # 6. Contorno de silueta
    outer_outline = silhouette_outline(alpha, outline_thickness=config.outline_thickness)
    ink_final = cv2.bitwise_and(cv2.bitwise_or(ink_core, outer_outline), alpha)

    # 7. Composición final
    tritone, _, _ = build_tritone(clean, alpha, ink_final, dark_gray=config.dark_gray)
    tritone = remove_white_specks(tritone, alpha, dark_gray=config.dark_gray, min_area=config.white_speck_area)
    tritone = remove_black_specks(tritone, alpha, dark_gray=config.dark_gray, min_area=config.black_speck_area)
    rgba = to_rgba(tritone, alpha)
    write_image(Path(output_path), rgba)

    # 8. Debug opcional
    if debug_dir:
        save_debug_frames(
            debug_dir,
            stem=Path(input_path).stem,
            frames={
                "01_gray": gray,
                "02_norm": norm,
                "03_score": score,
                "04_local_dark": local_dark,
                "05_bh1": bh1,
                "06_bh2": bh2,
                "07_line_mask0": mask0,
                "08_alpha": alpha,
                "09_clean": clean,
                "10_fill_map": fill_map,
                "11_ink_core": ink_core,
                "12_outer_outline": outer_outline,
                "13_ink_final": ink_final,
                "14_tritone": tritone,
                "15_rgba": rgba,
            },
        )


def iter_inputs(input_path: "str | Path", glob_pat: str = "*.png") -> List[Path]:
    """Devuelve la lista de ficheros de entrada (directorio o fichero único)."""
    p = Path(input_path)
    if p.is_dir():
        return sorted(p.glob(glob_pat))
    return [p]
