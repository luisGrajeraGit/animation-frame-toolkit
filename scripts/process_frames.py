#!/usr/bin/env python3
"""
process_frames.py — CLI unificado para animation-frame-toolkit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Procesa fotogramas de animación 2D con fondo blanco y genera PNGs RGBA.

Características:
  - Procesado paralelo (--workers N).
  - Configuración vía fichero JSON (--config preset.json).
  - Parámetros CLI sobreescriben el JSON.
  - Barra de progreso con tqdm (si está instalado).

Ejemplos de uso::

    # Frame único
    python scripts/process_frames.py input.png output.png

    # Carpeta completa, 4 hilos
    python scripts/process_frames.py media/Gato\ 01/EXPORT_frames/ outputs/ --workers 4

    # Usar preset JSON + override puntual
    python scripts/process_frames.py frames/ out/ --config presets/gato.json --dark-gray 30

    # Con debug
    python scripts/process_frames.py frames/ out/ --debug-dir /tmp/debug

    # Guardar config activa en JSON para reutilizar
    python scripts/process_frames.py frames/ out/ --save-config my_preset.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Permite ejecutar directamente sin instalar el paquete
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from animation_frame_toolkit.batch import process_batch
from animation_frame_toolkit.config import ProcessConfig
from animation_frame_toolkit.pipeline import iter_inputs, process_frame

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


# ------------------------------------------------------------------ #
# Parseo de argumentos                                                 #
# ------------------------------------------------------------------ #

def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="process_frames",
        description="Extrae personajes 2D (fondo blanco → RGBA). animation-frame-toolkit.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("input",  help="Imagen de entrada o carpeta con frames")
    ap.add_argument("output", help="Imagen de salida o carpeta de destino")

    io_grp = ap.add_argument_group("opciones de E/S")
    io_grp.add_argument("--glob",       default="*.png",
                        help="Patrón glob al leer carpetas")
    io_grp.add_argument("--debug-dir",  default=None,  metavar="DIR",
                        help="Guarda imágenes intermedias de debug aquí")
    io_grp.add_argument("--workers",    type=int, default=1,
                        help="Nº de procesos paralelos (1 = secuencial)")

    cfg_grp = ap.add_argument_group("configuración")
    cfg_grp.add_argument("--config",      default=None, metavar="FILE",
                         help="JSON con valores de ProcessConfig")
    cfg_grp.add_argument("--save-config", default=None, metavar="FILE",
                         help="Guarda la config resultante en un fichero JSON")

    param_grp = ap.add_argument_group("parámetros de procesado (anulan el JSON)")
    param_grp.add_argument("--dark-gray",          type=int, default=None)
    param_grp.add_argument("--max-line-width",      type=int, default=None)
    param_grp.add_argument("--line-thresh",         type=int, default=None)
    param_grp.add_argument("--abs-black",           type=int, default=None)
    param_grp.add_argument("--body-smooth",         type=int, default=None)
    param_grp.add_argument("--bg-lo",               type=int, default=None)
    param_grp.add_argument("--bg-hi",               type=int, default=None)
    param_grp.add_argument("--alpha-shrink",        type=int, default=None)
    param_grp.add_argument("--alpha-close",         type=int, default=None,
                           help="Radio del cierre morfológico para patas (0=desactivar)")
    param_grp.add_argument("--dark-thresh",         type=int, default=None,
                           help="Umbral de luminosidad para silueta oscura")
    param_grp.add_argument("--outline-thickness",   type=int, default=None)
    param_grp.add_argument("--white-speck-area",    type=int, default=None)
    param_grp.add_argument("--black-speck-area",    type=int, default=None)
    return ap


def _build_config(args: argparse.Namespace) -> ProcessConfig:
    """Construye ProcessConfig con prioridad: CLI > JSON > defaults."""
    config = ProcessConfig()

    if args.config:
        config = ProcessConfig.from_json(args.config)

    cli_overrides = {
        "dark_gray":        args.dark_gray,
        "max_line_width":   args.max_line_width,
        "line_thresh":      args.line_thresh,
        "abs_black":        args.abs_black,
        "body_smooth":      args.body_smooth,
        "bg_lo":            args.bg_lo,
        "bg_hi":            args.bg_hi,
        "alpha_shrink":     args.alpha_shrink,
        "alpha_close":      args.alpha_close,
        "dark_thresh":      args.dark_thresh,
        "outline_thickness": args.outline_thickness,
        "white_speck_area": args.white_speck_area,
        "black_speck_area": args.black_speck_area,
    }
    for key, val in cli_overrides.items():
        if val is not None:
            setattr(config, key, val)

    return config


# ------------------------------------------------------------------ #
# Punto de entrada                                                     #
# ------------------------------------------------------------------ #

def main(argv: list[str] | None = None) -> None:
    ap = _build_parser()
    args = ap.parse_args(argv)

    config = _build_config(args)

    if args.save_config:
        config.to_json(args.save_config)
        print(f"Config guardada en: {args.save_config}")

    inputs = iter_inputs(args.input, args.glob)
    out_path = Path(args.output)
    debug_dir = Path(args.debug_dir) if args.debug_dir else None

    # --- Frame único ---
    is_single = len(inputs) == 1 and not Path(args.input).is_dir()
    if is_single:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        process_frame(inputs[0], out_path, config, debug_dir)
        print(f"✓ {inputs[0].name} → {out_path}")
        return

    # --- Lote ---
    total = len(inputs)
    if total == 0:
        print("ERROR: No se encontraron imágenes con el patrón dado.", file=sys.stderr)
        sys.exit(1)

    print(f"Procesando {total} fotograma(s) · workers={args.workers}")

    if _HAS_TQDM:
        bar = tqdm(total=total, unit="frame", dynamic_ncols=True)
        callback = lambda p: bar.update(1)
    else:
        done = [0]
        def callback(p: Path) -> None:
            done[0] += 1
            print(f"  [{done[0]:>4}/{total}] {p.name}")

    process_batch(
        inputs, out_path, config,
        workers=args.workers,
        debug_dir=debug_dir,
        on_progress=callback,
    )

    if _HAS_TQDM:
        bar.close()

    print(f"\n✓ {total} fotograma(s) → {out_path}/")


if __name__ == "__main__":
    main()
