"""
animation_frame_toolkit.batch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Procesado en lote con soporte opcional de paralelismo por procesos.

Uso típico::

    from animation_frame_toolkit import ProcessConfig, process_batch
    from animation_frame_toolkit.pipeline import iter_inputs

    inputs = iter_inputs("media/Gato 01/EXPORT_frames")
    process_batch(inputs, "media/Gato 01/EXPORT_frames_alpha", workers=4)
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, List, Optional

from .config import ProcessConfig
from .pipeline import process_frame


# ------------------------------------------------------------------ #
# Worker (debe estar en top-level para ser picklable en multiproceso)  #
# ------------------------------------------------------------------ #

def _worker(args: tuple) -> Path:
    inp, out, config, debug_dir = args
    process_frame(inp, out, config, debug_dir)
    return Path(inp)


# ------------------------------------------------------------------ #
# API pública                                                          #
# ------------------------------------------------------------------ #

def process_batch(
    inputs: List["str | Path"],
    output_dir: "str | Path",
    config: Optional[ProcessConfig] = None,
    workers: int = 1,
    debug_dir: Optional["str | Path"] = None,
    on_progress: Optional[Callable[[Path], None]] = None,
) -> List[Path]:
    """Procesa una lista de imágenes, con soporte de paralelismo.

    Args:
        inputs:      Lista de rutas de imagen de entrada.
        output_dir:  Directorio de destino para los PNGs RGBA.
        config:      Parámetros de procesado (defaults si None).
        workers:     Nº de procesos paralelos. 1 = secuencial.
        debug_dir:   Carpeta para imágenes intermedias de debug.
        on_progress: Callback llamado con la ruta procesada al acabar cada frame.

    Returns:
        Lista de rutas de entrada procesadas en el orden en que terminaron.
    """
    if config is None:
        config = ProcessConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        (Path(inp), output_dir / (Path(inp).stem + ".png"), config, debug_dir)
        for inp in inputs
    ]

    processed: List[Path] = []

    if workers <= 1:
        for task in tasks:
            result = _worker(task)
            processed.append(result)
            if on_progress:
                on_progress(result)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_worker, t): t for t in tasks}
            for future in as_completed(futures):
                result = future.result()
                processed.append(result)
                if on_progress:
                    on_progress(result)

    return processed
