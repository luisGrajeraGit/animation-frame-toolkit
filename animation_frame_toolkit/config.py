"""
animation_frame_toolkit.config
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Dataclass centralizado con todos los parámetros de procesado.
Permite cargar/guardar configuraciones desde/hacia JSON.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path


@dataclasses.dataclass
class ProcessConfig:
    """Parámetros de procesado de un fotograma de animación 2D.

    Valores por defecto ajustados para personajes cartoon B/N con fondo blanco.
    """

    # --- Cuantización ---
    dark_gray: int = 25
    """Valor de gris oscuro para el cuerpo del personaje."""

    # --- Detección de líneas ---
    max_line_width: int = 5
    line_thresh: int = 16
    abs_black: int = 24

    # --- Limpieza de rellenos ---
    body_smooth: int = 2

    # --- Alpha (umbralización de fondo) ---
    bg_lo: int = 245
    bg_hi: int = 251
    alpha_shrink: int = 1
    alpha_close: int = 25
    """Radio (px) del cierre morfológico para recuperar grietas (0 = desactivar)."""
    dark_thresh: int = 180
    """Umbral de luminosidad para considerar un píxel 'oscuro' en la silueta."""

    # --- Contorno y limpieza final ---
    outline_thickness: int = 2
    white_speck_area: int = 8
    black_speck_area: int = 3

    # --- Defringe (anti-halo y limpieza de bordes blancos) ---
    defringe_width: int = 2
    """Anchura en píxeles de la corona exterior del alpha a examinar para defringe."""
    defringe_thresh: int = 220
    """Umbral de luminosidad: píxeles frontera con gris >= este valor se eliminan del alpha."""

    # ------------------------------------------------------------------ #
    # Serialización                                                        #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_dict(cls, d: dict) -> "ProcessConfig":
        valid_keys = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid_keys})

    @classmethod
    def from_json(cls, path: "str | Path") -> "ProcessConfig":
        with open(path) as fh:
            return cls.from_dict(json.load(fh))

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def to_json(self, path: "str | Path") -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))
