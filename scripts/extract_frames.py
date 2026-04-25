#!/usr/bin/env python3
"""Extract frames from a video into an `input/` folder (lossless PNG by default).

Usage:
  python3 scripts/extract_frames.py /path/to/video_or_dir

If a directory is provided the script will look for a video file inside it.
By default the output folder is `<video_parent>/input`. Use `--output` to override.

The script will prefer `ffmpeg` (if available) for speed and exact frame extraction.
If `ffmpeg` is not available it will try to use OpenCV (`cv2`).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def find_video_in_dir(d: Path) -> Path | None:
    # check immediate children first
    for p in sorted(d.iterdir()):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            return p
    # recursive fallback
    for p in d.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            return p
    return None


def run_ffmpeg_extract(video: Path, out_dir: Path, fmt: str = "png") -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found")

    out_pattern = str(out_dir / ("%06d." + fmt))
    # Use the codec matching the extension for lossless output (png/tiff)
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-vsync",
        "0",
        "-start_number",
        "0",
        "-y",
        out_pattern,
    ]

    # For some formats prefer explicit codec
    if fmt.lower() in ("png", "tiff", "tif"):
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-vsync",
            "0",
            "-start_number",
            "0",
            "-c:v",
            fmt.lower(),
            "-y",
            out_pattern,
        ]

    subprocess.run(cmd, check=True)


def run_opencv_extract(video: Path, out_dir: Path, fmt: str = "png") -> None:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover - runtime fallback
        raise RuntimeError("OpenCV (cv2) is not available") from exc

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video}")

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out_path = out_dir / f"{idx:06d}.{fmt}"
        if fmt.lower() == "png":
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        else:
            cv2.imwrite(str(out_path), frame)
        idx += 1

    cap.release()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extrae frames de un vídeo sin pérdida (PNG por defecto)")
    p.add_argument("source", help="Ruta de vídeo o carpeta que contiene el vídeo")
    p.add_argument("-o", "--output", help="Carpeta de salida. Por defecto: <video_parent>/input")
    p.add_argument("--format", choices=["png", "tiff", "jpg", "jpeg"], default="png",
                   help="Formato de salida (png es lossless)")
    p.add_argument("--ffmpeg", action="store_true", help="Forzar uso de ffmpeg (si está disponible)")
    p.add_argument("--overwrite", action="store_true", help="Borrar la carpeta de salida si existe")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    src = Path(args.source)

    if src.is_dir():
        video = find_video_in_dir(src)
        if video is None:
            print(f"No se encontró ningún archivo de vídeo en {src}", file=sys.stderr)
            raise SystemExit(2)
    elif src.is_file():
        video = src
    else:
        print(f"No existe la ruta: {src}", file=sys.stderr)
        raise SystemExit(2)

    out_dir = Path(args.output) if args.output else video.parent / "input"

    if out_dir.exists():
        if args.overwrite:
            shutil.rmtree(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
        else:
            if any(out_dir.iterdir()):
                print(f"La carpeta de salida {out_dir} ya existe y no está vacía. Use --overwrite para reemplazar.",
                      file=sys.stderr)
                raise SystemExit(2)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    fmt = args.format

    tried_ffmpeg = False
    if args.ffmpeg:
        tried_ffmpeg = True
        try:
            run_ffmpeg_extract(video, out_dir, fmt)
            print("Frames extraídos con ffmpeg en:", out_dir)
            return
        except Exception as exc:  # fall back to OpenCV
            print("ffmpeg falló:", exc, file=sys.stderr)

    # If ffmpeg is available and not explicitly disabled, prefer it for speed/accuracy
    if not tried_ffmpeg and shutil.which("ffmpeg"):
        try:
            run_ffmpeg_extract(video, out_dir, fmt)
            print("Frames extraídos con ffmpeg en:", out_dir)
            return
        except Exception as exc:
            print("ffmpeg falló, intentando OpenCV:", exc, file=sys.stderr)

    # final fallback to OpenCV
    try:
        run_opencv_extract(video, out_dir, fmt)
        print("Frames extraídos con OpenCV en:", out_dir)
    except Exception as exc:
        print("Error extrayendo frames:", exc, file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
