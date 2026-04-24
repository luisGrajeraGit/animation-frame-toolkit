"""python -m animation_frame_toolkit  →  delega al CLI unificado process_frames."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.process_frames import main  # noqa: E402

if __name__ == "__main__":
    main()
