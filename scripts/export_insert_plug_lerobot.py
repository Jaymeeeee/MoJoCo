#!/usr/bin/env python
"""Local HDF5-to-LeRobot conversion; run with ``conda activate lerobot``."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mujoco_tools.lerobot_export import main

if __name__ == "__main__":
    raise SystemExit(main())
