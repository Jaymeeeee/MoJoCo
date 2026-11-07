"""Paths for this editable simulation workspace, independent of shell cwd."""
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = WORKSPACE_ROOT / "assets"
CONFIG_DIR = WORKSPACE_ROOT / "config"
EXAMPLES_DIR = WORKSPACE_ROOT / "examples"
OUTPUTS_DIR = WORKSPACE_ROOT / "outputs"
DATA_DIR = OUTPUTS_DIR / "datasets"
CHECKS_DIR = OUTPUTS_DIR / "_checks"
