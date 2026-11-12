#!/usr/bin/env python3
"""Prepare the shared FR3 and Robotiq assets, or verify existing files offline.

Missing models are fetched at their pinned upstream commits. Existing assets
are verified against their provenance manifests and are never overwritten.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Asset preparation also works before installing this workspace as a package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mujoco_tools.assets import fetch_gripper, fetch_robot


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-only", action="store_true",
        help="Verify both existing assets without downloading or changing files.",
    )
    args = parser.parse_args(argv)
    for label, module in (("FR3", fetch_robot), ("Robotiq 2F-85", fetch_gripper)):
        operation = module.verify_model if args.verify_only else module.fetch_model
        manifest = operation(module.DEFAULT_DESTINATION)
        print(f"Verified {label}: {len(manifest['files'])} files at {manifest['commit']}")
        print(module.DEFAULT_DESTINATION)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
