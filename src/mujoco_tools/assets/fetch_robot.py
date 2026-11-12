#!/usr/bin/env python3
"""Fetch the official FR3 Menagerie asset at a pinned Git commit.

Only the franka_fr3 directory is checked out. The upstream LICENSE and README
are retained and SOURCE.json records the commit and SHA-256 of every file.
The downloader uses git, so Git's usual HTTPS/proxy configuration is honoured.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from mujoco_tools.paths import WORKSPACE_ROOT

REPOSITORY = "https://github.com/google-deepmind/mujoco_menagerie.git"
PINNED_COMMIT = "98a70c569c4c1b6da467a20012f766ee1f8dc521"
MODEL_SUBDIRECTORY = "franka_fr3"
PROJECT_ROOT = WORKSPACE_ROOT
DEFAULT_DESTINATION = PROJECT_ROOT / "assets" / "robots" / MODEL_SUBDIRECTORY


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model(destination: Path, expected_commit: str = PINNED_COMMIT) -> dict:
    """Check provenance and file contents without contacting the network."""
    manifest_path = destination / "SOURCE.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Missing model manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("commit") != expected_commit or manifest.get("repository") != REPOSITORY:
        raise RuntimeError("Robot provenance does not match the requested source/commit")
    expected_files = manifest.get("files", {})
    if not expected_files or not {"fr3.xml", "scene.xml", "LICENSE"}.issubset(expected_files):
        raise RuntimeError("Robot manifest lacks required model or licence files")
    actual_files = {
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*")
        if path.is_file() and path.name != "SOURCE.json"
    }
    if actual_files != set(expected_files):
        raise RuntimeError("Robot file inventory differs from SOURCE.json")
    for relative, expected_hash in expected_files.items():
        path = destination / relative
        if path.is_symlink() or not path.resolve().is_relative_to(destination.resolve()):
            raise RuntimeError(f"Unexpected path in robot manifest: {relative}")
        if _sha256(path) != expected_hash:
            raise RuntimeError(f"Robot checksum mismatch: {relative}")
    return manifest


def fetch_model(destination: Path = DEFAULT_DESTINATION, commit: str = PINNED_COMMIT) -> dict:
    """Fetch a model once; an existing checkout is verified, never overwritten."""
    destination = Path(destination).resolve()
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("commit must be a full 40-character lowercase Git SHA")
    if destination.exists():
        return verify_model(destination, commit)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fr3-fetch-") as temporary:
        checkout = Path(temporary) / "menagerie"
        checkout.mkdir()

        def git(*arguments: str) -> str:
            result = subprocess.run(
                ["git", "-C", str(checkout), *arguments],
                check=True, text=True, stdout=subprocess.PIPE,
            )
            return result.stdout.strip()

        git("init", "--quiet")
        git("remote", "add", "origin", REPOSITORY)
        git("config", "remote.origin.promisor", "true")
        git("config", "remote.origin.partialclonefilter", "blob:none")
        git("sparse-checkout", "init", "--cone")
        git("sparse-checkout", "set", MODEL_SUBDIRECTORY)
        git("fetch", "--quiet", "--depth=1", "--filter=blob:none", "origin", commit)
        git("checkout", "--quiet", "--detach", "FETCH_HEAD")
        actual_commit = git("rev-parse", "HEAD")
        if actual_commit != commit:
            raise RuntimeError(f"Git returned {actual_commit}, expected {commit}")
        source = checkout / MODEL_SUBDIRECTORY
        if any(path.is_symlink() for path in source.rglob("*")):
            raise RuntimeError("Unexpected symlink in upstream model")
        shutil.copytree(source, destination)
    manifest = {
        "repository": REPOSITORY,
        "commit": commit,
        "subdirectory": MODEL_SUBDIRECTORY,
        "source_url": f"https://github.com/google-deepmind/mujoco_menagerie/tree/{commit}/{MODEL_SUBDIRECTORY}",
        "license": "Apache-2.0; retained verbatim in LICENSE",
        "local_modifications": "None. Generated scenes modify a separate in-memory copy.",
        "files": {
            path.relative_to(destination).as_posix(): _sha256(path)
            for path in sorted(destination.rglob("*")) if path.is_file()
        },
    }
    (destination / "SOURCE.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return verify_model(destination, commit)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--commit", default=PINNED_COMMIT)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    manifest = (
        verify_model(args.destination.resolve(), args.commit)
        if args.verify_only else fetch_model(args.destination, args.commit)
    )
    print(f"Verified FR3: {len(manifest['files'])} files at {manifest['commit']}")
    print(args.destination.resolve())


if __name__ == "__main__":
    main()
