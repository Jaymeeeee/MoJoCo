#!/usr/bin/env python3
"""Inspect a local STL without guessing its units or changing its geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


def inspect_stl(path: str | Path, *, units: str) -> dict:
    """Return dimensions in SI units; the source file is never modified.

    STL does not contain an authoritative length unit. ``units`` is therefore
    required, rather than inferred from the mesh's size.
    """
    import numpy as np
    import trimesh

    source = Path(path).expanduser().resolve(strict=True)
    if source.suffix.lower() != ".stl":
        raise ValueError("Expected an .stl file")
    if units not in {"mm", "m"}:
        raise ValueError("units must be 'mm' or 'm'")
    mesh = trimesh.load_mesh(source, file_type="stl", process=True)
    if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
        raise ValueError("STL contains no triangular surface")
    if not np.isfinite(mesh.vertices).all():
        raise ValueError("STL contains non-finite vertex coordinates")

    scale = 0.001 if units == "mm" else 1.0
    mesh = mesh.copy()
    mesh.apply_scale(scale)
    vertices = np.unique(mesh.vertices, axis=0)
    if len(vertices) < 3:
        raise ValueError("At least three distinct vertices are needed")

    # PCA describes geometry orientation only. It does not choose the insertion
    # axis, fix CAD orientation, or assume that the long axis must be vertical.
    centered = vertices - vertices.mean(axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(centered.T @ centered / len(vertices))
    order = np.argsort(eigenvalues)[::-1]
    axes = eigenvectors[:, order].T
    # Resolve sign for reproducible reports; both signs denote the same axis.
    for row in axes:
        if row[np.argmax(np.abs(row))] < 0:
            row *= -1
    projected = centered @ axes.T
    valid_volume = bool(mesh.is_volume)
    warnings = []
    if not valid_volume:
        warnings.append("Mesh does not define a consistently oriented closed volume; volume and mass properties are not reported.")
    if not mesh.is_watertight:
        warnings.append("Mesh is not watertight; repair or separate visual/collision assets before using it for contact geometry.")
    warnings.append("The insertion axis and grasp pose must be specified by the user; PCA does not identify task semantics.")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    return {
        "file": str(source),
        "sha256": digest.hexdigest(),
        "input_units": units,
        "scale_to_meters": scale,
        "triangle_count": int(len(mesh.faces)),
        "vertex_count_after_merge": int(len(mesh.vertices)),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "valid_volume": valid_volume,
        "aabb_min_m": mesh.bounds[0].tolist(),
        "aabb_max_m": mesh.bounds[1].tolist(),
        "aabb_size_m": mesh.extents.tolist(),
        "aabb_size_mm": (mesh.extents * 1000).tolist(),
        "aabb_center_m": mesh.bounds.mean(axis=0).tolist(),
        "volume_m3": float(mesh.volume) if valid_volume else None,
        "center_of_mass_m": mesh.center_mass.tolist() if valid_volume else None,
        "principal_axes": {
            "method": "PCA of unique surface vertices; rows sorted by decreasing variance",
            "directions_in_original_frame": axes.tolist(),
            "projected_extent_m": np.ptp(projected, axis=0).tolist(),
            "variance_m2": eigenvalues[order].tolist(),
        },
        "warnings": warnings,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Local STL file (read only)")
    parser.add_argument("--units", required=True, choices=("mm", "m"), help="Explicit units of the STL coordinates")
    args = parser.parse_args(argv)
    try:
        report = inspect_stl(args.path, units=args.units)
    except (OSError, ValueError, ImportError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps({"ok": True, **report}, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
