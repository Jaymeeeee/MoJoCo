#!/usr/bin/env python3
"""Derive reusable convex colliders from the two supplied, unmodified STL files.

Run from the workspace's ``mujoco`` environment. The STL coordinates are
interpreted as millimetres. This preprocessing script requires numpy, scipy and
trimesh; scene construction itself only reads the resulting JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh


from mujoco_tools.paths import WORKSPACE_ROOT

ROOT = WORKSPACE_ROOT
DESTINATION = ROOT / "assets/objects/insert_plug/collision.json"


def _normalized_mesh(path: Path, centre_x: float, centre_z: float,
                     bottom_y: float) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh")
    if not mesh.is_watertight:
        raise ValueError(f"Expected a watertight STL: {path}")
    native = mesh.vertices.copy()
    # +90 degrees about X: native +Y becomes upright +Z.
    mesh.vertices = np.column_stack((native[:, 0] - centre_x,
                                     centre_z - native[:, 2],
                                     native[:, 1] - bottom_y))
    return mesh


def _slice(mesh: trimesh.Trimesh, normal, origin) -> trimesh.Trimesh:
    vertices, faces, _ = trimesh.intersections.slice_faces_plane(
        mesh.vertices, mesh.faces, plane_normal=normal, plane_origin=origin)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _piece(name: str, points: np.ndarray) -> tuple[dict, trimesh.Trimesh]:
    hull = trimesh.convex.convex_hull(points)
    if not hull.is_watertight or hull.volume <= 0:
        raise ValueError(f"Invalid convex part: {name}")
    # Keep faces as well as vertices: inspectable geometry and stable inertia.
    return {"name": name,
            "vertices": np.round(hull.vertices * 0.001, 12).tolist(),
            "faces": hull.faces.tolist(),
            "volume_m3": float(hull.volume * 1e-9)}, hull


def build() -> dict:
    holder_path = ROOT / "assets/放置桶.STL"
    bolt_path = ROOT / "assets/螺栓.STL"
    holder = _normalized_mesh(holder_path, 12.5, 12.5, 0.048290006816387177)
    bolt = _normalized_mesh(bolt_path, 13.856406211853027, 12.0, 0.0)
    floor = 5.048289775848389 - 0.048290006816387177
    top = float(holder.bounds[1, 2])
    vertices = holder.vertices
    radius = np.linalg.norm(vertices[:, :2], axis=1)
    rim = vertices[(vertices[:, 2] > top - 1e-5) & (radius < 6)]
    angles = np.sort(np.arctan2(rim[:, 1], rim[:, 0]))
    # Sector boundaries follow the actual bore vertices, so convex hulls cannot
    # bridge the hole. Vertical bands limit convexification of the flared foot.
    levels = [0.0, floor, 6.0, 7.0, 8.0, 9.0, 10.0, top]
    parts, hulls = [], []
    for band, (low, high) in enumerate(zip(levels[:-1], levels[1:])):
        sliced = _slice(holder, [0, 0, 1], [0, 0, low])
        sliced = _slice(sliced, [0, 0, -1], [0, 0, high])
        for sector, start in enumerate(angles):
            end = angles[(sector + 1) % len(angles)]
            if end < start:
                end += 2 * np.pi
            wedge = _slice(sliced, [-np.sin(start), np.cos(start), 0], [0, 0, 0])
            wedge = _slice(wedge, [np.sin(end), -np.cos(end), 0], [0, 0, 0])
            points = wedge.vertices
            if band == 0:
                points = np.vstack((points, [[0, 0, low], [0, 0, high]]))
            part, hull = _piece(f"band_{band:02d}_sector_{sector:02d}", points)
            parts.append(part)
            hulls.append(hull)
    shaft_vertices = bolt.vertices[(bolt.vertices[:, 2] <= 30.000001)
                                   & (np.linalg.norm(bolt.vertices[:, :2], axis=1) < 5.00001)]
    shaft, shaft_hull = _piece("shaft", shaft_vertices)
    head, head_hull = _piece("head", bolt.vertices[bolt.vertices[:, 2] >= 29.999999])
    bolt_error = abs(shaft_hull.volume + head_hull.volume - bolt.volume)
    if bolt_error > bolt.volume * 1e-4:
        raise ValueError(f"Convex bolt split does not match STL volume: {bolt_error}")
    return {
        "format_version": 1,
        "units": "metres",
        "generated_by": "python -m examples.insert_plug.environment.prepare_objects",
        "notes": [
            "Original STL units are inferred to be millimetres; STL has no unit metadata.",
            "Holder visual uses the original mesh. Collision is convex sectors clipped from it.",
            "Holder floor and bore are retained; exterior flutes and curved foot are locally convexified.",
            "Bolt has an exact hex head and convexified shaft preserving the rounded tip; its STL shaft triangulation is slightly nonconvex.",
        ],
        "holder": {
            "source": "assets/放置桶.STL",
            "sha256": hashlib.sha256(holder_path.read_bytes()).hexdigest(),
            "visual_scale": [0.001, 0.001, 0.001],
            "visual_position": [-0.0125, 0.0125, -0.000048290006816387177],
            "visual_quaternion": [2**-0.5, 2**-0.5, 0.0, 0.0],
            "floor_z": floor * 0.001,
            "rim_z": top * 0.001,
            "bore_nominal_radius": 0.0055,
            "source_volume_m3": float(holder.volume * 1e-9),
            "collision_volume_m3": float(sum(h.volume for h in hulls) * 1e-9),
            "height_bands_m": [value * 0.001 for value in levels],
            "angular_sectors": len(angles),
            "parts": parts,
        },
        "bolt": {
            "source": "assets/螺栓.STL",
            "sha256": hashlib.sha256(bolt_path.read_bytes()).hexdigest(),
            "visual_scale": [0.001, 0.001, 0.001],
            "visual_position": [-0.013856406211853027, 0.012, 0.0],
            "visual_quaternion": [2**-0.5, 2**-0.5, 0.0, 0.0],
            "shaft_radius": 0.005,
            "shaft_length": 0.030,
            "head_radius": 0.013856406211853027,
            "head_height": 0.0075,
            "source_volume_m3": float(bolt.volume * 1e-9),
            "collision_volume_m3": float((shaft_hull.volume + head_hull.volume) * 1e-9),
            "parts": [shaft, head],
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    manifest = build()
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_text(json.dumps(manifest, ensure_ascii=False, separators=(",", ":")) + "\n",
                           encoding="utf-8")
    holder = manifest["holder"]
    increase = holder["collision_volume_m3"] / holder["source_volume_m3"] - 1
    print(f"Wrote {DESTINATION.relative_to(ROOT)}: {len(holder['parts'])} holder parts, "
          f"2 bolt parts; holder collision volume increase {increase:.4%}")


if __name__ == "__main__":
    main()
