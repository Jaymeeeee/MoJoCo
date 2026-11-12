"""Convert insert_plug HDF5 episodes with the official LeRobot dataset API.

RGB uses H.264 videos. Depth uses LeRobot's native float32 TIFF image feature,
embedded losslessly in Parquet; it is never quantized into a depth video.
Run in the shared ``mujoco`` simulation, conversion and training environment.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import re

import h5py
import numpy as np

TASK = "Pick the plug from the orange holder and insert it into the green holder."
CAMERAS = ("head", "wrist")
JOINT_NAMES = [f"fr3_joint{i}" for i in range(1, 8)] + ["gripper_width"]
ACTION_TIMING = "active_control_target_at_observation; may change before next observation"
ALIGNED_ACTION_TIMING = "active_control_target_at_observation; held_until_next_observation"
REPO_ID = "local/insert_plug"


@dataclass(frozen=True)
class SourceEpisode:
    path: Path
    position_id: int
    source_episode_index: int


def discover_episodes(source_root: Path, positions: int = 10,
                      episodes_per_position: int = 10) -> list[SourceEpisode]:
    """Discover only the explicitly requested pos0..posN batches, never test data."""
    episodes = []
    for position in range(positions):
        folder = source_root / f"pos{position}"
        if not folder.is_dir():
            raise ValueError(f"Missing position directory: {folder}")
        candidates = []
        for subdir in folder.iterdir():
            match = re.fullmatch(r"episode_(\d+)", subdir.name)
            if not match or not subdir.is_dir():
                raise ValueError(f"Unexpected position entry: {subdir}")
            stem = subdir.name
            expected = {f"{stem}.{suffix}" for suffix in ("h5", "yaml", "xml", "mp4")}
            if {p.name for p in subdir.iterdir()} != expected:
                raise ValueError(f"Episode must contain the four completed recording files: {subdir}")
            candidates.append(SourceEpisode(subdir / f"{stem}.h5", position, int(match.group(1))))
        if len(candidates) != episodes_per_position:
            raise ValueError(f"{folder}: expected {episodes_per_position} episodes, got {len(candidates)}")
        episodes.extend(sorted(candidates, key=lambda item: item.source_episode_index))
    return episodes


def inspect_source(source: SourceEpisode) -> dict:
    with h5py.File(source.path, "r") as handle:
        if not handle.attrs.get("complete") or not handle.attrs.get("success"):
            raise ValueError(f"Incomplete or unsuccessful source: {source.path}")
        if handle.attrs.get("schema_version") != "insert-plug-v3":
            raise ValueError(f"Expected insert-plug-v3: {source.path}")
        metadata = json.loads(handle.attrs["metadata_json"])
        fps = float(metadata["sample_hz"])
        if not np.isfinite(fps) or fps <= 0 or fps != round(fps):
            raise ValueError("LeRobot export requires a positive integer nominal frame rate")
        if metadata.get("action_timing") not in (ACTION_TIMING, ALIGNED_ACTION_TIMING):
            raise ValueError(f"Unexpected action timing: {source.path}")
        times = handle["timestamp"][:]
        if len(times) < 2 or np.any(np.diff(times) <= 0) or not np.isfinite(times).all():
            raise ValueError(f"Invalid source timestamps: {source.path}")
        tolerance = float(metadata["physics_timestep"]) / 2 + 1e-8
        if not np.allclose(times, times[0] + np.arange(len(times)) / fps, atol=tolerance, rtol=0):
            raise ValueError(f"Source is not sampled at its declared nominal rate: {source.path}")
        if metadata["action_timing"] == ALIGNED_ACTION_TIMING:
            control_hz = float(metadata["control_hz"])
            period_ticks = 1 / (fps * float(metadata["physics_timestep"]))
            origin = metadata.get("sample_origin_s")
            hold = metadata.get("action_hold_duration_s")
            if (not np.isclose(control_hz, fps, rtol=1e-12, atol=0)
                    or not np.isclose(period_ticks, round(period_ticks), rtol=0, atol=1e-6)
                    or origin is None or not np.isfinite(origin)
                    or not np.isclose(origin * fps, round(origin * fps), rtol=0, atol=1e-8)
                    or not np.isclose(times[0], origin, rtol=0, atol=1e-8)
                    or not np.allclose(times, origin + np.arange(len(times)) / fps, rtol=0, atol=1e-8)
                    or hold is None or not np.isclose(hold, 1 / fps, rtol=1e-12, atol=0)):
                raise ValueError(f"Invalid held-until-next-observation action timing: {source.path}")
        height, width = handle["observations/images/head"].shape[1:3]
        shapes = {"action": (len(times), 8), "observations/joint_position": (len(times), 7),
                  "observations/gripper_width": (len(times), 1)}
        for camera in CAMERAS:
            shapes[f"observations/images/{camera}"] = (len(times), height, width, 3)
            shapes[f"observations/depth/{camera}"] = (len(times), height, width)
            depth = handle[f"observations/depth/{camera}"]
            if depth.dtype != np.float32 or depth.attrs.get("units") != "m":
                raise ValueError(f"Depth must be float32 meters: {source.path}")
        for key, shape in shapes.items():
            if handle[key].shape != shape:
                raise ValueError(f"Unexpected shape for {key}: {source.path}")
        actions = handle["action"][:]
        action_float32_exact = np.array_equal(actions, actions.astype(np.float32).astype(actions.dtype))
        if metadata["action_timing"] == ALIGNED_ACTION_TIMING and not action_float32_exact:
            raise ValueError(f"Aligned actions must preserve executed values exactly in float32: {source.path}")
        return dict(fps=int(fps), frames=len(times), height=height, width=width,
                    metadata=metadata, source_time_start=float(times[0]),
                    action_float32_conversion_lossless=bool(action_float32_exact))


def action_metadata(metadata: dict) -> dict:
    """Copy declared action timing, preserving legacy records without a hold."""
    return {key: metadata[key] for key in ("control_hz", "action_timing", "action_hold_duration_s")
            if key in metadata}


def validate_source_batch(descriptions: list[dict]) -> None:
    """Keep one camera configuration and one action timing contract per export."""
    first = descriptions[0]
    if any((d["fps"], d["height"], d["width"]) != (first["fps"], first["height"], first["width"])
           for d in descriptions):
        raise ValueError("Source camera dimensions/frame rates must agree")
    if any(action_metadata(d["metadata"]) != action_metadata(first["metadata"])
           for d in descriptions):
        raise ValueError("Source action timing/control rates/hold durations must agree; cannot mix contracts")


def make_features(height: int, width: int) -> dict:
    features = {
        "observation.state": {"dtype": "float32", "shape": (8,), "names": JOINT_NAMES},
        "action": {"dtype": "float32", "shape": (8,), "names": JOINT_NAMES},
        "source.position_id": {"dtype": "int64", "shape": (1,), "names": ["position_id"]},
        "source.episode_index": {"dtype": "int64", "shape": (1,), "names": ["source_episode_index"]},
        "source.timestamp": {"dtype": "float64", "shape": (1,), "names": ["simulation_seconds"]},
        "source.phase": {"dtype": "string", "shape": (1,), "names": None},
    }
    for camera in CAMERAS:
        features[f"observation.images.{camera}"] = {
            "dtype": "video", "shape": (3, height, width), "names": ["channels", "height", "width"]}
        features[f"observation.depth.{camera}"] = {
            "dtype": "image", "shape": (1, height, width), "names": ["channels", "height", "width"],
            "info": {"is_depth_map": True, "depth_unit": "m", "depth_convention": "camera_optical_axis_z"}}
    return features


def read_frame(handle: h5py.File, index: int, source: SourceEpisode) -> dict:
    phase = handle["phase"][index]
    if isinstance(phase, bytes):
        phase = phase.decode("utf8")
    frame = {
        "task": TASK,
        "observation.state": np.concatenate((handle["observations/joint_position"][index],
                                               handle["observations/gripper_width"][index])).astype(np.float32),
        "action": handle["action"][index].astype(np.float32),
        "source.position_id": np.array([source.position_id], dtype=np.int64),
        "source.episode_index": np.array([source.source_episode_index], dtype=np.int64),
        "source.timestamp": np.array([handle["timestamp"][index]], dtype=np.float64),
        "source.phase": str(phase),
    }
    for camera in CAMERAS:
        frame[f"observation.images.{camera}"] = handle[f"observations/images/{camera}"][index].transpose(2, 0, 1)
        depth = handle[f"observations/depth/{camera}"][index]
        if not np.isfinite(depth).all() or np.any(depth <= 0):
            raise ValueError(f"Invalid metric depth at {source.path}:{index}")
        frame[f"observation.depth.{camera}"] = depth[None, ...]
    return frame


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")


def _validate_depth_payload(root: Path, manifest: dict, sources: list[SourceEpisode]) -> dict:
    """Validate every scalar row and TIFF pixel in small Parquet batches."""
    from PIL import Image
    import pyarrow.parquet as pq
    hashes = [{camera: hashlib.sha256() for camera in CAMERAS} for _ in manifest["episodes"]]
    counts = Counter()
    current_episode = None
    expected = None
    for path in sorted((root / "data").rglob("*.parquet")):
        columns = ["episode_index", "action", "observation.state", "source.timestamp", "source.phase",
                   "source.position_id", "source.episode_index", *[f"observation.depth.{c}" for c in CAMERAS]]
        for batch in pq.ParquetFile(path).iter_batches(batch_size=16, columns=columns):
            for row in batch.to_pylist():
                episode = int(row["episode_index"])
                if current_episode != episode:
                    source = sources[episode]
                    with h5py.File(source.path, "r") as handle:
                        expected = {
                            "action": handle["action"][:].astype(np.float32),
                            "observation.state": np.concatenate((handle["observations/joint_position"][:],
                                                                 handle["observations/gripper_width"][:]), axis=1).astype(np.float32),
                            "source.timestamp": handle["timestamp"][:],
                            "source.phase": handle["phase"].asstr()[:],
                        }
                    current_episode = episode
                frame_index = counts[episode]
                for key in ("action", "observation.state"):
                    np.testing.assert_array_equal(row[key], expected[key][frame_index],
                                                  err_msg=f"{key} differs at episode {episode}, frame {frame_index}")
                np.testing.assert_array_equal(row["source.timestamp"], [expected["source.timestamp"][frame_index]])
                np.testing.assert_array_equal(row["source.position_id"], [source.position_id])
                np.testing.assert_array_equal(row["source.episode_index"], [source.source_episode_index])
                if row["source.phase"] != expected["source.phase"][frame_index]:
                    raise ValueError(f"Phase differs at episode {episode}, frame {frame_index}")
                counts[episode] += 1
                for camera in CAMERAS:
                    payload = row[f"observation.depth.{camera}"]
                    if not payload["bytes"]:
                        raise ValueError("Depth is not embedded in the dataset")
                    with Image.open(io.BytesIO(payload["bytes"])) as image:
                        pixels = np.asarray(image)
                    if pixels.dtype != np.float32:
                        raise ValueError("LeRobot depth precision changed")
                    hashes[episode][camera].update(pixels.tobytes())
    for index, episode in enumerate(manifest["episodes"]):
        if counts[index] != episode["frames"]:
            raise ValueError(f"Depth frame count differs for episode {index}")
        for camera in CAMERAS:
            if hashes[index][camera].hexdigest() != episode["depth_sha256"][camera]:
                raise ValueError(f"Depth pixels differ for episode {index}/{camera}")
    return dict(all_depth_pixels_bitwise_equal=True, depth_frames_checked=sum(counts.values()),
                all_actions_equal_source_float32=True, all_states_equal_source_float32=True,
                all_source_timestamps_exact=True, all_source_phases_equal=True,
                scalar_frames_checked=sum(counts.values()),
                action_float32_conversion_lossless=manifest["action_float32_conversion_lossless"])


def validate_export(root: Path, manifest: dict, sources: list[SourceEpisode]) -> dict:
    """Load through official LeRobot and verify episode boundaries and modalities."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    dataset = LeRobotDataset(repo_id=REPO_ID, root=root, video_backend="pyav", depth_output_unit="m")
    if dataset.num_episodes != len(sources) or dataset.num_frames != manifest["total_frames"]:
        raise ValueError("LeRobot total episode/frame counts differ from sources")
    checks = 0
    worst_rgb_error = 0.0
    for episode, source in zip(manifest["episodes"], sources, strict=True):
        with h5py.File(source.path, "r") as handle:
            for frame_index in sorted({0, episode["frames"] // 2, episode["frames"] - 1}):
                sample = dataset[episode["dataset_from_index"] + frame_index]
                expected = read_frame(handle, frame_index, source)
                for key in ("observation.state", "action", "source.position_id", "source.episode_index"):
                    np.testing.assert_array_equal(sample[key].numpy().reshape(-1), expected[key].reshape(-1))
                # LeRobot's generic torch transform casts Python float scalars
                # to float32; the Parquet source.timestamp column stays float64.
                np.testing.assert_allclose(sample["source.timestamp"].numpy().reshape(-1),
                                           expected["source.timestamp"], rtol=0, atol=5e-6)
                if int(sample["episode_index"]) != episode["episode_index"]:
                    raise ValueError("Episode boundary mismatch")
                for camera in CAMERAS:
                    np.testing.assert_array_equal(sample[f"observation.depth.{camera}"].numpy(),
                                                  expected[f"observation.depth.{camera}"])
                    rgb = sample[f"observation.images.{camera}"].numpy() * 255
                    error = float(np.abs(rgb - expected[f"observation.images.{camera}"].astype(float)).mean())
                    worst_rgb_error = max(worst_rgb_error, error)
                    if error > 12:
                        raise ValueError(f"RGB timestamp/camera mismatch for {source.path}:{frame_index}/{camera}")
                checks += 1
    return dict(official_reader_samples_checked=checks, worst_rgb_mean_absolute_error_255=worst_rgb_error,
                **_validate_depth_payload(root, manifest, sources))


def validate_published_path(root: Path, manifest: dict) -> dict:
    """Check relocated video paths and embedded depth through the official reader."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    dataset = LeRobotDataset(repo_id=REPO_ID, root=root, video_backend="pyav", depth_output_unit="m")
    indices = sorted({0, dataset.num_frames - 1,
                      *[ep["dataset_from_index"] for ep in manifest["episodes"][1::10]]})
    for index in indices:
        sample = dataset[index]
        for camera in CAMERAS:
            for key in (f"observation.images.{camera}", f"observation.depth.{camera}"):
                pixels = sample[key].numpy()
                if tuple(pixels.shape) != tuple(dataset.features[key]["shape"]) or not np.isfinite(pixels).all():
                    raise ValueError(f"Invalid published sample: {index}/{key}")
    return dict(published_reader_samples_checked=len(indices), published_reader_indices=indices)


def remove_empty_staging(root: Path) -> None:
    """Remove only empty image-writer directories; never delete recorded media."""
    images = root / "images"
    if images.is_dir():
        for folder in sorted((p for p in images.rglob("*") if p.is_dir()),
                             key=lambda p: len(p.parts), reverse=True):
            if not any(folder.iterdir()):
                folder.rmdir()
        if not any(images.iterdir()):
            images.rmdir()


def export_dataset(sources: list[SourceEpisode], output: Path, *,
                   source_root: Path, max_frames_per_episode: int | None = None) -> dict:
    """Stream episodes through the official writer and publish only after verification."""
    import importlib.metadata
    import av
    import datasets
    import torch
    from lerobot.configs import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    if not sources:
        raise ValueError("No episodes selected")
    if max_frames_per_episode is not None and max_frames_per_episode < 2:
        raise ValueError("At least two frames are required for a conversion check")
    output = output.expanduser().resolve()
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError(f"Refusing to overwrite {output} or {partial}")
    descriptions = [inspect_source(source) for source in sources]
    first = descriptions[0]
    validate_source_batch(descriptions)
    torch.set_num_threads(2)
    av.logging.set_level(av.logging.ERROR)
    datasets.disable_progress_bars()
    dataset = LeRobotDataset.create(
        repo_id=REPO_ID, root=partial, fps=first["fps"],
        features=make_features(first["height"], first["width"]),
        robot_type="franka_fr3_robotiq_2f85", video_backend="pyav", use_videos=True,
        rgb_encoder=RGBEncoderConfig(vcodec="h264", pix_fmt="yuv420p", crf=18, preset="fast", g=2),
        streaming_encoding=True, encoder_threads=2, metadata_buffer_size=10,
        data_files_size_in_mb=100, video_files_size_in_mb=100,
    )
    manifest = dict(
        format="lerobot-v3", lerobot_version=importlib.metadata.version("lerobot"), repo_id=REPO_ID,
        task=TASK, source_root=str(source_root.resolve()), fps=first["fps"], total_frames=0,
        **action_metadata(first["metadata"]),
        action_float32_conversion_lossless=all(d["action_float32_conversion_lossless"] for d in descriptions),
        policy_input_keys=["observation.images.head", "observation.images.wrist", "observation.state"],
        depth=dict(storage="float32 TIFF image bytes embedded in Parquet", units="m",
                   convention="camera_optical_axis_z", lossless=True, used_by_default_act=False),
        timestamps="LeRobot timestamp is frame_index/fps; source.timestamp preserves actual simulation time",
        episodes=[],
    )
    try:
        for index, (source, description) in enumerate(zip(sources, descriptions, strict=True)):
            count = min(description["frames"], max_frames_per_episode or description["frames"])
            hashes = {camera: hashlib.sha256() for camera in CAMERAS}
            with h5py.File(source.path, "r") as handle:
                for frame_index in range(count):
                    frame = read_frame(handle, frame_index, source)
                    for camera in CAMERAS:
                        hashes[camera].update(frame[f"observation.depth.{camera}"].tobytes())
                    dataset.add_frame(frame)
            dataset.save_episode(parallel_encoding=False)
            metadata = description["metadata"]
            manifest["episodes"].append(dict(
                episode_index=index, position_id=source.position_id,
                source_episode_index=source.source_episode_index,
                source_hdf5=os.path.relpath(source.path.resolve(), source_root.resolve()),
                frames=count, source_frames=description["frames"],
                dataset_from_index=manifest["total_frames"], dataset_to_index=manifest["total_frames"] + count,
                **action_metadata(metadata), physics_timestep=metadata["physics_timestep"],
                sample_origin_s=metadata.get("sample_origin_s", description["source_time_start"]),
                source_xy=metadata["config"]["board"]["source_xy"],
                target_xy=metadata["config"]["board"]["target_xy"],
                camera_calibration=metadata["camera_calibration"],
                depth_sha256={camera: hashes[camera].hexdigest() for camera in CAMERAS},
            ))
            manifest["total_frames"] += count
            _write_json(partial / "meta" / "export_manifest.json", manifest)
            print(f"Converted episode {index + 1}/{len(sources)}: pos{source.position_id}, {count} frames", flush=True)
        dataset.finalize()
    except BaseException:
        dataset.finalize()
        raise
    report = validate_export(partial, manifest, sources)
    report.update(episodes=len(sources), frames=manifest["total_frames"], fps=first["fps"])
    _write_json(partial / "meta" / "export_validation.json", report)
    partial.rename(output)
    report.update(validate_published_path(output, manifest))
    remove_empty_staging(output)
    _write_json(output / "meta" / "export_validation.json", report)
    report["output"] = str(output)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("outputs/datasets/insert_plug"))
    parser.add_argument("--output", type=Path, default=Path("outputs/datasets/insert_plug/lerobot"))
    parser.add_argument("--positions", type=int, default=10)
    parser.add_argument("--episodes-per-position", type=int, default=10)
    args = parser.parse_args(argv)
    if args.positions < 1 or args.episodes_per_position < 1:
        parser.error("Position and episode counts must be positive")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    sources = discover_episodes(args.source_root, args.positions, args.episodes_per_position)
    export_dataset(sources, args.output, source_root=args.source_root)
    return 0
