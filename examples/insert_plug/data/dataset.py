"""Streaming, explicit observation/action records for plug insertion demonstrations.

An observation at time t is recorded before a physics step, with the active
controller target in ``action``: seven joint targets (rad) then full gripper
opening (m). The controller may update between recorded observations. Privileged
poses are for diagnostics, not policy input.
This is a project HDF5 format; it is not a LeRobotDataset export.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np


SCHEMA_VERSION = "insert-plug-v3"
LEGACY_SCHEMA_VERSION = "peg-transfer-v1"
LEGACY_RGBD_SCHEMA_VERSION = "peg-transfer-v2"
ACTION_DESCRIPTION = "[fr3_joint1..fr3_joint7 target radians, full gripper opening meters]"
LEGACY_TIMING_DESCRIPTION = "observation_t_before_action_t; action_t applied until next observation"
TIMING_DESCRIPTION = "observation_t_before_physics_step; action_t is active control target and may change before next observation"
CAMERAS = ("head", "wrist")
LEGACY_CAMERAS = ("third_person", "wrist")
DEPTH_UNITS = "m"
DEPTH_CONVENTION = "camera_optical_axis_z"
FIELDS = {
    "joint_position": "observations/joint_position",
    "joint_velocity": "observations/joint_velocity",
    "gripper_width": "observations/gripper_width",
    "action": "action",
    "qpos": "simulation/qpos",
    "qvel": "simulation/qvel",
    "ctrl": "simulation/ctrl",
    "peg_pose": "privileged/peg_pose",
    "source_pose": "privileged/source_pose",
    "target_pose": "privileged/target_pose",
}
SIZES = {
    "joint_position": 7,
    "joint_velocity": 7,
    "gripper_width": 1,
    "action": 8,
    "peg_pose": 7,
    "source_pose": 7,
    "target_pose": 7,
}


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize metadata value of type {type(value).__name__}")


def _json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, default=_json_default, ensure_ascii=False, allow_nan=False)


def _feature_map(depth_enabled: bool) -> dict[str, str]:
    modalities = ("images", "depth") if depth_enabled else ("images",)
    return {
        f"observation.{modality}.{camera}": f"/observations/{modality}/{camera}"
        for modality in modalities for camera in CAMERAS
    }


def _sampling_grid(metadata: Mapping[str, Any]) -> tuple[float, float, float] | None:
    """Return the declared nearest-tick grid, rejecting ambiguous metadata."""
    mode = metadata.get("sampling_timing")
    if mode is None:
        return None
    if mode != "nearest_physics_tick":
        raise ValueError("Unsupported metadata.sampling_timing")
    try:
        hz, timestep, origin, control_hz = (
            float(metadata[key])
            for key in ("sample_hz", "physics_timestep", "sample_origin_s", "control_hz")
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Nearest-tick sampling requires sample_hz, physics_timestep, sample_origin_s and control_hz") from error
    if not np.isfinite([hz, timestep, origin, control_hz]).all() or min(hz, timestep, control_hz) <= 0 or origin < 0:
        raise ValueError("Sampling frequencies/timestep must be finite and positive; origin must be nonnegative")
    if hz * timestep > 1 + 1e-12:
        raise ValueError("sample_hz cannot exceed the physics frequency")
    ratio = 1 / (control_hz * timestep)
    if not np.isclose(ratio, round(ratio), rtol=0, atol=1e-8):
        raise ValueError("control_hz must divide the physics frequency")
    return hz, timestep, origin


class EpisodeWriter:
    """Write one episode to ``<path>.partial`` and publish it on finalize.

    Neither an existing final file nor an existing partial file is overwritten.
    The caller must explicitly finalize; a context manager exit only closes the
    partial file. This preserves interrupted runs without marking them complete.
    RGB cameras are named ``head`` and ``wrist``. Optional ``depth`` contains the
    same cameras as float32 optical-axis Z in meters, including finite far-plane
    values. Depth presence must remain constant throughout the episode.
    """

    def __init__(self, path: str | Path, metadata: Mapping[str, Any] | None = None):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.partial_path = self.path.with_name(self.path.name + ".partial")
        if self.path.exists():
            raise FileExistsError(self.path)
        self.metadata = dict(metadata or {})
        self._depth_enabled: bool | None = self.metadata.get("depth_enabled")
        if "depth_enabled" in self.metadata:
            if not isinstance(self._depth_enabled, (bool, np.bool_)):
                raise ValueError("metadata.depth_enabled must be a boolean")
            self._depth_enabled = bool(self._depth_enabled)
            self.metadata["depth_enabled"] = self._depth_enabled
        self.metadata["feature_map"] = _feature_map(bool(self._depth_enabled))
        if "sample_hz" in self.metadata:
            hz = float(self.metadata["sample_hz"])
            if not np.isfinite(hz) or hz <= 0:
                raise ValueError("metadata.sample_hz must be finite and positive")
        _sampling_grid(self.metadata)
        metadata_json = _json(self.metadata)
        self._file = h5py.File(self.partial_path, "x")
        self._file.attrs.update(
            schema_version=SCHEMA_VERSION,
            complete=False,
            steps=0,
            action_description=ACTION_DESCRIPTION,
            observation_timing=TIMING_DESCRIPTION,
            simulation_ctrl_timing="active actuator commands at observation_t, after current controller target was applied",
            pose_convention="[x,y,z,qw,qx,qy,qz], world frame, meters",
            privileged_policy_input=False,
            metadata_json=metadata_json,
        )
        self._datasets: dict[str, h5py.Dataset] = {}
        self.steps = 0
        self._last_timestamp: float | None = None
        self._closed = False
        self._finalized = False

    def _prepare(self, record: Mapping[str, Any]) -> dict[str, Any]:
        timestamp = float(record["timestamp"])
        if not np.isfinite(timestamp) or timestamp < 0:
            raise ValueError("timestamp must be finite and nonnegative")
        if self._last_timestamp is not None and timestamp <= self._last_timestamp:
            raise ValueError("timestamp must increase strictly between observations")
        phase = str(record["phase"])
        if not phase or "\x00" in phase:
            raise ValueError("phase must be a nonempty string without NUL")
        values: dict[str, Any] = {"timestamp": timestamp, "phase": phase}
        for key, dataset_path in FIELDS.items():
            value = np.asarray(record[key], dtype=np.float64)
            if key == "gripper_width" and value.ndim == 0:
                value = value.reshape(1)
            if value.ndim != 1 or value.size == 0:
                raise ValueError(f"{key} must be a nonempty vector")
            if key in SIZES and value.shape != (SIZES[key],):
                raise ValueError(f"{key} must have shape ({SIZES[key]},), got {value.shape}")
            if not np.isfinite(value).all():
                raise ValueError(f"{key} contains non-finite values")
            if key.endswith("_pose") and not np.isclose(np.linalg.norm(value[3:]), 1, atol=1e-3):
                raise ValueError(f"{key} quaternion must have unit length (wxyz)")
            values[dataset_path] = value
        if values["observations/gripper_width"][0] < -1e-6 or values["action"][-1] < 0:
            raise ValueError("gripper widths are full, nonnegative openings in meters")
        images = record["images"]
        if not isinstance(images, Mapping) or set(images) != set(CAMERAS):
            raise ValueError("images must contain exactly head and wrist cameras")
        depth_enabled = "depth" in record
        if self._depth_enabled is not None and depth_enabled != self._depth_enabled:
            raise ValueError("depth presence must match metadata.depth_enabled and remain constant")
        depth = record.get("depth")
        if depth_enabled and (not isinstance(depth, Mapping) or set(depth) != set(CAMERAS)):
            raise ValueError("depth must contain exactly head and wrist cameras")
        for camera in CAMERAS:
            image = np.asarray(images[camera])
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"images.{camera} must be uint8 RGB with shape (H,W,3)")
            if min(image.shape[:2]) <= 0:
                raise ValueError(f"images.{camera} has empty dimensions")
            values[f"observations/images/{camera}"] = image
            if depth_enabled:
                depth_image = np.asarray(depth[camera])
                if depth_image.dtype != np.float32 or depth_image.shape != image.shape[:2]:
                    raise ValueError(f"depth.{camera} must be float32 with shape {image.shape[:2]}")
                if not np.isfinite(depth_image).all() or np.any(depth_image <= 0):
                    raise ValueError(f"depth.{camera} must contain finite positive distances in meters")
                values[f"observations/depth/{camera}"] = depth_image
        for name, value in values.items():
            if name in self._datasets and np.shape(value) != self._datasets[name].shape[1:]:
                raise ValueError(f"Shape changed for {name}: {np.shape(value)}")
        return values

    def append(self, record: Mapping[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("EpisodeWriter is closed")
        values = self._prepare(record)  # Validate the entire frame before any resize.
        if not self._datasets:
            for name, value in values.items():
                shape = np.shape(value)
                dtype = h5py.string_dtype("utf-8") if name == "phase" else np.asarray(value).dtype
                visual = name.startswith(("observations/images/", "observations/depth/"))
                self._datasets[name] = self._file.create_dataset(
                    name,
                    shape=(0, *shape),
                    maxshape=(None, *shape),
                    chunks=(1, *shape) if visual else (128, *shape),
                    dtype=dtype,
                    compression="lzf",
                )
                if name.startswith("observations/depth/"):
                    self._datasets[name].attrs.update(units=DEPTH_UNITS, convention=DEPTH_CONVENTION)
        try:
            for name, value in values.items():
                dataset = self._datasets[name]
                dataset.resize(self.steps + 1, axis=0)
                dataset[self.steps] = value
        except Exception:
            for dataset in self._datasets.values():
                dataset.resize(self.steps, axis=0)
            self._file.flush()
            raise
        self.steps += 1
        self._depth_enabled = "depth" in record
        self.metadata.update(depth_enabled=self._depth_enabled, feature_map=_feature_map(self._depth_enabled))
        self._file.attrs["metadata_json"] = _json(self.metadata)
        self._last_timestamp = values["timestamp"]
        self._file.attrs["steps"] = self.steps
        # Flushing each frame also makes an interrupted partial inspectable.
        self._file.flush()

    def finalize(self, success: bool, failure_reason: str = "", **extra_metadata: Any) -> Path:
        if self._closed:
            raise RuntimeError("EpisodeWriter is closed")
        if self.steps == 0:
            raise ValueError("Cannot finalize an empty episode")
        if success and failure_reason:
            raise ValueError("A successful episode cannot have a failure reason")
        if not success and not failure_reason:
            failure_reason = "task_not_completed"
        if "depth_enabled" in extra_metadata and (
            not isinstance(extra_metadata["depth_enabled"], (bool, np.bool_))
            or bool(extra_metadata["depth_enabled"]) != self._depth_enabled
        ):
            raise ValueError("metadata.depth_enabled cannot change during finalize")
        if "feature_map" in extra_metadata and extra_metadata["feature_map"] != _feature_map(self._depth_enabled):
            raise ValueError("metadata.feature_map must match recorded RGB/depth datasets")
        metadata = {**self.metadata, **extra_metadata}
        metadata.update(
            success=bool(success), failure_reason=str(failure_reason), steps=self.steps,
            depth_enabled=self._depth_enabled, feature_map=_feature_map(self._depth_enabled),
        )
        metadata_json = _json(metadata)
        self.metadata = metadata
        self._file.attrs.update(
            metadata_json=metadata_json,
            success=bool(success),
            failure_reason=str(failure_reason),
            complete=True,
            steps=self.steps,
        )
        self._file.flush()
        self.close()
        # Hard-link publication is atomic and cannot clobber a concurrent writer.
        # Both paths are in one directory/filesystem; unlink removes the staging name.
        os.link(self.partial_path, self.path)
        self.partial_path.unlink()
        self._finalized = True
        return self.path

    def close(self) -> None:
        if not self._closed:
            self._file.close()
            self._closed = True

    def __enter__(self) -> "EpisodeWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def read_episode(
    path: str | Path, include_images: bool = True, *, include_depth: bool | None = None,
) -> dict[str, Any]:
    """Load historical v1/v2 and current v3 arrays with their HDF5 paths.

    ``include_images=False`` skips both RGB and depth unless ``include_depth``
    explicitly overrides it. Prefer streaming for large files.
    """
    if include_depth is None:
        include_depth = include_images
    arrays: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as handle:
        metadata = json.loads(handle.attrs.get("metadata_json", "{}"))

        def read(name: str, obj: Any) -> None:
            if not isinstance(obj, h5py.Dataset):
                return
            if not include_images and name.startswith("observations/images/"):
                return
            if not include_depth and name.startswith("observations/depth/"):
                return
            arrays[name] = obj.asstr()[...] if name == "phase" else obj[...]

        handle.visititems(read)
        return {"metadata": metadata, "data": arrays, "complete": bool(handle.attrs.get("complete", False))}


def validate_episode(path: str | Path) -> dict[str, Any]:
    """Validate shape, synchronization, numerical values and the declared schema.

    Reports file/schema errors rather than raising, making it suitable for batch
    quality checks. Images are scanned frame-wise and numerical arrays in chunks.
    Task success is reported separately; a failed rollout can be a valid recording.
    """
    result: dict[str, Any] = {
        "path": str(Path(path)), "valid": False, "errors": [], "warnings": [],
        "steps": 0, "success": False,
    }
    errors = result["errors"]
    try:
        with h5py.File(path, "r") as handle:
            if str(path).endswith(".partial") or not bool(handle.attrs.get("complete", False)):
                errors.append("Episode is incomplete or still has a .partial filename")
            schema_version = handle.attrs.get("schema_version")
            if schema_version not in (LEGACY_SCHEMA_VERSION, LEGACY_RGBD_SCHEMA_VERSION, SCHEMA_VERSION):
                errors.append("Unsupported or missing schema_version")
            timing = TIMING_DESCRIPTION if schema_version == SCHEMA_VERSION else LEGACY_TIMING_DESCRIPTION
            if handle.attrs.get("observation_timing") != timing:
                errors.append("Missing or unsupported observation/action timing convention")
            if handle.attrs.get("action_description") != ACTION_DESCRIPTION:
                errors.append("Missing or unsupported action convention")
            try:
                metadata = json.loads(handle.attrs.get("metadata_json", "{}"))
            except (TypeError, ValueError) as error:
                metadata = {}
                errors.append(f"Invalid metadata JSON: {error}")
            if not isinstance(metadata, dict):
                metadata = {}
                errors.append("metadata_json must encode an object")
            result["metadata"] = metadata
            result["success"] = bool(handle.attrs.get("success", False))
            cameras = LEGACY_CAMERAS if schema_version == LEGACY_SCHEMA_VERSION else CAMERAS
            expected = ["timestamp", "phase", *FIELDS.values(), *[f"observations/images/{c}" for c in cameras]]
            if schema_version in (SCHEMA_VERSION, LEGACY_RGBD_SCHEMA_VERSION):
                depth_enabled = metadata.get("depth_enabled")
                if not isinstance(depth_enabled, bool):
                    errors.append("metadata.depth_enabled must be a boolean")
                if metadata.get("feature_map") != _feature_map(bool(depth_enabled)):
                    errors.append("metadata.feature_map does not match the declared RGB/depth datasets")
                depth_present = "observations/depth" in handle
                if depth_present and depth_enabled is not True:
                    errors.append("Depth datasets are present but metadata.depth_enabled is not true")
                if depth_enabled or depth_present:
                    expected.extend(f"observations/depth/{c}" for c in CAMERAS)
                if "observations/images" in handle and isinstance(handle["observations/images"], h5py.Group):
                    if set(handle["observations/images"]) != set(CAMERAS):
                        errors.append("RGB camera names must be exactly head and wrist")
                if depth_present and isinstance(handle["observations/depth"], h5py.Group):
                    if set(handle["observations/depth"]) != set(CAMERAS):
                        errors.append("Depth camera names must be exactly head and wrist")
            for name in expected:
                if name not in handle or not isinstance(handle[name], h5py.Dataset):
                    errors.append(f"Missing dataset: {name}")
            if "timestamp" not in handle:
                return result
            times = handle["timestamp"][...]
            steps = len(times) if times.ndim else 0
            result["steps"] = steps
            if times.ndim != 1 or steps == 0:
                errors.append("timestamp must be a nonempty vector")
            elif not np.isfinite(times).all() or np.any(times < 0) or np.any(np.diff(times) <= 0):
                errors.append("timestamp must be finite, nonnegative and strictly increasing")
            elif schema_version == SCHEMA_VERSION and metadata.get("sampling_timing") is not None:
                try:
                    hz, timestep, origin = _sampling_grid(metadata)
                    ticks = np.floor(np.arange(steps, dtype=float) / (hz * timestep) + 0.5)
                    scheduled = origin + ticks * timestep
                    if not np.allclose(times, scheduled, rtol=0, atol=1e-8):
                        errors.append("timestamps do not match the declared nearest_physics_tick sampling grid")
                except (TypeError, ValueError) as error:
                    errors.append(f"Invalid sampling metadata: {error}")
            elif "sample_hz" in metadata:
                hz = float(metadata["sample_hz"])
                if not np.isfinite(hz) or hz <= 0:
                    errors.append("sample_hz must be finite and positive")
                elif steps > 1 and not np.allclose(np.diff(times), 1 / hz, rtol=0.01, atol=1e-6):
                    errors.append("timestamp intervals do not match sample_hz (1% tolerance)")
            if int(handle.attrs.get("steps", -1)) != steps:
                errors.append("steps attribute does not match timestamps")
            if metadata.get("steps") != steps:
                errors.append("metadata.steps does not match timestamps")
            if metadata.get("success") != result["success"]:
                errors.append("success metadata disagrees with the success attribute")
            if result["success"] and handle.attrs.get("failure_reason", ""):
                errors.append("Successful episode has a failure reason")
            if not result["success"] and not handle.attrs.get("failure_reason", ""):
                errors.append("Failed episode lacks a failure reason")
            for name in expected:
                if name not in handle or not isinstance(handle[name], h5py.Dataset):
                    continue
                dataset = handle[name]
                if dataset.ndim == 0 or dataset.shape[0] != steps:
                    errors.append(f"Frame count mismatch: {name}")
                    continue
                if name == "phase":
                    if dataset.ndim != 1:
                        errors.append("phase must have one label per observation")
                    elif any(not label for label in dataset.asstr()[...]):
                        errors.append("phase contains empty labels")
                    continue
                if name.startswith("observations/images/"):
                    if dataset.dtype != np.uint8 or dataset.ndim != 4 or dataset.shape[-1] != 3 or min(dataset.shape[1:3]) <= 0:
                        errors.append(f"Expected uint8 (T,H,W,3): {name}")
                    # Reading chunks also detects damaged compressed image data.
                    for start in range(steps):
                        dataset[start]
                    continue
                if name.startswith("observations/depth/"):
                    if dataset.dtype != np.float32 or dataset.ndim != 3 or min(dataset.shape[1:]) <= 0:
                        errors.append(f"Expected float32 (T,H,W): {name}")
                    image_path = name.replace("/depth/", "/images/")
                    if (
                        image_path in handle and isinstance(handle[image_path], h5py.Dataset)
                        and dataset.shape[1:] != handle[image_path].shape[1:3]
                    ):
                        errors.append(f"Depth dimensions do not match RGB: {name}")
                    if dataset.attrs.get("units") != DEPTH_UNITS:
                        errors.append(f"Depth units must be '{DEPTH_UNITS}': {name}")
                    if dataset.attrs.get("convention") != DEPTH_CONVENTION:
                        errors.append(f"Depth convention must be '{DEPTH_CONVENTION}': {name}")
                    for start in range(steps):
                        frame = dataset[start]
                        if not np.isfinite(frame).all() or np.any(frame <= 0):
                            errors.append(f"Depth must contain finite positive distances: {name}")
                            break
                    continue
                field = next((key for key, value in FIELDS.items() if value == name), None)
                if field in SIZES and dataset.shape[1:] != (SIZES[field],):
                    errors.append(f"Wrong shape for {name}: {dataset.shape}")
                elif field in ("qpos", "qvel", "ctrl") and (dataset.ndim != 2 or dataset.shape[1] == 0):
                    errors.append(f"Expected nonempty vectors: {name}")
                for start in range(0, steps, 256):
                    chunk = dataset[start:start + 256]
                    if not np.isfinite(chunk).all():
                        errors.append(f"Non-finite values: {name}")
                        break
                    if field and field.endswith("_pose") and chunk.ndim == 2 and chunk.shape[1] == 7:
                        if not np.allclose(np.linalg.norm(chunk[:, 3:], axis=1), 1, atol=1e-3):
                            errors.append(f"Non-unit quaternions: {name}")
                            break
                    if field in ("action", "gripper_width") and chunk.ndim == 2 and np.any(chunk[:, -1] < -1e-6):
                        errors.append(f"Negative full gripper opening: {name}")
                        break
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        errors.append(f"Cannot validate episode: {error}")
    result["valid"] = not errors
    return result
