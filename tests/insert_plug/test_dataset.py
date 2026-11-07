"""Tests for recording integrity, failure publication and corrupt episodes."""

import hashlib
import json

import h5py
import numpy as np
import pytest

from examples.insert_plug.dataset import EpisodeWriter, read_episode, validate_episode


def record(timestamp=0.0, with_depth=False):
    pose = np.array([0.4, 0.0, 0.02, 1.0, 0.0, 0.0, 0.0])
    frame = {
        "timestamp": timestamp,
        "phase": "approach",
        "images": {
            "head": np.zeros((8, 10, 3), dtype=np.uint8),
            "wrist": np.full((8, 10, 3), 127, dtype=np.uint8),
        },
        "joint_position": np.arange(7) / 10,
        "joint_velocity": np.zeros(7),
        "gripper_width": 0.02,
        "action": np.r_[np.arange(7) / 10, 0.02],
        "qpos": np.zeros(16), "qvel": np.zeros(15), "ctrl": np.zeros(9),
        "peg_pose": pose, "source_pose": pose, "target_pose": pose,
    }
    if with_depth:
        frame["depth"] = {
            "head": np.full((8, 10), 0.6, dtype=np.float32),
            "wrist": np.full((8, 10), 0.2, dtype=np.float32),
        }
        # Far-plane/background values remain finite metric depths.
        frame["depth"]["head"][0, 0] = np.float32(50.0)
    return frame


def test_synchronized_roundtrip_rejects_bad_frame_before_mutation(tmp_path):
    path = tmp_path / "episode.h5"
    with EpisodeWriter(path, {"seed": 7, "sample_hz": 20}) as writer:
        writer.append(record(0.0))
        with pytest.raises(ValueError, match="increase strictly"):
            writer.append(record(0.0))
        malformed = record(0.05)
        malformed["images"]["wrist"] = np.zeros((8, 10, 3), dtype=float)
        with pytest.raises(ValueError, match="uint8"):
            writer.append(malformed)
        assert writer.steps == 1
        writer.append(record(0.05))
        writer.finalize(success=True, final_metrics={"tip": np.array([0.4, 0.0, 0.02]), "error": np.float64(0.0)})
        writer.close()  # The CLI closes again in its finally block.
        writer.close()
    report = validate_episode(path)
    assert report["valid"], report["errors"]
    assert report["success"] and report["steps"] == 2
    json.dumps(report, allow_nan=False)
    assert report["metadata"]["final_metrics"]["tip"] == [0.4, 0.0, 0.02]
    episode = read_episode(path, include_images=False)
    assert episode["data"]["action"].shape == (2, 8)
    assert episode["data"]["phase"].tolist() == ["approach", "approach"]
    assert not any("images/" in key for key in episode["data"])
    assert episode["metadata"]["seed"] == 7
    assert not path.with_name(path.name + ".partial").exists()


def test_interruption_stays_partial_and_failed_rollout_can_finalize(tmp_path):
    interrupted = tmp_path / "interrupted.h5"
    with pytest.raises(RuntimeError, match="interrupted"):
        with EpisodeWriter(interrupted) as writer:
            writer.append(record())
            raise RuntimeError("interrupted")
    assert not interrupted.exists()
    partial = interrupted.with_name(interrupted.name + ".partial")
    assert partial.exists()
    assert not validate_episode(partial)["valid"]
    with pytest.raises(FileExistsError):
        EpisodeWriter(interrupted)
    failed = tmp_path / "failed.h5"
    with EpisodeWriter(failed, {"sample_hz": 20}) as writer:
        writer.append(record())
        writer.finalize(success=False, failure_reason="grasp_slipped")
    report = validate_episode(failed)
    assert report["valid"] and not report["success"]
    assert report["metadata"]["failure_reason"] == "grasp_slipped"


def test_validator_detects_desynchronization_and_numerical_corruption(tmp_path):
    path = tmp_path / "episode.h5"
    with EpisodeWriter(path, {"sample_hz": 20}) as writer:
        writer.append(record(0.0))
        writer.append(record(0.05))
        writer.finalize(success=True)
    with h5py.File(path, "r+") as handle:
        handle["timestamp"][1] = 0.0
        handle["action"][0, 0] = np.nan
        handle["observations/images/wrist"].resize(1, axis=0)
    report = validate_episode(path)
    assert not report["valid"]
    assert any("strictly increasing" in error for error in report["errors"])
    assert any("Non-finite" in error for error in report["errors"])
    assert any("Frame count mismatch" in error for error in report["errors"])


def test_rgbd_roundtrip_stores_metric_depth_and_flat_feature_names(tmp_path):
    path = tmp_path / "rgbd.h5"
    with EpisodeWriter(path, {"sample_hz": 20, "depth_enabled": True}) as writer:
        writer.append(record(0.0, with_depth=True))
        writer.append(record(0.05, with_depth=True))
        writer.finalize(success=True)
    report = validate_episode(path)
    assert report["valid"], report["errors"]
    assert report["metadata"]["depth_enabled"] is True
    assert report["metadata"]["feature_map"] == {
        "observation.images.head": "/observations/images/head",
        "observation.images.wrist": "/observations/images/wrist",
        "observation.depth.head": "/observations/depth/head",
        "observation.depth.wrist": "/observations/depth/wrist",
    }
    loaded = read_episode(path)
    with h5py.File(path, "r") as handle:
        assert handle.attrs["schema_version"] == "insert-plug-v3"
        assert "observations/images/third_person" not in handle
        for camera in ("head", "wrist"):
            depth = handle[f"observations/depth/{camera}"]
            assert depth.shape == (2, 8, 10)
            assert depth.dtype == np.float32
            assert depth.chunks == (1, 8, 10)
            assert depth.attrs["units"] == "m"
            assert depth.attrs["convention"] == "camera_optical_axis_z"
            np.testing.assert_array_equal(loaded["data"][f"observations/depth/{camera}"][0], record(with_depth=True)["depth"][camera])
    state_only = read_episode(path, include_images=False)["data"]
    assert not any(key.startswith(("observations/images/", "observations/depth/")) for key in state_only)
    depth_only = read_episode(path, include_images=False, include_depth=True)["data"]
    assert "observations/depth/head" in depth_only
    assert "observations/images/head" not in depth_only
    rgb_only = read_episode(path, include_depth=False)["data"]
    assert "observations/images/head" in rgb_only
    assert "observations/depth/head" not in rgb_only


@pytest.mark.parametrize("initial_depth", [False, True])
def test_depth_presence_cannot_change_after_first_frame(tmp_path, initial_depth):
    path = tmp_path / "episode.h5"
    with EpisodeWriter(path) as writer:
        writer.append(record(with_depth=initial_depth))
        with pytest.raises(ValueError, match="depth presence"):
            writer.append(record(0.05, with_depth=not initial_depth))
        assert writer.steps == 1
        assert all(dataset.shape[0] == 1 for dataset in writer._datasets.values())
        writer.append(record(0.05, with_depth=initial_depth))
        writer.finalize(success=True)
    report = validate_episode(path)
    assert report["valid"], report["errors"]
    assert report["metadata"]["depth_enabled"] is initial_depth


@pytest.mark.parametrize("declared_depth", [False, True])
def test_declared_depth_presence_is_checked_before_creating_datasets(tmp_path, declared_depth):
    with EpisodeWriter(tmp_path / "episode.h5", {"depth_enabled": declared_depth}) as writer:
        with pytest.raises(ValueError, match="depth presence"):
            writer.append(record(with_depth=not declared_depth))
        assert writer.steps == 0
        assert not writer._datasets
        assert list(writer._file) == []
        writer.append(record(with_depth=declared_depth))
        writer.finalize(success=True)


@pytest.mark.parametrize("invalid", ["missing_camera", "extra_camera", "float64", "shape", "nan", "inf", "zero", "negative"])
def test_bad_depth_is_rejected_without_desynchronizing_any_dataset(tmp_path, invalid):
    path = tmp_path / "episode.h5"
    with EpisodeWriter(path) as writer:
        writer.append(record(with_depth=True))
        malformed = record(0.05, with_depth=True)
        if invalid == "missing_camera":
            del malformed["depth"]["wrist"]
        elif invalid == "extra_camera":
            malformed["depth"]["third_person"] = malformed["depth"]["head"]
        elif invalid == "float64":
            malformed["depth"]["wrist"] = malformed["depth"]["wrist"].astype(np.float64)
        elif invalid == "shape":
            malformed["depth"]["wrist"] = np.ones((8, 9), dtype=np.float32)
        else:
            malformed["depth"]["wrist"][0, 0] = {"nan": np.nan, "inf": np.inf, "zero": 0.0, "negative": -0.1}[invalid]
        with pytest.raises(ValueError, match="depth"):
            writer.append(malformed)
        assert writer.steps == 1
        assert writer._last_timestamp == 0.0
        assert all(dataset.shape[0] == 1 for dataset in writer._datasets.values())
        writer.append(record(0.05, with_depth=True))
        writer.finalize(success=True)
    assert validate_episode(path)["valid"]


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("units", "Depth units"),
        ("convention", "Depth convention"),
        ("count", "Frame count mismatch"),
        ("missing_camera", "Missing dataset: observations/depth/wrist"),
        ("float64", "Expected float32"),
        ("dimensions", "Depth dimensions do not match RGB"),
        ("nan", "finite positive"),
        ("zero", "finite positive"),
        ("depth_enabled", "metadata.depth_enabled"),
        ("feature_map", "metadata.feature_map"),
    ],
)
def test_validator_detects_depth_schema_and_value_corruption(tmp_path, corruption, message):
    path = tmp_path / "episode.h5"
    with EpisodeWriter(path) as writer:
        writer.append(record(with_depth=True))
        writer.append(record(0.05, with_depth=True))
        writer.finalize(success=True)
    with h5py.File(path, "r+") as handle:
        depth = handle["observations/depth/wrist"]
        if corruption in ("units", "convention"):
            depth.attrs[corruption] = "incorrect"
        elif corruption == "count":
            depth.resize(1, axis=0)
        elif corruption == "missing_camera":
            del handle["observations/depth/wrist"]
        elif corruption in ("float64", "dimensions"):
            data = depth[...].astype(np.float64) if corruption == "float64" else depth[..., :-1]
            attrs = dict(depth.attrs)
            del handle["observations/depth/wrist"]
            handle.create_dataset("observations/depth/wrist", data=data).attrs.update(attrs)
        elif corruption in ("depth_enabled", "feature_map"):
            metadata = json.loads(handle.attrs["metadata_json"])
            metadata[corruption] = False if corruption == "depth_enabled" else {}
            handle.attrs["metadata_json"] = json.dumps(metadata)
        else:
            depth[0, 0, 0] = np.nan if corruption == "nan" else 0.0
    report = validate_episode(path)
    assert not report["valid"]
    assert any(message in error for error in report["errors"]), report["errors"]


def test_finalize_cannot_relabel_recorded_depth(tmp_path):
    path = tmp_path / "episode.h5"
    with EpisodeWriter(path) as writer:
        writer.append(record(with_depth=True))
        with pytest.raises(ValueError, match="depth_enabled"):
            writer.finalize(success=True, depth_enabled=False)
        with pytest.raises(ValueError, match="feature_map"):
            writer.finalize(success=True, feature_map={})
        assert not path.exists()
        writer.finalize(success=True)
    assert validate_episode(path)["valid"]


def test_legacy_v1_third_person_fixture_reads_and_validates_without_modification(tmp_path):
    path = tmp_path / "legacy_v1.h5"
    # Construct the historical layout independently of the v2 writer.
    frame = record()
    paths = {
        "joint_position": "observations/joint_position", "joint_velocity": "observations/joint_velocity",
        "gripper_width": "observations/gripper_width", "action": "action",
        "qpos": "simulation/qpos", "qvel": "simulation/qvel", "ctrl": "simulation/ctrl",
        "peg_pose": "privileged/peg_pose", "source_pose": "privileged/source_pose", "target_pose": "privileged/target_pose",
    }
    with h5py.File(path, "x") as handle:
        handle.attrs.update(
            schema_version="peg-transfer-v1", complete=True, steps=1, success=True, failure_reason="",
            observation_timing="observation_t_before_action_t; action_t applied until next observation",
            action_description="[fr3_joint1..fr3_joint7 target radians, full gripper opening meters]",
            metadata_json=json.dumps({"steps": 1, "success": True, "failure_reason": "", "sample_hz": 20}),
        )
        handle.create_dataset("timestamp", data=[0.0])
        handle.create_dataset("phase", data=["approach"], dtype=h5py.string_dtype("utf-8"))
        for key, dataset_path in paths.items():
            handle.create_dataset(dataset_path, data=np.atleast_1d(frame[key])[None, :])
        handle.create_dataset("observations/images/third_person", data=frame["images"]["head"][None])
        handle.create_dataset("observations/images/wrist", data=frame["images"]["wrist"][None])
    original_digest = hashlib.sha256(path.read_bytes()).digest()
    report = validate_episode(path)
    assert report["valid"], report["errors"]
    loaded = read_episode(path)
    assert "observations/images/third_person" in loaded["data"]
    assert "observations/images/head" not in loaded["data"]
    assert "depth_enabled" not in loaded["metadata"]
    assert hashlib.sha256(path.read_bytes()).digest() == original_digest


def test_legacy_v2_rgbd_reads_without_rewriting_and_retains_uniform_timing_check(tmp_path):
    path = tmp_path / "legacy_v2.h5"
    with EpisodeWriter(path, {"sample_hz": 20}) as writer:
        writer.append(record(0, with_depth=True))
        writer.append(record(.05, with_depth=True))
        writer.finalize(success=True)
    # v2 used the same RGB-D arrays with the original action-hold convention.
    with h5py.File(path, "r+") as handle:
        handle.attrs["schema_version"] = "peg-transfer-v2"
        handle.attrs["observation_timing"] = "observation_t_before_action_t; action_t applied until next observation"
    digest = hashlib.sha256(path.read_bytes()).digest()
    assert validate_episode(path)["valid"]
    assert read_episode(path)["data"]["observations/depth/head"].shape == (2, 8, 10)
    assert hashlib.sha256(path.read_bytes()).digest() == digest
    with h5py.File(path, "r+") as handle:
        handle["timestamp"][1] = .067
    assert not validate_episode(path)["valid"]


def test_v3_checks_real_nearest_tick_timestamps_and_current_target_semantics(tmp_path):
    path = tmp_path / "quantized_15hz.h5"
    metadata = dict(sample_hz=15, control_hz=50, physics_timestep=.001,
                    sample_origin_s=0., sampling_timing="nearest_physics_tick")
    with EpisodeWriter(path, metadata) as writer:
        for time in (0, .067, .133, .200, .267, .333, .400):
            writer.append(record(time))
        writer.finalize(success=True)
    result = validate_episode(path)
    assert result["valid"], result["errors"]
    with h5py.File(path, "r+") as handle:
        assert "may change before next observation" in handle.attrs["observation_timing"]
        assert "active actuator commands" in handle.attrs["simulation_ctrl_timing"]
        # Accepting arbitrary jitter would hide missed/shifted frames.
        handle["timestamp"][2] = .134
    result = validate_episode(path)
    assert not result["valid"]
    assert any("sampling grid" in message for message in result["errors"])


@pytest.mark.parametrize("changed", [{"sample_hz": 1001}, {"physics_timestep": 0},
                                    {"sample_origin_s": -1}, {"control_hz": 15}])
def test_invalid_nearest_tick_metadata_rejected_before_recording(tmp_path, changed):
    metadata = dict(sample_hz=15, control_hz=50, physics_timestep=.001,
                    sample_origin_s=0., sampling_timing="nearest_physics_tick")
    metadata.update(changed)
    path = tmp_path / "invalid.h5"
    with pytest.raises(ValueError):
        EpisodeWriter(path, metadata)
    assert not path.with_name(path.name + ".partial").exists()
