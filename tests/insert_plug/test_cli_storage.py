"""Storage contracts with synthetic short rollouts and the real HDF5 writer."""

import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from examples.insert_plug import cli, expert, replay, scene, simulation
from examples.insert_plug.dataset import validate_episode
from mujoco_tools.paths import WORKSPACE_ROOT


def _frame(timestamp):
    pose = np.array([0.4, 0.0, 0.02, 1.0, 0.0, 0.0, 0.0])
    return dict(
        timestamp=timestamp, phase="storage_fixture",
        images={name: np.zeros((8, 8, 3), dtype=np.uint8) for name in ("head", "wrist")},
        depth={name: np.full((8, 8), 0.5, dtype=np.float32) for name in ("head", "wrist")},
        joint_position=np.zeros(7), joint_velocity=np.zeros(7), gripper_width=0.02,
        action=np.r_[np.zeros(7), 0.02], qpos=np.zeros(16), qvel=np.zeros(15),
        ctrl=np.zeros(8), peg_pose=pose, source_pose=pose, target_pose=pose,
    )


@pytest.fixture
def short_collection(monkeypatch, tmp_path):
    """Replace motion/rendering only; exercise CLI and HDF5 persistence together."""
    class ShortSimulation:
        def __init__(self, path, config, render):
            self.config = config
            self.control_hz = 50
            self.depth_enabled = True

        def camera_calibration(self):
            return {"fixture": True}

        def close(self):
            pass

    class ShortExpert:
        def __init__(self, sim, **kwargs):
            from examples.insert_plug.recording import RecordingSampler
            self.writer = None
            self.plan_directory = kwargs.get("plan_directory")
            self.recording_sampler = RecordingSampler(15, 0.001, 0.0)

        def prepare(self):
            if self.plan_directory is not None:
                Path(self.plan_directory).mkdir()
                (Path(self.plan_directory) / "fixture.json").write_text("{}")

        def run(self):
            if self.writer is not None:
                for timestamp in (0.0, 0.067, 0.133):
                    self.writer.append(_frame(timestamp))
            # This is a persistence fixture, never a claimed successful robot task.
            return dict(
                success=False, failure_reason="synthetic_storage_test", steps=3,
                metrics={"success": False}, phases=[{"phase": "storage_fixture"}],
                planning=[{"checked": True}], execution_checks={"substeps_checked": 60},
                trajectory_files={},
            )

    def build_scene(config, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('<mujoco model="storage_fixture"/>')
        return path

    previews = []

    def preview(source, output):
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic preview path fixture")
        previews.append((Path(source), path))
        return path

    def head_video(source, output):
        assert Path(source).is_file()
        path = Path(output)
        path.write_bytes(b"synthetic head video path fixture")
        return path

    monkeypatch.setattr(simulation, "Simulation", ShortSimulation)
    monkeypatch.setattr(expert, "Expert", ShortExpert)
    monkeypatch.setattr(scene, "build_scene", build_scene)
    monkeypatch.setattr(replay, "generate_preview", preview)
    monkeypatch.setattr(replay, "generate_head_video", head_video)
    monkeypatch.setattr(cli, "OUTPUTS_DIR", tmp_path / "outputs")
    monkeypatch.setattr(cli, "_configure_gl", lambda backend: backend)
    args = SimpleNamespace(
        command="collect", config=None, output=str(tmp_path / "dataset" / "run_fixture"),
        backend="osmesa", width=None, control_hz=None, sample_hz=None, episodes=1, seed=7,
        randomize_mm=0.0, no_images=False, realtime=False, quiet=True, viewer=False,
        video=False,
    )
    return args, previews


@pytest.mark.parametrize("video", [False, True])
def test_collection_retains_four_files_and_embeds_execution_result(short_collection, video):
    args, previews = short_collection
    args.video = video
    assert cli.collect(args) == 2  # Deliberately failed synthetic task, valid recorded data.
    run = Path(args.output)
    assert {path.name for path in run.iterdir()} == {'episode_000007'}
    output = run / 'episode_000007'
    assert {path.name for path in output.iterdir()} == {
        "episode_000007.h5", "episode_000007.yaml", "episode_000007.xml", "episode_000007.mp4",
    }
    path = output / "episode_000007.h5"
    report = validate_episode(path)
    assert report["valid"], report["errors"]
    with h5py.File(path, "r") as handle:
        metadata = json.loads(handle.attrs["metadata_json"])
        assert metadata['model_path'] == 'episode_000007.xml'
        result = metadata["execution_result"]
        assert result["failure_reason"] == "synthetic_storage_test"
        assert result["phases"] == [{"phase": "storage_fixture"}]
        assert result["planning"] == [{"checked": True}]
        assert result["execution_checks"] == {"substeps_checked": 60}
        assert handle["observations/depth/head"].shape == (3, 8, 8)
    if video:
        expected = cli.OUTPUTS_DIR / "insert_plug" / "previews" / run.name / "preview.mp4"
        assert previews == [(path, expected)]
        assert not expected.is_relative_to(output)
    else:
        assert not previews


@pytest.mark.parametrize("suffix", [".h5", ".h5.partial", ".yaml", ".xml", ".mp4", ".partial.mp4"])
def test_existing_episode_component_is_never_overwritten(short_collection, suffix):
    args, _ = short_collection
    output = Path(args.output)
    output.mkdir(parents=True)
    original = output / f"episode_000007{suffix}"
    original.write_bytes(b"existing recording component")
    with pytest.raises(FileExistsError):
        cli.collect(args)
    assert original.read_bytes() == b"existing recording component"
    assert list(output.iterdir()) == [original]


def test_existing_episode_directory_is_never_reused(short_collection):
    args, _ = short_collection
    output = Path(args.output) / 'episode_000007'
    output.mkdir(parents=True)
    original = output / 'episode_000007.yaml'
    original.write_bytes(b'existing recording configuration')
    with pytest.raises(FileExistsError):
        cli.collect(args)
    assert original.read_bytes() == b'existing recording configuration'
    assert list(output.iterdir()) == [original]


def test_default_scene_and_dataset_locations_are_separate(monkeypatch):
    expected = WORKSPACE_ROOT / "outputs" / "datasets" / "insert_plug"
    for command in ("demo", "collect"):
        args = SimpleNamespace(command=command, no_images=False)
        assert cli._default_output(args).parent == expected
    assert cli._default_output(SimpleNamespace(command="demo", no_images=True)).parent == (
        WORKSPACE_ROOT / "outputs" / "_checks" / "insert_plug"
    )
    destinations = []
    monkeypatch.setattr(scene, "load_config", lambda path: {})
    monkeypatch.setattr(scene, "build_scene", lambda config, output: destinations.append(Path(output)))
    cli.main(["build"])
    assert destinations == [WORKSPACE_ROOT / "examples" / "insert_plug" / "scene.xml"]


def test_physics_only_run_has_no_recording_or_extra_artifacts(short_collection):
    args, previews = short_collection
    args.no_images = True
    assert cli.collect(args) == 2
    assert {path.name for path in (Path(args.output) / 'episode_000007').iterdir()} == {
        "episode_000007.yaml", "episode_000007.xml",
    }
    assert not previews
