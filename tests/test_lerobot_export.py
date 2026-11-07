"""Official API export tests run in the isolated lerobot environment."""
import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from mujoco_tools.lerobot_export import (
    ACTION_TIMING, ALIGNED_ACTION_TIMING, SourceEpisode, discover_episodes,
    _validate_depth_payload, export_dataset, inspect_source, validate_source_batch,
)


def source_episode(root, position, index, frames=4, *, aligned=False):
    stem = f"episode_{index:06d}"
    folder = root / f"pos{position}" / stem
    folder.mkdir(parents=True)
    for suffix in ("xml", "yaml", "mp4"):
        (folder / f"{stem}.{suffix}").write_bytes(b"source discovery fixture")
    path = folder / f"{stem}.h5"
    metadata = dict(sample_hz=10, control_hz=50, physics_timestep=.001,
                    action_timing=ACTION_TIMING, camera_calibration={},
                    config={"board": {"source_xy": [.45, -.07], "target_xy": [.45, .07]}})
    if aligned:
        metadata.update(control_hz=10, action_timing=ALIGNED_ACTION_TIMING,
                        action_hold_duration_s=.1, sample_origin_s=0)
    with h5py.File(path, "w") as f:
        f.attrs.update(complete=True, success=True, schema_version="insert-plug-v3",
                       metadata_json=json.dumps(metadata))
        f["timestamp"] = np.arange(frames) / 10
        f["phase"] = np.array(["move"] * frames, dtype=h5py.string_dtype())
        actions = np.arange(frames * 8, dtype=np.float64).reshape(frames, 8) / 100
        f["action"] = actions.astype(np.float32).astype(np.float64) if aligned else actions
        f["observations/joint_position"] = np.arange(frames * 7, dtype=np.float64).reshape(frames, 7) / 100
        f["observations/gripper_width"] = np.full((frames, 1), .03)
        for camera_index, camera in enumerate(("head", "wrist")):
            rgb = np.zeros((frames, 16, 16, 3), dtype=np.uint8)
            rgb[..., camera_index] = 128
            f[f"observations/images/{camera}"] = rgb
            depth = (np.arange(frames * 16 * 16, dtype=np.float32).reshape(frames, 16, 16)
                     / np.float32(937) + np.float32(.173 + camera_index))
            d = f.create_dataset(f"observations/depth/{camera}", data=depth)
            d.attrs["units"] = "m"
    return SourceEpisode(path, position, index)


def test_discovery_uses_only_complete_explicit_position_groups(tmp_path):
    source = source_episode(tmp_path, 0, 7)
    (tmp_path / "test").mkdir()
    assert discover_episodes(tmp_path, 1, 1) == [source]
    with pytest.raises(ValueError, match="expected 2"):
        discover_episodes(tmp_path, 1, 2)
    (source.path.parent / "unpublished.partial").touch()
    with pytest.raises(ValueError, match="four completed"):
        discover_episodes(tmp_path, 1, 1)


def test_unsuccessful_source_is_rejected(tmp_path):
    source = source_episode(tmp_path, 0, 7)
    with h5py.File(source.path, "r+") as f:
        f.attrs["success"] = False
    with pytest.raises(ValueError, match="unsuccessful"):
        inspect_source(source)


@pytest.mark.parametrize('aligned', [False, True])
def test_source_preserves_legacy_and_aligned_action_timing(tmp_path, aligned):
    source = source_episode(tmp_path, 0, 0, aligned=aligned)
    description = inspect_source(source)
    assert description['metadata']['action_timing'] == (ALIGNED_ACTION_TIMING if aligned else ACTION_TIMING)
    assert ('action_hold_duration_s' in description['metadata']) == aligned


@pytest.mark.parametrize('changes', [
    {'control_hz': 50}, {'action_hold_duration_s': .02},
    {'action_hold_duration_s': None}, {'sample_origin_s': .001}, {'sample_origin_s': None},
])
def test_inconsistent_aligned_action_timing_is_rejected(tmp_path, changes):
    source = source_episode(tmp_path, 0, 0, aligned=True)
    with h5py.File(source.path, 'r+') as handle:
        metadata = json.loads(handle.attrs['metadata_json'])
        metadata.update(changes)
        handle.attrs['metadata_json'] = json.dumps(metadata)
    with pytest.raises(ValueError, match='held-until-next-observation'):
        inspect_source(source)


def test_mixed_action_contracts_are_rejected_before_export(tmp_path):
    sources = [source_episode(tmp_path, 0, 0), source_episode(tmp_path, 1, 1, aligned=True)]
    with pytest.raises(ValueError, match='cannot mix contracts'):
        validate_source_batch([inspect_source(source) for source in sources])


def test_aligned_action_labels_cannot_be_changed_by_float32_conversion(tmp_path):
    source = source_episode(tmp_path, 0, 0, aligned=True)
    with h5py.File(source.path, 'r+') as handle:
        handle['action'][1, 0] = .123456789012345
    with pytest.raises(ValueError, match='preserve executed values exactly'):
        inspect_source(source)


def test_aligned_observations_cannot_drift_inside_their_control_intervals(tmp_path):
    source = source_episode(tmp_path, 0, 0, aligned=True)
    with h5py.File(source.path, 'r+') as handle:
        handle['timestamp'][1] += .0001
    with pytest.raises(ValueError, match='held-until-next-observation'):
        inspect_source(source)


@pytest.mark.skipif(importlib.util.find_spec("lerobot") is None, reason="requires conda env lerobot")
@pytest.mark.parametrize('aligned', [False, True])
def test_official_export_preserves_every_depth_pixel_and_episode_mapping(tmp_path, monkeypatch, aligned):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    sources = [source_episode(tmp_path / "raw", 0, 7, aligned=aligned),
               source_episode(tmp_path / "raw", 1, 12, aligned=aligned)]
    output = tmp_path / "dataset"
    result = export_dataset(sources, output, source_root=tmp_path / "raw")
    assert result["episodes"] == 2 and result["frames"] == 8
    assert result["all_depth_pixels_bitwise_equal"]
    assert result['scalar_frames_checked'] == 8
    assert result['all_actions_equal_source_float32']
    assert result['all_states_equal_source_float32']
    assert result['all_source_timestamps_exact'] and result['all_source_phases_equal']
    assert result['action_float32_conversion_lossless'] == aligned
    assert result["published_reader_samples_checked"] == 3
    assert not (output / "images").exists()
    assert not output.with_name("dataset.partial").exists()
    manifest = json.loads((output / 'meta' / 'export_manifest.json').read_text())
    for record in [manifest, *manifest['episodes']]:
        assert record['control_hz'] == (10 if aligned else 50)
        assert record['action_timing'] == (ALIGNED_ACTION_TIMING if aligned else ACTION_TIMING)
        assert ('action_hold_duration_s' in record) == aligned
        if aligned:
            assert record['action_hold_duration_s'] == .1
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    reader = LeRobotDataset("local/insert_plug", root=output, video_backend="pyav", depth_output_unit="m")
    assert int(reader[0]["source.episode_index"]) == 7
    assert int(reader[4]["source.episode_index"]) == 12
    assert reader[7]["observation.depth.head"].shape == (1, 16, 16)
    with pytest.raises(FileExistsError):
        export_dataset(sources, output, source_root=tmp_path / "raw")
    if aligned:
        # Frame 1 is outside the official reader's first/middle/last sample.
        # Full payload verification must still catch a changed training label.
        import pyarrow as pa
        import pyarrow.parquet as pq
        path = next((output / 'data').rglob('*.parquet'))
        table = pq.read_table(path)
        rows = table.to_pylist()
        rows[1]['action'][0] += 1
        pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
        with pytest.raises(AssertionError, match='action differs at episode 0, frame 1'):
            _validate_depth_payload(output, manifest, sources)
