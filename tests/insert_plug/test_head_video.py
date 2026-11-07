"""Exercise real MP4 encoding from small disposable head-camera recordings."""

import json

import h5py
import imageio.v2 as imageio
import numpy as np
import pytest

from examples.insert_plug.replay import generate_head_video


def _recording(path, schema="insert-plug-v3", shape=(24, 32), fps=15):
    frames = np.zeros((6, *shape, 3), dtype=np.uint8)
    frames[:3, ..., 0] = 240
    frames[3:, ..., 2] = 240
    camera = "third_person" if schema == "peg-transfer-v1" else "head"
    with h5py.File(path, "w") as handle:
        handle.attrs.update(schema_version=schema, steps=len(frames), metadata_json=json.dumps({"sample_hz": fps}))
        handle.create_dataset(f"observations/images/{camera}", data=frames)
        # A strongly different wrist stream detects accidental side-by-side export.
        wrist = np.zeros_like(frames)
        wrist[..., 1] = 240
        handle.create_dataset("observations/images/wrist", data=wrist)
        handle.create_dataset("timestamp", data=np.arange(len(frames)) / 15)
    return frames


@pytest.mark.parametrize("schema,shape", [
    ("insert-plug-v3", (24, 32)),
    ("peg-transfer-v2", (23, 31)),
    ("peg-transfer-v1", (24, 32)),
])
def test_head_video_preserves_frame_count_rate_shape_and_camera(tmp_path, schema, shape):
    source, output = tmp_path / "episode.h5", tmp_path / "episode.mp4"
    frames = _recording(source, schema=schema, shape=shape)
    assert generate_head_video(source, output) == output
    assert not output.with_name("episode.partial.mp4").exists()
    with imageio.get_reader(output) as reader:
        assert reader.count_frames() == len(frames)
        assert reader.get_meta_data()["fps"] == pytest.approx(15)
        decoded = [reader.get_data(i) for i in range(len(frames))]
    assert all(frame.shape == (*shape, 3) for frame in decoded)
    assert decoded[0][..., 0].mean() > 230
    assert decoded[0][..., 1:].mean() < 5
    assert decoded[-1][..., 2].mean() > 230
    assert decoded[-1][..., :2].mean() < 5


@pytest.mark.parametrize("suffix", [".mp4", ".partial.mp4"])
def test_head_video_does_not_overwrite_existing_output_or_partial(tmp_path, suffix):
    source, output = tmp_path / "episode.h5", tmp_path / "episode.mp4"
    _recording(source)
    existing = tmp_path / f"episode{suffix}"
    existing.write_bytes(b"keep this export")
    with pytest.raises(FileExistsError):
        generate_head_video(source, output)
    assert existing.read_bytes() == b"keep this export"


@pytest.mark.parametrize("fps", [0, -1, float("nan"), None])
def test_head_video_rejects_invalid_sampling_rate_without_leaving_partial(tmp_path, fps):
    source, output = tmp_path / "episode.h5", tmp_path / "episode.mp4"
    _recording(source, fps=fps)
    with pytest.raises(ValueError, match="sample_hz"):
        generate_head_video(source, output)
    assert not output.exists()
    assert not output.with_name("episode.partial.mp4").exists()


def test_head_video_rejects_inconsistent_frame_count(tmp_path):
    source, output = tmp_path / "episode.h5", tmp_path / "episode.mp4"
    _recording(source)
    with h5py.File(source, "r+") as handle:
        handle.attrs["steps"] = 5
    with pytest.raises(ValueError, match="frame count"):
        generate_head_video(source, output)
    assert not output.exists()
