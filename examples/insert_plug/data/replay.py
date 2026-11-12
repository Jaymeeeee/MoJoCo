"""Stream single-camera or side-by-side videos of recorded RGB observations.

This previews recorded images, not a physics re-simulation. No OpenGL context or
MuJoCo import is needed. Simulation state is stored separately in each HDF5 file.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _head_stream(handle: h5py.File) -> h5py.Dataset:
    """Resolve the logical head view, including explicitly identified v1 files."""
    schema = handle.attrs.get("schema_version")
    if isinstance(schema, bytes):
        schema = schema.decode("utf-8")
    if schema == "peg-transfer-v1":
        path = "observations/images/third_person"
    elif schema in ("peg-transfer-v2", "insert-plug-v3"):
        path = "observations/images/head"
    elif schema is None:
        # Historical image-only previews did not always declare a schema. Use
        # path detection only for these files; a declared v2 must contain head.
        path = "observations/images/head"
        if path not in handle:
            path = "observations/images/third_person"
    else:
        raise ValueError(f"Unsupported recording schema: {schema!r}")
    if path not in handle:
        raise ValueError(f"Recording is missing the head camera stream: {path}")
    return handle[path]


def generate_head_video(h5_path: str | Path, output_mp4: str | Path) -> Path:
    """Export every recorded head RGB frame at the declared sampling rate.

    The MP4 is a lossy viewing copy, with no labels, resizing, padding or wrist
    view. The HDF5 retains the original RGB values. Odd image dimensions use
    yuv444p because yuv420p requires an even width and height.
    """
    import os

    output = Path(output_mp4).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.stem + ".partial.mp4")
    with h5py.File(h5_path, "r") as handle:
        head = _head_stream(handle)
        if head.ndim != 4 or head.shape[-1] != 3 or head.dtype != np.uint8:
            raise ValueError("Head stream must be uint8 (T,H,W,3)")
        if min(head.shape[:3]) <= 0:
            raise ValueError("Head stream must contain nonempty RGB frames")
        if int(handle.attrs.get("steps", len(head))) != len(head):
            raise ValueError("Head frame count must match recorded steps")
        timestamps = handle.get("timestamp")
        if timestamps is not None and (timestamps.ndim != 1 or len(timestamps) != len(head)):
            raise ValueError("timestamp must have one value per head frame")
        metadata = json.loads(handle.attrs.get("metadata_json", "{}"))
        try:
            fps = float(metadata["sample_hz"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Head video requires metadata.sample_hz") from error
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("metadata.sample_hz must be finite and positive")
        # Reserve the temporary name exclusively before handing it to ffmpeg.
        # A pre-existing partial belongs to another/interrupted export.
        partial.touch(exist_ok=False)
        try:
            pixel_format = "yuv420p" if all(n % 2 == 0 for n in head.shape[1:3]) else "yuv444p"
            with imageio.get_writer(
                partial, fps=fps, codec="libx264", quality=8,
                macro_block_size=1, pixelformat=pixel_format,
            ) as writer:
                for index in range(len(head)):
                    writer.append_data(head[index])
            with imageio.get_reader(partial) as reader:
                if reader.count_frames() != len(head):
                    raise ValueError("Encoded video frame count does not match the HDF5 head stream")
            os.link(partial, output)  # Atomic publication, never replace another export.
            partial.unlink()
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
    return output


def generate_preview(
    h5_path: str | Path,
    output_mp4: str | Path,
    fps: float | None = None,
    *,
    annotated: bool = True,
) -> Path:
    """Stream recorded cameras into an MP4 at the recording's sampling rate.

    The head (third-person) camera is on the left and wrist camera on the right. A
    separate header labels the views, recorded simulation time and expert phase;
    ``annotated=False`` omits it. Neither view is cropped, scaled or painted over.
    """
    output = Path(output_mp4).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.stem + ".partial.mp4")
    if partial.exists():
        raise FileExistsError(partial)
    with h5py.File(h5_path, "r") as handle:
        head = _head_stream(handle)
        wrist = handle["observations/images/wrist"]
        if len(head) == 0 or len(head) != len(wrist):
            raise ValueError("Camera streams must be nonempty and have the same frame count")
        for stream in (head, wrist):
            if stream.ndim != 4 or stream.shape[-1] != 3 or stream.dtype != np.uint8:
                raise ValueError("Camera streams must be uint8 (T,H,W,3)")
        if fps is None:
            metadata = json.loads(handle.attrs.get("metadata_json", "{}"))
            fps = metadata.get("sample_hz")
            if fps is None:
                times = handle["timestamp"][...]
                if len(times) < 2 or np.any(np.diff(times) <= 0):
                    raise ValueError("Specify fps for a single frame or absent sampling metadata")
                fps = 1 / float(np.median(np.diff(times)))
        fps = float(fps)
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be finite and positive")
        width = head.shape[2] + wrist.shape[2]
        header_height = 0
        timestamps = phases = None
        if annotated:
            timestamps = handle.get("timestamp")
            phases = handle.get("phase")
            for name, values in (("timestamp", timestamps), ("phase", phases)):
                if values is not None and (values.ndim != 1 or len(values) != len(head)):
                    raise ValueError(f"{name} must have one value per camera frame")
            font_size = max(12, min(24, width // 50))
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", font_size)
            except OSError:
                font = ImageFont.load_default()
            line_height = font_size + 10
            header_height = 2 * line_height + 8
            header = Image.new("RGB", (width, header_height), "black")
            draw = ImageDraw.Draw(header)
            draw.text((10, 6), "Head (third-person)", font=font, fill="white", anchor="lt")
            draw.text((head.shape[2] + 10, 6), "Wrist", font=font, fill="white", anchor="lt")
        height = header_height + max(head.shape[1], wrist.shape[1])
        # Padding to an even shape preserves every source pixel for yuv420p.
        height += height % 2
        width += width % 2
        try:
            with imageio.get_writer(partial, fps=fps, codec="libx264", quality=8, macro_block_size=1) as writer:
                for index in range(len(head)):
                    frame = np.zeros((height, width, 3), dtype=np.uint8)
                    if annotated:
                        time_label = (
                            f"{float(timestamps[index]):.2f} s" if timestamps is not None else "unavailable"
                        )
                        phase = phases[index] if phases is not None else "unavailable"
                        if isinstance(phase, bytes):
                            phase = phase.decode("utf-8")
                        phase_label = str(phase).replace("_", " ").capitalize()
                        current_header = header.copy()
                        ImageDraw.Draw(current_header).text(
                            (10, line_height + 6),
                            f"Simulation time: {time_label}   |   Phase: {phase_label}",
                            font=font,
                            fill=(210, 220, 230),
                            anchor="lt",
                        )
                        frame[:header_height, :header.width] = np.asarray(current_header)
                    frame[header_height:header_height + head.shape[1], :head.shape[2]] = head[index]
                    offset = head.shape[2]
                    frame[header_height:header_height + wrist.shape[1], offset:offset + wrist.shape[2]] = wrist[index]
                    writer.append_data(frame)
            # Do not overwrite an existing preview if another process created it.
            import os

            os.link(partial, output)
            partial.unlink()
        except Exception:
            partial.unlink(missing_ok=True)
            raise
    return output
