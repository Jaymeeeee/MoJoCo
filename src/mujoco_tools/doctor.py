"""Isolated diagnostics for MuJoCo physics and rendering backends.

Importing this module never imports MuJoCo or modifies MUJOCO_GL. Each check
runs in a child Python process, since OpenGL context failures can abort the
process or leave libraries initialized for the wrong backend.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


_CHECK = r'''
import json
import sys
import numpy as np
import mujoco

mode = sys.argv[1]
xml = """<mujoco><option timestep="0.002"/><visual><global offwidth="96" offheight="64"/></visual>
<worldbody><light pos="0 0 3"/><geom type="plane" size="2 2 .1" rgba=".5 .5 .5 1"/>
<camera name="diagnostic" pos="0 -2 1" xyaxes="1 0 0 0 .4472136 .8944272"/>
<body pos="0 0 .3"><freejoint/><geom type="sphere" size=".05" mass=".1" rgba="1 .1 .1 1"/></body>
</worldbody></mujoco>"""
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
for _ in range(250):
    mujoco.mj_step(model, data)
if not np.isfinite(data.qpos).all() or not (0.035 < data.qpos[2] < 0.065):
    raise RuntimeError("Physics check failed: sphere did not settle on the ground")
result = {"ok": True, "version": mujoco.__version__, "simulated_seconds": float(data.time),
          "sphere_height_m": float(data.qpos[2])}
if mode == "render":
    renderer = mujoco.Renderer(model, height=64, width=96)
    try:
        renderer.update_scene(data, camera="diagnostic")
        pixels = renderer.render()
        if pixels.shape != (64, 96, 3) or pixels.dtype != np.uint8 or pixels.std() < 1:
            raise RuntimeError("Renderer returned an invalid or uniform image")
        result.update(image_shape=list(pixels.shape), image_std=float(pixels.std()))
        try:
            from OpenGL import GL
            for key, enum in (("gl_renderer", GL.GL_RENDERER), ("gl_vendor", GL.GL_VENDOR), ("gl_version", GL.GL_VERSION)):
                value = GL.glGetString(enum)
                result[key] = value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)
        except Exception as exc:
            result["renderer_identification_error"] = str(exc)
    finally:
        renderer.close()
print("PEG_DOCTOR_JSON=" + json.dumps(result))
'''


def _run_check(mode: str, backend: str, timeout: float) -> dict:
    env = os.environ.copy()
    env["MUJOCO_GL"] = backend
    if backend == "osmesa":
        env["PYOPENGL_PLATFORM"] = "osmesa"
    elif backend == "egl":
        env["PYOPENGL_PLATFORM"] = "egl"
    else:
        # Avoid inheriting PyOpenGL's incompatible EGL/OSMesa selection.
        env.pop("PYOPENGL_PLATFORM", None)
    start = time.monotonic()
    try:
        child = subprocess.run(
            [sys.executable, "-c", _CHECK, mode],
            env=env, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "backend": backend, "error": f"Check timed out after {timeout:g} seconds"}
    except OSError as exc:
        return {"ok": False, "backend": backend, "error": str(exc)}
    marker = "PEG_DOCTOR_JSON="
    payload = next((line[len(marker):] for line in reversed(child.stdout.splitlines()) if line.startswith(marker)), None)
    if child.returncode == 0 and payload:
        try:
            result = json.loads(payload)
        except json.JSONDecodeError:
            result = {"ok": False, "error": "Child returned malformed diagnostic JSON"}
    else:
        result = {"ok": False, "error": (child.stderr or child.stdout or "Child process returned no diagnostic result").strip()[-3500:]}
    result.update(backend=backend, returncode=child.returncode, elapsed_seconds=round(time.monotonic() - start, 3))
    return result


def _nvidia_status(timeout: float) -> dict:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return {"ok": False, "error": "nvidia-smi is not installed or not on PATH"}
    try:
        child = subprocess.run(
            [executable, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=min(timeout, 10),
        )
        return {"ok": child.returncode == 0, "returncode": child.returncode,
                "output": (child.stdout or child.stderr).strip()[-2000:]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": str(exc)}


def run_diagnostics(chosen_backend: str | None = None, timeout: float = 30.0) -> dict:
    """Probe physics plus requested/offscreen renderers without altering drivers.

    With no choice, EGL and OSMesa are checked; GLFW is also checked when a
    graphical display is present. A working EGL backend is preferred for
    offscreen rendering, but its availability alone does not prove GPU use.
    """
    if chosen_backend not in {None, "egl", "osmesa", "glfw"}:
        raise ValueError("chosen_backend must be egl, osmesa, glfw, or None")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    backends = [chosen_backend] if chosen_backend else ["egl", "osmesa"]
    if chosen_backend is None and (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        backends.append("glfw")
    with ThreadPoolExecutor(max_workers=len(backends) + 2) as pool:
        physics_future = pool.submit(_run_check, "physics", "disable", timeout)
        gpu_future = pool.submit(_nvidia_status, timeout)
        render_futures = {backend: pool.submit(_run_check, "render", backend, timeout) for backend in backends}
        physics = physics_future.result()
        nvidia = gpu_future.result()
        renderers = {backend: future.result() for backend, future in render_futures.items()}
    recommended = next((backend for backend in backends if renderers[backend].get("ok")), None)
    notes = []
    if not nvidia["ok"]:
        notes.append("NVIDIA driver access is unavailable in this session. A working OSMesa renderer can still collect images using the CPU.")
    if recommended == "osmesa":
        notes.append("Use MUJOCO_GL=osmesa for headless collection; software rendering is slower than GPU rendering.")
    if recommended == "egl":
        renderer_name = renderers["egl"].get("gl_renderer", "")
        if any(token in renderer_name.lower() for token in ("llvmpipe", "softpipe", "swrast", "swiftshader", "software")):
            notes.append(f"EGL reports a software renderer ({renderer_name}); successful EGL rendering does not imply NVIDIA GPU acceleration.")
        else:
            notes.append("EGL rendering succeeded; consult gl_renderer and gl_vendor for the actual rendering device.")
    if recommended is None:
        notes.append("No tested image renderer succeeded. Physics-only checks may pass, but image collection is not ready.")
    return {
        "ok": bool(physics.get("ok") and recommended),
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "display_present": bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")),
        "physics": physics,
        "renderers": renderers,
        "nvidia_smi": nvidia,
        "recommended_backend": recommended,
        "notes": notes,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("egl", "osmesa", "glfw"))
    parser.add_argument("--timeout", type=float, default=30.0, help="Maximum seconds for each isolated check")
    parser.add_argument("--output", type=Path, help="Also write diagnostic JSON to this file")
    args = parser.parse_args(argv)
    result = run_diagnostics(args.backend, args.timeout)
    output = json.dumps(result, indent=2, ensure_ascii=False)
    print(output)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
