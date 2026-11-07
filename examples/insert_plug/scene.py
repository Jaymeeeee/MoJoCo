"""Build the FR3, two real socket cavities, free peg and two-camera scene.

Coordinates and sizes are SI. The peg body origin is its shaft's bottom tip;
the head site is its grasp centre. Hole sites mark the top entrance planes.
The generated MJCF has no include dependency and references downloaded robot
meshes by absolute path. It can be opened directly with MuJoCo's viewer.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import yaml
from mujoco_tools.paths import WORKSPACE_ROOT, CONFIG_DIR

PROJECT_ROOT = WORKSPACE_ROOT
DEFAULT_CONFIG = CONFIG_DIR / "insert_plug.yaml"


def load_config(path: str | Path | None = None) -> dict:
    """Read a YAML scene config. Relative asset paths use the project root."""
    source = Path(path).expanduser().resolve() if path is not None else DEFAULT_CONFIG
    with source.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a configuration mapping: {source}")
    return config


def _values(sequence) -> str:
    return " ".join(f"{float(value):.12g}" for value in sequence)


def _asset_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _camera(parent: ET.Element, name: str, options: dict) -> None:
    position = np.asarray(options["position"], dtype=float)
    target = np.asarray(options["look_at"], dtype=float)
    # MuJoCo cameras look along local -Z, with local +Y up.
    backward = position - target
    if np.linalg.norm(backward) < 1e-9:
        raise ValueError(f"Camera {name} position equals its look-at point")
    backward /= np.linalg.norm(backward)
    up = np.array([0.0, 0.0, 1.0])
    if abs(float(backward @ up)) > 0.99:
        up = np.array([0.0, 1.0, 0.0])
    right = np.cross(up, backward)
    right /= np.linalg.norm(right)
    actual_up = np.cross(backward, right)
    ET.SubElement(parent, "camera", name=name, pos=_values(position),
                  xyaxes=_values(np.r_[right, actual_up]),
                  fovy=str(float(options["fovy"])), mode="fixed")


def _validate(config: dict) -> None:
    peg, board = config["peg"], config["board"]
    if not 0 < peg["shaft_radius"] < board["hole_radius"] < peg["head_radius"]:
        raise ValueError("Need 0 < shaft radius < hole radius < head radius")
    if board["bottom_thickness"] <= 0 or board["wall_height"] <= 0:
        raise ValueError("Board support and socket wall thicknesses must be positive")
    if board.get("type") != "stl_holders" and peg["shaft_length"] <= board["wall_height"]:
        raise ValueError("The initial head must be exposed above the board")
    locations = [np.asarray(board[key], dtype=float) for key in ("source_xy", "target_xy")]
    if any(p.shape != (2,) or not np.isfinite(p).all() for p in locations):
        raise ValueError("source_xy and target_xy must be two finite coordinates in metres")
    if board.get("type") == "stl_holders":
        import json
        manifest = json.loads(_asset_path(board["collision_manifest"]).read_text())
        vertices = np.concatenate([part['vertices'] for part in manifest['holder']['parts']])
        half = np.max(np.abs(vertices[:, :2]), axis=0)
        centre = np.asarray(config['table']['centre_xy'])
        extent = np.asarray(config['table']['half_size'][:2])
    else:
        segments = int(board["hole_segments"])
        if segments < 16 or segments % 8:
            raise ValueError("hole_segments must be at least 16 and divisible by 8")
        half = float(board["socket_half_width"])
        if half <= board["hole_radius"]:
            raise ValueError("Socket square must enclose its circular hole")
        centre, extent = np.asarray(board["centre_xy"]), np.asarray(board["half_size_xy"])
    for location in locations:
        if np.any(np.abs(location - centre) + half >= extent):
            raise ValueError("Both holder footprints must lie completely on the supporting table/board")
    if np.all(np.abs(locations[0] - locations[1]) < 2 * half):
        raise ValueError("The two socket squares overlap")
    if config["physics"]["timestep"] <= 0:
        raise ValueError("physics.timestep must be positive")


def _add_gripper(root: ET.Element, config: dict) -> None:
    options = config["gripper"]
    if options.get("type") == "robotiq_2f85":
        from mujoco_tools.robotiq import attach_gripper
        mount = attach_gripper(root, options)
        if config['robot'].get('gravity_compensation', False):
            for body in mount.iter('body'):
                body.set('gravcomp', '1')
        _camera(mount, "wrist", config["cameras"]["wrist"])
        return
    link = root.find(".//body[@name='fr3_link7']")
    if link is None:
        raise ValueError("FR3 model has no fr3_link7 attachment body")
    defaults = root.find("default")
    grip_default = ET.SubElement(defaults, "default", {"class": "task_gripper"})
    ET.SubElement(grip_default, "joint", type="slide", damping="1", armature="0.001",
                  frictionloss="0", limited="true", range=_values(options["joint_range"]))
    ET.SubElement(grip_default, "geom", friction=_values(options["friction"]),
                  condim="4", solref="0.004 1", solimp="0.99 0.999 0.0001",
                  contype="1", conaffinity="1", group="0")
    gravity = "1" if config["robot"].get("gravity_compensation", False) else "0"
    mount = ET.SubElement(link, "body", name="gripper_mount", childclass="task_gripper",
                          pos=_values(options["mount_position_in_link7"]), gravcomp=gravity)
    ET.SubElement(mount, "geom", name="gripper_palm", type="box", size="0.035 0.034 0.015",
                  pos="0 0 0.015", mass="0.20", rgba="0.17 0.19 0.22 1")
    tcp = np.asarray(options["tcp_position_in_mount"], dtype=float)
    ET.SubElement(mount, "site", name="tcp", pos=_values(tcp), size="0.0015",
                  rgba="0 1 0 0", group="4")
    pad_half = np.asarray(options["pad_half_size"], dtype=float)
    for side, sign, joint, actuator in zip(
        ("left", "right"), (1, -1), options["joint_names"], options["actuator_names"]
    ):
        finger = ET.SubElement(mount, "body", name=f"{side}_finger",
                               pos=_values([0, sign * pad_half[1], 0]), gravcomp=gravity)
        ET.SubElement(finger, "joint", name=joint, axis=_values([0, sign, 0]))
        # Stem ends at the pad. Its inward face shares the pad's aperture.
        stem_low, stem_high = 0.030, float(tcp[2] - pad_half[2])
        if stem_high <= stem_low:
            raise ValueError("TCP/pad must sit beyond the palm")
        ET.SubElement(finger, "geom", name=f"{side}_finger_stem", type="box",
                      size=_values([0.007, pad_half[1], (stem_high - stem_low) / 2]),
                      pos=_values([0, 0, (stem_high + stem_low) / 2]), mass="0.025",
                      rgba="0.23 0.25 0.28 1")
        ET.SubElement(finger, "geom", name=f"{side}_finger_pad", type="box",
                      size=_values(pad_half), pos=_values(tcp), mass="0.015",
                      rgba="0.05 0.055 0.06 1")
        ET.SubElement(root.find("actuator"), "position", name=actuator, joint=joint,
                      kp=str(options["kp"]), kv=str(options["kv"]),
                      ctrlrange=_values(options["joint_range"]), ctrllimited="true",
                      forcerange=_values([-options["force_limit"], options["force_limit"]]),
                      forcelimited="true")
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    # Jointed fingers overlap the fixed palm at the rail; this is intentional.
    for side in ("left", "right"):
        ET.SubElement(contact, "exclude", body1="gripper_mount", body2=f"{side}_finger")
    _camera(mount, "wrist", config["cameras"]["wrist"])


def _add_board(root: ET.Element, config: dict) -> tuple[float, float]:
    options = config["board"]
    if options.get("type") == "stl_holders":
        from .objects import add_board
        return add_board(root, config)
    world, assets = root.find("worldbody"), root.find("asset")
    floor = float(options["bottom_z"] + options["bottom_thickness"])
    top = floor + float(options["wall_height"])
    centre = np.asarray(options["centre_xy"], dtype=float)
    extent = np.asarray(options["half_size_xy"], dtype=float)
    board = ET.SubElement(world, "body", name="socket_board")
    common = {"friction": _values(options["friction"]), "condim": "3",
              "solref": "0.004 1", "solimp": "0.99 0.999 0.0001",
              "rgba": "0.24 0.31 0.37 1"}
    ET.SubElement(board, "geom", name="board_floor", type="box",
                  pos=_values([*centre, options["bottom_z"] + options["bottom_thickness"] / 2]),
                  size=_values([*extent, options["bottom_thickness"] / 2]), **common)
    locations = [np.asarray(options[key], dtype=float) for key in ("source_xy", "target_xy")]
    half, radius = float(options["socket_half_width"]), float(options["hole_radius"])
    # Rectangular cells fill the plate around two square socket patches.
    xs = sorted({centre[0] - extent[0], centre[0] + extent[0],
                 *(point[0] + delta for point in locations for delta in (-half, half))})
    ys = sorted({centre[1] - extent[1], centre[1] + extent[1],
                 *(point[1] + delta for point in locations for delta in (-half, half))})
    for ix, (x0, x1) in enumerate(zip(xs[:-1], xs[1:])):
        for iy, (y0, y1) in enumerate(zip(ys[:-1], ys[1:])):
            xy = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
            if any(np.all(np.abs(xy - point) < half - 1e-10) for point in locations):
                continue
            ET.SubElement(board, "geom", name=f"board_cell_{ix}_{iy}", type="box",
                          pos=_values([*xy, (floor + top) / 2]),
                          size=_values([(x1 - x0) / 2, (y1 - y0) / 2, (top - floor) / 2]),
                          **common)
    n = int(options["hole_segments"])
    # A 64-sided hole is within 4.3 micrometres of a 7mm circular hole.
    # Each wedge is its own convex mesh. No whole-board convex collision hull.
    for name, xy, colour in zip(("source", "target"), locations,
                                ("0.70 0.36 0.12 1", "0.10 0.53 0.48 1")):
        socket = ET.SubElement(board, "body", name=f"{name}_socket",
                               pos=_values([*xy, (floor + top) / 2]))
        for i in range(n):
            angles = [2 * math.pi * i / n, 2 * math.pi * (i + 1) / n]
            rays = [np.array([math.cos(angle), math.sin(angle)]) for angle in angles]
            polygon = [radius * rays[0], half / max(abs(rays[0])) * rays[0],
                       half / max(abs(rays[1])) * rays[1], radius * rays[1]]
            vertices = [[*point, z] for z in (-(top - floor) / 2, (top - floor) / 2)
                        for point in polygon]
            mesh_name = f"{name}_annulus_wedge_{i:02d}"
            ET.SubElement(assets, "mesh", name=mesh_name,
                          vertex=_values(np.array(vertices).ravel()))
            properties = dict(common, rgba=colour)
            ET.SubElement(socket, "geom", name=mesh_name, type="mesh", mesh=mesh_name,
                          **properties)
        ET.SubElement(board, "site", name=f"{name}_hole", pos=_values([*xy, top]),
                      size="0.001", rgba="1 1 1 0", group="4")
    return floor, top


def _add_peg(root: ET.Element, config: dict, floor: float) -> list[float]:
    options = config["peg"]
    if options.get("type") == "stl_bolt":
        from .objects import add_peg
        return add_peg(root, config, floor)
    position = [*config["board"]["source_xy"], floor + options["initial_floor_clearance"]]
    peg = ET.SubElement(root.find("worldbody"), "body", name="peg", pos=_values(position))
    ET.SubElement(peg, "freejoint", name="peg_free")
    radius, length = float(options["shaft_radius"]), float(options["shaft_length"])
    head_radius, head_height = float(options["head_radius"]), float(options["head_height"])
    head_z = length + head_height / 2
    volumes = np.array([radius**2 * length, head_radius**2 * head_height])
    masses = float(options["mass"]) * volumes / volumes.sum()
    common = {"friction": _values(options["friction"]), "condim": "4",
              "solref": "0.004 1", "solimp": "0.99 0.999 0.0001",
              "rgba": "0.92 0.65 0.16 1"}
    mesh_path = options.get("visual_mesh")
    if mesh_path:
        common["rgba"] = "0.92 0.65 0.16 0"
        common["group"] = "3"
        mesh_path = _asset_path(mesh_path)
        if not mesh_path.is_file():
            raise FileNotFoundError(mesh_path)
        ET.SubElement(root.find("asset"), "mesh", name="peg_visual_asset", file=str(mesh_path),
                      scale=_values(options["visual_mesh_scale"]))
        ET.SubElement(peg, "geom", name="peg_visual", type="mesh", mesh="peg_visual_asset",
                      pos=_values(options["visual_mesh_position"]),
                      quat=_values(options["visual_mesh_quaternion"]),
                      contype="0", conaffinity="0", mass="0", group="2",
                      rgba="0.92 0.65 0.16 1")
    ET.SubElement(peg, "geom", name="peg_shaft_geom", type="cylinder",
                  size=_values([radius, length / 2]), pos=_values([0, 0, length / 2]),
                  mass=str(masses[0]), **common)
    ET.SubElement(peg, "geom", name="peg_head_geom", type="cylinder",
                  size=_values([head_radius, head_height / 2]), pos=_values([0, 0, head_z]),
                  mass=str(masses[1]), **common)
    ET.SubElement(peg, "site", name="peg_tip", pos="0 0 0", size="0.001",
                  rgba="1 0 0 0", group="4")
    ET.SubElement(peg, "site", name="peg_head", pos=_values([0, 0, head_z]),
                  size="0.001", rgba="0 1 0 0", group="4")
    return position


def build_scene(config: dict, output_path: str | Path) -> Path:
    """Write the complete MJCF, leaving the verified upstream robot unmodified."""
    config = copy.deepcopy(config)
    _validate(config)
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    model_dir = _asset_path(config["robot"]["model_dir"])
    model_file = model_dir / config["robot"].get("model_file", "fr3.xml")
    if not model_file.is_file():
        raise FileNotFoundError(f"Missing FR3 model: {model_file}. Run python scripts/fetch_robot.py")
    root = ET.parse(model_file).getroot()
    root.set("model", "fr3_insert_plug")
    root.find("compiler").set("meshdir", str(model_dir / "assets"))
    option = root.find("option")
    for key, value in config["physics"].items():
        option.set(key, _values(value) if isinstance(value, (list, tuple)) else str(value))
    ET.SubElement(root, "size", nconmax="2000", njmax="6000")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth=str(config["cameras"]["width"]),
                  offheight=str(config["cameras"]["height"]), azimuth="135", elevation="-30")
    ET.SubElement(visual, "quality", shadowsize="2048", offsamples="4")
    ET.SubElement(visual, "map", znear="0.002", zfar="10")
    ET.SubElement(visual, "headlight", ambient="0.2 0.2 0.2", diffuse="0.35 0.35 0.35",
                  specular="0.1 0.1 0.1")
    ET.SubElement(root, "statistic", center="0.35 0 0.25", extent="1.2")
    world = root.find("worldbody")
    robot = world.find("body")
    robot.set("pos", _values(config["robot"]["base_position"]))
    if config["robot"].get("gravity_compensation", False):
        for body in robot.iter("body"):
            body.set("gravcomp", "1")
    _add_gripper(root, config)
    table = config["table"]
    table_z = table["surface_z"] - table["half_size"][2]
    ET.SubElement(world, "geom", name="table", type="box", size=_values(table["half_size"]),
                  pos=_values([*table["centre_xy"], table_z]), rgba="0.63 0.64 0.64 1",
                  friction="0.7 0.005 0.0001")
    ET.SubElement(world, "geom", name="ground", type="plane", size="2 2 0.1",
                  pos="0 0 -0.06", rgba="0.16 0.19 0.22 1")
    ET.SubElement(world, "light", name="key_light", pos="0.3 -0.4 1.5", dir="0 0 -1",
                  directional="true", diffuse="0.5 0.5 0.5", castshadow="true")
    ET.SubElement(world, "light", name="fill_light", pos="0.8 0.7 0.8", dir="-0.5 -0.5 -1",
                  diffuse="0.2 0.2 0.2", castshadow="false")
    floor, _ = _add_board(root, config)
    peg_position = _add_peg(root, config, floor)
    _camera(world, "third_person", config["cameras"]["third_person"])
    old_keyframes = root.find("keyframe")
    if old_keyframes is not None:
        root.remove(old_keyframes)
    home = config["robot"]["home_qpos"]
    if config["gripper"].get("type") == "robotiq_2f85":
        import mujoco
        from mujoco_tools.robotiq import Robotiq2F85
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        data = mujoco.MjData(model)
        for name, value in zip(config["robot"]["arm_joint_names"], home):
            data.joint(name).qpos[0] = value
            data.ctrl[model.actuator(name).id] = value
        Robotiq2F85(model, config["gripper"]).reset(data, config["gripper"]["open_width"])
        qpos, ctrl = data.qpos, data.ctrl
    else:
        opened = [config["gripper"]["open_position"]] * 2
        qpos, ctrl = [*home, *opened, *peg_position, 1, 0, 0, 0], [*home, *opened]
    keyframes = ET.SubElement(root, "keyframe")
    ET.SubElement(keyframes, "key", name="home", qpos=_values(qpos), ctrl=_values(ctrl))
    root.insert(0, ET.Comment(
        "Generated SI baseline. FR3 assets retain upstream Apache-2.0 licence. "
        "See the saved configuration for gripper and object model provenance. Robot gravity compensation is idealized; "
        "the peg is free and gravity-driven. Socket cavities use separate convex wedges."
    ))
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(destination, encoding="utf-8", xml_declaration=True)
    return destination
