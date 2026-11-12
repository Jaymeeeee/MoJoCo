"""Attach and control the pinned Menagerie Robotiq 2F-85 mechanism.

The public action is a full aperture in metres. Menagerie's single tendon
actuator instead accepts 0..255 and drives eight coupled hinge joints. Width
conversion uses the actual linkage geometry and inner contact faces, while
observations measure the deformed mechanism directly from two pad sites.
"""

from __future__ import annotations

import copy
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import brentq

from mujoco_tools.paths import WORKSPACE_ROOT

PREFIX = "robotiq_"
PAD_CONTACT_POSITION = (0.0, -0.0066, 0.028125)
JOINT_NAMES = tuple(
    f"{PREFIX}{side}_{part}_joint"
    for side in ("right", "left")
    for part in ("driver", "coupler", "spring_link", "follower")
)


def _values(values) -> str:
    return " ".join(f"{float(value):.12g}" for value in values)


def _rotation(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def _linkage(driver: float, pad_position=PAD_CONTACT_POSITION) -> tuple[np.ndarray, np.ndarray]:
    """Exact free linkage with its coupler against the native upper stop.

    Coordinates are the right mechanism's YZ plane in upstream ``base``.
    Constants are its unmodified body offsets, follower joint pivot, and
    connect-constraint anchor. A two-link triangle closes the mechanism;
    unlike setting every hinge to the same angle, this satisfies both loops.
    The pads tilt slightly as the opening changes, as in the source model.
    """
    spring_origin = np.array([0.0132, 0.0609])
    driver_origin = np.array([0.0306011, 0.054904])
    follower_pivot = np.array([-0.018, 0.0065])
    spring_arm = np.array([0.037, 0.044])
    coupler_anchor_from_driver = np.array([0.0375989, 0.043496])
    follower_origin = driver_origin + _rotation(driver) @ coupler_anchor_from_driver
    diagonal = follower_origin - spring_origin
    follower_arm = -follower_pivot
    cosine = (diagonal @ diagonal + spring_arm @ spring_arm - follower_arm @ follower_arm)
    cosine /= 2 * np.linalg.norm(diagonal) * np.linalg.norm(spring_arm)
    spring = (np.arctan2(diagonal[1], diagonal[0]) + np.arccos(np.clip(cosine, -1, 1))
              - np.arctan2(spring_arm[1], spring_arm[0]))
    remaining = diagonal - _rotation(spring) @ spring_arm
    pad_angle = (np.arctan2(remaining[1], remaining[0])
                 - np.arctan2(follower_arm[1], follower_arm[0]))
    qpos = np.array([driver, 0.0, spring, pad_angle - spring] * 2)
    pad_offset = np.array([-0.0189, 0.01352]) + np.asarray(pad_position, dtype=float)[1:]
    face = follower_origin + _rotation(pad_angle) @ pad_offset
    # base_mount is +7 mm and base is another +3.8 mm from the flange mount.
    centre = np.array([0.0, 0.0, 0.0108 + face[1]])
    return qpos, np.r_[2 * face[0], centre]


def _configuration(width: float, pad_position=PAD_CONTACT_POSITION) -> tuple[np.ndarray, np.ndarray]:
    maximum = float(_linkage(0, pad_position)[1][0])
    minimum = float(_linkage(0.8, pad_position)[1][0])
    if not minimum <= width <= maximum:
        raise ValueError(f"Robotiq aperture {width} m outside linkage range [{minimum}, {maximum}]")
    driver = brentq(lambda angle: _linkage(angle, pad_position)[1][0] - width, 0, 0.8,
                    xtol=1e-13)
    qpos, geometry = _linkage(driver, pad_position)
    return qpos, geometry[1:]


def attach_gripper(root: ET.Element, options: dict) -> ET.Element:
    """Merge the upstream mechanism and attach it at the FR3 flange.

    Source meshes, inertias, colliders, native friction, constraints, tendon,
    actuator and visual materials are preserved. Only names and mesh paths
    change; measurement sites and a fixed, width-calibrated TCP are added.
    Returns the mount body so callers can attach a wrist camera in that frame.
    """
    directory = Path(options.get("model_dir", "assets/grippers/robotiq_2f85")).expanduser()
    directory = directory.resolve() if directory.is_absolute() else (WORKSPACE_ROOT / directory).resolve()
    filename = directory / options.get("model_file", "2f85.xml")
    if not filename.is_file():
        raise FileNotFoundError(f"Missing Robotiq model: {filename}. Run python scripts/setup_assets.py")
    gripper = ET.parse(filename).getroot()
    link = root.find(".//body[@name='fr3_link7']")
    if link is None:
        raise ValueError("FR3 model has no fr3_link7 attachment body")
    if root.find(".//body[@name='gripper_mount']") is not None:
        raise ValueError("A gripper is already attached")

    # Material/default names such as black, base, visual and collision overlap
    # the FR3 asset. Explicitly name implicit mesh assets before namespacing.
    mesh_directory = directory / gripper.find("compiler").get("meshdir", ".")
    for mesh in gripper.findall("./asset/mesh"):
        mesh.set("name", mesh.get("name", Path(mesh.get("file")).stem))
        mesh.set("file", str((mesh_directory / mesh.get("file")).resolve()))
    reference_attributes = {
        "name", "class", "childclass", "mesh", "material", "joint", "joint1", "joint2",
        "body1", "body2", "tendon", "site", "site1", "site2", "geom1", "geom2",
    }
    for element in gripper.iter():
        for attribute in reference_attributes.intersection(element.attrib):
            element.set(attribute, PREFIX + element.get(attribute))
    for section in ("asset", "default", "contact", "equality", "tendon", "actuator"):
        source = gripper.find(section)
        if source is None:
            continue
        destination = root.find(section)
        if destination is None:
            destination = ET.SubElement(root, section)
        destination.extend(copy.deepcopy(list(source)))
    mount = ET.SubElement(link, "body", name="gripper_mount", childclass=PREFIX + "2f85",
                          pos=_values(options.get("mount_position_in_link7", [0, 0, 0.107])),
                          quat=_values(options.get("mount_quaternion_in_link7", [1, 0, 0, 0])))
    mount.extend(copy.deepcopy(list(gripper.find("worldbody"))))
    if options.get("gravity_compensation", False):
        for body in mount.iter("body"):
            body.set("gravcomp", "1")
    pad_position = options.get("pad_contact_position", PAD_CONTACT_POSITION)
    for side in ("left", "right"):
        pad = mount.find(f".//body[@name='{PREFIX}{side}_pad']")
        ET.SubElement(pad, "site", name=f"{PREFIX}{side}_pad_inner", pos=_values(pad_position),
                      size="0.001", rgba="0 1 0 0", group="4")
    _, tcp = _configuration(float(options.get("tcp_reference_width", 0.024)), pad_position)
    ET.SubElement(mount, "site", name="tcp", pos=_values(tcp), size="0.0015",
                  rgba="0 1 0 0", group="4")
    return mount


class Robotiq2F85:
    """Width-valued control and direct geometry measurements for a real 2F-85."""

    def __init__(self, model: mujoco.MjModel, options: dict):
        self.model = model
        self.options = options
        self.joint_ids = np.array([model.joint(name).id for name in JOINT_NAMES])
        self.qpos_addresses = model.jnt_qposadr[self.joint_ids]
        self.dof_addresses = model.jnt_dofadr[self.joint_ids]
        self.actuator_id = model.actuator(PREFIX + "fingers_actuator").id
        self.pad_site_ids = np.array([model.site(PREFIX + side + "_pad_inner").id
                                      for side in ("left", "right")])
        self.mount_id = model.body("gripper_mount").id
        self.pad_position = options.get("pad_contact_position", PAD_CONTACT_POSITION)
        self.opening_range = np.asarray(options.get("opening_range", [0.0, 0.085]), dtype=float)
        if (self.opening_range.shape != (2,) or not np.isfinite(self.opening_range).all()
                or self.opening_range[0] < 0 or self.opening_range[0] >= self.opening_range[1]):
            raise ValueError("opening_range must be two increasing, nonnegative metre values")
        for width in self.opening_range:
            _configuration(float(width), self.pad_position)
        self._cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}
        self._side_bodies = {}
        for side in ("left", "right"):
            for part in ("follower", "pad", "silicone_pad"):
                self._side_bodies[model.body(PREFIX + side + "_" + part).id] = side

    def configuration(self, width: float) -> tuple[np.ndarray, np.ndarray]:
        """Return exact reset hinges and nominal pad midpoint in mount frame."""
        width = float(width)
        if not np.isfinite(width) or not self.opening_range[0] <= width <= self.opening_range[1]:
            raise ValueError(f"Full gripper aperture outside configured limits: {width}")
        if width not in self._cache:
            if len(self._cache) > 512:
                self._cache.clear()
            self._cache[width] = _configuration(width, self.pad_position)
        qpos, centre = self._cache[width]
        return qpos.copy(), centre.copy()

    def control_for_width(self, width: float) -> float:
        """Convert an aperture to the native tendon position servo command."""
        qpos, _ = self.configuration(width)
        actuator = self.actuator_id
        # Native gain=0.3137255 and bias position=-100: target angle=ctrl*0.8/255.
        control = -self.model.actuator_biasprm[actuator, 1] * qpos[0]
        control /= self.model.actuator_gainprm[actuator, 0]
        low, high = self.model.actuator_ctrlrange[actuator]
        return float(np.clip(control, low, high))

    def command(self, data: mujoco.MjData, width: float) -> None:
        data.ctrl[self.actuator_id] = self.control_for_width(width)

    def preset_kinematics(self, data: mujoco.MjData, width: float) -> None:
        """Set an isolated IK/reset workspace; never call during physical rollout."""
        data.qpos[self.qpos_addresses] = self.configuration(width)[0]

    def reset(self, data: mujoco.MjData, width: float) -> None:
        self.preset_kinematics(data, width)
        data.qvel[self.dof_addresses] = 0
        self.command(data, width)

    def width(self, data: mujoco.MjData) -> float:
        """Measure inner-face separation along the mount X jaw-closing axis."""
        axis = data.xmat[self.mount_id].reshape(3, 3)[:, 0]
        left, right = data.site_xpos[self.pad_site_ids]
        return max(0.0, float(np.dot(right - left, axis)))

    def tcp_position(self, data: mujoco.MjData) -> np.ndarray:
        """Actual pad midpoint; shifts along tool Z as the linkage closes."""
        return data.site_xpos[self.pad_site_ids].mean(axis=0).copy()

    def contact_sides(self, data: mujoco.MjData, peg_body_id: int) -> set[str]:
        sides = set()
        for contact in data.contact:
            first, second = self.model.geom_bodyid[[contact.geom1, contact.geom2]]
            other = second if first == peg_body_id else first if second == peg_body_id else -1
            if other in self._side_bodies:
                sides.add(self._side_bodies[other])
        return sides
