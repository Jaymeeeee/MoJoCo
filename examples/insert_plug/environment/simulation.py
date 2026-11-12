"""Simulation boundary shared by the scripted expert and learned policies.

Only reset and the separate IK workspace write qpos. All rollout movement goes
through position actuators and mj_step, including grasping and the released peg.
"""
from __future__ import annotations

import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from mujoco_tools.cameras import CAMERA_NAMES, render_observations, camera_metadata


class Simulation:
    def __init__(self, model_path, config, render=True):
        self.config = config
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.ik_data = mujoco.MjData(self.model)
        self.arm_jids = np.array([self.model.joint(n).id for n in config['robot']['arm_joint_names']])
        self.arm_qadr = self.model.jnt_qposadr[self.arm_jids]
        self.arm_dadr = self.model.jnt_dofadr[self.arm_jids]
        self.arm_aids = np.array([next(i for i in range(self.model.nu)
            if self.model.actuator_trnid[i, 0] == j) for j in self.arm_jids])
        self.gripper = None
        if config['gripper'].get('type') == 'robotiq_2f85':
            from mujoco_tools.robotiq import Robotiq2F85
            self.gripper = Robotiq2F85(self.model, config['gripper'])
        else:
            self.finger_jids = np.array([self.model.joint(n).id for n in config['gripper']['joint_names']])
            self.finger_qadr = self.model.jnt_qposadr[self.finger_jids]
            self.finger_aids = np.array([self.model.actuator(n).id for n in config['gripper']['actuator_names']])
        self.tcp_id = self.model.site('tcp').id
        self.peg_id = self.model.body('peg').id
        self.control_hz = float(config['task']['control_hz'])
        ratio = 1.0 / self.control_hz / self.model.opt.timestep
        if not np.isclose(ratio, round(ratio), atol=1e-8):
            raise ValueError('control_hz must divide the physics frequency exactly')
        self.substeps = int(round(ratio))
        self.dt = self.substeps * self.model.opt.timestep
        self.renderer = None
        self._record_data = None
        self.depth_enabled = bool(config['cameras'].get('depth', {}).get('enabled', False))
        if render:
            self.renderer = mujoco.Renderer(self.model,
                height=int(config['cameras']['height']), width=int(config['cameras']['width']))
        self.viewer = None
        self.safety_callback = None
        self.reset()

    def reset(self):
        if self.model.nkey:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        else:
            mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.arm_qadr] = self.config['robot']['home_qpos']
        self.data.ctrl[self.arm_aids] = self.data.qpos[self.arm_qadr]
        if self.gripper is not None:
            self.gripper.reset(self.data, self.config['gripper']['open_width'])
        else:
            self.data.qpos[self.finger_qadr] = self.config['gripper']['open_position']
            self.data.ctrl[self.finger_aids] = self.config['gripper']['open_position']
        mujoco.mj_forward(self.model, self.data)
        self.last_target = self.data.qpos[self.arm_qadr].copy()

    def site(self, name):
        return self.data.site(name).xpos.copy()

    def render_images(self):
        if self.renderer is None:
            raise RuntimeError('This simulation was opened without rendering')
        return render_observations(self.renderer, self.data, depth_enabled=False)[0]

    def render_rgbd(self, data=None):
        if self.renderer is None:
            raise RuntimeError('This simulation was opened without rendering')
        return render_observations(self.renderer, self.data if data is None else data,
                                   depth_enabled=self.depth_enabled)

    def camera_calibration(self):
        return camera_metadata(self.model, self.data, self.config['cameras']['width'],
                               self.config['cameras']['height'])

    def observe(self, data=None):
        """Policy-visible state only. No object/goal ground truth is returned."""
        data = self.data if data is None else data
        images, depth = self.render_rgbd(data)
        observation = {f'observation.images.{name}': images[name] for name in CAMERA_NAMES}
        observation.update({f'observation.depth.{name}': depth[name] for name in depth})
        observation['observation.state'] = np.r_[data.qpos[self.arm_qadr],
                                                self.gripper_width(data)].astype(np.float32)
        return observation

    def gripper_width(self, data=None):
        """Actual full aperture in metres, independent of actuator transmission."""
        data = self.data if data is None else data
        if self.gripper is not None:
            return self.gripper.width(data)
        return float(data.qpos[self.finger_qadr].sum())

    def record(self, action, phase):
        # mj_step integrates qpos after computing geometry. At an interior
        # physics tick those derived poses can still describe the prior tick.
        # Refresh a copy so RGB, depth and poses match the recorded qpos/time,
        # without changing the live solver's warm-start state or contact path.
        if self._record_data is None:
            self._record_data = mujoco.MjData(self.model)
        data = self._record_data
        mujoco.mj_copyData(data, self.model, self.data)
        mujoco.mj_forward(self.model, data)
        obs = self.observe(data)
        record = dict(images={name: obs[f'observation.images.{name}'] for name in CAMERA_NAMES},
            joint_position=data.qpos[self.arm_qadr].copy(),
            joint_velocity=data.qvel[self.arm_dadr].copy(),
            gripper_width=np.array([self.gripper_width(data)]),
            timestamp=float(data.time), phase=phase,
            action=np.asarray(action).copy(), qpos=data.qpos.copy(),
            qvel=data.qvel.copy(), ctrl=data.ctrl.copy(),
            peg_pose=np.r_[data.xpos[self.peg_id], data.xquat[self.peg_id]],
            source_pose=np.r_[data.site('source_hole').xpos, [1., 0., 0., 0.]],
            target_pose=np.r_[data.site('target_hole').xpos, [1., 0., 0., 0.]])
        if self.depth_enabled:
            record['depth'] = {name: obs[f'observation.depth.{name}'] for name in CAMERA_NAMES}
        return record

    def step(self, action, *, before_substep=None):
        """Apply one control interval, optionally observing before physics ticks.

        ``before_substep(action)`` sees the active control target and the actual
        time immediately before each physics integration. It must not mutate
        live physics state; recording refreshes geometry on a separate copy.
        """
        action = np.asarray(action, dtype=float)
        if action.shape != (8,) or not np.isfinite(action).all():
            raise ValueError('Action must be eight finite values: seven joint targets and full aperture')
        limits = self.model.jnt_range[self.arm_jids]
        if np.any(action[:7] < limits[:, 0]) or np.any(action[:7] > limits[:, 1]):
            raise ValueError('Joint target outside robot limits')
        g = self.config['gripper']
        grange = g['opening_range'] if self.gripper is not None else np.array(g['joint_range']) * 2
        if not grange[0] <= action[7] <= grange[1]:
            raise ValueError('Full gripper aperture outside configured limits')
        self.data.ctrl[self.arm_aids] = action[:7]
        if self.gripper is not None:
            self.gripper.command(self.data, action[7])
        else:
            self.data.ctrl[self.finger_aids] = action[7] / 2
        self.last_target = action[:7].copy()
        previous_time = float(self.data.time)
        previous_warnings = self.data.warning.number.copy()
        if self.safety_callback is None and before_substep is None:
            mujoco.mj_step(self.model, self.data, nstep=self.substeps)
        else:
            for _ in range(self.substeps):
                if before_substep is not None:
                    before_substep(action)
                mujoco.mj_step(self.model, self.data)
                if self.safety_callback is not None:
                    self.safety_callback()
        mujoco.mj_forward(self.model, self.data)
        if not np.isclose(self.data.time - previous_time, self.dt, rtol=0, atol=1e-8):
            raise RuntimeError('Physics time discontinuity: possible automatic numerical reset')
        if np.any(self.data.warning.number > previous_warnings):
            raise RuntimeError(f'MuJoCo numerical warning: {self.data.warning.number.tolist()}')
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise RuntimeError('Nonfinite physics state')
        if self.viewer is not None:
            self.viewer.sync()

    def inverse_kinematics(self, position, rotation=None, seed=None):
        """Damped Jacobian IK in a separate data object; never teleports the robot."""
        if rotation is None:
            rotation = self.grasp_rotation()
        target = np.asarray(position, dtype=float)
        d = self.ik_data
        d.qpos[:] = self.data.qpos
        q = self.last_target.copy() if seed is None else np.array(seed, dtype=float)
        limits = self.model.jnt_range[self.arm_jids]
        jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        for _ in range(180):
            d.qpos[self.arm_qadr] = q
            mujoco.mj_forward(self.model, d)
            poserr = target - d.site_xpos[self.tcp_id]
            rot = d.site_xmat[self.tcp_id].reshape(3, 3)
            roterr = Rotation.from_matrix(rotation @ rot.T).as_rotvec()
            if np.linalg.norm(poserr) < 2e-5 and np.linalg.norm(roterr) < 2e-4:
                return q
            mujoco.mj_jacSite(self.model, d, jp, jr, self.tcp_id)
            jac = np.vstack([jp[:, self.arm_dadr], 0.2 * jr[:, self.arm_dadr]])
            err = np.r_[poserr, 0.2 * roterr]
            dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-6 * np.eye(6), err)
            dq *= min(1., 0.15 / max(np.linalg.norm(dq), 1e-12))
            q = np.clip(q + dq, limits[:, 0] + 1e-5, limits[:, 1] - 1e-5)
        raise RuntimeError(f'IK did not converge: position error {np.linalg.norm(poserr):.6f} m')

    def grasp_rotation(self):
        yaw = float(self.config['task'].get('grasp_yaw', 0.0))
        return Rotation.from_euler('z', yaw).as_matrix() @ np.diag([1., -1., -1.])

    def gripper_contacts(self):
        """Names of gripper bodies currently in contact with the free peg."""
        if self.gripper is not None:
            return self.gripper.contact_sides(self.data, self.peg_id)
        names = set()
        for c in self.data.contact:
            b1, b2 = self.model.geom_bodyid[[c.geom1, c.geom2]]
            if self.peg_id in (b1, b2):
                other = b2 if b1 == self.peg_id else b1
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, int(other)) or ''
                if 'finger' in name:
                    names.add(name)
        return names

    def metrics(self):
        tip, target = self.site('peg_tip'), self.site('target_hole')
        axis = self.data.xmat[self.peg_id].reshape(3, 3)[:, 2]
        floor = self.config['board']['bottom_z'] + self.config['board']['bottom_thickness']
        # A short shaft seats with its head on the rim; a longer shaft reaches
        # the cavity floor. Do not require the STL bolt to sink through its head.
        seated_tip_z = max(floor, target[2] - self.config['peg']['shaft_length'])
        pj = self.model.joint('peg_free').dofadr[0]
        speed = float(np.linalg.norm(self.data.qvel[pj:pj+3]))
        angular_speed = float(np.linalg.norm(self.data.qvel[pj+3:pj+6]))
        error = float(np.linalg.norm(tip[:2] - target[:2]))
        tilt = float(np.arccos(np.clip(axis[2], -1, 1)))
        clear = self.config['board']['hole_radius'] - self.config['peg']['shaft_radius']
        depth = float(target[2] - tip[2])
        floor_error = float(abs(tip[2] - floor))
        seating_error = float(abs(tip[2] - seated_tip_z))
        mouth_error = float('inf')
        if axis[2] > 0.1:
            intersection = tip + axis * (target[2] - tip[2]) / axis[2]
            mouth_error = float(np.linalg.norm(intersection[:2] - target[:2]))
        penetration = 0.
        for c in self.data.contact:
            if self.peg_id in self.model.geom_bodyid[[c.geom1, c.geom2]]:
                penetration = max(penetration, float(-c.dist))
        contacts = self.gripper_contacts()
        aperture = self.gripper_width()
        success = bool(error < clear and mouth_error < clear
            and seating_error < min(self.config['task']['seating_tolerance'], .0007)
            and penetration < .00015 and tilt < 0.10 and speed < .005 and angular_speed < .1 and not contacts
            and aperture > 2*self.config['peg']['head_radius'] + .004
            and self.site('tcp')[2] > self.site('peg_head')[2] + .04)
        return dict(success=success, radial_error_m=error, insertion_depth_m=depth,
            floor_error_m=floor_error, seating_error_m=seating_error,
            seated_tip_height_m=float(seated_tip_z), tilt_rad=tilt, linear_speed_m_s=speed,
            angular_speed_rad_s=angular_speed, mouth_axis_error_m=mouth_error if np.isfinite(mouth_error) else None,
            max_peg_penetration_m=penetration,
            gripper_contacts=sorted(contacts), aperture_m=aperture,
            peg_tip_xyz=tip.tolist(), tcp_xyz=self.site('tcp').tolist())

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
        if self.renderer is not None:
            self.renderer.close()
