"""Offline task trajectories for the configured two sockets.

This is a constrained waypoint planner, not a general obstacle-avoidance search.
It rejects unreachable poses and enforces joint motion bounds. Optional strict
diagnostics additionally reject colliding sampled paths. All
kinematics and collision queries use private MjData; the live simulation is never
stepped or teleported. Contact dynamics remain the executor's responsibility.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


class PlanningError(RuntimeError):
    """The configured task has no verified trajectory along its chosen corridor."""


def _smooth(u):
    return u**3 * (10 + u * (-15 + 6 * u))


def _quintic_positions(u, start, end, start_velocity, end_velocity, duration):
    """C2 Cartesian segment with specified endpoint velocities and zero acceleration."""
    u = np.asarray(u, dtype=float)[:, None]
    start, end = np.asarray(start), np.asarray(end)
    v0 = np.asarray(start_velocity)*duration
    v1 = np.asarray(end_velocity)*duration
    delta = end-start
    return (start + v0*u + (10*delta-6*v0-4*v1)*u**3
            + (-15*delta+8*v0+7*v1)*u**4 + (6*delta-3*v0-3*v1)*u**5)


def _motion_profile(u, profile='quintic', ramp_fraction=.2):
    """Normalized position with zero endpoint speed and acceleration.

    The cosine-velocity profile reaches cruise speed during a short ramp,
    instead of spreading the slow start over the entire extraction. Both
    profiles describe targets only; execution still holds each target at 10 Hz.
    """
    if profile == 'quintic':
        return _smooth(np.asarray(u))
    if profile != 'cosine_velocity':
        raise PlanningError(f'Unknown motion profile: {profile}')
    r = float(ramp_fraction)
    if not np.isfinite(r) or not 0 < r <= .5:
        raise PlanningError('extraction_ramp_fraction must be in (0, 0.5]')
    u = np.asarray(u, dtype=float)
    ramp = lambda t: .5*(t-r/np.pi*np.sin(np.pi*t/r))/(1-r)
    return np.where(u < r, ramp(u),
                    np.where(u > 1-r, 1-ramp(1-u), (u-r/2)/(1-r)))


def _profile_bounds(profile, ramp_fraction):
    if profile == 'quintic':
        return 1.875, 5.774
    # Validate the selector and ramp before computing analytic peak bounds.
    _motion_profile(np.array([0., 1.]), profile, ramp_fraction)
    return 1/(1-ramp_fraction), np.pi/(2*ramp_fraction*(1-ramp_fraction))


@dataclass
class TrajectoryPlan:
    time: np.ndarray
    arm_qpos: np.ndarray
    gripper_width: np.ndarray
    tcp_position: np.ndarray
    tcp_rotation: np.ndarray
    phase: np.ndarray
    stages: list[dict]
    validation: dict
    initial_arm_qpos: np.ndarray

    @property
    def actions(self):
        return np.column_stack((self.arm_qpos, self.gripper_width))

    def save(self, directory, stem='trajectory'):
        """Save readable stage poses and portable, pickle-free reference arrays."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        json_path, npz_path = directory / f'{stem}.json', directory / f'{stem}.npz'
        document = dict(format_version=1, units=dict(position='metres', angle='radians', time='seconds'),
                        planner='configured smooth Cartesian waypoints with sequential IK',
                        reference='Nominal actuator targets; contact execution is monitored separately.',
                        samples=len(self.time), duration=float(self.time[-1]),
                        initial_arm_qpos=self.initial_arm_qpos.tolist(), stages=self.stages,
                        validation=self.validation)
        json_path.write_text(json.dumps(document, indent=2, allow_nan=False), encoding='utf8')
        np.savez_compressed(npz_path, time=self.time, arm_qpos=self.arm_qpos,
                            gripper_width=self.gripper_width, tcp_position=self.tcp_position,
                            tcp_rotation=self.tcp_rotation, phase=self.phase, action=self.actions,
                            initial_arm_qpos=self.initial_arm_qpos)
        return dict(json=str(json_path.resolve()), npz=str(npz_path.resolve()))


class TaskPlanner:
    """Plan source (orange) → target (green), reading all positions from the scene.

    ``plan()`` preflights the whole nominal task. ``plan(start_phase='extract')``
    replans its suffix from the physically measured grasp, preserving the complete
    peg-to-TCP transform. ``plan(start_phase='align')`` refreshes that transform
    above the target before insertion. No source, target, or grasp position is a
    public argument.
    """

    def __init__(self, sim):
        self.sim, self.model, self.config = sim, sim.model, sim.config
        self.task = self.config['task']
        self.options = self.task.get('planning', {})
        self.strict_collisions = bool(self.options.get('strict_collision_checks', False))
        self.data = mujoco.MjData(self.model)
        self.ik_data = mujoco.MjData(self.model)
        self.base_qpos = sim.data.qpos.copy()
        self.data.qpos[:] = self.base_qpos
        self.ik_data.qpos[:] = self.base_qpos
        self.dt = float(sim.dt)
        self.tcp_id = self.model.site(self.task.get('tcp_site', 'tcp')).id
        self.tip_id = self.model.site(self.task.get('peg_tip_site', 'peg_tip')).id
        self.head_id = self.model.site(self.task.get('peg_head_site', 'peg_head')).id
        self.peg_id = sim.peg_id
        self.peg_qadr = int(self.model.joint('peg_free').qposadr[0])
        self.source = sim.site(self.task.get('source_site', 'source_hole'))
        self.target = sim.site(self.task.get('target_site', 'target_hole'))
        self.source_body = self.model.body('source_socket').id
        self.target_body = self.model.body('target_socket').id
        self.open = float(self.config['gripper'].get('open_width',
                         2*self.config['gripper'].get('open_position', 0.04)))
        self.closed = float(self.config['gripper'].get('grasp_width',
                           2*self.config['gripper'].get('close_position', 0.01)))
        self.joint_limits = self.model.jnt_range[sim.arm_jids]
        self.jp = np.zeros((3, self.model.nv))
        self.jr = self.jp.copy()
        self.pad_bodies = {}
        for bid in range(self.model.nbody):
            name = self.model.body(bid).name
            if ('pad' in name or 'finger' in name) and ('left' in name or 'right' in name):
                self.pad_bodies[bid] = 'left' if 'left' in name else 'right'
        self.checked = 0
        self.worst_allowed_penetration = 0.0
        self.contact_width = self.closed

    def _positive(self, key, default):
        value = float(self.options.get(key, default))
        if not np.isfinite(value) or value <= 0:
            raise PlanningError(f'task.planning.{key} must be a finite positive value')
        return value

    def _joint_bound(self, name, default):
        value = np.broadcast_to(np.asarray(self.options.get(name, default), dtype=float), (7,)).copy()
        if not np.isfinite(value).all() or np.any(value <= 0):
            raise PlanningError(f'task.planning.{name} must contain positive finite joint bounds')
        return value

    def _ik(self, position, rotation, seed, phase):
        q = np.array(seed, copy=True)
        d, m = self.ik_data, self.model
        for _ in range(220):
            d.qpos[self.sim.arm_qadr] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            ep = position - d.site_xpos[self.tcp_id]
            er = Rotation.from_matrix(rotation @ d.site_xmat[self.tcp_id].reshape(3, 3).T).as_rotvec()
            if np.linalg.norm(ep) < 1e-8 and np.linalg.norm(er) < 1e-7:
                return q
            mujoco.mj_jacSite(m, d, self.jp, self.jr, self.tcp_id)
            jac = np.vstack((self.jp[:, self.sim.arm_dadr], 0.2*self.jr[:, self.sim.arm_dadr]))
            dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-8*np.eye(6), np.r_[ep, 0.2*er])
            dq *= min(1.0, 0.12 / max(np.linalg.norm(dq), 1e-12))
            q = np.clip(q + dq, self.joint_limits[:, 0]+1e-6, self.joint_limits[:, 1]-1e-6)
        raise PlanningError(f'{phase}: unreachable TCP {np.round(position, 6).tolist()}; '
                            f'IK residual {np.linalg.norm(ep):.6g} m / {np.linalg.norm(er):.6g} rad. '
                            'Adjust robot.base_position, board positions, or task heights in the config.')

    def _set_peg(self, position, rotation):
        self.data.qpos[self.peg_qadr:self.peg_qadr+3] = position
        xyzw = Rotation.from_matrix(rotation).as_quat()
        self.data.qpos[self.peg_qadr+3:self.peg_qadr+7] = xyzw[[3, 0, 1, 2]]

    def _set_width(self, width):
        if self.sim.gripper is not None:
            self.sim.gripper.preset_kinematics(self.data, width)
        else:
            self.data.qpos[self.sim.finger_qadr] = width / 2

    def _contact_sides(self):
        sides = set()
        for c in self.data.contact:
            a, b = self.model.geom_bodyid[[c.geom1, c.geom2]]
            other = b if a == self.peg_id else a if b == self.peg_id else -1
            if other in self.pad_bodies:
                sides.add(self.pad_bodies[other])
        return sides

    def _contact_aperture(self, q, peg_position, peg_rotation):
        """Locate actual opposed pad/head contact, without compressing rigid meshes.

        The commanded aperture remains smaller to produce physical grip force.
        This private nominal linkage stops at the mesh contact aperture instead.
        """
        self.data.qpos[self.sim.arm_qadr] = q
        self._set_peg(peg_position, peg_rotation)
        low, high = self.closed, self.open
        self._set_width(low)
        mujoco.mj_forward(self.model, self.data)
        if len(self._contact_sides()) < 2:
            raise PlanningError('grasp: configured grasp_width cannot establish opposed finger/head contact. '
                                'Check gripper aperture, grasp yaw, and the initial peg pose.')
        for _ in range(22):
            middle = (low+high)/2
            self._set_width(middle)
            mujoco.mj_forward(self.model, self.data)
            if len(self._contact_sides()) >= 2:
                low = middle
            else:
                high = middle
        return max(self.closed, low - 1e-5)

    def _peg_pose(self, phase, progress, tcp, rotation):
        if phase in ('settle_initial', 'approach', 'descend_to_grasp', 'grasp'):
            return self.source_peg_position, self.source_peg_rotation
        if phase in ('retract', 'settle_final'):
            return self.target_peg_position, self.upright_peg_rotation
        position = tcp + rotation @ self.held_tip_local
        peg_rotation = rotation @ self.held_rotation
        if phase == 'release':
            # Check the released peg's complete vertical seating sweep. This is
            # geometry validation, not a claim to predict the free-fall timing.
            position = position + _smooth(progress) * (self.target_peg_position-position)
            peg_rotation = self.upright_peg_rotation
        return position, peg_rotation

    def _validate_contacts(self, phase, timestamp):
        m = self.model
        tolerance = self._positive('contact_tolerance', 0.00015)
        pad_tolerance = self._positive('grasp_contact_tolerance', 0.00075)
        for contact in self.data.contact:
            g1, g2 = int(contact.geom1), int(contact.geom2)
            a, b = (int(x) for x in m.geom_bodyid[[g1, g2]])
            penetration = max(0.0, -float(contact.dist))
            other = b if a == self.peg_id else a if b == self.peg_id else -1
            allowed = False
            if other in self.pad_bodies and phase in ('grasp', 'extract', 'transfer', 'align',
                                                     'shallow_insert', 'insertion_check', 'release'):
                allowed = penetration <= pad_tolerance
            if other == self.source_body and phase in ('settle_initial', 'approach',
                                                       'descend_to_grasp', 'grasp', 'extract'):
                allowed = penetration <= tolerance
            if other == self.target_body and phase in ('shallow_insert', 'insertion_check',
                                                       'release', 'retract', 'settle_final'):
                allowed = penetration <= tolerance
            if allowed:
                self.worst_allowed_penetration = max(self.worst_allowed_penetration, penetration)
                continue
            if penetration <= 1e-7:
                continue  # Numerical mesh tangency, not a penetrating collision.
            names = [m.geom(g).name or m.body(int(m.geom_bodyid[g])).name for g in (g1, g2)]
            raise PlanningError(f'{phase} at {timestamp:.3f}s: collision between {names[0]} and {names[1]} '
                                f'({penetration*1000:.3f} mm penetration). Adjust task clearance/approach '
                                'heights or configured fixture/robot positions; no unchecked motion was executed.')

    def _validate_sample(self, q, width, phase, progress, timestamp):
        if not self.strict_collisions:
            return
        self.data.qpos[self.sim.arm_qadr] = q
        self._set_width(max(width, self.contact_width) if phase not in
                        ('settle_initial', 'approach', 'descend_to_grasp', 'retract', 'settle_final') else width)
        mujoco.mj_kinematics(self.model, self.data)
        tcp = self.data.site_xpos[self.tcp_id].copy()
        rotation = self.data.site_xmat[self.tcp_id].reshape(3, 3).copy()
        peg_position, peg_rotation = self._peg_pose(phase, progress, tcp, rotation)
        self._set_peg(peg_position, peg_rotation)
        mujoco.mj_forward(self.model, self.data)
        self._validate_contacts(phase, timestamp)
        self.checked += 1

    def plan(self, start_phase=None, *, continue_from_target=False, initial_velocity=None):
        if start_phase not in (None, 'extract', 'align'):
            raise ValueError("start_phase must be None (whole task), 'extract', or 'align' (measured grasp suffix)")
        self.checked = 0
        self.worst_allowed_penetration = 0.0
        s, task, peg, board = self.sim, self.task, self.config['peg'], self.config['board']
        q0 = s.data.qpos[s.arm_qadr].copy()
        p0 = s.data.site_xpos[self.tcp_id].copy()
        r0 = s.data.site_xmat[self.tcp_id].reshape(3, 3).copy()
        tip = s.data.site_xpos[self.tip_id].copy()
        object_rotation = s.data.xmat[self.peg_id].reshape(3, 3).copy()
        floor = float(board['bottom_z'] + board['bottom_thickness'])
        source_seat = max(floor, float(self.source[2])-float(peg['shaft_length']))
        target_seat = max(floor, float(self.target[2])-float(peg['shaft_length']))
        clearance = float(board['hole_radius']-peg['shaft_radius'])
        if start_phase != 'align' and np.linalg.norm(tip[:2]-self.source[:2]) > clearance:
            raise PlanningError('Initial peg is outside the configured orange source bore; reset the scene before planning.')
        self.source_peg_position = np.r_[self.source[:2], source_seat]
        self.source_peg_rotation = object_rotation
        yaw = math.atan2(object_rotation[1, 0], object_rotation[0, 0])
        self.upright_peg_rotation = Rotation.from_euler('z', yaw).as_matrix()
        self.target_peg_position = np.r_[self.target[:2], target_seat]
        grasp_rotation = s.grasp_rotation()
        head_from_tip = s.data.site_xpos[self.head_id]-tip
        grasp_position = self.source_peg_position + head_from_tip
        if start_phase is not None:
            if len(s.gripper_contacts()) < 2:
                raise PlanningError(f'{start_phase}: measured suffix requires established contact on both fingers; '
                                    'hold the closed gripper and verify the grasp before lifting.')
            self.held_tip_local = r0.T @ (tip-p0)
            self.held_rotation = r0.T @ object_rotation
            grasp_rotation = self.upright_peg_rotation @ self.held_rotation.T
            self.contact_width = self._contact_aperture(q0, tip, object_rotation)
        else:
            if object_rotation[2, 2] < math.cos(0.02):
                raise PlanningError('Initial source peg is tilted; reset it upright before planning the vertical grasp.')
            self.held_tip_local = grasp_rotation.T @ (self.source_peg_position-grasp_position)
            self.held_rotation = grasp_rotation.T @ object_rotation
            grasp_q = self._ik(grasp_position, grasp_rotation, q0, 'grasp preflight')
            self.contact_width = self._contact_aperture(grasp_q, self.source_peg_position, object_rotation)
        if continue_from_target:
            if start_phase != 'align':
                raise ValueError('Moving continuation is only supported above the target')
            # Measure the held transform from actual physics above; continue
            # commands from the last target, not the servo-lagged actual pose.
            q0 = s.last_target.copy()
            self.ik_data.qpos[:] = s.data.qpos
            self.ik_data.qpos[s.arm_qadr] = q0
            mujoco.mj_kinematics(self.model, self.ik_data)
            p0 = self.ik_data.site_xpos[self.tcp_id].copy()
            r0 = self.ik_data.site_xmat[self.tcp_id].reshape(3, 3).copy()
        transport = float(task['transport_height'])
        approach = float(task['approach_height'])
        if not np.isfinite(transport) or not np.isfinite(approach) or approach <= 0:
            raise PlanningError('task transport_height and approach_height must be finite, with a positive approach height')
        lift_start = p0 if start_phase is not None else grasp_position
        lift_rotation = r0 if start_phase is not None else grasp_rotation
        lift = lift_start.copy()
        # transport_height is the requested absolute TCP height. An additional
        # minimum rise applies only when explicitly configured, not invisibly.
        lift[2] = max(transport, lift_start[2]+float(task.get('extraction_height', 0.)))
        held_tip_lift = lift + lift_rotation @ self.held_tip_local
        minimum_clearance = self._positive('transport_clearance', 0.015)
        if held_tip_lift[2] < max(self.source[2], self.target[2])+minimum_clearance:
            raise PlanningError('task.transport_height is too low to clear the source and target rims with the held peg')
        transfer = self.target - grasp_rotation @ self.held_tip_local
        transfer[2] = lift[2]
        preinsert_clearance = self._positive('preinsert_clearance', task.get('alignment_height', 0.015))
        if start_phase == 'align' and tip[2] < self.target[2]+preinsert_clearance-.001:
            raise PlanningError('align: held peg tip is too low for a lateral alignment correction; '
                                'it must clear the target rim by task.planning.preinsert_clearance '
                                '(within 1 mm) before replanning.')
        above = self.target + [0, 0, preinsert_clearance] - grasp_rotation @ self.held_tip_local
        shallow_depth = self._positive('insertion_depth', task.get('shallow_insertion_depth', 0.006))
        if not 0 < shallow_depth < self.target[2]-target_seat:
            raise PlanningError('task.planning.insertion_depth must enter the bore without driving the head into its rim')
        shallow = self.target - [0, 0, shallow_depth] - grasp_rotation @ self.held_tip_local
        retract = shallow + [0, 0, self._positive('retract_height', task.get('retract_height', 0.08))]
        specs = []
        def add(name, position, rotation, seconds, width):
            specs.append((name, np.asarray(position).copy(), np.asarray(rotation).copy(), float(seconds), float(width)))
        if start_phase is None:
            add('settle_initial', p0, r0, .3, self.open)
            add('approach', grasp_position+[0, 0, approach], grasp_rotation,
                task.get('approach_duration', 2.), self.open)
            add('descend_to_grasp', grasp_position, grasp_rotation,
                task.get('descent_duration', 1.5), self.open)
            add('grasp', grasp_position, grasp_rotation, .6, self.closed)
        if start_phase != 'align':
            add('extract', lift, lift_rotation, task.get('extraction_duration', 2.5), self.closed)
            add('transfer', transfer, grasp_rotation, task.get('transfer_duration', 2.), self.closed)
        alignment_seconds = float(task.get('alignment_duration', 1.5))
        if start_phase == 'align' and np.linalg.norm(above-p0) < .002:
            # A measured submillimetre correction need not spend another 1.5 s
            # almost motionless. Joint limits can still lengthen this minimum.
            alignment_seconds = self._positive('alignment_correction_duration', 1.5)
        add('align', above, grasp_rotation, alignment_seconds, self.closed)
        add('shallow_insert', shallow, grasp_rotation, task.get('insertion_duration', 3.), self.closed)
        add('insertion_check', shallow, grasp_rotation, .25, self.closed)
        add('release', shallow, grasp_rotation, task.get('release_duration', .5), self.open)
        add('retract', retract, grasp_rotation, 1., self.open)
        add('settle_final', retract, grasp_rotation, task['settling_seconds'], self.open)
        vmax = self._joint_bound('max_joint_velocity', .8)
        amax = self._joint_bound('max_joint_acceleration', 2.)
        cv = self._positive('max_cartesian_velocity', .12)
        ca = self._positive('max_cartesian_acceleration', .6)
        joint_step = self._positive('collision_joint_step', .015)
        cart_step = self._positive('collision_cartesian_step', .001)
        all_q, all_p, all_r, all_w, all_phase, stages = [], [], [], [], [], []
        q, p, r = q0.copy(), p0.copy(), r0.copy()
        width = float(s.gripper_width()) if start_phase else self.open
        if start_phase:
            width = self.closed
        elapsed = 0.
        continuous = bool(self.options.get('continuous_hover', False))
        velocity_start = np.zeros(3) if initial_velocity is None else np.asarray(initial_velocity, dtype=float)
        if velocity_start.shape != (3,) or not np.isfinite(velocity_start).all():
            raise PlanningError('initial_velocity must be a finite 3-vector')
        self._validate_sample(q0, width, specs[0][0], 0., 0.)
        maxima_v, maxima_a = np.zeros(7), np.zeros(7)
        for phase, end, rend, seconds, end_width in specs:
            # Empty-hand approach may move faster than compliant grasp transport.
            # Keep the loaded path's speed bound independent of this setting.
            phase_cv = self._positive('approach_max_cartesian_velocity', cv) if phase == 'approach' else cv
            velocity_end = np.zeros(3)
            if continuous and phase == 'approach':
                velocity_end[2] = -self._positive('grasp_approach_speed', .01)
            if continuous and phase in ('transfer', 'align'):
                velocity_end[2] = -self._positive('place_approach_speed', .006)
            flowing = bool(np.any(velocity_start) or np.any(velocity_end))
            profile = task.get('extraction_profile', 'quintic') if phase == 'extract' else 'quintic'
            ramp_fraction = float(task.get('extraction_ramp_fraction', .2))
            speed_bound, acceleration_bound = _profile_bounds(profile, ramp_fraction)
            distance = np.linalg.norm(end-p)
            seconds = max(seconds, speed_bound*distance/phase_cv, math.sqrt(acceleration_bound*distance/ca))
            rotations = Slerp([0, 1], Rotation.from_matrix(np.stack((r, rend))))
            qs = None
            for attempt in range(7):
                count = max(1, math.ceil(seconds/self.dt))
                fractions = np.arange(1, count+1)/count
                blend = _motion_profile(fractions, profile, ramp_fraction)
                positions = (_quintic_positions(fractions, p, end, velocity_start, velocity_end,
                                                count*self.dt) if flowing else
                             p + blend[:, None]*(end-p))
                matrices = rotations(blend).as_matrix()
                widths = width + blend*(end_width-width)
                seed, candidates = q.copy(), []
                for position, matrix in zip(positions, matrices):
                    seed = self._ik(position, matrix, seed, phase)
                    candidates.append(seed)
                qs = np.asarray(candidates)
                velocity = np.diff(np.vstack((q, qs)), axis=0)/self.dt
                acceleration = np.diff(np.vstack((np.zeros(7), velocity, np.zeros(7))), axis=0)/self.dt
                peak_v = np.max(np.abs(velocity), axis=0)
                peak_a = np.max(np.abs(acceleration), axis=0)
                factor = max(float(np.max(peak_v/vmax)), math.sqrt(float(np.max(peak_a/amax))))
                cart_velocity = np.diff(np.vstack((p, positions)), axis=0)/self.dt
                cart_acceleration = np.diff(np.vstack((velocity_start, cart_velocity, velocity_end)), axis=0)/self.dt
                factor = max(factor, float(np.max(np.linalg.norm(cart_velocity, axis=1)))/phase_cv,
                             math.sqrt(float(np.max(np.linalg.norm(cart_acceleration, axis=1)))/ca))
                if factor <= 1.00001:
                    break
                seconds = count*self.dt*factor*1.03
            else:
                raise PlanningError(f'{phase}: unable to satisfy joint speed/acceleration limits after retiming')
            start_index = len(all_q)
            previous_q, previous_p, previous_width = q, p, width
            for i, (qi, pi, wi) in enumerate(zip(qs, positions, widths)):
                subdivisions = max(1, math.ceil(np.max(np.abs(qi-previous_q))/joint_step),
                                   math.ceil(np.linalg.norm(pi-previous_p)/cart_step),
                                   math.ceil(abs(wi-previous_width)/.001))
                for sub in range(1, subdivisions+1):
                    f = sub/subdivisions
                    self._validate_sample(previous_q + f*(qi-previous_q),
                                          previous_width + f*(wi-previous_width), phase,
                                          (i+f)/count, elapsed+(i+f)*self.dt)
                previous_q, previous_p, previous_width = qi, pi, wi
            maxima_v, maxima_a = np.maximum(maxima_v, peak_v), np.maximum(maxima_a, peak_a)
            all_q.extend(qs); all_p.extend(positions); all_r.extend(matrices)
            all_w.extend(widths); all_phase.extend([phase]*count)
            duration = count*self.dt
            stages.append(dict(phase=phase, start_index=start_index, end_index=len(all_q),
                               start_time=elapsed, end_time=elapsed+duration, duration=duration,
                               start_position=p.tolist(), end_position=end.tolist(), end_pos=end.tolist(),
                               start_rotation=r.tolist(), end_rotation=rend.tolist(), width=end_width,
                               interpolation=('quintic Hermite position, smooth rotation' if flowing
                                              else f'{profile} Cartesian position and rotation'),
                               motion_profile='quintic_hermite' if flowing else profile,
                               start_velocity=velocity_start.tolist(), end_velocity=velocity_end.tolist(),
                               cartesian_velocity_limit_m_s=phase_cv,
                               ramp_fraction=ramp_fraction if profile == 'cosine_velocity' else None))
            elapsed += duration
            q, p, r, width = qs[-1], end, rend, end_width
            velocity_start = velocity_end
        validation = dict(valid=True, collision_samples=self.checked,
                          collision_checks_enabled=self.strict_collisions,
                          initial_state_checked=self.strict_collisions,
                          duration_seconds=elapsed, sample_interval_seconds=self.dt,
                          stage_bounds=[{key: stage[key] for key in
                                        ('phase', 'start_index', 'end_index', 'start_time', 'end_time', 'duration')}
                                        for stage in stages],
                          collision_joint_step_rad=joint_step, collision_cartesian_step_m=cart_step,
                          max_joint_velocity_rad_s=maxima_v.tolist(),
                          max_joint_acceleration_rad_s2=maxima_a.tolist(),
                          joint_velocity_limits_rad_s=vmax.tolist(), joint_acceleration_limits_rad_s2=amax.tolist(),
                          max_allowed_contact_penetration_m=(self.worst_allowed_penetration
                                                            if self.strict_collisions else None),
                          nominal_contact_aperture_m=float(self.contact_width),
                          source_site_xyz=self.source.tolist(), target_site_xyz=self.target.tolist(),
                          source_seated_tip_z=source_seat, target_seated_tip_z=target_seat,
                          held_tip_in_tcp=self.held_tip_local.tolist(), held_rotation_in_tcp=self.held_rotation.tolist(),
                          measured_grasp=start_phase is not None,
                          start_phase=start_phase or 'settle_initial',
                          scope=('Dense sampled collider checks; not continuous collision proof or contact dynamics.'
                                 if self.strict_collisions else
                                 'IK and motion bounds only. Physical contacts are resolved by MuJoCo during execution.'))
        return TrajectoryPlan(np.arange(1, len(all_q)+1)*self.dt, np.asarray(all_q), np.asarray(all_w),
                              np.asarray(all_p), np.asarray(all_r), np.asarray(all_phase), stages, validation, q0)
