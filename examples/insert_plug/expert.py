"""Execute smooth plans with grasp/task checks and real contact dynamics."""
from __future__ import annotations

from pathlib import Path
import time
import numpy as np
from scipy.spatial.transform import Rotation

from .planning import TaskPlanner, PlanningError
from .execution_guard import MotionGuard, ExecutionSafetyError
from .recording import RecordingSampler


class Expert:
    def __init__(self, sim, writer=None, realtime=False, verbose=True, plan_directory=None):
        self.sim, self.writer = sim, writer
        self.realtime, self.verbose = realtime, verbose
        self.plan_directory = Path(plan_directory) if plan_directory is not None else None
        g = sim.config['gripper']
        self.open = g['open_width'] if 'open_width' in g else 2*g['open_position']
        self.closed = g['grasp_width'] if 'grasp_width' in g else 2*g['close_position']
        self.options = sim.config['task'].get('planning', {})
        self.steps = 0
        self.phase_results, self.final_checks, self.grasp_history = [], [], []
        self.plans, self.plan_files = [], {}
        self.preflight_plan = None
        # Ordinary peg/holder and pad contacts belong to MuJoCo dynamics. The
        # old threshold-based abort policy is only an explicit diagnostic mode.
        self.guard = MotionGuard(sim) if self.options.get('strict_collision_checks', False) else None
        self.phase = 'settle_initial'
        self.max_fit_error = 0.
        self.max_insertion_error = 0.
        self.alignment_corrections = 0
        self.recording_sampler = RecordingSampler(
            sim.config.get('recording', {}).get('sample_hz', sim.control_hz),
            sim.model.opt.timestep, start_time=sim.data.time)

    def _record_substep(self, action, phase):
        if self.recording_sampler.due(self.sim.data.time):
            self.writer.append(self.sim.record(action, phase))

    def tick(self, target, width, phase):
        self.phase = phase
        if self.guard is not None:
            self.guard.phase = phase
        # Execute the exact float32 values retained by policy datasets. The
        # simulator promotes them to float64 without changing their values.
        action = np.r_[target, width].astype(np.float32).astype(np.float64)
        if self.writer is not None:
            self.sim.step(action, before_substep=lambda active: self._record_substep(active, phase))
        else:
            self.sim.step(action)
        self.steps += 1
        if phase in ('extract', 'transfer', 'align', 'shallow_insert', 'insertion_check'):
            if len(self.sim.gripper_contacts()) != 2:
                raise ExecutionSafetyError(f'{phase}: lost opposed jaw contact with the held bolt')
        if phase == 'settle_final':
            self.final_checks.append((float(self.sim.data.time), self.sim.metrics()['success']))
        if phase == 'grasp':
            r = self.sim.data.site_xmat[self.sim.tcp_id].reshape(3, 3)
            offset = r.T @ (self.sim.site('peg_tip')-self.sim.site('tcp'))
            self.grasp_history.append((offset, self.sim.gripper_contacts()))
        if phase in ('shallow_insert', 'insertion_check'):
            self._check_fit()
        if self.realtime:
            time.sleep(self.sim.dt)

    def hold(self, seconds, width, phase):
        target = self.sim.last_target.copy()
        for _ in range(max(1, int(np.ceil(seconds/self.sim.dt)))):
            self.tick(target, width, phase)

    def _tracking_status(self, stage):
        """Check the actual endpoint pose, not just the commanded target."""
        position, rotation = np.asarray(stage['end_position']), np.asarray(stage['end_rotation'])
        error = np.linalg.norm(self.sim.site('tcp')-position)
        actual = self.sim.data.site_xmat[self.sim.tcp_id].reshape(3, 3)
        angle = np.linalg.norm(Rotation.from_matrix(rotation @ actual.T).as_rotvec())
        speed = np.max(np.abs(self.sim.data.qvel[self.sim.arm_dadr]))
        good = (error <= self.options.get('tracking_position_tolerance', .00005)
                and angle <= self.options.get('tracking_rotation_tolerance', .0005)
                and speed <= self.options.get('tracking_joint_speed_tolerance', .01))
        return good, dict(position_error_m=float(error), rotation_error_rad=float(angle),
                          max_joint_speed_rad_s=float(speed))

    def _settle_tracking(self, stage, target, stable_since=None):
        """Count stability already observed at the end of the moving segment."""
        required = float(self.options.get('tracking_stable_seconds', .2))
        if not np.isfinite(required) or required < 0:
            raise ValueError('tracking_stable_seconds must be finite and nonnegative')
        started = float(self.sim.data.time)
        count = int(np.ceil(float(self.options.get('tracking_timeout', 2.))/self.sim.dt))
        for index in range(count+1):
            good, checks = self._tracking_status(stage)
            now = float(self.sim.data.time)
            stable_since = (now if stable_since is None else stable_since) if good else None
            if stable_since is not None and now-stable_since >= required-1e-9:
                return dict(**checks, tracking_wait_seconds=now-started,
                            tracking_stable_seconds=now-stable_since)
            if index < count:
                self.tick(target, stage['width'], stage['phase'])
        raise ExecutionSafetyError(f"{stage['phase']}: tracking did not settle: "
                                   f"position {checks['position_error_m']*1000:.3f} mm, "
                                   f"rotation {checks['rotation_error_rad']:.6f} rad")

    def _settle_grasp(self):
        """Continue once both pads are stable; fixed waits remain a legacy option."""
        timeout = float(self.options.get('grasp_settle_seconds', .8))
        if not self.options.get('adaptive_grasp_settle', False):
            self.hold(timeout, self.closed, 'grasp')
            return dict(**self._check_grasp(), grasp_wait_seconds=timeout)
        if not np.isfinite(timeout) or timeout < 0:
            raise ValueError('grasp_settle_seconds must be finite and nonnegative')
        started = float(self.sim.data.time)
        count = int(np.ceil(timeout/self.sim.dt))
        for index in range(count+1):
            try:
                checks = self._check_grasp()
                return dict(**checks, grasp_wait_seconds=float(self.sim.data.time)-started)
            except ExecutionSafetyError:
                if index == count:
                    raise
            self.tick(self.sim.last_target.copy(), self.closed, 'grasp')

    def _check_fit(self):
        """The shaft axis must fit the whole future insertion interval."""
        sim = self.sim
        tip, target = sim.site('peg_tip'), sim.site('target_hole')
        axis = sim.data.xmat[sim.peg_id].reshape(3, 3)[:, 2]
        tilt = float(np.arccos(np.clip(axis[2], -1., 1.)))
        if tilt > self.options.get('maximum_held_tilt', .003) or axis[2] <= 0:
            raise ExecutionSafetyError(f'Insertion blocked: held peg tilt {tilt:.6f} rad')
        floor = sim.config['board']['bottom_z']+sim.config['board']['bottom_thickness']
        seat = max(floor, target[2]-sim.config['peg']['shaft_length'])
        errors = [np.linalg.norm((tip+axis*(height-tip[2])/axis[2])[:2]-target[:2])
                  for height in (seat, target[2])]
        error = float(max(errors))
        self.max_fit_error = max(self.max_fit_error, error)
        if self.phase in ('shallow_insert', 'insertion_check'):
            self.max_insertion_error = max(self.max_insertion_error, error)
        clearance = sim.config['board']['hole_radius']-sim.config['peg']['shaft_radius']
        budget = clearance-float(self.options.get('alignment_margin', .00015))
        if budget <= 0 or error >= budget:
            raise ExecutionSafetyError(f'Insertion blocked: shaft-axis error {error*1000:.3f} mm '
                                       f'exceeds {budget*1000:.3f} mm fit budget')
        return dict(shaft_axis_error_m=error, fit_budget_m=budget, tilt_rad=tilt)

    def _check_grasp(self):
        # Include both endpoints: N samples cover only N-1 control intervals.
        count = max(3, int(np.ceil(.2/self.sim.dt))+1)
        history = self.grasp_history[-count:]
        if len(history) < count or any(len(sides) != 2 for _, sides in history):
            raise ExecutionSafetyError('Grasp did not maintain contact on both jaws for 200 ms')
        offsets = np.array([offset for offset, _ in history])
        drift = float(np.max(np.linalg.norm(offsets-offsets[-1], axis=1)))
        if drift > .0002:
            raise ExecutionSafetyError(f'Grasp still slipping: relative motion {drift*1000:.3f} mm')
        return dict(both_jaws_stable_seconds=.2, held_relative_drift_m=drift)

    def _execute_stage(self, plan, stage, check_alignment=True):
        phase = stage['phase']
        if phase in ('shallow_insert', 'release'):
            self._check_fit()
        if phase == 'release':
            depth = self.sim.site('target_hole')[2]-self.sim.site('peg_tip')[2]
            expected = float(self.options.get('insertion_depth', .006))
            if depth < expected-.0003 or len(self.sim.gripper_contacts()) != 2:
                raise ExecutionSafetyError(f'Release blocked: depth {depth*1000:.3f} mm or lost grasp')
        tracking = phase in ('approach', 'descend_to_grasp', 'extract', 'transfer',
                             'align', 'shallow_insert', 'retract')
        if self.options.get('continuous_free_space', False) and phase in ('approach', 'extract', 'retract'):
            # These are clearance waypoints, not contact operations. Continue
            # the validated targets without waiting for a 50-micrometre pose.
            # Extraction clearance and final seated/withdrawn success are still
            # checked below; real MuJoCo contact dynamics stay active.
            tracking = False
        if self.options.get('continuous_hover', False) and phase in ('approach', 'transfer', 'align'):
            tracking = False
        stable_since = None
        for index in range(stage['start_index'], stage['end_index']):
            self.tick(plan.arm_qpos[index], plan.gripper_width[index], phase)
            if tracking:
                good, _ = self._tracking_status(stage)
                stable_since = (float(self.sim.data.time) if stable_since is None else stable_since) if good else None
        checks = {}
        if tracking:
            checks.update(self._settle_tracking(stage, plan.arm_qpos[stage['end_index']-1], stable_since))
        if phase == 'descend_to_grasp':
            error = self.sim.site('tcp')-self.sim.site('peg_head')
            if np.linalg.norm(error[:2]) > .00015 or abs(error[2]) > .0004:
                raise ExecutionSafetyError(f'Grasp alignment failed: TCP/head offset {error.tolist()}')
        if phase == 'grasp':
            checks.update(self._settle_grasp())
        if phase == 'extract':
            if self.sim.site('peg_tip')[2] <= max(self.sim.site('source_hole')[2], self.sim.site('target_hole')[2])+.015:
                raise ExecutionSafetyError('Extraction did not establish transport clearance')
        if phase in ('shallow_insert', 'insertion_check') or (phase == 'align' and check_alignment):
            checks.update(self._check_fit())
        if phase == 'release':
            required = 2*self.sim.config['peg']['head_radius']+.005
            if self.sim.gripper_contacts() or self.sim.gripper_width() < required:
                raise ExecutionSafetyError('Retraction blocked: jaws have not released the bolt')
        result = dict(phase=phase, time=float(self.sim.data.time), checks=checks, **self.sim.metrics())
        self.phase_results.append(result)
        if self.verbose:
            print(f'{phase}: t={result["time"]:.2f}s tip={np.round(result["peg_tip_xyz"],5)} '
                  f'contacts={result["gripper_contacts"]}', flush=True)

    def _save_plan(self, plan, name):
        self.plans.append(plan)
        if self.plan_directory is not None:
            self.plan_files[name] = plan.save(self.plan_directory, stem=name)

    def run(self):
        previous_callback = self.sim.safety_callback
        try:
            # Solve the nominal IK and motion bounds before advancing physics.
            plan = self.preflight_plan if self.preflight_plan is not None else self.prepare()
            self.sim.safety_callback = self.guard or previous_callback
            for stage in plan.stages:
                self._execute_stage(plan, stage)
                if stage['phase'] == 'grasp':
                    break
            # Compliant real pads change the held pose: replan and validate the
            # complete suffix from the measured transform before extraction.
            suffix = TaskPlanner(self.sim).plan(start_phase='extract')
            self._save_plan(suffix, 'measured_grasp_trajectory')
            for stage in suffix.stages:
                self._execute_stage(suffix, stage)
                if stage['phase'] == 'transfer':
                    break
            # Reobserve at transport height, then correct alignment in free
            # space. The entry fit budget is never relaxed to accept a drift.
            continuous = bool(self.options.get('continuous_hover', False))
            incoming_velocity = stage.get('end_velocity', [0., 0., 0.])
            for attempt in range(1 if continuous else 3):
                insertion = TaskPlanner(self.sim).plan(
                    start_phase='align', continue_from_target=continuous,
                    initial_velocity=incoming_velocity if continuous else None)
                self._save_plan(insertion, f'target_alignment_{attempt}')
                self._execute_stage(insertion, insertion.stages[0], check_alignment=False)
                try:
                    fit = self._check_fit()
                    self.phase_results[-1]['checks'].update(fit)
                    break
                except ExecutionSafetyError:
                    if continuous or attempt == 2:
                        raise
                    self.alignment_corrections += 1
            for stage in insertion.stages[1:]:
                self._execute_stage(insertion, stage)
            return self.result('')
        except (PlanningError, ExecutionSafetyError) as error:
            return self.result(f'{type(error).__name__}: {error}')
        finally:
            self.sim.safety_callback = previous_callback

    def prepare(self):
        """Preflight before opening a dataset writer or advancing physics."""
        self.preflight_plan = TaskPlanner(self.sim).plan()
        self._save_plan(self.preflight_plan, 'nominal_trajectory')
        return self.preflight_plan

    def result(self, reason):
        metrics = self.sim.metrics()
        # Count elapsed observation time, not samples. At 10 Hz three positive
        # samples span only 200 ms; four are needed to establish at least 250 ms.
        stable_since = None
        for timestamp, good in reversed(self.final_checks):
            if not good:
                break
            stable_since = timestamp
        stable_span = (self.final_checks[-1][0]-stable_since) if stable_since is not None else 0.
        stable = bool(stable_span >= .25-1e-9)
        metrics['stable_for_final_250ms'] = stable
        metrics['final_success_observation_span_seconds'] = float(stable_span)
        success = bool(metrics['success'] and stable and not reason)
        contacts = (dict(collision_checks_enabled=True, **self.guard.summary()) if self.guard is not None
                    else dict(collision_checks_enabled=False, checks=0,
                              unsafe_contact_count=None, last_violation=None))
        return dict(success=success, failure_reason=reason or ('' if success else 'final_seating_check_failed'),
                    steps=self.steps, metrics=metrics, phases=self.phase_results,
                    trajectory_files=self.plan_files,
                    planning=[plan.validation for plan in self.plans],
                    execution_checks=dict(**contacts,
                        action_command_dtype='float32',
                        control_hold_seconds=float(self.sim.dt),
                        max_observed_alignment_error_m=self.max_fit_error,
                        max_insertion_axis_error_m=self.max_insertion_error,
                        alignment_corrections=self.alignment_corrections))
