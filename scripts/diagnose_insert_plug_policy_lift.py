#!/usr/bin/env python3
"""Passively record ACT targets, realized motion, and full predicted chunks.

The live physics, action queue, and returned policy tensors are untouched.
Forward kinematics and contact inspection use separate MjData objects. Full
predictions are copied during their actual forward calls, then decoded only
after execution, so diagnostics never call the model for extra predictions.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class LiftTrace:
    def __init__(self, sim, adapter):
        import mujoco
        self.sim, self.adapter = sim, adapter
        self.snapshot = copy.copy(sim)
        self.snapshot.data = mujoco.MjData(sim.model)
        self.kinematic_data = mujoco.MjData(sim.model)
        self.rows = []
        self.chunks = []
        self.original_step = sim.step
        self.original_predict = adapter.policy.predict_action_chunk

    def snapshot_state(self):
        import mujoco
        mujoco.mj_copyData(self.snapshot.data, self.sim.model, self.sim.data)
        mujoco.mj_forward(self.sim.model, self.snapshot.data)
        d = self.snapshot.data
        return dict(time=float(d.time), qpos=d.qpos.copy(),
                    joints=d.qpos[self.sim.arm_qadr].copy(),
                    width=self.snapshot.gripper_width(),
                    tcp=self.snapshot.site('tcp'), peg_tip=self.snapshot.site('peg_tip'),
                    contacts=sorted(self.snapshot.gripper_contacts()),
                    actuator_force=d.actuator_force[self.sim.arm_aids].copy())

    def fk(self, action, qpos):
        import mujoco
        self.kinematic_data.qpos[:] = qpos
        self.kinematic_data.qpos[self.sim.arm_qadr] = np.asarray(action)[:7]
        mujoco.mj_kinematics(self.sim.model, self.kinematic_data)
        return self.kinematic_data.site('tcp').xpos.copy()

    def predict(self, batch):
        result = self.original_predict(batch)
        self.chunks.append(dict(time=float(self.sim.data.time),
                                qpos=self.sim.data.qpos.copy(),
                                normalized=result.detach().cpu().numpy().copy()))
        return result

    def step(self, action, **kwargs):
        before = self.snapshot_state()
        target = self.fk(action, before['qpos'])
        try:
            return self.original_step(action, **kwargs)
        finally:
            after = self.snapshot_state()
            self.rows.append(dict(before=before, after=after,
                                  action=np.asarray(action).copy(), target_tcp=target))

    def install(self):
        self.sim.step = self.step
        self.adapter.policy.predict_action_chunk = self.predict

    def uninstall(self):
        self.sim.step = self.original_step
        self.adapter.policy.predict_action_chunk = self.original_predict

    def arrays(self):
        if not self.rows:
            raise ValueError('No executed commands were recorded')
        arrays = {f'{side}_{key}': np.asarray([row[side][key] for row in self.rows])
                  for side in ('before', 'after')
                  for key in ('time', 'qpos', 'joints', 'width', 'tcp', 'peg_tip', 'actuator_force')}
        for side in ('before', 'after'):
            arrays[f'{side}_dual_contact'] = np.asarray([
                set(row[side]['contacts']) == {'left', 'right'} for row in self.rows])
        arrays['action'] = np.asarray([row['action'] for row in self.rows])
        arrays['target_tcp'] = np.asarray([row['target_tcp'] for row in self.rows])
        arrays['actuator_force_range'] = self.sim.model.actuator_forcerange[self.sim.arm_aids].copy()
        arrays['actuator_force_limited'] = self.sim.model.actuator_forcelimited[self.sim.arm_aids].copy()
        arrays['chunk_time'] = np.asarray([chunk['time'] for chunk in self.chunks])
        arrays['chunk_qpos'] = np.asarray([chunk['qpos'] for chunk in self.chunks])
        normalized = np.concatenate([chunk['normalized'] for chunk in self.chunks])
        arrays['chunk_normalized_action'] = normalized
        # Execution has ended. These copies cannot change the actual queue or
        # actions that the model returned while the robot was moving.
        torch = self.adapter._torch
        with torch.inference_mode():
            decoded = self.adapter.postprocessor(torch.from_numpy(normalized.copy()))
            physical = decoded.detach().cpu().numpy().copy()
        arrays['chunk_action'] = physical
        arrays['chunk_target_tcp'] = np.asarray([
            [self.fk(action, chunk['qpos']) for action in actions]
            for chunk, actions in zip(self.chunks, physical)])
        return arrays


def distribution(values):
    values = np.asarray(values)
    return dict(min=float(np.min(values)), median=float(np.median(values)),
                max=float(np.max(values)), p95=float(np.quantile(values, .95)))


def summarize(arrays, result, n_action_steps):
    dual = arrays['after_dual_contact']
    starts = [i for i in range(max(0, len(dual)-2)) if dual[i:i+3].all()]
    report = dict(
        provenance='fresh diagnostic re-execution; not reconstructed from an earlier episode',
        stop_reason=result['stop_reason'], success=result['success'],
        control_hz=10, n_action_steps=n_action_steps, chunk_size=int(arrays['chunk_action'].shape[1]),
        rows=len(dual), actual_model_forward_calls=len(arrays['chunk_time']),
        model_calls_are_extra=False, live_state_writes_by_trace=False,
        limitations=['FK targets are kinematic arm targets, not guarantees that a held object clears a hole.',
                     'Three consecutive dual-pad contact samples identify a grasp window, not proof of a slip-free grasp.',
                     'Predictions beyond n_action_steps are unexecuted forecasts and may change on replanning.'])
    selected = arrays['chunk_action'][:, :n_action_steps].reshape(-1, 8)[:len(arrays['action'])]
    report['captured_chunk_prefix_vs_executed_action_max_abs'] = float(
        np.max(np.abs(selected-arrays['action'])))
    if not starts:
        report['stable_dual_contact_detected'] = False
        return report
    i = starts[0]
    start = float(arrays['after_time'][i])
    target = arrays['target_tcp'][i:]
    before, after = arrays['before_tcp'][i:], arrays['after_tcp'][i:]
    peg = arrays['after_peg_tip'][i:]
    joint_error = arrays['action'][i:, :7] - arrays['after_joints'][i:]
    report.update(stable_dual_contact_detected=True, grasp_start_time_s=start,
                  samples_after_grasp=len(target),
                  target_height_m=distribution(target[:, 2]), actual_height_m=distribution(after[:, 2]),
                  target_minus_actual_before_z_mm=distribution((target[:, 2]-before[:, 2])*1000),
                  target_minus_actual_after_z_mm=distribution((target[:, 2]-after[:, 2])*1000),
                  target_vs_actual_after_abs_z_error_mm=distribution(np.abs(target[:, 2]-after[:, 2])*1000),
                  commanded_target_net_rise_mm=float((target[-1, 2]-target[0, 2])*1000),
                  commanded_target_max_rise_mm=float((target[:, 2].max()-target[0, 2])*1000),
                  actual_tcp_net_rise_mm=float((after[-1, 2]-after[0, 2])*1000),
                  actual_tcp_max_rise_mm=float((after[:, 2].max()-after[0, 2])*1000),
                  peg_net_rise_mm=float((peg[-1, 2]-peg[0, 2])*1000),
                  peg_max_rise_mm=float((peg[:, 2].max()-peg[0, 2])*1000),
                  joint_tracking_abs_error_rad=distribution(np.abs(joint_error)),
                  commanded_width_m=distribution(arrays['action'][i:, 7]),
                  measured_width_m=distribution(arrays['after_width'][i:]),
                  dual_contact_fraction=float(np.mean(dual[i:])))
    limits = np.max(np.abs(arrays['actuator_force_range']), axis=1)
    forces = np.max(np.abs(arrays['after_actuator_force'][i:]), axis=0)
    report['arm_actuator_peak_abs_force'] = forces.tolist()
    report['arm_actuator_abs_force_limits'] = limits.tolist()
    report['arm_actuator_peak_fraction_of_limit'] = np.divide(
        forces, limits, out=np.zeros_like(forces), where=limits > 0).tolist()
    late = arrays['before_time'] >= max(start + 2., float(arrays['after_time'][-1]) - 10.)
    if late.any():
        report['last_ten_seconds'] = dict(
            start_time_s=float(arrays['before_time'][late][0]),
            target_height_m=distribution(arrays['target_tcp'][late, 2]),
            actual_height_m=distribution(arrays['after_tcp'][late, 2]),
            target_minus_actual_after_z_mm=distribution(
                (arrays['target_tcp'][late, 2]-arrays['after_tcp'][late, 2])*1000),
            commanded_target_net_rise_mm=float((arrays['target_tcp'][late, 2][-1]
                                                -arrays['target_tcp'][late, 2][0])*1000),
            actual_tcp_net_rise_mm=float((arrays['after_tcp'][late, 2][-1]
                                          -arrays['after_tcp'][late, 2][0])*1000),
            joint_tracking_abs_error_rad=distribution(np.abs(
                arrays['action'][late, :7]-arrays['after_joints'][late])))
    chunk_mask = arrays['chunk_time'] >= start - .1
    if chunk_mask.any():
        times = arrays['chunk_time'][chunk_mask]
        trajectories = arrays['chunk_target_tcp'][chunk_mask, :, 2]
        chunk_rows = []
        for t, heights in zip(times, trajectories):
            nearest = int(np.argmin(np.abs(arrays['before_time']-t)))
            current = arrays['before_tcp'][nearest, 2]
            chunk_rows.append(dict(time_s=float(t), actual_z_m=float(current),
                executed_prefix_max_rise_mm=float((heights[:n_action_steps].max()-current)*1000),
                full_chunk_max_rise_mm=float((heights.max()-current)*1000),
                full_chunk_last_rise_mm=float((heights[-1]-current)*1000),
                first_predicted_rise_over_1mm_step=(int(np.flatnonzero(heights-current > .001)[0])
                                                  if np.any(heights-current > .001) else None),
                first_predicted_rise_over_5mm_step=(int(np.flatnonzero(heights-current > .005)[0])
                                                  if np.any(heights-current > .005) else None)))
        report['post_grasp_predictions'] = chunk_rows
        report['executed_prefix_max_rise_mm'] = distribution([r['executed_prefix_max_rise_mm'] for r in chunk_rows])
        report['full_chunk_max_rise_mm'] = distribution([r['full_chunk_max_rise_mm'] for r in chunk_rows])
        report['chunks_with_rise_over_5mm_beyond_executed_prefix'] = sum(
            r['first_predicted_rise_over_5mm_step'] is not None and
            r['first_predicted_rise_over_5mm_step'] >= n_action_steps for r in chunk_rows)
        report['future_index_convention'] = 'zero-based: indices 0..9 are the executed prefix for n_action_steps=10'
    return report


def save_csv(path, arrays):
    fields = ['time_before_s', 'time_after_s', 'target_tcp_z_m', 'actual_tcp_z_before_m',
              'actual_tcp_z_after_m', 'peg_tip_z_m', 'target_minus_actual_after_z_mm',
              'dual_contact', 'command_width_m', 'actual_width_m', 'joint_tracking_max_error_rad']
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i in range(len(arrays['action'])):
            writer.writerow(dict(time_before_s=arrays['before_time'][i], time_after_s=arrays['after_time'][i],
                target_tcp_z_m=arrays['target_tcp'][i, 2],
                actual_tcp_z_before_m=arrays['before_tcp'][i, 2], actual_tcp_z_after_m=arrays['after_tcp'][i, 2],
                peg_tip_z_m=arrays['after_peg_tip'][i, 2],
                target_minus_actual_after_z_mm=(arrays['target_tcp'][i, 2]-arrays['after_tcp'][i, 2])*1000,
                dual_contact=bool(arrays['after_dual_contact'][i]), command_width_m=arrays['action'][i, 7],
                actual_width_m=arrays['after_width'][i], joint_tracking_max_error_rad=float(np.max(
                    np.abs(arrays['action'][i, :7]-arrays['after_joints'][i])))))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-root', type=Path, default=ROOT/'outputs/test/act_insert_plug_20261107_2044')
    parser.add_argument('--position', default='0')
    parser.add_argument('--seed', type=int, default=4)
    parser.add_argument('--duration', type=float, default=60)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    out = args.output.expanduser().resolve()
    if out.exists():
        raise FileExistsError(f'Preserving existing diagnostic: {out}')
    from examples.insert_plug.cli import _configure_gl
    _configure_gl('egl')
    from examples.insert_plug.scene import build_scene, load_config
    from examples.insert_plug.rollout import select_layouts, run_episode, _seed_policy
    from examples.insert_plug.simulation import Simulation
    from mujoco_tools.lerobot_policy import LeRobotPolicy
    batch = json.loads((args.batch_root/'eval_config.json').read_text())
    config = load_config(args.batch_root/'configs/scene_config.yaml')
    _, config = select_layouts(config, args.position, args.batch_root/'configs/positions_config.yaml')[0]
    policy = LeRobotPolicy(batch['checkpoint'], device='cuda', n_action_steps=batch['n_action_steps'])
    _seed_policy(args.seed)
    out.mkdir(parents=True)
    scene = build_scene(config, out/'scene.xml')
    sim = Simulation(scene, config, render=True)
    trace = LiftTrace(sim, policy)
    trace.install()
    try:
        result = run_episode(sim, policy, duration=args.duration, video_path=out/'rollout.mp4', quiet=False)
        trace.uninstall()
        arrays = trace.arrays()
        report = summarize(arrays, result, policy.n_action_steps)
        result.update(policy=policy.metadata(), position=f'pos{args.position}', seed=args.seed, config=config)
        report.update(checkpoint=batch['checkpoint'], position=f'pos{args.position}', seed=args.seed,
                      artifacts=dict(trace='trace.npz', samples='trace.csv', rollout='rollout.mp4', result='result.json'))
        np.savez_compressed(out/'trace.npz', **arrays)
        save_csv(out/'trace.csv', arrays)
        (out/'result.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
        (out/'diagnosis.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    finally:
        trace.uninstall()
        sim.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
