#!/usr/bin/env python3
"""Run the real expert without rendering and retain scalar motion diagnostics.

Each selected fixed layout runs once. This is a physics validation, not a new
training dataset: no cameras are rendered and no original episodes are changed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import hashlib
import json
import multiprocessing
from pathlib import Path
from tempfile import TemporaryDirectory
import time

import mujoco
import numpy as np
import yaml

from examples.insert_plug.control.expert import Expert
from examples.insert_plug.environment.layouts import head_visibility
from examples.insert_plug.environment.scene import build_scene, load_config
from examples.insert_plug.environment.simulation import Simulation
from mujoco_tools.paths import WORKSPACE_ROOT


def emit(**values):
    print(json.dumps(values, ensure_ascii=False, allow_nan=False), flush=True)


class TracedExpert(Expert):
    """Observe each actual command; all planning, guards and physics stay intact."""

    def __init__(self, sim):
        super().__init__(sim, verbose=False)
        self.trace = []
        self.fk_data = mujoco.MjData(sim.model)

    def tick(self, target, width, phase):
        sim = self.sim
        action = np.r_[target, width].astype(np.float32).astype(np.float64)
        self.fk_data.qpos[:] = sim.data.qpos
        self.fk_data.qpos[sim.arm_qadr] = action[:7]
        mujoco.mj_kinematics(sim.model, self.fk_data)
        row = dict(
            time=float(sim.data.time), phase=phase, action=action,
            qpos=sim.data.qpos.copy(), qvel=sim.data.qvel.copy(),
            tcp=sim.site('tcp'), peg_tip=sim.site('peg_tip'),
            tcp_rotation=sim.data.site_xmat[sim.tcp_id].copy().reshape(3, 3),
            target_tcp=self.fk_data.site_xpos[sim.tcp_id].copy(),
            width=float(sim.gripper_width()), contacts=len(sim.gripper_contacts()),
            completed=False,
        )
        try:
            super().tick(target, width, phase)
            row['completed'] = True
        finally:
            row.update(end_time=float(sim.data.time), end_tcp=sim.site('tcp'),
                       end_peg_tip=sim.site('peg_tip'), end_width=float(sim.gripper_width()),
                       end_qpos=sim.data.qpos.copy(), end_qvel=sim.data.qvel.copy())
            self.trace.append(row)


def motion_statistics(rows, dt, *, phase_results=None, rim_height=None):
    if not rows:
        return dict(commands=0, completed_commands=0, transitions=0)
    action = np.asarray([r['action'] for r in rows])
    tcp = np.asarray([r['tcp'] for r in rows])
    width = np.asarray([r['width'] for r in rows])
    rotations = np.asarray([r['tcp_rotation'] for r in rows])
    same = np.all(action[1:] == action[:-1], axis=1)
    angle = np.degrees(np.arccos(np.clip(
        (np.einsum('nij,nij->n', rotations[:-1], rotations[1:])-1)/2, -1, 1)))
    near_static = ((np.linalg.norm(np.diff(tcp, axis=0), axis=1) < .0001)
                   & (np.abs(np.diff(width)) < .0001) & (angle < .01))
    end_tcp = np.asarray([r['end_tcp'] for r in rows])
    peg_tip = np.asarray([r['peg_tip'] for r in rows])
    end_peg_tip = np.asarray([r['end_peg_tip'] for r in rows])
    intervals = np.asarray([r['end_time']-r['time'] for r in rows])
    actual_step_mm = np.linalg.norm(end_tcp-tcp, axis=1)*1000
    actual_speed_mm_s = np.divide(actual_step_mm, intervals,
                                  out=np.zeros_like(actual_step_mm), where=intervals > 0)
    target_tcp = np.asarray([r['target_tcp'] for r in rows])
    phase_names = np.asarray([r['phase'] for r in rows])
    boundaries = []
    window_count = max(1, int(np.ceil(.5/dt)))
    for index in np.flatnonzero(phase_names[1:] != phase_names[:-1])+1:
        begin, end = max(0, index-window_count), min(len(rows), index+window_count)
        repeated_before = 0
        previous = index-1
        while previous > 0 and phase_names[previous-1] == phase_names[index-1] and same[previous-1]:
            repeated_before += 1
            previous -= 1
        boundaries.append(dict(
            previous_phase=str(phase_names[index-1]), phase=str(phase_names[index]),
            time_s=float(rows[index]['time']),
            target_tcp_increment_at_boundary_mm=float(np.linalg.norm(
                target_tcp[index]-target_tcp[index-1])*1000),
            actual_tcp_speed_before_mm_s=float(actual_speed_mm_s[index-1]),
            actual_tcp_speed_after_mm_s=float(actual_speed_mm_s[index]),
            actual_tcp_step_before_mm=float(actual_step_mm[index-1]),
            actual_tcp_step_after_mm=float(actual_step_mm[index]),
            preceding_repeated_target_intervals=repeated_before,
            preceding_repeated_target_seconds=repeated_before*dt,
            window_start_s=float(rows[begin]['time']), window_end_s=float(rows[end-1]['end_time']),
            window_min_actual_tcp_speed_mm_s=float(actual_speed_mm_s[begin:end].min()),
            window_median_actual_tcp_speed_mm_s=float(np.median(actual_speed_mm_s[begin:end])),
            window_repeated_target_intervals=int(same[begin:end-1].sum()),
        ))
    carrying = np.isin(phase_names, ['extract', 'transfer', 'align', 'shallow_insert', 'insertion_check'])
    transferring = phase_names == 'transfer'
    heights = dict(
        max_tcp_z_m=float(max(tcp[:, 2].max(), end_tcp[:, 2].max())),
        max_peg_tip_z_m=float(max(peg_tip[:, 2].max(), end_peg_tip[:, 2].max())),
        carried_max_tcp_z_m=(float(max(tcp[carrying, 2].max(), end_tcp[carrying, 2].max()))
                             if carrying.any() else None),
        rim_height_m=rim_height,
        transfer_min_tip_clearance_m=(float(min(peg_tip[transferring, 2].min(),
                                                end_peg_tip[transferring, 2].min())-rim_height)
                                      if transferring.any() and rim_height is not None else None),
    )
    stage_waits = [dict(
        phase=stage['phase'], end_time_s=float(stage['time']),
        tracking_wait_seconds=stage.get('checks', {}).get('tracking_wait_seconds'),
        grasp_wait_seconds=stage.get('checks', {}).get('grasp_wait_seconds'),
    ) for stage in (phase_results or [])]
    phases = {}
    for name in dict.fromkeys(r['phase'] for r in rows):
        indices = [i for i, row in enumerate(rows) if row['phase'] == name]
        selected = [rows[i] for i in indices]
        phases[name] = dict(
            commands=len(indices), duration_seconds=sum(r['end_time']-r['time'] for r in selected),
            repeated_targets=sum(bool(same[i-1]) for i in indices if i > 0),
        )
        if name in ('extract', 'shallow_insert'):
            sign = 1 if name == 'extract' else -1
            origin = selected[0]['tcp'][2]
            elapsed = np.array([r['end_time']-selected[0]['time'] for r in selected])
            motion = sign*np.array([r['end_tcp'][2]-origin for r in selected])*1000
            index = np.flatnonzero(elapsed >= 1.-1e-8)
            phases[name]['actual_tcp_motion_after_1s_mm'] = (
                float(motion[index[0]]) if len(index) else None)
            index = np.flatnonzero(motion >= 5.)
            phases[name]['actual_tcp_5mm_time_seconds'] = (
                float(elapsed[index[0]]) if len(index) else None)
    return dict(
        commands=len(rows), completed_commands=sum(r['completed'] for r in rows),
        transitions=len(same), repeated_full_action_transitions=int(same.sum()),
        repeated_full_action_fraction=float(same.mean()) if len(same) else None,
        near_static_transitions=int(near_static.sum()),
        near_static_fraction=float(near_static.mean()) if len(near_static) else None,
        commanded_interval_seconds=dt, phases=phases,
        phase_boundaries=boundaries, stage_waits=stage_waits, heights=heights,
    )


def run_position(base, position, output, margin_px=5.):
    started = time.monotonic()
    config = copy.deepcopy(base)
    for key in ('source_xy', 'target_xy'):
        config['board'][key] = list(position[key])
    name = position['name']
    destination = Path(output)/name
    destination.mkdir()
    (destination/'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf8')
    with TemporaryDirectory(prefix=f'insert_plug_expert_check_{name}_') as temp:
        scene = build_scene(config, Path(temp)/'scene.xml')
        sim = Simulation(scene, config, render=False)
        expert = TracedExpert(sim)
        try:
            initial_visibility = head_visibility(sim, margin_px)
            try:
                result = expert.run()
            except Exception as error:
                result = expert.result(f'{type(error).__name__}: {error}')
                result['runtime_error'] = True
            final_visibility = head_visibility(sim, margin_px)
        finally:
            sim.close()
        stats = motion_statistics(expert.trace, sim.dt, phase_results=result.get('phases'),
                                  rim_height=float(max(sim.site('source_hole')[2], sim.site('target_hole')[2])))
        checks = result['execution_checks']
        contact_checks_enabled = checks.get('collision_checks_enabled', True)
        expected_checks = stats['completed_commands']*sim.substeps
        physics_checks_complete = ((expected_checks > 0 and checks['checks'] == expected_checks)
                                   if contact_checks_enabled else None)
        contact_clean = ((checks['unsafe_contact_count'] == 0 and checks['last_violation'] is None)
                         if contact_checks_enabled else None)
        all_commands_completed = (stats['commands'] == stats['completed_commands'] == result['steps'])
        visibility_passed = initial_visibility['all_inside'] and final_visibility['all_inside']
        contact_validation = not contact_checks_enabled or (physics_checks_complete and contact_clean)
        validated_success = bool(result['success'] and contact_validation
                                 and all_commands_completed and not result.get('runtime_error', False)
                                 and visibility_passed)
        arrays = {key: np.asarray([row[key] for row in expert.trace])
                  for key in expert.trace[0]} if expert.trace else {}
        np.savez_compressed(destination/'trace.npz', **arrays)
        document = dict(position=name, source_xy=position['source_xy'], target_xy=position['target_xy'],
                        seed=int(name.removeprefix('pos'))*10,
                        randomized=False, wall_seconds=time.monotonic()-started,
                        simulated_seconds=float(sim.data.time),
                        visibility=dict(initial=initial_visibility, final=final_visibility),
                        visibility_passed=bool(visibility_passed),
                        physics_checks=checks['checks'], expected_physics_checks=expected_checks,
                        collision_checks_enabled=contact_checks_enabled,
                        physics_checks_complete=physics_checks_complete,
                        all_commands_completed=bool(all_commands_completed),
                        contact_clean=contact_clean, validated_success=validated_success,
                        result=result, motion=stats)
        (destination/'result.json').write_text(json.dumps(document, indent=2, allow_nan=False)+'\n')
        emit(event='position_complete', position=name, success=validated_success,
             failure_reason=result['failure_reason'], commands=stats['commands'],
             physics_checks_complete=physics_checks_complete, contact_clean=contact_clean,
             visibility_passed=bool(visibility_passed),
             simulated_seconds=float(sim.data.time), wall_seconds=document['wall_seconds'])
        return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=WORKSPACE_ROOT/'config/insert_plug.yaml')
    parser.add_argument('--positions-config', type=Path,
                        default=WORKSPACE_ROOT/'config/insert_plug_positions.yaml')
    parser.add_argument('--position', action='append', help='Select a layout; repeat for several.')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--output', type=Path, required=True, help='A fresh directory under outputs/_checks.')
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    checks = (WORKSPACE_ROOT/'outputs/_checks').resolve()
    if not output.is_relative_to(checks) or output == checks:
        parser.error('--output must be a new subdirectory under outputs/_checks')
    if args.workers < 1:
        parser.error('--workers must be positive')
    manifest = yaml.safe_load(args.positions_config.read_text())
    margin_px = float(manifest.get('required_edge_margin_px', 5.))
    if not np.isfinite(margin_px) or margin_px < 5.:
        parser.error('Position manifest required_edge_margin_px must be finite and at least 5 pixels')
    positions = manifest['positions']
    names = {p['name'] for p in positions}
    if args.position:
        if set(args.position)-names:
            parser.error(f'Unknown positions: {sorted(set(args.position)-names)}')
        positions = [p for p in positions if p['name'] in args.position]
    if not positions:
        parser.error('No positions selected')
    base = load_config(args.config)
    output.mkdir(parents=True, exist_ok=False)
    (output/'base_config.yaml').write_text(yaml.safe_dump(base, sort_keys=False), encoding='utf8')
    (output/'positions_config.yaml').write_text(args.positions_config.read_text(), encoding='utf8')
    emit(event='validation_start', positions=[p['name'] for p in positions], output=str(output))
    started = time.monotonic()
    if args.workers == 1:
        results = [run_position(base, p, output, margin_px) for p in positions]
    else:
        with ProcessPoolExecutor(max_workers=args.workers,
                                 mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = [pool.submit(run_position, base, p, output, margin_px) for p in positions]
            results = [future.result() for future in as_completed(futures)]
    results.sort(key=lambda r: int(r['position'].removeprefix('pos')))
    transitions = sum(r['motion']['transitions'] for r in results)
    repeated = sum(r['motion'].get('repeated_full_action_transitions', 0) for r in results)
    document = dict(
        format_version=1, config=str(args.config.resolve()),
        config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
        control_hz=base['task']['control_hz'], physics_timestep_seconds=base['physics']['timestep'],
        rendered=False, recorded_training_data=False, randomized=False,
        wall_seconds=time.monotonic()-started, positions=len(results),
        successes=sum(r['validated_success'] for r in results),
        task_successes=sum(r['result']['success'] for r in results),
        all_successful=all(r['validated_success'] for r in results),
        collision_checks_enabled=all(r['collision_checks_enabled'] for r in results),
        physics_checks_complete=(all(r['physics_checks_complete'] for r in results)
                                 if all(r['collision_checks_enabled'] for r in results) else None),
        contact_clean=(all(r['contact_clean'] for r in results)
                       if all(r['collision_checks_enabled'] for r in results) else None),
        visibility_passed=all(r['visibility_passed'] for r in results),
        required_edge_margin_px=margin_px,
        physics_checks=sum(r['physics_checks'] for r in results),
        expected_physics_checks=sum(r['expected_physics_checks'] for r in results),
        commands=sum(r['motion']['commands'] for r in results), transitions=transitions,
        repeated_full_action_transitions=repeated,
        repeated_full_action_fraction=repeated/transitions if transitions else None,
        definitions=dict(
            trace='One attempted command per row: pre-action state plus final state after the actual interval.',
            action='Exact eight float32 values executed, represented losslessly as float64.',
            repeats='Full eight-dimensional adjacent targets identical; does not imply physical immobility.',
            near_static='TCP <0.1 mm, rotation <0.01 degrees, aperture <0.1 mm per adjacent interval.',
            lift='Actual TCP motion from phase-entry state; after 1 s includes completed commands only.',
            phase_boundaries='Actual speed is endpoint displacement divided by the executed control interval; this is a 100 ms interval average, not instantaneous speed. Local windows span up to 0.5 s on each side.',
            preceding_repeated_targets='Consecutive identical full targets at the end of the preceding phase, excluding its first endpoint command; repeated targets do not imply actual immobility.',
            stage_waits='Reported execution waits, including repeated alignment attempts; null means no tracking/grasp wait was reported for that stage.',
            heights='World Z in metres. Carried peak excludes initial home and final retraction; transfer clearance uses actual pre/post-control peg-tip heights above the higher rim.',
            seeds='Fixed-layout seed labels match the first episode per position; no stochastic changes.',
            visibility='Unmodified initial/final head_visibility reports, including all mesh vertices; occlusion is not tested.',
            physics_checks_complete='When explicitly enabled, guard calls equal completed commands times physics substeps. Null when contact diagnostics are disabled.',
            validated_success='Physical success, all commands completed, no runtime errors, initial/final visibility; full guard coverage and clean contacts additionally required only when explicitly enabled.',
        ), results=results,
    )
    baseline_path = checks/'insert_plug_motion_density/report.json'
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text())
        document['previous_dataset_reference'] = dict(
            source=str(baseline_path), episodes=baseline['episodes'], frames=baseline['frames'],
            repeated_full_action_fraction=baseline['unchanged_full_action_fraction'],
            mean_duration_seconds=baseline['frames']*.1/baseline['episodes'],
        )
    (output/'summary.json').write_text(json.dumps(document, indent=2, allow_nan=False)+'\n')
    emit(event='validation_complete', successes=document['successes'], positions=len(results),
         summary=str(output/'summary.json'))
    return 0 if document['all_successful'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
