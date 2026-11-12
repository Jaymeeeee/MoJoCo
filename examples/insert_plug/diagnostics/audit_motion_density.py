#!/usr/bin/env python3
"""Read scalar trajectory data and measure exact holds and slow-motion windows.

No simulator integration, image/depth decoding, dataset changes, or policy runs.
FK evaluates saved qpos and absolute action targets without advancing time.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import h5py
import mujoco
import numpy as np

from mujoco_tools.paths import WORKSPACE_ROOT

ROOT = WORKSPACE_ROOT


def read(path):
    path = path.resolve()
    with h5py.File(path, 'r') as f:
        return dict(path=str(path), action=f['action'][:], qpos=f['simulation/qpos'][:],
                    qvel=f['simulation/qvel'][:], time=f['timestamp'][:],
                    phase=f['phase'].asstr()[:], width=f['observations/gripper_width'][:].reshape(-1),
                    meta=json.loads(f.attrs['metadata_json']))


def fingerprint(row):
    h = hashlib.sha256()
    for name in ('action', 'qpos', 'qvel', 'time'):
        h.update(row[name].tobytes())
    h.update('|'.join(row['phase']).encode())
    return h.hexdigest()


def runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return [(int(a), int(b)) for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]


def first_cross(time, displacement, threshold):
    indices = np.flatnonzero(displacement >= threshold)
    return float(time[indices[0]]) if len(indices) else None


def representative(row):
    path = Path(row['path'])
    model = mujoco.MjModel.from_xml_path(str(path.with_suffix('.xml')))
    data = mujoco.MjData(model)
    arm = np.array([model.joint(n).qposadr[0] for n in row['meta']['config']['robot']['arm_joint_names']])
    tcp = model.site('tcp').id
    peg = model.site('peg_tip').id
    actual_p, actual_r, target_p, peg_p = [], [], [], []
    for q, a in zip(row['qpos'], row['action']):
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        actual_p.append(data.site_xpos[tcp].copy())
        actual_r.append(data.site_xmat[tcp].copy().reshape(3, 3))
        peg_p.append(data.site_xpos[peg].copy())
        data.qpos[arm] = a[:7]
        mujoco.mj_kinematics(model, data)
        target_p.append(data.site_xpos[tcp].copy())
    actual_p, target_p, peg_p = map(np.asarray, (actual_p, target_p, peg_p))
    rotations = np.asarray(actual_r)
    cos_angle = (np.einsum('nij,nij->n', rotations[:-1], rotations[1:])-1)/2
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
    actual_distance = np.linalg.norm(np.diff(actual_p, axis=0), axis=1)*1000
    target_distance = np.linalg.norm(np.diff(target_p, axis=0), axis=1)*1000
    width_change = np.abs(np.diff(row['width']))*1000
    peg_distance = np.linalg.norm(np.diff(peg_p, axis=0), axis=1)*1000
    result = dict(source=str(path.relative_to(ROOT)), frames=len(row['time']),
                  duration_s=len(row['time'])*.1, transitions=len(row['time'])-1,
                  slow_motion_thresholds={}, phases=[], transitions_of_interest=[])
    for mm in (.01, .1):
        mask = (actual_distance < mm) & (angle < .01) & (width_change < mm)
        result['slow_motion_thresholds'][str(mm)] = dict(
            tcp_step_mm_below=mm, rotation_step_deg_below=.01, actual_gripper_step_mm_below=mm,
            qualifying_intervals=int(mask.sum()),
            fraction=float(mask.mean()), longest_contiguous_seconds=max([b-a for a, b in runs(mask)] or [0])*.1,
            robot_and_peg_tip_qualifying_intervals=int((mask & (peg_distance < mm)).sum()))
    full_same = np.r_[False, np.all(np.diff(row['action'], axis=0) == 0, axis=1)]
    arm_same = np.r_[False, np.all(np.diff(row['action'][:, :7], axis=0) == 0, axis=1)]
    result['constant_action_runs'] = [dict(
        first_repeated_frame=a, repeated_intervals=b-a, repeated_duration_s=(b-a)*.1,
        start_s=float(row['time'][a-1]), phases=list(dict.fromkeys(row['phase'][a:b])))
        for a, b in runs(full_same)]
    phase_edges = np.r_[0, np.flatnonzero(row['phase'][1:] != row['phase'][:-1])+1, len(row['phase'])]
    for begin, end in zip(phase_edges[:-1], phase_edges[1:]):
        name = str(row['phase'][begin])
        tail = end-1
        while tail > begin and np.array_equal(row['action'][tail-1], row['action'][end-1]):
            tail -= 1
        result['phases'].append(dict(phase=name, first_frame=int(begin), frames=int(end-begin),
            start_s=float(row['time'][begin]), duration_s=float((end-begin)*.1),
            unchanged_8d_targets=int(full_same[begin:end].sum()),
            unchanged_arm_targets=int(arm_same[begin:end].sum()),
            terminal_constant_target_frames=int(end-tail),
            terminal_constant_target_duration_s=float((end-tail)*.1)))
        if name in {'extract', 'align', 'shallow_insert'}:
            sign = 1 if name == 'extract' else -1
            local_t = row['time'][begin:end]-row['time'][begin]
            previous = max(0, begin-1)
            target_delta = sign*(target_p[begin:end, 2]-target_p[previous, 2])*1000
            actual_delta = sign*(actual_p[begin:end, 2]-actual_p[begin, 2])*1000
            sample = {}
            for elapsed in (.9, 1., 2., 3.):
                found = np.flatnonzero(local_t >= elapsed-1e-8)
                if len(found):
                    j = found[0]
                    sample[str(elapsed)] = dict(target_displacement_mm=float(target_delta[j]),
                                               actual_displacement_mm=float(actual_delta[j]))
            result['transitions_of_interest'].append(dict(phase=name,
                previous_phase=str(row['phase'][previous]), start_s=float(row['time'][begin]),
                direction='up' if sign == 1 else 'down',
                first_command_displacement_mm=float(target_delta[0]),
                total_target_displacement_mm=float(target_delta[-1]),
                total_actual_displacement_mm=float(actual_delta[-1]),
                first_target_1mm_s=first_cross(local_t, target_delta, 1),
                first_target_5mm_s=first_cross(local_t, target_delta, 5),
                first_actual_1mm_s=first_cross(local_t, actual_delta, 1),
                first_actual_5mm_s=first_cross(local_t, actual_delta, 5), samples=sample))
    series = [dict(frame=i, time_s=float(t), phase=str(phase),
                   target_tcp_z_mm=float(target_p[i, 2]*1000), actual_tcp_z_mm=float(actual_p[i, 2]*1000),
                   peg_tip_z_mm=float(peg_p[i, 2]*1000), target_width_mm=float(row['action'][i, 7]*1000),
                   actual_width_mm=float(row['width'][i]*1000),
                   target_delta_mm=float(target_distance[i-1]) if i else None,
                   actual_delta_mm=float(actual_distance[i-1]) if i else None,
                   rotation_delta_deg=float(angle[i-1]) if i else None,
                   peg_tip_delta_mm=float(peg_distance[i-1]) if i else None,
                   action_exactly_repeated=bool(full_same[i]))
              for i, (t, phase) in enumerate(zip(row['time'], row['phase']))]
    return result, series


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'outputs/datasets/insert_plug')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/_checks/insert_plug_motion_density')
    args = parser.parse_args()
    total, phases, hashes, reps = Counter(), defaultdict(Counter), defaultdict(set), {}
    position_counts = Counter()
    files = sorted(args.source.glob('pos*/episode_*/*.h5'))
    if len(files) != 100:
        raise ValueError(f'Expected 100 files, found {len(files)}')
    for path in files:
        row = read(path)
        a, ph = row['action'], row['phase']
        same = np.all(a[1:] == a[:-1], axis=1)
        arm = np.all(a[1:, :7] == a[:-1, :7], axis=1)
        total.update(frames=len(a), transitions=len(a)-1, unchanged_full_action=int(same.sum()),
                     unchanged_arm_action=int(arm.sum()),
                     stationary_arm_changing_gripper=int((arm & ~same).sum()))
        position = path.parent.parent.name
        position_counts[position] += 1
        hashes[position].add(fingerprint(row))
        reps.setdefault(position, row)
        for phase in np.unique(ph):
            mask = ph[1:] == phase
            phases[str(phase)].update(frames=int((ph == phase).sum()),
                                     unchanged_full_action=int((same & mask).sum()),
                                     unchanged_arm_action=int((arm & mask).sum()))
    if any(len(v) != 1 for v in hashes.values()):
        raise ValueError('Representatives cannot summarize nonidentical scalar trajectories within position')
    args.output.mkdir(parents=True, exist_ok=True)
    representatives = {}
    for position, row in reps.items():
        representative_stats, series = representative(row)
        representatives[position] = representative_stats
        with (args.output/f'{position}_timeline.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(series[0]))
            writer.writeheader(); writer.writerows(series)
    doc = dict(source=str(args.source.resolve()), episodes=len(files), **total,
        unchanged_full_action_fraction=total['unchanged_full_action']/total['transitions'],
        unchanged_arm_action_fraction=total['unchanged_arm_action']/total['transitions'],
        phases=dict(phases), scalar_trajectory_unique_per_position={k: len(v) for k, v in hashes.items()},
        episodes_per_position=dict(position_counts),
        scalar_trajectory_unique_total=len(set.union(*hashes.values())),
        slow_motion_full_batch={},
        definitions=dict(exact_action_repeats='Adjacent full 8D absolute targets are bitwise equal; excludes first frame of each episode. This does not imply no actual motion.',
                         arm_repeats='Only seven arm targets equal; gripper can still move.',
                         hashing='Actions, saved qpos/qvel, time and phase only; RGB/depth were not compared.',
                         interval_duration_s=.1,
                         slow_motion='Actual TCP translation and actual aperture change below stated mm threshold, rotation below .01 degrees over a 100 ms adjacent interval. This is robot motion, not whole-scene stillness; a separate count also requires peg-tip translation below the threshold.',
                         target_displacement_baseline='Last target in preceding phase; actual displacement baseline is observation at start of current phase.',
                         onset_time='Elapsed time from the first observation/action pair of the phase. The first target is already a small step into the trajectory; its paired actual observation is before executing it, so actual displacement at elapsed zero is zero.',
                         repeats_by_phase='A repeated transition is assigned to the destination frame phase. Repeated intervals measure equal command duration, not observed physical stationarity.',
                         terminal_constant_duration='Includes the endpoint command itself; not solely the explicitly inserted tracking wait.'),
        limitations=['Near-static thresholds describe motion, not dispensable samples; submillimetre alignment and contact settling can be physically necessary.',
                     'Duplicate scalar trajectories do not prove identical camera pixels.',
                     'Causality for training requires a controlled retraining comparison.'],
        representatives=representatives)
    for key in ('0.01', '0.1'):
        count = sum(r['slow_motion_thresholds'][key]['qualifying_intervals']*position_counts[p]
                    for p, r in representatives.items())
        peg_count = sum(r['slow_motion_thresholds'][key]['robot_and_peg_tip_qualifying_intervals']*position_counts[p]
                        for p, r in representatives.items())
        doc['slow_motion_full_batch'][key] = dict(qualifying_intervals=count, fraction=count/total['transitions'],
            robot_and_peg_tip_qualifying_intervals=peg_count, robot_and_peg_tip_fraction=peg_count/total['transitions'])
    (args.output/'report.json').write_text(json.dumps(doc, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({k: v for k, v in doc.items() if k not in {'representatives','definitions','limitations','phases'}}, indent=2))


if __name__ == '__main__':
    main()
