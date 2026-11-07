#!/usr/bin/env python3
"""Independently audit a fixed-position ACT batch; write only with --write.

Physical failures (including genuine timeouts and contact stops) and reported
policy runtime failures are evaluation outcomes. Artifact completeness, valid
execution, and worker-process failures are reported separately.
No simulator, policy, GPU, or training dataset is loaded by this script.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import yaml

ROOT = Path(__file__).resolve().parents[1]
POSITIONS = tuple(f'pos{i}' for i in range(10))
REPETITIONS = 10
PHYSICAL_REASONS = {'success', 'timeout', 'contact_violation',
                    'task_success_with_contact_violations'}
INFRASTRUCTURE_REASONS = {'runtime_error', 'interrupted', 'viewer_closed'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_json(path):
    def reject_constant(value):
        raise ValueError(f'Nonfinite JSON constant {value}')
    value = json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=reject_constant)
    require(isinstance(value, dict), 'JSON report must be an object')
    return value


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def timestamp(seconds):
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat()


def resolved_checkpoint(path):
    path = Path(path).expanduser().resolve()
    return path / 'pretrained_model' if (path / 'pretrained_model').is_dir() else path


def close(actual, expected, message, tolerance=1e-9):
    require(type(actual) in (int, float) and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=0, abs_tol=tolerance), message)


def regular_file(path):
    require(path.is_file() and not path.is_symlink() and path.stat().st_size > 0,
            f'Missing, empty, or nonregular artifact: {path.name}')


def check_video(path, expected_frames, width, height, mode='full'):
    """Full mode counts every decoded frame without copying RGB arrays to Python."""
    regular_file(path)
    evidence = dict(check=mode, bytes=path.stat().st_size, full_decode=False)
    if mode == 'none':
        return evidence
    import imageio_ffmpeg
    reader = imageio_ffmpeg.read_frames(str(path))
    try:
        metadata = next(reader)
    finally:
        reader.close()
    require(tuple(metadata['size']) == (width, height), 'Video dimensions differ from side-by-side cameras')
    close(metadata['fps'], 10, 'Video FPS must be 10', tolerance=.001)
    close(metadata['duration'], expected_frames / 10,
          'Video duration disagrees with reported frame count', tolerance=.11)
    evidence.update(width=width, height=height, fps=metadata['fps'], duration_s=metadata['duration'])
    if mode == 'full':
        process = subprocess.run([
            imageio_ffmpeg.get_ffmpeg_exe(), '-nostdin', '-v', 'error', '-xerror',
            '-threads', '1', '-i', str(path), '-map', '0:v:0', '-an',
            '-threads', '1', '-fps_mode', 'passthrough', '-f', 'null', '-',
            '-progress', 'pipe:1', '-nostats',
        ], capture_output=True, text=True, timeout=300, check=False)
        require(process.returncode == 0 and not process.stderr.strip(),
                f'Full video decode failed: {process.stderr.strip() or process.returncode}')
        counts = [int(line.split('=', 1)[1]) for line in process.stdout.splitlines()
                  if line.startswith('frame=')]
        require(counts and counts[-1] == expected_frames,
                f'Decoded video frame count {counts[-1] if counts else None} != {expected_frames}')
        require('progress=end' in process.stdout, 'Video decode did not reach end of stream')
        evidence.update(decoded_frames=counts[-1], full_decode=True)
    return evidence


def audit_episode(folder, result, *, position, repetition, checkpoint, layout,
                  video_check='full', n_action_steps=10, duration=60.0):
    """Check saved evidence without changing an episode or trusting group totals."""
    for key, expected in [('position', position), ('repetition', repetition),
                          ('episode_index', repetition),
                          ('seed', int(position[3:]) * REPETITIONS + repetition)]:
        require(result.get(key) == expected and type(result.get(key)) is type(expected),
                f'{key} must be {expected!r}, got {result.get(key)!r}')
    policy = result['policy']
    require(Path(policy['checkpoint']).expanduser().resolve() == checkpoint, 'Checkpoint path mismatch')
    require(policy.get('policy_type') == 'act', 'Expected ACT policy')
    require(type(policy.get('n_action_steps')) is int and policy['n_action_steps'] == n_action_steps,
            f'n_action_steps must be {n_action_steps}')
    config = result['config']
    for actual, expected, name in [
        (result.get('control_hz'), 10, 'control_hz'),
        (result.get('action_hold_s'), .1, 'action_hold_s'),
        (result.get('physics_timestep_s'), .001, 'physics_timestep_s'),
        (config['task']['control_hz'], 10, 'config control_hz'),
        (config['physics']['timestep'], .001, 'config timestep'),
        (result.get('requested_duration_s'), duration, 'requested_duration_s'),
    ]:
        close(actual, expected, f'{name} mismatch')
    for key in ('source_xy', 'target_xy'):
        actual = config['board'][key]
        require(isinstance(actual, list) and len(actual) == 2, f'{key} must be a coordinate pair')
        for value, expected in zip(actual, layout[key]):
            close(value, expected, f'{key} differs from vetted fixed position')
    require(result.get('stop_on_contact') is True, 'stop_on_contact must be true')
    for key in ('success', 'task_success', 'contact_clean'):
        require(type(result.get(key)) is bool, f'{key} must be a boolean')
    require(result['success'] == (result['task_success'] and result['contact_clean']),
            'success must combine task_success and contact_clean')
    contacts = result['contacts']
    unsafe = contacts['unsafe_contact_count']
    require(type(unsafe) is int and unsafe >= 0, 'Invalid unsafe_contact_count')
    require(result['contact_clean'] == (unsafe == 0), 'Contact count and contact_clean disagree')
    reason = result['stop_reason']
    require(reason in PHYSICAL_REASONS | INFRASTRUCTURE_REASONS, f'Unknown stop_reason {reason!r}')
    require(result['success'] == (reason == 'success'), 'Success flag and stop reason disagree')
    require(not result['task_success'] or reason in {'success', 'task_success_with_contact_violations'},
            'task_success and stop reason disagree')
    if reason == 'contact_violation':
        require(unsafe > 0, 'Contact stop has no unsafe contacts')
    if reason == 'runtime_error':
        require(bool(result.get('runtime_error')), 'Runtime error lacks diagnostic')
    elif reason not in {'contact_violation', 'interrupted'}:
        require(result.get('runtime_error') is None, 'Unexpected runtime_error diagnostic')
    for key in ('completed_control_steps', 'attempted_control_steps', 'selected_actions',
                'physics_checks', 'video_frames'):
        require(type(result.get(key)) is int and result[key] >= 0, f'Invalid {key}')
    completed, attempted = result['completed_control_steps'], result['attempted_control_steps']
    require(completed <= attempted <= result['selected_actions'], 'Control/action counts disagree')
    require(contacts.get('checks') == result['physics_checks'], 'Contact/physics check counts disagree')
    close(result['simulation_seconds'], result['physics_checks'] * .001,
          'Simulation duration and physics check count disagree', tolerance=1e-7)
    require(0 <= result['simulation_seconds'] <= duration + 1e-7, 'Simulation duration exceeds requested duration')
    if reason in PHYSICAL_REASONS:
        require(result['video_frames'] == attempted + 1, 'Video must include every attempted action and final view')
        require(result['policy'].get('calls') == result['selected_actions'], 'Policy call count disagrees')
    if reason == 'timeout':
        require(completed == attempted == round(duration * 10), 'Timeout did not finish requested control steps')
        close(result['simulation_seconds'], duration, 'Timeout stopped early', tolerance=1e-7)
    if result['task_success']:
        require(result['final_metrics']['success'] is True, 'Task success lacks successful final metrics')
        require(result['continuous_success_seconds'] + 1e-9 >= result['success_required_seconds'] >= .25,
                'Task success did not satisfy continuous stability requirement')
    require(result.get('scene') == 'scene.xml', 'Scene link must reference local scene.xml')
    scene_path = folder / 'scene.xml'
    regular_file(scene_path)
    scene = ET.parse(scene_path).getroot()
    require(scene.tag == 'mujoco', 'Scene XML is not a MuJoCo model')
    close(float(scene.find('option').get('timestep')), .001, 'XML timestep mismatch')
    for side in ('source', 'target'):
        body = scene.find(f'.//body[@name="{side}_socket"]')
        require(body is not None, f'XML lacks {side} holder')
        coordinates = [float(value) for value in body.get('pos', '').split()]
        require(len(coordinates) == 3, f'XML {side} holder position is invalid')
        for value, expected in zip(coordinates[:2], layout[f'{side}_xy']):
            close(value, expected, f'XML {side} holder differs from vetted position')
    require(result.get('video') == 'rollout.mp4', 'Video link must reference local rollout.mp4')
    require(result.get('video_layout') == 'head_left_wrist_right', 'Video layout mismatch')
    require(result['video_frames'] > 0, 'Video has no reported frames')
    video = check_video(folder / 'rollout.mp4', result['video_frames'],
                        int(config['cameras']['width']) * 2, int(config['cameras']['height']), video_check)
    # These describe repeated observed outcomes, not a proof of trajectory equality.
    outcome = {key: result.get(key) for key in (
        'success', 'task_success', 'contact_clean', 'stop_reason', 'simulation_seconds',
        'completed_control_steps', 'physics_checks', 'final_metrics', 'last_action', 'contacts')}
    return dict(video_audit=video, config_sha256=digest(config),
                scene_sha256=hashlib.sha256(scene_path.read_bytes()).hexdigest(),
                outcome_sha256=digest(outcome))


def check_position_summary(folder, rows, checkpoint):
    summary = load_json(folder / 'summary.json')
    require(summary.get('episodes_requested') == REPETITIONS, 'Position summary requested count mismatch')
    require(summary.get('episodes_completed') == len(rows), 'Position summary completed count mismatch')
    require(summary.get('complete') == (len(rows) == REPETITIONS), 'Position summary complete flag mismatch')
    require(summary.get('successes') == sum(row.get('success') is True for row in rows),
            'Position summary success count mismatch')
    require(Path(summary['checkpoint']).expanduser().resolve() == checkpoint, 'Position summary checkpoint mismatch')
    reported = summary['results']
    require(len(reported) == len(rows), 'Position summary row count mismatch')
    for saved, actual in zip(reported, rows):
        for key in ('episode_index', 'position', 'success', 'task_success', 'contact_clean',
                    'stop_reason', 'simulation_seconds', 'unsafe_contact_count'):
            require(saved.get(key) == actual.get(key), f'Position summary row {key} mismatch')
        require(saved.get('report') == str(Path(actual['report']).relative_to(folder.name)),
                'Position summary report link mismatch')
        require(saved.get('video') == str(Path(actual['video']).relative_to(folder.name)),
                'Position summary video link mismatch')


def process_failures(document, source):
    """Support launcher metadata without letting absent optional metadata pass as evidence."""
    failures = []
    for key in ('process_results', 'processes', 'jobs', 'results'):
        rows = document.get(key, [])
        if isinstance(rows, dict):
            rows = [dict(position=name, **value) if isinstance(value, dict)
                    else dict(position=name, returncode=value) for name, value in rows.items()]
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, dict):
                code = row.get('returncode', row.get('return_code', row.get('exit_code')))
                command = row.get('command', [])
                # The task's rollout CLI deliberately exits 2 when any physical
                # episode fails. Do not confuse that contract with process failure.
                outcome_exit = (code == 2 and isinstance(command, list)
                                and 'examples.insert_plug' in command and 'rollout' in command
                                and row.get('complete') is True and row.get('episodes_completed') == 10)
                if code is not None and code != 0 and not outcome_exit:
                    failures.append(f'{source}: {row.get("position", "worker")} process exit {code}')
                if row.get('error'):
                    failures.append(f'{source}: {row.get("position", "worker")}: {row["error"]}')
                if row.get('completed_at') and row.get('complete') is False:
                    failures.append(f'{source}: {row.get("position", "worker")} worker incomplete')
    for name, code in document.get('return_codes', {}).items():
        if code is not None and code != 0:
            failures.append(f'{source}: {name} process exit {code}')
    if source == 'launch_summary.json' and document.get('completed_at') and document.get('complete') is False:
        failures.append(f'{source}: launcher finished with an incomplete batch')
    return failures


def summarize_batch(root, *, checkpoint, positions_config=ROOT / 'config/insert_plug_positions.yaml',
                    video_check='full', n_action_steps=10, duration=60.0):
    """Read a batch and return (summary, rows); never write artifacts here."""
    root = Path(root).expanduser().resolve()
    require(root.is_dir() and not root.is_symlink(), f'Batch root does not exist: {root}')
    require(video_check in {'full', 'metadata', 'none'}, 'Unknown video audit mode')
    checkpoint = resolved_checkpoint(checkpoint)
    manifest = yaml.safe_load(Path(positions_config).read_text(encoding='utf-8'))
    layouts = {row['name']: row for row in manifest['positions']}
    require(set(layouts) == set(POSITIONS), 'Vetted manifest must contain exactly pos0 through pos9')
    errors, pending, infrastructure, rows, groups = [], [], [], [], []
    metadata = {}
    for name in ('eval_config.json', 'launch_summary.json'):
        path = root / name
        if path.exists():
            try:
                metadata[name] = load_json(path)
                infrastructure.extend(process_failures(metadata[name], name))
            except (ValueError, OSError, TypeError, AttributeError) as exc:
                errors.append(f'{name}: {exc}')
    checkpoint_config = checkpoint / 'config.json'
    checkpoint_config_sha256 = (hashlib.sha256(checkpoint_config.read_bytes()).hexdigest()
                                if checkpoint_config.is_file() else None)
    unexpected = sorted(path.name for path in root.glob('pos*') if path.is_dir() and path.name not in POSITIONS)
    if unexpected:
        errors.append(f'Unexpected position entries: {unexpected}')
    for position in POSITIONS:
        folder = root / position
        group_rows = []
        expected_names = {f'episode_{i:06d}' for i in range(REPETITIONS)}
        extra = sorted(path.name for path in folder.glob('episode_*') if path.name not in expected_names)
        if extra:
            errors.append(f'{position}: Unexpected episodes: {extra}')
        for repetition in range(REPETITIONS):
            episode = folder / f'episode_{repetition:06d}'
            report = episode / 'result.json'
            relative = str(report.relative_to(root))
            if not report.exists():
                pending.append(relative)
                continue
            row = dict(position=position, repetition=repetition, episode_index=repetition,
                       report=relative, scene=str((episode / 'scene.xml').relative_to(root)),
                       video=str((episode / 'rollout.mp4').relative_to(root)), audit_valid=False)
            try:
                regular_file(report)
                result = load_json(report)
                for key in ('seed', 'success', 'task_success', 'contact_clean', 'stop_reason',
                            'runtime_error', 'simulation_seconds', 'wall_seconds', 'video_frames',
                            'completed_control_steps', 'physics_checks'):
                    row[key] = result.get(key)
                row['unsafe_contact_count'] = result.get('contacts', {}).get('unsafe_contact_count')
                row['result_written_at'] = timestamp(report.stat().st_mtime)
                if row.get('stop_reason') in {'interrupted', 'viewer_closed'}:
                    infrastructure.append(f'{relative}: {row["stop_reason"]}: {row.get("runtime_error")}')
                if checkpoint_config_sha256 is not None:
                    require(result.get('policy', {}).get('config_sha256') == checkpoint_config_sha256,
                            'Policy config_sha256 differs from checkpoint config.json')
                row.update(audit_episode(episode, result, position=position, repetition=repetition,
                                         checkpoint=checkpoint, layout=layouts[position], video_check=video_check,
                                         n_action_steps=n_action_steps, duration=duration))
                row['audit_valid'] = True
            except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError,
                    ET.ParseError, subprocess.SubprocessError) as exc:
                row['audit_error'] = str(exc)
                errors.append(f'{relative}: {exc}')
            rows.append(row)
            group_rows.append(row)
        summary_path = folder / 'summary.json'
        if not summary_path.exists():
            pending.append(str(summary_path.relative_to(root)))
        else:
            try:
                regular_file(summary_path)
                check_position_summary(folder, group_rows, checkpoint)
            except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
                errors.append(f'{position}/summary.json: {exc}')
        for key in ('config_sha256', 'scene_sha256'):
            if len({row[key] for row in group_rows if key in row}) > 1:
                errors.append(f'{position}: {key} changed between fixed-position repetitions')
        outcome_hashes = {row['outcome_sha256'] for row in group_rows if 'outcome_sha256' in row}
        counts = totals(group_rows)
        groups.append(dict(position=position, **counts, source_xy=layouts[position]['source_xy'],
                           target_xy=layouts[position]['target_xy'],
                           unique_recorded_outcomes=len(outcome_hashes),
                           identical_recorded_outcomes=(len(group_rows) == REPETITIONS and
                                                       all(row['audit_valid'] for row in group_rows)
                                                       and len(outcome_hashes) == 1),
                           summary=str(summary_path.relative_to(root)),
                           reports=[row['report'] for row in group_rows]))
    infrastructure = sorted(set(infrastructure))
    audited = sum(row['audit_valid'] for row in rows)
    complete = len(rows) == 100 and not pending and not errors
    counts = totals(rows)
    written = sorted(row['result_written_at'] for row in rows if 'result_written_at' in row)
    report = dict(format_version=1, root=str(root), checkpoint=str(checkpoint),
                  checkpoint_config_sha256=checkpoint_config_sha256,
                  audited_at=datetime.now(timezone.utc).isoformat(),
                  started_at=metadata.get('eval_config.json', {}).get('started_at'),
                  completed_at=metadata.get('launch_summary.json', {}).get('completed_at'),
                  expected_episodes=100, expected_positions=list(POSITIONS), repetitions_per_position=10,
                  complete=complete, reports_complete=len(rows) == 100,
                  audit_valid=not errors, infrastructure_ok=not infrastructure,
                  audit_execution_ok=not infrastructure and counts['runtime_errors'] == 0,
                  success=complete and counts['successes'] == 100,
                  **counts, episodes_audited=audited,
                  control_hz=10, action_hold_s=.1, physics_timestep_s=.001,
                  n_action_steps=n_action_steps, requested_duration_s=duration,
                  video_check=video_check,
                  all_videos_fully_decoded=(audited == 100 and all(
                      row.get('video_audit', {}).get('full_decode') is True for row in rows)),
                  fixed_positions=True, randomized_initial_conditions=False,
                  repetition_note=('Each position repeats the same fixed initial scene with seeds 10*i+j. '
                                   'Identical recorded outcomes are reported explicitly; these repeats do not '
                                   'measure robustness to randomized initial conditions or prove identical full trajectories.'),
                  first_result_written_at=written[0] if written else None,
                  last_result_written_at=written[-1] if written else None,
                  positions=groups, errors=errors, missing_artifacts=pending,
                  infrastructure_failures=infrastructure,
                  metadata=metadata, artifacts=dict(summary='summary.json', episodes='episodes.csv'))
    return report, rows


def totals(rows):
    reasons = Counter(row.get('stop_reason') if isinstance(row.get('stop_reason'), str)
                      else 'unreadable_report' for row in rows)
    return dict(episodes_completed=len(rows), successes=sum(row.get('success') is True for row in rows),
                task_successes=sum(row.get('task_success') is True for row in rows),
                contact_clean=sum(row.get('contact_clean') is True for row in rows),
                success_rate=sum(row.get('success') is True for row in rows) / len(rows) if rows else None,
                stop_reasons=dict(sorted(reasons.items())), timeouts=reasons['timeout'],
                runtime_errors=reasons['runtime_error'], contact_stops=reasons['contact_violation'],
                contact_violation_episodes=sum(row.get('contact_clean') is False for row in rows),
                unsafe_contact_count=sum(row['unsafe_contact_count'] for row in rows
                                         if type(row.get('unsafe_contact_count')) is int
                                         and row['unsafe_contact_count'] >= 0))


def write_reports(root, summary, rows):
    """Called only by the explicit --write CLI; never alter source reports."""
    root = Path(root).resolve()
    require(str(root) == summary['root'], 'Report destination must be the audited batch root')
    fields = ['position', 'repetition', 'episode_index', 'seed', 'success', 'task_success', 'contact_clean',
              'stop_reason', 'runtime_error', 'simulation_seconds', 'wall_seconds', 'completed_control_steps',
              'physics_checks', 'unsafe_contact_count', 'video_frames', 'result_written_at', 'audit_valid',
              'audit_error', 'report', 'scene', 'video', 'config_sha256', 'scene_sha256', 'outcome_sha256']
    for name in ('summary.json', 'episodes.csv'):
        require(not (root / name).is_symlink(), f'Refusing to overwrite symlink {name}')
    with (root / 'episodes.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    (root / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False,
                                                 allow_nan=False) + '\n', encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--positions-config', type=Path, default=ROOT / 'config/insert_plug_positions.yaml')
    parser.add_argument('--n-action-steps', type=int, default=10)
    parser.add_argument('--duration', type=float, default=60)
    parser.add_argument('--video-check', choices=('full', 'metadata', 'none'), default='full')
    parser.add_argument('--allow-incomplete', action='store_true',
                        help='Allow missing in-progress reports, but keep complete=false; other errors still fail')
    parser.add_argument('--write', action='store_true', help='Write summary.json and episodes.csv inside --root')
    args = parser.parse_args(argv)
    try:
        summary, rows = summarize_batch(args.root, checkpoint=args.checkpoint,
                                       positions_config=args.positions_config, video_check=args.video_check,
                                       n_action_steps=args.n_action_steps, duration=args.duration)
        if args.write:
            write_reports(args.root, summary, rows)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f'Audit failed: {exc}\n')
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    accepted = (summary['audit_valid'] and summary['infrastructure_ok']
                and (summary['complete'] or args.allow_incomplete))
    return 0 if accepted else 1


if __name__ == '__main__':
    raise SystemExit(main())
