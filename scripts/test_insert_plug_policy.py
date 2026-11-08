#!/usr/bin/env python3
"""Run 100 fixed-layout ACT evaluations in isolated headless subprocesses.

Run with the mujoco_policy environment's Python. Exit 0 means all 100 reports
were completed and validated; physical task success is reported separately.
Existing output directories are never overwritten or implicitly resumed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
EPISODES = 10
LOG_LIMIT = 4 * 1024 * 1024
ENVIRONMENT = dict(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   HF_HUB_OFFLINE='1', PYTHONUNBUFFERED='1')


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
                         encoding='utf-8')
    temporary.replace(path)


def workspace_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return dict(sha256=digest.hexdigest(), bytes=path.stat().st_size)


def child_environment():
    environment = os.environ.copy()
    environment.pop('PYTHONPATH', None)
    environment.update(ENVIRONMENT)
    return environment


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy-config', default='config/insert_plug_policy.yaml')
    parser.add_argument('--checkpoint', help='Override the configured checkpoint')
    parser.add_argument('--output', help='New directory; defaults to outputs/test/<training-run>')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--backend', choices=('egl', 'osmesa'), default='egl')
    parser.add_argument('--device', choices=('cuda', 'cpu', 'auto'), default='cuda')
    parser.add_argument('--n-action-steps', type=int, help='Default: policy YAML value')
    parser.add_argument('--duration', type=float, help='Default: policy YAML value')
    parser.add_argument('--stop-on-contact', action=argparse.BooleanOptionalAction, default=None,
                        help='Opt in to stopping at a diagnostic contact violation')
    parser.add_argument('--contact-diagnostics', action=argparse.BooleanOptionalAction, default=None,
                        help='Opt in to contact diagnostics without changing task success')
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error('--workers must be positive')
    return args


def prepare(args):
    """Validate inputs and freeze provenance before any simulation is launched."""
    policy_path = workspace_path(args.policy_config)
    policy = yaml.safe_load(policy_path.read_text(encoding='utf-8'))
    allowed = {'checkpoint', 'scene_config', 'positions_config', 'device',
               'n_action_steps', 'duration', 'position', 'stop_on_contact', 'contact_diagnostics'}
    if not isinstance(policy, dict) or set(policy) - allowed:
        raise ValueError(f'Invalid policy configuration: {policy_path}')
    requested = workspace_path(args.checkpoint or policy['checkpoint'])
    # This resolver validates the saved processor state files without loading ML.
    from mujoco_tools.lerobot_policy import resolve_checkpoint
    checkpoint = resolve_checkpoint(requested)
    step = checkpoint.parent if checkpoint.name == 'pretrained_model' else checkpoint
    training_run = (step.parent.parent.parent.name if step.parent.name == 'checkpoints'
                    and step.parent.parent.name == 'artifacts' else
                    f'{step.name}_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}')
    output = workspace_path(args.output or f'outputs/test/{training_run}')
    if output.exists():
        raise FileExistsError(f'Output already exists; preserved without changes: {output}')
    duration = args.duration if args.duration is not None else policy.get('duration', 60.)
    horizon = (args.n_action_steps if args.n_action_steps is not None else
               policy.get('n_action_steps', 10))
    stop_on_contact = (args.stop_on_contact if args.stop_on_contact is not None else
                       policy.get('stop_on_contact', False))
    contact_diagnostics = (args.contact_diagnostics if args.contact_diagnostics is not None else
                           policy.get('contact_diagnostics', False))
    if type(stop_on_contact) is not bool or type(contact_diagnostics) is not bool:
        raise ValueError('stop_on_contact and contact_diagnostics must be booleans')
    contact_diagnostics = contact_diagnostics or stop_on_contact
    contact_policy = 'stop' if stop_on_contact else 'diagnostic' if contact_diagnostics else 'off'
    saved = json.loads((checkpoint / 'config.json').read_text(encoding='utf-8'))
    if saved.get('type') != 'act' or type(horizon) is not int or not 1 <= horizon <= saved['chunk_size']:
        raise ValueError('An ACT checkpoint and n_action_steps within its chunk_size are required')
    if saved.get('temporal_ensemble_coeff') is not None and horizon != 1:
        raise ValueError('ACT temporal ensembling requires n_action_steps=1')
    if not math.isfinite(duration) or duration <= 0 or not math.isclose(duration * 10, round(duration * 10), abs_tol=1e-8):
        raise ValueError('duration must be positive and a multiple of 0.1 seconds')
    scene_path = workspace_path(policy.get('scene_config', 'config/insert_plug.yaml'))
    positions_path = workspace_path(policy.get('positions_config', 'config/insert_plug_positions.yaml'))
    sources = dict(policy_config=policy_path, scene_config=scene_path, positions_config=positions_path)
    source_bytes = {name: path.read_bytes() for name, path in sources.items()}
    scene = yaml.safe_load(source_bytes['scene_config'])
    manifest = yaml.safe_load(source_bytes['positions_config'])
    positions = manifest['positions']
    if [row['name'] for row in positions] != [f'pos{i}' for i in range(10)]:
        raise ValueError('Position manifest must contain pos0 through pos9 exactly, in order')
    for row in positions:
        for key in ('source_xy', 'target_xy'):
            if len(row[key]) != 2 or not all(math.isfinite(value) for value in row[key]):
                raise ValueError(f'Invalid fixed layout: {row["name"]}/{key}')
    if scene['task']['control_hz'] != 10 or scene['physics']['timestep'] != .001:
        raise ValueError('Evaluation requires 10 Hz control and 1 kHz physics')
    configuration = dict(
        format_version=1, started_at=timestamp(), completed_at=None,
        output=str(output), training_run=training_run, checkpoint=str(checkpoint),
        checkpoint_requested=str(requested), checkpoint_step=step.name,
        python=sys.executable, workers=args.workers, backend=args.backend, device=args.device,
        n_action_steps=horizon, duration=duration, duration_s=duration,
        episodes_per_position=EPISODES, episodes_requested=100,
        control_hz=10, physics_hz=1000, video=True, viewer=False,
        stop_on_contact=stop_on_contact, contact_diagnostics=contact_diagnostics,
        contact_policy=contact_policy, randomization=False, environment=ENVIRONMENT,
        unset_environment=['PYTHONPATH'],
        positions=[dict(name=row['name'], source_xy=row['source_xy'], target_xy=row['target_xy'],
                        seed_start=i * EPISODES, seeds=list(range(i * EPISODES, (i + 1) * EPISODES)))
                   for i, row in enumerate(positions)],
        checkpoint_files={str(path.relative_to(checkpoint)): fingerprint(path)
                          for path in sorted(checkpoint.rglob('*')) if path.is_file()},
        source_files={name: dict(path=str(path), **fingerprint(path)) for name, path in sources.items()},
        scene_snapshot=scene, policy_snapshot=policy, positions_snapshot=manifest,
        process_results=[],
    )
    configuration.update({name: str(path) for name, path in sources.items()})
    output.mkdir(parents=True, exist_ok=False)
    (output / 'configs').mkdir()
    for name, contents in source_bytes.items():
        (output / 'configs' / f'{name}.yaml').write_bytes(contents)
    write_json(output / 'eval_config.json', configuration)
    return configuration


def command_for(configuration, index):
    output = Path(configuration['output'])
    return [configuration['python'], '-m', 'examples.insert_plug', 'rollout',
            '--policy-config', str(output / 'configs/policy_config.yaml'),
            '--config', str(output / 'configs/scene_config.yaml'),
            '--positions-config', str(output / 'configs/positions_config.yaml'),
            '--checkpoint', configuration['checkpoint'], '--position', str(index),
            '--episodes', str(EPISODES), '--seed', str(index * EPISODES),
            '--backend', configuration['backend'], '--device', configuration['device'],
            '--n-action-steps', str(configuration['n_action_steps']),
            '--duration', str(configuration['duration']),
            '--stop-on-contact' if configuration['stop_on_contact'] else '--no-stop-on-contact',
            '--contact-diagnostics' if configuration['contact_diagnostics'] else '--no-contact-diagnostics',
            '--output', str(output / f'pos{index}'), '--quiet']


def validate_position(configuration, index):
    """Accept failed physical outcomes only when every requested record exists."""
    directory = Path(configuration['output']) / f'pos{index}'
    summary = json.loads((directory / 'summary.json').read_text(encoding='utf-8'))
    if not summary.get('complete') or summary.get('episodes_requested') != EPISODES or summary.get('episodes_completed') != EPISODES:
        raise ValueError(f'Incomplete summary: {directory}')
    if len(summary.get('results', [])) != EPISODES:
        raise ValueError(f'Expected {EPISODES} summary results: {directory}')
    expected_dirs = {f'episode_{i:06d}' for i in range(EPISODES)}
    if {path.name for path in directory.glob('episode_*')} != expected_dirs:
        raise ValueError(f'Unexpected episode directories: {directory}')
    layout = configuration['positions'][index]
    expected_scene = deepcopy(configuration['scene_snapshot'])
    expected_scene['board'].update(source_xy=layout['source_xy'], target_xy=layout['target_xy'])
    successes = 0
    for repeat, row in enumerate(summary['results']):
        folder = directory / f'episode_{repeat:06d}'
        report = json.loads((folder / 'result.json').read_text(encoding='utf-8'))
        expected = dict(episode_index=repeat, repetition=repeat, position=f'pos{index}',
                        seed=index * EPISODES + repeat, requested_duration_s=configuration['duration'],
                        stop_on_contact=configuration['stop_on_contact'],
                        contact_diagnostics=configuration['contact_diagnostics'],
                        contact_policy=configuration['contact_policy'],
                        control_hz=10, action_hold_s=.1, physics_timestep_s=.001,
                        scene='scene.xml', video='rollout.mp4')
        if any(report.get(key) != value for key, value in expected.items()) or report.get('config') != expected_scene:
            raise ValueError(f'Report settings differ: {folder}')
        contacts = report.get('contacts', {})
        if contacts.get('enabled') is not configuration['contact_diagnostics']:
            raise ValueError(f'Contact diagnostic mode differs: {folder}')
        if not configuration['contact_diagnostics'] and (
                report.get('contact_clean') is not None or contacts.get('unsafe_contact_count') is not None
                or contacts.get('checks') != 0 or report.get('contact_checks') != 0):
            raise ValueError(f'Disabled contact diagnostics must remain unmeasured: {folder}')
        policy = report.get('policy', {})
        if (policy.get('checkpoint') != configuration['checkpoint'] or
                policy.get('n_action_steps') != configuration['n_action_steps'] or
                policy.get('config_sha256') != configuration['checkpoint_files']['config.json']['sha256']):
            raise ValueError(f'Policy provenance differs: {folder}')
        if configuration['device'] != 'auto' and policy.get('device') != configuration['device']:
            raise ValueError(f'Policy device differs: {folder}')
        if report.get('stop_reason') in {'interrupted', 'viewer_closed'}:
            raise ValueError(f'Interrupted episode: {folder}')
        if (row.get('report') != f'{folder.name}/result.json' or
                row.get('position') != f'pos{index}' or row.get('episode_index') != repeat or
                row.get('success') != report.get('success')):
            raise ValueError(f'Summary/report mismatch: {folder}')
        for name in ('scene.xml', 'rollout.mp4'):
            if not (folder / name).is_file() or (folder / name).stat().st_size == 0:
                raise ValueError(f'Missing artifact: {folder / name}')
        successes += bool(report['success'])
    if summary.get('successes') != successes:
        raise ValueError(f'Summary success count differs: {directory}')
    return dict(episodes_completed=EPISODES, physical_successes=successes,
                physical_success_rate=successes / EPISODES)


def run_position(configuration, index):
    name = f'pos{index}'
    command = command_for(configuration, index)
    started = time.monotonic()
    result = dict(position=name, started_at=timestamp(), command=command,
                  returncode=None, complete=False, episodes_completed=0)
    print(json.dumps(dict(event='position_start', position=name, seed=index * EPISODES)), flush=True)
    tail = bytearray()
    try:
        with subprocess.Popen(command, cwd=ROOT, env=child_environment(),
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as process:
            while chunk := process.stdout.read(65536):
                tail.extend(chunk)
                if len(tail) > LOG_LIMIT:
                    del tail[:-LOG_LIMIT]
            result['returncode'] = process.wait()
        if result['returncode'] not in (0, 2):
            raise RuntimeError(f'Rollout process exited {result["returncode"]}')
        result.update(validate_position(configuration, index))
        result['complete'] = True
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        (Path(configuration['output']) / f'{name}.log').write_bytes(tail)
    result.update(completed_at=timestamp(), wall_seconds=time.monotonic() - started)
    print(json.dumps(dict(event='position_complete' if result['complete'] else 'position_failed',
                          **result)), flush=True)
    return result


def launch(configuration):
    output = Path(configuration['output'])
    results = []
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=configuration['workers']) as pool:
        futures = [pool.submit(run_position, configuration, index) for index in range(10)]
        for future in as_completed(futures):
            results.append(future.result())
            results.sort(key=lambda row: int(row['position'][3:]))
            summary = dict(started_at=configuration['started_at'], completed_at=None,
                           complete=len(results) == 10 and all(row['complete'] for row in results),
                           episodes_requested=100,
                           episodes_completed=sum(row['episodes_completed'] for row in results),
                           physical_successes=sum(row.get('physical_successes', 0) for row in results),
                           process_results=results, wall_seconds=time.monotonic() - started)
            write_json(output / 'launch_summary.json', summary)
    summary['completed_at'] = timestamp()
    configuration.update(completed_at=summary['completed_at'], process_results=results)
    write_json(output / 'eval_config.json', configuration)
    write_json(output / 'launch_summary.json', summary)
    print(json.dumps(dict(event='batch_complete' if summary['complete'] else 'batch_failed',
                          output=str(output), episodes_completed=summary['episodes_completed'],
                          physical_successes=summary['physical_successes'])), flush=True)
    return summary


def main(argv=None):
    summary = launch(prepare(parse_args(argv)))
    return 0 if summary['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
