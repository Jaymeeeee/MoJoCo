"""Evaluate ACT and Diffusion on one frozen layout manifest with a shared pool.

Each model retains the existing 100-episode evaluator and independent full-video
audit. Exit zero means complete, valid execution; physical task failures remain
reported outcomes. No output directory is overwritten or implicitly resumed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import yaml

from . import evaluate
from . import summarize


MODEL_ORDER = ('diffusion', 'act')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--act-checkpoint', required=True)
    parser.add_argument('--diffusion-checkpoint', required=True)
    parser.add_argument('--positions-config', required=True)
    parser.add_argument('--output', required=True, help='New comparison directory')
    parser.add_argument('--act-output', help='New ACT batch directory; defaults to OUTPUT/act')
    parser.add_argument('--diffusion-output', help='New Diffusion batch directory; defaults to OUTPUT/diffusion')
    parser.add_argument('--policy-config', default='config/insert_plug_policy.yaml')
    parser.add_argument('--workers', type=int, default=4, help='One shared worker limit across both models')
    parser.add_argument('--n-action-steps', type=int, default=32)
    parser.add_argument('--duration', type=float, default=60.)
    parser.add_argument('--device', choices=('cuda', 'cpu', 'auto'), default='cuda')
    parser.add_argument('--backend', choices=('egl', 'osmesa'), default='egl')
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error('--workers must be positive')
    return args


def preflight(args):
    """Read and validate critical inputs before claiming any output directory."""
    from mujoco_tools.lerobot_policy import resolve_checkpoint

    output = evaluate.workspace_path(args.output)
    model_outputs = {kind: evaluate.workspace_path(getattr(args, f'{kind}_output') or output/kind)
                     for kind in MODEL_ORDER}
    paths = [output, *model_outputs.values()]
    if len(set(paths)) != 3:
        raise ValueError('Comparison and model output directories must be distinct')
    for path in paths:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f'Output already exists; preserved without changes: {path}')
    for kind, path in model_outputs.items():
        other = model_outputs['act' if kind == 'diffusion' else 'diffusion']
        if output.is_relative_to(path) or other.is_relative_to(path):
            raise ValueError('Model outputs cannot contain the comparison or other model output directory')
        if path.is_relative_to(output/'configs'):
            raise ValueError('The comparison configs directory is reserved for frozen inputs')
    if (not math.isfinite(args.duration) or args.duration <= 0
            or not math.isclose(args.duration*10, round(args.duration*10), abs_tol=1e-8)):
        raise ValueError('duration must be positive and a multiple of 0.1 seconds')
    if type(args.workers) is not int or args.workers < 1:
        raise ValueError('workers must be a positive integer')
    checkpoints, checkpoint_configs = {}, {}
    for kind in MODEL_ORDER:
        path = resolve_checkpoint(evaluate.workspace_path(getattr(args, f'{kind}_checkpoint')))
        saved = json.loads((path/'config.json').read_text())
        if saved.get('type') != kind:
            raise ValueError(f'{kind} checkpoint must declare type={kind!r}')
        if kind == 'act' and saved.get('temporal_ensemble_coeff') is not None:
            raise ValueError('Paired comparison requires ACT temporal ensembling disabled in the checkpoint')
        evaluate.policy_settings(saved, args.n_action_steps)
        checkpoints[kind], checkpoint_configs[kind] = path, saved
    policy_path = evaluate.workspace_path(args.policy_config)
    policy = yaml.safe_load(policy_path.read_bytes())
    allowed = {'checkpoint', 'scene_config', 'positions_config', 'device', 'n_action_steps',
               'duration', 'position', 'stop_on_contact', 'contact_diagnostics'}
    if not isinstance(policy, dict) or set(policy)-allowed:
        raise ValueError(f'Invalid policy configuration: {policy_path}')
    scene_path = evaluate.workspace_path(policy.get('scene_config', 'config/insert_plug.yaml'))
    scene = yaml.safe_load(scene_path.read_bytes())
    if scene['task']['control_hz'] != 10 or scene['physics']['timestep'] != .001:
        raise ValueError('Comparison requires 10 Hz control and 1 kHz physics')
    positions_path = evaluate.workspace_path(args.positions_config)
    positions_bytes = positions_path.read_bytes()
    manifest = yaml.safe_load(positions_bytes)
    if [row['name'] for row in manifest['positions']] != [f'pos{i}' for i in range(10)]:
        raise ValueError('Manifest must contain pos0 through pos9 exactly, in order')
    for row in manifest['positions']:
        for key in ('source_xy', 'target_xy'):
            if len(row[key]) != 2 or not all(math.isfinite(value) for value in row[key]):
                raise ValueError(f'Invalid coordinate pair: {row["name"]}/{key}')
    return dict(output=output, model_outputs=model_outputs, checkpoints=checkpoints,
                checkpoint_configs=checkpoint_configs, policy_config=policy_path,
                positions_config=positions_path, positions_bytes=positions_bytes,
                positions_manifest_sha256=hashlib.sha256(positions_bytes).hexdigest(),
                positions_split=manifest.get('split'))


def prepare_comparison(args):
    planned = preflight(args)
    output = planned['output']
    output.mkdir(parents=True, exist_ok=False)
    (output/'configs').mkdir()
    manifest = output/'configs/positions_config.yaml'
    manifest.write_bytes(planned['positions_bytes'])
    configs = {}
    for kind in MODEL_ORDER:
        argv = ['--policy-config', str(planned['policy_config']),
                '--checkpoint', str(planned['checkpoints'][kind]),
                '--positions-config', str(manifest), '--output', str(planned['model_outputs'][kind]),
                '--workers', str(args.workers), '--n-action-steps', str(args.n_action_steps),
                '--duration', str(args.duration), '--device', args.device, '--backend', args.backend,
                '--no-stop-on-contact', '--no-contact-diagnostics']
        configs[kind] = evaluate.prepare(evaluate.parse_args(argv))
    for kind, config in configs.items():
        if (config['policy_type'] != kind
                or config['positions_manifest_sha256'] != planned['positions_manifest_sha256']
                or config['n_action_steps'] != args.n_action_steps or config['duration'] != args.duration):
            raise ValueError(f'{kind} configuration differs from the preflight comparison contract')
    for key in ('positions', 'positions_snapshot', 'scene_snapshot', 'contact_policy', 'device', 'backend'):
        if configs['act'][key] != configs['diffusion'][key]:
            raise ValueError(f'Model configurations must share identical {key}')
    configuration = dict(
        format_version=1, output=str(output), started_at=evaluate.timestamp(), completed_at=None,
        workers=args.workers, scheduling='single_shared_pool_diffusion_positions_first',
        job_order=[dict(model=kind, position=f'pos{i}') for kind in MODEL_ORDER for i in range(10)],
        positions_config_requested=str(planned['positions_config']), positions_config=str(manifest),
        positions_manifest_sha256=planned['positions_manifest_sha256'], positions_split=planned['positions_split'],
        layouts=configs['act']['positions'],
        n_action_steps=args.n_action_steps, duration_s=args.duration, control_hz=10, physics_hz=1000,
        device=args.device, backend=args.backend, act_temporal_ensemble_coeff=None,
        episodes_per_position=10, episodes_per_model=100, episodes_requested=200,
        models={kind: dict(checkpoint=configs[kind]['checkpoint'], output=configs[kind]['output'])
                for kind in MODEL_ORDER},
        comparison_note='Both models use the same fixed layouts and seed indices. Repetitions are not independent unseen layouts; report descriptive comparisons only.')
    evaluate.write_json(output/'comparison_config.json', configuration)
    return configuration, configs


def _save_model_progress(config, results, started):
    done = len(results) == 10
    if done and config.get('completed_at') is None:
        config['completed_at'] = evaluate.timestamp()
    config['process_results'] = sorted(results, key=lambda row: int(row['position'][3:]))
    summary = dict(started_at=config['started_at'], completed_at=config.get('completed_at'),
                   complete=done and all(row.get('complete') is True for row in results),
                   episodes_requested=100, episodes_completed=sum(row.get('episodes_completed', 0) for row in results),
                   physical_successes=sum(row.get('physical_successes', 0) for row in results),
                   process_results=config['process_results'], wall_seconds=time.monotonic()-started)
    evaluate.write_json(Path(config['output'])/'eval_config.json', config)
    evaluate.write_json(Path(config['output'])/'launch_summary.json', summary)
    return summary


def _mean(rows, key, *, successful_only=False):
    values = [row[key] for row in rows if (not successful_only or row.get('success') is True)
              and type(row.get(key)) in (int, float) and math.isfinite(row[key])]
    return float(statistics.mean(values)) if values else None


def make_comparison(configuration, audits, episode_rows):
    models = {}
    for kind in MODEL_ORDER:
        audit = audits[kind]
        rows = episode_rows[kind]
        models[kind] = dict(
            output=configuration['models'][kind]['output'], checkpoint=configuration['models'][kind]['checkpoint'],
            report=str(Path(configuration['models'][kind]['output'])/'summary.json'),
            complete=audit.get('complete') is True, audit_valid=audit.get('audit_valid') is True,
            infrastructure_ok=audit.get('infrastructure_ok') is True,
            audit_execution_ok=audit.get('audit_execution_ok') is True,
            all_videos_fully_decoded=audit.get('all_videos_fully_decoded') is True,
            episodes_completed=audit.get('episodes_completed', 0), successes=audit.get('successes', 0),
            success_rate=audit.get('success_rate'), timeouts=audit.get('timeouts', 0),
            runtime_errors=audit.get('runtime_errors', 0), audit_error=audit.get('audit_error'),
            mean_simulation_seconds=_mean(rows, 'simulation_seconds'),
            mean_success_simulation_seconds=_mean(rows, 'simulation_seconds', successful_only=True),
            mean_wall_seconds=_mean(rows, 'wall_seconds'))
    positions = []
    for index in range(10):
        position = f'pos{index}'
        layout = configuration['layouts'][index]
        row = dict(position=position, source_xy=layout['source_xy'], target_xy=layout['target_xy'])
        for kind in MODEL_ORDER:
            selected = [entry for entry in episode_rows[kind] if entry.get('position') == position]
            successes = sum(entry.get('success') is True for entry in selected)
            row.update({f'{kind}_episodes': len(selected), f'{kind}_successes': successes,
                        f'{kind}_success_rate': successes/len(selected) if selected else None,
                        f'{kind}_timeouts': sum(entry.get('stop_reason') == 'timeout' for entry in selected),
                        f'{kind}_runtime_errors': sum(entry.get('stop_reason') == 'runtime_error' for entry in selected),
                        f'{kind}_mean_simulation_seconds': _mean(selected, 'simulation_seconds'),
                        f'{kind}_mean_success_simulation_seconds': _mean(selected, 'simulation_seconds', successful_only=True),
                        f'{kind}_mean_wall_seconds': _mean(selected, 'wall_seconds')})
        row['success_rate_difference_diffusion_minus_act'] = (
            row['diffusion_success_rate']-row['act_success_rate']
            if row['diffusion_success_rate'] is not None and row['act_success_rate'] is not None else None)
        positions.append(row)
    complete = all(model['complete'] for model in models.values())
    audit_valid = all(model['audit_valid'] for model in models.values())
    execution_ok = all(model['infrastructure_ok'] and model['audit_execution_ok'] for model in models.values())
    videos_complete = all(model['all_videos_fully_decoded'] for model in models.values())
    return dict(**configuration, complete=complete, audit_valid=audit_valid, execution_ok=execution_ok,
                all_videos_fully_decoded=videos_complete,
                comparison_ready=complete and audit_valid and execution_ok and videos_complete,
                episodes_completed=sum(model['episodes_completed'] for model in models.values()),
                model_results=models, positions=positions,
                artifacts=dict(summary='summary.json', positions='positions.csv', configuration='comparison_config.json'))


def launch_comparison(configuration, configs):
    output = Path(configuration['output'])
    started = time.monotonic()
    results = {kind: [] for kind in MODEL_ORDER}
    progress = {kind: _save_model_progress(configs[kind], [], started) for kind in MODEL_ORDER}
    # Exactly one pool bounds all subprocesses; Diffusion jobs are enqueued first.
    with ThreadPoolExecutor(max_workers=configuration['workers']) as pool:
        futures = {pool.submit(evaluate.run_position, configs[kind], index): (kind, index)
                   for kind in MODEL_ORDER for index in range(10)}
        for future in as_completed(futures):
            kind, index = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = dict(position=f'pos{index}', returncode=None, complete=False, episodes_completed=0,
                              completed_at=evaluate.timestamp(), error=f'{type(exc).__name__}: {exc}')
            results[kind].append(result)
            progress[kind] = _save_model_progress(configs[kind], results[kind], started)
            evaluate.write_json(output/'progress.json', dict(started_at=configuration['started_at'],
                                completed_at=None, models=progress, wall_seconds=time.monotonic()-started))
    audits, rows = {}, {}
    for kind in MODEL_ORDER:
        config = configs[kind]
        print(json.dumps(dict(event='comparison_audit_start', model=kind, output=config['output'])), flush=True)
        try:
            audit, entries = summarize.summarize_batch(
                config['output'], checkpoint=config['checkpoint'],
                positions_config=Path(config['output'])/'configs/positions_config.yaml',
                video_check='full', n_action_steps=configuration['n_action_steps'], duration=configuration['duration_s'])
            summarize.write_reports(config['output'], audit, entries)
            audits[kind], rows[kind] = audit, entries
        except Exception as exc:
            audits[kind], rows[kind] = dict(audit_error=f'{type(exc).__name__}: {exc}'), []
            evaluate.write_json(Path(config['output'])/'audit_failure.json', audits[kind])
    configuration['completed_at'] = evaluate.timestamp()
    configuration['wall_seconds'] = time.monotonic()-started
    evaluate.write_json(output/'comparison_config.json', configuration)
    summary = make_comparison(configuration, audits, rows)
    evaluate.write_json(output/'summary.json', summary)
    with (output/'positions.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary['positions'][0]))
        writer.writeheader()
        writer.writerows(summary['positions'])
    evaluate.write_json(output/'progress.json', dict(started_at=configuration['started_at'],
                        completed_at=configuration['completed_at'], models=progress,
                        comparison_ready=summary['comparison_ready'], wall_seconds=configuration['wall_seconds']))
    print(json.dumps(dict(event='comparison_complete' if summary['comparison_ready'] else 'comparison_failed',
                          output=str(output), models=summary['model_results'])), flush=True)
    return summary


def main(argv=None):
    configuration, configs = prepare_comparison(parse_args(argv))
    summary = launch_comparison(configuration, configs)
    return 0 if summary['comparison_ready'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
