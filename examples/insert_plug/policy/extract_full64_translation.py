"""Recover full initial forecasts for the physically qualified pos2 experiment.

ACT reuses saved 64-step predictions. Diffusion is sampled on saved RGB-D/state
inputs with a passive hook before its n32 return slice; no simulator steps are
executed. Seed20 must reproduce the qualified experiment's saved first chunk.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import json

import numpy as np

from .translation_probe import forward_kinematics, sha256, write_json


CONDITIONS = ('baseline', 'x_plus', 'x_minus', 'y_plus', 'y_minus')


def rng(torch):
    return dict(cpu=torch.get_rng_state().cpu().numpy().copy(),
                cuda=np.asarray([v.cpu().numpy().copy() for v in torch.cuda.get_rng_state_all()],
                                dtype=np.uint8))


def capture(adapter, observation, seed):
    from .rollout import _seed_policy
    adapter.reset()
    _seed_policy(seed)
    module = adapter.policy.diffusion
    original = module.conditional_sample
    samples = []

    def sample(*args, **kwargs):
        result = original(*args, **kwargs)
        samples.append(result.detach().cpu().numpy().copy())
        return result

    start = rng(adapter._torch)
    module.conditional_sample = sample
    try:
        selected = adapter.select_action(observation)
    finally:
        module.conditional_sample = original
    end = rng(adapter._torch)
    if len(samples) != 1 or samples[0].shape != (1, 64, 8):
        raise ValueError('Expected exactly one complete 64-step conditional sample')
    normalized = samples[0][0]
    with adapter._torch.inference_mode():
        actions = adapter.postprocessor(adapter._torch.from_numpy(normalized.copy())).cpu().numpy().copy()
    np.testing.assert_array_equal(actions[0], selected)
    if not np.isfinite(actions).all():
        raise ValueError('Nonfinite full forecast')
    return dict(actions=actions, normalized=normalized, rng_start=start['cpu'], rng_end=end['cpu'],
                cuda_rng_start=start['cuda'], cuda_rng_end=end['cuda'])


def run(source, output):
    import torch
    from ..environment.simulation import Simulation
    from mujoco_tools.lerobot_policy import LeRobotPolicy

    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    summary = json.loads((source/'summary.json').read_text())
    if summary['status'] != 'complete' or not summary['baseline_gate_passed']:
        raise ValueError('Source experiment must have physically qualified baselines')
    output.mkdir(parents=True, exist_ok=True)
    if (output/'manifest.json').exists() or (output/'data').exists():
        raise FileExistsError('Preserving previously extracted forecasts')
    paths = [source/'summary.json', source/'protocol.json', source/'audit.json']
    for condition in CONDITIONS:
        paths += [source/'scenes/pos2'/f'{condition}{ext}' for ext in ('.npz', '.json', '.xml')]
        paths += [source/'episodes'/m/condition/'trace.npz' for m in ('act', 'diffusion')]
    checkpoint = Path(summary['episodes']['diffusion']['baseline']['policy']['checkpoint'])
    paths += [p for p in checkpoint.iterdir() if p.is_file()]
    protected = {str(p): sha256(p) for p in paths}
    manifest = dict(created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        status='running', source_experiment=str(source), position='pos2',
        source_baseline_physical_seed=20, n_action_steps=32, horizon=64, n_obs_steps=1,
        conditions=list(CONDITIONS), seeds={'act': [20], 'diffusion': list(range(20, 25))},
        model_inferences={'act': 0, 'diffusion': 25}, new_physical_rollouts=0,
        act_method='Reuse saved first 64-action normalized/physical chunk and FK, without inference.',
        diffusion_method='Passive conditional_sample hook in n32 adapter; exact original saved RGB-D/state; no n_action_steps or sampler changes.',
        coordinates='World-frame TCP xyz in metres from individual absolute joint targets through FK.',
        statistics='At each query/condition/coordinate independently, mean and observed min/max across listed seeds. ACT has n=1, so no variability band.',
        limitations=['DP seed20 reproduces the physically qualified rollout; other seeds are inference samples only.',
                     'Initial-observation 64-step forecasts are not the replanned 6.4-second trajectory. Only q0..31 of the first chunk were executed in the n32 experiment.',
                     'Seed spread is not a confidence interval or variation across layouts.',
                     'These predicted absolute positions are not measured robot motion.'],
        source_sha256=protected, extraction_source_sha256=sha256(Path(__file__)),
        seed20_verification={}, data={})
    write_json(output/'manifest.json', manifest)

    for model in ('act', 'diffusion'):
        (output/'data'/model).mkdir(parents=True)
    adapter = LeRobotPolicy(checkpoint, device='cuda', n_action_steps=32)
    if adapter.n_obs_steps != 1 or adapter.horizon != 64:
        raise ValueError('Expected current RGB-D n_obs_steps=1, horizon=64 checkpoint')
    base_rng = None
    for condition in CONDITIONS:
        scene_path = source/'scenes/pos2'/f'{condition}.xml'
        meta = json.loads((source/'scenes/pos2'/f'{condition}.json').read_text())
        with np.load(source/'scenes/pos2'/f'{condition}.npz') as saved:
            observation = {key: saved[key].copy() for key in saved.files if key.startswith('observation.')}
            source_xyz = saved['peg'].copy()
            delta = saved['delta'].copy()
            initial_qpos = saved['qpos'].copy()
        sim = Simulation(scene_path, meta['config'], render=False)
        np.testing.assert_array_equal(sim.data.qpos, initial_qpos)
        try:
            with np.load(source/'episodes/act'/condition/'trace.npz') as saved:
                actions = saved['chunk_action'][0].copy()
                normalized = saved['chunk_normalized_action'][0].copy()
                tcp = saved['chunk_target_tcp'][0].copy()
                fresh, _ = forward_kinematics(sim, actions)
                np.testing.assert_array_equal(tcp, fresh)
            act_path = output/'data/act'/f'{condition}.npz'
            np.savez_compressed(act_path, actions=actions[None], normalized=normalized[None],
                                tcp=tcp[None], seeds=np.asarray([20]), delta=delta,
                                source_xyz=source_xyz, initial_qpos=initial_qpos)

            values = []
            for seed in range(20, 25):
                prediction = capture(adapter, observation, seed)
                tcp, rotation = forward_kinematics(sim, prediction['actions'])
                prediction.update(tcp=tcp, tcp_rotation=rotation)
                if seed == 20:
                    with np.load(source/'episodes/diffusion'/condition/'trace.npz') as saved:
                        np.testing.assert_array_equal(prediction['normalized'][:32], saved['chunk_normalized_action'][0])
                        np.testing.assert_array_equal(prediction['actions'][:32], saved['chunk_action'][0])
                        np.testing.assert_array_equal(prediction['tcp'][:32], saved['chunk_target_tcp'][0])
                        for phase in ('start', 'end'):
                            np.testing.assert_array_equal(prediction[f'rng_{phase}'], saved[f'chunk_rng_cpu_{phase}'][0])
                            np.testing.assert_array_equal(prediction[f'cuda_rng_{phase}'], saved[f'chunk_rng_cuda_{phase}'][0])
                    manifest['seed20_verification'][condition] = dict(normalized_first32_exact=True,
                        physical_first32_exact=True, fk_first32_exact=True, full_cpu_cuda_rng_exact=True)
                values.append(prediction)
            data = {key: np.asarray([row[key] for row in values]) for key in values[0]}
            if base_rng is None:
                base_rng = {key: data[key].copy() for key in ('rng_start', 'rng_end', 'cuda_rng_start', 'cuda_rng_end')}
            else:
                for key, expected in base_rng.items():
                    np.testing.assert_array_equal(data[key], expected)
            dp_path = output/'data/diffusion'/f'{condition}.npz'
            np.savez_compressed(dp_path, **data, seeds=np.arange(20, 25), delta=delta,
                                source_xyz=source_xyz, initial_qpos=initial_qpos)
            manifest['data'][condition] = dict(act=str(act_path.relative_to(output)),
                diffusion=str(dp_path.relative_to(output)), source_xyz_m=source_xyz.tolist(),
                delta_m=delta.tolist(), full_rng_paired=True)
            write_json(output/'manifest.json', manifest)
            print(json.dumps(dict(event='condition_complete', condition=condition,
                                  act_predictions_reused=1, diffusion_inference_samples=5)), flush=True)
        finally:
            sim.close()
    manifest['protected_files_unchanged'] = all(sha256(Path(p)) == h for p, h in protected.items())
    if not manifest['protected_files_unchanged']:
        raise RuntimeError('Source artifact or checkpoint changed')
    manifest['status'] = 'complete'
    write_json(output/'manifest.json', manifest)
    print(json.dumps(dict(event='complete', new_physical_rollouts=0, new_model_inferences=25)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run(args.source.resolve(), args.output.resolve())


if __name__ == '__main__':
    main()
