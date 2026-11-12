"""Shared scheduling and comparable evidence without running GPU simulations."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from examples.insert_plug.policy import compare_generalization as comparison


@pytest.fixture
def inputs(tmp_path):
    checkpoints = {}
    for kind in ('act', 'diffusion'):
        path = tmp_path/kind/'artifacts/checkpoints/030000/pretrained_model'
        path.mkdir(parents=True)
        saved = (dict(type='act', chunk_size=64, n_action_steps=64, temporal_ensemble_coeff=None)
                 if kind == 'act' else dict(type='diffusion', horizon=64, n_obs_steps=1, n_action_steps=64,
                                           noise_scheduler_type='DDPM', num_train_timesteps=100,
                                           num_inference_steps=None, normalization_mapping={'ACTION': 'MIN_MAX'}))
        (path/'config.json').write_text(json.dumps(saved))
        (path/'model.safetensors').write_bytes(b'fixture')
        for name in ('policy_preprocessor.json', 'policy_postprocessor.json'):
            (path/name).write_text('{"steps": []}')
        checkpoints[kind] = path
    scene = tmp_path/'scene.yaml'
    scene.write_text(yaml.safe_dump(dict(task={'control_hz': 10}, physics={'timestep': .001},
                                         board={'source_xy': [0, 0], 'target_xy': [0, .1]})))
    manifest = tmp_path/'unseen.yaml'
    manifest.write_text(yaml.safe_dump(dict(split='unseen_v1', positions=[
        dict(name=f'pos{i}', source_xy=[.451+i*.001, -.061], target_xy=[.451+i*.001, .061])
        for i in range(10)])))
    policy = tmp_path/'policy.yaml'
    policy.write_text(yaml.safe_dump(dict(checkpoint=str(checkpoints['act']), scene_config=str(scene),
                                         positions_config=str(manifest), n_action_steps=50, duration=60.)))
    output = tmp_path/'comparison'
    argv = ['--act-checkpoint', str(checkpoints['act']), '--diffusion-checkpoint', str(checkpoints['diffusion']),
            '--positions-config', str(manifest), '--policy-config', str(policy), '--output', str(output),
            '--workers', '14']
    return dict(argv=argv, output=output, checkpoints=checkpoints, manifest=manifest)


def fake_execution(monkeypatch, *, runtime_error=False, crash=None):
    pools, submissions, audit_calls = [], [], []

    class Future:
        def __init__(self, function, config, index):
            self.args = (function, config, index)

        def result(self):
            function, config, index = self.args
            return function(config, index)

    class Pool:
        def __init__(self, *, max_workers):
            pools.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def submit(self, function, config, index):
            submissions.append((config['policy_type'], index))
            return Future(function, config, index)

    def run_position(config, index):
        kind = config['policy_type']
        if crash == (kind, index):
            raise RuntimeError('synthetic worker failure')
        # Exercise finalized-at timing: the priority model finishes before ACT.
        if kind == 'act' and index == 0:
            diffusion_path = Path(config['output']).parent/'diffusion/launch_summary.json'
            assert json.loads(diffusion_path.read_text())['completed_at'] is not None
        return dict(position=f'pos{index}', complete=True, episodes_completed=10,
                    physical_successes=7 if kind == 'diffusion' else 3,
                    returncode=2, completed_at='synthetic-completion')

    def audit(root, **kwargs):
        config = json.loads((Path(root)/'eval_config.json').read_text())
        kind = config['policy_type']
        audit_calls.append((kind, kwargs))
        entries = []
        for index in range(10):
            for repetition in range(10):
                success = repetition < (7 if kind == 'diffusion' else 3)
                reason = 'success' if success else 'timeout'
                if runtime_error and kind == 'act' and index == 9 and repetition == 9:
                    reason = 'runtime_error'
                entries.append(dict(position=f'pos{index}', repetition=repetition, seed=index*10+repetition,
                                    success=success, stop_reason=reason,
                                    simulation_seconds=20. if success else 60., wall_seconds=8. if success else 20.))
        successes = sum(row['success'] for row in entries)
        runtime_errors = sum(row['stop_reason'] == 'runtime_error' for row in entries)
        infrastructure = all(row['complete'] for row in config['process_results'])
        summary = dict(root=str(root), complete=True, audit_valid=True, infrastructure_ok=infrastructure,
                       audit_execution_ok=infrastructure and not runtime_errors, all_videos_fully_decoded=True,
                       episodes_completed=100, successes=successes, success_rate=successes/100,
                       timeouts=100-successes-runtime_errors, runtime_errors=runtime_errors)
        return summary, entries

    monkeypatch.setattr(comparison, 'ThreadPoolExecutor', Pool)
    monkeypatch.setattr(comparison, 'as_completed', lambda futures: list(futures))
    monkeypatch.setattr(comparison.evaluate, 'run_position', run_position)
    monkeypatch.setattr(comparison.summarize, 'summarize_batch', audit)
    return pools, submissions, audit_calls


def test_one_shared_pool_prioritizes_diffusion_and_comparison_retains_physical_failures(inputs, monkeypatch):
    pools, submissions, audits = fake_execution(monkeypatch)
    assert comparison.main(inputs['argv']) == 0
    assert pools == [14]
    assert submissions == [(kind, index) for kind in ('diffusion', 'act') for index in range(10)]
    assert [kind for kind, _ in audits] == ['diffusion', 'act']
    for kind, args in audits:
        assert args['video_check'] == 'full' and args['n_action_steps'] == 32 and args['duration'] == 60.
        assert args['positions_config'] == inputs['output']/kind/'configs/positions_config.yaml'
    report = json.loads((inputs['output']/'summary.json').read_text())
    assert report['comparison_ready'] and report['episodes_completed'] == 200
    assert report['model_results']['act']['successes'] == 30
    assert report['model_results']['diffusion']['successes'] == 70
    assert report['model_results']['act']['timeouts'] == 70
    row = report['positions'][0]
    assert row['success_rate_difference_diffusion_minus_act'] == pytest.approx(.4)
    assert row['act_mean_simulation_seconds'] == 48.
    assert row['diffusion_mean_simulation_seconds'] == 32.
    assert row['act_mean_success_simulation_seconds'] == row['diffusion_mean_success_simulation_seconds'] == 20.
    assert row['source_xy'] == [.451, -.061]
    assert (inputs['output']/'positions.csv').is_file()
    manifests = []
    for kind in ('act', 'diffusion'):
        config = json.loads((inputs['output']/kind/'eval_config.json').read_text())
        assert config['completed_at'] and len(config['process_results']) == 10
        assert [seed for p in config['positions'] for seed in p['seeds']] == list(range(100))
        assert config['n_action_steps'] == 32 and config['duration'] == 60.
        manifests.append(config['positions_manifest_sha256'])
    assert manifests[0] == manifests[1] == report['positions_manifest_sha256']
    assert report['act_temporal_ensemble_coeff'] is None


@pytest.mark.parametrize('existing', ['comparison', 'act', 'diffusion'])
def test_any_existing_output_is_preserved_before_other_directories_are_created(inputs, existing):
    root = inputs['output'].parent
    outputs = dict(comparison=inputs['output'], act=root/'act-output', diffusion=root/'diffusion-output')
    outputs[existing].mkdir()
    sentinel = outputs[existing]/'preserved.txt'
    sentinel.write_text('original')
    args = comparison.parse_args([*inputs['argv'], '--act-output', str(outputs['act']),
                                  '--diffusion-output', str(outputs['diffusion'])])
    with pytest.raises(FileExistsError, match='preserved'):
        comparison.prepare_comparison(args)
    assert sentinel.read_text() == 'original'
    assert all(not path.exists() for kind, path in outputs.items() if kind != existing)


@pytest.mark.parametrize('kind,updates,message', [
    ('act', {'type': 'diffusion'}, 'type='),
    ('diffusion', {'type': 'act'}, 'type='),
    ('act', {'temporal_ensemble_coeff': .01}, 'temporal ensembling disabled'),
    ('diffusion', {'n_obs_steps': 64}, 'n_action_steps'),
])
def test_checkpoint_guards_fail_before_mutating_outputs(inputs, kind, updates, message):
    path = inputs['checkpoints'][kind]/'config.json'
    config = json.loads(path.read_text())
    config.update(updates)
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match=message):
        comparison.prepare_comparison(comparison.parse_args(inputs['argv']))
    assert not inputs['output'].exists()


def test_prepared_models_cannot_use_different_manifest_hashes(inputs, monkeypatch):
    original = comparison.evaluate.prepare
    def altered(args):
        config = original(args)
        if config['policy_type'] == 'act':
            config['positions_manifest_sha256'] = 'wrong'
        return config
    monkeypatch.setattr(comparison.evaluate, 'prepare', altered)
    with pytest.raises(ValueError, match='preflight comparison contract'):
        comparison.prepare_comparison(comparison.parse_args(inputs['argv']))
    assert not (inputs['output']/'progress.json').exists()


@pytest.mark.parametrize('runtime_error,crash', [(True, None), (False, ('act', 4))])
def test_runtime_or_infrastructure_failure_is_not_successful_comparison(inputs, monkeypatch, runtime_error, crash):
    fake_execution(monkeypatch, runtime_error=runtime_error, crash=crash)
    assert comparison.main(inputs['argv']) == 1
    report = json.loads((inputs['output']/'summary.json').read_text())
    assert not report['comparison_ready'] and not report['execution_ok']
    assert report['model_results']['diffusion']['successes'] == 70
