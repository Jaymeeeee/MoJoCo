"""Exercise batch completion and failure handling without loading an ACT model."""
from copy import deepcopy
import io
import json
from pathlib import Path

import pytest
import yaml

from scripts import test_insert_plug_policy as launcher


@pytest.fixture
def configuration(tmp_path):
    model = tmp_path / 'outputs/train/example_run/artifacts/checkpoints/030000/pretrained_model'
    model.mkdir(parents=True)
    (model / 'config.json').write_text(json.dumps(dict(type='act', chunk_size=100)))
    (model / 'model.safetensors').write_bytes(b'synthetic model')
    for filename in ('policy_preprocessor.json', 'policy_postprocessor.json'):
        (model / filename).write_text('{"steps": []}')
    scene = tmp_path / 'scene.yaml'
    scene.write_text(yaml.safe_dump(dict(board=dict(source_xy=[0., 0.], target_xy=[0., .1]),
                                         task=dict(control_hz=10), physics=dict(timestep=.001))))
    positions = tmp_path / 'positions.yaml'
    positions.write_text(yaml.safe_dump(dict(positions=[
        dict(name=f'pos{i}', source_xy=[i * .01, 0.], target_xy=[.4, .1]) for i in range(10)])))
    policy = tmp_path / 'policy.yaml'
    policy.write_text(yaml.safe_dump(dict(checkpoint=str(model.parent), scene_config=str(scene),
                                          positions_config=str(positions), n_action_steps=10, duration=60.)))
    args = launcher.parse_args(['--policy-config', str(policy), '--output', str(tmp_path / 'results'),
                                '--workers', '3'])
    return launcher.prepare(args)


def write_position(configuration, index, count=10, success=False):
    directory = Path(configuration['output']) / f'pos{index}'
    directory.mkdir()
    layout = configuration['positions'][index]
    scene = deepcopy(configuration['scene_snapshot'])
    scene['board'].update(source_xy=layout['source_xy'], target_xy=layout['target_xy'])
    rows = []
    for repeat in range(count):
        folder = directory / f'episode_{repeat:06d}'
        folder.mkdir()
        report = dict(episode_index=repeat, repetition=repeat, position=f'pos{index}',
                      seed=index * 10 + repeat, requested_duration_s=60., config=scene,
                      stop_on_contact=True, control_hz=10, action_hold_s=.1, physics_timestep_s=.001,
                      scene='scene.xml', video='rollout.mp4', stop_reason='success' if success else 'timeout',
                      success=success, policy=dict(checkpoint=configuration['checkpoint'],
                      n_action_steps=10, device='cuda',
                      config_sha256=configuration['checkpoint_files']['config.json']['sha256']))
        (folder / 'result.json').write_text(json.dumps(report))
        (folder / 'scene.xml').write_text('<mujoco/>')
        (folder / 'rollout.mp4').write_bytes(b'synthetic media; decoding belongs to the audit')
        rows.append(dict(episode_index=repeat, position=f'pos{index}', success=success,
                         report=f'{folder.name}/result.json'))
    (directory / 'summary.json').write_text(json.dumps(dict(complete=count == 10,
        episodes_requested=10, episodes_completed=count, successes=count if success else 0, results=rows)))


def fake_process(monkeypatch, returncode=2):
    class Process:
        def __init__(self, command, **kwargs):
            self.stdout = io.BytesIO(b'policy process log\n')

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def wait(self):
            return returncode

    monkeypatch.setattr(launcher.subprocess, 'Popen', Process)


def test_freezes_configs_hashes_and_global_seed_blocks(configuration):
    assert configuration['training_run'] == 'example_run'
    assert [row['seeds'] for row in configuration['positions']] == [list(range(i * 10, i * 10 + 10)) for i in range(10)]
    assert configuration['checkpoint_files']['model.safetensors']['sha256']
    assert (Path(configuration['output']) / 'configs/scene_config.yaml').read_text() == Path(configuration['scene_config']).read_text()
    command = launcher.command_for(configuration, 9)
    assert command[0] == launcher.sys.executable
    for key, value in {'--position': '9', '--seed': '90', '--episodes': '10', '--backend': 'egl',
                       '--device': 'cuda', '--n-action-steps': '10', '--duration': '60.0'}.items():
        assert command[command.index(key) + 1] == value
    assert '--viewer' not in command and '--continue-on-contact' not in command


def test_child_environment_does_not_mutate_launcher(monkeypatch):
    monkeypatch.setenv('PYTHONPATH', '/wrong/environment')
    monkeypatch.setenv('OMP_NUM_THREADS', '96')
    child = launcher.child_environment()
    assert 'PYTHONPATH' not in child
    assert all(child[key] == '1' for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'HF_HUB_OFFLINE'))
    assert launcher.os.environ['OMP_NUM_THREADS'] == '96'
    assert launcher.os.environ['PYTHONPATH'] == '/wrong/environment'


def test_existing_output_is_preserved(configuration):
    path = Path(configuration['output']) / 'eval_config.json'
    original = path.read_bytes()
    args = launcher.parse_args(['--policy-config', configuration['policy_config'], '--output', configuration['output']])
    with pytest.raises(FileExistsError, match='preserved'):
        launcher.prepare(args)
    assert path.read_bytes() == original


def test_complete_failed_physical_outcomes_are_valid_runs(configuration, monkeypatch):
    write_position(configuration, 0)
    fake_process(monkeypatch, 2)
    result = launcher.run_position(configuration, 0)
    assert result['complete'] and result['episodes_completed'] == 10
    assert result['physical_successes'] == 0 and result['returncode'] == 2
    assert (Path(configuration['output']) / 'pos0.log').read_text() == 'policy process log\n'


@pytest.mark.parametrize('count,returncode', [(9, 2), (10, 1)])
def test_incomplete_or_crashed_process_is_never_accepted(configuration, monkeypatch, count, returncode):
    write_position(configuration, 0, count=count)
    fake_process(monkeypatch, returncode)
    result = launcher.run_position(configuration, 0)
    assert not result['complete'] and result['error']
    assert (Path(configuration['output']) / 'pos0/episode_000000/result.json').is_file()


@pytest.mark.parametrize('field,value', [('seed', 99), ('requested_duration_s', 30.), ('stop_reason', 'interrupted')])
def test_completion_checks_report_identity_and_settings(configuration, field, value):
    write_position(configuration, 0)
    path = Path(configuration['output']) / 'pos0/episode_000000/result.json'
    report = json.loads(path.read_text())
    report[field] = value
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        launcher.validate_position(configuration, 0)


def test_batch_separates_100_completions_from_physical_success(configuration, monkeypatch):
    for index in range(10):
        write_position(configuration, index, success=index == 0)
    fake_process(monkeypatch)
    summary = launcher.launch(configuration)
    assert summary['complete'] and summary['episodes_completed'] == 100
    assert summary['physical_successes'] == 10
    saved = json.loads((Path(configuration['output']) / 'eval_config.json').read_text())
    assert saved['completed_at'] and len(saved['process_results']) == 10
    assert [row['position'] for row in saved['process_results']] == [f'pos{i}' for i in range(10)]


def test_batch_writes_failure_manifest_without_erasing_partial_records(configuration, monkeypatch):
    for index in range(10):
        write_position(configuration, index, count=9 if index == 2 else 10)
    fake_process(monkeypatch)
    summary = launcher.launch(configuration)
    assert not summary['complete'] and summary['episodes_completed'] == 90
    saved = json.loads((Path(configuration['output']) / 'launch_summary.json').read_text())
    assert saved['process_results'][2]['position'] == 'pos2'
    assert saved['process_results'][2]['error']
    assert (Path(configuration['output']) / 'pos2/episode_000008/result.json').is_file()
