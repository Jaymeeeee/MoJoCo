"""Exercise batch completion and failure handling without loading policy models."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path

import pytest
import yaml

from examples.insert_plug.policy import evaluate as launcher


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
                      stop_on_contact=configuration['stop_on_contact'],
                      contact_diagnostics=configuration['contact_diagnostics'],
                      contact_policy=configuration['contact_policy'],
                      contact_clean=True if configuration['contact_diagnostics'] else None,
                      contact_checks=60000 if configuration['contact_diagnostics'] else 0,
                      contacts=dict(enabled=configuration['contact_diagnostics'],
                                    checks=60000 if configuration['contact_diagnostics'] else 0,
                                    unsafe_contact_count=0 if configuration['contact_diagnostics'] else None),
                      control_hz=10, action_hold_s=.1, physics_timestep_s=.001,
                      scene='scene.xml', video='rollout.mp4', stop_reason='success' if success else 'timeout',
                      success=success, policy=dict(checkpoint=configuration['checkpoint'],
                      **configuration['policy_settings'], device='cuda',
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
    assert '--no-stop-on-contact' in command and '--no-contact-diagnostics' in command
    assert configuration['contact_policy'] == 'off'
    assert configuration['stop_on_contact'] is False and configuration['contact_diagnostics'] is False


def test_positions_override_freezes_new_layouts_without_changing_policy_defaults(configuration):
    output = Path(configuration['output']).with_name('generalization')
    positions_path = output.with_name('generalization.yaml')
    manifest = deepcopy(configuration['positions_snapshot'])
    manifest['split'] = 'generalization_v1'
    for row in manifest['positions']:
        row['source_xy'][0] += .003
        row['target_xy'][1] += .005
    positions_path.write_text(yaml.safe_dump(manifest))
    original_manifest = positions_path.read_bytes()
    original_policy = Path(configuration['policy_config']).read_bytes()
    configured = launcher.prepare(launcher.parse_args([
        '--policy-config', configuration['policy_config'], '--positions-config', str(positions_path),
        '--output', str(output), '--n-action-steps', '32']))
    assert configured['positions_config'] == str(positions_path.resolve())
    assert configured['positions_split'] == 'generalization_v1'
    assert configured['positions_manifest_sha256'] == hashlib.sha256(original_manifest).hexdigest()
    assert configured['source_files']['positions_config']['sha256'] == configured['positions_manifest_sha256']
    assert configured['positions_snapshot'] == manifest
    assert configured['positions'][0]['source_xy'] == manifest['positions'][0]['source_xy']
    assert configured['positions'][9]['seeds'] == list(range(90, 100))
    assert Path(configuration['policy_config']).read_bytes() == original_policy
    frozen = output/'configs/positions_config.yaml'
    assert frozen.read_bytes() == original_manifest
    command = launcher.command_for(configured, 0)
    assert command[command.index('--positions-config')+1] == str(frozen)
    write_position(configured, 0)
    assert launcher.validate_position(configured, 0)['episodes_completed'] == 10
    positions_path.write_text('# edited after preparation\n'+original_manifest.decode())
    assert frozen.read_bytes() == original_manifest
    report_path = output/'pos0/episode_000000/result.json'
    report = json.loads(report_path.read_text())
    report['config']['board']['source_xy'] = configuration['positions'][0]['source_xy']
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='Report settings differ'):
        launcher.validate_position(configured, 0)


@pytest.mark.parametrize('flags,mode,stop,diagnostics', [
    (['--stop-on-contact'], 'stop', True, True),
    (['--contact-diagnostics'], 'diagnostic', False, True),
])
def test_contact_options_are_frozen_and_forwarded(configuration, flags, mode, stop, diagnostics):
    output = str(Path(configuration['output']).with_name(mode))
    args = launcher.parse_args(['--policy-config', configuration['policy_config'], '--output', output, *flags])
    configured = launcher.prepare(args)
    assert configured['contact_policy'] == mode
    assert configured['stop_on_contact'] is stop and configured['contact_diagnostics'] is diagnostics
    command = launcher.command_for(configured, 0)
    assert ('--stop-on-contact' if stop else '--no-stop-on-contact') in command
    assert '--contact-diagnostics' in command
    write_position(configured, 0)
    assert launcher.validate_position(configured, 0)['episodes_completed'] == 10


def test_policy_contact_defaults_can_be_explicitly_disabled(configuration):
    policy_path = Path(configuration['policy_config'])
    policy = yaml.safe_load(policy_path.read_text())
    policy.update(stop_on_contact=True, contact_diagnostics=True)
    policy_path.write_text(yaml.safe_dump(policy))
    args = launcher.parse_args(['--policy-config', str(policy_path), '--output',
                               str(Path(configuration['output']).with_name('disabled')),
                               '--no-stop-on-contact', '--no-contact-diagnostics'])
    configured = launcher.prepare(args)
    assert configured['contact_policy'] == 'off'


def test_disabled_contact_diagnostics_cannot_claim_zero_contacts(configuration):
    write_position(configuration, 0)
    path = Path(configuration['output'])/'pos0/episode_000000/result.json'
    report = json.loads(path.read_text())
    report['contacts']['unsafe_contact_count'] = 0
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='remain unmeasured'):
        launcher.validate_position(configuration, 0)


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


@pytest.mark.parametrize('kind,modality,heldout', [
    ('act', 'rgb', False), ('act', 'rgbd', True),
    ('diffusion', 'rgb', True), ('diffusion', 'rgbd', False),
])
def test_default_batch_name_uses_model_variant_split_and_local_test_start(
        configuration, monkeypatch, kind, modality, heldout):
    root = Path(configuration['policy_config']).parent
    monkeypatch.setattr(launcher, 'ROOT', root)
    monkeypatch.setattr(launcher, 'timestamp', lambda: '2026-11-11T16:30:45+00:00')
    model = Path(configuration['checkpoint'])
    old_run = model.parents[3]
    new_run = old_run.with_name(f'{kind}_insert_plug_v1_{modality}_20261108_154950')
    old_run.rename(new_run)
    model = new_run / model.relative_to(old_run)
    saved = dict(type=kind, chunk_size=64, horizon=64, n_obs_steps=1,
                 input_features={'observation.images.head': dict(type='VISUAL', shape=[3, 256, 256])})
    if modality == 'rgbd':
        saved['depth_camera_map'] = {'observation.images.head': 'observation.depth.head'}
    (model / 'config.json').write_text(json.dumps(saved))
    original_checkpoint = (model / 'config.json').read_bytes()
    policy_path = Path(configuration['policy_config'])
    policy = yaml.safe_load(policy_path.read_text())
    policy['checkpoint'] = str(model)
    policy_path.write_text(yaml.safe_dump(policy))
    manifest_path = Path(configuration['positions_config'])
    (root / 'config').mkdir()
    (root / 'config/insert_plug_positions.yaml').write_bytes(manifest_path.read_bytes())
    if heldout:
        manifest = yaml.safe_load(manifest_path.read_text())
        manifest['split'] = 'heldout_coordinate_range'
        for row in manifest['positions']:
            row['source_xy'][0] += .003
        manifest_path.write_text(yaml.safe_dump(manifest))
    args = launcher.parse_args(['--policy-config', str(policy_path)])
    configured = launcher.prepare(args)
    split = 'ood' if heldout else 'ind'
    assert configured['output'] == str(root / 'outputs/test' /
        f'{kind}_insert_plug_v1_{modality}_{split}_20261112_003045')
    assert configured['evaluation_split'] == split
    assert configured['output_name_timezone'] == 'Asia/Shanghai'
    assert configured['output_name_timestamp_source'] == 'started_at'
    assert configured['started_at'] == '2026-11-11T16:30:45+00:00'
    assert (model / 'config.json').read_bytes() == original_checkpoint
    with pytest.raises(FileExistsError, match='preserved'):
        launcher.prepare(args)


def test_unknown_layout_requires_explicit_split_for_default_name(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, 'ROOT', tmp_path)
    saved = dict(type='act', input_features={'observation.images.head': {}})
    manifest = dict(positions=[dict(source_xy=[0., 0.], target_xy=[.4, .1])])
    values = (saved, 'act_insert_plug_v1_rgb_20261108_154950', manifest)
    start = '2026-11-11T16:30:45+00:00'
    with pytest.raises(ValueError, match='specify --split or --output'):
        launcher.default_output_name(*values, None, start)
    name, split = launcher.default_output_name(*values, 'ood', start)
    assert name == 'act_insert_plug_v1_rgb_ood_20261112_003045'
    assert split == 'ood'


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


def diffusion_configuration(configuration, *, n_obs_steps=1, n_action_steps=32):
    saved = dict(type='diffusion', horizon=64, n_obs_steps=n_obs_steps, n_action_steps=64,
                 noise_scheduler_type='DDPM', num_train_timesteps=100, num_inference_steps=None,
                 normalization_mapping=dict(VISUAL='MEAN_STD', STATE='MIN_MAX', ACTION='MIN_MAX'))
    checkpoint = Path(configuration['checkpoint'])
    (checkpoint / 'config.json').write_text(json.dumps(saved))
    args = launcher.parse_args(['--policy-config', configuration['policy_config'],
                               '--output', str(Path(configuration['output']).with_name('diffusion')),
                               '--n-action-steps', str(n_action_steps)])
    return launcher.prepare(args), saved


def test_diffusion_execution_window_preserves_horizon_scheduler_and_seed_blocks(configuration):
    configured, saved = diffusion_configuration(configuration)
    assert configured['policy_type'] == 'diffusion'
    assert configured['checkpoint_config'] == saved
    assert configured['policy_settings']['horizon'] == 64
    assert configured['policy_settings']['n_obs_steps'] == 1
    assert configured['policy_settings']['num_inference_steps'] == 100
    assert configured['policy_settings']['checkpoint_num_inference_steps'] is None
    assert configured['policy_settings']['checkpoint_n_action_steps'] == 64
    assert configured['policy_settings']['n_action_steps'] == 32
    assert configured['episodes_requested'] == 100
    assert [seed for position in configured['positions'] for seed in position['seeds']] == list(range(100))
    command = launcher.command_for(configured, 9)
    assert command[command.index('--n-action-steps')+1] == '32'
    assert json.loads((Path(configured['checkpoint'])/'config.json').read_text()) == saved
    write_position(configured, 0)
    assert launcher.validate_position(configured, 0)['episodes_completed'] == 10


@pytest.mark.parametrize('n_obs_steps,n_action_steps', [(1, 65), (2, 64), (0, 32), (65, 1)])
def test_diffusion_rejects_unavailable_execution_window(configuration, n_obs_steps, n_action_steps):
    with pytest.raises(ValueError, match='Diffusion|diffusion'):
        diffusion_configuration(configuration, n_obs_steps=n_obs_steps, n_action_steps=n_action_steps)


@pytest.mark.parametrize('field,value', [('policy_type', 'act'), ('horizon', 32),
                                        ('num_inference_steps', 25), ('checkpoint_n_action_steps', 32)])
def test_diffusion_report_cannot_change_model_or_sampling_settings(configuration, field, value):
    configured, _ = diffusion_configuration(configuration)
    write_position(configured, 0)
    path = Path(configured['output'])/'pos0/episode_000000/result.json'
    report = json.loads(path.read_text())
    report['policy'][field] = value
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='temporal/model settings'):
        launcher.validate_position(configured, 0)
