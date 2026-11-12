"""Adapter contracts; model weights are exercised by the real rollout smoke run."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

import mujoco_tools.lerobot_policy as adapter


def checkpoint(tmp_path):
    path = tmp_path / '030000' / 'pretrained_model'
    path.mkdir(parents=True)
    config = dict(type='act', chunk_size=4, n_action_steps=4,
                  pretrained_backbone_weights='ImageNet', temporal_ensemble_coeff=None,
                  input_features={
                      'observation.images.head': {'type': 'VISUAL', 'shape': [3, 2, 4]},
                      'observation.state': {'type': 'STATE', 'shape': [8]}},
                  output_features={'action': {'type': 'ACTION', 'shape': [8]}})
    (path / 'config.json').write_text(json.dumps(config))
    (path / 'model.safetensors').write_bytes(b'fixture')
    for prefix in ('pre', 'post'):
        (path / f'{prefix}_stats.safetensors').write_bytes(b'fixture')
        (path / f'policy_{prefix}processor.json').write_text(json.dumps({
            'steps': [{'registry_name': 'normalizer_processor',
                       'state_file': f'{prefix}_stats.safetensors'}]}))
    return path


def test_resolves_both_checkpoint_directory_forms(tmp_path):
    path = checkpoint(tmp_path)
    assert adapter.resolve_checkpoint(path) == path
    assert adapter.resolve_checkpoint(path.parent) == path


def test_missing_normalization_statistics_fail_before_import(tmp_path):
    path = checkpoint(tmp_path)
    (path / 'pre_stats.safetensors').unlink()
    with pytest.raises(FileNotFoundError, match='saved processor statistics'):
        adapter.LeRobotPolicy(path)


def test_checkpoint_statistics_cannot_escape_checkpoint(tmp_path):
    path = checkpoint(tmp_path)
    (path / 'policy_preprocessor.json').write_text(json.dumps({
        'steps': [{'state_file': '../outside.safetensors'}]}))
    with pytest.raises(ValueError, match='escapes checkpoint'):
        adapter.resolve_checkpoint(path)


@pytest.mark.parametrize('update,match', [
    ({'type': 'unknown'}, 'supports ACT and Diffusion'),
    ({'output_features': {}}, 'output feature'),
    ({'input_features': {'privileged.target': {'type': 'STATE', 'shape': [3]}}}, 'Unsupported policy feature'),
    ({'input_features': {'observation.images.head': {'type': 'VISUAL', 'shape': [1, 2, 4]}}}, 'Unsupported policy feature'),
])
def test_unsupported_checkpoint_contracts_fail_early(tmp_path, update, match):
    path = checkpoint(tmp_path)
    cfg = json.loads((path / 'config.json').read_text())
    cfg.update(update)
    (path / 'config.json').write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match=match):
        adapter.LeRobotPolicy(path)


@pytest.mark.parametrize('steps', [0, -1, 5, 1.5, True])
def test_execution_prefix_must_fit_chunk(tmp_path, steps):
    with pytest.raises(ValueError, match='n_action_steps'):
        adapter.LeRobotPolicy(checkpoint(tmp_path), n_action_steps=steps)


@pytest.fixture
def fake_policy(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    path = checkpoint(tmp_path)
    calls = {}

    class Config:
        @classmethod
        def from_pretrained(cls, folder, **kwargs):
            assert kwargs == {'local_files_only': True}
            saved = json.loads((folder / 'config.json').read_text())
            saved.setdefault('depth_camera_map', {})
            saved.setdefault('depth_unit', 'm')
            return SimpleNamespace(**saved)

    class Policy:
        resets = 0

        @classmethod
        def from_pretrained(cls, folder, config, **kwargs):
            calls['config'] = deepcopy(config)
            assert kwargs == {'local_files_only': True, 'strict': True}
            return cls()

        def eval(self):
            calls['eval'] = True

        def reset(self):
            self.resets += 1
            self.pending = []

        def select_action(self, batch):
            assert torch.is_inference_mode_enabled()
            calls['normalized_batch'] = deepcopy(batch)
            if not self.pending:
                self.pending = [torch.full((1, 8), float(i)) for i in (1, 2)]
            return self.pending.pop(0)

    class Processor:
        resets = 0

        def __init__(self, prefix):
            self.prefix = prefix

        def reset(self):
            self.resets += 1

        def __call__(self, value):
            if self.prefix == 'pre':
                calls['raw_batch'] = deepcopy(value)
                # Emulate saved image/state normalization and batching.
                return {key: (tensor.unsqueeze(0) - .5) / .25 for key, tensor in value.items()}
            # Emulate saved action mean 3/std 2; must happen exactly once.
            return value * 2 + 3

    def processors(folder, **kwargs):
        calls['processors'] = kwargs
        return Processor('pre'), Processor('post')

    monkeypatch.setattr(adapter, '_load_backend', lambda: (
        torch, SimpleNamespace(__version__='fixture', __file__='fixture'), Config, Policy, processors))
    instance = adapter.LeRobotPolicy(path.parent, device='cpu', n_action_steps=2)
    return instance, calls, path


def observation():
    rgb = np.zeros((2, 4, 3), dtype=np.uint8)
    rgb[..., 0], rgb[..., 1], rgb[..., 2] = 255, 128, 0
    return {'observation.images.head': rgb, 'observation.state': np.arange(8, dtype=np.float64),
            'privileged.target_pose': np.ones(7), 'observation.depth.head': np.ones((2, 4))}


def test_images_scale_once_and_saved_processors_are_applied(fake_policy):
    instance, calls, _ = fake_policy
    obs = observation()
    before = deepcopy(obs)
    action = instance.select_action(obs)
    batch = calls['raw_batch']
    assert set(batch) == set(instance.input_features)
    assert tuple(batch['observation.images.head'].shape) == (3, 2, 4)
    np.testing.assert_allclose(batch['observation.images.head'][:, 0, 0].numpy(), [1, 128/255, 0])
    np.testing.assert_allclose(calls['normalized_batch']['observation.images.head'][0, :, 0, 0].numpy(),
                               [2, (128/255-.5)/.25, -2], rtol=1e-5, atol=2e-7)
    assert action.dtype == np.float32 and action.shape == (8,)
    np.testing.assert_array_equal(action, np.full(8, 5, np.float32))
    for key in obs:
        np.testing.assert_array_equal(obs[key], before[key])


def test_in_memory_overrides_do_not_modify_checkpoint(fake_policy):
    instance, calls, path = fake_policy
    cfg = calls['config']
    assert cfg.device == 'cpu' and cfg.n_action_steps == 2 and cfg.chunk_size == 4
    assert cfg.pretrained_backbone_weights is None
    assert calls['processors'] == {
        'preprocessor_overrides': {'device_processor': {'device': 'cpu'}},
        'postprocessor_overrides': {'device_processor': {'device': 'cpu'}}}
    saved = json.loads((path / 'config.json').read_text())
    assert saved['n_action_steps'] == 4 and saved['pretrained_backbone_weights'] == 'ImageNet'
    assert calls['eval']
    json.dumps(instance.metadata())


@pytest.mark.parametrize('saved_coeff,override,expected', [
    (None, None, None),
    (None, 0, 0.0),
    (None, 0.01, 0.01),
    (0.2, None, 0.2),
    (0.2, 0, 0.0),
    (0.2, -0.1, -0.1),
])
def test_temporal_ensemble_override_reaches_model_without_changing_checkpoint(
        fake_policy, saved_coeff, override, expected):
    _, calls, path = fake_policy
    saved = json.loads((path / 'config.json').read_text())
    saved['temporal_ensemble_coeff'] = saved_coeff
    (path / 'config.json').write_text(json.dumps(saved))
    before = {file.name: file.read_bytes() for file in path.iterdir()}
    instance = adapter.LeRobotPolicy(
        path, device='cpu', n_action_steps=1, temporal_ensemble_coeff=override)
    assert calls['config'].temporal_ensemble_coeff == expected
    assert calls['config'].n_action_steps == 1
    assert instance.temporal_ensemble_coeff == expected
    assert instance.metadata()['temporal_ensemble_coeff'] == expected
    assert instance.metadata()['checkpoint_temporal_ensemble_coeff'] == saved_coeff
    assert before == {file.name: file.read_bytes() for file in path.iterdir()}
    json.dumps(instance.metadata(), allow_nan=False)


@pytest.mark.parametrize('saved_coeff,override', [(None, 0), (None, .01), (.2, None), (.2, 0)])
def test_temporal_ensemble_requires_one_action_step_before_backend_load(
        tmp_path, monkeypatch, saved_coeff, override):
    path = checkpoint(tmp_path)
    saved = json.loads((path / 'config.json').read_text())
    saved['temporal_ensemble_coeff'] = saved_coeff
    (path / 'config.json').write_text(json.dumps(saved))

    def unexpected_backend_load():
        raise AssertionError('Invalid ensemble settings must fail before loading the backend')

    monkeypatch.setattr(adapter, '_load_backend', unexpected_backend_load)
    with pytest.raises(ValueError, match='temporal ensembling requires n_action_steps=1'):
        adapter.LeRobotPolicy(path, n_action_steps=2, temporal_ensemble_coeff=override)


@pytest.mark.parametrize('coefficient', [True, '0.01', float('nan'), float('inf'),
                                        -float('inf'), 1j, {}, 10**400])
def test_invalid_temporal_ensemble_override_fails_before_backend_load(tmp_path, monkeypatch, coefficient):
    def unexpected_backend_load():
        raise AssertionError('Invalid ensemble settings must fail before loading the backend')

    monkeypatch.setattr(adapter, '_load_backend', unexpected_backend_load)
    with pytest.raises(ValueError, match='finite real number'):
        adapter.LeRobotPolicy(checkpoint(tmp_path), temporal_ensemble_coeff=coefficient)


def test_invalid_saved_temporal_ensemble_coefficient_is_validated(tmp_path, monkeypatch):
    path = checkpoint(tmp_path)
    saved = json.loads((path / 'config.json').read_text())
    saved['temporal_ensemble_coeff'] = 'invalid'
    (path / 'config.json').write_text(json.dumps(saved))

    def unexpected_backend_load():
        raise AssertionError('Invalid saved ensemble coefficient must fail before loading the backend')

    monkeypatch.setattr(adapter, '_load_backend', unexpected_backend_load)
    with pytest.raises(ValueError, match='finite real number'):
        adapter.LeRobotPolicy(path)


def test_episode_reset_discards_pending_actions_and_processor_state(fake_policy):
    instance, _, _ = fake_policy
    first = instance.select_action(observation())
    instance.reset()
    assert instance.calls == 0 and instance.total_inference_seconds == 0
    assert instance.policy.resets == instance.preprocessor.resets == instance.postprocessor.resets == 2
    np.testing.assert_array_equal(instance.select_action(observation()), first)
    assert instance.metadata()['calls'] == 1


@pytest.mark.parametrize('key,value,match', [
    ('observation.images.head', np.zeros((3, 2, 4), np.uint8), 'HWC'),
    ('observation.images.head', np.zeros((2, 4, 3), np.float32), 'uint8'),
    ('observation.state', np.zeros((1, 8)), 'shape'),
    ('observation.state', np.full(8, np.nan), 'nonfinite'),
])
def test_invalid_inputs_fail_before_policy(fake_policy, key, value, match):
    instance, calls, _ = fake_policy
    obs = observation()
    obs[key] = value
    with pytest.raises(ValueError, match=match):
        instance.select_action(obs)
    assert 'normalized_batch' not in calls


def test_missing_input_has_useful_error(fake_policy):
    instance, _, _ = fake_policy
    with pytest.raises(ValueError, match='Missing policy observation'):
        instance.select_action({})


@pytest.mark.parametrize('shape,value,match', [((1, 9), 0., 'shape'), ((1, 8), float('nan'), 'nonfinite')])
def test_invalid_outputs_are_rejected(fake_policy, shape, value, match):
    instance, _, _ = fake_policy
    instance.policy.select_action = lambda batch: instance._torch.full(shape, value)
    with pytest.raises(ValueError, match=match):
        instance.select_action(observation())


def diffusion_checkpoint(tmp_path, **updates):
    path = checkpoint(tmp_path)
    config = json.loads((path / 'config.json').read_text())
    config.pop('chunk_size')
    config.pop('temporal_ensemble_coeff')
    config.update(type='diffusion', horizon=64, n_obs_steps=1, n_action_steps=64,
                  noise_scheduler_type='DDPM', num_train_timesteps=100,
                  num_inference_steps=None,
                  normalization_mapping={'VISUAL': 'MEAN_STD', 'STATE': 'MIN_MAX', 'ACTION': 'MIN_MAX'})
    config.update(updates)
    (path / 'config.json').write_text(json.dumps(config))
    return path


@pytest.mark.parametrize('updates,steps,match', [
    ({'horizon': 0}, 1, 'horizon'),
    ({'horizon': True}, 1, 'horizon'),
    ({'n_obs_steps': 0}, 1, 'n_obs_steps'),
    ({'n_obs_steps': None}, 1, 'n_obs_steps'),
    ({'n_obs_steps': 65}, 1, 'n_obs_steps'),
    ({'n_obs_steps': True}, 1, 'n_obs_steps'),
    ({'n_obs_steps': 2}, 64, 'n_action_steps'),
    ({}, 65, 'n_action_steps'),
    ({}, True, 'n_action_steps'),
])
def test_diffusion_execution_prefix_accounts_for_observation_history(
        tmp_path, monkeypatch, updates, steps, match):
    def unexpected_backend_load():
        raise AssertionError('Invalid Diffusion settings must fail before importing its backend')

    monkeypatch.setattr(adapter, '_load_diffusion_backend', unexpected_backend_load)
    with pytest.raises(ValueError, match=match):
        adapter.LeRobotPolicy(diffusion_checkpoint(tmp_path, **updates), n_action_steps=steps)


@pytest.mark.parametrize('coefficient', [0, .01])
def test_diffusion_rejects_act_temporal_ensemble_override(tmp_path, monkeypatch, coefficient):
    def unexpected_backend_load():
        raise AssertionError('ACT-only options must fail before importing Diffusion')

    monkeypatch.setattr(adapter, '_load_diffusion_backend', unexpected_backend_load)
    with pytest.raises(ValueError, match='only supported for ACT'):
        adapter.LeRobotPolicy(diffusion_checkpoint(tmp_path), temporal_ensemble_coeff=coefficient)


@pytest.fixture
def lightweight_diffusion(monkeypatch):
    """Use real LeRobot observation/action queues with a cheap sampling model."""
    torch = pytest.importorskip('torch')
    module = pytest.importorskip('lerobot.policies.diffusion.modeling_diffusion')
    calls = {'histories': []}

    class Config:
        @classmethod
        def from_pretrained(cls, folder, **kwargs):
            assert kwargs == {'local_files_only': True}
            saved = json.loads((folder / 'config.json').read_text())
            saved.setdefault('depth_camera_map', {})
            saved.setdefault('depth_unit', 'm')
            return SimpleNamespace(**saved)

    class Sampler(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config

        def generate_actions(self, batch, noise=None):
            assert torch.is_inference_mode_enabled()
            calls['histories'].append(deepcopy(batch))
            return torch.linspace(-.5, .5, self.config.n_action_steps)[None, :, None].expand(1, -1, 8)

    class Policy(module.DiffusionPolicy):
        @classmethod
        def from_pretrained(cls, folder, config, **kwargs):
            assert kwargs == {'local_files_only': True, 'strict': True}
            calls['config'] = deepcopy(config)
            instance = cls.__new__(cls)
            torch.nn.Module.__init__(instance)
            config.image_features = {key: value for key, value in config.input_features.items()
                                     if value['type'] == 'VISUAL'
                                     and key not in config.depth_camera_map.values()}
            config.env_state_feature = None
            instance.config = config
            instance.diffusion = Sampler(config)
            instance.resets = 0
            instance.reset()
            return instance

        def reset(self):
            super().reset()
            self.resets += 1

    class Processor:
        def __init__(self, prefix):
            self.prefix = prefix
            self.resets = 0

        def reset(self):
            self.resets += 1

        def __call__(self, value):
            if self.prefix == 'pre':
                return {key: (tensor.unsqueeze(0) / 5 - 1 if key == 'observation.state'
                              else tensor.unsqueeze(0)) for key, tensor in value.items()}
            # Saved MIN_MAX inverse with physical action range [-2, 6].
            return (value + 1) * 4 - 2

    def processors(folder, **kwargs):
        calls['processors'] = kwargs
        return Processor('pre'), Processor('post')

    def unexpected_act_backend_load():
        raise AssertionError('Diffusion must not load the ACT backend')

    monkeypatch.setattr(adapter, '_load_backend', unexpected_act_backend_load)
    monkeypatch.setattr(adapter, '_load_diffusion_backend', lambda: (
        torch, SimpleNamespace(__version__='fixture', __file__='fixture'), Config, Policy, processors))
    return calls


@pytest.mark.parametrize('saved_inference_steps,effective_steps', [(None, 100), (25, 25)])
def test_diffusion_load_preserves_horizon_scheduler_and_saved_processors(
        tmp_path, lightweight_diffusion, saved_inference_steps, effective_steps):
    path = diffusion_checkpoint(tmp_path, num_inference_steps=saved_inference_steps)
    before = {file.name: file.read_bytes() for file in path.iterdir()}
    instance = adapter.LeRobotPolicy(path, device='cpu', n_action_steps=32)
    calls = lightweight_diffusion
    config = calls['config']
    assert config.n_action_steps == 32 and config.horizon == 64 and config.n_obs_steps == 1
    assert config.num_inference_steps == saved_inference_steps and config.noise_scheduler_type == 'DDPM'
    assert config.pretrained_backbone_weights is None
    assert config.normalization_mapping['STATE'] == config.normalization_mapping['ACTION'] == 'MIN_MAX'
    assert not hasattr(config, 'temporal_ensemble_coeff')
    assert not instance.policy.training
    assert calls['processors'] == {
        'preprocessor_overrides': {'device_processor': {'device': 'cpu'}},
        'postprocessor_overrides': {'device_processor': {'device': 'cpu'}}}
    metadata = instance.metadata()
    assert metadata['policy_type'] == 'diffusion'
    assert metadata['horizon'] == metadata['chunk_size'] == metadata['checkpoint_n_action_steps'] == 64
    assert metadata['n_action_steps'] == 32 and metadata['n_obs_steps'] == 1
    assert metadata['noise_scheduler_type'] == 'DDPM' and metadata['num_train_timesteps'] == 100
    assert metadata['num_inference_steps'] == effective_steps
    assert metadata['checkpoint_num_inference_steps'] == saved_inference_steps
    assert metadata['temporal_ensemble_coeff'] is None
    assert before == {file.name: file.read_bytes() for file in path.iterdir()}
    json.dumps(metadata, allow_nan=False)


def test_diffusion_updates_history_during_cached_actions_and_resets_every_pipeline(tmp_path, lightweight_diffusion):
    path = diffusion_checkpoint(tmp_path, horizon=4, n_obs_steps=2, n_action_steps=3)
    instance = adapter.LeRobotPolicy(path, device='cpu', n_action_steps=2)
    obs = observation()
    actions = []
    for value in (1., 2., 3.):
        obs['observation.state'] = np.full(8, value)
        actions.append(instance.select_action(obs))
    histories = lightweight_diffusion['histories']
    assert len(histories) == 2
    assert instance.chunk_size == 3
    np.testing.assert_allclose(histories[0]['observation.state'][0, :, 0].numpy(), [-.8, -.8])
    np.testing.assert_allclose(histories[1]['observation.state'][0, :, 0].numpy(), [-.6, -.4])
    assert tuple(histories[0]['observation.images'].shape) == (1, 2, 1, 3, 2, 4)
    np.testing.assert_array_equal(actions, np.repeat([[0.], [4.], [0.]], 8, axis=1))
    assert instance.calls == 3
    resets = instance.policy.resets
    instance.reset()
    assert instance.policy.resets == resets + 1
    assert instance.preprocessor.resets == instance.postprocessor.resets == 2
    assert instance.calls == 0 and instance.total_inference_seconds == 0
    assert all(not queue for queue in instance.policy._queues.values())
    # The pending second action is discarded; a fresh padded history is built.
    np.testing.assert_array_equal(instance.select_action(obs), actions[0])
    np.testing.assert_allclose(histories[-1]['observation.state'][0, :, 0].numpy(), [-.4, -.4])


def enable_rgbd_checkpoint(path):
    """Extend a fixture with both aligned RGB-D views, retaining its policy type."""
    config = json.loads((path / 'config.json').read_text())
    config['input_features']['observation.images.wrist'] = {'type': 'VISUAL', 'shape': [3, 2, 4]}
    config['depth_camera_map'] = {
        f'observation.images.{camera}': f'observation.depth.{camera}' for camera in ('head', 'wrist')}
    config['depth_unit'] = 'm'
    for key in config['depth_camera_map'].values():
        config['input_features'][key] = {'type': 'VISUAL', 'shape': [1, 2, 4]}
    (path / 'config.json').write_text(json.dumps(config))
    return config


def rgbd_observation(layout='HW'):
    obs = observation()
    obs['observation.images.wrist'] = obs['observation.images.head'].copy()
    for camera, offset in [('head', .3), ('wrist', .1)]:
        depth = np.arange(8, dtype=np.float32).reshape(2, 4) / np.float32(100) + np.float32(offset)
        if layout == 'HWC':
            depth = depth[..., None]
        elif layout == 'CHW':
            depth = depth[None, ...]
        obs[f'observation.depth.{camera}'] = depth
    return obs


@pytest.mark.parametrize('layout', ['HW', 'HWC', 'CHW'])
def test_rgbd_preserves_metric_depth_and_saved_normalization(fake_policy, layout):
    _, calls, path = fake_policy
    config = enable_rgbd_checkpoint(path)
    before_files = {file.name: file.read_bytes() for file in path.iterdir()}
    instance = adapter.LeRobotPolicy(path, device='cpu', n_action_steps=2)
    obs = rgbd_observation(layout)
    before_obs = deepcopy(obs)
    instance.select_action(obs)
    for key in instance.depth_features:
        expected = before_obs[key].reshape(1, 2, 4)
        raw = calls['raw_batch'][key]
        assert raw.dtype == instance._torch.float32 and tuple(raw.shape) == (1, 2, 4)
        np.testing.assert_array_equal(raw.numpy(), expected)
        np.testing.assert_allclose(calls['normalized_batch'][key][0].numpy(), (expected - .5) / .25)
        # The adapter must own memory even when the simulator already supplies CHW float32.
        instance._prepare_observation(obs)[key].zero_()
        np.testing.assert_array_equal(obs[key], before_obs[key])
    assert calls['config'].depth_camera_map == config['depth_camera_map']
    assert calls['config'].depth_unit == 'm'
    metadata = instance.metadata()
    assert instance.requires_depth and metadata['requires_depth']
    assert metadata['depth_camera_map'] == config['depth_camera_map']
    assert metadata['depth_unit'] == 'm'
    assert metadata['depth_features'] == list(instance.depth_features)
    assert before_files == {file.name: file.read_bytes() for file in path.iterdir()}
    json.dumps(metadata, allow_nan=False)


@pytest.mark.parametrize('value,match', [
    (np.ones((2, 4), np.uint16), 'float32'),
    (np.ones((2, 4), np.uint8), 'float32'),
    (np.ones((2, 4), np.float64), 'float32'),
    (np.ones((2, 4), bool), 'float32'),
    (np.ones((2, 4, 3), np.float32), 'HW'),
    (np.full((2, 4), np.nan, np.float32), 'nonfinite'),
    (np.full((2, 4), np.inf, np.float32), 'nonfinite'),
    (np.full((2, 4), -.1, np.float32), 'negative'),
])
def test_rgbd_rejects_invalid_depth_before_policy(fake_policy, value, match):
    _, calls, path = fake_policy
    enable_rgbd_checkpoint(path)
    instance = adapter.LeRobotPolicy(path, device='cpu')
    obs = rgbd_observation()
    obs['observation.depth.head'] = value
    with pytest.raises(ValueError, match=match):
        instance.select_action(obs)
    assert 'normalized_batch' not in calls


@pytest.mark.parametrize('camera', ['head', 'wrist'])
def test_rgbd_requires_both_declared_depth_observations(fake_policy, camera):
    _, calls, path = fake_policy
    enable_rgbd_checkpoint(path)
    instance = adapter.LeRobotPolicy(path, device='cpu')
    obs = rgbd_observation()
    key = f'observation.depth.{camera}'
    del obs[key]
    with pytest.raises(ValueError, match=f'Missing policy observation: {key}'):
        instance.select_action(obs)
    assert 'normalized_batch' not in calls


@pytest.mark.parametrize('problem,match', [
    ('missing_unit', 'depth_unit'), ('millimetres', 'depth_unit'),
    ('invalid_map', 'depth_camera_map'), ('duplicate_depth', 'distinct depth'),
    ('missing_depth_feature', 'Missing paired depth'), ('unaligned_depth', 'spatial shapes'),
    ('wrong_channels', 'Unsupported policy feature'), ('undeclared_depth', 'Unsupported policy feature'),
    ('partial_map', 'Unsupported policy feature'), ('extra_rgb', 'cover every RGB'),
])
def test_rgbd_rejects_ambiguous_checkpoint_before_backend_load(tmp_path, monkeypatch, problem, match):
    path = checkpoint(tmp_path)
    config = enable_rgbd_checkpoint(path)
    if problem == 'missing_unit':
        config.pop('depth_unit')
    elif problem == 'millimetres':
        config['depth_unit'] = 'mm'
    elif problem == 'invalid_map':
        config['depth_camera_map'] = None
    elif problem == 'duplicate_depth':
        config['depth_camera_map']['observation.images.wrist'] = 'observation.depth.head'
    elif problem == 'missing_depth_feature':
        config['input_features'].pop('observation.depth.head')
    elif problem == 'unaligned_depth':
        config['input_features']['observation.depth.head']['shape'] = [1, 4, 4]
    elif problem == 'wrong_channels':
        config['input_features']['observation.depth.head']['shape'] = [3, 2, 4]
    elif problem == 'undeclared_depth':
        config['depth_camera_map'] = {}
    elif problem == 'partial_map':
        config['depth_camera_map'].pop('observation.images.wrist')
    elif problem == 'extra_rgb':
        config['input_features']['observation.images.other'] = {'type': 'VISUAL', 'shape': [3, 2, 4]}
    (path / 'config.json').write_text(json.dumps(config))

    def unexpected_backend_load():
        raise AssertionError('Invalid RGB-D settings must fail before importing the backend')

    monkeypatch.setattr(adapter, '_load_backend', unexpected_backend_load)
    with pytest.raises(ValueError, match=match):
        adapter.LeRobotPolicy(path)


def test_rgb_checkpoint_does_not_require_or_forward_depth(fake_policy):
    instance, calls, _ = fake_policy
    obs = observation()
    obs['observation.depth.head'] = 'unused invalid depth'
    instance.select_action(obs)
    assert 'observation.depth.head' not in calls['raw_batch']
    assert not instance.requires_depth and not instance.depth_camera_map and not instance.depth_features
    assert instance.depth_unit is None and instance.metadata()['depth_unit'] is None


def test_diffusion_rgbd_keeps_depth_paired_in_observation_history(tmp_path, lightweight_diffusion):
    path = diffusion_checkpoint(tmp_path, horizon=4, n_obs_steps=2, n_action_steps=3)
    enable_rgbd_checkpoint(path)
    instance = adapter.LeRobotPolicy(path, device='cpu', n_action_steps=2)
    obs = rgbd_observation()
    for offset in (0., .01, .02):
        for camera, value in [('head', .5), ('wrist', .2)]:
            obs[f'observation.depth.{camera}'] = np.full((2, 4), value + offset, np.float32)
        instance.select_action(obs)
    histories = lightweight_diffusion['histories']
    assert len(histories) == 2
    for history in histories:
        assert tuple(history['observation.images'].shape) == (1, 2, 2, 4, 2, 4)
    np.testing.assert_allclose(histories[0]['observation.images'][0, :, 0, 3, 0, 0].numpy(), [.5, .5])
    np.testing.assert_allclose(histories[0]['observation.images'][0, :, 1, 3, 0, 0].numpy(), [.2, .2])
    np.testing.assert_allclose(histories[1]['observation.images'][0, :, 0, 3, 0, 0].numpy(), [.51, .52])
    np.testing.assert_allclose(histories[1]['observation.images'][0, :, 1, 3, 0, 0].numpy(), [.21, .22])
    assert instance.metadata()['depth_unit'] == 'm'
