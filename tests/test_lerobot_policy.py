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
    ({'type': 'diffusion'}, 'supports ACT'),
    ({'output_features': {}}, 'output feature'),
    ({'input_features': {'privileged.target': {'type': 'STATE', 'shape': [3]}}}, 'Unsupported ACT feature'),
    ({'input_features': {'observation.images.head': {'type': 'VISUAL', 'shape': [1, 2, 4]}}}, 'Unsupported ACT feature'),
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
            return SimpleNamespace(**json.loads((folder / 'config.json').read_text()))

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
