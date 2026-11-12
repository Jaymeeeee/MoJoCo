"""Camera interventions preserve paired inputs and policy execution history."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from mujoco_tools.policy_ablation import mask_camera, predict_action_chunks, random_patch_mask


def observation(offset=0):
    image = np.arange(4*8*3, dtype=np.uint8).reshape(4, 8, 3) + offset
    return {'observation.images.head': image,
            'observation.images.wrist': image[::-1].copy(),
            'observation.state': np.arange(8, dtype=np.float32) + offset}


@pytest.fixture
def adapter():
    torch = pytest.importorskip('torch')

    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._action_queue = [torch.arange(8)]
            self.temporal_ensembler = SimpleNamespace(ensembled_actions=torch.arange(24).reshape(1, 3, 8),
                                                     ensembled_actions_count=torch.ones(3, 1))
            self.predict_calls = 0

        def predict_action_chunk(self, batch):
            assert torch.is_inference_mode_enabled()
            self.predict_calls += 1
            state = batch['observation.state']
            head = batch['observation.images.head'].mean((1, 2, 3))[:, None]
            wrist = batch['observation.images.wrist'].mean((1, 2, 3))[:, None]
            return (state+head+2*wrist)[:, None] + torch.arange(3)[None, :, None]

        def select_action(self, batch):
            raise AssertionError('Shadow prediction must never select an action')

    def prepare(obs):
        return {key: torch.from_numpy(np.array(value, dtype=np.float32, copy=True))
                for key, value in obs.items()}

    def preprocess(batch):
        return {**{key: (value.unsqueeze(0)-2)/3 for key, value in batch.items()},
                'action': None, 'complementary_data': {}}

    def postprocess(flat):
        assert flat.ndim == 2
        # Deliberately in-place to verify returned normalized predictions stay
        # normalized and do not alias the physical output.
        return flat.mul_(2).add_(5)

    policy = Policy().eval()
    return SimpleNamespace(_torch=torch, policy=policy, preprocessor=preprocess,
                           postprocessor=postprocess, _prepare_observation=prepare,
                           input_features={key: {} for key in observation()},
                           chunk_size=3, action_shape=(8,), calls=7)


def test_batched_predictions_match_independent_inputs_and_preserve_history(adapter):
    observations = [observation(0), observation(10), observation(20)]
    before = deepcopy(observations)
    queue = [value.clone() for value in adapter.policy._action_queue]
    ensemble = deepcopy(adapter.policy.temporal_ensembler)
    predictions = predict_action_chunks(adapter, observations)
    singles = [predict_action_chunks(adapter, [item]) for item in observations]
    assert predictions.normalized.shape == predictions.physical.shape == (3, 3, 8)
    assert predictions.normalized.dtype == predictions.physical.dtype == np.float32
    np.testing.assert_array_equal(predictions.normalized, np.concatenate([s.normalized for s in singles]))
    np.testing.assert_array_equal(predictions.physical, np.concatenate([s.physical for s in singles]))
    np.testing.assert_array_equal(predictions.physical, predictions.normalized*2+5)
    for old, current in zip(before, observations, strict=True):
        for key in old:
            np.testing.assert_array_equal(old[key], current[key])
    assert adapter.calls == 7
    for old, current in zip(queue, adapter.policy._action_queue, strict=True):
        assert adapter._torch.equal(old, current)
    assert adapter._torch.equal(ensemble.ensembled_actions, adapter.policy.temporal_ensembler.ensembled_actions)
    assert adapter._torch.equal(ensemble.ensembled_actions_count, adapter.policy.temporal_ensembler.ensembled_actions_count)
    assert not np.shares_memory(predictions.normalized, predictions.physical)


def test_empty_training_and_nonfinite_predictions_rejected(adapter):
    with pytest.raises(ValueError, match='At least one'):
        predict_action_chunks(adapter, [])
    adapter.policy.train()
    with pytest.raises(ValueError, match='evaluation mode'):
        predict_action_chunks(adapter, [observation()])
    adapter.policy.eval()
    adapter.policy.predict_action_chunk = lambda batch: adapter._torch.full((1, 3, 8), float('nan'))
    with pytest.raises(ValueError, match='nonfinite normalized'):
        predict_action_chunks(adapter, [observation()])


@pytest.mark.parametrize('fill', ['mean', 'black'])
def test_mask_changes_only_selected_camera_pixels_and_preserves_inputs(fill):
    obs = observation()
    before = deepcopy(obs)
    mask = np.zeros((4, 8), dtype=bool)
    mask[:2, ::2] = True
    out = mask_camera(obs, 'observation.images.head', mask, fill)
    assert out is not obs and out['observation.images.head'] is not obs['observation.images.head']
    for key in obs:
        np.testing.assert_array_equal(obs[key], before[key])
    for key in ('observation.images.wrist', 'observation.state'):
        np.testing.assert_array_equal(out[key], obs[key])
    np.testing.assert_array_equal(out['observation.images.head'][~mask], obs['observation.images.head'][~mask])
    color = np.rint(obs['observation.images.head'].mean((0, 1))).astype(np.uint8) if fill == 'mean' else np.zeros(3, np.uint8)
    np.testing.assert_array_equal(out['observation.images.head'][mask], np.tile(color, (mask.sum(), 1)))


def test_seeded_patch_mask_exact_coverage_and_camera_pairing():
    first = random_patch_mask((256, 256), .5, np.random.default_rng(41))
    same = random_patch_mask((256, 256), .5, np.random.default_rng(41))
    other = random_patch_mask((256, 256), .5, np.random.default_rng(42))
    assert first.dtype == bool and first.sum() == 256*256//2
    np.testing.assert_array_equal(first, same)
    assert not np.array_equal(first, other)
    patches = first.reshape(8, 32, 8, 32).transpose(0, 2, 1, 3).reshape(64, -1)
    assert np.all(patches == patches[:, :1]) and patches[:, 0].sum() == 32
    assert not random_patch_mask((16, 24), 0, np.random.default_rng(0)).any()
    assert random_patch_mask((16, 24), 1, np.random.default_rng(0)).all()


@pytest.mark.parametrize('shape,fraction,grid', [((15, 16), .5, 8), ((16, 16, 3), .5, 8),
                                               ((16, 16), 1.1, 8), ((16, 16), float('nan'), 8),
                                               ((16, 16), .5, 0)])
def test_invalid_patch_masks_fail_explicitly(shape, fraction, grid):
    with pytest.raises(ValueError):
        random_patch_mask(shape, fraction, np.random.default_rng(0), grid)


def test_mask_validation():
    obs = observation()
    with pytest.raises(ValueError, match='boolean'):
        mask_camera(obs, 'observation.images.head', np.ones((4, 8)))
    with pytest.raises(ValueError, match='fill'):
        mask_camera(obs, 'observation.images.head', np.ones((4, 8), bool), 'noise')
