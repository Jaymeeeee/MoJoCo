"""Attention extraction must preserve actions, token layout and hook lifetime."""
from types import SimpleNamespace

import numpy as np
import pytest

from mujoco_tools.act_attention import ACTAttentionTrace


torch = pytest.importorskip('torch')


class FeatureMap(torch.nn.Module):
    def forward(self, image):
        return {'feature_map': image}


class TinyACT(torch.nn.Module):
    """Real torch cross-attention with the local ACT memory-token contract."""

    def __init__(self, *, state=True, env=False):
        super().__init__()
        self.config = SimpleNamespace(
            type='act', chunk_size=3,
            image_features={'observation.images.wrist': object(), 'observation.images.head': object()},
            robot_state_feature=object() if state else None,
            env_state_feature=object() if env else None,
        )
        self.backbone = FeatureMap()
        layer = torch.nn.Module()
        layer.multihead_attn = torch.nn.MultiheadAttention(4, 2, dropout=0)
        self.decoder = torch.nn.Module()
        self.decoder.layers = torch.nn.ModuleList([layer])
        self.register_buffer('queries', torch.tensor([
            [[0.1, 0.4, -0.2, 0.8]],
            [[0.6, -0.3, 0.7, 0.5]],
            [[-0.8, 0.2, 0.9, -0.1]],
        ]))
        self.need_weights = True
        self.average_attn_weights = True
        self.extra_token = False
        self.fail_after_attention = False
        self.eval()

    def memory(self, images):
        batch = images[0].shape[0]
        count = 1 + bool(self.config.robot_state_feature) + bool(self.config.env_state_feature)
        special = torch.arange(count * batch * 4, dtype=torch.float32).reshape(count, batch, 4) / 5
        spatial = [self.backbone(image)['feature_map'].flatten(2).permute(2, 0, 1) for image in images]
        if self.extra_token:
            spatial.append(torch.zeros(1, batch, 4))
        return torch.cat([special, *spatial], dim=0)

    def forward(self, images):
        memory = self.memory(images)
        result, _ = self.decoder.layers[-1].multihead_attn(
            self.queries.expand(-1, memory.shape[1], -1), memory, memory,
            need_weights=self.need_weights, average_attn_weights=self.average_attn_weights)
        if self.fail_after_attention:
            raise RuntimeError('injected failure after attention')
        return result.transpose(0, 1)


@pytest.fixture
def images():
    # Asymmetric and different camera grids catch transposed/misordered slices.
    return [torch.arange(24, dtype=torch.float32).reshape(1, 4, 2, 3) / 10,
            torch.arange(24, dtype=torch.float32).reshape(1, 4, 3, 2).flip(-1) / 20]


def assert_no_hooks(model):
    for module in model.modules():
        assert not module._forward_hooks
        assert not module._forward_pre_hooks


@pytest.mark.parametrize('state,env', [(False, False), (True, False), (False, True), (True, True)])
def test_passive_capture_preserves_output_and_spatial_token_order(images, state, env):
    torch.manual_seed(11)
    model = TinyACT(state=state, env=env)
    with torch.inference_mode():
        baseline = model(images)
        memory = model.memory(images)
        _, expected = model.decoder.layers[-1].multihead_attn(model.queries, memory, memory)
        with ACTAttentionTrace(model) as trace:
            monitored = model(images)
            assert torch.equal(baseline, monitored)
            assert trace.capture_count == 1
            capture = trace.latest
            np.testing.assert_array_equal(capture.token_weights, expected[0].numpy())
            names = ['latent'] + (['state'] if state else []) + (['environment_state'] if env else [])
            assert tuple(capture.special_token_weights) == tuple(names)
            for index, name in enumerate(names):
                np.testing.assert_array_equal(capture.special_token_weights[name], expected[0, :, index])
            offset = len(names)
            for key, shape in zip(model.config.image_features, ((2, 3), (3, 2)), strict=True):
                assert capture.grid_shapes[key] == shape
                np.testing.assert_array_equal(capture.camera_weights[key], expected[0, :, offset:offset + 6].reshape(3, *shape))
                offset += 6
            np.testing.assert_allclose(capture.token_weights.sum(-1), 1, atol=1e-6)
            with pytest.raises(ValueError, match='read-only'):
                capture.camera_weights['observation.images.head'][0, 0, 0] = 0
            model(images)
            assert trace.capture_count == 2
        assert_no_hooks(model)
        model(images)
        assert trace.capture_count == 2
        trace.clear()
        assert trace.latest is None
        assert trace.capture_count == 2


def test_partial_forward_error_does_not_publish_capture_and_removes_hooks(images):
    model = TinyACT()
    with ACTAttentionTrace(model) as trace:
        with torch.inference_mode():
            model(images)
            first = trace.latest
            model.fail_after_attention = True
            with pytest.raises(RuntimeError, match='injected failure'):
                model(images)
        assert trace.latest is first
        assert trace.capture_count == 1
        assert_no_hooks(model)


@pytest.mark.parametrize('setting,value,match', [
    ('need_weights', False, 'did not return attention weights'),
    ('average_attn_weights', False, 'head-averaged'),
    ('extra_token', True, 'memory token count mismatch'),
])
def test_unsupported_attention_rejected_without_leaving_hooks(images, setting, value, match):
    model = TinyACT()
    setattr(model, setting, value)
    with ACTAttentionTrace(model) as trace:
        with pytest.raises(ValueError, match=match):
            model(images)
        assert trace.latest is None
        assert trace.capture_count == 0
        assert_no_hooks(model)


def test_batch_larger_than_one_rejected(images):
    model = TinyACT()
    with ACTAttentionTrace(model):
        with pytest.raises(ValueError, match='batch size one'):
            model([image.expand(2, -1, -1, -1) for image in images])
        assert_no_hooks(model)


def test_camera_reordering_rejected_before_hooks():
    model = TinyACT()
    with pytest.raises(ValueError, match='same order'):
        ACTAttentionTrace(model, image_keys=reversed(model.config.image_features))
    assert_no_hooks(model)


def test_training_mode_rejected_without_changing_model():
    model = TinyACT().train()
    with pytest.raises(ValueError, match='model.eval'):
        with ACTAttentionTrace(model):
            pass
    assert model.training
    assert_no_hooks(model)


def test_context_error_removes_hooks():
    model = TinyACT()
    with pytest.raises(RuntimeError, match='caller failure'):
        with ACTAttentionTrace(model):
            raise RuntimeError('caller failure')
    assert_no_hooks(model)
