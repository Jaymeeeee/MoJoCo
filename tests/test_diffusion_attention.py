"""Spatial pooling diagnostics preserve predictions, RNG, and camera geometry."""
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from mujoco_tools.diffusion_attention import DiffusionAttentionTrace


torch = pytest.importorskip('torch')


class TinySpatialSoftmax(torch.nn.Module):
    def __init__(self, shape, num_kp=2):
        super().__init__()
        self._in_c, self._in_h, self._in_w = shape
        self._out_c = num_kp if num_kp is not None else shape[0]
        self.nets = torch.nn.Conv2d(shape[0], num_kp, 1) if num_kp is not None else None
        x, y = np.meshgrid(np.linspace(-1, 1, self._in_w), np.linspace(-1, 1, self._in_h))
        self.register_buffer('pos_grid', torch.tensor(np.stack([x.ravel(), y.ravel()], -1), dtype=torch.float32))
        self.corrupt = False

    def forward(self, features):
        if self.nets is not None:
            features = self.nets(features)
        probabilities = torch.softmax(features.reshape(-1, self._in_h*self._in_w), -1)
        result = (probabilities @ self.pos_grid).reshape(-1, self._out_c, 2)
        return result + .01 if self.corrupt else result


class TinyEncoder(torch.nn.Module):
    def __init__(self, shape, *, projection=True, real_pool=False):
        super().__init__()
        self.resize, self.do_crop = None, False
        self.backbone = torch.nn.Conv2d(3, 4, 1)
        pool_type = TinySpatialSoftmax
        if real_pool:
            from lerobot.policies.diffusion.modeling_diffusion import SpatialSoftmax
            pool_type = SpatialSoftmax
        self.pool = pool_type((4, *shape[-2:]), 2 if projection else None)
        self.out = torch.nn.Linear(self.pool._out_c*2, 4)
        self.fail_after_pool = False
        self.backbone_calls = 0

    def forward(self, images):
        self.backbone_calls += 1
        points = self.pool(self.backbone(images)).flatten(1)
        if self.fail_after_pool:
            raise RuntimeError('injected encoder error')
        return self.out(points)


class TinyDiffusion(torch.nn.Module):
    """Match encoder invocation order plus stochastic, cached predictions."""
    def __init__(self, *, projection=True, real_pool=False):
        super().__init__()
        self.config = SimpleNamespace(
            type='diffusion', n_obs_steps=1,
            use_separate_rgb_encoder_per_camera=True, crop_shape=None, resize_shape=None,
            image_features={
                'observation.images.wrist': SimpleNamespace(shape=[3, 2, 3]),
                'observation.images.head': SimpleNamespace(shape=[3, 3, 2]),
            })
        self.rgb_encoder = torch.nn.ModuleList([
            TinyEncoder(feature.shape, projection=projection, real_pool=real_pool)
            for feature in self.config.image_features.values()])
        self.queue = deque()
        self.eval()

    def generate_actions(self, images):
        features = torch.cat([encoder(image) for encoder, image in zip(self.rgb_encoder, images, strict=True)], -1)
        # Draws emulate stochastic denoising and detect changes to RNG state.
        return features.unsqueeze(1) + torch.randn(1, 4, 8) + torch.randn(1, 4, 8)

    def select_action(self, images):
        if not self.queue:
            self.queue.extend(self.generate_actions(images).transpose(0, 1))
        return self.queue.popleft()


@pytest.fixture
def images():
    return [torch.arange(18, dtype=torch.float32).reshape(1, 3, 2, 3)/19,
            torch.arange(18, dtype=torch.float32).reshape(1, 3, 3, 2).flip(-1)/23]


def assert_no_hooks(model):
    for module in model.modules():
        assert not module._forward_hooks
        assert not module._forward_pre_hooks


@pytest.mark.parametrize('projection', [True, False])
def test_probabilities_keypoints_preserve_outputs_rng_and_camera_order(images, projection):
    torch.manual_seed(9)
    model = TinyDiffusion(projection=projection)
    parameters = {key: value.clone() for key, value in model.state_dict().items()}
    initial_rng = torch.get_rng_state().clone()
    with torch.inference_mode():
        baseline = model.generate_actions(images)
        final_rng = torch.get_rng_state().clone()
        torch.set_rng_state(initial_rng)
        with DiffusionAttentionTrace(model) as trace:
            monitored = model.generate_actions(images)
            assert torch.equal(monitored, baseline)
            assert torch.equal(torch.get_rng_state(), final_rng)
            assert trace.capture_count == 1
            capture = trace.latest
            assert tuple(capture.camera_weights) == tuple(model.config.image_features)
            for index, (key, grid) in enumerate(zip(trace.image_keys, [(2, 3), (3, 2)], strict=True)):
                k = 2 if projection else 4
                weights = capture.camera_weights[key]
                assert weights.shape == (k, *grid)
                assert capture.grid_shapes[key] == grid
                assert capture.keypoint_counts[key] == k
                assert weights.dtype == np.float32
                np.testing.assert_allclose(weights.sum((-2, -1)), 1, atol=1e-6)
                np.testing.assert_array_equal(capture.mean_weights[key], weights.mean(0))
                assert capture.mean_weights[key].sum() == pytest.approx(1)
                coordinates = model.rgb_encoder[index].pool.pos_grid.numpy()
                np.testing.assert_allclose(weights.reshape(k, -1)@coordinates,
                                           capture.keypoints[key], atol=1e-7)
                assert capture.keypoint_reconstruction_max_abs_error[key] == 0.
                for array in [weights, capture.mean_weights[key], capture.keypoints[key]]:
                    assert not array.flags.writeable
            # One extra real backbone call per camera, not an additional pass to trace.
            assert [encoder.backbone_calls for encoder in model.rgb_encoder] == [2, 2]
        assert_no_hooks(model)
    for key, expected in parameters.items():
        assert torch.equal(model.state_dict()[key], expected)


def test_cached_actions_do_not_create_new_capture_and_context_can_be_reused(images):
    model = TinyDiffusion()
    trace = DiffusionAttentionTrace(model)
    with torch.inference_mode(), trace:
        model.select_action(images)
        first = trace.latest
        for _ in range(3):
            model.select_action([image + .5 for image in images])
            assert trace.latest is first
            assert trace.capture_count == 1
        model.select_action([image + .5 for image in images])
        assert trace.latest is not first and trace.capture_count == 2
        trace.clear()
        assert trace.latest is None and trace.capture_count == 2
    assert_no_hooks(model)
    with torch.inference_mode(), trace:
        model.generate_actions(images)
        assert trace.latest is not None and trace.capture_count == 3
    assert_no_hooks(model)


@pytest.mark.parametrize('camera', [0, 1])
def test_partial_encoder_failure_preserves_previous_capture_and_cleans_hooks(images, camera):
    model = TinyDiffusion()
    with torch.inference_mode(), DiffusionAttentionTrace(model) as trace:
        model.generate_actions(images)
        first = trace.latest
        model.rgb_encoder[camera].fail_after_pool = True
        with pytest.raises(RuntimeError, match='injected encoder error'):
            model.generate_actions(images)
        assert trace.latest is first and trace.capture_count == 1
        assert_no_hooks(model)


def test_reconstruction_detects_changed_pool_semantics(images):
    model = TinyDiffusion()
    model.rgb_encoder[1].pool.corrupt = True
    with DiffusionAttentionTrace(model) as trace:
        with pytest.raises(ValueError, match='do not reconstruct'):
            model.generate_actions(images)
        assert trace.latest is None and trace.capture_count == 0
        assert_no_hooks(model)


@pytest.mark.parametrize('setting,value,match', [
    ('type', 'act', 'config.type'),
    ('n_obs_steps', 2, 'n_obs_steps=1'),
    ('use_separate_rgb_encoder_per_camera', False, 'separate RGB'),
    ('crop_shape', (2, 2), 'no image crop or resize'),
    ('resize_shape', (2, 2), 'no image crop or resize'),
])
def test_unsupported_config_rejected_without_hooks(setting, value, match):
    model = TinyDiffusion()
    setattr(model.config, setting, value)
    with pytest.raises(ValueError, match=match):
        DiffusionAttentionTrace(model)
    assert_no_hooks(model)


def test_reordered_camera_names_rejected():
    model = TinyDiffusion()
    with pytest.raises(ValueError, match='same order'):
        DiffusionAttentionTrace(model, image_keys=reversed(model.config.image_features))
    assert_no_hooks(model)


def test_actual_call_order_rejected(images):
    model = TinyDiffusion()
    with DiffusionAttentionTrace(model) as trace:
        with pytest.raises(ValueError, match='configured order'):
            model.rgb_encoder[1](images[1])
        assert trace.latest is None
        assert_no_hooks(model)


@pytest.mark.parametrize('training_when', ['before', 'after'])
def test_training_mode_rejected_without_changing_it(images, training_when):
    model = TinyDiffusion()
    if training_when == 'before':
        model.train()
        with pytest.raises(ValueError, match='model.eval'):
            with DiffusionAttentionTrace(model):
                pass
    else:
        with DiffusionAttentionTrace(model):
            model.train()
            with pytest.raises(ValueError, match='model.eval'):
                model.generate_actions(images)
    assert model.training
    assert_no_hooks(model)


def test_batch_size_and_shape_rejected(images):
    model = TinyDiffusion()
    with DiffusionAttentionTrace(model):
        with pytest.raises(ValueError, match='batch size one'):
            model.generate_actions([image.repeat(2, 1, 1, 1) for image in images])
    assert_no_hooks(model)
    with DiffusionAttentionTrace(model):
        with pytest.raises(ValueError, match='image shape differs'):
            model.generate_actions([images[0].transpose(-1, -2), images[1]])
    assert_no_hooks(model)


def test_nonfinite_probabilities_rejected(images):
    model = TinyDiffusion()
    bad_images = [images[0] * float('nan'), images[1]]
    with DiffusionAttentionTrace(model) as trace:
        with pytest.raises(ValueError, match='finite and nonnegative'):
            model.generate_actions(bad_images)
        assert trace.latest is None
        assert_no_hooks(model)


def test_caller_error_removes_hooks():
    model = TinyDiffusion()
    with pytest.raises(RuntimeError, match='caller failed'):
        with DiffusionAttentionTrace(model):
            raise RuntimeError('caller failed')
    assert_no_hooks(model)


def test_capture_clear_discards_incomplete_camera_group(images):
    model = TinyDiffusion()
    with torch.inference_mode(), DiffusionAttentionTrace(model) as trace:
        model.rgb_encoder[0](images[0])
        assert trace.latest is None and trace.capture_count == 0
        trace.clear()
        model.generate_actions(images)
        assert trace.capture_count == 1


def test_real_lerobot_spatial_pool_matches_actual_output(images):
    pytest.importorskip('lerobot.policies.diffusion.modeling_diffusion')
    torch.manual_seed(15)
    model = TinyDiffusion(real_pool=True)
    initial_rng = torch.get_rng_state().clone()
    with torch.inference_mode():
        baseline = model.generate_actions(images)
        final_rng = torch.get_rng_state().clone()
        torch.set_rng_state(initial_rng)
        with DiffusionAttentionTrace(model) as trace:
            captured = model.generate_actions(images)
            assert torch.equal(captured, baseline)
            assert torch.equal(torch.get_rng_state(), final_rng)
            assert trace.capture_count == 1
            assert all(error == 0 for error in trace.latest.keypoint_reconstruction_max_abs_error.values())
    assert_no_hooks(model)
