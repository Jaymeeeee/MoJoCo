"""Spatial activation capture preserves outputs, RNG, camera order and scope."""
from types import SimpleNamespace

import numpy as np
import pytest

from mujoco_tools.act_features import ACTFeatureTrace


torch = pytest.importorskip('torch')


class FeatureMap(torch.nn.Module):
    def forward(self, image):
        return {'feature_map': image}


class TinyACT(torch.nn.Module):
    """The real shared backbone -> projection order, with a known projection."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(type='act', image_features={
            'observation.images.wrist': object(), 'observation.images.head': object()})
        self.backbone = FeatureMap()
        self.encoder_img_feat_input_proj = torch.nn.Conv2d(2, 3, 1)
        with torch.no_grad():
            self.encoder_img_feat_input_proj.weight.copy_(torch.tensor(
                [[1., -2.], [-.5, .25], [2., 1.]]).reshape(3, 2, 1, 1))
            self.encoder_img_feat_input_proj.bias.copy_(torch.tensor([.5, -1., .25]))
        self.fail_after_features = False
        self.consume_random = False
        self.eval()

    def forward(self, images):
        features = [self.encoder_img_feat_input_proj(self.backbone(image)['feature_map'])
                    for image in images]
        if self.fail_after_features:
            raise RuntimeError('injected model failure')
        result = torch.cat([x.flatten(1) for x in features], dim=1)
        return result + torch.rand(()) if self.consume_random else result


@pytest.fixture
def images():
    # Reversed camera names and asymmetric grids reveal accidental sorting or
    # transposed maps; signed channels distinguish RMS from a channel mean.
    return [torch.tensor([[[[1., -2., 3.], [4., 0., -1.]],
                            [[-3., 2., 1.], [0., 4., -2.]]]]),
            torch.tensor([[[[-2., 0.], [1., 3.], [4., 2.]],
                            [[1., 2.], [-3., 4.], [0., -1.]]]])]


def test_exact_features_rms_camera_order_and_independent_readonly_copies(images):
    model = TinyACT()
    expected_backbone = [x.numpy().copy() for x in images]
    weight = np.array([[1., -2.], [-.5, .25], [2., 1.]], np.float32)
    bias = np.array([.5, -1., .25], np.float32)[None, :, None, None]
    with torch.inference_mode(), ACTFeatureTrace(model) as trace:
        model(images)
        capture = trace.latest
        assert tuple(capture.camera_backbone_features) == tuple(model.config.image_features)
        assert tuple(capture.camera_projected_features) == tuple(model.config.image_features)
        assert capture.grid_shapes == {'observation.images.wrist': (2, 3),
                                       'observation.images.head': (3, 2)}
        for key, expected in zip(model.config.image_features, expected_backbone, strict=True):
            projected = np.einsum('dc,bchw->bdhw', weight, expected) + bias
            np.testing.assert_array_equal(capture.camera_backbone_features[key], expected)
            np.testing.assert_array_equal(capture.camera_projected_features[key], projected)
            for actual, source in [(capture.backbone_rms[key], expected),
                                   (capture.projected_rms[key], projected)]:
                np.testing.assert_allclose(actual, np.sqrt((source[0] ** 2).mean(axis=0)), rtol=1e-7)
                assert actual.shape == capture.grid_shapes[key]
                assert not actual.flags.writeable
            assert not capture.camera_backbone_features[key].flags.writeable
            assert not capture.camera_projected_features[key].flags.writeable
        images[0].zero_()
        np.testing.assert_array_equal(capture.camera_backbone_features['observation.images.wrist'], expected_backbone[0])
        with pytest.raises(ValueError, match='read-only'):
            capture.projected_rms['observation.images.head'][0, 0] = 0
        assert trace.capture_count == 1


def test_capture_preserves_output_and_exact_rng_consumption(images):
    model = TinyACT()
    model.consume_random = True
    with torch.inference_mode():
        initial_rng = torch.get_rng_state().clone()
        baseline = model(images)
        expected_rng = torch.get_rng_state().clone()
        torch.set_rng_state(initial_rng)
        with ACTFeatureTrace(model) as trace:
            monitored = model(images)
            assert torch.equal(monitored, baseline)
            assert torch.equal(torch.get_rng_state(), expected_rng)
            assert trace.capture_count == 1
        saved = trace.latest
        model(images)
        assert trace.capture_count == 1
        assert trace.latest is saved
        trace.clear()
        assert trace.latest is None
        assert trace.capture_count == 1


def test_model_exception_discards_partial_capture_and_deactivates_trace(images):
    model = TinyACT()
    with torch.inference_mode(), ACTFeatureTrace(model) as trace:
        model(images)
        previous = trace.latest
        model.fail_after_features = True
        with pytest.raises(RuntimeError, match='injected model failure'):
            model(images)
        assert trace.latest is previous
        assert trace.capture_count == 1
        model.fail_after_features = False
        model(images)
        assert trace.capture_count == 1
    with trace:
        model(images)
        assert trace.capture_count == 2


@pytest.mark.parametrize('invalid,match', [
    (lambda x: x.expand(2, -1, -1, -1), 'batch size one'),
    (lambda x: x[0], 'shape'),
    (lambda x: x[:, :0], 'nonempty'),
    (lambda x: x * float('nan'), 'finite'),
    (lambda x: x.long(), 'floating point'),
])
def test_invalid_backbone_maps_rejected_without_partial_capture(images, invalid, match):
    model = TinyACT()
    with ACTFeatureTrace(model) as trace:
        with pytest.raises(ValueError, match=match):
            model([invalid(images[0]), images[1]])
        assert trace.latest is None
        assert trace.capture_count == 0
        model(images)
        assert trace.capture_count == 0


@pytest.mark.parametrize('camera_count', [1, 3])
def test_each_configured_camera_must_run_once(images, camera_count):
    model = TinyACT()
    with ACTFeatureTrace(model) as trace:
        with pytest.raises(ValueError, match='camera'):
            model((images + [images[0]])[:camera_count])
        assert trace.latest is None


def test_projection_must_preserve_spatial_grid(images):
    model = TinyACT()
    model.encoder_img_feat_input_proj = torch.nn.AvgPool2d(2)
    model.eval()
    with ACTFeatureTrace(model) as trace:
        with pytest.raises(ValueError, match='spatial shapes'):
            model(images)
        assert trace.latest is None


def test_nonfinite_projection_rejected(images):
    model = TinyACT()
    with torch.no_grad():
        model.encoder_img_feat_input_proj.bias.fill_(float('inf'))
    with ACTFeatureTrace(model) as trace:
        with pytest.raises(ValueError, match='projected feature map.*finite'):
            model(images)
        assert trace.latest is None


def test_eval_and_camera_order_are_required(images):
    model = TinyACT()
    with pytest.raises(ValueError, match='same order'):
        ACTFeatureTrace(model, image_keys=reversed(model.config.image_features))
    model.train()
    with pytest.raises(ValueError, match='model.eval'):
        with ACTFeatureTrace(model):
            pass
    assert model.training
    model.eval()
    with ACTFeatureTrace(model) as trace:
        model.backbone.train()
        with pytest.raises(ValueError, match='model.eval'):
            model(images)
        assert trace.capture_count == 0


def test_caller_exception_closes_trace(images):
    model = TinyACT()
    with pytest.raises(RuntimeError, match='caller failure'):
        with ACTFeatureTrace(model) as trace:
            raise RuntimeError('caller failure')
    model(images)
    assert trace.capture_count == 0
