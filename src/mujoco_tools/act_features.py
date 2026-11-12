"""Passively capture spatial ACT image features before Transformer mixing.

The backbone output is the local ACT ResNet ``layer4`` feature map. The projected
output is the learned 1x1 ``encoder_img_feat_input_proj`` output, before spatial
flattening and before positional encoding is used by the Transformer. ACT passes
position embeddings separately; they are not included in these feature maps.
Channel RMS measures activation magnitude, not attention, probability, keypoints,
or causal importance of a camera or pixel.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ACTFeatureCapture:
    """One successful batch-one ACT forward in configured camera order.

    Feature arrays are read-only float32 CPU copies with shape ``(1, C, H, W)``
    or ``(1, D, H, W)``. RMS arrays have shape ``(H, W)`` and contain
    ``sqrt(mean_channels(feature ** 2))``, without spatial normalization. Camera
    dictionaries retain the insertion order of ``config.image_features``.
    """

    camera_backbone_features: dict[str, np.ndarray]
    camera_projected_features: dict[str, np.ndarray]
    backbone_rms: dict[str, np.ndarray]
    projected_rms: dict[str, np.ndarray]
    grid_shapes: dict[str, tuple[int, int]]


def _channel_rms(feature: np.ndarray) -> np.ndarray:
    # Float64 accumulation avoids overflow when squaring finite float32 values.
    rms = np.sqrt(np.mean(np.square(feature[0], dtype=np.float64), axis=0)).astype(np.float32)
    rms.setflags(write=False)
    return rms


class ACTFeatureTrace:
    """Read real backbone/projection outputs without changing the forward.

    Use ``with ACTFeatureTrace(policy.model) as trace: ...``. A capture is
    published only after a successful model forward. Cached action calls leave
    ``latest`` and ``capture_count`` unchanged. No model calls, RNG operations,
    parameter changes, or replacement forward results are introduced.

    Supports the local ACT architecture, evaluation mode, and batch size one.
    The shared backbone and image projection must each run exactly once per
    camera, alternating in ``config.image_features`` order. The caller must
    associate captures with their actual source observations.
    """

    def __init__(self, model, *, image_keys=None):
        import torch

        self._torch = torch
        self.model = model
        config = getattr(model, 'config', None)
        if getattr(config, 'type', None) != 'act':
            raise ValueError('ACT feature tracing requires config.type="act"')
        configured_keys = tuple(getattr(config, 'image_features', {}))
        if not configured_keys or len(set(configured_keys)) != len(configured_keys):
            raise ValueError('ACT feature tracing requires configured image features')
        if image_keys is not None and tuple(image_keys) != configured_keys:
            raise ValueError('image_keys must match ACT config.image_features in the same order')
        self.image_keys = configured_keys
        self._backbone = getattr(model, 'backbone', None)
        self._projection = getattr(model, 'encoder_img_feat_input_proj', None)
        if not isinstance(self._backbone, torch.nn.Module) or not isinstance(self._projection, torch.nn.Module):
            raise ValueError('Unsupported ACT architecture: expected backbone and encoder_img_feat_input_proj')
        if self._backbone is self._projection:
            raise ValueError('ACT backbone and image projection must be distinct modules')
        self.latest: ACTFeatureCapture | None = None
        self.capture_count = 0
        self._handles = []
        self._backbone_features = []
        self._projected_features = []
        self._in_forward = False

    def _check_eval(self):
        if any(module.training for module in self.model.modules()):
            self._fail('ACT feature tracing requires model.eval() for all submodules')

    def __enter__(self):
        if self._handles:
            raise RuntimeError('ACTFeatureTrace is already active')
        self._check_eval()
        try:
            self._handles.append(self.model.register_forward_pre_hook(self._begin_forward))
            self._handles.append(self._backbone.register_forward_hook(self._capture_backbone))
            self._handles.append(self._projection.register_forward_hook(self._capture_projection))
            self._handles.append(self.model.register_forward_hook(self._finish_forward, always_call=True))
        except Exception:
            self.close()
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        """Remove hooks and discard pending data, retaining the last capture."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._backbone_features.clear()
        self._projected_features.clear()
        self._in_forward = False

    def clear(self):
        """Forget the last capture, retaining the monotonic capture count."""
        if self._in_forward:
            raise RuntimeError('Cannot clear ACT features during a model forward')
        self.latest = None

    def _fail(self, message):
        self.close()
        raise ValueError(message)

    def _begin_forward(self, module, args):
        self._check_eval()
        if self._in_forward:
            self._fail('ACT feature tracing does not support nested or concurrent model forwards')
        if tuple(getattr(module.config, 'image_features', {})) != self.image_keys:
            self._fail('ACT config.image_features changed after trace construction')
        self._in_forward = True
        self._backbone_features = []
        self._projected_features = []

    def _copy_feature(self, value, name):
        if not isinstance(value, self._torch.Tensor) or value.ndim != 4:
            self._fail(f'ACT {name} must have shape (B, C, H, W)')
        if value.shape[0] != 1:
            self._fail('ACT feature tracing supports batch size one')
        if min(value.shape[1:]) < 1:
            self._fail(f'ACT {name} must have nonempty channel and spatial dimensions')
        if not value.is_floating_point():
            self._fail(f'ACT {name} must be floating point')
        result = value.detach().float().cpu().numpy().copy()
        if not np.isfinite(result).all():
            self._fail(f'ACT {name} must contain finite float32 values')
        result.setflags(write=False)
        return result

    def _capture_backbone(self, module, args, output):
        if not self._in_forward:
            return
        if len(self._backbone_features) >= len(self.image_keys):
            self._fail('ACT backbone ran more than once per configured camera')
        if len(self._backbone_features) != len(self._projected_features):
            self._fail('ACT backbone and projection calls must alternate per camera')
        feature = output.get('feature_map') if isinstance(output, dict) else None
        self._backbone_features.append(self._copy_feature(feature, 'backbone feature_map'))

    def _capture_projection(self, module, args, output):
        if not self._in_forward:
            return
        if len(self._backbone_features) != len(self._projected_features) + 1:
            self._fail('ACT projection must run exactly once after each camera backbone')
        feature = self._copy_feature(output, 'projected feature map')
        if feature.shape[-2:] != self._backbone_features[-1].shape[-2:]:
            self._fail('ACT backbone and projection spatial shapes must match per camera')
        self._projected_features.append(feature)

    def _finish_forward(self, module, args, output):
        if not self._in_forward:
            return
        if output is None:
            # always_call=True invokes this after a model exception. Preserve
            # that exception and the previous completed capture, not partials.
            self.close()
            return
        if (len(self._backbone_features) != len(self.image_keys)
                or len(self._projected_features) != len(self.image_keys)):
            self._fail('ACT backbone/projection call counts must match configured cameras')
        backbone = dict(zip(self.image_keys, self._backbone_features, strict=True))
        projected = dict(zip(self.image_keys, self._projected_features, strict=True))
        self.latest = ACTFeatureCapture(
            camera_backbone_features=backbone,
            camera_projected_features=projected,
            backbone_rms={key: _channel_rms(value) for key, value in backbone.items()},
            projected_rms={key: _channel_rms(value) for key, value in projected.items()},
            grid_shapes={key: tuple(value.shape[-2:]) for key, value in backbone.items()},
        )
        self.capture_count += 1
        self._backbone_features = []
        self._projected_features = []
        self._in_forward = False
