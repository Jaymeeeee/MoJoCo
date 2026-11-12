"""Capture the image encoder's spatial pooling in LeRobot Diffusion inference.

This Diffusion architecture uses convolutional encoders and a conditional U-Net,
not an action-query Transformer. Each SpatialSoftmax channel is a separately
normalized probability map whose expectation is one learned keypoint. The mean
map below weights keypoints equally; it is neither action-conditioned attention
nor causal pixel attribution. Every camera's mean map sums to one, so total map
mass cannot rank cameras by their influence on actions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DiffusionAttentionCapture:
    """One completed pass through all configured image encoders.

    Camera weights have shape ``(keypoints, height, width)``; mean weights have
    shape ``(height, width)``. Keypoints are the actual SpatialSoftmax outputs,
    with shape ``(keypoints, 2)`` in normalized ``(x, y)`` coordinates. All arrays
    are read-only float32 CPU copies. This capture concerns visual encoding;
    the later diffusion sampler may still fail. Archive it only after the
    caller's action-selection call has returned successfully.
    """

    camera_weights: dict[str, np.ndarray]
    mean_weights: dict[str, np.ndarray]
    keypoints: dict[str, np.ndarray]
    grid_shapes: dict[str, tuple[int, int]]
    keypoint_counts: dict[str, int]
    keypoint_reconstruction_max_abs_error: dict[str, float]


class DiffusionAttentionTrace:
    """Read real spatial logits without rerunning an encoder or changing it.

    Use ``with DiffusionAttentionTrace(adapter.policy.diffusion) as trace``.
    Hooks on each learned 1x1 projection read its actual logits, reconstruct the
    channel-wise softmax, and check the result against the pool's actual output.
    Captures are published only after every encoder returns. ``select_action``
    calls that consume cached actions do not create additional captures.

    The supported local architecture has separate encoders, batch size one,
    one observation step, and no image crop or resize. Rejecting other layouts
    avoids silently overlaying weights on the wrong images. No forward method,
    arguments, return value, parameters, mode, or RNG state is modified.
    """

    def __init__(self, model, *, image_keys=None):
        import torch

        self._torch = torch
        self.model = model
        config = getattr(model, 'config', None)
        if getattr(config, 'type', None) != 'diffusion':
            raise ValueError('Diffusion spatial tracing requires config.type="diffusion"')
        if getattr(config, 'n_obs_steps', None) != 1:
            raise ValueError('Diffusion spatial tracing currently requires n_obs_steps=1')
        if getattr(config, 'use_separate_rgb_encoder_per_camera', None) is not True:
            raise ValueError('Diffusion spatial tracing requires separate RGB encoders per camera')
        if getattr(config, 'crop_shape', None) is not None or getattr(config, 'resize_shape', None) is not None:
            raise ValueError('Diffusion spatial overlays currently require no image crop or resize')
        configured = tuple(getattr(config, 'image_features', {}))
        if not configured or len(set(configured)) != len(configured):
            raise ValueError('Diffusion spatial tracing requires configured image features')
        if image_keys is not None and tuple(image_keys) != configured:
            raise ValueError('image_keys must match config.image_features in the same order')
        self.image_keys = configured
        encoders = getattr(model, 'rgb_encoder', None)
        if not isinstance(encoders, torch.nn.ModuleList) or len(encoders) != len(configured):
            raise ValueError('Unsupported Diffusion architecture: expected one rgb_encoder per camera')
        self._encoders = tuple(encoders)
        self._pools = []
        self._image_shapes = {}
        for key, encoder in zip(configured, encoders, strict=True):
            if getattr(encoder, 'resize', None) is not None or getattr(encoder, 'do_crop', False):
                raise ValueError('Diffusion spatial overlays require encoders without crop or resize')
            shape = tuple(getattr(config.image_features[key], 'shape', ()))
            if len(shape) != 3 or shape[0] != 3 or any(type(n) is not int or n < 1 for n in shape):
                raise ValueError(f'Invalid RGB image feature shape for {key}: {shape}')
            self._image_shapes[key] = shape
            pool = getattr(encoder, 'pool', None)
            if not isinstance(pool, torch.nn.Module):
                raise ValueError('Unsupported Diffusion encoder: expected SpatialSoftmax pool')
            dimensions = [getattr(pool, name, None) for name in ('_in_c', '_in_h', '_in_w', '_out_c')]
            if any(type(n) is not int or n < 1 for n in dimensions):
                raise ValueError('Unsupported SpatialSoftmax feature/keypoint dimensions')
            grid = getattr(pool, 'pos_grid', None)
            if not isinstance(grid, torch.Tensor) or tuple(grid.shape) != (pool._in_h * pool._in_w, 2):
                raise ValueError('Unsupported SpatialSoftmax normalized position grid')
            projection = getattr(pool, 'nets', None)
            if projection is not None and not isinstance(projection, torch.nn.Conv2d):
                raise ValueError('Unsupported SpatialSoftmax keypoint projection')
            self._pools.append(pool)
        if len({id(encoder) for encoder in encoders}) != len(encoders):
            raise ValueError('Separate RGB encoders must be distinct modules')
        if len({id(pool) for pool in self._pools}) != len(encoders):
            raise ValueError('Separate RGB encoders must have distinct SpatialSoftmax pools')
        self.capture_count = 0
        self.latest: DiffusionAttentionCapture | None = None
        self._handles = []
        self._pending = {}
        self._active_index = None
        self._active_weights = None
        self._active_expected_xy = None
        self._active_pool_result = None

    def __enter__(self):
        if self._handles:
            raise RuntimeError('DiffusionAttentionTrace is already active')
        if self.model.training or any(encoder.training for encoder in self._encoders):
            raise ValueError('Diffusion spatial tracing requires model.eval()')
        try:
            for index, (encoder, pool) in enumerate(zip(self._encoders, self._pools, strict=True)):
                self._handles.append(encoder.register_forward_pre_hook(
                    lambda module, args, i=index: self._begin_encoder(i, module, args)))
                self._handles.append(pool.register_forward_pre_hook(
                    lambda module, args, i=index: self._begin_pool(i, module, args)))
                if pool.nets is not None:
                    self._handles.append(pool.nets.register_forward_hook(
                        lambda module, args, output, i=index: self._capture_logits(i, output)))
                self._handles.append(pool.register_forward_hook(
                    lambda module, args, output, i=index: self._capture_keypoints(i, module, output),
                    always_call=True))
                self._handles.append(encoder.register_forward_hook(
                    lambda module, args, output, i=index: self._finish_encoder(i, output),
                    always_call=True))
        except Exception:
            self.close()
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _clear_pending(self):
        self._pending.clear()
        self._active_index = None
        self._active_weights = None
        self._active_expected_xy = None
        self._active_pool_result = None

    def close(self):
        """Remove hooks and partial work, preserving the last complete capture."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._clear_pending()

    def clear(self):
        """Clear pending/latest data while preserving the monotonic count."""
        if self._active_index is not None:
            raise RuntimeError('Cannot clear Diffusion spatial capture inside an encoder forward')
        self._clear_pending()
        self.latest = None

    def _fail(self, message):
        self.close()
        raise ValueError(message)

    @staticmethod
    def _readonly(array):
        result = np.asarray(array, dtype=np.float32).copy()
        result.setflags(write=False)
        return result

    def _begin_encoder(self, index, module, args):
        if self.model.training or module.training:
            self._fail('Diffusion spatial tracing requires model.eval()')
        if self._active_index is not None or index != len(self._pending):
            self._fail('Diffusion encoders must run once per camera in configured order')
        key = self.image_keys[index]
        image = args[0] if len(args) == 1 else None
        if not isinstance(image, self._torch.Tensor) or image.ndim != 4 or image.shape[0] != 1:
            self._fail('Diffusion spatial tracing currently supports RGB batch size one')
        if tuple(image.shape[1:]) != self._image_shapes[key]:
            self._fail(f'Diffusion encoder image shape differs from configured {key}')
        self._active_index = index
        self._active_weights = None
        self._active_expected_xy = None
        self._active_pool_result = None

    def _begin_pool(self, index, module, args):
        if self._active_index is None:
            return
        if self._active_index != index or module.training:
            self._fail('Unexpected SpatialSoftmax invocation or training mode')
        feature = args[0] if len(args) == 1 else None
        expected = (1, module._in_c, module._in_h, module._in_w)
        if not isinstance(feature, self._torch.Tensor) or tuple(feature.shape) != expected:
            self._fail(f'SpatialSoftmax input shape differs from expected {expected}')
        if module.nets is None:
            self._capture_logits(index, feature)

    def _capture_logits(self, index, logits):
        if self._active_index is None:
            return
        if self._active_index != index or self._active_weights is not None:
            self._fail('SpatialSoftmax logits must occur exactly once for the active camera')
        pool = self._pools[index]
        expected = (1, pool._out_c, pool._in_h, pool._in_w)
        if not isinstance(logits, self._torch.Tensor) or tuple(logits.shape) != expected:
            self._fail(f'SpatialSoftmax logits shape differs from expected {expected}')
        # This operates on the real projection output. It never reruns the
        # projection/backbone and performs no random operation.
        with self._torch.no_grad():
            attention = self._torch.softmax(logits.detach().reshape(pool._out_c, -1), dim=-1)
            expected_xy = attention @ pool.pos_grid
            weights = attention.reshape(pool._out_c, pool._in_h, pool._in_w).float().cpu().numpy()
            reconstructed = expected_xy.float().cpu().numpy()
        if not np.isfinite(weights).all() or np.any(weights < 0):
            self._fail('SpatialSoftmax probabilities must be finite and nonnegative')
        if not np.allclose(weights.sum(axis=(-2, -1)), 1, atol=2e-3, rtol=2e-3):
            self._fail('Each SpatialSoftmax keypoint probability map must sum to one')
        self._active_weights = self._readonly(weights)
        self._active_expected_xy = self._readonly(reconstructed)

    def _capture_keypoints(self, index, module, output):
        if self._active_index is None:
            return
        if output is None:
            self.close()
            return
        if self._active_index != index or self._active_weights is None:
            self._fail('SpatialSoftmax returned without captured logits')
        if self._active_pool_result is not None:
            self._fail('SpatialSoftmax must return once per active camera')
        if not isinstance(output, self._torch.Tensor) or tuple(output.shape) != (1, module._out_c, 2):
            self._fail('SpatialSoftmax output must have shape (1, keypoints, 2)')
        points = output[0].detach().float().cpu().numpy()
        if not np.isfinite(points).all():
            self._fail('SpatialSoftmax keypoints must be finite')
        error = float(np.max(np.abs(points - self._active_expected_xy)))
        if not np.allclose(points, self._active_expected_xy, rtol=2e-3, atol=2e-5):
            self._fail('Captured probabilities do not reconstruct the actual SpatialSoftmax keypoints')
        self._active_pool_result = dict(
            weights=self._active_weights,
            mean=self._readonly(self._active_weights.mean(axis=0)),
            keypoints=self._readonly(points),
            error=error,
        )

    def _finish_encoder(self, index, output):
        if self._active_index is None:
            return
        if output is None:
            self.close()
            return
        if self._active_index != index or self._active_pool_result is None:
            self._fail('Diffusion encoder completed without validated spatial pooling')
        self._pending[self.image_keys[index]] = self._active_pool_result
        self._active_index = None
        self._active_weights = None
        self._active_expected_xy = None
        self._active_pool_result = None
        if len(self._pending) == len(self.image_keys):
            self.latest = DiffusionAttentionCapture(
                camera_weights={key: row['weights'] for key, row in self._pending.items()},
                mean_weights={key: row['mean'] for key, row in self._pending.items()},
                keypoints={key: row['keypoints'] for key, row in self._pending.items()},
                grid_shapes={key: row['weights'].shape[-2:] for key, row in self._pending.items()},
                keypoint_counts={key: row['weights'].shape[0] for key, row in self._pending.items()},
                keypoint_reconstruction_max_abs_error={key: row['error'] for key, row in self._pending.items()},
            )
            self.capture_count += 1
            self._pending.clear()
