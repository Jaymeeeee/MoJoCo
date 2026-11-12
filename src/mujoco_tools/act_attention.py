"""Passively capture LeRobot ACT decoder cross-attention during inference.

The returned weights address spatially indexed *encoder memory*. Encoder
self-attention has already mixed the image and state tokens, so these maps are
not causal pixel attribution and do not describe an individual action dimension.
Only the final decoder layer is captured; heads are averaged by PyTorch.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ACTAttentionCapture:
    """One completed batch-one prediction, with raw attention for every query.

    ``token_weights`` has shape (queries, memory tokens); camera arrays have
    shape (queries, height, width). Camera and special-token weights partition
    each attention row, whose sum is one. Arrays are read-only CPU copies.
    """

    camera_weights: dict[str, np.ndarray]
    token_weights: np.ndarray
    special_token_weights: dict[str, np.ndarray]
    grid_shapes: dict[str, tuple[int, int]]
    decoder_layer_index: int


class ACTAttentionTrace:
    """Read existing ACT attention outputs without changing model computation.

    Use as ``with ACTAttentionTrace(policy.model) as trace: ...``. A capture is
    made only when the ACT model actually runs. Cached ``select_action`` calls
    therefore leave ``latest`` and ``capture_count`` unchanged. The caller must
    associate query indices with cached actions and retain their source images.

    The supported layout is the local LeRobot ACT implementation: latent,
    optional robot/environment state, then row-major image features in config
    order. This collector supports evaluation mode and batch size one. It never
    requests new attention weights or changes forward arguments/return values.
    """

    def __init__(self, model, *, image_keys=None):
        import torch

        self._torch = torch
        self.model = model
        config = getattr(model, 'config', None)
        if getattr(config, 'type', None) != 'act':
            raise ValueError('ACT attention requires a LeRobot ACT model with config.type="act"')
        configured_keys = tuple(getattr(config, 'image_features', {}))
        if not configured_keys or len(set(configured_keys)) != len(configured_keys):
            raise ValueError('ACT attention requires at least one configured image feature')
        if image_keys is not None and tuple(image_keys) != configured_keys:
            raise ValueError('image_keys must match ACT config.image_features in the same order')
        self.image_keys = configured_keys
        self._special_names = ['latent']
        if getattr(config, 'robot_state_feature', None):
            self._special_names.append('state')
        if getattr(config, 'env_state_feature', None):
            self._special_names.append('environment_state')
        self._chunk_size = getattr(config, 'chunk_size', None)
        if type(self._chunk_size) is not int or self._chunk_size < 1:
            raise ValueError('ACT attention requires a positive integer config.chunk_size')
        self._backbone = getattr(model, 'backbone', None)
        layers = getattr(getattr(model, 'decoder', None), 'layers', None)
        if not isinstance(self._backbone, torch.nn.Module) or layers is None or not len(layers):
            raise ValueError('Unsupported ACT architecture: expected backbone and decoder.layers')
        self.decoder_layer_index = len(layers) - 1
        self._cross_attention = getattr(layers[-1], 'multihead_attn', None)
        if not isinstance(self._cross_attention, torch.nn.MultiheadAttention):
            raise ValueError('Unsupported ACT architecture: final decoder cross-attention must be MultiheadAttention')
        self.capture_count = 0
        self.latest: ACTAttentionCapture | None = None
        self._handles = []
        self._grid_shapes = []
        self._pending = None
        self._in_forward = False

    def __enter__(self):
        if self._handles:
            raise RuntimeError('ACTAttentionTrace is already active')
        if self.model.training:
            raise ValueError('ACT attention visualization requires model.eval()')
        try:
            self._handles.append(self.model.register_forward_pre_hook(self._begin_forward))
            self._handles.append(self._backbone.register_forward_hook(self._capture_grid))
            self._handles.append(self._cross_attention.register_forward_hook(self._capture_weights))
            self._handles.append(self.model.register_forward_hook(self._finish_forward, always_call=True))
        except Exception:
            self.close()
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        """Remove all hooks, preserving the last successful capture."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._grid_shapes.clear()
        self._pending = None
        self._in_forward = False

    def clear(self):
        """Forget the latest capture while keeping the monotonic capture count."""
        if self._in_forward:
            raise RuntimeError('Cannot clear ACT attention during a model forward')
        self.latest = None

    def _fail(self, message):
        self.close()
        raise ValueError(message)

    def _begin_forward(self, module, args):
        if module.training:
            self._fail('ACT attention visualization requires model.eval()')
        if self._in_forward:
            self._fail('ACT attention does not support nested or concurrent model forwards')
        self._in_forward = True
        self._grid_shapes = []
        self._pending = None

    def _capture_grid(self, module, args, output):
        if not self._in_forward:
            return
        feature_map = output.get('feature_map') if isinstance(output, dict) else None
        if not isinstance(feature_map, self._torch.Tensor) or feature_map.ndim != 4:
            self._fail('Unsupported ACT backbone output: expected feature_map with shape (B, C, H, W)')
        if feature_map.shape[0] != 1:
            self._fail('ACT attention currently supports batch size one')
        if min(feature_map.shape[-2:]) < 1:
            self._fail('ACT backbone produced an empty spatial feature map')
        self._grid_shapes.append(tuple(feature_map.shape[-2:]))
        if len(self._grid_shapes) > len(self.image_keys):
            self._fail('ACT backbone was called more than once per configured camera')

    def _capture_weights(self, module, args, output):
        if not self._in_forward:
            return
        if self._pending is not None:
            self._fail('ACT final decoder cross-attention was called more than once')
        weights = output[1] if isinstance(output, tuple) and len(output) == 2 else None
        if not isinstance(weights, self._torch.Tensor):
            self._fail('ACT decoder did not return attention weights; passive tracing requires need_weights=True')
        if weights.ndim != 3 or weights.shape[0] != 1:
            self._fail('ACT attention expects head-averaged weights with shape (1, queries, memory tokens)')
        if weights.shape[1] != self._chunk_size:
            self._fail('ACT attention query count does not match config.chunk_size')
        if len(self._grid_shapes) != len(self.image_keys):
            self._fail('ACT backbone call count does not match configured image features')
        expected_tokens = len(self._special_names) + sum(h * w for h, w in self._grid_shapes)
        if weights.shape[2] != expected_tokens:
            self._fail(f'ACT memory token count mismatch: expected {expected_tokens}, got {weights.shape[2]}')
        tokens = weights[0].detach().float().cpu().numpy().copy()
        if not np.isfinite(tokens).all() or np.any(tokens < 0):
            self._fail('ACT attention weights must be finite and nonnegative')
        if not np.allclose(tokens.sum(axis=-1), 1, atol=2e-3, rtol=2e-3):
            self._fail('ACT attention rows do not sum to one; check evaluation mode and attention implementation')
        tokens.setflags(write=False)
        special = {name: tokens[:, i] for i, name in enumerate(self._special_names)}
        cameras = {}
        offset = len(self._special_names)
        for key, (height, width) in zip(self.image_keys, self._grid_shapes, strict=True):
            cameras[key] = tokens[:, offset:offset + height * width].reshape(-1, height, width)
            offset += height * width
        self._pending = ACTAttentionCapture(
            camera_weights=cameras, token_weights=tokens,
            special_token_weights=special,
            grid_shapes=dict(zip(self.image_keys, self._grid_shapes, strict=True)),
            decoder_layer_index=self.decoder_layer_index,
        )

    def _finish_forward(self, module, args, output):
        if not self._in_forward:
            return
        if output is None:
            # PyTorch calls this hook with None if the model raised. Leave the
            # original exception intact and do not publish a partial capture.
            self.close()
            return
        if self._pending is None:
            self._fail('ACT model completed without a final decoder cross-attention capture')
        self.latest = self._pending
        self.capture_count += 1
        self._pending = None
        self._grid_shapes.clear()
        self._in_forward = False
