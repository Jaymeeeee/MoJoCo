"""Paired image interventions and stateless ACT action-chunk inference.

These helpers measure sensitivity at fixed observations. They do not advance a
simulator or the policy's cached-action/temporal-ensemble state.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class ActionChunkPredictions:
    """Independent predictions in saved normalization space and physical units."""

    normalized: np.ndarray
    physical: np.ndarray


def predict_action_chunks(adapter, observations):
    """Predict a batch of independent chunks without selecting/executing actions.

    Each observation uses the adapter's saved preprocessing. The model is called
    via ``predict_action_chunk`` directly, bypassing both its action queue and
    temporal ensembler. The ACT adapter's processors must be stateless, as in
    the supported local checkpoints. Returned owned float32 arrays have shape
    ``(batch, chunk_size, action_dimension)``.
    """
    torch = adapter._torch
    observations = list(observations)
    if not observations:
        raise ValueError('At least one observation is required')
    if getattr(adapter.policy, 'training', False):
        raise ValueError('Ablation inference requires the policy in evaluation mode')
    if len(adapter.action_shape) != 1:
        raise ValueError('Ablation inference requires vector actions')
    expected = (len(observations), adapter.chunk_size, *adapter.action_shape)
    with torch.inference_mode():
        prepared = [adapter.preprocessor(adapter._prepare_observation(observation))
                    for observation in observations]
        # LeRobot pipelines also return transition placeholders such as
        # action=None. Only configured observation features enter the model.
        keys = tuple(adapter.input_features)
        if any(not set(keys).issubset(batch) for batch in prepared):
            raise ValueError('Preprocessed observations are missing configured features')
        for batch in prepared:
            for key in keys:
                value = batch[key]
                if not isinstance(value, torch.Tensor) or value.ndim < 1 or value.shape[0] != 1:
                    raise ValueError(f'Preprocessed feature {key} must be a batch-one tensor')
        batch = {key: torch.cat([item[key] for item in prepared], dim=0) for key in keys}
        prediction = adapter.policy.predict_action_chunk(batch)
        if not isinstance(prediction, torch.Tensor) or tuple(prediction.shape) != expected:
            raise ValueError(f'ACT chunk shape must be {expected}; got {getattr(prediction, "shape", None)}')
        if not bool(torch.isfinite(prediction).all()):
            raise ValueError('ACT produced nonfinite normalized action predictions')
        # Copy before postprocessing, including protection from an in-place
        # unnormalizer. Flatten only the batch/time axes for saved processors.
        normalized = prediction.detach().to(device='cpu', dtype=torch.float32).numpy().copy()
        flat = prediction.reshape(-1, adapter.action_shape[0]).clone()
        physical_tensor = adapter.postprocessor(flat)
        expected_flat = (expected[0] * expected[1], expected[2])
        if not isinstance(physical_tensor, torch.Tensor) or tuple(physical_tensor.shape) != expected_flat:
            raise ValueError(f'ACT postprocessor shape must be {expected_flat}')
        physical = physical_tensor.detach().to(device='cpu', dtype=torch.float32).numpy().copy().reshape(expected)
    if not np.isfinite(physical).all():
        raise ValueError('ACT produced nonfinite physical action predictions')
    return ActionChunkPredictions(normalized=normalized, physical=physical)


def mask_camera(observation, camera, mask, fill='mean'):
    """Replace selected RGB pixels in one copied camera image.

    ``mean`` uses the source image's rounded per-channel mean; ``black`` uses
    zero RGB. Unmasked camera/state arrays are shared read-only by convention;
    neither the supplied dictionary nor any of its arrays is mutated.
    """
    image = np.asarray(observation[camera])
    mask = np.asarray(mask)
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
        raise ValueError('Camera image must be HWC RGB uint8')
    if mask.dtype != np.bool_ or mask.shape != image.shape[:2]:
        raise ValueError('Mask must be a boolean array matching image height and width')
    if fill == 'mean':
        color = np.rint(image.mean(axis=(0, 1), dtype=np.float64)).astype(np.uint8)
    elif fill == 'black':
        color = np.zeros(3, dtype=np.uint8)
    else:
        raise ValueError('fill must be mean or black')
    masked = image.copy()
    masked[mask] = color
    result = dict(observation)
    result[camera] = masked
    return result


def random_patch_mask(shape, fraction, rng, grid_size=8):
    """Choose a fixed number of equal-size spatial cells using a supplied RNG.

    ``shape`` is ``(height, width)``. The masked cell count is the nearest integer
    to ``fraction * grid_size**2``. Reuse the returned mask for paired cameras.
    """
    if len(shape) != 2 or any(isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n <= 0 for n in shape):
        raise ValueError('shape must contain positive integer height and width')
    if isinstance(grid_size, bool) or not isinstance(grid_size, (int, np.integer)) or grid_size < 1:
        raise ValueError('grid_size must be a positive integer')
    if any(n % grid_size for n in shape):
        raise ValueError('Image height and width must be divisible by grid_size')
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float, np.number)) or not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError('fraction must be a finite number between zero and one')
    count = round(float(fraction) * grid_size**2)
    cells = np.zeros(grid_size**2, dtype=bool)
    cells[rng.choice(grid_size**2, size=count, replace=False)] = True
    cells = cells.reshape(grid_size, grid_size)
    return cells.repeat(shape[0] // grid_size, axis=0).repeat(shape[1] // grid_size, axis=1)
