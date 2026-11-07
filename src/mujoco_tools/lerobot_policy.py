"""Load a local LeRobot ACT checkpoint for a simulator's observation/action loop.

This module is independent of MuJoCo scenes and imports PyTorch/LeRobot lazily.
It accepts unbatched RGB uint8 images and physical state vectors, and returns
an unbatched, unnormalized action. The caller owns action units, limits and time.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def resolve_checkpoint(checkpoint) -> Path:
    """Accept either a training-step directory or its pretrained_model child."""
    path = Path(checkpoint).expanduser().resolve()
    if (path / 'pretrained_model').is_dir():
        path /= 'pretrained_model'
    required = ('config.json', 'model.safetensors', 'policy_preprocessor.json',
                'policy_postprocessor.json')
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f'Incomplete local LeRobot checkpoint at {path}; missing {", ".join(missing)}. '
            'Supply the whole pretrained_model directory, including processor statistics.')
    for filename in required[2:]:
        document = json.loads((path / filename).read_text())
        for step in document.get('steps', []):
            state_file = step.get('state_file')
            if state_file:
                state_path = (path / state_file).resolve()
                if not state_path.is_relative_to(path):
                    raise ValueError(f'{filename}: processor state_file escapes checkpoint directory')
                if not state_path.is_file():
                    raise FileNotFoundError(f'Missing saved processor statistics: {state_path}')
    return path


def _feature_schema(config):
    """Reject unsupported layouts before importing heavy policy dependencies."""
    inputs, outputs = config.get('input_features'), config.get('output_features')
    if not isinstance(inputs, dict) or not inputs:
        raise ValueError('Checkpoint must declare nonempty input_features')
    if not isinstance(outputs, dict) or set(outputs) != {'action'}:
        raise ValueError('Checkpoint must declare exactly one output feature: action')
    for name, feature in [*inputs.items(), *outputs.items()]:
        shape = feature.get('shape', [])
        if not shape or any(type(n) is not int or n <= 0 for n in shape):
            raise ValueError(f'Invalid checkpoint shape for {name}: {shape}')
        kind = feature.get('type')
        if name == 'action':
            valid = kind == 'ACTION' and len(shape) == 1
        elif kind == 'VISUAL':
            valid = name.startswith('observation.images.') and len(shape) == 3 and shape[0] == 3
        else:
            valid = kind in {'STATE', 'ENV'} and name in {
                'observation.state', 'observation.environment_state'} and len(shape) == 1
        if not valid:
            raise ValueError(f'Unsupported ACT feature {name}: {feature}; '
                             'expected RGB images, state vectors, and one action vector')
    return deepcopy(inputs), deepcopy(outputs)


def _load_backend():
    try:
        import torch
        import lerobot
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.processor import load_pretrained_policy_processors
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            'ACT deployment requires compatible PyTorch, torchvision, and LeRobot in the '
            'inference Python environment. Use the project deployment launcher/setup; '
            'do not install these into the simulator environment to repair this error. '
            f'Original import error: {exc}') from exc
    return torch, lerobot, ACTConfig, ACTPolicy, load_pretrained_policy_processors


class LeRobotPolicy:
    """Scene-independent ACT adapter using the checkpoint's saved processors.

    ``n_action_steps`` controls how many cached actions LeRobot executes before
    predicting another chunk. It does not change chunk_size or simulator time.
    All input images must be HWC RGB uint8; accepting only this form avoids
    accidental double normalization. Extra observation keys are never forwarded.
    """

    def __init__(self, checkpoint, device='auto', n_action_steps=1):
        self.checkpoint = resolve_checkpoint(checkpoint)
        config_bytes = (self.checkpoint / 'config.json').read_bytes()
        saved = json.loads(config_bytes)
        if saved.get('type') != 'act':
            raise ValueError(f"Unsupported policy type {saved.get('type')!r}; this adapter currently supports ACT")
        self.input_features, self.output_features = _feature_schema(saved)
        self.action_shape = tuple(self.output_features['action']['shape'])
        self.chunk_size = saved.get('chunk_size')
        if type(self.chunk_size) is not int or self.chunk_size < 1:
            raise ValueError('ACT checkpoint chunk_size must be a positive integer')
        if type(n_action_steps) is not int or not 1 <= n_action_steps <= self.chunk_size:
            raise ValueError(f'n_action_steps must be an integer in [1, {self.chunk_size}]')
        if saved.get('temporal_ensemble_coeff') is not None and n_action_steps != 1:
            raise ValueError('ACT temporal ensembling requires n_action_steps=1')
        self.n_action_steps = n_action_steps
        torch, lerobot, config_cls, policy_cls, load_processors = _load_backend()
        self._torch = torch
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = str(torch.device(device))
        if self.device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but this PyTorch environment cannot access CUDA; use device=cpu or a CUDA-enabled environment')

        # Do not alter the checkpoint. Loading a full trained backbone makes a
        # second ImageNet download unnecessary and enables fully local inference.
        config = config_cls.from_pretrained(self.checkpoint, local_files_only=True)
        config.device = self.device
        config.n_action_steps = n_action_steps
        config.pretrained_backbone_weights = None
        self.policy = policy_cls.from_pretrained(
            self.checkpoint, config=config, local_files_only=True, strict=True)
        self.policy.eval()
        self.preprocessor, self.postprocessor = load_processors(
            self.checkpoint,
            preprocessor_overrides={'device_processor': {'device': self.device}},
            postprocessor_overrides={'device_processor': {'device': 'cpu'}},
        )
        self._provenance = dict(
            checkpoint=str(self.checkpoint), policy_type='act', device=self.device,
            chunk_size=self.chunk_size, n_action_steps=self.n_action_steps,
            checkpoint_n_action_steps=saved.get('n_action_steps'),
            temporal_ensemble_coeff=saved.get('temporal_ensemble_coeff'),
            input_features=self.input_features, output_features=self.output_features,
            config_sha256=hashlib.sha256(config_bytes).hexdigest(),
            model_bytes=(self.checkpoint / 'model.safetensors').stat().st_size,
            lerobot_version=getattr(lerobot, '__version__', 'unknown'),
            lerobot_module=str(getattr(lerobot, '__file__', 'unknown')),
            torch_version=str(torch.__version__),
            rgb_input='HWC RGB uint8 -> CHW float32 / 255 -> saved preprocessor',
            action_output='saved postprocessor -> unbatched float32 physical action',
        )
        self.reset()

    def reset(self):
        """Reset policy queues and both processor pipelines for a new episode."""
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()
        self.calls = 0
        self.total_inference_seconds = 0.0
        self.last_inference_seconds = 0.0

    def _prepare_observation(self, observation):
        batch = {}
        for name, feature in self.input_features.items():
            if name not in observation:
                raise ValueError(f'Missing policy observation: {name}')
            value = np.asarray(observation[name])
            expected = tuple(feature['shape'])
            if feature['type'] == 'VISUAL':
                image_shape = (expected[1], expected[2], expected[0])
                if value.shape != image_shape or value.dtype != np.uint8:
                    raise ValueError(f'{name} must be RGB uint8 with HWC shape {image_shape}; '
                                     f'got {value.shape}, {value.dtype}')
                array = np.ascontiguousarray(value.transpose(2, 0, 1), dtype=np.float32) / np.float32(255)
            else:
                if value.shape != expected or not np.issubdtype(value.dtype, np.number):
                    raise ValueError(f'{name} must have shape {expected} and numeric dtype; '
                                     f'got {value.shape}, {value.dtype}')
                # Own this memory: processors must not mutate simulator arrays.
                array = np.array(value, dtype=np.float32, copy=True)
                if not np.isfinite(array).all():
                    raise ValueError(f'{name} contains nonfinite state values')
            batch[name] = self._torch.from_numpy(array)
        return batch

    def select_action(self, observation) -> np.ndarray:
        """Return one physical action; the caller executes one control interval."""
        start = time.perf_counter()
        with self._torch.inference_mode():
            batch = self.preprocessor(self._prepare_observation(observation))
            normalized_action = self.policy.select_action(batch)
            result = self.postprocessor(normalized_action)
            if not isinstance(result, self._torch.Tensor):
                raise TypeError('ACT postprocessor did not return an action tensor')
            if tuple(result.shape) != (1, *self.action_shape):
                raise ValueError(f'ACT returned shape {tuple(result.shape)}; expected {(1, *self.action_shape)}')
            action = result[0].detach().to(device='cpu', dtype=self._torch.float32).numpy().copy()
        if not np.isfinite(action).all():
            raise ValueError('ACT returned nonfinite action values')
        self.last_inference_seconds = time.perf_counter() - start
        self.total_inference_seconds += self.last_inference_seconds
        self.calls += 1
        return action

    def metadata(self):
        """Serializable provenance and current episode's end-to-end inference time."""
        return dict(deepcopy(self._provenance), calls=self.calls,
                    total_inference_seconds=self.total_inference_seconds,
                    last_inference_seconds=self.last_inference_seconds,
                    mean_inference_seconds=(self.total_inference_seconds / self.calls if self.calls else None))
