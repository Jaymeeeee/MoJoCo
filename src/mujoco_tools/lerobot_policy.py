"""Load a local LeRobot ACT or Diffusion checkpoint for a simulator control loop.

This module is independent of MuJoCo scenes and imports PyTorch/LeRobot lazily.
It accepts unbatched RGB uint8 images, optional paired float32 metric depth,
and physical state vectors, and returns an unbatched, unnormalized action.
The caller owns action units, limits and time.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from numbers import Real
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
    depth_map = config.get('depth_camera_map', {})
    if not isinstance(depth_map, dict) or not all(
            isinstance(rgb, str) and isinstance(depth, str) for rgb, depth in depth_map.items()):
        raise ValueError('depth_camera_map must map RGB feature names to depth feature names')
    depth_keys = set(depth_map.values())
    if depth_map:
        if config.get('depth_unit') != 'm':
            raise ValueError("RGB-D checkpoint depth_unit must be 'm' (metres)")
        if len(depth_keys) != len(depth_map) or set(depth_map) & depth_keys:
            raise ValueError('depth_camera_map must pair each RGB camera with a distinct depth feature')
    for name, feature in [*inputs.items(), *outputs.items()]:
        shape = feature.get('shape', [])
        if not shape or any(type(n) is not int or n <= 0 for n in shape):
            raise ValueError(f'Invalid checkpoint shape for {name}: {shape}')
        kind = feature.get('type')
        if name == 'action':
            valid = kind == 'ACTION' and len(shape) == 1
        elif kind == 'VISUAL':
            if name in depth_keys:
                valid = name.startswith('observation.depth.') and len(shape) == 3 and shape[0] == 1
            else:
                valid = name.startswith('observation.images.') and len(shape) == 3 and shape[0] == 3
        else:
            valid = kind in {'STATE', 'ENV'} and name in {
                'observation.state', 'observation.environment_state'} and len(shape) == 1
        if not valid:
            raise ValueError(f'Unsupported policy feature {name}: {feature}; '
                             'expected RGB images, explicitly paired metric depth, state vectors, '
                             'and one action vector')
    if depth_map:
        rgb_keys = {name for name, feature in inputs.items()
                    if feature.get('type') == 'VISUAL' and name not in depth_keys}
        if set(depth_map) != rgb_keys:
            raise ValueError('depth_camera_map must cover every RGB camera without adding cameras')
        for rgb_key, depth_key in depth_map.items():
            if depth_key not in inputs:
                raise ValueError(f'Missing paired depth input feature: {depth_key}')
            if inputs[depth_key]['shape'][1:] != inputs[rgb_key]['shape'][1:]:
                raise ValueError(f'RGB and depth spatial shapes must match: {rgb_key}, {depth_key}')
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
            'shared mujoco Python environment. Use scripts/setup_mujoco_env.sh. '
            f'Original import error: {exc}') from exc
    return torch, lerobot, ACTConfig, ACTPolicy, load_pretrained_policy_processors


def _load_diffusion_backend():
    """Keep optional Diffusion dependencies out of the ACT loading path."""
    try:
        import torch
        import lerobot
        from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
        from lerobot.processor import load_pretrained_policy_processors
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            'Diffusion deployment requires compatible PyTorch, torchvision, LeRobot and '
            'diffusers in the shared mujoco Python environment. '
            f'Use scripts/setup_mujoco_env.sh. Original import error: {exc}') from exc
    return torch, lerobot, DiffusionConfig, DiffusionPolicy, load_pretrained_policy_processors


class LeRobotPolicy:
    """Scene-independent ACT/Diffusion adapter using saved processor statistics.

    ``n_action_steps`` controls how many cached actions LeRobot executes before
    predicting another chunk. It does not change the trained prediction horizon
    or simulator time. For Diffusion, ``chunk_size`` is the usable future span
    (``horizon - n_obs_steps + 1``); observation history stays owned by LeRobot.
    ACT-only ``temporal_ensemble_coeff=None`` preserves the checkpoint setting.
    A numeric override enables ensembling and requires ``n_action_steps=1``;
    zero uniformly weights the predictions aligned to the current control step.
    RGB inputs must be HWC uint8. RGB-D checkpoints also require their declared
    depth inputs as float32 metres, in HW, HWC with one channel, or CHW form.
    Depth values pass unchanged to the saved preprocessor; only RGB is divided
    by 255. Extra observation keys are never forwarded.
    """

    def __init__(self, checkpoint, device='auto', n_action_steps=1, temporal_ensemble_coeff=None):
        self.checkpoint = resolve_checkpoint(checkpoint)
        config_bytes = (self.checkpoint / 'config.json').read_bytes()
        saved = json.loads(config_bytes)
        self.policy_type = saved.get('type')
        if self.policy_type not in {'act', 'diffusion'}:
            raise ValueError(f"Unsupported policy type {self.policy_type!r}; this adapter supports ACT and Diffusion")
        self.input_features, self.output_features = _feature_schema(saved)
        self.depth_camera_map = deepcopy(saved.get('depth_camera_map', {}))
        self.requires_depth = bool(self.depth_camera_map)
        self.depth_features = tuple(self.depth_camera_map.values())
        self.depth_unit = saved['depth_unit'] if self.requires_depth else None
        self.action_shape = tuple(self.output_features['action']['shape'])
        self.n_obs_steps = saved.get('n_obs_steps', 1 if self.policy_type == 'act' else None)
        if self.policy_type == 'diffusion':
            self.horizon = saved.get('horizon')
            if type(self.horizon) is not int or self.horizon < 1:
                raise ValueError('Diffusion checkpoint horizon must be a positive integer')
            if type(self.n_obs_steps) is not int or not 1 <= self.n_obs_steps <= self.horizon:
                raise ValueError('Diffusion checkpoint n_obs_steps must be an integer in [1, horizon]')
            self.chunk_size = self.horizon - self.n_obs_steps + 1
            if temporal_ensemble_coeff is not None:
                raise ValueError('temporal_ensemble_coeff is only supported for ACT, not Diffusion')
            ensemble_coeff = None
        else:
            self.chunk_size = saved.get('chunk_size')
            if type(self.chunk_size) is not int or self.chunk_size < 1:
                raise ValueError('ACT checkpoint chunk_size must be a positive integer')
            ensemble_coeff = (saved.get('temporal_ensemble_coeff')
                              if temporal_ensemble_coeff is None else temporal_ensemble_coeff)
        if ensemble_coeff is not None:
            if isinstance(ensemble_coeff, bool) or not isinstance(ensemble_coeff, Real):
                raise ValueError('temporal_ensemble_coeff must be a finite real number or None')
            try:
                ensemble_coeff = float(ensemble_coeff)
            except (OverflowError, ValueError) as exc:
                raise ValueError('temporal_ensemble_coeff must be a finite real number or None') from exc
            if not math.isfinite(ensemble_coeff):
                raise ValueError('temporal_ensemble_coeff must be a finite real number or None')
        self.temporal_ensemble_coeff = ensemble_coeff
        if type(n_action_steps) is not int or not 1 <= n_action_steps <= self.chunk_size:
            raise ValueError(f'n_action_steps must be an integer in [1, {self.chunk_size}]')
        if self.temporal_ensemble_coeff is not None and n_action_steps != 1:
            raise ValueError('ACT temporal ensembling requires n_action_steps=1')
        self.n_action_steps = n_action_steps
        backend = _load_backend if self.policy_type == 'act' else _load_diffusion_backend
        torch, lerobot, config_cls, policy_cls, load_processors = backend()
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
        if self.policy_type == 'act':
            config.temporal_ensemble_coeff = self.temporal_ensemble_coeff
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
            checkpoint=str(self.checkpoint), policy_type=self.policy_type, device=self.device,
            chunk_size=self.chunk_size, n_action_steps=self.n_action_steps,
            n_obs_steps=self.n_obs_steps,
            checkpoint_n_action_steps=saved.get('n_action_steps'),
            temporal_ensemble_coeff=self.temporal_ensemble_coeff,
            checkpoint_temporal_ensemble_coeff=saved.get('temporal_ensemble_coeff'),
            input_features=self.input_features, output_features=self.output_features,
            config_sha256=hashlib.sha256(config_bytes).hexdigest(),
            model_bytes=(self.checkpoint / 'model.safetensors').stat().st_size,
            lerobot_version=getattr(lerobot, '__version__', 'unknown'),
            lerobot_module=str(getattr(lerobot, '__file__', 'unknown')),
            torch_version=str(torch.__version__),
            rgb_input='HWC RGB uint8 -> CHW float32 / 255 -> saved preprocessor',
            requires_depth=self.requires_depth,
            depth_camera_map=deepcopy(self.depth_camera_map),
            depth_features=list(self.depth_features), depth_unit=self.depth_unit,
            depth_input=('HW/HWC1/CHW float32 metres -> CHW unchanged -> saved preprocessor'
                         if self.requires_depth else None),
            action_output='saved postprocessor -> unbatched float32 physical action',
        )
        if self.policy_type == 'diffusion':
            self._provenance.update(
                horizon=config.horizon,
                noise_scheduler_type=config.noise_scheduler_type,
                num_train_timesteps=config.num_train_timesteps,
                num_inference_steps=(config.num_train_timesteps if config.num_inference_steps is None
                                     else config.num_inference_steps),
                checkpoint_num_inference_steps=saved.get('num_inference_steps'),
                normalization_mapping=deepcopy(saved.get('normalization_mapping')),
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
            if name in self.depth_features:
                hw_shape = expected[1:]
                hwc_shape = (*hw_shape, 1)
                if value.dtype != np.float32 or value.shape not in (hw_shape, hwc_shape, expected):
                    raise ValueError(f'{name} must be metric depth float32 in metres with HW {hw_shape}, '
                                     f'HWC {hwc_shape}, or CHW {expected}; got {value.shape}, {value.dtype}')
                if value.shape == hw_shape:
                    value = value[None, ...]
                elif value.shape == hwc_shape:
                    value = value.transpose(2, 0, 1)
                # Keep full metric precision and own storage before saved processors run.
                array = np.array(value, dtype=np.float32, order='C', copy=True)
                if not np.isfinite(array).all() or np.any(array < 0):
                    raise ValueError(f'{name} contains nonfinite or negative metric depth values')
            elif feature['type'] == 'VISUAL':
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
                raise TypeError('Policy postprocessor did not return an action tensor')
            if tuple(result.shape) != (1, *self.action_shape):
                raise ValueError(f'Policy returned shape {tuple(result.shape)}; expected {(1, *self.action_shape)}')
            action = result[0].detach().to(device='cpu', dtype=self._torch.float32).numpy().copy()
        if not np.isfinite(action).all():
            raise ValueError('Policy returned nonfinite action values')
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
