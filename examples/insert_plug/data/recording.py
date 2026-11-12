"""Sample observations on real physics ticks independently of control updates."""

from __future__ import annotations

import math

LEGACY_ACTION_TIMING = 'active_control_target_at_observation; may change before next observation'
ALIGNED_ACTION_TIMING = 'active_control_target_at_observation; held_until_next_observation'


class RecordingSampler:
    """Select the nearest physics tick to each nominal sample time.

    For 15 Hz observations and 1 ms physics this selects 0, 67, 133, 200, ...
    ticks. Timestamps are always the actual simulator times; neither images nor
    states are interpolated. A missed scheduled tick is an error, not a silently
    skipped frame. The schedule is anchored once so control/phase boundaries do
    not restart sampling or introduce drift.
    """

    def __init__(self, sample_hz, physics_timestep, start_time=0.0):
        self.sample_hz = float(sample_hz)
        self.physics_timestep = float(physics_timestep)
        self.start_time = float(start_time)
        if not math.isfinite(self.physics_timestep) or self.physics_timestep <= 0:
            raise ValueError('physics_timestep must be finite and positive')
        if not math.isfinite(self.sample_hz) or not 0 < self.sample_hz <= 1 / self.physics_timestep:
            raise ValueError('sample_hz must be positive and no greater than the physics frequency')
        if not math.isfinite(self.start_time) or self.start_time < 0:
            raise ValueError('start_time must be finite and nonnegative')
        self.samples = 0
        self._next_tick = 0
        self._last_tick = -1

    def due(self, actual_time):
        """Consume a scheduled observation at the supplied actual physics time."""
        elapsed_ticks = (float(actual_time) - self.start_time) / self.physics_timestep
        if not math.isfinite(elapsed_ticks):
            raise ValueError('Recording time must be finite')
        tick = int(round(elapsed_ticks))
        if tick < 0 or not math.isclose(elapsed_ticks, tick, abs_tol=1e-6, rel_tol=0):
            raise RuntimeError('Recording time is not on the configured physics grid')
        if tick < self._last_tick:
            raise RuntimeError('Recording time moved backwards')
        self._last_tick = tick
        if tick < self._next_tick:
            return False
        if tick > self._next_tick:
            raise RuntimeError(f'Missed recording tick {self._next_tick}; received {tick}')
        self.samples += 1
        self._next_tick = int(math.floor(
            self.samples / (self.sample_hz * self.physics_timestep) + 0.5))
        return True

    def metadata(self, control_hz):
        """Describe the active target without promising an unaligned hold.

        Controls start on the simulation's zero-origin grid and keep their
        target for an integer number of physics ticks. Equal nominal rates
        alone do not establish this contract if the recording origin is
        shifted or the nominal period lies between physics ticks.
        """
        control_hz = float(control_hz)
        if not math.isfinite(control_hz) or control_hz <= 0:
            raise ValueError('control_hz must be finite and positive')
        period_ticks = 1 / (control_hz * self.physics_timestep)
        origin_periods = self.start_time * control_hz
        aligned = (
            math.isclose(self.sample_hz, control_hz, rel_tol=1e-12, abs_tol=0)
            and math.isclose(period_ticks, round(period_ticks), rel_tol=0, abs_tol=1e-6)
            and math.isclose(origin_periods, round(origin_periods), rel_tol=0, abs_tol=1e-8)
        )
        metadata = dict(
            sample_hz=self.sample_hz,
            control_hz=control_hz,
            physics_timestep=self.physics_timestep,
            sample_origin_s=self.start_time,
            sampling_timing='nearest_physics_tick',
            max_sample_time_error_s=self.physics_timestep / 2,
            action_timing=ALIGNED_ACTION_TIMING if aligned else LEGACY_ACTION_TIMING,
        )
        if aligned:
            metadata['action_hold_duration_s'] = 1 / control_hz
        return metadata
