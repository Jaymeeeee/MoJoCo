"""Physical grasp qualification and signed paired translation measurements.

Qualification is passive: no contacts, actions or simulator states are changed.
Call ``sample`` on independently forwarded physics snapshots, not stale derived
poses after integration. Aperture commands never enter the qualification rule.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy

import numpy as np


class GraspLiftMonitor:
    """Require a stable opposed grasp and physical clearance of the source.

    Every qualifying window covers at least ``hold_seconds`` including both
    endpoints. Drift is measured in TCP coordinates relative to the last sample
    in the window, following the existing expert's 0.2 mm grasp tolerance.
    Missing contacts or a sample gap greater than 1.5 sampling periods breaks a
    contact segment. A successful report selects the first stable grasp in the
    segment that actually leads to the first verified lift.
    """

    def __init__(self, hold_seconds=.2, drift_tolerance_m=.0002,
                 clearance_m=.005, sample_period_s=.001):
        values = (hold_seconds, drift_tolerance_m, clearance_m, sample_period_s)
        if not all(np.isfinite(v) and v > 0 for v in values):
            raise ValueError('Qualification thresholds must be finite and positive')
        self.hold_seconds, self.drift_tolerance_m, self.clearance_m, self.sample_period_s = (
            float(v) for v in values)
        self._window = deque()
        self._last_time = None
        self._segment_grasp = None
        self._lift_grasp = None
        self._grasp_events = []
        self._lift_event = None
        self._samples = 0
        self._contact_samples = 0
        self._max_tip_clearance = None

    def _break_segment(self):
        self._window.clear()
        self._segment_grasp = None

    def sample(self, time_s, tcp_xyz, tcp_rotation, peg_tip_xyz, source_rim_z,
               opposing_contacts, action_index):
        """Append one physics sample; ``action_index`` is zero-based or -1 initially."""
        tcp, rotation, tip = (np.asarray(v, dtype=float)
                              for v in (tcp_xyz, tcp_rotation, peg_tip_xyz))
        if tcp.shape != (3,) or tip.shape != (3,) or rotation.shape != (3, 3):
            raise ValueError('Expected TCP/tip shape (3,) and rotation shape (3, 3)')
        if not (all(np.isfinite(v).all() for v in (tcp, rotation, tip))
                and np.isfinite(time_s) and np.isfinite(source_rim_z)):
            raise ValueError('Physics samples must be finite')
        if isinstance(action_index, (bool, np.bool_)) or not isinstance(action_index, (int, np.integer)) or action_index < -1:
            raise ValueError('action_index must be an integer >= -1')
        contacts = np.asarray(opposing_contacts)
        if contacts.shape not in ((), (2,)) or contacts.dtype.kind != 'b':
            raise ValueError('opposing_contacts must be a boolean or two boolean pad contacts')
        both = bool(contacts.all())
        now = float(time_s)
        if self._last_time is not None:
            elapsed = now - self._last_time
            if elapsed <= 0:
                raise ValueError('Physics sample times must increase strictly')
            if elapsed > 1.5 * self.sample_period_s + 1e-10:
                self._break_segment()
        self._last_time = now
        self._samples += 1
        clearance = float(tip[2] - source_rim_z)
        self._max_tip_clearance = (clearance if self._max_tip_clearance is None
                                   else max(clearance, self._max_tip_clearance))
        if not both:
            self._break_segment()
            return
        self._contact_samples += 1
        offset = rotation.T @ (tip - tcp)
        row = dict(time_s=now, offset=offset, clearance=clearance,
                   action_index=int(action_index), tcp=tcp.copy(), tip=tip.copy())
        self._window.append(row)
        # Retain the oldest sample required to cover the complete duration.
        # The tolerance only absorbs floating-point timestamps, not missing time.
        cutoff = now - self.hold_seconds
        while len(self._window) >= 2 and self._window[1]['time_s'] <= cutoff + 1e-10:
            self._window.popleft()
        start = self._window[0]
        span = now - start['time_s']
        if span < self.hold_seconds - 1e-10:
            return
        offsets = np.asarray([entry['offset'] for entry in self._window])
        drift = float(np.max(np.linalg.norm(offsets - offset, axis=1)))
        if drift > self.drift_tolerance_m + 1e-12:
            return
        minimum_clearance = min(entry['clearance'] for entry in self._window)
        event = dict(start_time_s=start['time_s'], confirmed_time_s=now,
                     start_action_index=start['action_index'],
                     confirmed_action_index=int(action_index), window_samples=len(self._window),
                     window_seconds=span, max_relative_drift_m=drift,
                     min_tip_clearance_m=minimum_clearance,
                     start_tcp_m=start['tcp'].tolist(), confirmed_tcp_m=tcp.tolist(),
                     start_peg_tip_m=start['tip'].tolist(), confirmed_peg_tip_m=tip.tolist())
        if self._segment_grasp is None:
            self._segment_grasp = event
            self._grasp_events.append(event)
        if self._lift_event is None and minimum_clearance >= self.clearance_m - 1e-12:
            self._lift_event = event
            self._lift_grasp = self._segment_grasp

    def report(self):
        """Return a JSON-safe snapshot; a stable contact alone never means passed."""
        first = self._grasp_events[0] if self._grasp_events else None
        return deepcopy(dict(
            passed=self._lift_event is not None,
            grasp_event=self._lift_grasp if self._lift_event is not None else first,
            first_observed_grasp_event=first, grasp_events=self._grasp_events,
            grasp_leads_to_verified_lift=self._lift_event is not None,
            lift_event=self._lift_event,
            samples=self._samples, opposing_contact_samples=self._contact_samples,
            max_tip_clearance_m=self._max_tip_clearance,
            criteria=dict(hold_seconds=self.hold_seconds, drift_tolerance_m=self.drift_tolerance_m,
                          clearance_m=self.clearance_m, sample_period_s=self.sample_period_s,
                          maximum_sample_gap_s=1.5 * self.sample_period_s,
                          drift_reference='last sample of each window, in TCP coordinates',
                          lift_reference='peg tip above source rim throughout the complete window'),
        ))


def displacement_metrics(base_point, shifted_point, delta):
    """World-frame displacement, signed response along the object shift, and errors."""
    base, shifted, delta = (np.asarray(v, dtype=float) for v in (base_point, shifted_point, delta))
    if any(v.shape != (3,) or not np.isfinite(v).all() for v in (base, shifted, delta)):
        raise ValueError('Points and delta must be finite vectors of shape (3,)')
    distance = float(np.linalg.norm(delta))
    if distance <= 0:
        raise ValueError('delta must be nonzero')
    direction = delta / distance
    displacement = shifted - base
    along = float(displacement @ direction)
    return dict(baseline_tcp_m=base.tolist(), shifted_tcp_m=shifted.tolist(),
                displacement_m=displacement.tolist(), along_m=along, gain=along / distance,
                orthogonal_m=float(np.linalg.norm(displacement - along * direction)),
                residual_m=float(np.linalg.norm(displacement - delta)))


def paired_grasp_metrics(base_fk, shifted_fk, base_actual, shifted_actual, delta,
                         baseline_event, shifted_event=None):
    """Compare equal control intervals and, separately, physical grasp events.

    Arrays contain one sample per executed action, not overlapping full chunks.
    FK samples are absolute command targets; actual samples are states at each
    control interval's end. Event indices identify the interval containing the
    1 kHz qualification time; exact event poses remain in the event records.
    """
    arrays = [np.asarray(v, dtype=float) for v in (base_fk, shifted_fk, base_actual, shifted_actual)]
    if (arrays[0].ndim != 2 or arrays[0].shape[1:] != (3,) or not len(arrays[0])
            or any(v.shape != arrays[0].shape or not np.isfinite(v).all() for v in arrays)):
        raise ValueError('All trajectories must be finite and have the same nonempty shape (T, 3)')
    # Validate delta even when an event is missing.
    displacement_metrics(np.zeros(3), np.zeros(3), delta)

    def index(event):
        if event is None:
            return None
        i = event['confirmed_action_index']
        if isinstance(i, bool) or not isinstance(i, (int, np.integer)) or not 0 <= i < len(arrays[0]):
            raise ValueError('Event action index is outside the trajectory')
        return int(i)

    base_index, shifted_index = index(baseline_event), index(shifted_event)

    def endpoint(bi, si):
        if bi is None or si is None:
            return None
        return dict(base_action_index=bi, shifted_action_index=si, same_action_index=bi == si,
                    fk=displacement_metrics(arrays[0][bi], arrays[1][si], delta),
                    actual=displacement_metrics(arrays[2][bi], arrays[3][si], delta))

    return dict(
        translation_m=np.asarray(delta, dtype=float).tolist(),
        baseline_confirmed_grasp=endpoint(base_index, base_index),
        independent_confirmed_grasps=endpoint(base_index, shifted_index),
        actual_sample_timing='end of control interval containing grasp confirmation; not exact event time',
        action_semantics='separate FK of each executed absolute joint target; never sum absolute joints',
    )
