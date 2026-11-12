"""Signed translation sensitivity of paired absolute-action FK trajectories.

Inputs are Cartesian targets obtained by forward kinematics of *each* absolute
joint command. The helpers never add joint targets or equate a commanded
aperture with a physically successful grasp.
"""
from __future__ import annotations

import numpy as np


def first_closure_index(actions, closure_width_m=.026, consecutive_samples=2):
    """First command in a persistent aperture-threshold crossing, or ``None``.

    Accept an aperture vector or an eight-dimensional action trajectory. This
    is a commanded-closure proxy only; contacts and object retention are absent.
    """
    values = np.asarray(actions, dtype=float)
    if values.ndim == 2 and values.shape[1] == 8:
        values = values[:, 7]
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError('Expected finite aperture vector or actions with shape (T, 8)')
    if not np.isfinite(closure_width_m) or closure_width_m <= 0:
        raise ValueError('closure_width_m must be finite and positive')
    if type(consecutive_samples) is not int or consecutive_samples < 1:
        raise ValueError('consecutive_samples must be a positive integer')
    closed = values <= closure_width_m
    for index in range(len(closed) - consecutive_samples + 1):
        if closed[index:index + consecutive_samples].all():
            return index
    return None


def trajectory_metrics(base_tcp, shifted_tcp, base_actions, shifted_actions,
                       delta, initial_tcp, base_grasp, shifted_grasp,
                       control_hz=10., closure_width_m=.026):
    """Return JSON-safe paired endpoint and telescoping-displacement metrics.

    ``base_tcp`` and ``shifted_tcp`` are ``(T, 3)`` FK command trajectories in
    the same world frame, with common initial TCP and aligned query indices.
    ``delta`` is the actual object translation in that frame. Grasp references
    are object-relative geometric target points, used only for error reporting.

    Fixed query endpoints and the baseline closure query compare equal forecast
    times. Independently selected closure queries can differ and are explicitly
    labeled. Minimum Z is an auxiliary geometric proxy, never a grasp label.
    """
    base_tcp, shifted_tcp = (np.asarray(value, dtype=float)
                             for value in (base_tcp, shifted_tcp))
    base_actions, shifted_actions = (np.asarray(value, dtype=float)
                                     for value in (base_actions, shifted_actions))
    if base_tcp.ndim != 2 or base_tcp.shape[1:] != (3,) or not len(base_tcp):
        raise ValueError('base_tcp must have nonempty shape (T, 3)')
    if shifted_tcp.shape != base_tcp.shape:
        raise ValueError('Paired TCP trajectories must have equal shape (T, 3)')
    count = len(base_tcp)
    if base_actions.shape != (count, 8) or shifted_actions.shape != (count, 8):
        raise ValueError('Actions must have shape (T, 8), aligned with TCP trajectories')
    vectors = [np.asarray(value, dtype=float)
               for value in (delta, initial_tcp, base_grasp, shifted_grasp)]
    if any(value.shape != (3,) for value in vectors):
        raise ValueError('Translation, initial TCP and grasp references must have shape (3,)')
    delta, initial_tcp, base_grasp, shifted_grasp = vectors
    if not all(np.isfinite(value).all() for value in
               (base_tcp, shifted_tcp, base_actions, shifted_actions, *vectors)):
        raise ValueError('All trajectories, actions and reference points must be finite')
    distance = float(np.linalg.norm(delta))
    if distance <= 0:
        raise ValueError('Object translation must be nonzero')
    if not np.isfinite(control_hz) or control_hz <= 0:
        raise ValueError('control_hz must be finite and positive')
    direction = delta / distance
    base_close = first_closure_index(base_actions, closure_width_m)
    shifted_close = first_closure_index(shifted_actions, closure_width_m)

    def endpoint(base_query, shifted_query):
        if base_query is None or shifted_query is None:
            return None
        base_point, shifted_point = base_tcp[base_query], shifted_tcp[shifted_query]
        displacement = shifted_point - base_point
        along = float(displacement @ direction)
        base_error, shifted_error = base_point - base_grasp, shifted_point - shifted_grasp
        return dict(
            base_query=int(base_query), shifted_query=int(shifted_query),
            base_command_time_s=float(base_query / control_hz),
            shifted_command_time_s=float(shifted_query / control_hz),
            same_query=bool(base_query == shifted_query),
            baseline_tcp_m=base_point.tolist(), shifted_tcp_m=shifted_point.tolist(),
            displacement_m=displacement.tolist(), along_m=along, gain=along / distance,
            orthogonal_m=float(np.linalg.norm(displacement - along * direction)),
            residual_m=float(np.linalg.norm(displacement - delta)),
            baseline_grasp_xy_error_m=float(np.linalg.norm(base_error[:2])),
            shifted_grasp_xy_error_m=float(np.linalg.norm(shifted_error[:2])),
            baseline_grasp_z_error_m=float(base_error[2]),
            shifted_grasp_z_error_m=float(shifted_error[2]),
            baseline_grasp_error_m=float(np.linalg.norm(base_error)),
            shifted_grasp_error_m=float(np.linalg.norm(shifted_error)),
        )

    def telescoping(points):
        increments = np.diff(np.vstack((initial_tcp, points)), axis=0)
        summed = increments.sum(axis=0)
        terminal = points[-1] - initial_tcp
        error = float(np.linalg.norm(summed - terminal))
        if not np.allclose(summed, terminal, rtol=1e-10, atol=1e-10):
            raise AssertionError('Successive TCP increments do not telescope to terminal minus initial')
        return dict(sum_successive_tcp_displacements_m=summed.tolist(),
                    terminal_minus_initial_m=terminal.tolist(),
                    identity_error_m=error, verified=True)

    min_z_query = int(np.argmin(base_tcp[:, 2]))
    return dict(
        action_semantics='absolute joint targets; FK is evaluated separately for every query',
        target_semantics='kinematic command targets, not physically executed TCP positions',
        frames=count, control_hz=float(control_hz), translation_m=delta.tolist(),
        translation_distance_m=distance,
        closure=dict(baseline_found=base_close is not None, shifted_found=shifted_close is not None,
                     baseline_query=base_close, shifted_query=shifted_close,
                     threshold_m=float(closure_width_m), consecutive_samples=2,
                     semantics='commanded aperture proxy, not observed physical grasp'),
        endpoints=dict(query31=endpoint(31, 31) if count > 31 else None,
                       last=endpoint(count - 1, count - 1),
                       baseline_first_closure=endpoint(base_close, base_close),
                       independent_first_closures=endpoint(base_close, shifted_close),
                       baseline_minimum_z=endpoint(min_z_query, min_z_query)),
        telescoping=dict(baseline=telescoping(base_tcp), shifted=telescoping(shifted_tcp)),
        limitations=[
            'Aperture threshold crossings and minimum height are proxies, not detected physical grasps.',
            'Early approach targets need not shift by the full object displacement before reaching the object.',
            'Independent closure queries may refer to different prediction times.',
            'Reduced translation gain alone does not isolate an internal spatial-representation cause.',
        ],
    )
