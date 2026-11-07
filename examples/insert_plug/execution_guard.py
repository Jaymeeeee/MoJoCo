"""Contact guards evaluated at the physics frequency during task execution.

The guard observes contacts only. It never changes a pose, actuator, collision
mask or constraint. Intended pad compression and upward head support on a rim
are distinct from the shaft striking a wall or the robot striking a fixture.
"""
from __future__ import annotations

import numpy as np


class ExecutionSafetyError(RuntimeError):
    """An actual physics contact violates the task's contact policy."""


class MotionGuard:
    SOURCE_PHASES = frozenset({
        'settle_initial', 'approach', 'descend_to_grasp', 'grasp', 'extract',
    })
    TARGET_PHASES = frozenset({
        'shallow_insert', 'insertion_check', 'release', 'retract', 'settle_final',
    })
    GRASP_PHASES = frozenset({
        'grasp', 'extract', 'transfer', 'align', 'shallow_insert',
        'insertion_check', 'release',
    })

    def __init__(self, sim):
        self.sim = sim
        model = sim.model
        options = sim.config['task'].get('planning', {})
        self.fixture_tolerance = float(options.get('fixture_penetration_tolerance', 5e-5))
        self.support_tolerance = float(options.get('support_penetration_tolerance', 1e-3))
        self.pad_tolerance = float(options.get('pad_penetration_tolerance', 7.5e-4))
        self.self_tolerance = float(options.get('self_penetration_tolerance', 1.5e-4))
        for name, value in (('fixture', self.fixture_tolerance),
                            ('support', self.support_tolerance),
                            ('pad', self.pad_tolerance), ('self', self.self_tolerance)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f'{name} penetration tolerance must be finite and positive')
        self.phase = 'settle_initial'
        self.checks = 0
        self.unsafe_contact_count = 0
        self.peak_fixture_penetration_m = 0.0
        self.peak_shaft_fixture_penetration_m = 0.0
        self.peak_support_penetration_m = 0.0
        self.peak_pad_penetration_m = 0.0
        self.peak_robot_fixture_penetration_m = 0.0
        self.peak_self_penetration_m = 0.0
        self.last_violation = None
        self.body_names = [model.body(i).name or f'body_{i}' for i in range(model.nbody)]
        self.geom_names = [model.geom(i).name or f'geom_{i}' for i in range(model.ngeom)]
        self.robot_root = model.body('fr3_link0').id
        self.robot_bodies = set()
        for body in range(model.nbody):
            ancestor = body
            while ancestor:
                if ancestor == self.robot_root:
                    self.robot_bodies.add(body)
                    break
                ancestor = int(model.body_parentid[ancestor])
        self.pad_bodies = {
            body for body in self.robot_bodies
            if self.body_names[body] in {
                'robotiq_left_pad', 'robotiq_right_pad',
                'robotiq_left_silicone_pad', 'robotiq_right_silicone_pad',
                'left_finger', 'right_finger',
            }
        }
        self.rims = {side: float(sim.site(f'{side}_hole')[2]) for side in ('source', 'target')}

    def _fail(self, contact, reason):
        self.unsafe_contact_count += 1
        first, second = int(contact.geom1), int(contact.geom2)
        self.last_violation = dict(
            phase=self.phase, time_s=float(self.sim.data.time), reason=reason,
            geom1=self.geom_names[first], geom2=self.geom_names[second],
            penetration_m=max(0.0, float(-contact.dist)),
        )
        raise ExecutionSafetyError(
            f'{self.phase}: {reason}: {self.geom_names[first]} / '
            f'{self.geom_names[second]} at t={self.sim.data.time:.3f}s '
            f'(penetration {max(0.0, -contact.dist):.6f} m)'
        )

    def _adjacent(self, first, second):
        model = self.sim.model
        # Fixed pieces share their welded body's collision relation. Exempt
        # only immediate mechanical neighbours, never all robot ancestors or
        # pairs that happened to overlap in an initial configuration.
        first, second = int(model.body_weldid[first]), int(model.body_weldid[second])
        if first == second:
            return True
        parents = model.body_parentid
        return (int(model.body_weldid[parents[first]]) == second
                or int(model.body_weldid[parents[second]]) == first)

    def __call__(self):
        self.checks += 1
        model, data = self.sim.model, self.sim.data
        for contact in data.contact:
            penetration = max(0.0, float(-contact.dist))
            if penetration <= 1e-7:
                continue
            first, second = int(contact.geom1), int(contact.geom2)
            b1, b2 = map(int, model.geom_bodyid[[first, second]])
            robot1, robot2 = b1 in self.robot_bodies, b2 in self.robot_bodies
            if robot1 and robot2:
                if not self._adjacent(b1, b2):
                    self.peak_self_penetration_m = max(self.peak_self_penetration_m, penetration)
                    if penetration > self.self_tolerance:
                        self._fail(contact, 'nonadjacent robot self collision')
                continue
            if self.sim.peg_id in (b1, b2):
                peg_first = b1 == self.sim.peg_id
                peg_geom, other_geom = (first, second) if peg_first else (second, first)
                other_body = b2 if peg_first else b1
                if other_body in self.pad_bodies and 'head' in self.geom_names[peg_geom]:
                    self.peak_pad_penetration_m = max(self.peak_pad_penetration_m, penetration)
                    if self.phase not in self.GRASP_PHASES:
                        self._fail(contact, 'grasp pads touch the peg outside a grasp stage')
                    if penetration > self.pad_tolerance:
                        self._fail(contact, 'excessive grasp pad compression')
                    continue
                if other_body in self.robot_bodies:
                    self._fail(contact, 'peg contacts robot outside grasp pads')
                self.peak_fixture_penetration_m = max(self.peak_fixture_penetration_m, penetration)
                name = self.body_names[other_body]
                side = 'source' if name == 'source_socket' else 'target' if name == 'target_socket' else None
                allowed = ((side == 'source' and self.phase in self.SOURCE_PHASES)
                           or (side == 'target' and self.phase in self.TARGET_PHASES))
                if not allowed:
                    self._fail(contact, 'peg contacts a fixture outside its allowed stage')
                # Contact-frame +X points from geom1 to geom2. The normal
                # force on the peg must point up for a legitimate rim support.
                normal_z = float(contact.frame[2]) * (-1 if peg_first else 1)
                support = ('head' in self.geom_names[peg_geom] and normal_z > 0.9
                           and abs(float(contact.pos[2]) - self.rims[side]) < 0.002)
                if support:
                    self.peak_support_penetration_m = max(self.peak_support_penetration_m, penetration)
                    if penetration > self.support_tolerance:
                        self._fail(contact, 'excessive head penetration into supporting rim')
                else:
                    if 'shaft' in self.geom_names[peg_geom]:
                        self.peak_shaft_fixture_penetration_m = max(
                            self.peak_shaft_fixture_penetration_m, penetration)
                    if penetration > self.fixture_tolerance:
                        self._fail(contact, 'peg strikes holder wall or rim')
                continue
            if robot1 or robot2:
                robot_body, other_geom = (b1, second) if robot1 else (b2, first)
                # The fixed base is intentionally mounted on the support.
                if robot_body == self.robot_root and self.geom_names[other_geom] in {'table', 'ground'}:
                    continue
                self.peak_robot_fixture_penetration_m = max(
                    self.peak_robot_fixture_penetration_m, penetration)
                self._fail(contact, 'robot or gripper contacts a fixture')

    def summary(self):
        return dict(
            checks=self.checks, unsafe_contact_count=self.unsafe_contact_count,
            peak_fixture_penetration_m=self.peak_fixture_penetration_m,
            peak_shaft_fixture_penetration_m=self.peak_shaft_fixture_penetration_m,
            peak_support_penetration_m=self.peak_support_penetration_m,
            peak_pad_penetration_m=self.peak_pad_penetration_m,
            peak_robot_fixture_penetration_m=self.peak_robot_fixture_penetration_m,
            peak_self_penetration_m=self.peak_self_penetration_m,
            fixture_penetration_tolerance_m=self.fixture_tolerance,
            support_penetration_tolerance_m=self.support_tolerance,
            last_violation=self.last_violation,
        )
