"""Stage-independent contact diagnostics for open-loop action replay.

This observer uses MotionGuard's bodies, adjacency rules and penetration
tolerances. It permits physically valid grasping and either holder's upward
head support throughout replay, without consulting an expert or its stages.
Every violating contact is counted; a bounded sample retains its details.
"""
from __future__ import annotations

from collections import Counter

from ..control.execution_guard import MotionGuard


class ReplayContactMonitor(MotionGuard):
    """Read-only callback suitable for ``Simulation.safety_callback``.

    Calling the monitor never raises for a contact-policy violation and never
    changes the simulation. Counts and peaks cover every call and contact;
    ``violations`` retains the first 1,000 detailed violations to bound memory.
    """

    SOURCE_PHASES = frozenset({'action_replay'})
    TARGET_PHASES = frozenset({'action_replay'})
    GRASP_PHASES = frozenset({'action_replay'})
    VIOLATION_SAMPLE_LIMIT = 1000

    def __init__(self, sim):
        super().__init__(sim)
        self.phase = 'action_replay'
        self.first_violation = None
        self.violations_by_reason = Counter()
        self.violations = []

    def _fail(self, contact, reason):
        self.unsafe_contact_count += 1
        self.violations_by_reason[reason] += 1
        violation = dict(
            phase=self.phase, time_s=float(self.sim.data.time), reason=reason,
            geom1=self.geom_names[int(contact.geom1)],
            geom2=self.geom_names[int(contact.geom2)],
            penetration_m=max(0.0, float(-contact.dist)),
        )
        if self.first_violation is None:
            self.first_violation = violation
        self.last_violation = violation
        if len(self.violations) < self.VIOLATION_SAMPLE_LIMIT:
            self.violations.append(violation)

    def __call__(self):
        # MotionGuard's loop assumes _fail raises. Its robot/unknown-fixture
        # branches would otherwise fall through to rims[None]. Keep its same
        # physical classification here, with explicit per-contact completion.
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
                peg_geom = first if peg_first else second
                other_body = b2 if peg_first else b1
                peg_name = self.geom_names[peg_geom]
                if other_body in self.pad_bodies and 'head' in peg_name:
                    self.peak_pad_penetration_m = max(self.peak_pad_penetration_m, penetration)
                    if penetration > self.pad_tolerance:
                        self._fail(contact, 'excessive grasp pad compression')
                    continue
                if other_body in self.robot_bodies:
                    self._fail(contact, 'peg contacts robot outside grasp pads')
                    continue
                self.peak_fixture_penetration_m = max(self.peak_fixture_penetration_m, penetration)
                name = self.body_names[other_body]
                side = ('source' if name == 'source_socket' else
                        'target' if name == 'target_socket' else None)
                if side is None:
                    if 'shaft' in peg_name:
                        self.peak_shaft_fixture_penetration_m = max(
                            self.peak_shaft_fixture_penetration_m, penetration)
                    self._fail(contact, 'peg contacts a fixture other than source or target holder')
                    continue
                # MuJoCo's contact-frame +X points from geom1 to geom2.
                normal_z = float(contact.frame[2]) * (-1 if peg_first else 1)
                support = ('head' in peg_name and normal_z > 0.9
                           and abs(float(contact.pos[2]) - self.rims[side]) < 0.002)
                if support:
                    self.peak_support_penetration_m = max(self.peak_support_penetration_m, penetration)
                    if penetration > self.support_tolerance:
                        self._fail(contact, 'excessive head penetration into supporting rim')
                else:
                    if 'shaft' in peg_name:
                        self.peak_shaft_fixture_penetration_m = max(
                            self.peak_shaft_fixture_penetration_m, penetration)
                    if penetration > self.fixture_tolerance:
                        self._fail(contact, 'peg strikes holder wall or rim')
                continue
            if robot1 or robot2:
                robot_body, other_geom = (b1, second) if robot1 else (b2, first)
                if robot_body == self.robot_root and self.geom_names[other_geom] in {'table', 'ground'}:
                    continue
                self.peak_robot_fixture_penetration_m = max(
                    self.peak_robot_fixture_penetration_m, penetration)
                self._fail(contact, 'robot or gripper contacts a fixture')

    def summary(self):
        summary = super().summary()
        summary.update(
            first_violation=self.first_violation,
            violations_by_reason=dict(self.violations_by_reason),
            violation_samples_retained=len(self.violations),
            violation_samples_truncated=self.unsafe_contact_count > len(self.violations),
            pad_penetration_tolerance_m=self.pad_tolerance,
            self_penetration_tolerance_m=self.self_tolerance,
        )
        return summary
