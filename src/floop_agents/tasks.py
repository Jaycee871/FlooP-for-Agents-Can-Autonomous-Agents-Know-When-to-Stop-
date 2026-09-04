from __future__ import annotations

from dataclasses import dataclass
import math
import random


@dataclass(frozen=True)
class Task:
    """A synthetic reasoning task with an observable utility trajectory.

    utility_curve[s] is the best task utility available after s reasoning steps.
    Index 0 is the pre-reasoning baseline.
    """

    task_id: str
    family: str
    utility_curve: tuple[float, ...]
    solved_at: int | None = None

    @property
    def horizon(self) -> int:
        return len(self.utility_curve) - 1

    def utility(self, step: int) -> float:
        step = max(0, min(step, self.horizon))
        return self.utility_curve[step]

    def marginal(self, step: int) -> float:
        if step <= 0:
            return 0.0
        return self.utility(step) - self.utility(step - 1)


def _solvable_task(i: int, rng: random.Random, horizon: int) -> Task:
    solved_at = rng.randint(4, 9)
    curve = [0.0]
    for s in range(1, horizon + 1):
        if s <= solved_at:
            u = min(1.0, 0.14 * s + 0.02 * rng.random())
        else:
            u = max(0.90, 1.0 - 0.006 * (s - solved_at))
        curve.append(round(u, 6))
    curve[solved_at] = 1.0
    return Task(f"solvable-{i:03d}", "solvable", tuple(curve), solved_at=solved_at)


def _unsatisfiable_task(i: int, rng: random.Random, horizon: int) -> Task:
    curve = [0.0]
    for s in range(1, horizon + 1):
        u = 0.20 * (1.0 - math.exp(-0.65 * s))
        u += rng.uniform(-0.003, 0.003)
        curve.append(round(max(0.0, u), 6))
    return Task(f"unsatisfiable-{i:03d}", "unsatisfiable", tuple(curve), solved_at=None)


def _open_ended_task(i: int, rng: random.Random, horizon: int) -> Task:
    curve = [0.0]
    for s in range(1, horizon + 1):
        u = 0.78 * (1.0 - math.exp(-0.28 * s))
        u += 0.012 * math.log1p(s)
        u += rng.uniform(-0.002, 0.002)
        curve.append(round(max(curve[-1], u), 6))
    return Task(f"open-ended-{i:03d}", "open_ended", tuple(curve), solved_at=None)


def make_task_suite(
    n_per_family: int = 10,
    *,
    seed: int = 7,
    horizon: int = 30,
) -> list[Task]:
    """Create deterministic solvable, unsatisfiable, and open-ended tasks."""
    if n_per_family <= 0:
        raise ValueError("n_per_family must be positive")
    if horizon < 10:
        raise ValueError("horizon must be at least 10")

    rng = random.Random(seed)
    tasks: list[Task] = []
    for i in range(n_per_family):
        tasks.append(_solvable_task(i, rng, horizon))
        tasks.append(_unsatisfiable_task(i, rng, horizon))
        tasks.append(_open_ended_task(i, rng, horizon))
    return tasks
