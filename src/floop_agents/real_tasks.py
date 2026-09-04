from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


ScoreFn = Callable[[str], float]


@dataclass(frozen=True)
class RealTask:
    task_id: str
    family: str
    prompt: str
    score_fn: ScoreFn
    success_threshold: float = 1.0

    def score(self, candidate: str) -> float:
        value = float(self.score_fn(candidate.strip()))
        return max(0.0, min(1.0, value))

    def is_success(self, candidate: str) -> bool:
        return self.score(candidate) >= self.success_threshold


def _exact(expected: str) -> ScoreFn:
    expected_norm = expected.strip().upper()

    def score(candidate: str) -> float:
        return 1.0 if candidate.strip().upper() == expected_norm else 0.0

    return score


def _open_ended_score(candidate: str) -> float:
    """Deterministic checklist for the open-ended smoke task.

    This is intentionally simple: it validates trace collection and post-success
    continuation plumbing, not prose quality in the general case.
    """

    text = candidate.lower()
    words = candidate.split()
    checks = [
        "agent" in text,
        "termination" in text or "stop" in text,
        "evidence" in text,
        len(words) <= 25,
        len(words) >= 7,
    ]
    return sum(checks) / len(checks)


def smoke_tasks() -> list[RealTask]:
    return [
        RealTask(
            task_id="solvable_arithmetic",
            family="solvable",
            prompt=(
                "Compute 37 * 24 - 19. Your candidate field must contain only the final integer, "
                "with no explanation."
            ),
            score_fn=_exact("869"),
        ),
        RealTask(
            task_id="unsat_modular",
            family="unsatisfiable",
            prompt=(
                "Find an integer x satisfying both x mod 6 = 1 and x mod 3 = 2. "
                "If you can establish that no such integer exists, put exactly NO_SOLUTION in the candidate field."
            ),
            score_fn=_exact("NO_SOLUTION"),
        ),
        RealTask(
            task_id="open_ended_microabstract",
            family="open_ended",
            prompt=(
                "Write one sentence of at most 25 words describing why autonomous-agent stopping should be evaluated. "
                "The sentence must explicitly mention an agent, termination (or stopping), and evidence."
            ),
            score_fn=_open_ended_score,
        ),
    ]
