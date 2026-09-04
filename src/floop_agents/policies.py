from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from .metrics import oracle_stop_step
from .tasks import Task


@dataclass(frozen=True)
class StopDecision:
    stop: bool
    reason: str = ""


class StoppingPolicy(Protocol):
    name: str

    def decide(
        self,
        *,
        task: Task,
        step: int,
        history: Sequence[float],
        step_cost: float,
    ) -> StopDecision:
        ...


@dataclass
class BoundedPolicy:
    bound: int = 12
    name: str = "bounded"

    def decide(self, *, task: Task, step: int, history: Sequence[float], step_cost: float) -> StopDecision:
        if step >= self.bound:
            return StopDecision(True, f"fixed_bound={self.bound}")
        return StopDecision(False)


@dataclass
class UnboundedPolicy:
    """Never halts endogenously; the benchmark harness supplies a safety cap."""

    name: str = "unbounded"

    def decide(self, *, task: Task, step: int, history: Sequence[float], step_cost: float) -> StopDecision:
        return StopDecision(False)


@dataclass
class MetaStopPolicy:
    """Stop after sustained low marginal value using only observed history."""

    threshold: float = 0.012
    patience: int = 2
    min_steps: int = 3
    name: str = "meta_stop"

    def decide(self, *, task: Task, step: int, history: Sequence[float], step_cost: float) -> StopDecision:
        if step < max(self.min_steps, self.patience):
            return StopDecision(False)
        if len(history) < self.patience + 1:
            return StopDecision(False)
        marginals = [
            history[-j] - history[-j - 1]
            for j in range(1, self.patience + 1)
        ]
        cutoff = max(self.threshold, step_cost)
        if all(delta <= cutoff for delta in marginals):
            return StopDecision(True, f"marginal_value<={cutoff:.4f} for {self.patience} steps")
        return StopDecision(False)


@dataclass
class VerifierStopPolicy:
    """Privileged verifier baseline using the benchmark's oracle stop point."""

    name: str = "verifier_stop"

    def decide(self, *, task: Task, step: int, history: Sequence[float], step_cost: float) -> StopDecision:
        oracle = oracle_stop_step(task, step_cost)
        if step >= oracle:
            return StopDecision(True, f"verifier_oracle={oracle}")
        return StopDecision(False)
