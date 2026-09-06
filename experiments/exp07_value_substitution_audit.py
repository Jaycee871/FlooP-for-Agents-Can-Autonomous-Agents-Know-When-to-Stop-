from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

from floop_agents.providers import NvidiaClient
from floop_agents.search_arena import oracle_state, query_action_value
from floop_agents.stopping_metrics import local_stopping_metrics

from exp04_calibrated_state_ablation import highest_mass_remaining_candidate
from exp05_bellman_decomposition import WEDGE_CONDITIONS, make_wedge_spec, state_diagnostics
from exp05b_transition_audit import parse_transition_response
from exp06_repeated_transition_audit import REPLICATION_CONDITION_IDS


EXP07_ARMS = {"oracle_successor_state", "oracle_successor_value"}
EXP07_CONDITION_IDS = set(REPLICATION_CONDITION_IDS)


EXP07_SYSTEM_PROMPT = """You are an autonomous search agent in a stopping experiment.
Your goal is to maximize expected NET utility.

You will receive exact statistics for the CURRENT Bayesian state and, depending on
the experimental arm, exact information about the state that follows a fixed query
if that query MISSES. Treat every quantity explicitly labelled EXACT as authoritative.
Do not replace an exact supplied value with your own alternative estimate.

Return ONLY one JSON object with exactly these keys:
{
  "decision": "STOP or QUERY",
  "query": 1,
  "evaluated_query": 1,
  "current_hit_probability": 0.0,
  "miss_probability": 0.0,
  "predicted_target_exists_after_miss": 0.0,
  "predicted_best_hit_probability_after_miss": 0.0,
  "predicted_best_candidate_after_miss": 1,
  "estimated_value_after_miss": 0.0,
  "full_expected_net_gain": 0.0,
  "reason": "brief rationale"
}

If decision is QUERY, query must equal evaluated_query. If decision is STOP, set
query to null but still fill all counterfactual fields. Probabilities must be in
[0,1]. estimated_value_after_miss must be nonnegative because stopping is available
in the successor state. full_expected_net_gain may be negative.

Use the Bellman composition
full expected net gain = - query cost
                         + current hit probability * reward
                         + miss probability * value after miss.
Choose QUERY iff this value is positive; otherwise choose STOP.
Do not wrap the JSON in markdown.
"""


class Exp07Client(NvidiaClient):
    """NVIDIA client with the same bounded reasoning controls used in Exp06."""

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        reasoning_effort: str | None = None,
    ):
        configured_tokens = int(os.getenv("EXP07_MAX_TOKENS", str(max_tokens)))
        mode = os.getenv("EXP07_REASONING_MODE", "preserve").strip().lower()
        if mode not in {"none", "preserve"}:
            raise ValueError("EXP07_REASONING_MODE must be 'none' or 'preserve'")
        effective_reasoning = reasoning_effort if mode == "preserve" else None
        return super().chat(
            messages,
            temperature=temperature,
            max_tokens=max(max_tokens, configured_tokens),
            reasoning_effort=effective_reasoning,
        )


def condition_by_id(condition_id: str):
    if condition_id not in EXP07_CONDITION_IDS:
        raise ValueError(f"Unsupported Exp07 condition: {condition_id}")
    for condition in WEDGE_CONDITIONS:
        if condition.condition_id == condition_id:
            return condition
    raise ValueError(f"Condition not found in WEDGE_CONDITIONS: {condition_id}")


def _world_header(spec, evaluated_query: int) -> str:
    current = oracle_state(spec, [])
    target_masses = ", ".join(
        f"{candidate}:{mass:.9f}" for candidate, mass in sorted(spec.target_masses.items())
    )
    return f"""SEARCH WORLD
There are {spec.n_candidates} candidate locations. At most one target exists.
Prior probability that a target exists somewhere: {spec.existence_prob:.9f}
Probability no target exists: {spec.no_target_mass:.9f}
Target mass by candidate: {target_masses}
Reward for finding the target: {spec.reward:.9f}
Cost per query: {spec.query_cost:.9f}
No candidates have yet been queried.

EXACT CURRENT-STATE STATISTICS
Posterior probability that a target currently exists: {current.posterior_target_exists:.9f}
Highest current one-query hit probability: {current.next_hit_probability:.9f}
Candidate to evaluate: {evaluated_query}
Exact miss probability for that candidate: {1.0 - current.next_hit_probability:.9f}
"""


def prompt_for_value_substitution(spec, evaluated_query: int, arm: str) -> str:
    if arm not in EXP07_ARMS:
        raise ValueError(f"Unsupported Exp07 arm: {arm}")

    successor = oracle_state(spec, [evaluated_query])
    successor_candidate = highest_mass_remaining_candidate(spec, [evaluated_query])
    exact_value_after_miss = max(0.0, successor.continuation_value)

    prompt = _world_header(spec, evaluated_query)
    prompt += f"""
EXACT POST-MISS SUCCESSOR STATE
Assume candidate {evaluated_query} is queried and the result is a MISS.
Exact posterior probability that a target still exists: {successor.posterior_target_exists:.9f}
Exact best remaining one-query hit probability: {successor.next_hit_probability:.9f}
Exact best remaining candidate: {successor_candidate}

Copy these exact successor-state quantities into the corresponding JSON fields.
"""

    if arm == "oracle_successor_state":
        prompt += """
EXPERIMENTAL ARM: ORACLE SUCCESSOR STATE
The successor state is supplied exactly, but the value of the best future policy is
NOT supplied. Estimate the optimal future policy value from this exact post-miss
state. Then use that estimate in the Bellman composition, report the resulting full
expected net gain, and choose STOP or QUERY from its sign.
"""
    else:
        prompt += f"""
EXPERIMENTAL ARM: ORACLE SUCCESSOR VALUE
The successor state and its optimal continuation value are both supplied exactly.
Exact optimal value after the miss: {exact_value_after_miss:.9f}

Copy this exact value into estimated_value_after_miss. Do not re-estimate it. Compose
the full expected net gain from the supplied current hit probability, miss
probability, reward, cost, and exact value after miss. Then choose STOP or QUERY
from the sign of that composed value.
"""

    return prompt


def _safe_number(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(result) or math.isinf(result):
        return 0.0
    return result


def probe_value_substitution(client: Exp07Client, condition, arm: str) -> dict[str, Any]:
    spec = make_wedge_spec(condition)
    queried: list[int] = []
    evaluated_query = highest_mass_remaining_candidate(spec, queried)
    if evaluated_query is None:
        raise RuntimeError("No candidate available for Exp07 audit")

    diagnostics = state_diagnostics(spec, queried)
    current = oracle_state(spec, queried)
    exact_current_hit = current.next_hit_probability
    exact_miss_probability = 1.0 - exact_current_hit

    successor_queried = [evaluated_query]
    successor = oracle_state(spec, successor_queried)
    exact_successor_candidate = highest_mass_remaining_candidate(spec, successor_queried)
    exact_successor_exists = successor.posterior_target_exists
    exact_successor_hit = successor.next_hit_probability
    exact_successor_value = max(0.0, successor.continuation_value)
    exact_q_value = query_action_value(spec, queried, evaluated_query)

    response = client.chat(
        [
            {"role": "system", "content": EXP07_SYSTEM_PROMPT},
            {"role": "user", "content": prompt_for_value_substitution(spec, evaluated_query, arm)},
        ],
        temperature=0.1,
        max_tokens=3072,
        reasoning_effort="low",
    )
    action = parse_transition_response(response.content)

    evaluated_query_matches = action["evaluated_query"] == evaluated_query
    query_valid = action["decision"] != "QUERY" or action["query"] == evaluated_query
    action_valid = evaluated_query_matches and query_valid

    recomposed = (
        -spec.query_cost
        + float(action["current_hit_probability"]) * spec.reward
        + float(action["miss_probability"]) * float(action["estimated_value_after_miss"])
    )

    stopping = local_stopping_metrics(
        agent_decision=action["decision"],
        oracle_continuation_value=float(diagnostics["oracle_advantage"]),
        agent_reported_continuation_value=float(action["full_expected_net_gain"]),
        chosen_query_value=(
            exact_q_value if action["decision"] == "QUERY" and action_valid else None
        ),
    )

    return {
        "condition_id": condition.condition_id,
        "query_cost": condition.query_cost,
        "arm": arm,
        "model": response.model,
        **action,
        **diagnostics,
        "fixed_evaluated_query": evaluated_query,
        "action_valid": action_valid,
        "exact_current_hit_probability": exact_current_hit,
        "exact_miss_probability": exact_miss_probability,
        "exact_target_exists_after_miss": exact_successor_exists,
        "exact_best_hit_probability_after_miss": exact_successor_hit,
        "exact_best_candidate_after_miss": exact_successor_candidate,
        "exact_value_after_miss": exact_successor_value,
        "exact_evaluated_query_value": exact_q_value,
        "current_hit_probability_error": (
            float(action["current_hit_probability"]) - exact_current_hit
        ),
        "miss_probability_error": (
            float(action["miss_probability"]) - exact_miss_probability
        ),
        "target_exists_after_miss_error": (
            float(action["predicted_target_exists_after_miss"]) - exact_successor_exists
        ),
        "best_hit_after_miss_error": (
            float(action["predicted_best_hit_probability_after_miss"]) - exact_successor_hit
        ),
        "successor_best_candidate_correct": (
            action["predicted_best_candidate_after_miss"] == exact_successor_candidate
        ),
        "future_value_after_miss_error": (
            float(action["estimated_value_after_miss"]) - exact_successor_value
        ),
        "full_evaluated_query_value_error": (
            float(action["full_expected_net_gain"]) - exact_q_value
        ),
        "reported_decomposition_closure_error": (
            float(action["full_expected_net_gain"]) - recomposed
        ),
        "decision_agreement": action["decision"] == diagnostics["oracle_decision"],
        "decision_matches_reported_value_sign": (
            (action["decision"] == "QUERY" and float(action["full_expected_net_gain"]) > 0)
            or (action["decision"] == "STOP" and float(action["full_expected_net_gain"]) <= 0)
        ),
        "oracle_successor_state_supplied": True,
        "oracle_successor_value_supplied": arm == "oracle_successor_value",
        "premature_stop_regret": stopping.premature_stop_regret,
        "excess_continuation_regret": stopping.excess_continuation_regret,
        "query_selection_regret": stopping.query_selection_regret,
        "local_decision_regret": stopping.local_decision_regret,
        "reasoning_chars": len(response.reasoning) if response.reasoning else 0,
        "prompt_tokens": response.prompt_tokens,
        "completion_tokens": response.completion_tokens,
        "total_tokens": response.total_tokens,
    }


def run_single_probe(
    *,
    arm: str,
    condition_id: str,
    repeat_index: int,
    out_path: Path,
) -> dict[str, Any]:
    if arm not in EXP07_ARMS:
        raise ValueError(f"Unsupported Exp07 arm: {arm}")
    if repeat_index < 0:
        raise ValueError("repeat_index must be non-negative")

    client = Exp07Client.from_env()
    model = client.resolve_model()
    condition = condition_by_id(condition_id)
    reasoning_mode = os.getenv("EXP07_REASONING_MODE", "preserve").strip().lower()

    try:
        record = probe_value_substitution(client, condition, arm)
        record["repeat"] = repeat_index
        record["replication_model"] = model
        record["reasoning_mode"] = reasoning_mode
    except Exception as exc:
        record = {
            "condition_id": condition.condition_id,
            "query_cost": condition.query_cost,
            "arm": arm,
            "repeat": repeat_index,
            "replication_model": model,
            "reasoning_mode": reasoning_mode,
            "error": f"{type(exc).__name__}: {exc}",
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))
    print(
        "PROBE_VALIDITY: "
        + ("PASS" if "error" not in record else "FAIL")
        + f" model={model} arm={arm} condition={condition_id} repeat={repeat_index}"
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=sorted(EXP07_ARMS))
    parser.add_argument("--condition-id", required=True)
    parser.add_argument("--repeat-index", required=True, type=int)
    parser.add_argument("--out", type=Path, default=Path("results/exp07_matrix/probe.json"))
    parser.add_argument("--require-valid", action="store_true")
    args = parser.parse_args()

    record = run_single_probe(
        arm=args.arm,
        condition_id=args.condition_id,
        repeat_index=args.repeat_index,
        out_path=args.out,
    )
    if args.require_valid and "error" in record:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
