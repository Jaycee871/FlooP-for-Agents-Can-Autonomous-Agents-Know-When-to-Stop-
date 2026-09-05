from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from floop_agents.providers import NvidiaClient
from floop_agents.search_arena import oracle_state, query_action_value
from floop_agents.stopping_metrics import local_stopping_metrics

from exp04_calibrated_state_ablation import highest_mass_remaining_candidate
from exp05_bellman_decomposition import (
    WEDGE_CONDITIONS,
    make_wedge_spec,
    state_diagnostics,
)


AUDIT_CONDITION_IDS = {
    "cost_0090_wedge",
    "cost_0100_wedge",
    "cost_0120_wedge",
    "cost_0125_wedge",
    "cost_0130_dynamic_stop",
}


TRANSITION_SYSTEM_PROMPT = """You are an autonomous search agent in a stopping experiment.
Your goal is to maximize expected NET utility.

You will be given exact statistics for the CURRENT Bayesian state and one fixed
candidate whose continuation consequences must be evaluated. Imagine that this
candidate is queried and the query MISSES. Predict the Bayesian information state
that would follow that miss, then estimate the value of the best future policy.

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
query to null but still fill every counterfactual successor-state field.
Probabilities must be between 0 and 1. estimated_value_after_miss must be
nonnegative because stopping remains available in the successor state.
full_expected_net_gain may be negative. Do not wrap the JSON in markdown.
"""


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(result) or math.isinf(result):
        return default
    return result


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_transition_response(text: str) -> dict[str, Any]:
    raw = text.strip()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"Model did not return JSON: {raw[:400]}")
        obj = json.loads(raw[start : end + 1])

    decision = str(obj.get("decision", "STOP")).strip().upper()
    if decision not in {"STOP", "QUERY"}:
        decision = "STOP"

    query = _int_or_none(obj.get("query")) if decision == "QUERY" else None
    evaluated_query = _int_or_none(obj.get("evaluated_query"))

    return {
        "decision": decision,
        "query": query,
        "evaluated_query": evaluated_query,
        "current_hit_probability": max(0.0, min(1.0, _number(obj.get("current_hit_probability")))),
        "miss_probability": max(0.0, min(1.0, _number(obj.get("miss_probability")))),
        "predicted_target_exists_after_miss": max(
            0.0, min(1.0, _number(obj.get("predicted_target_exists_after_miss")))
        ),
        "predicted_best_hit_probability_after_miss": max(
            0.0, min(1.0, _number(obj.get("predicted_best_hit_probability_after_miss")))
        ),
        "predicted_best_candidate_after_miss": _int_or_none(
            obj.get("predicted_best_candidate_after_miss")
        ),
        "estimated_value_after_miss": max(
            0.0, min(1.0, _number(obj.get("estimated_value_after_miss")))
        ),
        "full_expected_net_gain": max(
            -1.0, min(1.0, _number(obj.get("full_expected_net_gain")))
        ),
        "reason": str(obj.get("reason", "")).strip()[:600],
    }


def prompt_for_transition_audit(spec, evaluated_query: int) -> str:
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

COUNTERFACTUAL AUDIT
Imagine candidate {evaluated_query} is queried next and the result is a MISS.
Predict the posterior state after that hypothetical miss. In particular, report:
1. probability a target still exists somewhere after the miss;
2. best remaining one-query hit probability after the miss;
3. candidate attaining that best remaining hit probability;
4. value of the optimal future policy from that miss-updated state.

Then combine the current query and hypothetical successor value to estimate the full
expected net gain of initiating the query now, and choose STOP or QUERY.
Do not assume the correct STOP/QUERY action is encoded in any supplied statistic.
"""


def probe(client: NvidiaClient, condition) -> dict[str, Any]:
    spec = make_wedge_spec(condition)
    queried: list[int] = []
    evaluated_query = highest_mass_remaining_candidate(spec, queried)
    if evaluated_query is None:
        raise RuntimeError("No candidate available for transition audit")

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
            {"role": "system", "content": TRANSITION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt_for_transition_audit(spec, evaluated_query)},
        ],
        temperature=0.1,
        max_tokens=3072,
        reasoning_effort="low",
    )
    action = parse_transition_response(response.content)

    evaluated_query_matches = action["evaluated_query"] == evaluated_query
    query_valid = (
        action["decision"] != "QUERY"
        or action["query"] == evaluated_query
    )
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
        "arm": "transition_audit",
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
        "premature_stop_regret": stopping.premature_stop_regret,
        "excess_continuation_regret": stopping.excess_continuation_regret,
        "query_selection_regret": stopping.query_selection_regret,
        "local_decision_regret": stopping.local_decision_regret,
        "reasoning_chars": len(response.reasoning) if response.reasoning else 0,
        "prompt_tokens": response.prompt_tokens,
        "completion_tokens": response.completion_tokens,
        "total_tokens": response.total_tokens,
    }


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in records if "error" not in r]
    return {
        "n": len(valid),
        "n_errors": len(records) - len(valid),
        "decision_agreement_rate": (
            sum(bool(r["decision_agreement"]) for r in valid) / len(valid) if valid else None
        ),
        "decision_value_self_consistency": (
            sum(bool(r["decision_matches_reported_value_sign"]) for r in valid) / len(valid)
            if valid else None
        ),
        "mean_abs_current_hit_error": _mean([
            abs(float(r["current_hit_probability_error"])) for r in valid
        ]),
        "mean_abs_successor_existence_error": _mean([
            abs(float(r["target_exists_after_miss_error"])) for r in valid
        ]),
        "mean_abs_successor_hit_error": _mean([
            abs(float(r["best_hit_after_miss_error"])) for r in valid
        ]),
        "successor_candidate_accuracy": (
            sum(bool(r["successor_best_candidate_correct"]) for r in valid) / len(valid)
            if valid else None
        ),
        "mean_abs_future_value_error": _mean([
            abs(float(r["future_value_after_miss_error"])) for r in valid
        ]),
        "mean_abs_full_q_error": _mean([
            abs(float(r["full_evaluated_query_value_error"])) for r in valid
        ]),
        "mean_abs_closure_error": _mean([
            abs(float(r["reported_decomposition_closure_error"])) for r in valid
        ]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/exp05b_transition_audit"),
    )
    args = parser.parse_args()

    client = NvidiaClient.from_env()
    print(f"Selected NVIDIA model: {client.resolve_model()}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    probes_path = args.out_dir / "probes.jsonl"
    records: list[dict[str, Any]] = []

    conditions = [c for c in WEDGE_CONDITIONS if c.condition_id in AUDIT_CONDITION_IDS]
    with probes_path.open("w", encoding="utf-8", buffering=1) as handle:
        for condition in conditions:
            try:
                record = probe(client, condition)
            except Exception as exc:
                record = {
                    "condition_id": condition.condition_id,
                    "query_cost": condition.query_cost,
                    "arm": "transition_audit",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                f"{condition.condition_id} oracle={record.get('oracle_decision')} "
                f"agent={record.get('decision')} "
                f"successor_exist_error={record.get('target_exists_after_miss_error')} "
                f"future_value_error={record.get('future_value_after_miss_error')} "
                f"error={record.get('error')}"
            )

    summary = summarize(records)
    summary["model"] = next((r.get("model") for r in records if r.get("model")), client.model)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
