from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from floop_agents.providers import NvidiaClient
from floop_agents.search_arena import SearchSpec, oracle_state


SYSTEM_PROMPT = """You are an autonomous search agent in a stopping experiment.
Your goal is to maximize expected NET utility, not to search until certainty.
At each state you may either STOP or QUERY exactly one unqueried candidate.
Return ONLY one JSON object with exactly these keys:
{
  "decision": "STOP or QUERY",
  "query": 1,
  "confidence": 0.0,
  "expected_net_gain": 0.0,
  "reason": "brief decision rationale"
}
If decision is STOP, set query to null. confidence must be in [0,1].
expected_net_gain should be your estimate, in utility units, of the net value of continuing from the current state; it may be negative.
Do not wrap the JSON in markdown.
"""


@dataclass(frozen=True)
class Condition:
    condition_id: str
    query_cost: float


CONDITIONS = (
    Condition("cost_003", 0.03),
    Condition("cost_005", 0.05),
    Condition("cost_008", 0.08),
)


def make_spec(condition: Condition) -> SearchSpec:
    return SearchSpec(
        n_candidates=50,
        existence_prob=0.8,
        alpha=1.5,
        query_cost=condition.query_cost,
        reward=1.0,
    )


def _number(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(x) or math.isinf(x):
        return default
    return x


def parse_action(text: str) -> dict[str, Any]:
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

    query_raw = obj.get("query")
    query: int | None = None
    if decision == "QUERY":
        try:
            query = int(query_raw)
        except (TypeError, ValueError):
            query = None

    return {
        "decision": decision,
        "query": query,
        "confidence": max(0.0, min(1.0, _number(obj.get("confidence"), 0.0))),
        "expected_net_gain": max(-1.0, min(1.0, _number(obj.get("expected_net_gain"), 0.0))),
        "reason": str(obj.get("reason", "")).strip()[:500],
    }


def oracle_boundary(spec: SearchSpec) -> tuple[int, list[int]]:
    """Return first failed-query count where the oracle chooses STOP.

    The queried path follows the oracle's own best-query ordering. The returned
    path contains the queries that must have failed to reach the boundary state.
    """

    queried: list[int] = []
    while len(queried) <= spec.n_candidates:
        state = oracle_state(spec, queried)
        if state.decision == "STOP" or state.best_query is None:
            return len(queried), queried
        queried.append(int(state.best_query))
    raise RuntimeError("Oracle failed to produce a stopping boundary")


def prompt_for_state(spec: SearchSpec, queried: list[int]) -> str:
    history = [{"query": q, "outcome": "MISS"} for q in queried]
    return f"""SEARCH WORLD
Candidates are the integers 1 through {spec.n_candidates}.
Before searching, probability that NO target exists = {1.0 - spec.existence_prob:.6f}.
Probability that a target exists = {spec.existence_prob:.6f}.
Conditional on existence, target i has probability proportional to 1 / i^{spec.alpha}.
There is at most one target.
Each QUERY costs {spec.query_cost:.6f} utility.
Finding the target pays {spec.reward:.6f} utility and ends the episode.
STOP pays no additional reward and incurs no additional cost.

The hidden world is fixed before you act. You do NOT know whether a target exists or where it is.
Every query in HISTORY has already failed. Treat this as the current information state even if you personally would have stopped earlier.
A failed query is evidence and should update your stopping judgment.
Optimize expected net utility. Do not confuse 'not found yet' with 'no target exists'.

FAILED QUERIES SO FAR: {len(queried)}
HISTORY: {json.dumps(history, ensure_ascii=False)}

Choose STOP or QUERY one integer that has not already been queried.
"""


def sign_decision(value: float, tol: float = 1e-12) -> str:
    return "QUERY" if value > tol else "STOP"


def run_condition(client: NvidiaClient, condition: Condition) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    spec = make_spec(condition)
    boundary, oracle_path = oracle_boundary(spec)
    max_k = min(spec.n_candidates, boundary + 2)
    records: list[dict[str, Any]] = []

    for k in range(max_k + 1):
        queried = oracle_path[:k]
        if k > len(oracle_path):
            # Continue the same descending-prior path beyond the oracle boundary
            # so counterfactual post-boundary states remain well defined.
            queried = list(range(1, k + 1))

        oracle = oracle_state(spec, queried)
        response = client.chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_for_state(spec, queried)},
            ],
            temperature=0.1,
            max_tokens=4096,
            reasoning_effort="low",
        )
        action = parse_action(response.content)

        valid_query = (
            action["decision"] == "QUERY"
            and action["query"] is not None
            and 1 <= action["query"] <= spec.n_candidates
            and action["query"] not in queried
        )
        query_matches_oracle: bool | None = None
        if action["decision"] == "QUERY":
            query_matches_oracle = valid_query and action["query"] == oracle.best_query

        myopic_gain = -spec.query_cost + oracle.next_hit_probability * spec.reward
        signed_gain_error = float(action["expected_net_gain"]) - oracle.advantage
        expected_sign_decision = sign_decision(float(action["expected_net_gain"]))

        records.append(
            {
                "condition_id": condition.condition_id,
                "query_cost": spec.query_cost,
                "failed_queries": k,
                "queried": queried,
                "model": response.model,
                **action,
                "valid_query": valid_query if action["decision"] == "QUERY" else None,
                "oracle_decision": oracle.decision,
                "oracle_best_query": oracle.best_query,
                "oracle_advantage": oracle.advantage,
                "oracle_posterior_target_exists": oracle.posterior_target_exists,
                "oracle_next_hit_probability": oracle.next_hit_probability,
                "myopic_one_step_gain": myopic_gain,
                "decision_agreement": action["decision"] == oracle.decision,
                "query_matches_oracle": query_matches_oracle,
                "signed_gain_error": signed_gain_error,
                "expected_gain_abs_error": abs(signed_gain_error),
                "myopic_gain_abs_error": abs(float(action["expected_net_gain"]) - myopic_gain),
                "reported_value_sign_matches_oracle": expected_sign_decision == oracle.decision,
                "decision_matches_reported_value_sign": action["decision"] == expected_sign_decision,
                "reasoning_chars": len(response.reasoning) if response.reasoning else 0,
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "total_tokens": response.total_tokens,
            }
        )

    first_stop = next((int(r["failed_queries"]) for r in records if r["decision"] == "STOP"), None)
    later_queries_after_stop = 0
    if first_stop is not None:
        later_queries_after_stop = sum(
            r["decision"] == "QUERY" and int(r["failed_queries"]) > first_stop for r in records
        )

    query_records = [r for r in records if r["decision"] == "QUERY"]
    summary = {
        "condition_id": condition.condition_id,
        "query_cost": spec.query_cost,
        "oracle_stop_boundary": boundary,
        "agent_first_stop_boundary": first_stop,
        "boundary_displacement": (first_stop - boundary) if first_stop is not None else None,
        "boundary_censored": first_stop is None,
        "n_probes": len(records),
        "decision_agreement_rate": sum(bool(r["decision_agreement"]) for r in records) / len(records),
        "query_choice_agreement_rate": (
            sum(bool(r["query_matches_oracle"]) for r in query_records) / len(query_records)
            if query_records
            else None
        ),
        "reported_value_sign_accuracy": (
            sum(bool(r["reported_value_sign_matches_oracle"]) for r in records) / len(records)
        ),
        "decision_value_self_consistency": (
            sum(bool(r["decision_matches_reported_value_sign"]) for r in records) / len(records)
        ),
        "mean_signed_gain_error": sum(float(r["signed_gain_error"]) for r in records) / len(records),
        "mean_expected_gain_abs_error": sum(float(r["expected_gain_abs_error"]) for r in records) / len(records),
        "mean_myopic_gain_abs_error": sum(float(r["myopic_gain_abs_error"]) for r in records) / len(records),
        "fraction_reported_gain_closer_to_myopic_than_oracle": (
            sum(float(r["myopic_gain_abs_error"]) < float(r["expected_gain_abs_error"]) for r in records)
            / len(records)
        ),
        "post_stop_query_count": later_queries_after_stop,
        "nonmonotone_stop_curve": later_queries_after_stop > 0,
    }
    return records, summary


def aggregate(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    def mean(key: str) -> float | None:
        vals = [float(s[key]) for s in summaries if s.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    return {
        "n_conditions": len(summaries),
        "mean_decision_agreement_rate": mean("decision_agreement_rate"),
        "mean_reported_value_sign_accuracy": mean("reported_value_sign_accuracy"),
        "mean_decision_value_self_consistency": mean("decision_value_self_consistency"),
        "mean_signed_gain_error": mean("mean_signed_gain_error"),
        "mean_expected_gain_abs_error": mean("mean_expected_gain_abs_error"),
        "mean_boundary_displacement": mean("boundary_displacement"),
        "n_nonmonotone_stop_curves": sum(bool(s["nonmonotone_stop_curve"]) for s in summaries),
        "conditions": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("results/exp03_boundary_sweep"))
    args = parser.parse_args()

    client = NvidiaClient.from_env()
    print(f"Selected NVIDIA model: {client.resolve_model()}")

    all_records: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        records, summary = run_condition(client, condition)
        all_records.extend(records)
        summaries.append(summary)
        print(
            f"{condition.condition_id}: oracle_boundary={summary['oracle_stop_boundary']} "
            f"agent_boundary={summary['agent_first_stop_boundary']} "
            f"displacement={summary['boundary_displacement']} "
            f"agreement={summary['decision_agreement_rate']:.3f} "
            f"nonmonotone={summary['nonmonotone_stop_curve']}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "probes.jsonl").open("w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    result = aggregate(summaries)
    result["model"] = all_records[0]["model"] if all_records else client.model
    (args.out_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in result.items() if k != "conditions"}, indent=2))


if __name__ == "__main__":
    main()
