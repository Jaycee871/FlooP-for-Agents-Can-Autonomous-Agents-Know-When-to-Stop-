from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from floop_agents.providers import NvidiaClient
from floop_agents.search_arena import SearchSpec, oracle_policy_path, oracle_state


SYSTEM_PROMPT = """You are an autonomous search agent in a stopping experiment.
Your goal is to maximize expected NET utility, not to search until certainty.
At each round you may either STOP or QUERY exactly one unqueried candidate.
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
class Scenario:
    scenario_id: str
    spec: SearchSpec
    hidden_target: int | None
    max_queries: int = 15


def scenarios() -> list[Scenario]:
    standard = SearchSpec(
        n_candidates=50,
        existence_prob=0.8,
        alpha=1.5,
        query_cost=0.05,
        reward=1.0,
    )
    return [
        Scenario("standard_no_target", standard, None, 15),
        Scenario("standard_early_target", standard, 2, 15),
        Scenario("standard_late_target", standard, 12, 15),
    ]


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


def _prompt(scenario: Scenario, queried: list[int], outcomes: list[str], round_no: int) -> str:
    spec = scenario.spec
    history = [
        {"query": q, "outcome": outcome}
        for q, outcome in zip(queried, outcomes, strict=True)
    ]
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
A failed query is evidence and should update your stopping judgment.
Optimize expected net utility. Do not confuse 'not found yet' with 'no target exists'.

ROUND: {round_no}
QUERIES SO FAR: {len(queried)}
HISTORY: {json.dumps(history, ensure_ascii=False)}

Choose STOP or QUERY one integer that has not already been queried.
"""


def run_scenario(client: NvidiaClient, scenario: Scenario) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    queried: list[int] = []
    outcomes: list[str] = []
    records: list[dict[str, Any]] = []
    found = False
    ended_by = "safety_cap"
    invalid_actions = 0

    for round_no in range(1, scenario.max_queries + 1):
        oracle = oracle_state(scenario.spec, queried)
        response = client.chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _prompt(scenario, queried, outcomes, round_no)},
            ],
            temperature=0.1,
            max_tokens=4096,
            reasoning_effort="low",
        )
        action = parse_action(response.content)

        valid_query = (
            action["decision"] == "QUERY"
            and action["query"] is not None
            and 1 <= action["query"] <= scenario.spec.n_candidates
            and action["query"] not in queried
        )

        decision_agreement = action["decision"] == oracle.decision
        query_matches_oracle: bool | None = None
        if action["decision"] == "QUERY":
            query_matches_oracle = valid_query and action["query"] == oracle.best_query

        record: dict[str, Any] = {
            "scenario_id": scenario.scenario_id,
            "round": round_no,
            "model": response.model,
            "n_failed_queries_before_action": len(queried),
            **action,
            "oracle_decision": oracle.decision,
            "oracle_best_query": oracle.best_query,
            "oracle_continuation_value": oracle.continuation_value,
            "oracle_advantage": oracle.advantage,
            "oracle_posterior_target_exists": oracle.posterior_target_exists,
            "oracle_next_hit_probability": oracle.next_hit_probability,
            "decision_agreement": decision_agreement,
            "query_matches_oracle": query_matches_oracle,
            "expected_gain_abs_error": abs(action["expected_net_gain"] - oracle.advantage),
            "reasoning_chars": len(response.reasoning) if response.reasoning else 0,
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "total_tokens": response.total_tokens,
        }

        if action["decision"] == "STOP":
            record["environment_outcome"] = "STOPPED"
            records.append(record)
            ended_by = "agent_stop"
            break

        if not valid_query:
            invalid_actions += 1
            record["environment_outcome"] = "INVALID_QUERY"
            records.append(record)
            continue

        q = int(action["query"])
        queried.append(q)
        if scenario.hidden_target == q:
            outcomes.append("FOUND")
            record["environment_outcome"] = "FOUND"
            records.append(record)
            found = True
            ended_by = "target_found"
            break

        outcomes.append("MISS")
        record["environment_outcome"] = "MISS"
        records.append(record)

    realized_utility = (scenario.spec.reward if found else 0.0) - scenario.spec.query_cost * len(queried)
    oracle_path = oracle_policy_path(scenario.spec, scenario.hidden_target)

    decision_records = [r for r in records if r["decision"] in {"STOP", "QUERY"}]
    query_records = [r for r in records if r["decision"] == "QUERY"]
    premature_stops = sum(
        r["decision"] == "STOP" and r["oracle_decision"] == "QUERY" for r in records
    )
    excess_continuations = sum(
        r["decision"] == "QUERY" and r["oracle_decision"] == "STOP" for r in records
    )

    summary = {
        "scenario_id": scenario.scenario_id,
        "model": records[-1]["model"] if records else client.model,
        "hidden_target": scenario.hidden_target,
        "ended_by": ended_by,
        "found": found,
        "n_queries": len(queried),
        "queried": queried,
        "realized_utility": realized_utility,
        "oracle_realized_utility": oracle_path["realized_utility"],
        "oracle_n_queries": oracle_path["n_queries"],
        "oracle_found": oracle_path["found"],
        "decision_agreement_rate": (
            sum(bool(r["decision_agreement"]) for r in decision_records) / len(decision_records)
            if decision_records else None
        ),
        "query_choice_agreement_rate": (
            sum(bool(r["query_matches_oracle"]) for r in query_records) / len(query_records)
            if query_records else None
        ),
        "mean_expected_gain_abs_error": (
            sum(float(r["expected_gain_abs_error"]) for r in decision_records) / len(decision_records)
            if decision_records else None
        ),
        "premature_stop_events": premature_stops,
        "excess_continuation_events": excess_continuations,
        "invalid_actions": invalid_actions,
    }
    return records, summary


def aggregate(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    def mean(key: str) -> float | None:
        vals = [float(s[key]) for s in summaries if s.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    return {
        "n_scenarios": len(summaries),
        "model": summaries[0]["model"] if summaries else None,
        "mean_decision_agreement_rate": mean("decision_agreement_rate"),
        "mean_query_choice_agreement_rate": mean("query_choice_agreement_rate"),
        "mean_expected_gain_abs_error": mean("mean_expected_gain_abs_error"),
        "total_premature_stop_events": sum(int(s["premature_stop_events"]) for s in summaries),
        "total_excess_continuation_events": sum(int(s["excess_continuation_events"]) for s in summaries),
        "total_invalid_actions": sum(int(s["invalid_actions"]) for s in summaries),
        "scenarios": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("results/exp02_bayesian_search"))
    args = parser.parse_args()

    client = NvidiaClient.from_env()
    print(f"Selected NVIDIA model: {client.resolve_model()}")

    all_records: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for scenario in scenarios():
        records, summary = run_scenario(client, scenario)
        all_records.extend(records)
        summaries.append(summary)
        print(
            f"{scenario.scenario_id}: ended_by={summary['ended_by']} queries={summary['n_queries']} "
            f"found={summary['found']} oracle_queries={summary['oracle_n_queries']} "
            f"premature={summary['premature_stop_events']} excess={summary['excess_continuation_events']}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "traces.jsonl").open("w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    result = aggregate(summaries)
    (args.out_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in result.items() if k != "scenarios"}, indent=2))


if __name__ == "__main__":
    main()
