from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

from floop_agents.stopping_metrics import local_stopping_metrics

from exp05_bellman_decomposition import make_wedge_spec, state_diagnostics
from exp07_value_substitution_audit import Exp07Client, condition_by_id
from exp09_recursive_depth_ladder import EXP09_CONDITION_IDS, build_depth_chain


EXP10_ARMS = {"implicit_narrative", "explicit_bellman"}
EXP10_CONDITION_IDS = set(EXP09_CONDITION_IDS)
EXP10_BACKUP_DEPTH = 3


COMMON_SYSTEM = """You are an autonomous search agent in a stopping experiment.
Your goal is to maximize expected NET utility.

All quantities labelled EXACT are authoritative. At every state you may either STOP
or QUERY the indicated unqueried candidate. A miss is not terminal: after a miss you
may still stop or continue querying according to the information supplied below.

Return ONLY one JSON object with exactly these keys:
{
  "decision": "STOP or QUERY",
  "query": 1,
  "evaluated_query": 1,
  "estimated_value_after_miss": 0.0,
  "full_expected_net_gain": 0.0,
  "reason": "brief rationale"
}

`estimated_value_after_miss` is your estimate of the optimal net value from the
state immediately after the current query misses. `full_expected_net_gain` is your
estimate of the current query's net value versus stopping now. If decision is QUERY,
query must equal evaluated_query. If decision is STOP, set query to null. Do not
wrap the JSON in markdown.
"""


EXPLICIT_INSTRUCTION = """
REPRESENTATION CONDITION: EXPLICIT BELLMAN

Use the following Bellman recurrence explicitly and work backward from the supplied
boundary value:

V(s_j) = max(0,
    - query_cost
    + hit_probability(s_j) * reward
    + miss_probability(s_j) * V(s_{j+1}))

Then compute the current query value:
Q(s_0,q_0) = - query_cost
              + hit_probability(s_0,q_0) * reward
              + miss_probability(s_0,q_0) * V(s_1).

Report V(s_1) as estimated_value_after_miss and Q(s_0,q_0) as
full_expected_net_gain. Choose QUERY iff Q(s_0,q_0) is positive; otherwise STOP.
"""


IMPLICIT_INSTRUCTION = """
REPRESENTATION CONDITION: IMPLICIT NARRATIVE

Use the complete trajectory information below to decide whether querying the current
candidate is worth its cost. You are not given precomputed values for the
intermediate post-miss states. Determine whatever intermediate quantities you need
from the task description. Do not assume a miss ends the process.

No particular mathematical decomposition is required. Report your estimate of the
optimal value immediately after the current miss and your estimate of the current
query's net value, then choose STOP or QUERY.
"""


def _number(value: Any, *, default: float = 0.0) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(x) or math.isinf(x):
        return default
    return x


def _integer_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_exp10_response(text: str) -> dict[str, Any]:
    raw = text.strip()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"Model did not return JSON: {raw[:400]}")
        obj = json.loads(raw[start : end + 1])

    decision = str(obj.get("decision", "")).strip().upper()
    if decision not in {"STOP", "QUERY"}:
        raise ValueError(f"Invalid decision: {decision}")

    query = _integer_or_none(obj.get("query")) if decision == "QUERY" else None
    evaluated_query = _integer_or_none(obj.get("evaluated_query"))
    value_after_miss = _number(
        obj.get("estimated_value_after_miss"), default=float("nan")
    )
    full_q = _number(obj.get("full_expected_net_gain"), default=float("nan"))
    if math.isnan(value_after_miss) or value_after_miss < 0.0 or math.isnan(full_q):
        raise ValueError("Invalid reported value")

    return {
        "decision": decision,
        "query": query,
        "evaluated_query": evaluated_query,
        "estimated_value_after_miss": value_after_miss,
        "full_expected_net_gain": full_q,
        "reason": str(obj.get("reason", "")).strip()[:800],
    }


def common_numeric_prompt(spec) -> tuple[str, dict[str, Any]]:
    chain = build_depth_chain(spec, EXP10_BACKUP_DEPTH)
    current = chain["current"]
    q0 = chain["current_query"]

    lines = [
        "SEARCH WORLD",
        f"Number of candidates: {spec.n_candidates}",
        f"Reward for finding the target: {spec.reward:.9f}",
        f"Cost of each query: {spec.query_cost:.9f}",
        "",
        "EXACT CURRENT FACTS",
        f"Current candidate: {q0}",
        f"Probability current query hits: {current.next_hit_probability:.9f}",
        f"Probability current query misses: {1.0 - current.next_hit_probability:.9f}",
    ]

    for depth in range(1, EXP10_BACKUP_DEPTH + 1):
        row = chain["state_rows"][depth]
        lines.extend(
            [
                "",
                f"EXACT FACTS AFTER {depth} CONSECUTIVE MISS(ES)",
                f"Next candidate available under the supplied optimal trajectory: {row['candidate']}",
                f"Posterior probability a target still exists: {row['state'].posterior_target_exists:.9f}",
                f"Probability that next query hits: {row['hit']:.9f}",
                f"Probability that next query misses: {row['miss']:.9f}",
            ]
        )

    boundary_depth = EXP10_BACKUP_DEPTH + 1
    boundary = chain["boundary"]
    lines.extend(
        [
            "",
            f"EXACT FACTS AFTER {boundary_depth} CONSECUTIVE MISSES",
            f"Posterior probability a target still exists: {boundary.posterior_target_exists:.9f}",
            f"Exact optimal net continuation value from this boundary state: {chain['boundary_value']:.9f}",
        ]
    )
    return "\n".join(lines), chain


def prompt_for_arm(spec, arm: str) -> tuple[str, dict[str, Any]]:
    if arm not in EXP10_ARMS:
        raise ValueError(f"Unsupported Exp10 arm: {arm}")
    numeric, chain = common_numeric_prompt(spec)
    instruction = EXPLICIT_INSTRUCTION if arm == "explicit_bellman" else IMPLICIT_INSTRUCTION
    return numeric + "\n\n" + instruction, chain


def probe_representation(client: Exp07Client, condition, arm: str) -> dict[str, Any]:
    spec = make_wedge_spec(condition)
    diagnostics = state_diagnostics(spec, [])
    prompt, chain = prompt_for_arm(spec, arm)
    q0 = int(chain["current_query"])
    exact_v1 = float(chain["exact_backups"][1])
    exact_q0 = float(chain["exact_current_q"])

    response = client.chat(
        [
            {"role": "system", "content": COMMON_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=3072,
        reasoning_effort="low",
    )
    action = parse_exp10_response(response.content)

    action_valid = (
        action["evaluated_query"] == q0
        and (action["decision"] != "QUERY" or action["query"] == q0)
    )
    reported_v1 = float(action["estimated_value_after_miss"])
    reported_q0 = float(action["full_expected_net_gain"])
    current = chain["current"]
    recomposed_q0 = (
        -spec.query_cost
        + current.next_hit_probability * spec.reward
        + (1.0 - current.next_hit_probability) * reported_v1
    )

    stopping = local_stopping_metrics(
        agent_decision=action["decision"],
        oracle_continuation_value=float(diagnostics["oracle_advantage"]),
        agent_reported_continuation_value=reported_q0,
        chosen_query_value=(
            exact_q0 if action["decision"] == "QUERY" and action_valid else None
        ),
    )

    return {
        "condition_id": condition.condition_id,
        "query_cost": condition.query_cost,
        "arm": arm,
        "backup_depth": EXP10_BACKUP_DEPTH,
        "model": response.model,
        **action,
        **diagnostics,
        "action_valid": action_valid,
        "fixed_evaluated_query": q0,
        "exact_boundary_value": chain["boundary_value"],
        "exact_first_successor_value": exact_v1,
        "exact_current_query_value": exact_q0,
        "first_successor_value_error": reported_v1 - exact_v1,
        "current_query_value_error": reported_q0 - exact_q0,
        "reported_current_closure_error": reported_q0 - recomposed_q0,
        "decision_agreement": action["decision"] == diagnostics["oracle_decision"],
        "decision_matches_reported_value_sign": (
            (action["decision"] == "QUERY" and reported_q0 > 0)
            or (action["decision"] == "STOP" and reported_q0 <= 0)
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


def run_single_probe(
    *,
    arm: str,
    condition_id: str,
    repeat_index: int,
    out_path: Path,
) -> dict[str, Any]:
    if arm not in EXP10_ARMS:
        raise ValueError(f"Unsupported Exp10 arm: {arm}")
    if condition_id not in EXP10_CONDITION_IDS:
        raise ValueError(f"Unsupported Exp10 condition: {condition_id}")
    if repeat_index < 0:
        raise ValueError("repeat_index must be non-negative")

    client = Exp07Client.from_env()
    model = client.resolve_model()
    condition = condition_by_id(condition_id)
    reasoning_mode = os.getenv("EXP07_REASONING_MODE", "preserve").strip().lower()

    try:
        record = probe_representation(client, condition, arm)
        record["repeat"] = repeat_index
        record["replication_model"] = model
        record["reasoning_mode"] = reasoning_mode
    except Exception as exc:
        record = {
            "condition_id": condition.condition_id,
            "query_cost": condition.query_cost,
            "arm": arm,
            "backup_depth": EXP10_BACKUP_DEPTH,
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
    parser.add_argument("--arm", required=True, choices=sorted(EXP10_ARMS))
    parser.add_argument("--condition-id", required=True)
    parser.add_argument("--repeat-index", required=True, type=int)
    parser.add_argument("--out", type=Path, default=Path("results/exp10_matrix/probe.json"))
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
