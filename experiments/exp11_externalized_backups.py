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
from exp10_representation_ablation import common_numeric_prompt


EXP11_ARMS = {"formula_only", "forced_backups"}
EXP11_CONDITION_IDS = {
    "cost_0100_wedge",
    "cost_0120_wedge",
    "cost_0125_wedge",
    "cost_0130_dynamic_stop",
}
EXP11_BACKUP_DEPTH = 3


COMMON_SYSTEM = """You are an autonomous search agent in a stopping experiment.
Your goal is to maximize expected NET utility.

All quantities labelled EXACT are authoritative. A miss is not terminal: after a
miss you may still STOP or continue QUERYING according to the supplied trajectory.

Use this Bellman recurrence exactly:
V(s_j) = max(0,
    - query_cost
    + hit_probability(s_j) * reward
    + miss_probability(s_j) * V(s_{j+1}))

Then compute the current query value:
Q(s_0,q_0) = - query_cost
              + hit_probability(s_0,q_0) * reward
              + miss_probability(s_0,q_0) * V(s_1).

Choose QUERY iff Q(s_0,q_0) is positive; otherwise choose STOP.
Do not wrap the JSON in markdown.
"""


FORMULA_ONLY_INSTRUCTION = """
EXTERNALIZATION CONDITION: FORMULA ONLY

Work backward from the supplied boundary value using the Bellman recurrence. You may
compute intermediate state values as needed, but do NOT report those intermediate
values. Return ONLY this JSON object:
{
  "decision": "STOP or QUERY",
  "query": 1,
  "evaluated_query": 1,
  "estimated_value_after_miss": 0.0,
  "full_expected_net_gain": 0.0,
  "reason": "brief rationale"
}

`estimated_value_after_miss` is V(s_1). If decision is STOP, set query to null.
"""


FORCED_BACKUPS_INSTRUCTION = """
EXTERNALIZATION CONDITION: FORCED INTERMEDIATE BACKUPS

Work backward from the supplied boundary value using the SAME Bellman recurrence.
You MUST explicitly report every intermediate state value so the computation can be
audited. Return ONLY this JSON object:
{
  "decision": "STOP or QUERY",
  "query": 1,
  "evaluated_query": 1,
  "backups": [
    {"state_depth": 3, "value": 0.0},
    {"state_depth": 2, "value": 0.0},
    {"state_depth": 1, "value": 0.0}
  ],
  "estimated_value_after_miss": 0.0,
  "full_expected_net_gain": 0.0,
  "reason": "brief rationale"
}

Report exactly one backup for state depths 3, 2, and 1, in that order.
`estimated_value_after_miss` must equal the reported value for state depth 1.
If decision is STOP, set query to null.
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


def _load_json(text: str) -> dict[str, Any]:
    raw = text.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"Model did not return JSON: {raw[:400]}")
        return json.loads(raw[start : end + 1])


def parse_response(text: str, arm: str) -> dict[str, Any]:
    obj = _load_json(text)
    decision = str(obj.get("decision", "")).strip().upper()
    if decision not in {"STOP", "QUERY"}:
        raise ValueError(f"Invalid decision: {decision}")

    query = _integer_or_none(obj.get("query")) if decision == "QUERY" else None
    evaluated_query = _integer_or_none(obj.get("evaluated_query"))
    v1 = _number(obj.get("estimated_value_after_miss"), default=float("nan"))
    q0 = _number(obj.get("full_expected_net_gain"), default=float("nan"))
    if math.isnan(v1) or v1 < 0 or math.isnan(q0):
        raise ValueError("Invalid reported value")

    parsed: dict[str, Any] = {
        "decision": decision,
        "query": query,
        "evaluated_query": evaluated_query,
        "estimated_value_after_miss": v1,
        "full_expected_net_gain": q0,
        "reason": str(obj.get("reason", "")).strip()[:800],
    }

    if arm == "forced_backups":
        raw_backups = obj.get("backups")
        if not isinstance(raw_backups, list):
            raise ValueError("backups must be a list")
        backups: dict[int, float] = {}
        observed_order: list[int] = []
        for item in raw_backups:
            if not isinstance(item, dict):
                raise ValueError("Each backup entry must be an object")
            depth = _integer_or_none(item.get("state_depth"))
            value = _number(item.get("value"), default=float("nan"))
            if depth not in {1, 2, 3} or math.isnan(value) or value < 0:
                raise ValueError(f"Invalid backup entry: {item}")
            if depth in backups:
                raise ValueError(f"Duplicate backup depth: {depth}")
            backups[depth] = value
            observed_order.append(depth)
        if set(backups) != {1, 2, 3}:
            raise ValueError(f"Missing backup depths: {sorted(backups)}")
        if observed_order != [3, 2, 1]:
            raise ValueError(f"Backups must be reported in order [3,2,1], got {observed_order}")
        parsed["reported_backups"] = backups
        parsed["value_after_miss_consistency_error"] = v1 - backups[1]
    else:
        parsed["reported_backups"] = None
        parsed["value_after_miss_consistency_error"] = None

    return parsed


def prompt_for_arm(spec, arm: str) -> tuple[str, dict[str, Any]]:
    if arm not in EXP11_ARMS:
        raise ValueError(f"Unsupported Exp11 arm: {arm}")
    numeric, chain = common_numeric_prompt(spec)
    instruction = FORMULA_ONLY_INSTRUCTION if arm == "formula_only" else FORCED_BACKUPS_INSTRUCTION
    return numeric + "\n\n" + instruction, chain


def probe_externalization(client: Exp07Client, condition, arm: str) -> dict[str, Any]:
    spec = make_wedge_spec(condition)
    diagnostics = state_diagnostics(spec, [])
    prompt, chain = prompt_for_arm(spec, arm)
    q0 = int(chain["current_query"])
    exact_backups = {int(k): float(v) for k, v in chain["exact_backups"].items()}
    exact_v1 = exact_backups[1]
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
    action = parse_response(response.content, arm)

    action_valid = action["evaluated_query"] == q0 and (action["decision"] != "QUERY" or action["query"] == q0)
    reported_v1 = float(action["estimated_value_after_miss"])
    reported_q0 = float(action["full_expected_net_gain"])
    current = chain["current"]
    recomposed_q0 = -spec.query_cost + current.next_hit_probability * spec.reward + (1.0 - current.next_hit_probability) * reported_v1

    backup_errors = None
    backup_closure_errors = None
    max_abs_backup_error = None
    max_abs_backup_closure_error = None
    if arm == "forced_backups":
        reported = action["reported_backups"]
        backup_errors = {str(depth): float(reported[depth]) - exact_backups[depth] for depth in (1, 2, 3)}
        closure = {}
        for depth in (3, 2, 1):
            row = chain["state_rows"][depth]
            next_value = float(chain["boundary_value"]) if depth == 3 else float(reported[depth + 1])
            recomposed = max(0.0, -spec.query_cost + row["hit"] * spec.reward + row["miss"] * next_value)
            closure[str(depth)] = float(reported[depth]) - recomposed
        backup_closure_errors = closure
        max_abs_backup_error = max(abs(v) for v in backup_errors.values())
        max_abs_backup_closure_error = max(abs(v) for v in closure.values())

    stopping = local_stopping_metrics(
        agent_decision=action["decision"],
        oracle_continuation_value=float(diagnostics["oracle_advantage"]),
        agent_reported_continuation_value=reported_q0,
        chosen_query_value=exact_q0 if action["decision"] == "QUERY" and action_valid else None,
    )

    return {
        "condition_id": condition.condition_id,
        "query_cost": condition.query_cost,
        "arm": arm,
        "backup_depth": EXP11_BACKUP_DEPTH,
        "model": response.model,
        **action,
        **diagnostics,
        "action_valid": action_valid,
        "fixed_evaluated_query": q0,
        "exact_boundary_value": float(chain["boundary_value"]),
        "exact_backups": {str(k): v for k, v in exact_backups.items()},
        "exact_first_successor_value": exact_v1,
        "exact_current_query_value": exact_q0,
        "first_successor_value_error": reported_v1 - exact_v1,
        "current_query_value_error": reported_q0 - exact_q0,
        "reported_current_closure_error": reported_q0 - recomposed_q0,
        "backup_errors": backup_errors,
        "backup_closure_errors": backup_closure_errors,
        "max_abs_backup_error": max_abs_backup_error,
        "max_abs_backup_closure_error": max_abs_backup_closure_error,
        "decision_agreement": action["decision"] == diagnostics["oracle_decision"],
        "decision_matches_reported_value_sign": (action["decision"] == "QUERY" and reported_q0 > 0) or (action["decision"] == "STOP" and reported_q0 <= 0),
        "premature_stop_regret": stopping.premature_stop_regret,
        "excess_continuation_regret": stopping.excess_continuation_regret,
        "query_selection_regret": stopping.query_selection_regret,
        "local_decision_regret": stopping.local_decision_regret,
        "reasoning_chars": len(response.reasoning) if response.reasoning else 0,
        "prompt_tokens": response.prompt_tokens,
        "completion_tokens": response.completion_tokens,
        "total_tokens": response.total_tokens,
    }


def run_single_probe(*, arm: str, condition_id: str, repeat_index: int, out_path: Path) -> dict[str, Any]:
    if arm not in EXP11_ARMS:
        raise ValueError(f"Unsupported Exp11 arm: {arm}")
    if condition_id not in EXP11_CONDITION_IDS:
        raise ValueError(f"Unsupported Exp11 condition: {condition_id}")
    if repeat_index < 0:
        raise ValueError("repeat_index must be non-negative")

    client = Exp07Client.from_env()
    model = client.resolve_model()
    condition = condition_by_id(condition_id)
    reasoning_mode = os.getenv("EXP07_REASONING_MODE", "preserve").strip().lower()

    try:
        record = probe_externalization(client, condition, arm)
        record["repeat"] = repeat_index
        record["replication_model"] = model
        record["reasoning_mode"] = reasoning_mode
    except Exception as exc:
        record = {
            "condition_id": condition.condition_id,
            "query_cost": condition.query_cost,
            "arm": arm,
            "backup_depth": EXP11_BACKUP_DEPTH,
            "repeat": repeat_index,
            "replication_model": model,
            "reasoning_mode": reasoning_mode,
            "error": f"{type(exc).__name__}: {exc}",
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))
    print("PROBE_VALIDITY: " + ("PASS" if "error" not in record else "FAIL") + f" model={model} arm={arm} condition={condition_id} repeat={repeat_index}")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=sorted(EXP11_ARMS))
    parser.add_argument("--condition-id", required=True)
    parser.add_argument("--repeat-index", required=True, type=int)
    parser.add_argument("--out", type=Path, default=Path("results/exp11_matrix/probe.json"))
    parser.add_argument("--require-valid", action="store_true")
    args = parser.parse_args()

    record = run_single_probe(arm=args.arm, condition_id=args.condition_id, repeat_index=args.repeat_index, out_path=args.out)
    if args.require_valid and "error" in record:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
