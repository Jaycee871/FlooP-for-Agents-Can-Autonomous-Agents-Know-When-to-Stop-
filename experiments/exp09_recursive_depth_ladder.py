from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

from floop_agents.search_arena import oracle_state, query_action_value
from floop_agents.stopping_metrics import local_stopping_metrics

from exp04_calibrated_state_ablation import highest_mass_remaining_candidate
from exp05_bellman_decomposition import make_wedge_spec, state_diagnostics
from exp07_value_substitution_audit import Exp07Client, condition_by_id


EXP09_CONDITION_IDS = {
    "cost_0100_wedge",
    "cost_0120_wedge",
    "cost_0125_wedge",
    "cost_0130_dynamic_stop",
}
EXP09_BACKUP_DEPTHS = {2, 3}


EXP09_SYSTEM_PROMPT = """You are an autonomous search agent in a stopping experiment.
Your goal is to maximize expected NET utility.

All state probabilities and candidates labelled EXACT are authoritative. The only
value supplied by the oracle is the continuation value at the final boundary state.
You must work backward through the explicitly listed states using Bellman backups.

For each state s_j, use:
V(s_j) = max(0,
    - query cost
    + hit_probability(s_j) * reward
    + miss_probability(s_j) * V(s_{j+1}))

After reconstructing V(s_1), compute the current query value:
Q(s_0,q_0) = - query cost
              + hit_probability(s_0,q_0) * reward
              + miss_probability(s_0,q_0) * V(s_1).

Choose QUERY iff Q(s_0,q_0) is positive; otherwise choose STOP.

Return ONLY one JSON object with exactly these keys:
{
  "decision": "STOP or QUERY",
  "query": 1,
  "evaluated_query": 1,
  "backup_depth": 2,
  "backups": [
    {"state_depth": 2, "value": 0.0},
    {"state_depth": 1, "value": 0.0}
  ],
  "estimated_value_after_miss": 0.0,
  "full_expected_net_gain": 0.0,
  "reason": "brief rationale"
}

Report exactly one backup entry for every state depth from backup_depth down to 1.
`estimated_value_after_miss` must equal your reported V(s_1). If decision is QUERY,
`query` must equal `evaluated_query`; if decision is STOP, set `query` to null.
Do not wrap the JSON in markdown.
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


def parse_depth_response(text: str, expected_depth: int) -> dict[str, Any]:
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
        raise ValueError(f"Invalid decision: {decision}")

    reported_depth = _integer_or_none(obj.get("backup_depth"))
    if reported_depth != expected_depth:
        raise ValueError(
            f"backup_depth mismatch: expected {expected_depth}, got {reported_depth}"
        )

    raw_backups = obj.get("backups")
    if not isinstance(raw_backups, list):
        raise ValueError("backups must be a list")

    backups: dict[int, float] = {}
    for item in raw_backups:
        if not isinstance(item, dict):
            continue
        depth = _integer_or_none(item.get("state_depth"))
        if depth is None or depth < 1 or depth > expected_depth:
            continue
        if depth in backups:
            raise ValueError(f"Duplicate backup state_depth={depth}")
        value = _number(item.get("value"), default=float("nan"))
        if math.isnan(value) or value < 0.0:
            raise ValueError(f"Invalid backup value at depth {depth}")
        backups[depth] = value

    expected = set(range(1, expected_depth + 1))
    if set(backups) != expected:
        raise ValueError(
            f"Missing backup depths: expected {sorted(expected)}, got {sorted(backups)}"
        )

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
        "backup_depth": reported_depth,
        "reported_backups": backups,
        "estimated_value_after_miss": value_after_miss,
        "full_expected_net_gain": full_q,
        "reason": str(obj.get("reason", "")).strip()[:800],
    }


def build_depth_chain(spec, backup_depth: int) -> dict[str, Any]:
    if backup_depth not in EXP09_BACKUP_DEPTHS:
        raise ValueError(f"Unsupported Exp09 backup depth: {backup_depth}")

    current = oracle_state(spec, [])
    q0 = highest_mass_remaining_candidate(spec, [])
    if q0 is None:
        raise RuntimeError("No current candidate available")

    queried = [q0]
    state_rows: dict[int, dict[str, Any]] = {}
    for depth in range(1, backup_depth + 1):
        state = oracle_state(spec, queried)
        candidate = highest_mass_remaining_candidate(spec, queried)
        if candidate is None:
            raise RuntimeError(f"No candidate available at state depth {depth}")
        state_rows[depth] = {
            "state": state,
            "candidate": candidate,
            "hit": state.next_hit_probability,
            "miss": 1.0 - state.next_hit_probability,
        }
        queried.append(candidate)

    boundary = oracle_state(spec, queried)
    boundary_value = max(0.0, boundary.continuation_value)

    exact_backups: dict[int, float] = {}
    identity_errors: dict[int, float] = {}
    next_value = boundary_value
    for depth in range(backup_depth, 0, -1):
        row = state_rows[depth]
        value = max(
            0.0,
            -spec.query_cost
            + row["hit"] * spec.reward
            + row["miss"] * next_value,
        )
        oracle_value = max(0.0, row["state"].continuation_value)
        exact_backups[depth] = value
        identity_errors[depth] = value - oracle_value
        next_value = value

    exact_current_q = query_action_value(spec, [], q0)
    return {
        "current": current,
        "current_query": q0,
        "state_rows": state_rows,
        "boundary": boundary,
        "boundary_value": boundary_value,
        "exact_backups": exact_backups,
        "identity_errors": identity_errors,
        "exact_current_q": exact_current_q,
    }


def depth_prompt(spec, backup_depth: int) -> tuple[str, dict[str, Any]]:
    chain = build_depth_chain(spec, backup_depth)
    current = chain["current"]
    q0 = chain["current_query"]

    lines = [
        "SEARCH WORLD",
        f"Candidates: {spec.n_candidates}",
        f"Reward: {spec.reward:.9f}",
        f"Query cost: {spec.query_cost:.9f}",
        "",
        "EXACT CURRENT STATE s_0",
        f"Current candidate q_0: {q0}",
        f"Hit probability: {current.next_hit_probability:.9f}",
        f"Miss probability: {1.0 - current.next_hit_probability:.9f}",
        "",
        f"BACKUP DEPTH: {backup_depth}",
    ]

    for depth in range(1, backup_depth + 1):
        row = chain["state_rows"][depth]
        lines.extend(
            [
                "",
                f"EXACT STATE s_{depth} (after {depth} consecutive miss(es))",
                f"Oracle-optimal candidate q_{depth}: {row['candidate']}",
                f"Posterior target-exists probability: {row['state'].posterior_target_exists:.9f}",
                f"Hit probability: {row['hit']:.9f}",
                f"Miss probability: {row['miss']:.9f}",
            ]
        )

    boundary_depth = backup_depth + 1
    boundary = chain["boundary"]
    lines.extend(
        [
            "",
            f"EXACT ORACLE BOUNDARY VALUE AT s_{boundary_depth}",
            f"Posterior target-exists probability: {boundary.posterior_target_exists:.9f}",
            f"Exact optimal continuation value V(s_{boundary_depth}): {chain['boundary_value']:.9f}",
            "",
            "No intermediate continuation values are supplied.",
            f"Starting from V(s_{boundary_depth}), compute exactly {backup_depth} Bellman backups",
            "backward to V(s_1), then compute Q(s_0,q_0), then choose STOP or QUERY.",
        ]
    )
    return "\n".join(lines), chain


def probe_depth_ladder(client: Exp07Client, condition, backup_depth: int) -> dict[str, Any]:
    spec = make_wedge_spec(condition)
    diagnostics = state_diagnostics(spec, [])
    prompt, chain = depth_prompt(spec, backup_depth)
    q0 = int(chain["current_query"])

    response = client.chat(
        [
            {"role": "system", "content": EXP09_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=4096,
        reasoning_effort="low",
    )
    action = parse_depth_response(response.content, backup_depth)

    action_valid = (
        action["evaluated_query"] == q0
        and (action["decision"] != "QUERY" or action["query"] == q0)
    )

    reported_backups = action["reported_backups"]
    exact_backups = chain["exact_backups"]
    backup_errors = {
        str(depth): reported_backups[depth] - exact_backups[depth]
        for depth in range(1, backup_depth + 1)
    }

    closure_errors: dict[str, float] = {}
    for depth in range(backup_depth, 0, -1):
        row = chain["state_rows"][depth]
        next_value = (
            chain["boundary_value"]
            if depth == backup_depth
            else reported_backups[depth + 1]
        )
        recomposed = max(
            0.0,
            -spec.query_cost
            + row["hit"] * spec.reward
            + row["miss"] * next_value,
        )
        closure_errors[str(depth)] = reported_backups[depth] - recomposed

    current = chain["current"]
    recomposed_current = (
        -spec.query_cost
        + current.next_hit_probability * spec.reward
        + (1.0 - current.next_hit_probability) * reported_backups[1]
    )
    exact_current_q = float(chain["exact_current_q"])

    stopping = local_stopping_metrics(
        agent_decision=action["decision"],
        oracle_continuation_value=float(diagnostics["oracle_advantage"]),
        agent_reported_continuation_value=float(action["full_expected_net_gain"]),
        chosen_query_value=(
            exact_current_q if action["decision"] == "QUERY" and action_valid else None
        ),
    )

    return {
        "condition_id": condition.condition_id,
        "query_cost": condition.query_cost,
        "backup_depth": backup_depth,
        "model": response.model,
        **action,
        **diagnostics,
        "action_valid": action_valid,
        "fixed_evaluated_query": q0,
        "exact_boundary_value": chain["boundary_value"],
        "exact_backups": {str(k): v for k, v in exact_backups.items()},
        "oracle_identity_errors": {
            str(k): v for k, v in chain["identity_errors"].items()
        },
        "max_abs_oracle_identity_error": max(
            abs(v) for v in chain["identity_errors"].values()
        ),
        "backup_errors": backup_errors,
        "backup_closure_errors": closure_errors,
        "max_abs_backup_error": max(abs(v) for v in backup_errors.values()),
        "max_abs_backup_closure_error": max(abs(v) for v in closure_errors.values()),
        "value_after_miss_consistency_error": (
            float(action["estimated_value_after_miss"]) - reported_backups[1]
        ),
        "exact_first_successor_value": exact_backups[1],
        "first_successor_value_error": (
            float(action["estimated_value_after_miss"]) - exact_backups[1]
        ),
        "exact_current_query_value": exact_current_q,
        "current_query_value_error": (
            float(action["full_expected_net_gain"]) - exact_current_q
        ),
        "reported_current_closure_error": (
            float(action["full_expected_net_gain"]) - recomposed_current
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


def run_single_probe(
    *,
    condition_id: str,
    backup_depth: int,
    repeat_index: int,
    out_path: Path,
) -> dict[str, Any]:
    if condition_id not in EXP09_CONDITION_IDS:
        raise ValueError(f"Unsupported Exp09 condition: {condition_id}")
    if backup_depth not in EXP09_BACKUP_DEPTHS:
        raise ValueError(f"Unsupported Exp09 backup depth: {backup_depth}")
    if repeat_index < 0:
        raise ValueError("repeat_index must be non-negative")

    client = Exp07Client.from_env()
    model = client.resolve_model()
    condition = condition_by_id(condition_id)
    reasoning_mode = os.getenv("EXP07_REASONING_MODE", "preserve").strip().lower()

    try:
        record = probe_depth_ladder(client, condition, backup_depth)
        record["repeat"] = repeat_index
        record["replication_model"] = model
        record["reasoning_mode"] = reasoning_mode
    except Exception as exc:
        record = {
            "condition_id": condition.condition_id,
            "query_cost": condition.query_cost,
            "backup_depth": backup_depth,
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
        + f" model={model} depth={backup_depth} condition={condition_id} repeat={repeat_index}"
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition-id", required=True)
    parser.add_argument("--backup-depth", required=True, type=int)
    parser.add_argument("--repeat-index", required=True, type=int)
    parser.add_argument("--out", type=Path, default=Path("results/exp09_matrix/probe.json"))
    parser.add_argument("--require-valid", action="store_true")
    args = parser.parse_args()

    record = run_single_probe(
        condition_id=args.condition_id,
        backup_depth=args.backup_depth,
        repeat_index=args.repeat_index,
        out_path=args.out,
    )
    if args.require_valid and "error" in record:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
