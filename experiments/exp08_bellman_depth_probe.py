from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from floop_agents.search_arena import oracle_state, query_action_value
from floop_agents.stopping_metrics import local_stopping_metrics

from exp04_calibrated_state_ablation import highest_mass_remaining_candidate
from exp05_bellman_decomposition import make_wedge_spec, state_diagnostics
from exp05b_transition_audit import parse_transition_response
from exp07_value_substitution_audit import (
    EXP07_CONDITION_IDS,
    Exp07Client,
    condition_by_id,
)


EXP08_CONDITION_IDS = set(EXP07_CONDITION_IDS)


EXP08_SYSTEM_PROMPT = """You are an autonomous search agent in a stopping experiment.
Your goal is to maximize expected NET utility.

You will receive exact statistics for the current state, the state after the first
query misses, and an exact oracle boundary value after one additional miss. Treat
all quantities explicitly labelled EXACT as authoritative. The first successor
value itself is deliberately NOT supplied: you must construct it with exactly one
Bellman backup.

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

Copy the exact supplied state probabilities/candidate into the corresponding JSON
fields. estimated_value_after_miss must be the value you compute for the FIRST
successor state. full_expected_net_gain is the value you then compute for the
CURRENT query.

Use these equations:
V(first successor) = max(0,
    - query cost
    + first-successor hit probability * reward
    + first-successor miss probability * V(second successor))

Q(current query) =
    - query cost
    + current hit probability * reward
    + current miss probability * V(first successor)

Choose QUERY iff Q(current query) is positive; otherwise choose STOP.
If decision is QUERY, query must equal evaluated_query. If decision is STOP, set
query to null. Do not wrap the JSON in markdown.
"""


def depth2_prompt(spec, evaluated_query: int) -> tuple[str, dict[str, Any]]:
    current = oracle_state(spec, [])
    successor_queried = [evaluated_query]
    successor = oracle_state(spec, successor_queried)
    next_query = highest_mass_remaining_candidate(spec, successor_queried)
    if next_query is None:
        raise RuntimeError("Exp08 requires a nonterminal first successor state")

    second_successor_queried = [evaluated_query, next_query]
    second_successor = oracle_state(spec, second_successor_queried)
    second_successor_value = max(0.0, second_successor.continuation_value)
    successor_value = max(0.0, successor.continuation_value)

    one_backup_value = max(
        0.0,
        -spec.query_cost
        + successor.next_hit_probability * spec.reward
        + (1.0 - successor.next_hit_probability) * second_successor_value,
    )

    target_masses = ", ".join(
        f"{candidate}:{mass:.9f}" for candidate, mass in sorted(spec.target_masses.items())
    )
    prompt = f"""SEARCH WORLD
There are {spec.n_candidates} candidate locations. At most one target exists.
Prior probability that a target exists somewhere: {spec.existence_prob:.9f}
Probability no target exists: {spec.no_target_mass:.9f}
Target mass by candidate: {target_masses}
Reward for finding the target: {spec.reward:.9f}
Cost per query: {spec.query_cost:.9f}
No candidates have yet been queried.

EXACT CURRENT STATE
Candidate to evaluate now: {evaluated_query}
Current hit probability for that candidate: {current.next_hit_probability:.9f}
Current miss probability: {1.0 - current.next_hit_probability:.9f}

EXACT FIRST SUCCESSOR STATE
Assume candidate {evaluated_query} is queried and MISSES.
Posterior probability a target still exists: {successor.posterior_target_exists:.9f}
Oracle-optimal next candidate: {next_query}
Hit probability of that next candidate: {successor.next_hit_probability:.9f}
Miss probability of that next candidate: {1.0 - successor.next_hit_probability:.9f}

EXACT SECOND-SUCCESSOR BOUNDARY VALUE
Now assume candidate {next_query} is also queried and MISSES.
Exact optimal continuation value from that second-successor state:
{second_successor_value:.9f}

IMPORTANT
The exact value of the FIRST successor state is intentionally NOT given.
Compute it yourself with exactly one Bellman backup using the supplied
second-successor boundary value. Then use your computed first-successor value in
the current-state Bellman equation and choose STOP or QUERY from the sign.
"""

    exact = {
        "current": current,
        "successor": successor,
        "next_query": next_query,
        "second_successor": second_successor,
        "second_successor_value": second_successor_value,
        "successor_value": successor_value,
        "one_backup_value": one_backup_value,
    }
    return prompt, exact


def probe_depth2(client: Exp07Client, condition) -> dict[str, Any]:
    spec = make_wedge_spec(condition)
    queried: list[int] = []
    evaluated_query = highest_mass_remaining_candidate(spec, queried)
    if evaluated_query is None:
        raise RuntimeError("No candidate available for Exp08 audit")

    diagnostics = state_diagnostics(spec, queried)
    prompt, exact = depth2_prompt(spec, evaluated_query)
    current = exact["current"]
    successor = exact["successor"]
    next_query = exact["next_query"]
    second_successor = exact["second_successor"]
    exact_v2 = float(exact["second_successor_value"])
    exact_v1 = float(exact["successor_value"])
    one_backup_value = float(exact["one_backup_value"])
    exact_q0 = query_action_value(spec, queried, evaluated_query)

    response = client.chat(
        [
            {"role": "system", "content": EXP08_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=3072,
        reasoning_effort="low",
    )
    action = parse_transition_response(response.content)

    evaluated_query_matches = action["evaluated_query"] == evaluated_query
    query_valid = action["decision"] != "QUERY" or action["query"] == evaluated_query
    action_valid = evaluated_query_matches and query_valid

    reported_v1 = float(action["estimated_value_after_miss"])
    reported_q0 = float(action["full_expected_net_gain"])
    recomposed_q0 = (
        -spec.query_cost
        + float(action["current_hit_probability"]) * spec.reward
        + float(action["miss_probability"]) * reported_v1
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
        "model": response.model,
        **action,
        **diagnostics,
        "fixed_evaluated_query": evaluated_query,
        "action_valid": action_valid,
        "exact_current_hit_probability": current.next_hit_probability,
        "exact_current_miss_probability": 1.0 - current.next_hit_probability,
        "exact_target_exists_after_first_miss": successor.posterior_target_exists,
        "exact_best_hit_probability_after_first_miss": successor.next_hit_probability,
        "exact_best_candidate_after_first_miss": next_query,
        "exact_second_successor_target_exists": second_successor.posterior_target_exists,
        "exact_second_successor_next_hit": second_successor.next_hit_probability,
        "exact_second_successor_value": exact_v2,
        "exact_first_successor_value": exact_v1,
        "exact_one_backup_value": one_backup_value,
        "exact_current_query_value": exact_q0,
        "oracle_identity_error": one_backup_value - exact_v1,
        "one_backup_value_error": reported_v1 - one_backup_value,
        "first_successor_value_error": reported_v1 - exact_v1,
        "current_query_value_error": reported_q0 - exact_q0,
        "reported_current_closure_error": reported_q0 - recomposed_q0,
        "current_hit_probability_error": (
            float(action["current_hit_probability"]) - current.next_hit_probability
        ),
        "first_successor_existence_error": (
            float(action["predicted_target_exists_after_miss"])
            - successor.posterior_target_exists
        ),
        "first_successor_hit_error": (
            float(action["predicted_best_hit_probability_after_miss"])
            - successor.next_hit_probability
        ),
        "first_successor_candidate_correct": (
            action["predicted_best_candidate_after_miss"] == next_query
        ),
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
    condition_id: str,
    repeat_index: int,
    out_path: Path,
) -> dict[str, Any]:
    if repeat_index < 0:
        raise ValueError("repeat_index must be non-negative")
    if condition_id not in EXP08_CONDITION_IDS:
        raise ValueError(f"Unsupported Exp08 condition: {condition_id}")

    client = Exp07Client.from_env()
    model = client.resolve_model()
    condition = condition_by_id(condition_id)
    reasoning_mode = os.getenv("EXP07_REASONING_MODE", "preserve").strip().lower()

    try:
        record = probe_depth2(client, condition)
        record["repeat"] = repeat_index
        record["replication_model"] = model
        record["reasoning_mode"] = reasoning_mode
    except Exception as exc:
        record = {
            "condition_id": condition.condition_id,
            "query_cost": condition.query_cost,
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
        + f" model={model} condition={condition_id} repeat={repeat_index}"
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition-id", required=True)
    parser.add_argument("--repeat-index", required=True, type=int)
    parser.add_argument("--out", type=Path, default=Path("results/exp08_matrix/probe.json"))
    parser.add_argument("--require-valid", action="store_true")
    args = parser.parse_args()

    record = run_single_probe(
        condition_id=args.condition_id,
        repeat_index=args.repeat_index,
        out_path=args.out,
    )
    if args.require_valid and "error" in record:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
