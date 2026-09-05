from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from floop_agents.providers import NvidiaClient
from floop_agents.search_arena import SearchSpec, oracle_state, query_action_value
from floop_agents.stopping_metrics import local_stopping_metrics

from exp03_boundary_sweep import SYSTEM_PROMPT, parse_action, sign_decision
from exp04_calibrated_state_ablation import (
    calibrated_prompt,
    highest_mass_remaining_candidate,
)


@dataclass(frozen=True)
class WedgeCondition:
    condition_id: str
    query_cost: float


# A deliberately flat prior creates states in which every single next query looks
# unprofitable in isolation while a sequence of queries still has positive value.
# This makes the future option value decision-relevant rather than merely additive.
WEDGE_CONDITIONS = (
    WedgeCondition("cost_0080_myopic_positive", 0.080),
    WedgeCondition("cost_0090_wedge", 0.090),
    WedgeCondition("cost_0100_wedge", 0.100),
    WedgeCondition("cost_0120_wedge", 0.120),
    WedgeCondition("cost_0125_wedge", 0.125),
    WedgeCondition("cost_0130_dynamic_stop", 0.130),
    WedgeCondition("cost_0140_stop_control", 0.140),
)


def make_wedge_spec(condition: WedgeCondition) -> SearchSpec:
    return SearchSpec(
        n_candidates=10,
        existence_prob=0.8,
        alpha=0.05,
        query_cost=condition.query_cost,
        reward=1.0,
    )


BELLMAN_SYSTEM_PROMPT = """You are an autonomous search agent in a stopping experiment.
Your goal is to maximize expected NET utility, not to search until certainty.
At each state you may STOP or QUERY exactly one unqueried candidate.

Before your final action, explicitly decompose the best continuation you can identify.
A miss is NOT terminal: after a miss, the future policy may query again or stop.
Do not use only the immediate one-step gain. Estimate the option value preserved after a miss.

Return ONLY one JSON object with exactly these keys:
{
  "decision": "STOP or QUERY",
  "query": 1,
  "evaluated_query": 1,
  "confidence": 0.0,
  "immediate_hit_value": 0.0,
  "query_cost": 0.0,
  "miss_probability": 0.0,
  "estimated_value_after_miss": 0.0,
  "full_expected_net_gain": 0.0,
  "reason": "brief decision rationale"
}

'evaluated_query' is the unqueried candidate whose continuation value you are
pricing. If decision is QUERY, query must equal evaluated_query. If decision is
STOP, set query to null but still report evaluated_query and the decomposition
that justified stopping. confidence, probabilities, and nonnegative component
values must be numerically valid. full_expected_net_gain may be negative.
Do not wrap the JSON in markdown.
"""


def _number(value: Any, default: float = 0.0) -> float:
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


def parse_bellman_action(text: str) -> dict[str, Any]:
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

    query = _integer_or_none(obj.get("query")) if decision == "QUERY" else None
    evaluated_query = _integer_or_none(obj.get("evaluated_query"))

    return {
        "decision": decision,
        "query": query,
        "evaluated_query": evaluated_query,
        "confidence": max(0.0, min(1.0, _number(obj.get("confidence"), 0.0))),
        "immediate_hit_value": max(0.0, min(1.0, _number(obj.get("immediate_hit_value"), 0.0))),
        "reported_query_cost": max(0.0, min(1.0, _number(obj.get("query_cost"), 0.0))),
        "miss_probability": max(0.0, min(1.0, _number(obj.get("miss_probability"), 0.0))),
        "estimated_value_after_miss": max(0.0, min(1.0, _number(obj.get("estimated_value_after_miss"), 0.0))),
        "full_expected_net_gain": max(-1.0, min(1.0, _number(obj.get("full_expected_net_gain"), 0.0))),
        "reason": str(obj.get("reason", "")).strip()[:500],
    }


def posterior_hit_probability(spec: SearchSpec, queried: list[int], query: int) -> float:
    queried_set = set(queried)
    masses = spec.target_masses
    if query in queried_set or query not in masses:
        raise ValueError("query is unavailable in the current state")
    remaining_mass = sum(m for i, m in masses.items() if i not in queried_set)
    denominator = spec.no_target_mass + remaining_mass
    if denominator <= 0:
        return 0.0
    return masses[query] / denominator


def bellman_prompt(spec: SearchSpec, queried: list[int]) -> str:
    return calibrated_prompt(spec, queried) + """
BELLMAN DECOMPOSITION
Price the best continuation candidate you can identify using:
full continuation value = - query cost
                          + immediate hit value
                          + miss probability * estimated optimal value after a miss.

Here 'immediate hit value' means P(hit on evaluated_query) times the target reward.
'estimated optimal value after a miss' means the value of the best future policy
from that miss-updated state, where the future policy may either query again or stop.
Report the components and then choose STOP or QUERY.
"""


def _valid_query(spec: SearchSpec, queried: list[int], query: int | None) -> bool:
    return (
        query is not None
        and 1 <= query <= spec.n_candidates
        and query not in set(queried)
    )


def state_diagnostics(spec: SearchSpec, queried: list[int]) -> dict[str, Any]:
    oracle = oracle_state(spec, queried)
    myopic_gain = oracle.next_hit_probability * spec.reward - spec.query_cost
    return {
        "oracle_decision": oracle.decision,
        "oracle_advantage": oracle.advantage,
        "oracle_posterior_target_exists": oracle.posterior_target_exists,
        "oracle_next_hit_probability": oracle.next_hit_probability,
        "myopic_one_step_gain": myopic_gain,
        "myopic_decision": sign_decision(myopic_gain),
        "is_option_value_wedge": myopic_gain <= 0.0 and oracle.advantage > 0.0,
        "oracle_myopic_disagree": sign_decision(myopic_gain) != oracle.decision,
    }


def _stopping_metrics_for_action(
    *,
    spec: SearchSpec,
    queried: list[int],
    decision: str,
    query: int | None,
    reported_value: float,
    oracle_advantage: float,
):
    chosen_query_value: float | None = None
    if decision == "QUERY" and _valid_query(spec, queried, query):
        chosen_query_value = query_action_value(spec, queried, int(query))
    if decision == "STOP" or chosen_query_value is not None:
        return local_stopping_metrics(
            agent_decision=decision,
            oracle_continuation_value=oracle_advantage,
            agent_reported_continuation_value=reported_value,
            chosen_query_value=chosen_query_value,
        )
    return None


def probe_calibrated(client: NvidiaClient, condition: WedgeCondition) -> dict[str, Any]:
    spec = make_wedge_spec(condition)
    queried: list[int] = []
    diagnostics = state_diagnostics(spec, queried)
    highest_candidate = highest_mass_remaining_candidate(spec, queried)

    response = client.chat(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": calibrated_prompt(spec, queried)},
        ],
        temperature=0.1,
        max_tokens=2048,
        reasoning_effort="low",
    )
    action = parse_action(response.content)
    valid_query = action["decision"] != "QUERY" or _valid_query(spec, queried, action["query"])
    stopping = _stopping_metrics_for_action(
        spec=spec,
        queried=queried,
        decision=action["decision"],
        query=action["query"],
        reported_value=float(action["expected_net_gain"]),
        oracle_advantage=float(diagnostics["oracle_advantage"]),
    )

    return {
        "condition_id": condition.condition_id,
        "query_cost": condition.query_cost,
        "arm": "calibrated",
        "queried": queried,
        "model": response.model,
        **action,
        "highest_mass_remaining_candidate": highest_candidate,
        "action_valid": valid_query,
        **diagnostics,
        "decision_agreement": action["decision"] == diagnostics["oracle_decision"],
        "reported_value_sign_matches_oracle": sign_decision(float(action["expected_net_gain"])) == diagnostics["oracle_decision"],
        "decision_matches_reported_value_sign": action["decision"] == sign_decision(float(action["expected_net_gain"])),
        "premature_stop_regret": stopping.premature_stop_regret if stopping else None,
        "excess_continuation_regret": stopping.excess_continuation_regret if stopping else None,
        "query_selection_regret": stopping.query_selection_regret if stopping else None,
        "local_decision_regret": stopping.local_decision_regret if stopping else None,
        "reasoning_chars": len(response.reasoning) if response.reasoning else 0,
        "prompt_tokens": response.prompt_tokens,
        "completion_tokens": response.completion_tokens,
        "total_tokens": response.total_tokens,
    }


def probe_bellman(client: NvidiaClient, condition: WedgeCondition) -> dict[str, Any]:
    spec = make_wedge_spec(condition)
    queried: list[int] = []
    diagnostics = state_diagnostics(spec, queried)
    highest_candidate = highest_mass_remaining_candidate(spec, queried)

    response = client.chat(
        [
            {"role": "system", "content": BELLMAN_SYSTEM_PROMPT},
            {"role": "user", "content": bellman_prompt(spec, queried)},
        ],
        temperature=0.1,
        max_tokens=2048,
        reasoning_effort="low",
    )
    action = parse_bellman_action(response.content)

    evaluated_valid = _valid_query(spec, queried, action["evaluated_query"])
    query_valid = action["decision"] != "QUERY" or _valid_query(spec, queried, action["query"])
    query_matches_evaluated = action["decision"] != "QUERY" or action["query"] == action["evaluated_query"]
    action_valid = evaluated_valid and query_valid and query_matches_evaluated

    exact_hit: float | None = None
    exact_miss: float | None = None
    exact_after_miss: float | None = None
    exact_evaluated_q_value: float | None = None
    exact_immediate_hit_value: float | None = None

    if evaluated_valid:
        q_eval = int(action["evaluated_query"])
        exact_hit = posterior_hit_probability(spec, queried, q_eval)
        exact_miss = 1.0 - exact_hit
        exact_after_miss = max(0.0, oracle_state(spec, [*queried, q_eval]).continuation_value)
        exact_evaluated_q_value = query_action_value(spec, queried, q_eval)
        exact_immediate_hit_value = exact_hit * spec.reward

    stopping = _stopping_metrics_for_action(
        spec=spec,
        queried=queried,
        decision=action["decision"],
        query=action["query"],
        reported_value=float(action["full_expected_net_gain"]),
        oracle_advantage=float(diagnostics["oracle_advantage"]),
    )

    recomposed_reported_gain = (
        -float(action["reported_query_cost"])
        + float(action["immediate_hit_value"])
        + float(action["miss_probability"]) * float(action["estimated_value_after_miss"])
    )

    return {
        "condition_id": condition.condition_id,
        "query_cost": condition.query_cost,
        "arm": "bellman",
        "queried": queried,
        "model": response.model,
        **action,
        "highest_mass_remaining_candidate": highest_candidate,
        "evaluated_query_valid": evaluated_valid,
        "action_valid": action_valid,
        **diagnostics,
        "exact_hit_probability_evaluated_query": exact_hit,
        "exact_miss_probability_evaluated_query": exact_miss,
        "exact_immediate_hit_value": exact_immediate_hit_value,
        "exact_value_after_miss": exact_after_miss,
        "exact_evaluated_query_value": exact_evaluated_q_value,
        "immediate_term_error": float(action["immediate_hit_value"]) - exact_immediate_hit_value if exact_immediate_hit_value is not None else None,
        "query_cost_error": float(action["reported_query_cost"]) - spec.query_cost,
        "miss_probability_error": float(action["miss_probability"]) - exact_miss if exact_miss is not None else None,
        "future_option_value_omission_error": float(action["estimated_value_after_miss"]) - exact_after_miss if exact_after_miss is not None else None,
        "full_evaluated_query_value_error": float(action["full_expected_net_gain"]) - exact_evaluated_q_value if exact_evaluated_q_value is not None else None,
        "reported_decomposition_closure_error": float(action["full_expected_net_gain"]) - recomposed_reported_gain,
        "decision_agreement": action["decision"] == diagnostics["oracle_decision"],
        "reported_value_sign_matches_oracle": sign_decision(float(action["full_expected_net_gain"])) == diagnostics["oracle_decision"],
        "decision_matches_reported_value_sign": action["decision"] == sign_decision(float(action["full_expected_net_gain"])),
        "premature_stop_regret": stopping.premature_stop_regret if stopping else None,
        "excess_continuation_regret": stopping.excess_continuation_regret if stopping else None,
        "query_selection_regret": stopping.query_selection_regret if stopping else None,
        "local_decision_regret": stopping.local_decision_regret if stopping else None,
        "reasoning_chars": len(response.reasoning) if response.reasoning else 0,
        "prompt_tokens": response.prompt_tokens,
        "completion_tokens": response.completion_tokens,
        "total_tokens": response.total_tokens,
    }


def safe_probe(fn: Callable[[], dict[str, Any]], *, condition_id: str, arm: str) -> dict[str, Any]:
    try:
        return fn()
    except Exception as exc:
        return {"condition_id": condition_id, "arm": arm, "error": f"{type(exc).__name__}: {exc}"}


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid_records = [r for r in records if "error" not in r]
    arms: dict[str, dict[str, Any]] = {}
    for arm in ("calibrated", "bellman"):
        rr = [r for r in valid_records if r["arm"] == arm]
        wedge_rr = [r for r in rr if bool(r["is_option_value_wedge"])]
        arms[arm] = {
            "n": len(rr),
            "n_errors": sum(r["arm"] == arm and "error" in r for r in records),
            "decision_agreement_rate": sum(bool(r["decision_agreement"]) for r in rr) / len(rr) if rr else None,
            "wedge_n": len(wedge_rr),
            "wedge_decision_agreement_rate": sum(bool(r["decision_agreement"]) for r in wedge_rr) / len(wedge_rr) if wedge_rr else None,
            "reported_value_sign_accuracy": sum(bool(r["reported_value_sign_matches_oracle"]) for r in rr) / len(rr) if rr else None,
            "decision_value_self_consistency": sum(bool(r["decision_matches_reported_value_sign"]) for r in rr) / len(rr) if rr else None,
            "invalid_action_rate": sum(not bool(r.get("action_valid", False)) for r in rr) / len(rr) if rr else None,
            "mean_local_decision_regret": _mean([float(r["local_decision_regret"]) for r in rr if r.get("local_decision_regret") is not None]),
        }

    bellman = [r for r in valid_records if r["arm"] == "bellman"]
    bellman_wedge = [r for r in bellman if bool(r["is_option_value_wedge"])]
    arms["bellman"].update({
        "mean_future_option_value_omission_error": _mean([float(r["future_option_value_omission_error"]) for r in bellman if r.get("future_option_value_omission_error") is not None]),
        "mean_wedge_future_option_value_omission_error": _mean([float(r["future_option_value_omission_error"]) for r in bellman_wedge if r.get("future_option_value_omission_error") is not None]),
        "mean_abs_future_option_value_error": _mean([abs(float(r["future_option_value_omission_error"])) for r in bellman if r.get("future_option_value_omission_error") is not None]),
        "mean_abs_full_evaluated_query_value_error": _mean([abs(float(r["full_evaluated_query_value_error"])) for r in bellman if r.get("full_evaluated_query_value_error") is not None]),
        "mean_abs_decomposition_closure_error": _mean([abs(float(r["reported_decomposition_closure_error"])) for r in bellman if r.get("reported_decomposition_closure_error") is not None]),
    })

    paired = []
    for condition in WEDGE_CONDITIONS:
        cal = next((r for r in valid_records if r["condition_id"] == condition.condition_id and r["arm"] == "calibrated"), None)
        bell = next((r for r in valid_records if r["condition_id"] == condition.condition_id and r["arm"] == "bellman"), None)
        if cal is None or bell is None:
            continue
        paired.append({
            "condition_id": condition.condition_id,
            "query_cost": condition.query_cost,
            "oracle_decision": cal["oracle_decision"],
            "myopic_decision": cal["myopic_decision"],
            "myopic_one_step_gain": cal["myopic_one_step_gain"],
            "oracle_advantage": cal["oracle_advantage"],
            "is_option_value_wedge": cal["is_option_value_wedge"],
            "calibrated_decision": cal["decision"],
            "bellman_decision": bell["decision"],
            "calibrated_correct": cal["decision_agreement"],
            "bellman_correct": bell["decision_agreement"],
            "calibrated_regret": cal.get("local_decision_regret"),
            "bellman_regret": bell.get("local_decision_regret"),
            "bellman_option_error": bell.get("future_option_value_omission_error"),
        })

    wedge_pairs = [p for p in paired if bool(p["is_option_value_wedge"])]
    corrected = sum((not p["calibrated_correct"]) and p["bellman_correct"] for p in wedge_pairs)
    harmed = sum(p["calibrated_correct"] and (not p["bellman_correct"]) for p in wedge_pairs)
    return {
        "world": {"n_candidates": 10, "existence_prob": 0.8, "alpha": 0.05, "reward": 1.0},
        "conditions": [asdict(c) for c in WEDGE_CONDITIONS],
        "arms": arms,
        "paired_n": len(paired),
        "wedge_paired_n": len(wedge_pairs),
        "wedge_calibrated_wrong_bellman_correct": corrected,
        "wedge_calibrated_correct_bellman_wrong": harmed,
        "wedge_net_corrected_pairs": corrected - harmed,
        "pairs": paired,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("results/exp05_bellman_decomposition"))
    args = parser.parse_args()

    client = NvidiaClient.from_env()
    print(f"Selected NVIDIA model: {client.resolve_model()}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    probes_path = args.out_dir / "probes.jsonl"
    records: list[dict[str, Any]] = []

    with probes_path.open("w", encoding="utf-8", buffering=1) as f:
        for condition in WEDGE_CONDITIONS:
            spec = make_wedge_spec(condition)
            diagnostics = state_diagnostics(spec, [])
            print(
                f"{condition.condition_id}: cost={condition.query_cost:.3f} "
                f"myopic={diagnostics['myopic_one_step_gain']:.6f} "
                f"oracle={diagnostics['oracle_advantage']:.6f} "
                f"wedge={diagnostics['is_option_value_wedge']}"
            )

            calibrated = safe_probe(lambda c=condition: probe_calibrated(client, c), condition_id=condition.condition_id, arm="calibrated")
            records.append(calibrated)
            f.write(json.dumps(calibrated, ensure_ascii=False) + "\n")
            print(f"  calibrated decision={calibrated.get('decision')} error={calibrated.get('error')}")

            bellman = safe_probe(lambda c=condition: probe_bellman(client, c), condition_id=condition.condition_id, arm="bellman")
            records.append(bellman)
            f.write(json.dumps(bellman, ensure_ascii=False) + "\n")
            print(
                f"  bellman decision={bellman.get('decision')} "
                f"option_error={bellman.get('future_option_value_omission_error')} "
                f"error={bellman.get('error')}"
            )

    result = summarize(records)
    result["model"] = next((r.get("model") for r in records if r.get("model")), client.model)
    (args.out_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "pairs"}, indent=2))


if __name__ == "__main__":
    main()
