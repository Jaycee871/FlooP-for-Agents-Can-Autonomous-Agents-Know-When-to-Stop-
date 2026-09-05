from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

from floop_agents.providers import NvidiaClient
from floop_agents.search_arena import oracle_state

from exp03_boundary_sweep import (
    CONDITIONS,
    SYSTEM_PROMPT,
    make_spec,
    oracle_boundary,
    parse_action,
    prompt_for_state,
    sign_decision,
)
from exp04_calibrated_state_ablation import highest_mass_remaining_candidate


DYNAMIC_REMINDER = (
    "\nDECISION REMINDER\n"
    "Remember that after another miss you may still choose to query again or stop. "
    "Decide the expected NET value of the full continuation policy, not only the immediate next query.\n"
)


def instruction_only_prompt(spec, queried: list[int]) -> str:
    return prompt_for_state(spec, queried) + DYNAMIC_REMINDER


def state_calibrated_prompt(spec, queried: list[int]) -> str:
    oracle = oracle_state(spec, queried)
    highest_candidate = highest_mass_remaining_candidate(spec, queried)
    return (
        prompt_for_state(spec, queried)
        + "\nCALIBRATED STATE STATISTICS\n"
        + "The following posterior quantities are exact and may be trusted; you do not need to recompute them.\n"
        + f"Posterior probability that a target still exists: {oracle.posterior_target_exists:.9f}\n"
        + f"Highest one-query hit probability among remaining candidates: {oracle.next_hit_probability:.9f}\n"
        + f"Candidate attaining that highest one-query hit probability: {highest_candidate}\n"
        + "These statistics do NOT reveal whether continuing is optimal. They were computed independently of the STOP/CONTINUE decision.\n"
        + DYNAMIC_REMINDER
    )


def probe(client: NvidiaClient, condition, k: int, arm: str) -> dict[str, Any]:
    spec = make_spec(condition)
    boundary, oracle_path = oracle_boundary(spec)
    queried = oracle_path[:k] if k <= len(oracle_path) else list(range(1, k + 1))
    oracle = oracle_state(spec, queried)
    highest_candidate = highest_mass_remaining_candidate(spec, queried)

    if arm == "instruction_only":
        user_prompt = instruction_only_prompt(spec, queried)
    elif arm == "state_calibrated":
        user_prompt = state_calibrated_prompt(spec, queried)
    else:
        raise ValueError(f"unknown arm: {arm}")

    response = client.chat(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=2048,
        reasoning_effort="low",
    )
    action = parse_action(response.content)

    valid_query = (
        action["decision"] == "QUERY"
        and action["query"] is not None
        and 1 <= action["query"] <= spec.n_candidates
        and action["query"] not in queried
    )
    action_valid = action["decision"] == "STOP" or valid_query
    reported_sign = sign_decision(float(action["expected_net_gain"]))

    return {
        "condition_id": condition.condition_id,
        "query_cost": condition.query_cost,
        "oracle_boundary": boundary,
        "failed_queries": k,
        "arm": arm,
        "queried": queried,
        "model": response.model,
        **action,
        "action_valid": action_valid,
        "valid_query": valid_query if action["decision"] == "QUERY" else None,
        "highest_mass_remaining_candidate": highest_candidate,
        "oracle_decision": oracle.decision,
        "oracle_advantage": oracle.advantage,
        "oracle_posterior_target_exists": oracle.posterior_target_exists,
        "oracle_next_hit_probability": oracle.next_hit_probability,
        "decision_agreement": action["decision"] == oracle.decision,
        "reported_value_sign_matches_oracle": reported_sign == oracle.decision,
        "decision_matches_reported_value_sign": action["decision"] == reported_sign,
        "signed_gain_error": float(action["expected_net_gain"]) - oracle.advantage,
        "expected_gain_abs_error": abs(float(action["expected_net_gain"]) - oracle.advantage),
        "reasoning_chars": len(response.reasoning) if response.reasoning else 0,
        "prompt_tokens": response.prompt_tokens,
        "completion_tokens": response.completion_tokens,
        "total_tokens": response.total_tokens,
    }


def safe_probe(
    fn: Callable[[], dict[str, Any]],
    *,
    condition_id: str,
    failed_queries: int,
    arm: str,
) -> dict[str, Any]:
    try:
        return fn()
    except Exception as exc:
        return {
            "condition_id": condition_id,
            "failed_queries": failed_queries,
            "arm": arm,
            "error": f"{type(exc).__name__}: {exc}",
        }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in records if "error" not in r]
    arms: dict[str, dict[str, Any]] = {}
    for arm in ("instruction_only", "state_calibrated"):
        rr = [r for r in valid if r["arm"] == arm]
        pre = [r for r in rr if int(r["failed_queries"]) < int(r["oracle_boundary"])]
        post = [r for r in rr if int(r["failed_queries"]) >= int(r["oracle_boundary"])]
        arms[arm] = {
            "n": len(rr),
            "n_errors": sum(r["arm"] == arm and "error" in r for r in records),
            "decision_agreement_rate": sum(bool(r["decision_agreement"]) for r in rr) / len(rr) if rr else None,
            "reported_value_sign_accuracy": sum(bool(r["reported_value_sign_matches_oracle"]) for r in rr) / len(rr) if rr else None,
            "decision_value_self_consistency": sum(bool(r["decision_matches_reported_value_sign"]) for r in rr) / len(rr) if rr else None,
            "mean_expected_gain_abs_error": sum(float(r["expected_gain_abs_error"]) for r in rr) / len(rr) if rr else None,
            "premature_stop_rate_pre_boundary": sum(r["decision"] == "STOP" for r in pre) / len(pre) if pre else None,
            "excess_query_rate_at_or_after_boundary": sum(r["decision"] == "QUERY" for r in post) / len(post) if post else None,
            "invalid_action_rate": sum(not bool(r["action_valid"]) for r in rr) / len(rr) if rr else None,
        }

    paired = []
    keys = sorted({(r["condition_id"], int(r["failed_queries"])) for r in valid})
    for condition_id, k in keys:
        instruction = next((r for r in valid if r["condition_id"] == condition_id and int(r["failed_queries"]) == k and r["arm"] == "instruction_only"), None)
        calibrated = next((r for r in valid if r["condition_id"] == condition_id and int(r["failed_queries"]) == k and r["arm"] == "state_calibrated"), None)
        if instruction is None or calibrated is None:
            continue
        paired.append({
            "condition_id": condition_id,
            "failed_queries": k,
            "oracle_decision": instruction["oracle_decision"],
            "instruction_only_decision": instruction["decision"],
            "state_calibrated_decision": calibrated["decision"],
            "instruction_only_correct": instruction["decision_agreement"],
            "state_calibrated_correct": calibrated["decision_agreement"],
            "instruction_only_abs_error": instruction["expected_gain_abs_error"],
            "state_calibrated_abs_error": calibrated["expected_gain_abs_error"],
        })

    corrected = sum((not p["instruction_only_correct"]) and p["state_calibrated_correct"] for p in paired)
    harmed = sum(p["instruction_only_correct"] and (not p["state_calibrated_correct"]) for p in paired)
    return {
        "arms": arms,
        "paired_n": len(paired),
        "instruction_wrong_state_correct": corrected,
        "instruction_correct_state_wrong": harmed,
        "net_state_corrected_pairs": corrected - harmed,
        "pairs": paired,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("results/exp04b_instruction_matched_state_ablation"))
    args = parser.parse_args()

    client = NvidiaClient.from_env()
    print(f"Selected NVIDIA model: {client.resolve_model()}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    probes_path = args.out_dir / "probes.jsonl"
    records: list[dict[str, Any]] = []

    with probes_path.open("w", encoding="utf-8", buffering=1) as f:
        for condition in CONDITIONS:
            spec = make_spec(condition)
            boundary, _ = oracle_boundary(spec)
            ks = sorted({max(0, boundary - 2), max(0, boundary - 1), boundary, min(spec.n_candidates, boundary + 1)})
            for k in ks:
                for arm in ("instruction_only", "state_calibrated"):
                    record = safe_probe(
                        lambda c=condition, kk=k, a=arm: probe(client, c, kk, a),
                        condition_id=condition.condition_id,
                        failed_queries=k,
                        arm=arm,
                    )
                    records.append(record)
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    print(
                        f"{condition.condition_id} k={k} arm={arm} "
                        f"oracle={record.get('oracle_decision')} agent={record.get('decision')} "
                        f"error={record.get('error')}"
                    )

    result = summarize(records)
    result["model"] = next((r.get("model") for r in records if r.get("model")), client.model)
    (args.out_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "pairs"}, indent=2))


if __name__ == "__main__":
    main()
