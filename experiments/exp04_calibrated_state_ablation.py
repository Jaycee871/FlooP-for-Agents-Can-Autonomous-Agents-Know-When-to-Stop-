from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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


def highest_mass_remaining_candidate(spec, queried: list[int]) -> int | None:
    """Return the highest-prior unqueried candidate without consulting STOP/CONTINUE."""

    queried_set = set(queried)
    remaining = [(i, mass) for i, mass in spec.target_masses.items() if i not in queried_set]
    if not remaining:
        return None
    return max(remaining, key=lambda item: (item[1], -item[0]))[0]


def calibrated_prompt(spec, queried: list[int]) -> str:
    oracle = oracle_state(spec, queried)
    highest_candidate = highest_mass_remaining_candidate(spec, queried)
    base = prompt_for_state(spec, queried)
    return (
        base
        + "\nCALIBRATED STATE STATISTICS\n"
        + "The following posterior quantities are exact and may be trusted; you do not need to recompute them.\n"
        + f"Posterior probability that a target still exists: {oracle.posterior_target_exists:.9f}\n"
        + f"Highest one-query hit probability among remaining candidates: {oracle.next_hit_probability:.9f}\n"
        + f"Candidate attaining that highest one-query hit probability: {highest_candidate}\n"
        + "These statistics do NOT reveal whether continuing is optimal. They were computed independently of the STOP/CONTINUE decision. "
        + "Remember that after another miss you may still choose to query again or stop. Decide the expected NET value of the full continuation policy, not only the immediate next query.\n"
    )


def probe(client: NvidiaClient, condition, k: int, arm: str) -> dict[str, Any]:
    spec = make_spec(condition)
    boundary, oracle_path = oracle_boundary(spec)
    queried = oracle_path[:k] if k <= len(oracle_path) else list(range(1, k + 1))
    oracle = oracle_state(spec, queried)
    highest_candidate = highest_mass_remaining_candidate(spec, queried)

    user_prompt = prompt_for_state(spec, queried) if arm == "raw" else calibrated_prompt(spec, queried)
    response = client.chat(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
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
        "valid_query": valid_query if action["decision"] == "QUERY" else None,
        "highest_mass_remaining_candidate": highest_candidate,
        "oracle_decision": oracle.decision,
        "oracle_best_query": oracle.best_query,
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


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    arms: dict[str, dict[str, Any]] = {}
    for arm in ("raw", "calibrated"):
        rr = [r for r in records if r["arm"] == arm]
        pre = [r for r in rr if int(r["failed_queries"]) < int(r["oracle_boundary"])]
        post = [r for r in rr if int(r["failed_queries"]) >= int(r["oracle_boundary"])]
        arms[arm] = {
            "n": len(rr),
            "decision_agreement_rate": sum(bool(r["decision_agreement"]) for r in rr) / len(rr),
            "reported_value_sign_accuracy": sum(bool(r["reported_value_sign_matches_oracle"]) for r in rr) / len(rr),
            "decision_value_self_consistency": sum(bool(r["decision_matches_reported_value_sign"]) for r in rr) / len(rr),
            "mean_expected_gain_abs_error": sum(float(r["expected_gain_abs_error"]) for r in rr) / len(rr),
            "premature_stop_rate_pre_boundary": (
                sum(r["decision"] == "STOP" for r in pre) / len(pre) if pre else None
            ),
            "excess_query_rate_at_or_after_boundary": (
                sum(r["decision"] == "QUERY" for r in post) / len(post) if post else None
            ),
            "invalid_query_rate": (
                sum(r["decision"] == "QUERY" and not bool(r["valid_query"]) for r in rr)
                / max(1, sum(r["decision"] == "QUERY" for r in rr))
            ),
        }

    paired = []
    keys = sorted({(r["condition_id"], r["failed_queries"]) for r in records})
    for condition_id, k in keys:
        raw = next(r for r in records if r["condition_id"] == condition_id and r["failed_queries"] == k and r["arm"] == "raw")
        cal = next(r for r in records if r["condition_id"] == condition_id and r["failed_queries"] == k and r["arm"] == "calibrated")
        paired.append(
            {
                "condition_id": condition_id,
                "failed_queries": k,
                "oracle_decision": raw["oracle_decision"],
                "raw_decision": raw["decision"],
                "calibrated_decision": cal["decision"],
                "raw_correct": raw["decision_agreement"],
                "calibrated_correct": cal["decision_agreement"],
                "raw_abs_error": raw["expected_gain_abs_error"],
                "calibrated_abs_error": cal["expected_gain_abs_error"],
            }
        )

    corrected = sum((not p["raw_correct"]) and p["calibrated_correct"] for p in paired)
    harmed = sum(p["raw_correct"] and (not p["calibrated_correct"]) for p in paired)
    return {
        "arms": arms,
        "paired_n": len(paired),
        "raw_wrong_calibrated_correct": corrected,
        "raw_correct_calibrated_wrong": harmed,
        "net_corrected_pairs": corrected - harmed,
        "pairs": paired,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("results/exp04_calibrated_state_ablation"))
    args = parser.parse_args()

    client = NvidiaClient.from_env()
    print(f"Selected NVIDIA model: {client.resolve_model()}")

    records: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        spec = make_spec(condition)
        boundary, _ = oracle_boundary(spec)
        ks = sorted({max(0, boundary - 2), max(0, boundary - 1), boundary, min(spec.n_candidates, boundary + 1)})
        for k in ks:
            for arm in ("raw", "calibrated"):
                record = probe(client, condition, k, arm)
                records.append(record)
                print(
                    f"{condition.condition_id} k={k} arm={arm} "
                    f"oracle={record['oracle_decision']} agent={record['decision']} "
                    f"gain={record['expected_net_gain']} true={record['oracle_advantage']:.5f}"
                )

    result = summarize(records)
    result["model"] = records[0]["model"] if records else client.model

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "probes.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    (args.out_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in result.items() if k != "pairs"}, indent=2))


if __name__ == "__main__":
    main()
