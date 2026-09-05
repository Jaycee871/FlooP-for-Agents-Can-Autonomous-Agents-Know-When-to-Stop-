from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from floop_agents.providers import NvidiaClient

from exp05_bellman_decomposition import WEDGE_CONDITIONS
from exp05b_transition_audit import probe


REPLICATION_CONDITION_IDS = {
    "cost_0100_wedge",
    "cost_0120_wedge",
    "cost_0125_wedge",
    "cost_0130_dynamic_stop",
}


class NeutralReasoningClient(NvidiaClient):
    """Cross-model wrapper that avoids provider-specific reasoning controls.

    Exp 05b calls `chat(..., reasoning_effort="low")`. Some NVIDIA-hosted models
    do not expose the same reasoning-control surface. For the cross-model replication,
    ignore that optional control and hold the task prompt itself constant.

    EXP06_MAX_TOKENS can raise the generation budget for models that spend a large
    fraction of their completion budget on hidden/provider-side reasoning before
    emitting the required JSON answer.
    """

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        reasoning_effort: str | None = None,
    ):
        configured = os.getenv("EXP06_MAX_TOKENS")
        effective_max_tokens = max_tokens
        if configured:
            effective_max_tokens = max(max_tokens, int(configured))
        return super().chat(
            messages,
            temperature=temperature,
            max_tokens=effective_max_tokens,
            reasoning_effort=None,
        )


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _stdev(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) >= 2 else None


def describe(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean": _mean(values),
        "median": _median(values),
        "stdev": _stdev(values),
        "mean_abs": _mean([abs(v) for v in values]),
    }


def sign_matches_oracle(record: dict[str, Any]) -> bool:
    reported = float(record["full_expected_net_gain"])
    oracle = str(record["oracle_decision"])
    return (reported > 0 and oracle == "QUERY") or (reported <= 0 and oracle == "STOP")


def coupling_direction(record: dict[str, Any]) -> str:
    value = float(record["full_expected_net_gain"])
    action = str(record["decision"])
    if value > 0 and action == "STOP":
        return "positive_value_to_stop"
    if value <= 0 and action == "QUERY":
        return "nonpositive_value_to_query"
    return "consistent"


def scientific_validity(
    records: list[dict[str, Any]],
    *,
    repeats: int,
    condition_ids: list[str],
) -> dict[str, Any]:
    """Separate workflow completion from scientific completeness.

    A replication is scientifically complete only when every expected
    condition×repeat probe exists and none is an error record. This prevents a green
    workflow, partial artifact, or timeout-truncated run from being mistaken for a
    valid experiment.
    """

    expected_records = repeats * len(condition_ids)
    valid_records = [r for r in records if "error" not in r]
    error_records = [r for r in records if "error" in r]
    counts = Counter(str(r.get("condition_id")) for r in valid_records)
    expected_counts = {condition_id: repeats for condition_id in condition_ids}
    observed_counts = {condition_id: counts.get(condition_id, 0) for condition_id in condition_ids}
    per_condition_complete = all(
        observed_counts[condition_id] == expected_counts[condition_id]
        for condition_id in condition_ids
    )
    complete = (
        len(records) == expected_records
        and len(valid_records) == expected_records
        and not error_records
        and per_condition_complete
    )
    return {
        "expected_records": expected_records,
        "observed_records": len(records),
        "valid_records": len(valid_records),
        "error_records": len(error_records),
        "expected_counts_by_condition": expected_counts,
        "valid_counts_by_condition": observed_counts,
        "per_condition_complete": per_condition_complete,
        "complete": complete,
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in records if "error" not in r]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in valid:
        grouped[str(record["condition_id"])].append(record)

    def metric(name: str, rr: list[dict[str, Any]]) -> dict[str, Any]:
        return describe([float(r[name]) for r in rr])

    by_condition: dict[str, Any] = {}
    for condition_id, rr in sorted(grouped.items()):
        directions = [coupling_direction(r) for r in rr]
        by_condition[condition_id] = {
            "n": len(rr),
            "decision_agreement_count": sum(bool(r["decision_agreement"]) for r in rr),
            "decision_agreement_rate": sum(bool(r["decision_agreement"]) for r in rr) / len(rr),
            "reported_value_sign_accuracy": sum(sign_matches_oracle(r) for r in rr) / len(rr),
            "decision_value_self_consistency": sum(
                bool(r["decision_matches_reported_value_sign"]) for r in rr
            ) / len(rr),
            "successor_candidate_accuracy": sum(
                bool(r["successor_best_candidate_correct"]) for r in rr
            ) / len(rr),
            "coupling_directions": {
                "consistent": directions.count("consistent"),
                "positive_value_to_stop": directions.count("positive_value_to_stop"),
                "nonpositive_value_to_query": directions.count("nonpositive_value_to_query"),
            },
            "current_hit_error": metric("current_hit_probability_error", rr),
            "successor_existence_error": metric("target_exists_after_miss_error", rr),
            "successor_hit_error": metric("best_hit_after_miss_error", rr),
            "future_value_error": metric("future_value_after_miss_error", rr),
            "full_q_error": metric("full_evaluated_query_value_error", rr),
            "closure_error": metric("reported_decomposition_closure_error", rr),
            "local_decision_regret": describe([
                float(r["local_decision_regret"])
                for r in rr
                if r.get("local_decision_regret") is not None
            ]),
        }

    all_directions = [coupling_direction(r) for r in valid]
    overall = {
        "n": len(valid),
        "n_errors": len(records) - len(valid),
        "decision_agreement_rate": (
            sum(bool(r["decision_agreement"]) for r in valid) / len(valid) if valid else None
        ),
        "reported_value_sign_accuracy": (
            sum(sign_matches_oracle(r) for r in valid) / len(valid) if valid else None
        ),
        "decision_value_self_consistency": (
            sum(bool(r["decision_matches_reported_value_sign"]) for r in valid) / len(valid)
            if valid else None
        ),
        "successor_candidate_accuracy": (
            sum(bool(r["successor_best_candidate_correct"]) for r in valid) / len(valid)
            if valid else None
        ),
        "coupling_directions": {
            "consistent": all_directions.count("consistent"),
            "positive_value_to_stop": all_directions.count("positive_value_to_stop"),
            "nonpositive_value_to_query": all_directions.count("nonpositive_value_to_query"),
        },
        "current_hit_error": metric("current_hit_probability_error", valid) if valid else describe([]),
        "successor_existence_error": metric("target_exists_after_miss_error", valid) if valid else describe([]),
        "successor_hit_error": metric("best_hit_after_miss_error", valid) if valid else describe([]),
        "future_value_error": metric("future_value_after_miss_error", valid) if valid else describe([]),
        "full_q_error": metric("full_evaluated_query_value_error", valid) if valid else describe([]),
        "closure_error": metric("reported_decomposition_closure_error", valid) if valid else describe([]),
    }

    return {
        "overall": overall,
        "by_condition": by_condition,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/exp06_repeated_transition_audit"),
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Exit non-zero unless every expected probe is present and error-free.",
    )
    args = parser.parse_args()
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")

    client = NeutralReasoningClient.from_env()
    model = client.resolve_model()
    print(f"Selected NVIDIA model: {model}")
    print("Cross-model replication mode: provider-specific reasoning_effort disabled")
    if os.getenv("EXP06_MAX_TOKENS"):
        print(f"Exp06 minimum generation budget: {os.environ['EXP06_MAX_TOKENS']} tokens")

    safe_model_name = model.replace("/", "__").replace(":", "_")
    model_dir = args.out_dir / safe_model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    probes_path = model_dir / "probes.jsonl"

    conditions = [
        condition
        for condition in WEDGE_CONDITIONS
        if condition.condition_id in REPLICATION_CONDITION_IDS
    ]

    records: list[dict[str, Any]] = []
    with probes_path.open("w", encoding="utf-8", buffering=1) as handle:
        for repeat in range(args.repeats):
            for condition in conditions:
                try:
                    record = probe(client, condition)
                    record["repeat"] = repeat
                    record["replication_model"] = model
                    record["reasoning_effort_disabled"] = True
                except Exception as exc:
                    record = {
                        "condition_id": condition.condition_id,
                        "query_cost": condition.query_cost,
                        "repeat": repeat,
                        "replication_model": model,
                        "reasoning_effort_disabled": True,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                records.append(record)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(
                    f"repeat={repeat} {condition.condition_id} "
                    f"oracle={record.get('oracle_decision')} agent={record.get('decision')} "
                    f"future_error={record.get('future_value_after_miss_error')} "
                    f"coupling={coupling_direction(record) if 'error' not in record else 'error'} "
                    f"error={record.get('error')}"
                )

    condition_ids = [c.condition_id for c in conditions]
    result = summarize(records)
    result["model"] = model
    result["repeats"] = args.repeats
    result["conditions"] = condition_ids
    result["reasoning_effort_disabled"] = True
    result["scientific_validity"] = scientific_validity(
        records,
        repeats=args.repeats,
        condition_ids=condition_ids,
    )
    (model_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.require_complete and not result["scientific_validity"]["complete"]:
        print("SCIENTIFIC_VALIDITY_GATE: FAIL")
        raise SystemExit(2)
    print("SCIENTIFIC_VALIDITY_GATE: PASS")


if __name__ == "__main__":
    main()
