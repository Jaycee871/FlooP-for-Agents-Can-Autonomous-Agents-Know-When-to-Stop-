from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import exp05b_transition_audit as audit
from floop_agents.providers import NvidiaClient

from exp05_bellman_decomposition import WEDGE_CONDITIONS


TARGET_IDS = {"cost_0125_wedge", "cost_0130_dynamic_stop"}

EXPLICIT_STOP_RULE = """

NORMATIVE STOP BASELINE
For this experiment, STOP has incremental continuation value exactly 0.
The field full_expected_net_gain is the incremental value of initiating the evaluated
query relative to STOP. Therefore your final decision MUST obey this mapping:

- if full_expected_net_gain > 0, choose QUERY;
- if full_expected_net_gain <= 0, choose STOP.

Do not compare full_expected_net_gain with any other baseline or with a separate
"current value." The only decision comparison is QUERY's incremental continuation
value versus STOP = 0.
"""


def coupling_direction(record: dict[str, Any]) -> str:
    value = float(record["full_expected_net_gain"])
    action = str(record["decision"])
    if value > 0 and action == "STOP":
        return "positive_value_to_stop"
    if value <= 0 and action == "QUERY":
        return "nonpositive_value_to_query"
    return "consistent"


def run_probe(client: NvidiaClient, condition) -> dict[str, Any]:
    original_prompt = audit.TRANSITION_SYSTEM_PROMPT
    try:
        audit.TRANSITION_SYSTEM_PROMPT = original_prompt + EXPLICIT_STOP_RULE
        record = audit.probe(client, condition)
    finally:
        audit.TRANSITION_SYSTEM_PROMPT = original_prompt
    record["arm"] = "explicit_stop_baseline"
    record["stop_baseline_explicit"] = True
    record["coupling_direction"] = coupling_direction(record)
    return record


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in records if "error" not in r]
    by_condition: dict[str, Any] = {}
    for condition_id in sorted({str(r["condition_id"]) for r in valid}):
        rr = [r for r in valid if r["condition_id"] == condition_id]
        counts = Counter(coupling_direction(r) for r in rr)
        by_condition[condition_id] = {
            "n": len(rr),
            "decision_agreement_rate": (
                sum(bool(r["decision_agreement"]) for r in rr) / len(rr) if rr else None
            ),
            "decision_value_self_consistency": (
                sum(bool(r["decision_matches_reported_value_sign"]) for r in rr) / len(rr)
                if rr else None
            ),
            "coupling_directions": dict(counts),
            "mean_future_value_error": (
                sum(float(r["future_value_after_miss_error"]) for r in rr) / len(rr)
                if rr else None
            ),
            "mean_full_q_error": (
                sum(float(r["full_evaluated_query_value_error"]) for r in rr) / len(rr)
                if rr else None
            ),
        }

    counts = Counter(coupling_direction(r) for r in valid)
    return {
        "n": len(valid),
        "n_errors": len(records) - len(valid),
        "decision_agreement_rate": (
            sum(bool(r["decision_agreement"]) for r in valid) / len(valid) if valid else None
        ),
        "decision_value_self_consistency": (
            sum(bool(r["decision_matches_reported_value_sign"]) for r in valid) / len(valid)
            if valid else None
        ),
        "coupling_directions": dict(counts),
        "by_condition": by_condition,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/exp05c_explicit_stop_baseline"),
    )
    args = parser.parse_args()
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")

    client = NvidiaClient.from_env()
    print(f"Selected NVIDIA model: {client.resolve_model()}")

    conditions = [c for c in WEDGE_CONDITIONS if c.condition_id in TARGET_IDS]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    probes_path = args.out_dir / "probes.jsonl"
    records: list[dict[str, Any]] = []

    with probes_path.open("w", encoding="utf-8", buffering=1) as handle:
        for repeat in range(args.repeats):
            for condition in conditions:
                try:
                    record = run_probe(client, condition)
                    record["repeat"] = repeat
                except Exception as exc:
                    record = {
                        "condition_id": condition.condition_id,
                        "query_cost": condition.query_cost,
                        "repeat": repeat,
                        "arm": "explicit_stop_baseline",
                        "stop_baseline_explicit": True,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                records.append(record)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(
                    f"repeat={repeat} {condition.condition_id} "
                    f"oracle={record.get('oracle_decision')} agent={record.get('decision')} "
                    f"gain={record.get('full_expected_net_gain')} "
                    f"coupling={record.get('coupling_direction')} "
                    f"error={record.get('error')}"
                )

    result = summarize(records)
    result["model"] = next((r.get("model") for r in records if r.get("model")), client.model)
    result["repeats"] = args.repeats
    result["stop_baseline_explicit"] = True
    (args.out_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
