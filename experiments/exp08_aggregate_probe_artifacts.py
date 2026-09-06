from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from exp08_bellman_depth_probe import EXP08_CONDITION_IDS


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


def load_probe_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("probe.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            record = {
                "replication_model": "artifact-parse-error",
                "condition_id": "unknown",
                "repeat": -1,
                "error": f"{type(exc).__name__}: {exc}",
            }
        record["artifact_path"] = str(path)
        records.append(record)
    return records


def duplicate_keys(records: list[dict[str, Any]]) -> list[tuple[str, str, int]]:
    keys = [
        (
            str(r.get("replication_model")),
            str(r.get("condition_id")),
            int(r.get("repeat", -1)),
        )
        for r in records
    ]
    counts = Counter(keys)
    return sorted(key for key, count in counts.items() if count > 1)


def sign_matches_oracle(record: dict[str, Any]) -> bool:
    reported = float(record["full_expected_net_gain"])
    oracle = str(record["oracle_decision"])
    return (reported > 0 and oracle == "QUERY") or (reported <= 0 and oracle == "STOP")


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in records if "error" not in r]
    if not valid:
        return {"n": 0, "n_errors": len(records)}

    return {
        "n": len(valid),
        "n_errors": len(records) - len(valid),
        "decision_agreement_rate": sum(bool(r["decision_agreement"]) for r in valid) / len(valid),
        "reported_value_sign_accuracy": sum(sign_matches_oracle(r) for r in valid) / len(valid),
        "decision_value_self_consistency": (
            sum(bool(r["decision_matches_reported_value_sign"]) for r in valid) / len(valid)
        ),
        "first_successor_candidate_accuracy": (
            sum(bool(r["first_successor_candidate_correct"]) for r in valid) / len(valid)
        ),
        "oracle_identity_error": describe([
            float(r["oracle_identity_error"]) for r in valid
        ]),
        "one_backup_value_error": describe([
            float(r["one_backup_value_error"]) for r in valid
        ]),
        "first_successor_value_error": describe([
            float(r["first_successor_value_error"]) for r in valid
        ]),
        "current_query_value_error": describe([
            float(r["current_query_value_error"]) for r in valid
        ]),
        "reported_current_closure_error": describe([
            float(r["reported_current_closure_error"]) for r in valid
        ]),
        "local_decision_regret": describe([
            float(r["local_decision_regret"])
            for r in valid
            if r.get("local_decision_regret") is not None
        ]),
    }


def scientific_validity(
    records: list[dict[str, Any]],
    *,
    model: str,
    repeats: int,
    identity_tolerance: float = 1e-9,
) -> dict[str, Any]:
    expected_keys = {
        (model, condition_id, repeat)
        for condition_id in EXP08_CONDITION_IDS
        for repeat in range(repeats)
    }
    valid_records = [r for r in records if "error" not in r]
    observed_keys = {
        (
            str(r.get("replication_model")),
            str(r.get("condition_id")),
            int(r.get("repeat", -1)),
        )
        for r in valid_records
    }
    error_records = [r for r in records if "error" in r]
    missing = sorted(expected_keys - observed_keys)
    unexpected = sorted(observed_keys - expected_keys)
    identity_violations = [
        {
            "condition_id": r.get("condition_id"),
            "repeat": r.get("repeat"),
            "oracle_identity_error": r.get("oracle_identity_error"),
        }
        for r in valid_records
        if abs(float(r.get("oracle_identity_error", float("inf")))) > identity_tolerance
    ]
    complete = (
        len(records) == len(expected_keys)
        and not error_records
        and not missing
        and not unexpected
        and not identity_violations
    )
    return {
        "expected_records": len(expected_keys),
        "observed_records": len(records),
        "valid_records": len(valid_records),
        "error_records": len(error_records),
        "missing_probe_keys": [list(key) for key in missing],
        "unexpected_probe_keys": [list(key) for key in unexpected],
        "oracle_identity_tolerance": identity_tolerance,
        "oracle_identity_violations": identity_violations,
        "complete": complete,
    }


def aggregate_model(records: list[dict[str, Any]], *, model: str, repeats: int) -> dict[str, Any]:
    model_records = [r for r in records if str(r.get("replication_model")) == model]
    by_condition = {
        condition_id: summarize([
            r for r in model_records if str(r.get("condition_id")) == condition_id
        ])
        for condition_id in sorted(EXP08_CONDITION_IDS)
    }
    return {
        "model": model,
        "repeats": repeats,
        "conditions": sorted(EXP08_CONDITION_IDS),
        "overall": summarize(model_records),
        "by_condition": by_condition,
        "scientific_validity": scientific_validity(
            model_records,
            model=model,
            repeats=repeats,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    records = load_probe_records(args.input_dir)
    duplicates = duplicate_keys(records)
    by_model = {
        model: aggregate_model(records, model=model, repeats=args.repeats)
        for model in args.model
    }
    expected_models = set(args.model)
    unexpected_models = sorted(
        {
            str(r.get("replication_model"))
            for r in records
            if str(r.get("replication_model")) not in expected_models
        }
    )
    complete = (
        not duplicates
        and not unexpected_models
        and all(v["scientific_validity"]["complete"] for v in by_model.values())
    )
    result = {
        "models": by_model,
        "n_probe_files": len(records),
        "duplicate_probe_keys": [list(key) for key in duplicates],
        "unexpected_models": unexpected_models,
        "complete": complete,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("SCIENTIFIC_VALIDITY_GATE: " + ("PASS" if complete else "FAIL"))
    if args.require_complete and not complete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
