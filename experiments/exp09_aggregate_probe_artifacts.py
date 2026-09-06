from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from exp09_recursive_depth_ladder import EXP09_BACKUP_DEPTHS, EXP09_CONDITION_IDS


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
                "backup_depth": -1,
                "condition_id": "unknown",
                "repeat": -1,
                "error": f"{type(exc).__name__}: {exc}",
            }
        record["artifact_path"] = str(path)
        records.append(record)
    return records


def duplicate_keys(records: list[dict[str, Any]]) -> list[tuple[str, int, str, int]]:
    keys = [
        (
            str(r.get("replication_model")),
            int(r.get("backup_depth", -1)),
            str(r.get("condition_id")),
            int(r.get("repeat", -1)),
        )
        for r in records
    ]
    counts = Counter(keys)
    return sorted(key for key, count in counts.items() if count > 1)


def sign_matches_oracle(record: dict[str, Any]) -> bool:
    q = float(record["full_expected_net_gain"])
    oracle = str(record["oracle_decision"])
    return (q > 0 and oracle == "QUERY") or (q <= 0 and oracle == "STOP")


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in records if "error" not in r]
    if not valid:
        return {"n": 0, "n_errors": len(records)}

    flattened_backup_errors = [
        float(value)
        for record in valid
        for value in record.get("backup_errors", {}).values()
    ]
    flattened_closure_errors = [
        float(value)
        for record in valid
        for value in record.get("backup_closure_errors", {}).values()
    ]

    return {
        "n": len(valid),
        "n_errors": len(records) - len(valid),
        "decision_agreement_rate": sum(bool(r["decision_agreement"]) for r in valid) / len(valid),
        "reported_value_sign_accuracy": sum(sign_matches_oracle(r) for r in valid) / len(valid),
        "decision_value_self_consistency": (
            sum(bool(r["decision_matches_reported_value_sign"]) for r in valid) / len(valid)
        ),
        "action_valid_rate": sum(bool(r["action_valid"]) for r in valid) / len(valid),
        "first_successor_value_error": describe([
            float(r["first_successor_value_error"]) for r in valid
        ]),
        "current_query_value_error": describe([
            float(r["current_query_value_error"]) for r in valid
        ]),
        "all_backup_value_errors": describe(flattened_backup_errors),
        "all_backup_closure_errors": describe(flattened_closure_errors),
        "max_abs_backup_error": describe([
            float(r["max_abs_backup_error"]) for r in valid
        ]),
        "max_abs_backup_closure_error": describe([
            float(r["max_abs_backup_closure_error"]) for r in valid
        ]),
        "value_after_miss_consistency_error": describe([
            float(r["value_after_miss_consistency_error"]) for r in valid
        ]),
        "current_closure_error": describe([
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
        (model, depth, condition_id, repeat)
        for depth in EXP09_BACKUP_DEPTHS
        for condition_id in EXP09_CONDITION_IDS
        for repeat in range(repeats)
    }
    valid = [r for r in records if "error" not in r]
    observed_keys = {
        (
            str(r.get("replication_model")),
            int(r.get("backup_depth", -1)),
            str(r.get("condition_id")),
            int(r.get("repeat", -1)),
        )
        for r in valid
    }
    errors = [r for r in records if "error" in r]
    missing = sorted(expected_keys - observed_keys)
    unexpected = sorted(observed_keys - expected_keys)
    identity_violations = [
        {
            "backup_depth": r.get("backup_depth"),
            "condition_id": r.get("condition_id"),
            "repeat": r.get("repeat"),
            "max_abs_oracle_identity_error": r.get("max_abs_oracle_identity_error"),
        }
        for r in valid
        if abs(float(r.get("max_abs_oracle_identity_error", float("inf")))) > identity_tolerance
    ]
    complete = (
        len(records) == len(expected_keys)
        and not errors
        and not missing
        and not unexpected
        and not identity_violations
    )
    return {
        "expected_records": len(expected_keys),
        "observed_records": len(records),
        "valid_records": len(valid),
        "error_records": len(errors),
        "missing_probe_keys": [list(k) for k in missing],
        "unexpected_probe_keys": [list(k) for k in unexpected],
        "oracle_identity_tolerance": identity_tolerance,
        "oracle_identity_violations": identity_violations,
        "complete": complete,
    }


def aggregate_model(records: list[dict[str, Any]], *, model: str, repeats: int) -> dict[str, Any]:
    model_records = [r for r in records if str(r.get("replication_model")) == model]
    by_depth: dict[str, Any] = {}
    for depth in sorted(EXP09_BACKUP_DEPTHS):
        depth_records = [r for r in model_records if int(r.get("backup_depth", -1)) == depth]
        by_depth[str(depth)] = {
            "overall": summarize(depth_records),
            "by_condition": {
                condition_id: summarize([
                    r for r in depth_records if str(r.get("condition_id")) == condition_id
                ])
                for condition_id in sorted(EXP09_CONDITION_IDS)
            },
        }
    return {
        "model": model,
        "repeats": repeats,
        "backup_depths": sorted(EXP09_BACKUP_DEPTHS),
        "conditions": sorted(EXP09_CONDITION_IDS),
        "overall": summarize(model_records),
        "by_depth": by_depth,
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
        "duplicate_probe_keys": [list(k) for k in duplicates],
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
