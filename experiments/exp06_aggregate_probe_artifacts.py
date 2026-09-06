from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from exp06_repeated_transition_audit import (
    REPLICATION_CONDITION_IDS,
    scientific_validity,
    summarize,
)


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
                "artifact_path": str(path),
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


def aggregate_model(
    records: list[dict[str, Any]],
    *,
    model: str,
    repeats: int,
) -> dict[str, Any]:
    model_records = [r for r in records if str(r.get("replication_model")) == model]
    condition_ids = sorted(REPLICATION_CONDITION_IDS)
    summary = summarize(model_records)
    summary["model"] = model
    summary["repeats"] = repeats
    summary["conditions"] = condition_ids
    summary["scientific_validity"] = scientific_validity(
        model_records,
        repeats=repeats,
        condition_ids=condition_ids,
    )
    return summary


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
    by_model: dict[str, Any] = {}
    for model in args.model:
        by_model[model] = aggregate_model(records, model=model, repeats=args.repeats)

    unexpected_models = sorted(
        {
            str(r.get("replication_model"))
            for r in records
            if str(r.get("replication_model")) not in set(args.model)
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
