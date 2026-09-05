from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("results/exp06_repeated_transition_audit")


def load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compact_model_summary(payload: dict[str, Any]) -> dict[str, Any]:
    overall = payload.get("overall", {})
    return {
        "model": payload.get("model"),
        "repeats": payload.get("repeats"),
        "n": overall.get("n"),
        "n_errors": overall.get("n_errors"),
        "decision_agreement_rate": overall.get("decision_agreement_rate"),
        "reported_value_sign_accuracy": overall.get("reported_value_sign_accuracy"),
        "decision_value_self_consistency": overall.get("decision_value_self_consistency"),
        "successor_candidate_accuracy": overall.get("successor_candidate_accuracy"),
        "coupling_directions": overall.get("coupling_directions"),
        "current_hit_mean_abs": (overall.get("current_hit_error") or {}).get("mean_abs"),
        "successor_existence_mean_abs": (overall.get("successor_existence_error") or {}).get("mean_abs"),
        "successor_hit_mean_abs": (overall.get("successor_hit_error") or {}).get("mean_abs"),
        "future_value_mean": (overall.get("future_value_error") or {}).get("mean"),
        "future_value_median": (overall.get("future_value_error") or {}).get("median"),
        "future_value_stdev": (overall.get("future_value_error") or {}).get("stdev"),
        "future_value_mean_abs": (overall.get("future_value_error") or {}).get("mean_abs"),
        "full_q_mean_abs": (overall.get("full_q_error") or {}).get("mean_abs"),
        "closure_mean_abs": (overall.get("closure_error") or {}).get("mean_abs"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/exp06_cross_model_summary.json"),
    )
    args = parser.parse_args()

    summaries: list[dict[str, Any]] = []
    for summary_path in sorted(args.root.glob("*/summary.json")):
        payload = load_summary(summary_path)
        compact = compact_model_summary(payload)
        compact["source"] = str(summary_path)
        summaries.append(compact)

    result = {
        "n_models": len(summaries),
        "models": summaries,
        "interpretation_guard": (
            "Compare mechanisms descriptively unless repeated samples and model coverage "
            "are sufficient for inferential claims. Provider/model-availability failures "
            "must not be counted as model decisions."
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
