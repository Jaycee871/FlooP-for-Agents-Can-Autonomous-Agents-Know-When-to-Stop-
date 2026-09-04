#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from floop_agents.metrics import episode_metrics
from floop_agents.policies import (
    BoundedPolicy,
    MetaStopPolicy,
    UnboundedPolicy,
    VerifierStopPolicy,
)
from floop_agents.simulate import run_episode
from floop_agents.tasks import make_task_suite


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-family", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--step-cost", type=float, default=0.01)
    ap.add_argument("--output-dir", default="results")
    args = ap.parse_args()

    tasks = make_task_suite(
        args.n_per_family,
        seed=args.seed,
        horizon=args.horizon,
    )
    policies = [
        BoundedPolicy(bound=12),
        UnboundedPolicy(),
        MetaStopPolicy(threshold=0.012, patience=2, min_steps=3),
        VerifierStopPolicy(),
    ]

    rows = []
    for task in tasks:
        for policy in policies:
            result = run_episode(
                task,
                policy,
                external_cap=args.horizon,
                step_cost=args.step_cost,
            )
            rows.append(episode_metrics(task, result, step_cost=args.step_cost))

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "synthetic_pilot.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["policy"]].append(row)

    summary = {}
    for policy, items in grouped.items():
        summary[policy] = {
            "n": len(items),
            "mean_termination_regret": mean([float(x["termination_regret"]) for x in items]),
            "mean_excess_continuation_ratio": mean([float(x["excess_continuation_ratio"]) for x in items]),
            "premature_stop_rate": mean([float(bool(x["premature_stop"])) for x in items]),
            "failure_to_halt_rate": mean([float(bool(x["failure_to_halt"])) for x in items]),
            "mean_stop_step": mean([float(x["stop_step"]) for x in items]),
        }

    summary_path = out_dir / "synthetic_pilot_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"\nWrote {csv_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
