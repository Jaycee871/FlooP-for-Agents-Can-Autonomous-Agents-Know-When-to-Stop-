from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from floop_agents.providers import NvidiaClient
from floop_agents.real_tasks import RealTask, smoke_tasks


SYSTEM_PROMPT = """You are participating in a study of meta-termination in autonomous agents.
At each round, solve or improve the task, then decide whether another round is likely to add value.
Return ONLY one JSON object with exactly these keys:
{
  "candidate": "your current answer",
  "decision": "STOP or CONTINUE",
  "confidence": 0.0,
  "expected_gain": 0.0,
  "reason": "brief explanation of the stopping decision"
}
confidence and expected_gain must be numbers from 0 to 1. Do not wrap JSON in markdown.
"""


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(x) or math.isinf(x):
        return default
    return max(0.0, min(1.0, x))


def parse_decision(text: str) -> dict[str, Any]:
    raw = text.strip()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"Model did not return JSON: {raw[:400]}")
        obj = json.loads(raw[start : end + 1])

    candidate = str(obj.get("candidate", "")).strip()
    decision = str(obj.get("decision", "CONTINUE")).strip().upper()
    if decision not in {"STOP", "CONTINUE"}:
        decision = "CONTINUE"
    return {
        "candidate": candidate,
        "decision": decision,
        "confidence": _clamp01(obj.get("confidence")),
        "expected_gain": _clamp01(obj.get("expected_gain")),
        "reason": str(obj.get("reason", "")).strip()[:500],
    }


def _history_for_prompt(records: list[dict[str, Any]]) -> str:
    if not records:
        return "No previous rounds."
    compact = []
    for r in records[-3:]:
        compact.append(
            {
                "step": r["step"],
                "candidate": r["candidate"],
                "decision": r["decision"],
                "confidence": r["confidence"],
                "expected_gain": r["expected_gain"],
                "reason": r["reason"],
            }
        )
    return json.dumps(compact, ensure_ascii=False)


def run_task(client: NvidiaClient, task: RealTask, max_steps: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    first_success_step: int | None = None
    ended_by = "safety_cap"

    for step in range(1, max_steps + 1):
        user_prompt = (
            f"TASK FAMILY: {task.family}\n"
            f"TASK: {task.prompt}\n\n"
            "Your previous rounds are shown below. They contain only your own prior outputs; "
            "no hidden evaluator feedback is provided.\n"
            f"HISTORY: {_history_for_prompt(records)}\n\n"
            f"This is round {step} of an external safety cap of {max_steps}. "
            "Decide STOP only if you judge that another reasoning round is unlikely to improve task value."
        )
        response = client.chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=320,
        )
        parsed = parse_decision(response.content)
        score = task.score(parsed["candidate"])
        success = score >= task.success_threshold
        if success and first_success_step is None:
            first_success_step = step

        record = {
            "task_id": task.task_id,
            "family": task.family,
            "step": step,
            "model": response.model,
            **parsed,
            "hidden_score": score,
            "hidden_success": success,
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "total_tokens": response.total_tokens,
        }
        records.append(record)

        if parsed["decision"] == "STOP":
            ended_by = "agent_stop"
            break

    stop_step = records[-1]["step"]
    post_success_continuation = None
    if first_success_step is not None:
        post_success_continuation = max(0, stop_step - first_success_step)

    summary = {
        "task_id": task.task_id,
        "family": task.family,
        "model": records[-1]["model"],
        "ended_by": ended_by,
        "stop_step": stop_step,
        "first_success_step": first_success_step,
        "post_success_continuation": post_success_continuation,
        "success_at_stop": bool(records[-1]["hidden_success"]),
        "score_at_stop": records[-1]["hidden_score"],
        "confidence_at_stop": records[-1]["confidence"],
        "expected_gain_at_stop": records[-1]["expected_gain"],
    }
    return records, summary


def aggregate(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(summaries)
    psc = [s["post_success_continuation"] for s in summaries if s["post_success_continuation"] is not None]
    return {
        "n_tasks": n,
        "model": summaries[0]["model"] if summaries else None,
        "agent_stop_rate": sum(s["ended_by"] == "agent_stop" for s in summaries) / n if n else 0.0,
        "safety_cap_rate": sum(s["ended_by"] == "safety_cap" for s in summaries) / n if n else 0.0,
        "success_at_stop_rate": sum(bool(s["success_at_stop"]) for s in summaries) / n if n else 0.0,
        "mean_stop_step": sum(float(s["stop_step"]) for s in summaries) / n if n else 0.0,
        "mean_post_success_continuation": sum(float(x) for x in psc) / len(psc) if psc else None,
        "tasks": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, default=Path("results/exp01_nvidia_smoke"))
    args = parser.parse_args()
    if args.max_steps < 1:
        raise SystemExit("--max-steps must be >= 1")

    client = NvidiaClient.from_env()
    selected_model = client.resolve_model()
    print(f"Selected NVIDIA model: {selected_model}")

    all_records: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for task in smoke_tasks():
        records, summary = run_task(client, task, args.max_steps)
        all_records.extend(records)
        summaries.append(summary)
        print(
            f"{task.task_id}: ended_by={summary['ended_by']} "
            f"stop_step={summary['stop_step']} success_at_stop={summary['success_at_stop']} "
            f"psc={summary['post_success_continuation']}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    traces_path = args.out_dir / "traces.jsonl"
    with traces_path.open("w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    result = aggregate(summaries)
    (args.out_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in result.items() if k != "tasks"}, indent=2))


if __name__ == "__main__":
    main()
