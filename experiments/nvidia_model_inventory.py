from __future__ import annotations

import json
import os
from pathlib import Path

from floop_agents.providers.nvidia import NvidiaClient


def main() -> None:
    api_key = os.getenv("NVIDIA_API_KEY") or os.getenv("NVDIA_API_KEY") or ""
    if not api_key:
        raise RuntimeError("NVIDIA API key is required")

    client = NvidiaClient(api_key=api_key, timeout=90, max_retries=2)
    model_ids = client.list_models()
    chat_candidates = client._chat_candidates(model_ids)
    non_openai = [m for m in chat_candidates if not m.lower().startswith("openai/")]

    payload = {
        "total_models": len(model_ids),
        "chat_candidates": chat_candidates,
        "non_openai_chat_candidates": non_openai,
    }

    out_dir = Path("results/nvidia_model_inventory")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "models.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
