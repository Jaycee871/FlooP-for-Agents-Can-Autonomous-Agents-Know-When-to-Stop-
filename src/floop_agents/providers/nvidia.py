from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class NvidiaResponse:
    content: str
    model: str
    reasoning: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class NvidiaClient:
    """Minimal OpenAI-compatible client for NVIDIA's hosted inference API.

    The adapter intentionally uses only the Python standard library so the benchmark
    stays lightweight. It never logs or serializes the API key. Transient network,
    rate-limit, and server failures are retried with bounded exponential backoff so
    long experiment sweeps do not fail because of a single slow request.
    """

    RETRYABLE_HTTP_CODES = {408, 409, 425, 429, 500, 502, 503, 504}

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        timeout: float = 180.0,
        max_retries: int = 3,
        retry_backoff: float = 2.0,
    ) -> None:
        if not api_key:
            raise ValueError("NVIDIA API key is required")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if retry_backoff < 0:
            raise ValueError("retry_backoff must be non-negative")
        self._api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

    @classmethod
    def from_env(cls) -> "NvidiaClient":
        api_key = os.getenv("NVIDIA_API_KEY") or os.getenv("NVDIA_API_KEY") or ""
        model = os.getenv("NVIDIA_MODEL") or None
        timeout = float(os.getenv("NVIDIA_TIMEOUT", "180"))
        max_retries = int(os.getenv("NVIDIA_MAX_RETRIES", "3"))
        retry_backoff = float(os.getenv("NVIDIA_RETRY_BACKOFF", "2"))
        return cls(
            api_key=api_key,
            model=model,
            timeout=timeout,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
        )

    def _make_request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None,
    ) -> urllib.request.Request:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        return urllib.request.Request(
            f"{self.base_url}/{path.lstrip('/')}",
            data=body,
            method=method,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": "floop-agents/0.1",
            },
        )

    def _sleep_before_retry(self, attempt: int) -> None:
        delay = self.retry_backoff * (2**attempt)
        if delay > 0:
            time.sleep(delay)

    def _request_json(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        last_error: BaseException | None = None
        for attempt in range(self.max_retries + 1):
            request = self._make_request(method, path, payload)
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                last_error = exc
                detail = exc.read().decode("utf-8", errors="replace")[:1000]
                if exc.code in self.RETRYABLE_HTTP_CODES and attempt < self.max_retries:
                    self._sleep_before_retry(attempt)
                    continue
                raise RuntimeError(f"NVIDIA API HTTP {exc.code}: {detail}") from exc
            except (urllib.error.URLError, TimeoutError, ConnectionResetError) as exc:
                last_error = exc
                if attempt < self.max_retries:
                    self._sleep_before_retry(attempt)
                    continue
                reason = getattr(exc, "reason", str(exc))
                raise RuntimeError(
                    f"NVIDIA API connection failed after {attempt + 1} attempts: {reason}"
                ) from exc

        raise RuntimeError(f"NVIDIA API request failed: {last_error}")

    def list_models(self) -> list[str]:
        payload = self._request_json("GET", "/models")
        ids: list[str] = []
        for item in payload.get("data", []):
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                ids.append(item["id"])
        return ids

    @staticmethod
    def _chat_candidates(model_ids: Iterable[str]) -> list[str]:
        excluded = ("embed", "rerank", "retrieval", "vision-embed", "nvclip")
        preferred_markers = (
            "instruct", "chat", "reason", "llama", "qwen", "mistral", "nemotron",
            "gpt-oss", "glm", "gemma", "deepseek",
        )
        ids = [m for m in model_ids if not any(term in m.lower() for term in excluded)]
        preferred = [m for m in ids if any(term in m.lower() for term in preferred_markers)]
        return preferred or ids

    def resolve_model(self) -> str:
        if self.model:
            return self.model
        model_ids = self.list_models()
        candidates = self._chat_candidates(model_ids)
        if not candidates:
            raise RuntimeError("No plausible chat-capable NVIDIA model was found")

        priorities = (
            "openai/gpt-oss-20b",
            "google/gemma-3-27b-it",
            "nvidia/llama-3.3-nemotron-super-49b-v1",
            "z-ai/glm-5.1",
            "deepseek-ai/deepseek-v4-flash",
        )
        for preferred in priorities:
            for model in candidates:
                if model.lower() == preferred:
                    self.model = model
                    return model

        self.model = candidates[0]
        return self.model

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        reasoning_effort: str | None = None,
    ) -> NvidiaResponse:
        model = self.resolve_model()
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if reasoning_effort is not None:
            payload["reasoning_effort"] = reasoning_effort

        data = self._request_json("POST", "/chat/completions", payload)
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"NVIDIA API returned no choices: {str(data)[:800]}")
        message = choices[0].get("message", {})
        content = message.get("content")
        reasoning = message.get("reasoning_content") or message.get("reasoning")
        if not isinstance(content, str) or not content.strip():
            finish_reason = choices[0].get("finish_reason")
            raise RuntimeError(
                "NVIDIA API returned no final content "
                f"(finish_reason={finish_reason!r}, reasoning_chars={len(reasoning) if isinstance(reasoning, str) else 0}). "
                "Increase max_tokens or lower reasoning_effort."
            )
        usage = data.get("usage") or {}
        return NvidiaResponse(
            content=content,
            model=str(data.get("model") or model),
            reasoning=reasoning if isinstance(reasoning, str) else None,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
        )
