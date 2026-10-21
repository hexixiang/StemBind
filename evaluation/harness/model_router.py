"""Route model names to provider-specific call implementations.

Real backends use an OpenAI-compatible chat/completions endpoint (`common.vlm`).
For tests / `--dry-run` we expose a `StubModelCaller` which never touches the
network; it returns deterministic, model-aware synthetic traces.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

# Re-use the existing openai-compatible helper without duplicating.
_DATA_PROCESSING_DIR = Path(__file__).resolve().parents[2] / "data_processing"
if str(_DATA_PROCESSING_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_PROCESSING_DIR))


class ModelCaller(Protocol):
    def __call__(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
        timeout: int = 120,
        extra: dict[str, Any] | None = None,
    ) -> "ChatResult":
        ...


@dataclass
class ChatResult:
    content: str
    reasoning_content: str = ""
    raw_response: dict[str, Any] | None = None

    @property
    def final_text(self) -> str:
        return self.content or ""


# --- Real (network) caller ------------------------------------------------


@dataclass
class OpenAICompatibleCaller:
    api_base: str
    api_key: str

    def _normalize(self) -> str:
        base = (self.api_base or "").rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions"

    def __call__(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
        timeout: int = 120,
        extra: dict[str, Any] | None = None,
    ) -> ChatResult:
        from common.vlm import call_openai_compatible, response_parts  # type: ignore

        payload_extra = _env_extra_payload()
        if extra:
            payload_extra.update(extra)
        resp = call_openai_compatible(
            api_base=self._normalize(),
            api_key=self.api_key,
            model=model,
            messages=messages,
            temperature=temperature,
            timeout=timeout,
            extra_payload=payload_extra or None,
        )
        parts = response_parts(resp)
        return ChatResult(
            content=parts.get("content", ""),
            reasoning_content=parts.get("reasoning_content", ""),
            raw_response=resp,
        )


def _env_extra_payload() -> dict[str, Any]:
    """Read optional OpenAI-compatible request-body extras from env.

    vLLM exposes provider-specific settings such as `top_k` and
    `chat_template_kwargs` as top-level request JSON fields. The OpenAI Python
    SDK calls this `extra_body`; our urllib caller passes the same fields through
    via `extra_payload`.
    """
    payload: dict[str, Any] = {}
    for name in ("OPENAI_EXTRA_PAYLOAD_JSON", "OPENAI_EXTRA_BODY_JSON", "VLLM_EXTRA_BODY_JSON"):
        raw = os.environ.get(name)
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"{name} must be valid JSON.") from exc
        if not isinstance(obj, dict):
            raise RuntimeError(f"{name} must decode to a JSON object.")
        payload.update(obj)

    max_tokens = os.environ.get("OPENAI_MAX_TOKENS")
    if max_tokens:
        payload["max_tokens"] = int(max_tokens)

    enable_thinking = os.environ.get("QWEN_ENABLE_THINKING")
    if enable_thinking is not None:
        chat_kwargs = dict(payload.get("chat_template_kwargs") or {})
        chat_kwargs["enable_thinking"] = enable_thinking.lower() in {"1", "true", "yes", "on"}
        payload["chat_template_kwargs"] = chat_kwargs

    return payload


# --- Stub caller (dry-run / unit tests) -----------------------------------


@dataclass
class StubModelCaller:
    """Deterministic synthetic caller that never hits the network.

    The answer letter is derived from a hash of (model, task_id) so results are
    reproducible across runs. Short natural trace mimics a real model to let
    the judge path exercise itself.
    """

    letters: tuple[str, ...] = ("A", "B", "C", "D")

    def __call__(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
        timeout: int = 120,
        extra: dict[str, Any] | None = None,
    ) -> ChatResult:
        text_parts: list[str] = []
        for m in messages:
            content = m.get("content", "")
            if isinstance(content, list):
                for c in content:
                    if c.get("type") == "text":
                        text_parts.append(c.get("text", ""))
            elif isinstance(content, str):
                text_parts.append(content)
        key = f"{model}||{''.join(text_parts)[:2048]}".encode("utf-8")
        h = hashlib.sha256(key).hexdigest()
        letter = self.letters[int(h[:2], 16) % len(self.letters)]
        trace = (
            "[stub trace] Inspect the known panels, induce the visible rule, "
            "then apply it to the missing slot and compare the options.\n"
            f"Leaning towards option {letter}.\n"
            f"<ANSWER>{letter}</ANSWER>"
        )
        return ChatResult(content=trace, reasoning_content="")


# --- Factory --------------------------------------------------------------


def build_caller(*, dry_run: bool = True, api_base: str | None = None, api_key: str | None = None) -> ModelCaller:
    if dry_run:
        return StubModelCaller()
    base = api_base or os.environ.get("OPENAI_API_BASE")
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not base:
        raise RuntimeError("Live caller requires OPENAI_API_BASE.")
    if not key:
        raise RuntimeError("Live caller requires OPENAI_API_KEY.")
    return OpenAICompatibleCaller(api_base=base, api_key=key)


# --- Known model registry (advisory, not enforced) ------------------------

# Lightweight advisory manifest. The CLI accepts arbitrary model names and does
# not hardcode provider-specific branching.
KNOWN_MODELS: dict[str, dict[str, str]] = {
    "gpt-4o": {"family": "openai", "role": "closed_flagship"},
    "gpt-o3": {"family": "openai", "role": "thinking"},
    "gpt-5.4-mini": {"family": "openai", "role": "closed_mini"},
    "gemini-2.5-pro": {"family": "google", "role": "closed_flagship"},
    "gemini-2.5-flash": {"family": "google", "role": "closed_mini"},
    "claude-sonnet-4.5": {"family": "anthropic", "role": "closed_flagship"},
    "claude-opus-4.6": {"family": "anthropic", "role": "judge"},
    "qwen2.5-vl-3b-instruct": {"family": "qwen", "role": "scale_grad"},
    "qwen2.5-vl-7b-instruct": {"family": "qwen", "role": "scale_grad"},
    "qwen2.5-vl-32b-instruct": {"family": "qwen", "role": "scale_grad"},
    "qwen2.5-vl-72b-instruct": {"family": "qwen", "role": "scale_grad"},
    "qvq-72b": {"family": "qwen", "role": "thinking"},
    "kimi-vl": {"family": "moonshot", "role": "mid_tier"},
    "kimi-vl-thinking": {"family": "moonshot", "role": "thinking"},
    "internvl3-38b": {"family": "shanghai-ai", "role": "mid_tier"},
    "internvl3-78b": {"family": "shanghai-ai", "role": "mid_tier"},
    "minicpm-v-2.6": {"family": "openbmb", "role": "mid_tier"},
    "deepseek-vl2": {"family": "deepseek", "role": "mid_tier"},
    "glm-4v-thinking": {"family": "zhipu", "role": "thinking"},
}


def describe(model: str) -> dict[str, str]:
    return KNOWN_MODELS.get(model, {"family": "unknown", "role": "unknown"})


# Type alias for callers that want to install a custom caller (tests/mocks).
FactoryFn = Callable[..., ModelCaller]
