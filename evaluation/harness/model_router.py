"""Model callers: an OpenAI-compatible HTTP client and a deterministic stub.

The live caller talks to any `/chat/completions` endpoint (OpenAI, vLLM,
SGLang, ...) configured by OPENAI_API_BASE and OPENAI_API_KEY. The stub never
touches the network and is used by `--dry-run`.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class ChatResult:
    content: str
    reasoning_content: str = ""

    @property
    def final_text(self) -> str:
        return self.content or ""


class ModelCaller(Protocol):
    def __call__(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
        timeout: int = 120,
        extra: dict[str, Any] | None = None,
    ) -> ChatResult:
        ...


# --- Live caller ----------------------------------------------------------


def _salvage_partial_response(body: bytes) -> dict[str, Any] | None:
    """Recover the message text from a truncated JSON body, if possible."""
    text = body.decode("utf-8", errors="replace")
    decoder = json.JSONDecoder()
    for key in ('"content"', '"reasoning_content"', '"reasoning"'):
        start = 0
        while (idx := text.find(key, start)) >= 0:
            start = idx + len(key)
            colon = text.find(":", start)
            if colon < 0:
                break
            try:
                value, _ = decoder.raw_decode(text[colon + 1:].lstrip())
            except json.JSONDecodeError:
                continue
            if isinstance(value, str) and value.strip():
                field = key.strip('"')
                message = {"content": value} if field == "content" else {"content": "", field: value}
                return {"choices": [{"message": message}]}
    return None


def _post_json(url: str, api_key: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            try:
                body = resp.read()
            except http.client.IncompleteRead as exc:
                salvaged = _salvage_partial_response(exc.partial)
                if salvaged is None:
                    raise RuntimeError(f"IncompleteRead ({len(exc.partial)} bytes read)") from exc
                return salvaged
            return json.loads(body.decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000].replace("\n", " ")
        raise RuntimeError(f"HTTP Error {exc.code}: {exc.reason}; body={detail}") from exc


def _message_parts(response: dict[str, Any]) -> tuple[str, str]:
    """Return (content, reasoning_content) from a chat/completions response."""
    if not response.get("choices"):
        return "", ""
    message = response["choices"][0].get("message", {})
    content = message.get("content") or ""
    if isinstance(content, list):
        content = "\n".join(
            part.get("text", "") for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    return str(content), str(reasoning)


@dataclass
class OpenAICompatibleCaller:
    api_base: str
    api_key: str

    def __call__(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
        timeout: int = 120,
        extra: dict[str, Any] | None = None,
    ) -> ChatResult:
        url = self.api_base.rstrip("/")
        if not url.endswith("/chat/completions"):
            url += "/chat/completions"
        payload = {"model": model, "messages": messages, "temperature": temperature, **(extra or {})}
        content, reasoning = _message_parts(_post_json(url, self.api_key, payload, timeout))
        return ChatResult(content=content, reasoning_content=reasoning)


# --- Stub caller ----------------------------------------------------------


class StubModelCaller:
    """Deterministic offline caller for `--dry-run` and the smoke test.

    The answer letter is derived from a hash of (model, prompt), so results are
    reproducible across runs.
    """

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
                text_parts.extend(c.get("text", "") for c in content if c.get("type") == "text")
            elif isinstance(content, str):
                text_parts.append(content)
        digest = hashlib.sha256(f"{model}||{''.join(text_parts)[:2048]}".encode("utf-8")).hexdigest()
        letter = "ABCD"[int(digest[:2], 16) % 4]
        trace = (
            "[stub trace] Inspect the known panels, induce the visible rule, "
            "then apply it to the missing slot and compare the options.\n"
            f"Leaning towards option {letter}.\n"
            f"<ANSWER>{letter}</ANSWER>"
        )
        return ChatResult(content=trace)


def build_caller(*, dry_run: bool) -> ModelCaller:
    if dry_run:
        return StubModelCaller()
    base = os.environ.get("OPENAI_API_BASE")
    key = os.environ.get("OPENAI_API_KEY")
    if not base or not key:
        raise RuntimeError("Live mode requires OPENAI_API_BASE and OPENAI_API_KEY.")
    return OpenAICompatibleCaller(api_base=base, api_key=key)
