from __future__ import annotations

import base64
import http.client
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def encode_image(image_path: Path) -> str:
    with image_path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def image_data_url(image_path: Path, mime_type: str = "image/png") -> str:
    return f"data:{mime_type};base64,{encode_image(image_path)}"


def extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    code_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    if code_match:
        return json.loads(code_match.group(1))

    decoder = json.JSONDecoder()
    start = stripped.find("{")
    while start >= 0:
        try:
            obj, _end = decoder.raw_decode(stripped[start:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            start = stripped.find("{", start + 1)
        else:
            break

    raise ValueError("Could not parse JSON from model response")


def _salvage_openai_partial_response(body: bytes) -> dict[str, Any] | None:
    text = body.decode("utf-8", errors="replace")
    decoder = json.JSONDecoder()
    for key in ('"content"', '"reasoning_content"', '"reasoning"'):
        search_from = 0
        while True:
            key_idx = text.find(key, search_from)
            if key_idx < 0:
                break
            colon_idx = text.find(":", key_idx + len(key))
            if colon_idx < 0:
                break
            value_idx = colon_idx + 1
            while value_idx < len(text) and text[value_idx].isspace():
                value_idx += 1
            try:
                value, _ = decoder.raw_decode(text[value_idx:])
            except json.JSONDecodeError:
                search_from = key_idx + len(key)
                continue
            if isinstance(value, str) and value.strip():
                field = key.strip('"')
                message = {"content": value}
                if field != "content":
                    message = {"content": "", field: value}
                return {"choices": [{"message": message}]}
            search_from = key_idx + len(key)
    return None


def call_openai_compatible(
    *,
    api_base: str,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float,
    timeout: int,
    extra_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
    }
    if os.environ.get("OPENAI_OMIT_TEMPERATURE", "").lower() not in {"1", "true", "yes", "on"}:
        payload["temperature"] = temperature
    if extra_payload:
        payload.update(extra_payload)
    req = urllib.request.Request(
        api_base,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            try:
                body = resp.read()
            except http.client.IncompleteRead as exc:
                body = exc.partial
                try:
                    return json.loads(body.decode("utf-8"))
                except Exception as parse_exc:  # noqa: BLE001
                    salvaged = _salvage_openai_partial_response(body)
                    if salvaged is not None:
                        return salvaged
                    raise RuntimeError(
                        f"IncompleteRead({len(body)} bytes read); partial_json_parse_failed"
                    ) from parse_exc
            return json.loads(body.decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        detail = body[:1000].replace("\n", " ")
        raise RuntimeError(f"HTTP Error {exc.code}: {exc.reason}; body={detail}") from exc


def response_text(response: dict[str, Any]) -> str:
    parts = response_parts(response)
    if parts["content"]:
        return parts["content"]
    if parts["reasoning_content"]:
        return parts["reasoning_content"]
    return json.dumps(response, ensure_ascii=False)


def response_parts(response: dict[str, Any]) -> dict[str, str]:
    if response.get("choices"):
        message = response["choices"][0].get("message", {})
        content = message.get("content", "")
        reasoning = (
            message.get("reasoning_content", "")
            or message.get("reasoning", "")
            or ""
        )
        content_text = ""
        if isinstance(content, str):
            content_text = content
        elif isinstance(content, list):
            content_text = "\n".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        return {
            "content": content_text,
            "reasoning_content": str(reasoning or ""),
        }
    if response.get("content"):
        content = response["content"]
        if isinstance(content, list):
            return {
                "content": "\n".join(part.get("text", "") for part in content if isinstance(part, dict)),
                "reasoning_content": "",
            }
    return {"content": "", "reasoning_content": ""}
