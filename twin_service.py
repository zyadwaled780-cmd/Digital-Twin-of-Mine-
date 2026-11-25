from __future__ import annotations

import json
import re
from typing import Any, Callable

from context import build_turn_system_prompt, get_active_knowledge_diagnostics
from input_guard import validate_and_classify_input
from providers import ProviderConfig, get_provider_diagnostics, run_chat_turn
from tools import handle_tool_calls, tools

MAX_BODY_BYTES = 65536
_VERCEL_ORIGIN_RE = re.compile(r"^https://[a-z0-9-]+\.vercel\.app$", re.IGNORECASE)
_LOCAL_ORIGIN_RE = re.compile(r"^http://(?:127\.0\.0\.1|localhost)(?::\d{1,5})?$", re.IGNORECASE)
_GITHUB_PAGES_ORIGIN = "https://zyadwaled780-cmd.github.io"


def extract_text(content: Any) -> str:
    """Normalize chat message content (str or multimodal block list) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
        return "\n".join(p for p in parts if p)
    return str(content) if content is not None else ""


def normalize_history(history: Any, max_turns: int = 20) -> list[dict[str, str]]:
    """Convert raw chat history into bounded OpenAI-compatible role/content dicts."""
    clean: list[dict[str, str]] = []
    for entry in history or []:
        if not isinstance(entry, dict):
            continue
        role = entry.get("role")
        if role not in ("user", "assistant"):
            continue
        text = extract_text(entry.get("content")).strip()
        if text:
            clean.append({"role": role, "content": text})
    if max_turns > 0 and len(clean) > max_turns:
        return clean[-max_turns:]
    return clean


def process_chat_turn(
    message: Any,
    history: Any = None,
    *,
    chat_runner: Callable[..., str] | None = None,
    config: ProviderConfig | None = None,
) -> dict[str, Any]:
    """Run input guards, CV-only retrieval, and configured LLM providers without UI coupling."""
    text = extract_text(message).strip()
    if not text:
        return {
            "ok": False,
            "rejected": True,
            "empty": True,
            "error": "Please type a message before sending.",
            "reply": "",
        }

    clean_history = normalize_history(history)
    is_valid, rejection_msg = validate_and_classify_input(text, history=clean_history)
    if not is_valid:
        return {
            "ok": True,
            "rejected": True,
            "empty": False,
            "reply": rejection_msg or "",
        }

    system_prompt = build_turn_system_prompt(query=text, history=clean_history)
    resolved_config = config or ProviderConfig.from_env()
    runner = chat_runner or run_chat_turn
    reply_text = runner(
        user_message=text,
        history=clean_history,
        system_prompt=system_prompt,
        tools=tools,
        tool_handler=handle_tool_calls,
        config=resolved_config,
    )
    return {
        "ok": True,
        "rejected": False,
        "empty": False,
        "reply": reply_text,
    }


def get_health_payload() -> dict[str, Any]:
    """Return safe, non-secret diagnostics about active CV knowledge and provider readiness."""
    knowledge_meta = get_active_knowledge_diagnostics()
    provider_meta = get_provider_diagnostics()
    return {
        "status": "ok",
        "knowledge": knowledge_meta,
        "providers": provider_meta,
    }


def resolve_cors_origin(origin: str | None) -> str:
    """Allow portfolio, Vercel preview/prod, and local origins; default to wildcard for stateless API."""
    cleaned = (origin or "").strip().rstrip("/")
    if (
        cleaned == _GITHUB_PAGES_ORIGIN
        or _VERCEL_ORIGIN_RE.fullmatch(cleaned)
        or _LOCAL_ORIGIN_RE.fullmatch(cleaned)
    ):
        return cleaned
    return "*"


def write_cors_headers(handler_obj: Any, methods: str) -> None:
    """Write standard CORS and no-cache security headers onto a BaseHTTPRequestHandler."""
    origin = resolve_cors_origin(handler_obj.headers.get("Origin"))
    handler_obj.send_header("Access-Control-Allow-Origin", origin)
    handler_obj.send_header("Access-Control-Allow-Methods", methods)
    handler_obj.send_header("Access-Control-Allow-Headers", "Content-Type")
    handler_obj.send_header("Vary", "Origin")
    handler_obj.send_header("Cache-Control", "no-store")


def send_json_response(
    handler_obj: Any,
    status_code: int,
    payload: dict[str, Any],
    *,
    methods: str,
) -> None:
    """Serialize payload as UTF-8 JSON with CORS headers and send HTTP response."""
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler_obj.send_response(status_code)
    write_cors_headers(handler_obj, methods)
    handler_obj.send_header("Content-Type", "application/json; charset=utf-8")
    handler_obj.send_header("Content-Length", str(len(body)))
    handler_obj.end_headers()
    handler_obj.wfile.write(body)


def send_options_response(handler_obj: Any, methods: str) -> None:
    """Respond to CORS preflight OPTIONS requests with HTTP 204."""
    handler_obj.send_response(204)
    write_cors_headers(handler_obj, methods)
    handler_obj.end_headers()


class PayloadTooLargeError(ValueError):
    """Raised when an HTTP JSON request body exceeds MAX_BODY_BYTES."""


def read_json_object_body(handler_obj: Any, max_bytes: int = MAX_BODY_BYTES) -> dict[str, Any]:
    """Read and validate a bounded UTF-8 JSON object from a BaseHTTPRequestHandler."""
    raw_len = handler_obj.headers.get("Content-Length")
    if raw_len is None or not str(raw_len).strip():
        raise ValueError("Request body must be valid JSON.")
    try:
        content_length = int(str(raw_len).strip())
    except ValueError as exc:
        raise ValueError("Invalid Content-Length header.") from exc
    if content_length < 0:
        raise ValueError("Invalid Content-Length header.")
    if content_length > max_bytes:
        raise PayloadTooLargeError("Request body too large.")
    if content_length == 0:
        raise ValueError("Request body must be a JSON object.")

    raw_bytes = handler_obj.rfile.read(content_length)
    if len(raw_bytes) > max_bytes:
        raise PayloadTooLargeError("Request body too large.")
    try:
        text = raw_bytes.decode("utf-8")
        data = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Request body must be valid UTF-8 JSON.") from exc
    if not isinstance(data, dict):
        raise ValueError("Request body must be a JSON object.")
    return data
