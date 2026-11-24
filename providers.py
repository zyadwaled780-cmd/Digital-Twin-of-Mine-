from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
import time
from typing import Any, Callable
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
import openai

GROQ_DEFAULT_URL = "https://api.groq.com/openai/v1"
GROQ_DEFAULT_MODEL = "openai/gpt-oss-120b"
GEMINI_DEFAULT_MODEL = "gemini-4-hydrogen"
DEFAULT_TIMEOUT_SECONDS = 20.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_BACKOFF_BASE = 0.4
MAX_TOOL_ITERATIONS = 5


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    cleaned = value.strip().lower()
    if cleaned in ("1", "true", "yes", "on"):
        return True
    if cleaned in ("0", "false", "no", "off"):
        return False
    return default


def _parse_float(value: str | None, default: float, min_val: float = 0.5, max_val: float = 120.0) -> float:
    if not value:
        return default
    try:
        parsed = float(value.strip())
    except ValueError:
        return default
    return parsed if min_val <= parsed <= max_val else default


def _parse_int(value: str | None, default: int, min_val: int = 0, max_val: int = 5) -> int:
    if not value:
        return default
    try:
        parsed = int(value.strip())
    except ValueError:
        return default
    return parsed if min_val <= parsed <= max_val else default


@dataclass(frozen=True)
class ProviderConfig:
    primary_provider: str = "groq"
    fallback_provider: str = "gemini"
    fallback_enabled: bool = False
    fallback_on_auth_error: bool = False
    groq_api_key: str = ""
    groq_model: str = GROQ_DEFAULT_MODEL
    groq_base_url: str = GROQ_DEFAULT_URL
    gemini_api_key: str = ""
    gemini_model: str = GEMINI_DEFAULT_MODEL
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_MAX_RETRIES
    backoff_base_seconds: float = DEFAULT_BACKOFF_BASE

    @classmethod
    def from_env(cls) -> "ProviderConfig":
        primary = os.getenv("LLM_PRIMARY_PROVIDER", "groq").strip().lower() or "groq"
        fallback = os.getenv("LLM_FALLBACK_PROVIDER", "gemini").strip().lower() or "gemini"
        return cls(
            primary_provider=primary,
            fallback_provider=fallback,
            fallback_enabled=_parse_bool(os.getenv("LLM_FALLBACK_ENABLED"), default=False),
            fallback_on_auth_error=_parse_bool(os.getenv("LLM_FALLBACK_ON_AUTH_ERROR"), default=False),
            groq_api_key=os.getenv("GROQ_API_KEY", "").strip(),
            groq_model=os.getenv("GROQ_MODEL", GROQ_DEFAULT_MODEL).strip() or GROQ_DEFAULT_MODEL,
            groq_base_url=os.getenv("GROQ_BASE_URL", GROQ_DEFAULT_URL).strip() or GROQ_DEFAULT_URL,
            gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
            gemini_model=os.getenv("GEMINI_MODEL", GEMINI_DEFAULT_MODEL).strip() or GEMINI_DEFAULT_MODEL,
            timeout_seconds=_parse_float(os.getenv("LLM_TIMEOUT_SECONDS"), DEFAULT_TIMEOUT_SECONDS),
            max_retries=_parse_int(os.getenv("LLM_MAX_RETRIES"), DEFAULT_MAX_RETRIES),
        )


@dataclass(frozen=True)
class ToolCallRequest:
    id: str
    name: str
    arguments_json: str


@dataclass(frozen=True)
class ProviderTurnResult:
    text: str | None
    tool_calls: list[ToolCallRequest]
    provider: str
    model: str
    safety_refused: bool = False


class ProviderInvocationError(RuntimeError):
    """Sanitized provider failure with explicit retry and fallback eligibility."""

    def __init__(
        self,
        *,
        category: str,
        provider: str,
        model: str,
        retryable: bool,
        fallback_eligible: bool,
        user_message: str,
    ):
        super().__init__(f"{provider} ({model}) failed: {category}")
        self.category = category
        self.provider = provider
        self.model = model
        self.retryable = retryable
        self.fallback_eligible = fallback_eligible
        self.user_message = user_message


def _normalize_gemini_schema(schema: Any) -> Any:
    """Normalize JSON schema union types like ['string', 'null'] to single OpenAPI/Gemini types."""
    if not isinstance(schema, dict):
        return schema
    normalized = dict(schema)
    raw_type = normalized.get("type")
    if isinstance(raw_type, list):
        non_null = [t for t in raw_type if t != "null"]
        normalized["type"] = non_null[0] if non_null else "string"
        if "null" in raw_type:
            normalized["nullable"] = True
    props = normalized.get("properties")
    if isinstance(props, dict):
        normalized["properties"] = {k: _normalize_gemini_schema(v) for k, v in props.items()}
    items = normalized.get("items")
    if isinstance(items, dict):
        normalized["items"] = _normalize_gemini_schema(items)
    return normalized


def convert_tools_to_gemini(openai_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI function tool schemas into Gemini Interactions FunctionParam dicts."""
    converted: list[dict[str, Any]] = []
    for item in openai_tools or []:
        if item.get("type") != "function":
            continue
        fn = item.get("function") or {}
        name = fn.get("name")
        if not name:
            continue
        tool_def: dict[str, Any] = {
            "type": "function",
            "name": name,
        }
        if fn.get("description"):
            tool_def["description"] = fn["description"]
        if fn.get("parameters"):
            tool_def["parameters"] = _normalize_gemini_schema(fn["parameters"])
        converted.append(tool_def)
    return converted


def _canonical_args_key(tool_name: str, raw_args: str) -> tuple[str, str]:
    try:
        parsed = json.loads(raw_args or "{}")
        normalized = json.dumps(parsed, sort_keys=True, ensure_ascii=False)
    except json.JSONDecodeError:
        normalized = (raw_args or "").strip()
    return (tool_name, normalized)


class TurnToolExecutor:
    """In-memory per-turn tool deduplication across retries and fallback.

    Note: This prevents duplicate external side effects only within a single
    in-memory turn execution; it is NOT durable exactly-once execution across
    separate turns, separate requests, or process crashes/restarts.
    """

    def __init__(self, handler_fn: Callable[[list[Any]], list[dict[str, Any]]]):
        self._handler_fn = handler_fn
        self._ledger: dict[tuple[str, str], dict[str, Any]] = {}
        self.executed_calls: list[tuple[str, str]] = []

    def execute(self, tool_calls: list[ToolCallRequest]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        uncached: list[Any] = []
        uncached_keys: list[tuple[str, str]] = []

        for call in tool_calls:
            key = _canonical_args_key(call.name, call.arguments_json)
            if key in self._ledger:
                cached = dict(self._ledger[key])
                cached["tool_call_id"] = call.id
                results.append(cached)
            else:
                shim = _ToolCallShim(id=call.id, name=call.name, arguments=call.arguments_json)
                uncached.append(shim)
                uncached_keys.append(key)
                results.append({"_pending_id": call.id})

        if uncached:
            fresh_results = self._handler_fn(uncached)
            by_id = {res.get("tool_call_id"): res for res in fresh_results}
            for shim, key in zip(uncached, uncached_keys):
                res = by_id.get(
                    shim.id,
                    {"role": "tool", "content": json.dumps({"error": "No tool result"}), "tool_call_id": shim.id},
                )
                self._ledger[key] = dict(res)
                self.executed_calls.append(key)
            results = [by_id.get(r["_pending_id"], r) if "_pending_id" in r else r for r in results]

        return results


@dataclass
class _FunctionShim:
    name: str
    arguments: str


class _ToolCallShim:
    def __init__(self, id: str, name: str, arguments: str):
        self.id = id
        self.function = _FunctionShim(name=name, arguments=arguments)


def _build_openai_messages(
    system_prompt: str,
    history: list[dict[str, str]],
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    for ev in events:
        etype = ev["type"]
        if etype == "user":
            messages.append({"role": "user", "content": ev["text"]})
        elif etype == "assistant_tool_calls":
            messages.append(
                {
                    "role": "assistant",
                    "content": ev.get("text") or None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": tc.arguments_json},
                        }
                        for tc in ev["tool_calls"]
                    ],
                }
            )
        elif etype == "tool_result":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": ev["tool_call_id"],
                    "content": ev["content"],
                }
            )
    return messages


def _build_gemini_steps(
    history: list[dict[str, str]],
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for entry in history:
        role = entry.get("role")
        text = entry.get("content", "")
        if not text:
            continue
        if role == "user":
            steps.append({"type": "user_input", "content": [{"type": "text", "text": text}]})
        elif role == "assistant":
            steps.append({"type": "model_output", "content": [{"type": "text", "text": text}]})

    for ev in events:
        etype = ev["type"]
        if etype == "user":
            steps.append({"type": "user_input", "content": [{"type": "text", "text": ev["text"]}]})
        elif etype == "assistant_tool_calls":
            for tc in ev["tool_calls"]:
                try:
                    parsed_args = json.loads(tc.arguments_json or "{}")
                    if not isinstance(parsed_args, dict):
                        parsed_args = {}
                except json.JSONDecodeError:
                    parsed_args = {}
                steps.append(
                    {
                        "type": "function_call",
                        "id": tc.id,
                        "name": tc.name,
                        "arguments": parsed_args,
                    }
                )
        elif etype == "tool_result":
            steps.append(
                {
                    "type": "function_result",
                    "call_id": ev["tool_call_id"],
                    "name": ev["name"],
                    "result": [{"type": "text", "text": ev["content"]}],
                }
            )
    return steps


class GroqProvider:
    name = "groq"

    def __init__(self, config: ProviderConfig, client_factory=openai.OpenAI):
        self.config = config
        self._client_factory = client_factory
        self._client = None

    def is_configured(self) -> bool:
        return bool(self.config.groq_api_key)

    def _get_client(self):
        if self._client is None:
            # Set SDK max_retries=0 so retries are strictly bounded by _invoke_with_retries().
            self._client = self._client_factory(
                api_key=self.config.groq_api_key,
                base_url=self.config.groq_base_url,
                timeout=self.config.timeout_seconds,
                max_retries=0,
            )
        return self._client

    def classify_error(self, exc: Exception) -> ProviderInvocationError:
        model = self.config.groq_model
        if isinstance(exc, ProviderInvocationError):
            return exc
        if isinstance(exc, openai.ContentFilterFinishReasonError):
            return ProviderInvocationError(
                category="safety_refusal",
                provider=self.name,
                model=model,
                retryable=False,
                fallback_eligible=False,
                user_message="I cannot answer that request due to content safety policies.",
            )
        if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)):
            return ProviderInvocationError(
                category="auth_error",
                provider=self.name,
                model=model,
                retryable=False,
                fallback_eligible=self.config.fallback_on_auth_error,
                user_message="The Digital Twin provider authentication failed. Please check server configuration.",
            )
        if isinstance(exc, openai.APITimeoutError):
            return ProviderInvocationError(
                category="timeout",
                provider=self.name,
                model=model,
                retryable=True,
                fallback_eligible=True,
                user_message="The Digital Twin timed out while generating a reply. Please try again.",
            )
        if isinstance(exc, openai.APIConnectionError):
            return ProviderInvocationError(
                category="connection_error",
                provider=self.name,
                model=model,
                retryable=True,
                fallback_eligible=True,
                user_message="Could not reach the AI provider. Please try again in a moment.",
            )
        if isinstance(exc, openai.RateLimitError):
            return ProviderInvocationError(
                category="rate_limit",
                provider=self.name,
                model=model,
                retryable=True,
                fallback_eligible=True,
                user_message="The Digital Twin is temporarily busy. Please try again shortly.",
            )
        if isinstance(exc, openai.InternalServerError):
            return ProviderInvocationError(
                category="server_error",
                provider=self.name,
                model=model,
                retryable=True,
                fallback_eligible=True,
                user_message="The AI provider is experiencing temporary issues. Please try again.",
            )
        if isinstance(exc, openai.APIStatusError):
            code = getattr(exc, "status_code", 0)
            if code in (401, 403):
                return ProviderInvocationError(
                    category="auth_error",
                    provider=self.name,
                    model=model,
                    retryable=False,
                    fallback_eligible=self.config.fallback_on_auth_error,
                    user_message="The Digital Twin provider authentication failed. Please check server configuration.",
                )
            if code in (429, 500, 502, 503, 504):
                cat = "rate_limit" if code == 429 else "server_error"
                return ProviderInvocationError(
                    category=cat,
                    provider=self.name,
                    model=model,
                    retryable=True,
                    fallback_eligible=True,
                    user_message="The AI provider is temporarily unavailable. Please try again shortly.",
                )
            body = getattr(exc, "body", None)
            err_code = body.get("code") if isinstance(body, dict) else None
            if not err_code and isinstance(body, dict) and isinstance(body.get("error"), dict):
                err_code = body["error"].get("code")
            if err_code == "tool_use_failed":
                return ProviderInvocationError(
                    category="tool_use_failed",
                    provider=self.name,
                    model=model,
                    retryable=True,
                    fallback_eligible=True,
                    user_message="I had trouble formatting that request. Please try sending your message again.",
                )
            return ProviderInvocationError(
                category="bad_request",
                provider=self.name,
                model=model,
                retryable=False,
                fallback_eligible=False,
                user_message="The Digital Twin encountered a request configuration error.",
            )
        return ProviderInvocationError(
            category="unexpected_error",
            provider=self.name,
            model=model,
            retryable=False,
            fallback_eligible=False,
            user_message="Sorry, I'm having trouble connecting right now. Please try again in a moment.",
        )

    def generate_step(
        self,
        *,
        system_prompt: str,
        history: list[dict[str, str]],
        events: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ProviderTurnResult:
        if not self.is_configured():
            raise ProviderInvocationError(
                category="not_configured",
                provider=self.name,
                model=self.config.groq_model,
                retryable=False,
                fallback_eligible=False,
                user_message="GROQ_API_KEY is not configured.",
            )

        client = self._get_client()
        messages = _build_openai_messages(system_prompt, history, events)
        try:
            response = client.chat.completions.create(
                model=self.config.groq_model,
                messages=messages,
                tools=tools,
                temperature=0.2,
            )
        except Exception as exc:
            raise self.classify_error(exc) from exc

        choices = getattr(response, "choices", None)
        if not choices or not isinstance(choices, list):
            raise ProviderInvocationError(
                category="malformed_response",
                provider=self.name,
                model=self.config.groq_model,
                retryable=False,
                fallback_eligible=True,
                user_message="The AI provider returned an unexpected empty response.",
            )

        choice = choices[0]
        if getattr(choice, "finish_reason", None) == "content_filter":
            raise ProviderInvocationError(
                category="safety_refusal",
                provider=self.name,
                model=self.config.groq_model,
                retryable=False,
                fallback_eligible=False,
                user_message="I cannot answer that request due to content safety policies.",
            )

        msg = getattr(choice, "message", None)
        if msg is None:
            raise ProviderInvocationError(
                category="malformed_response",
                provider=self.name,
                model=self.config.groq_model,
                retryable=False,
                fallback_eligible=True,
                user_message="The AI provider returned an incomplete message object.",
            )

        if getattr(msg, "refusal", None):
            return ProviderTurnResult(
                text=str(msg.refusal),
                tool_calls=[],
                provider=self.name,
                model=self.config.groq_model,
                safety_refused=True,
            )

        raw_calls = getattr(msg, "tool_calls", None) or []
        parsed_calls: list[ToolCallRequest] = []
        for idx, tc in enumerate(raw_calls):
            fn = getattr(tc, "function", None)
            fn_name = getattr(fn, "name", None) if fn is not None else None
            if not fn_name:
                raise ProviderInvocationError(
                    category="malformed_response",
                    provider=self.name,
                    model=self.config.groq_model,
                    retryable=False,
                    fallback_eligible=True,
                    user_message="The AI provider returned a malformed tool request.",
                )
            parsed_calls.append(
                ToolCallRequest(
                    id=str(getattr(tc, "id", "") or f"groq_call_{idx}"),
                    name=str(fn_name),
                    arguments_json=str(getattr(fn, "arguments", None) or "{}"),
                )
            )

        return ProviderTurnResult(
            text=getattr(msg, "content", None),
            tool_calls=parsed_calls,
            provider=self.name,
            model=self.config.groq_model,
        )


class GeminiProvider:
    name = "gemini"

    def __init__(self, config: ProviderConfig, client_factory=genai.Client):
        self.config = config
        self._client_factory = client_factory
        self._client = None

    def is_configured(self) -> bool:
        return bool(self.config.gemini_api_key)

    def _get_client(self):
        if self._client is None:
            # Disable SDK internal retry loop (attempts=1) so _invoke_with_retries() is the single retry owner.
            # HttpOptions.timeout expects milliseconds in google-genai.
            http_options = genai_types.HttpOptions(
                timeout=int(self.config.timeout_seconds * 1000),
                retry_options=genai_types.HttpRetryOptions(attempts=1),
            )
            self._client = self._client_factory(
                api_key=self.config.gemini_api_key,
                http_options=http_options,
            )
        return self._client

    def classify_error(self, exc: Exception) -> ProviderInvocationError:
        model = self.config.gemini_model
        exc_name = type(exc).__name__
        if isinstance(exc, ProviderInvocationError):
            return exc
        if isinstance(exc, TimeoutError) or exc_name in ("APITimeoutError", "ReadTimeout", "ConnectTimeout"):
            return ProviderInvocationError(
                category="timeout",
                provider=self.name,
                model=model,
                retryable=True,
                fallback_eligible=True,
                user_message="The backup AI provider timed out. Please try again.",
            )
        if isinstance(exc, ConnectionError) or exc_name in ("APIConnectionError", "ConnectError"):
            return ProviderInvocationError(
                category="connection_error",
                provider=self.name,
                model=model,
                retryable=True,
                fallback_eligible=True,
                user_message="Could not reach the backup AI provider. Please try again.",
            )

        raw_code = getattr(exc, "status_code", None)
        if raw_code is None:
            raw_code = getattr(exc, "code", 0)
        try:
            code = int(raw_code or 0)
        except (TypeError, ValueError):
            code = 0

        compat_api_names = (
            "RateLimitError",
            "InternalServerError",
            "AuthenticationError",
            "PermissionDeniedError",
            "BadRequestError",
            "APIStatusError",
            "NotFoundError",
            "ConflictError",
            "UnprocessableEntityError",
        )
        if (
            isinstance(exc, genai_errors.APIError)
            or isinstance(getattr(exc, "status_code", None), int)
            or exc_name in compat_api_names
        ):
            msg = str(getattr(exc, "message", "") or exc or "").lower()
            if "safety" in msg or "blocked" in msg:
                return ProviderInvocationError(
                    category="safety_refusal",
                    provider=self.name,
                    model=model,
                    retryable=False,
                    fallback_eligible=False,
                    user_message="I cannot answer that request due to content safety policies.",
                )
            if code in (401, 403) or exc_name in ("AuthenticationError", "PermissionDeniedError"):
                return ProviderInvocationError(
                    category="auth_error",
                    provider=self.name,
                    model=model,
                    retryable=False,
                    fallback_eligible=self.config.fallback_on_auth_error,
                    user_message="Gemini authentication failed. Please check GEMINI_API_KEY.",
                )
            if (
                code in (429, 500, 502, 503, 504)
                or isinstance(exc, genai_errors.ServerError)
                or exc_name in ("RateLimitError", "InternalServerError")
            ):
                cat = "rate_limit" if (code == 429 or exc_name == "RateLimitError") else "server_error"
                return ProviderInvocationError(
                    category=cat,
                    provider=self.name,
                    model=model,
                    retryable=True,
                    fallback_eligible=True,
                    user_message="The backup AI provider is temporarily unavailable. Please try again shortly.",
                )
            return ProviderInvocationError(
                category="bad_request",
                provider=self.name,
                model=model,
                retryable=False,
                fallback_eligible=False,
                user_message="Gemini rejected the request configuration.",
            )
        return ProviderInvocationError(
            category="unexpected_error",
            provider=self.name,
            model=model,
            retryable=False,
            fallback_eligible=False,
            user_message="Sorry, I'm having trouble connecting right now. Please try again in a moment.",
        )

    def generate_step(
        self,
        *,
        system_prompt: str,
        history: list[dict[str, str]],
        events: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ProviderTurnResult:
        if not self.is_configured():
            raise ProviderInvocationError(
                category="not_configured",
                provider=self.name,
                model=self.config.gemini_model,
                retryable=False,
                fallback_eligible=False,
                user_message="GEMINI_API_KEY is not configured.",
            )

        client = self._get_client()
        gemini_steps = _build_gemini_steps(history, events)
        gemini_tools = convert_tools_to_gemini(tools)
        try:
            interaction = client.interactions.create(
                model=self.config.gemini_model,
                system_instruction=system_prompt,
                input=gemini_steps,
                tools=gemini_tools,
                store=False,
                timeout=self.config.timeout_seconds,
            )
        except Exception as exc:
            raise self.classify_error(exc) from exc

        if interaction is None:
            raise ProviderInvocationError(
                category="malformed_response",
                provider=self.name,
                model=self.config.gemini_model,
                retryable=False,
                fallback_eligible=True,
                user_message="Gemini returned an empty response.",
            )

        status = str(getattr(interaction, "status", "") or "").lower()
        if status == "failed":
            errors = getattr(interaction, "errors", None) or []
            err_text = " ".join(str(e) for e in errors).lower()
            if "safety" in err_text or "policy" in err_text or "block" in err_text:
                raise ProviderInvocationError(
                    category="safety_refusal",
                    provider=self.name,
                    model=self.config.gemini_model,
                    retryable=False,
                    fallback_eligible=False,
                    user_message="I cannot answer that request due to content safety policies.",
                )
            raise ProviderInvocationError(
                category="server_error",
                provider=self.name,
                model=self.config.gemini_model,
                retryable=True,
                fallback_eligible=True,
                user_message="The AI provider failed to complete the request.",
            )

        steps = getattr(interaction, "steps", None)
        if steps is not None and not isinstance(steps, list):
            raise ProviderInvocationError(
                category="malformed_response",
                provider=self.name,
                model=self.config.gemini_model,
                retryable=False,
                fallback_eligible=True,
                user_message="Gemini returned malformed interaction steps.",
            )

        tool_calls: list[ToolCallRequest] = []
        text_parts: list[str] = []
        for idx, step in enumerate(steps or []):
            stype = getattr(step, "type", "")
            if stype == "function_call":
                fn_name = str(getattr(step, "name", "") or "").strip()
                if not fn_name:
                    raise ProviderInvocationError(
                        category="malformed_response",
                        provider=self.name,
                        model=self.config.gemini_model,
                        retryable=False,
                        fallback_eligible=True,
                        user_message="Gemini returned a malformed function call.",
                    )
                args_obj = getattr(step, "arguments", None)
                if args_obj is None:
                    args_obj = {}
                elif not isinstance(args_obj, dict):
                    raise ProviderInvocationError(
                        category="malformed_response",
                        provider=self.name,
                        model=self.config.gemini_model,
                        retryable=False,
                        fallback_eligible=True,
                        user_message="Gemini returned non-object function arguments.",
                    )
                tool_calls.append(
                    ToolCallRequest(
                        id=str(getattr(step, "id", "") or f"gemini_call_{idx}"),
                        name=fn_name,
                        arguments_json=json.dumps(args_obj, ensure_ascii=False),
                    )
                )
            elif stype == "model_output":
                for block in getattr(step, "content", None) or []:
                    if getattr(block, "type", "") == "text" and getattr(block, "text", None):
                        text_parts.append(str(block.text))

        output_text = getattr(interaction, "output_text", None) or (
            "\n".join(text_parts) if text_parts else None
        )
        if not tool_calls and not output_text:
            raise ProviderInvocationError(
                category="malformed_response",
                provider=self.name,
                model=self.config.gemini_model,
                retryable=False,
                fallback_eligible=True,
                user_message="Gemini returned neither text nor tool calls.",
            )

        return ProviderTurnResult(
            text=output_text,
            tool_calls=tool_calls,
            provider=self.name,
            model=self.config.gemini_model,
        )


_EMAIL_STRIP_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_URL_STRIP_RE = re.compile(r"(?:https?://|github\.com/|linkedin\.com/)\S+", re.IGNORECASE)
_ARABIC_LETTER_RE = re.compile(r"[\u0621-\u064A\u0671-\u06D3]")


def _enrich_tool_result_with_language_reminder(
    content: str,
    user_message: str,
    history: list[dict[str, str]],
) -> str:
    """Inject a non-persistent reply language reminder into JSON tool results sent back to the LLM."""
    cleaned_user = _EMAIL_STRIP_RE.sub(" ", _URL_STRIP_RE.sub(" ", user_message or "")).strip()
    is_arabic = bool(_ARABIC_LETTER_RE.search(cleaned_user))
    if not is_arabic and not any(ch.isalpha() for ch in cleaned_user):
        for entry in reversed(history or []):
            if isinstance(entry, dict) and entry.get("role") == "user":
                hist_clean = _EMAIL_STRIP_RE.sub(
                    " ", _URL_STRIP_RE.sub(" ", str(entry.get("content", "")))
                ).strip()
                if _ARABIC_LETTER_RE.search(hist_clean):
                    is_arabic = True
                    break
                if any(ch.isalpha() for ch in hist_clean):
                    break

    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return content
    if not isinstance(payload, dict):
        return content

    payload["reply_language_reminder"] = (
        "أجب على المستخدم باللغة العربية المصرية فقط، ولا تتحول للإنجليزية. اسم زياد بالعربي: زياد وليد طلعت."
        if is_arabic
        else "Reply strictly in English (only English and Arabic are supported)."
    )
    return json.dumps(payload, ensure_ascii=False)


def _log_provider_failure(err: ProviderInvocationError) -> None:
    """Log sanitized provider failure info without keys or private prompts."""
    print(
        f"Provider failure: provider={err.provider} model={err.model} "
        f"category={err.category} retryable={err.retryable}",
        flush=True,
    )


def _build_provider_instance(name: str, config: ProviderConfig):
    if name == "groq":
        return GroqProvider(config)
    if name == "gemini":
        return GeminiProvider(config)
    raise ValueError(f"Unsupported provider name: {name}")


def _invoke_with_retries(
    provider,
    *,
    system_prompt: str,
    history: list[dict[str, str]],
    events: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    max_retries: int,
    backoff_base: float,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> ProviderTurnResult:
    last_err: ProviderInvocationError | None = None
    for attempt in range(max_retries + 1):
        try:
            return provider.generate_step(
                system_prompt=system_prompt,
                history=history,
                events=events,
                tools=tools,
            )
        except ProviderInvocationError as err:
            last_err = err
            _log_provider_failure(err)
            if not err.retryable or attempt >= max_retries:
                raise err
            delay = min(backoff_base * (2**attempt), 4.0)
            sleep_fn(delay)
    assert last_err is not None
    raise last_err


def run_chat_turn(
    *,
    user_message: str,
    history: list[dict[str, str]],
    system_prompt: str,
    tools: list[dict[str, Any]],
    tool_handler: Callable[[list[Any]], list[dict[str, Any]]],
    config: ProviderConfig | None = None,
    primary_provider=None,
    fallback_provider=None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> str:
    """Execute one non-streaming chat turn with Groq primary, optional Gemini fallback, and per-turn tool dedupe."""
    cfg = config or ProviderConfig.from_env()
    primary = primary_provider or _build_provider_instance(cfg.primary_provider, cfg)
    fallback = None
    if cfg.fallback_enabled and cfg.fallback_provider != cfg.primary_provider:
        fallback = fallback_provider or _build_provider_instance(cfg.fallback_provider, cfg)

    events: list[dict[str, Any]] = [{"type": "user", "text": user_message}]
    executor = TurnToolExecutor(tool_handler)
    active_provider = primary
    used_fallback = False

    if not primary.is_configured():
        if cfg.fallback_on_auth_error and fallback is not None and fallback.is_configured():
            print(
                f"Primary provider={primary.name} not configured; using fallback provider={fallback.name}",
                flush=True,
            )
            active_provider = fallback
            used_fallback = True
        else:
            return (
                f"The Digital Twin is temporarily unavailable "
                f"({primary.name.upper()} API key is not configured)."
            )

    for _ in range(MAX_TOOL_ITERATIONS + 1):
        try:
            turn_res = _invoke_with_retries(
                active_provider,
                system_prompt=system_prompt,
                history=history,
                events=events,
                tools=tools,
                max_retries=cfg.max_retries,
                backoff_base=cfg.backoff_base_seconds,
                sleep_fn=sleep_fn,
            )
        except ProviderInvocationError as err:
            if (
                err.fallback_eligible
                and not used_fallback
                and fallback is not None
                and fallback.is_configured()
            ):
                print(
                    f"Falling back from provider={active_provider.name} to provider={fallback.name} "
                    f"(reason={err.category})",
                    flush=True,
                )
                active_provider = fallback
                used_fallback = True
                try:
                    turn_res = _invoke_with_retries(
                        active_provider,
                        system_prompt=system_prompt,
                        history=history,
                        events=events,
                        tools=tools,
                        max_retries=cfg.max_retries,
                        backoff_base=cfg.backoff_base_seconds,
                        sleep_fn=sleep_fn,
                    )
                except ProviderInvocationError as fb_err:
                    return fb_err.user_message
            else:
                if (
                    err.fallback_eligible
                    and not used_fallback
                    and cfg.fallback_enabled
                    and fallback is not None
                    and not fallback.is_configured()
                ):
                    print(
                        f"Fallback skipped: provider={fallback.name} is enabled but API key is missing.",
                        flush=True,
                    )
                return err.user_message

        if turn_res.safety_refused:
            return turn_res.text or "I cannot answer that request due to content safety policies."

        if not turn_res.tool_calls:
            return turn_res.text or "I couldn't generate a response right now. Please try again."

        tool_rounds_so_far = sum(1 for ev in events if ev["type"] == "assistant_tool_calls")
        if tool_rounds_so_far >= MAX_TOOL_ITERATIONS:
            return "I hit a processing limit while handling your request. Please try rephrasing your question."

        events.append(
            {
                "type": "assistant_tool_calls",
                "text": turn_res.text,
                "tool_calls": turn_res.tool_calls,
            }
        )
        tool_results = executor.execute(turn_res.tool_calls)
        name_by_id = {tc.id: tc.name for tc in turn_res.tool_calls}
        for res in tool_results:
            call_id = str(res.get("tool_call_id", ""))
            enriched_content = _enrich_tool_result_with_language_reminder(
                str(res.get("content", "")),
                user_message,
                history,
            )
            events.append(
                {
                    "type": "tool_result",
                    "tool_call_id": call_id,
                    "name": name_by_id.get(call_id, ""),
                    "content": enriched_content,
                }
            )

    return "I hit a processing limit while handling your request. Please try rephrasing your question."


def get_provider_diagnostics(config: ProviderConfig | None = None) -> dict[str, Any]:
    """Return non-sensitive provider configuration diagnostics."""
    cfg = config or ProviderConfig.from_env()
    return {
        "primary_provider": cfg.primary_provider,
        "primary_configured": (
            bool(cfg.groq_api_key) if cfg.primary_provider == "groq" else bool(cfg.gemini_api_key)
        ),
        "groq_model": cfg.groq_model,
        "fallback_enabled": cfg.fallback_enabled,
        "fallback_provider": cfg.fallback_provider,
        "fallback_configured": (
            bool(cfg.gemini_api_key) if cfg.fallback_provider == "gemini" else bool(cfg.groq_api_key)
        ),
        "gemini_model": cfg.gemini_model,
        "streaming_used": False,
    }
