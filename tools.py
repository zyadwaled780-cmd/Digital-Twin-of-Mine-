from datetime import datetime, timezone
import json
import os
from pathlib import Path
import gspread
from google.oauth2.service_account import Credentials
import requests

TELEGRAM_TIMEOUT_SECONDS = 10
GOOGLE_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]
CONTACTS_HEADERS = ["timestamp", "name", "email", "notes"]
UNKNOWN_QUESTIONS_HEADERS = ["timestamp", "question"]


def _utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp string."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_google_sheets_config() -> tuple[str, str, str]:
    """Return (sheet_id, creds_json, creds_file) from environment variables.

    Precedence: if both GOOGLE_CREDENTIALS_JSON and GOOGLE_CREDENTIALS_FILE are set,
    GOOGLE_CREDENTIALS_JSON takes precedence (intended for hosted environments).
    """
    sheet_id = os.getenv("GOOGLE_SHEET_ID", "").strip()
    creds_json = os.getenv("GOOGLE_CREDENTIALS_JSON", "").strip()
    creds_file = os.getenv("GOOGLE_CREDENTIALS_FILE", "").strip()
    return sheet_id, creds_json, creds_file


def _get_gspread_client():
    """Authorize and return (gspread_client, sheet_id), or (None, None) if unconfigured."""
    sheet_id, creds_json, creds_file = get_google_sheets_config()

    if not sheet_id or (not creds_json and not creds_file):
        return None, None

    if creds_json:
        info = json.loads(creds_json)
        if not isinstance(info, dict):
            raise ValueError("GOOGLE_CREDENTIALS_JSON must decode to a JSON object")
        creds = Credentials.from_service_account_info(info, scopes=GOOGLE_SCOPES)
    else:
        creds_path = Path(creds_file)
        if not creds_path.is_file():
            raise FileNotFoundError("GOOGLE_CREDENTIALS_FILE does not exist")
        creds = Credentials.from_service_account_file(str(creds_path), scopes=GOOGLE_SCOPES)

    return gspread.authorize(creds), sheet_id


def _get_or_create_worksheet(spreadsheet, tab_name: str, min_cols: int):
    """Fetch a worksheet by name, creating it safely even under concurrent creation."""
    try:
        return spreadsheet.worksheet(tab_name)
    except gspread.WorksheetNotFound:
        try:
            return spreadsheet.add_worksheet(
                title=tab_name,
                rows=1000,
                cols=max(min_cols, 4),
            )
        except gspread.exceptions.APIError:
            return spreadsheet.worksheet(tab_name)


def append_to_sheet_with_status(tab_name: str, headers: list[str], row: list[str]) -> dict:
    """Append a row to Google Sheets and classify outcome as confirmed, skipped, uncertain, or failed.

    Timeout or connection drop after sending an append request returns status="uncertain"
    and is NEVER auto-retried to avoid duplicate non-idempotent row writes.
    """
    try:
        client, sheet_id = _get_gspread_client()
        if client is None or not sheet_id:
            print(f"Google Sheets logging skipped ({tab_name}): not configured", flush=True)
            return {"status": "skipped", "reason": "not_configured"}

        spreadsheet = client.open_by_key(sheet_id)
        worksheet = _get_or_create_worksheet(spreadsheet, tab_name, len(headers))

        existing_first_row = worksheet.row_values(1)
        if not existing_first_row and not worksheet.col_values(1):
            worksheet.append_row(headers, value_input_option="USER_ENTERED")

        worksheet.append_row(row, value_input_option="USER_ENTERED")
        print(f"Google Sheets logging succeeded ({tab_name})", flush=True)
        return {"status": "confirmed"}
    except (TimeoutError, requests.Timeout, ConnectionError, requests.ConnectionError) as exc:
        print(f"Google Sheets logging uncertain ({tab_name}): {type(exc).__name__}", flush=True)
        return {"status": "uncertain", "error": type(exc).__name__}
    except Exception as exc:
        # Log exception class only so credential or PII strings are never printed.
        print(f"Google Sheets logging failed ({tab_name}): {type(exc).__name__}", flush=True)
        return {"status": "failed", "error": type(exc).__name__}


def append_to_sheet(tab_name: str, headers: list[str], row: list[str]) -> bool:
    """Append a row to a Google Sheet tab and return True only when confirmed."""
    return append_to_sheet_with_status(tab_name, headers, row).get("status") == "confirmed"


def send_telegram_notification(message: str) -> dict:
    """Send a Telegram notification and return explicit confirmed/uncertain/failed/not_configured status.

    Never auto-retries on timeout or connection loss so non-idempotent alerts are not duplicated.
    """
    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not telegram_token or not telegram_chat_id:
        return {"ok": False, "status": "not_configured", "error": "Telegram credentials not configured"}

    prefix = os.getenv("TELEGRAM_MESSAGE_PREFIX", "")
    text = message if not prefix or message.startswith(prefix) else f"{prefix}{message}"
    url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
    payload = {"chat_id": telegram_chat_id, "text": text}
    try:
        response = requests.post(url, data=payload, timeout=TELEGRAM_TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()
        print("Telegram notification status:", response.status_code, flush=True)
        if isinstance(data, dict) and data.get("ok") is True:
            result = dict(data)
            result["status"] = "confirmed"
            return result
        return {"ok": False, "status": "failed", "error": "TelegramAPIRejected"}
    except (requests.Timeout, requests.ConnectionError, TimeoutError, ConnectionError) as exc:
        print(f"Telegram notification uncertain: {type(exc).__name__}", flush=True)
        return {"ok": False, "status": "uncertain", "error": type(exc).__name__}
    except (requests.RequestException, ValueError) as exc:
        print(f"Telegram notification failed: {type(exc).__name__}", flush=True)
        return {"ok": False, "status": "failed", "error": type(exc).__name__}


def _aggregate_record_outcome(telegram_result: dict, sheets_result: dict) -> dict:
    """Synthesize confirmed/uncertain/failed tool response without exaggerating delivery."""
    tg_status = telegram_result.get("status") or ("confirmed" if telegram_result.get("ok") else "failed")
    sh_status = sheets_result.get("status", "failed")

    if tg_status == "confirmed" or sh_status == "confirmed":
        return {
            "recorded": "ok",
            "status": "confirmed",
            "telegram": tg_status,
            "sheets": sh_status,
        }
    if tg_status == "uncertain" or sh_status == "uncertain":
        return {
            "recorded": "uncertain",
            "status": "uncertain",
            "telegram": tg_status,
            "sheets": sh_status,
        }
    return {
        "recorded": "failed",
        "status": "failed",
        "telegram": tg_status,
        "sheets": sh_status,
        "reason": telegram_result.get("error", "Notification and logging unavailable"),
    }


def record_user_details(
    email: str,
    name: str | None = None,
    notes: str | None = None,
) -> dict:
    """Record a visitor's contact request via Telegram and best-effort Google Sheets."""
    clean_email = (email or "").strip()
    if not clean_email or "@" not in clean_email or "." not in clean_email.split("@", 1)[-1] or " " in clean_email:
        return {
            "recorded": "failed",
            "status": "failed",
            "error": "Please provide a valid email address.",
        }

    clean_name = (name or "").strip() or "Not provided"
    clean_notes = (notes or "").strip() or "Not provided"

    telegram_result = send_telegram_notification(
        f"📩 New portfolio inquiry for Zyad:\nName: {clean_name}\nEmail: {clean_email}\nNotes: {clean_notes}"
    )
    sheets_result = append_to_sheet_with_status(
        tab_name="contacts",
        headers=CONTACTS_HEADERS,
        row=[_utc_timestamp(), clean_name, clean_email, clean_notes],
    )
    return _aggregate_record_outcome(telegram_result, sheets_result)


def record_unknown_question(question: str) -> dict:
    """Record an unanswered visitor question via Telegram and best-effort Google Sheets."""
    telegram_result = send_telegram_notification(
        f"❓ Unanswered portfolio question:\n{question}"
    )
    sheets_result = append_to_sheet_with_status(
        tab_name="unknown_questions",
        headers=UNKNOWN_QUESTIONS_HEADERS,
        row=[_utc_timestamp(), question],
    )
    return _aggregate_record_outcome(telegram_result, sheets_result)


record_user_details_json = {
    "name": "record_user_details",
    "description": (
        "Use this when a visitor wants Zyad to contact them or follow up on a question. "
        "Collect both the visitor's name and valid email before calling when possible; "
        "pass 'Not provided' if the visitor declined to share a name."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "email": {"type": "string", "description": "Visitor's valid email address"},
            "name": {
                "type": ["string", "null"],
                "description": "Visitor's full or first name (use 'Not provided' if unavailable)",
            },
            "notes": {
                "type": ["string", "null"],
                "description": "Summary of the visitor's question, request, or topic for Zyad",
            },
        },
        "required": ["email"],
    },
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Use this ONLY when a visitor asks a genuine question about Zyad (such as an unlisted skill, missing qualification, or undocumented profile fact) that is not answered in the retrieved evidence. Never call for greetings, generic small talk, or off-topic questions.",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "The specific question about Zyad not covered in the CV/context"},
        },
        "required": ["question"],
    },
}

tools = [
    {"type": "function", "function": record_user_details_json},
    {"type": "function", "function": record_unknown_question_json},
]

TOOL_HANDLERS = {
    "record_user_details": record_user_details,
    "record_unknown_question": record_unknown_question,
}


def handle_tool_calls(tool_calls, dedupe_cache: dict[tuple[str, str], dict] | None = None) -> list[dict]:
    """Execute model-requested tool calls safely with optional in-memory per-turn deduplication."""
    results = []
    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        raw_args = tool_call.function.arguments or "{}"
        try:
            arguments = json.loads(raw_args)
        except json.JSONDecodeError:
            result = {"error": f"Invalid JSON arguments for tool '{tool_name}'"}
            results.append(
                {"role": "tool", "content": json.dumps(result), "tool_call_id": tool_call.id}
            )
            continue

        if not isinstance(arguments, dict):
            result = {"error": f"Arguments for tool '{tool_name}' must be a JSON object"}
            results.append(
                {"role": "tool", "content": json.dumps(result), "tool_call_id": tool_call.id}
            )
            continue

        cache_key = (tool_name, json.dumps(arguments, sort_keys=True, ensure_ascii=False))
        if dedupe_cache is not None and cache_key in dedupe_cache:
            results.append(
                {
                    "role": "tool",
                    "content": json.dumps(dedupe_cache[cache_key]),
                    "tool_call_id": tool_call.id,
                }
            )
            continue

        print(f"Tool called: {tool_name}", flush=True)
        handler = TOOL_HANDLERS.get(tool_name)
        if handler is None:
            result = {"error": f"Unknown tool: {tool_name}"}
        else:
            try:
                result = handler(**arguments)
                if dedupe_cache is not None:
                    dedupe_cache[cache_key] = result
            except TypeError as exc:
                result = {"error": f"Invalid arguments for '{tool_name}': {exc}"}

        results.append(
            {"role": "tool", "content": json.dumps(result), "tool_call_id": tool_call.id}
        )
    return results
