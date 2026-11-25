from __future__ import annotations

from http.server import BaseHTTPRequestHandler
from pathlib import Path
import sys

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

load_dotenv(override=False)

from twin_service import (
    PayloadTooLargeError,
    process_chat_turn,
    read_json_object_body,
    send_json_response,
    send_options_response,
)

ALLOWED_METHODS = "POST, OPTIONS"


class handler(BaseHTTPRequestHandler):
    """Vercel Python serverless handler for Digital Twin chat turns."""

    def do_OPTIONS(self) -> None:
        send_options_response(self, ALLOWED_METHODS)

    def do_POST(self) -> None:
        try:
            body = read_json_object_body(self)
        except PayloadTooLargeError as exc:
            send_json_response(self, 413, {"error": str(exc)}, methods=ALLOWED_METHODS)
            return
        except ValueError as exc:
            send_json_response(self, 400, {"error": str(exc)}, methods=ALLOWED_METHODS)
            return

        message = body.get("message")
        history = body.get("history")
        if history is not None and not isinstance(history, list):
            send_json_response(
                self,
                400,
                {"error": "Field 'history' must be a list of chat messages."},
                methods=ALLOWED_METHODS,
            )
            return

        try:
            result = process_chat_turn(message, history=history)
        except Exception:
            send_json_response(
                self,
                500,
                {"error": "The Digital Twin encountered an unexpected error. Please try again."},
                methods=ALLOWED_METHODS,
            )
            return

        if result.get("empty"):
            send_json_response(
                self,
                400,
                {"error": result.get("error", "Please type a message before sending."), "rejected": True},
                methods=ALLOWED_METHODS,
            )
            return

        send_json_response(
            self,
            200,
            {"reply": result.get("reply", ""), "rejected": bool(result.get("rejected", False))},
            methods=ALLOWED_METHODS,
        )

    def log_message(self, format: str, *args: object) -> None:
        return
