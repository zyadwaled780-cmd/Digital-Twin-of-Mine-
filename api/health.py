from __future__ import annotations

from http.server import BaseHTTPRequestHandler
from pathlib import Path
import sys

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

load_dotenv(override=False)

from twin_service import get_health_payload, send_json_response, send_options_response

ALLOWED_METHODS = "GET, OPTIONS"


class handler(BaseHTTPRequestHandler):
    """Vercel Python serverless handler for non-secret Digital Twin diagnostics."""

    def do_OPTIONS(self) -> None:
        send_options_response(self, ALLOWED_METHODS)

    def do_GET(self) -> None:
        try:
            payload = get_health_payload()
            send_json_response(self, 200, payload, methods=ALLOWED_METHODS)
        except Exception:
            send_json_response(
                self,
                500,
                {"status": "error", "error": "Unable to read Digital Twin diagnostics."},
                methods=ALLOWED_METHODS,
            )

    def log_message(self, format: str, *args: object) -> None:
        return
