import argparse
import json
import os
from types import SimpleNamespace
from unittest.mock import patch
from dotenv import load_dotenv

from tools import handle_tool_calls, record_unknown_question, record_user_details


def run_smoke_test(live_telegram: bool = False) -> int:
    """Verify tool handlers offline by default; pass live_telegram=True for [TEST] Telegram alerts."""
    load_dotenv(override=False)

    if not live_telegram:
        sample_calls = [
            SimpleNamespace(
                id="dry_1",
                function=SimpleNamespace(
                    name="record_unknown_question",
                    arguments=json.dumps({"question": "[DRY-RUN] Does Zyad know Rust?"}),
                ),
            ),
            SimpleNamespace(
                id="dry_2",
                function=SimpleNamespace(
                    name="record_user_details",
                    arguments=json.dumps(
                        {
                            "email": "dryrun@example.com",
                            "name": "Dry Run User",
                            "notes": "Offline validation",
                        }
                    ),
                ),
            ),
        ]
        with patch(
            "tools.send_telegram_notification",
            return_value={"ok": True, "status": "confirmed"},
        ), patch(
            "tools.append_to_sheet_with_status",
            return_value={"status": "skipped", "reason": "dry_run"},
        ):
            results = handle_tool_calls(sample_calls)
        print("[DRY-RUN] Offline tool dispatch results:", [r["content"] for r in results])
        print("[DRY-RUN] No live Telegram messages or Google Sheets rows were written. Pass --live for Telegram.")
        return 0

    has_tg = bool(os.getenv("TELEGRAM_BOT_TOKEN", "").strip() and os.getenv("TELEGRAM_CHAT_ID", "").strip())
    if not has_tg:
        print("[SKIP] TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is not set.")
        return 1

    # Force [TEST] prefix and suppress production Sheets writes during tool smoke test.
    with patch.dict(os.environ, {"TELEGRAM_MESSAGE_PREFIX": "[TEST] "}, clear=False), patch(
        "tools.append_to_sheet_with_status",
        return_value={"status": "skipped", "reason": "suppressed_in_test_tools"},
    ):
        result1 = record_unknown_question("[TEST] Does Zyad know Rust?")
        result2 = record_user_details(
            email="test@example.com",
            name="[TEST] Test User",
            notes="[TEST] Tool smoke check (Sheets skipped)",
        )
    print("Live Telegram tool results (Sheets skipped):", result1, result2)
    return 0 if (result1.get("recorded") == "ok" and result2.get("recorded") == "ok") else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Tool dispatch smoke check (offline dry-run by default).")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Send [TEST]-prefixed Telegram alerts (never writes fake rows to production Sheets).",
    )
    args = parser.parse_args()
    return run_smoke_test(live_telegram=args.live)


if __name__ == "__main__":
    raise SystemExit(main())
