import argparse
import os
from dotenv import load_dotenv
from tools import send_telegram_notification


def run_smoke_test(live: bool = False) -> int:
    """Verify Telegram config offline by default; pass live=True to send a [TEST] message."""
    load_dotenv(override=False)

    has_token = bool(os.getenv("TELEGRAM_BOT_TOKEN", "").strip())
    has_chat_id = bool(os.getenv("TELEGRAM_CHAT_ID", "").strip())
    print(f"Telegram configured: token={has_token}, chat_id={has_chat_id}")

    message = "[TEST] Digital Twin Telegram smoke check"
    if not live:
        print(f"[DRY-RUN] Would send Telegram message: {message!r}. Pass --live to send.")
        return 0

    if not (has_token and has_chat_id):
        print("[SKIP] TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is not set.")
        return 1

    result = send_telegram_notification(message)
    ok = bool(result.get("ok"))
    print(f"Telegram response ok={ok}")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Telegram smoke check (offline dry-run by default).")
    parser.add_argument("--live", action="store_true", help="Send a real [TEST] message to Telegram.")
    args = parser.parse_args()
    return run_smoke_test(live=args.live)


if __name__ == "__main__":
    raise SystemExit(main())
