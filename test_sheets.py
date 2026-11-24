import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
from dotenv import load_dotenv

from tools import (
    CONTACTS_HEADERS,
    UNKNOWN_QUESTIONS_HEADERS,
    append_to_sheet,
    get_google_sheets_config,
)


def run_sheets_smoke_test(live: bool = False) -> int:
    """Validate Google Sheets config offline by default; pass live=True to write tagged test rows."""
    load_dotenv(override=False)

    sheet_id, creds_json, creds_file = get_google_sheets_config()
    test_sheet_id = os.getenv("GOOGLE_TEST_SHEET_ID", "").strip() or sheet_id
    cred_source = (
        "GOOGLE_CREDENTIALS_JSON"
        if creds_json
        else ("GOOGLE_CREDENTIALS_FILE" if creds_file else "none")
    )
    file_exists = bool(creds_file and Path(creds_file).is_file())

    print(
        f"Google Sheets config: target_sheet_set={bool(test_sheet_id)}, "
        f"using_test_sheet_override={bool(os.getenv('GOOGLE_TEST_SHEET_ID', '').strip())}, "
        f"credential_source={cred_source}, file_exists={file_exists}"
    )

    if not live:
        print(
            "[DRY-RUN] No Google Sheets API calls made. "
            "Pass --live to write [TEST]-tagged rows into 'test_contacts' and 'test_unknown_questions'."
        )
        return 0

    if not test_sheet_id or cred_source == "none":
        print("[SKIP] Missing GOOGLE_TEST_SHEET_ID/GOOGLE_SHEET_ID or Google credentials.")
        return 1
    if cred_source == "GOOGLE_CREDENTIALS_FILE" and not file_exists:
        print("[SKIP] GOOGLE_CREDENTIALS_FILE points to a missing file.")
        return 1

    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    tag = f"[TEST {ts}]"

    # Always use GOOGLE_TEST_SHEET_ID if provided and write only to dedicated test tabs.
    original_sheet_id = os.environ.get("GOOGLE_SHEET_ID")
    os.environ["GOOGLE_SHEET_ID"] = test_sheet_id
    try:
        ok_contact = append_to_sheet(
            tab_name="test_contacts",
            headers=CONTACTS_HEADERS,
            row=[ts, f"{tag} Audit Tester", "test@example.com", f"{tag} Safe smoke test"],
        )
        ok_question = append_to_sheet(
            tab_name="test_unknown_questions",
            headers=UNKNOWN_QUESTIONS_HEADERS,
            row=[ts, f"{tag} Smoke test unknown question"],
        )
    finally:
        if original_sheet_id is None:
            os.environ.pop("GOOGLE_SHEET_ID", None)
        else:
            os.environ["GOOGLE_SHEET_ID"] = original_sheet_id

    if ok_contact and ok_question:
        print(
            f"Live Sheets smoke test succeeded. Check tabs 'test_contacts' and 'test_unknown_questions' "
            f"for rows tagged '{tag}' and delete those test rows/tabs after verification."
        )
        return 0

    print("[ERROR] Live Sheets smoke test failed. See console log above for exception type.")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Google Sheets smoke check (offline dry-run by default).")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Write uniquely tagged [TEST] rows to test_contacts and test_unknown_questions tabs.",
    )
    args = parser.parse_args()
    return run_sheets_smoke_test(live=args.live)


if __name__ == "__main__":
    raise SystemExit(main())
