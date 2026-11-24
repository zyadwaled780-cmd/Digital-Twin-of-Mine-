# Zyad Waled Talat — AI Digital Twin

An AI-powered digital twin chatbot that helps visitors explore Zyad Waled Talat's background, skills, and projects through conversation.

Built with **Python**, **Gradio** (`ChatInterface`), **Groq** (primary LLM) + opt-in **Google Gemini** (fallback), **Telegram** notifications, and optional **Google Sheets** logging.

## Supported Features

- **Strict English & Egyptian Arabic Replies:** Responds in English or Egyptian Arabic matching the visitor's message, keeps Arabic responses in Arabic even after English tool calls (`record_unknown_question`, `record_user_details`), spells the owner's Arabic name consistently (`زياد وليد طلعت`), and keeps URLs/technical tokens strictly ASCII.
- **Mixed RTL/LTR Isolation & Accessibility:** Automatic block directionality (`dir="rtl"` / `dir="ltr"` via `MutationObserver`) plus inline `<bdi dir="ltr">` wrapping for technical tokens (`C#`, `C++`, `2025/2026`, emails, GitHub URLs) inside Arabic messages.
- **High-Contrast Light & n8n Dark Themes:** Full paired Light Mode and Dark Mode token palettes with readable contrast on chat bubbles, code badges, tables, links, and buttons.
- **Strict CV-Only Grounded Retrieval (RAG):** Retrieves section- and item-level evidence exclusively from the active CV ([knowledge/cv_current.md](knowledge/cv_current.md) compiled into [knowledge/index.json](knowledge/index.json) via `ingest_cv.py`), with bilingual keyword expansion, Arabic morphology normalization, and typo tolerance.
- **Empty-Input & Nonsense Protection:** Client-side submit button guard, Gradio server-side `validator`, and pre-LLM [input_guard.py](input_guard.py) prevent empty/whitespace submissions and obvious gibberish from calling LLMs or external tools.
- **Visitor Follow-Up & Question Logging:** Collects visitor name and email for inquiries (`record_user_details`) and logs genuine unanswered CV questions (`record_unknown_question`) via Telegram alerts and optional Google Sheets.
- **Responsive & Embed-Ready Layout:** Tested across desktop, narrow mobile (`~390px`), and portfolio `<iframe>` integration ([docs/TWIN_EMBED.md](docs/TWIN_EMBED.md)).

## Knowledge & Privacy Workflow

- **Active CV:** [knowledge/cv_current.md](knowledge/cv_current.md) (referenced by [knowledge/active_source.json](knowledge/active_source.json)).
- **Private Contact Hygiene:** Personal phone and email lines are removed from tracked [knowledge/cv_current.md](knowledge/cv_current.md) and redacted by default in [knowledge/index.json](knowledge/index.json). Raw local backups (`cv.pdf`, `cv.original.pdf`) and `.env` are excluded via [.gitignore](.gitignore).
- **Supplemental Notes:** [knowledge/supplemental_profile.json](knowledge/supplemental_profile.json) holds quarantined legacy context; `CV_ONLY_MODE` defaults to `true` so answers stay strictly grounded in the active CV.

To update the active CV and atomically rebuild [knowledge/index.json](knowledge/index.json):

```bash
python ingest_cv.py --source knowledge/cv_current.md --set-active
```

## Local Setup

1. Install pinned dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Copy [.env.example](.env.example) to `.env` and fill in your credentials locally (never commit `.env`):
   - **Primary LLM (required for live chat):** `GROQ_API_KEY` (default `GROQ_MODEL=openai/gpt-oss-120b`)
   - **Optional Fallback LLM:** `GEMINI_API_KEY`, `GEMINI_MODEL=gemini-4-hydrogen`, `LLM_FALLBACK_ENABLED=true` *(disabled by default; when enabled, visitor queries and retrieved CV chunks are sent to Gemini if Groq hits eligible errors)*
   - **Optional Telegram Alerts:** `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`
   - **Optional Google Sheets Logging:** `GOOGLE_SHEET_ID`, `GOOGLE_CREDENTIALS_JSON` or `GOOGLE_CREDENTIALS_FILE`
3. Start the Gradio app:
   ```bash
   python app.py
   ```
   The server binds to `0.0.0.0:$PORT` (default `7860`) and logs `Open locally in your browser: http://127.0.0.1:7860`.

## Testing & Verification

Run the offline unit & regression test suite (zero live API calls):

```bash
python -B -m unittest test_unit.py -v
```

Standalone smoke scripts default to **offline dry-run mode** and require `--live` to call external services:
- `python test.py [--provider auto|groq|gemini] [--live]`
- `python test_telegram.py [--live]` (uses `[TEST]` prefix)
- `python test_tools.py [--live]` (uses `[TEST]` prefix and skips production Sheets)
- `python test_sheets.py [--live]` (writes `[TEST]`-tagged rows only to `test_contacts` / `test_unknown_questions`)
- `python test_ui_mock.py` (launches offline Gradio UI test harness on `http://127.0.0.1:7861`)

## Deployment & Portfolio Integration

- **Render Blueprint / Web Service:** Defined in [render.yaml](render.yaml) (`buildCommand: pip install -r requirements.txt`, `startCommand: python app.py`, `healthCheckPath: /`). Full environment variable and deployment checklist in [DEPLOY.md](DEPLOY.md).
- **Portfolio Integration Guide:** See [docs/TWIN_EMBED.md](docs/TWIN_EMBED.md).
