# Deploying Zyad's Digital Twin on Render (Free Web Service)

This app runs purely from environment variables (`load_dotenv(override=False)` is a safe no-op when `.env` is absent), resolves `knowledge/` assets relative to the application root regardless of working directory, and binds to `0.0.0.0:$PORT` (defaulting to `7860` locally).

## 1. Pre-Push Security & Privacy Check
1. Confirm `.env` and any service-account `.json` files are ignored by `.gitignore` before initializing Git or committing.
2. Confirm no secrets appear in `render.yaml`, `app.py`, `providers.py`, `tools.py`, `knowledge/index.json`, or logs.
3. **Private contact info status:** Private phone and email lines were removed from `knowledge/cv_current.md` and `knowledge/index.json` redacts private contact patterns by default. Legacy `cv.pdf` and `cv.original.pdf` are ignored in `.gitignore` so personal phone/email never enter the public GitHub repository.
4. **Gemini fallback privacy note:** If you enable `LLM_FALLBACK_ENABLED=true`, visitor prompts and retrieved profile evidence will be sent to Google Gemini whenever Groq falls back. Keep `LLM_FALLBACK_ENABLED=false` unless you want cross-provider fallback active.
5. **Production execution mode:** `app.py` uses non-streaming provider calls (`chat()` returns complete responses); tool deduplication (`TurnToolExecutor`) is in-memory per turn only and not durable across process restarts.

## 2. Render Web Service Configuration
Deploy via [render.yaml](render.yaml) (Blueprint) or configure a Web Service in the Render Dashboard:
- **Runtime:** `Python`
- **Plan:** `Free` (`0.1 CPU`, `512 MB RAM`; spins down after ~15 minutes of inactivity)
- **Build Command:**
  ```bash
  pip install -r requirements.txt
  ```
- **Start Command:**
  ```bash
  python app.py
  ```
- **Health Check Path:** `/` (Gradio serves HTTP `200` on `/` and `/config`)

## 3. Environment Variables
Configure these under **Environment → Environment Variables** in Render (never commit values):

| Variable | Required | Default / Notes |
| :--- | :--- | :--- |
| `GROQ_API_KEY` | Yes (for primary Groq) | Groq API key; if missing, UI renders and returns an unavailable notice rather than fabricating replies |
| `GROQ_MODEL` | Optional | `openai/gpt-oss-120b` |
| `LLM_PRIMARY_PROVIDER` | Optional | `groq` (set to `gemini` for explicit Gemini-only testing) |
| `LLM_FALLBACK_PROVIDER` | Optional | `gemini` |
| `LLM_FALLBACK_ENABLED` | Optional | `false` (must be `true` to enable fallback) |
| `LLM_FALLBACK_ON_AUTH_ERROR` | Optional | `false` (by default, auth/config errors fail loudly rather than falling back silently) |
| `GEMINI_API_KEY` | Optional | Required only when Gemini is primary or fallback is enabled |
| `GEMINI_MODEL` | Optional | `gemini-4-hydrogen` |
| `TELEGRAM_BOT_TOKEN` | Optional | Bot token for lead/unknown-question alerts |
| `TELEGRAM_CHAT_ID` | Optional | Owner Telegram chat ID |
| `GOOGLE_SHEET_ID` | Optional | Spreadsheet ID from the Google Sheets URL |
| `GOOGLE_CREDENTIALS_JSON` | Optional | Service-account JSON key as a single-line string (takes precedence over `GOOGLE_CREDENTIALS_FILE`) |

> Render injects `PORT` dynamically (commonly `10000`). Do not hardcode `PORT` unless intentionally overriding it.

## 4. Deployment Verification Checklist
*(Render deployment is **NOT VERIFIED** until executed against a live Render service.)*
1. Confirm `GROQ_API_KEY` (and optional `GEMINI_API_KEY` / Telegram / Google Sheets variables) are set in Render.
2. Check startup logs for `Active knowledge: cv=knowledge/cv_current.md sha256=...`, `LLM config: ...`, and `Running on local URL:  http://0.0.0.0:<PORT>`.
3. Open `https://YOUR-SERVICE-NAME.onrender.com/` (and `/config`) and confirm HTTP `200` and the dark Gradio UI load cleanly.
4. Test one live English question, one Egyptian Arabic question, and a follow-up question grounded in the active CV.
5. Run one explicitly authorized `[TEST]` tool check (contact or unknown question) if Telegram/Sheets are enabled, then remove any test row from Google Sheets.
6. Inspect Render logs to verify zero leaked credentials or PII.

## 5. Cold Starts on Free Tier
Render's free web service spins down after ~15 minutes of inactivity and typically takes **~30–60 seconds** to wake up on the first request. See [docs/TWIN_EMBED.md](docs/TWIN_EMBED.md).

---

## 6. Alternative: Deploying on Vercel (Free Hobby Tier, No Credit Card)

This repository also includes a Gradio-free Python serverless API and standalone/embeddable Light/Dark bilingual web UI configured for Vercel's free Hobby plan:

- **Static UI:** [public/index.html](public/index.html) (served at `/`, reads `?__theme=light|dark`, supports English + Egyptian Arabic RTL/LTR `<bdi dir="ltr">` isolation, and works both standalone and inside an `<iframe>`).
- **Python Serverless Functions:** [api/chat.py](api/chat.py) (`POST /api/chat`) and [api/health.py](api/health.py) (`GET /api/health`), backed by [twin_service.py](twin_service.py).
- **Packaging & Config:** [pyproject.toml](pyproject.toml) (excludes `gradio` on Vercel so serverless bundles stay small) and [vercel.json](vercel.json) (`framework: null`, `outputDirectory: "public"`, `frame-ancestors` CSP allowing `https://zyadwaled780-cmd.github.io`), plus [.vercelignore](.vercelignore).

### Vercel Setup Steps
1. Sign in at `https://vercel.com/new` with GitHub on the **Hobby (Free)** tier (no credit card required).
2. Import **`zyadwaled780-cmd/Digital-Twin-of-Mine-`** (`main` branch).
3. Keep the detected project settings (`vercel.json` + `pyproject.toml` configure `public/` and `api/*.py`).
4. Under **Environment Variables**, add secrets (never commit values):
   - `GROQ_API_KEY` *(required)*
   - `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` *(optional — Telegram alerts)*
   - `GEMINI_API_KEY` and `LLM_FALLBACK_ENABLED` = `true` *(optional — Gemini fallback)*
   - `GOOGLE_SHEET_ID` and `GOOGLE_CREDENTIALS_JSON` *(optional — Google Sheets)*
5. Click **Deploy**, wait for completion, verify `GET /api/health` (`200`) and live English + Arabic chats on your `https://YOUR-PROJECT.vercel.app` URL, then configure `DIGITAL_TWIN_CONFIG.liveUrl` in the portfolio repository.
