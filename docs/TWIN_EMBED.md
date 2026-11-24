# Portfolio Handoff: Embedding or Linking the Digital Twin

This document describes how the separate n8n-style portfolio frontend connects to the deployed Digital Twin Gradio service.

## 1. Production URL Placeholder
Replace `https://YOUR-SERVICE-NAME.onrender.com` with your live Render web service URL:

```text
https://YOUR-SERVICE-NAME.onrender.com
```

## 2. Responsive `<iframe>` Embed Example
Use a responsive wrapper so the iframe adapts cleanly across desktop drawers/modals and mobile screens without horizontal overflow:

```html
<div class="twin-frame-shell">
  <iframe
    id="digital-twin-frame"
    src="https://YOUR-SERVICE-NAME.onrender.com"
    title="Zyad Waled Talat — AI Digital Twin Chat"
    loading="lazy"
    allow="clipboard-write"
    referrerpolicy="no-referrer-when-downgrade"
    style="width: 100%; height: 100%; border: 0; display: block;"
  ></iframe>
</div>
```

Recommended responsive CSS for the portfolio container:

```css
.twin-frame-shell {
  width: min(100%, 460px);
  height: min(80dvh, 660px);
  min-height: 480px;
  border-radius: 12px;
  overflow: hidden;
  background: #0e0f14;
}

@media (max-width: 600px) {
  .twin-frame-shell {
    width: 100%;
    height: min(82dvh, 640px);
    min-height: 420px;
  }
}
```

- **Mobile sizing & scrolling behavior:** Verified locally at `390x844` (direct) and in a `400x600` iframe harness with zero horizontal overflow (`scrollWidth <= innerWidth`). Because the Gradio layout height exceeds `600px` when example prompts are displayed, the iframe scrolls vertically inside its own viewport; `autofocus=False` is set in `app.py` so the iframe stays scrolled at the top (`<h1>` title visible) on initial load rather than jumping to the bottom input box.

## 3. New-Tab Fallback Link
Always pair the iframe with a visible fallback link so visitors on strict privacy shields, blocked third-party contexts, or slow connections can open the twin directly:

```html
<a
  href="https://YOUR-SERVICE-NAME.onrender.com"
  target="_blank"
  rel="noopener noreferrer"
>
  Open Digital Twin in a new tab ↗
</a>
```

## 4. Cold-Start Loading, Failure Handling & Provider Readiness
1. **Cold-start loading state (~30–60s on Render Free Tier):** Show a loading overlay/skeleton inside `.twin-frame-shell` until the iframe's `load` event fires (e.g., *"Waking up the Digital Twin… First load can take up to 60 seconds."*).
2. **Timeout / failure fallback:** If `load` does not fire within ~15–20 seconds (or `error` triggers), keep waiting in the background while highlighting the **"Open Digital Twin in a new tab"** fallback link and letting the user retry reloading the frame.
3. **Optional warm-up ping:** When the visitor clicks the portfolio's manual trigger node, optionally send a non-blocking `fetch("https://YOUR-SERVICE-NAME.onrender.com", { mode: "no-cors" }).catch(() => {})` so the Render container begins waking before the user reaches the chat node.
4. **Important — Iframe `load` does NOT prove backend/provider readiness:** Even if the Gradio HTML loads with HTTP `200`, the Groq primary (or optional Gemini fallback) backend could still be unconfigured or experiencing an upstream outage (in which case the twin replies `"The Digital Twin is temporarily unavailable..."`). Only a real chat round-trip confirms end-to-end AI readiness.

## 5. Iframe Restrictions Verified Locally
- **Verified in local `http://127.0.0.1:8766` -> `http://127.0.0.1:7861` harness (and earlier `7860` harness):** Gradio did **not** block framing via `X-Frame-Options` or `Content-Security-Policy: frame-ancestors`, and `autofocus=False` eliminated the browser's `Blocked autofocusing in a cross-origin subframe` warning.
- **Gradio `6.28.0` upstream caveats:**
  - `BrowserState` may log `Error reading from localStorage: SyntaxError: Unexpected end of JSON input` if an existing browser profile holds stale `_saved_conversations_0` ciphertext from a previous server restart on the same origin (harmless fallback to `[]`).
  - On narrow screens (`~390px`), `gr.Chatbot(buttons=["copy"])` removes the `"share"` and `"copy_all"` header overlays; Gradio's scroll-to-bottom button may still appear near the bottom center during long scrolled threads.
- **Not yet verified against Render HTTPS or the real portfolio repo:** Exact response headers on `*.onrender.com` and embedding inside the actual portfolio origin remain **NOT VERIFIED** until deployed and tested inside the portfolio codebase.

## 6. Checklist for Testing in the Real Portfolio Repository
1. Replace `https://YOUR-SERVICE-NAME.onrender.com` with the live Render URL.
2. Verify the iframe loads inside the portfolio on desktop (`>=1024px`) and mobile (`~390px`) with no horizontal scroll or clipped controls.
3. Confirm the iframe opens scrolled to the top (`Zyad Waled Talat — Digital Twin` title visible) and does not steal focus from the portfolio workflow.
4. Test the cold-start loading overlay and the 15s fallback link (`Open Digital Twin in a new tab`).
5. Send one real English message and one Arabic message inside the embedded iframe to confirm end-to-end Groq responses, copy/scroll behavior, and readable links/lists.
6. Check browser DevTools Console/Network for any cross-origin, CSP, or cookie warnings on the live portfolio domain.
