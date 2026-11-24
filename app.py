import os
from dotenv import load_dotenv

# Load local .env before importing modules that may read environment variables.
# override=False ensures hosted platform environment variables take precedence.
load_dotenv(override=False)

import gradio as gr

from context import build_turn_system_prompt, get_active_knowledge_diagnostics
from input_guard import validate_and_classify_input
from providers import (
    GEMINI_DEFAULT_MODEL,
    GROQ_DEFAULT_MODEL,
    GROQ_DEFAULT_URL,
    MAX_TOOL_ITERATIONS,
    ProviderConfig,
    get_provider_diagnostics,
    run_chat_turn,
)
from tools import tools, handle_tool_calls

GROQ_URL = GROQ_DEFAULT_URL
MODEL = GROQ_DEFAULT_MODEL
GEMINI_MODEL = GEMINI_DEFAULT_MODEL


def get_server_config() -> tuple[str, int]:
    """Return (host, port) from environment variables with Render-friendly defaults."""
    host = os.getenv("HOST", "").strip() or "0.0.0.0"
    raw_port = os.getenv("PORT", "7860").strip()
    if not raw_port:
        return host, 7860
    try:
        port = int(raw_port)
    except ValueError:
        print("WARNING: Invalid PORT value; falling back to 7860.", flush=True)
        return host, 7860
    if not (1 <= port <= 65535):
        print("WARNING: Out-of-range PORT value; falling back to 7860.", flush=True)
        return host, 7860
    return host, port


def log_startup_diagnostics() -> None:
    """Log non-sensitive active knowledge metadata and provider readiness checks."""
    knowledge_meta = get_active_knowledge_diagnostics()
    print(
        f"Active knowledge: cv={knowledge_meta['active_cv_file']} "
        f"sha256={knowledge_meta['active_cv_sha256_prefix']} "
        f"public_chunks={knowledge_meta['public_chunk_count']}",
        flush=True,
    )

    provider_meta = get_provider_diagnostics()
    print(
        f"LLM config: primary={provider_meta['primary_provider']} "
        f"primary_configured={provider_meta['primary_configured']} "
        f"fallback_enabled={provider_meta['fallback_enabled']} "
        f"fallback={provider_meta['fallback_provider']} "
        f"fallback_configured={provider_meta['fallback_configured']}",
        flush=True,
    )
    if not provider_meta["primary_configured"]:
        print(
            f"WARNING: Primary provider ({provider_meta['primary_provider'].upper()}) API key is not set; "
            "Digital Twin chat will report unavailable until configured.",
            flush=True,
        )
    if not (os.getenv("TELEGRAM_BOT_TOKEN", "").strip() and os.getenv("TELEGRAM_CHAT_ID", "").strip()):
        print(
            "INFO: Telegram notification credentials are not set; Telegram alerts are disabled.",
            flush=True,
        )


def _extract_text(content) -> str:
    """Normalize Gradio message content (str or multimodal block list) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
        return "\n".join(p for p in parts if p)
    return str(content) if content is not None else ""


def _normalize_history(history) -> list[dict]:
    """Convert Gradio chat history into clean OpenAI-compatible role/content dicts."""
    clean = []
    for entry in history or []:
        if not isinstance(entry, dict):
            continue
        role = entry.get("role")
        if role not in ("user", "assistant"):
            continue
        text = _extract_text(entry.get("content"))
        if text:
            clean.append({"role": role, "content": text})
    return clean


def validate_user_message(message):
    """Reject empty or whitespace-only messages in Gradio before adding a chat bubble."""
    text = _extract_text(message).strip()
    if not text:
        return gr.validate(False, "Please type a message before sending.")
    return gr.validate(True, "")


def chat(message, history):
    """Handle a visitor turn using input validation, CV-only RAG, and Groq/Gemini providers."""
    text = _extract_text(message).strip()
    clean_history = _normalize_history(history)
    is_valid, rejection_msg = validate_and_classify_input(text, history=clean_history)
    if not is_valid:
        return rejection_msg

    system_prompt = build_turn_system_prompt(query=text, history=clean_history)
    config = ProviderConfig.from_env()

    return run_chat_turn(
        user_message=text,
        history=clean_history,
        system_prompt=system_prompt,
        tools=tools,
        tool_handler=handle_tool_calls,
        config=config,
    )


N8N_DARK_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.orange,
    neutral_hue=gr.themes.colors.slate,
).set(
    body_background_fill="#f7f8fb",
    body_background_fill_dark="#0e0f14",
    body_text_color="#181c26",
    body_text_color_dark="#e8eaf0",
    body_text_color_subdued="#525d73",
    body_text_color_subdued_dark="#9aa3b2",
    background_fill_primary="#ffffff",
    background_fill_primary_dark="#151821",
    background_fill_secondary="#f1f3f8",
    background_fill_secondary_dark="#1b1f2b",
    block_background_fill="#ffffff",
    block_background_fill_dark="#151821",
    panel_background_fill="#f1f3f8",
    panel_background_fill_dark="#1b1f2b",
    border_color_primary="#d6dce8",
    border_color_primary_dark="#2a3040",
    border_color_accent="#ff6d5a",
    border_color_accent_dark="#ff6d5a",
    color_accent_soft="#fff0ed",
    color_accent_soft_dark="#2a2024",
    code_background_fill="#eef1f6",
    code_background_fill_dark="#232838",
    table_even_background_fill="#ffffff",
    table_even_background_fill_dark="#151821",
    table_odd_background_fill="#f6f8fb",
    table_odd_background_fill_dark="#1b1f2b",
    table_border_color="#d6dce8",
    table_border_color_dark="#2a3040",
    input_background_fill="#ffffff",
    input_background_fill_dark="#1b1f2b",
    input_border_color="#cbd3e3",
    input_border_color_dark="#2e3547",
    input_border_color_focus="#ff6d5a",
    input_border_color_focus_dark="#ff6d5a",
    input_placeholder_color="#6b7280",
    input_placeholder_color_dark="#8b95a7",
    link_text_color="#d94836",
    link_text_color_dark="#ff8c7a",
    link_text_color_hover="#b83525",
    link_text_color_hover_dark="#ffaba0",
    link_text_color_visited="#c24130",
    link_text_color_visited_dark="#ff9e8f",
    button_primary_background_fill="#ff6d5a",
    button_primary_background_fill_dark="#ff6d5a",
    button_primary_background_fill_hover="#e85b48",
    button_primary_background_fill_hover_dark="#ff8270",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#ffffff",
    button_secondary_background_fill="#eef1f6",
    button_secondary_background_fill_dark="#232838",
    button_secondary_background_fill_hover="#e2e7f0",
    button_secondary_background_fill_hover_dark="#2e3547",
    button_secondary_border_color="#d6dce8",
    button_secondary_border_color_dark="#363e54",
    button_secondary_text_color="#181c26",
    button_secondary_text_color_dark="#e8eaf0",
)
N8N_THEME = N8N_DARK_THEME

BIDI_DARK_CSS = """
:root, html, body {
    color-scheme: light;
}
html.dark, body.dark, .dark {
    color-scheme: dark;
}
.message-content, .prose, .message {
    min-width: 0;
    max-width: 100%;
}
.prose [dir="rtl"] {
    direction: rtl;
    text-align: right;
    unicode-bidi: isolate;
}
.prose [dir="ltr"] {
    direction: ltr;
    text-align: left;
    unicode-bidi: isolate;
}
.prose :is(ul, ol) {
    padding-inline-start: 1.5em;
    padding-inline-end: 0;
    list-style-position: outside;
}
.prose li {
    text-align: start;
}
.prose :is(pre, code, a[href], bdi) {
    direction: ltr;
    unicode-bidi: isolate;
}
.prose code {
    background: var(--code-background-fill);
    color: var(--body-text-color);
    border-radius: 4px;
    padding: 0.1em 0.35em;
}
.prose pre code {
    background: transparent;
    padding: 0;
}
.prose table {
    width: 100%;
    table-layout: auto;
    border-collapse: collapse;
}
.prose th, .prose td {
    text-align: start;
    white-space: normal;
    overflow-wrap: break-word;
    word-break: normal;
}
.prose a[href] {
    white-space: normal;
    overflow-wrap: anywhere;
    word-break: break-word;
}
textarea {
    unicode-bidi: plaintext;
    text-align: start;
}
"""

BIDI_OBSERVER_JS = """
() => {
    const ARABIC_RE = /[\\u0600-\\u06FF\\u0750-\\u077F\\u08A0-\\u08FF\\uFB50-\\uFDFF\\uFE70-\\uFEFF]/g;
    const LATIN_RE = /[A-Za-z]/g;
    const STRONG_RE = /[\\u0600-\\u06FF\\u0750-\\u077F\\u08A0-\\u08FF\\uFB50-\\uFDFF\\uFE70-\\uFEFFA-Za-z]/;
    const BLOCK_SELECTOR = "p, ul, ol, li, blockquote, h1, h2, h3, h4, h5, h6, table, th, td";
    const LTR_RUN_RE = /(?:https?:\\/\\/[^\\s\\u0600-\\u06FF]+|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}|\\.?[A-Za-z0-9]+(?:[#+.\\/_-]+[A-Za-z0-9]+)*[#+-]*(?:\\s+[A-Za-z0-9][A-Za-z0-9._#+-]*)*)/g;
    let isMutatingBidi = false;

    function ownTextForDirection(el) {
        const clone = el.cloneNode(true);
        const tag = el.tagName;
        if (tag === "UL" || tag === "OL" || tag === "LI") {
            clone.querySelectorAll("ul, ol, pre, code, a[href]").forEach((n) => n.remove());
        } else {
            clone.querySelectorAll("pre, code, a[href]").forEach((n) => n.remove());
        }
        return (clone.textContent || "").trim();
    }

    function detectDirection(text) {
        if (!text) return null;
        const firstStrong = text.match(STRONG_RE);
        if (!firstStrong) return null;
        const arabicMatches = text.match(ARABIC_RE) || [];
        if (arabicMatches.length === 0) return "ltr";
        const firstIsArabic = /[\\u0600-\\u06FF\\u0750-\\u077F\\u08A0-\\u08FF\\uFB50-\\uFDFF\\uFE70-\\uFEFF]/.test(firstStrong[0]);
        if (firstIsArabic) return "rtl";
        const latinMatches = text.match(LATIN_RE) || [];
        const total = arabicMatches.length + latinMatches.length;
        return total > 0 && arabicMatches.length / total >= 0.25 ? "rtl" : "ltr";
    }

    function isolateInlineLtrSpans(root) {
        const rtlBlocks = Array.from(root.querySelectorAll ? root.querySelectorAll('.prose [dir="rtl"]') : []);
        if (root.matches && root.matches('.prose [dir="rtl"]')) {
            rtlBlocks.unshift(root);
        }
        for (const rtlEl of rtlBlocks) {
            const walker = document.createTreeWalker(rtlEl, NodeFilter.SHOW_TEXT);
            const textNodes = [];
            let cur;
            while ((cur = walker.nextNode())) {
                const parent = cur.parentElement;
                if (!parent) continue;
                if (parent.closest("pre, code, a[href], bdi, script, style, textarea, input")) continue;
                if (parent.closest('[dir="ltr"]')) continue;
                LTR_RUN_RE.lastIndex = 0;
                if (LTR_RUN_RE.test(cur.nodeValue || "")) {
                    textNodes.push(cur);
                }
            }
            for (const textNode of textNodes) {
                const value = textNode.nodeValue || "";
                LTR_RUN_RE.lastIndex = 0;
                let lastIndex = 0;
                let match;
                let wrapped = 0;
                const frag = document.createDocumentFragment();
                while ((match = LTR_RUN_RE.exec(value)) !== null) {
                    const start = match.index;
                    const rawToken = match[0];
                    const cleanToken = rawToken.replace(/[.,;:!?)]+$/, "");
                    const trailing = rawToken.slice(cleanToken.length);
                    if (!cleanToken) continue;
                    if (start > lastIndex) {
                        frag.appendChild(document.createTextNode(value.slice(lastIndex, start)));
                    }
                    const bdi = document.createElement("bdi");
                    bdi.setAttribute("dir", "ltr");
                    bdi.setAttribute("data-twin-ltr", "1");
                    bdi.textContent = cleanToken;
                    frag.appendChild(bdi);
                    if (trailing) {
                        frag.appendChild(document.createTextNode(trailing));
                    }
                    lastIndex = start + rawToken.length;
                    wrapped += 1;
                }
                if (wrapped > 0) {
                    if (lastIndex < value.length) {
                        frag.appendChild(document.createTextNode(value.slice(lastIndex)));
                    }
                    textNode.replaceWith(frag);
                }
            }
        }
    }

    function applyBidiToRoot(root) {
        if (!root || root.nodeType !== 1 || isMutatingBidi) return;
        isMutatingBidi = true;
        try {
            const blocks = Array.from(root.querySelectorAll(BLOCK_SELECTOR));
            if (root.matches && root.matches(BLOCK_SELECTOR)) {
                blocks.unshift(root);
            }

            for (const el of blocks) {
                if (!el.closest(".prose")) continue;
                const dir = detectDirection(ownTextForDirection(el));
                if (dir && el.getAttribute("dir") !== dir) {
                    el.setAttribute("dir", dir);
                }
            }
            const ltrElements = Array.from(
                root.querySelectorAll ? root.querySelectorAll(".prose pre, .prose code, .prose a[href], .prose bdi") : []
            );
            if (root.matches && root.matches(".prose pre, .prose code, .prose a[href], .prose bdi")) {
                ltrElements.unshift(root);
            }
            for (const el of ltrElements) {
                if (el.getAttribute("dir") !== "ltr") {
                    el.setAttribute("dir", "ltr");
                }
            }
            isolateInlineLtrSpans(root);
        } finally {
            if (window.__twinBidiObserver) {
                window.__twinBidiObserver.takeRecords();
            }
            isMutatingBidi = false;
        }
    }

    applyBidiToRoot(document.body);

    if (!window.__twinBidiObserver) {
        const observer = new MutationObserver((mutations) => {
            if (isMutatingBidi) return;
            for (const mutation of mutations) {
                if (mutation.type === "characterData" && mutation.target.parentElement) {
                    applyBidiToRoot(mutation.target.parentElement.closest(".prose") || mutation.target.parentElement);
                }
                for (const node of mutation.addedNodes) {
                    if (node.nodeType === 1 && !node.matches?.("bdi[data-twin-ltr]")) {
                        applyBidiToRoot(node);
                    }
                }
            }
        });
        window.__twinBidiObserver = observer;
        observer.observe(document.body, { childList: true, subtree: true, characterData: true });
    }

    if (!window.__twinEmptyInputGuard) {
        function findChatInput() {
            return document.querySelector('textarea[placeholder], textarea');
        }
        function findSubmitButton(textarea) {
            if (!textarea) return null;
            const container = textarea.closest('.block, .gradio-row, form') || textarea.parentElement?.parentElement;
            return container ? container.querySelector('button.submit-button, button[aria-label*="Submit" i], button[type="submit"]') : null;
        }
        function syncEmptySubmitState() {
            const ta = findChatInput();
            const btn = findSubmitButton(ta);
            if (!ta) return;
            if (ta.getAttribute('dir') !== 'auto') {
                ta.setAttribute('dir', 'auto');
            }
            if (!btn) return;
            const empty = ta.value.trim().length === 0;
            btn.disabled = empty;
            btn.setAttribute('aria-disabled', empty ? 'true' : 'false');
        }
        document.addEventListener('input', (e) => {
            if (e.target && e.target.tagName === 'TEXTAREA') {
                syncEmptySubmitState();
            }
        }, true);
        document.addEventListener('keydown', (e) => {
            if (e.target && e.target.tagName === 'TEXTAREA' && e.key === 'Enter' && !e.shiftKey) {
                if (e.target.value.trim().length === 0) {
                    e.preventDefault();
                    e.stopImmediatePropagation();
                }
            }
        }, true);
        document.addEventListener('click', (e) => {
            const ta = findChatInput();
            const btn = findSubmitButton(ta);
            if (btn && (e.target === btn || btn.contains(e.target)) && ta && ta.value.trim().length === 0) {
                e.preventDefault();
                e.stopImmediatePropagation();
            }
        }, true);
        syncEmptySubmitState();
        window.__twinEmptyInputGuard = true;
    }
}
"""


def create_chat_interface(chat_fn=chat) -> gr.ChatInterface:
    """Create the configured Gradio ChatInterface without launching a server."""
    return gr.ChatInterface(
        fn=chat_fn,
        chatbot=gr.Chatbot(
            show_label=False,
            buttons=["copy"],
            scale=1,
            autoscroll=True,
        ),
        validator=validate_user_message,
        title="Zyad Waled Talat — Digital Twin",
        description="Ask me about Zyad's AI & workflow automation work, projects, skills, or background (English & Arabic supported).",
        autofocus=False,
        examples=[
            "Tell me about Zyad's background, education, and experience.",
            "What projects has Zyad built?",
            "What technical skills and courses does Zyad have?",
            "How can I reach out to Zyad for a project or role?",
        ],
    )


demo = create_chat_interface(chat)

if __name__ == "__main__":
    log_startup_diagnostics()
    host, port = get_server_config()
    if host == "0.0.0.0":
        print(f"Open locally in your browser: http://127.0.0.1:{port}", flush=True)
    demo.launch(
        server_name=host,
        server_port=port,
        theme=N8N_DARK_THEME,
        css=BIDI_DARK_CSS,
        js=BIDI_OBSERVER_JS,
    )
