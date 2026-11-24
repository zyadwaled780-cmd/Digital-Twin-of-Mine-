from __future__ import annotations

import re
from typing import Any

EMAIL_INLINE_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
URL_INLINE_RE = re.compile(r"(?:https?://|github\.com/|linkedin\.com/)\S+", re.IGNORECASE)
ARABIC_CHAR_RE = re.compile(r"[\u0600-\u06FF]")
WORD_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
CJK_HANGUL_KANA_RE = re.compile(r"[\u4E00-\u9FFF\u3040-\u30FF\uAC00-\uD7AF]")
REPEATED_CHAR_RE = re.compile(r"(.)\1{3,}")

ALLOWED_SHORT_TECH_AND_GREETINGS = {
    "hi", "hey", "hello", "hola", "yo", "ok", "okay", "yes", "no", "skip", "thanks", "thx",
    "bye", "cv", "ai", "ml", "ui", "ux", "sql", "rag", "n8n", "api", "llm", "llms", "git",
    "gpa", "bis", "depi", "gci", "php", "css", "html", "http", "https", "json", "xml",
    "csv", "pdf", "npm", "jsx", "tsx", "grpc", "aws", "gcp", "go", "rust", "dart", "java",
    "utest", "c", "cpp", "csharp", "react", "node", "next", "vue", "flask", "django",
    "docker", "ollama", "python", "flutter", "supabase", "fastapi", "gradio", "test", "help",
    "سلام", "السلام", "اهلا", "أهلا", "اهلين", "مرحبا", "ازيك", "إزيك", "عامل", "ايه", "إيه",
    "شكرا", "شكرًا", "تمام", "ماشي", "نعم", "لا", "ايوه", "أيوه", "حاضر", "مين", "فين", "امتى", "إمتى",
    "كم", "كام", "هل", "عن", "من", "ما", "هو", "هي", "ليه", "ازاي", "إزاي",
}

ALLOWED_CV_NUMBERS = {"2024", "2025", "2026", "2028", "3.4", "3", "4"}

KEYBOARD_SMASH_SUBSTRINGS = (
    "asdf", "sdfg", "dfgh", "fghj", "ghjk", "hjkl",
    "qwer", "wert", "erty", "rtyu", "tyui", "yuio", "uiop",
    "zxcv", "xcvb", "cvbn", "vbnm",
    "qazwsx", "wsxedc", "edcrfv",
    "ضصثق", "صثقف", "ثقفغ", "قفغع", "فغعه", "غعهخ", "عهخح",
    "هعغف", "عغفق", "غفقث", "فقثص", "قثصض",
    "شسيب", "سيبل", "يبلا", "بلات", "لاتن", "اتنم", "تنمك",
    "منتب", "نتبل", "تبلا", "بلاي", "لايس", "ايسش",
    "ئءؤر", "ءؤرئ", "ؤرلا", "رلاى", "لاىة",
    "زوةى", "وةىر", "ةىرؤ",
)

CONTACT_PROMPT_MARKERS = (
    "email", "name", "contact", "reach you", "reach out", "follow up",
    "إيميل", "ايميل", "البريد", "اسمك", "الاسم", "تواصل", "يتواصل",
)


def _rejection_message(text: str) -> str:
    if ARABIC_CHAR_RE.search(text):
        return (
            "من فضلك اكتب سؤالك أو رسالتك بشكل واضح عن زياد أو مهاراته أو مشاريعه عشان أقدر أساعدك 🙂"
        )
    return (
        "Please enter a clear question or message about Zyad, his skills, or his projects so I can help. 🙂"
    )


def _is_after_contact_prompt(history: list[dict[str, Any]] | None) -> bool:
    for entry in reversed(history or []):
        if isinstance(entry, dict) and entry.get("role") == "assistant":
            content = str(entry.get("content", "")).lower()
            return any(marker in content for marker in CONTACT_PROMPT_MARKERS)
    return False


def _is_keyboard_smash(token_lower: str) -> bool:
    if token_lower in ALLOWED_SHORT_TECH_AND_GREETINGS:
        return False
    return any(sub in token_lower for sub in KEYBOARD_SMASH_SUBSTRINGS)


def _is_gibberish_latin_word(token_lower: str) -> bool:
    if token_lower in ALLOWED_SHORT_TECH_AND_GREETINGS:
        return False
    if any(ch.isdigit() for ch in token_lower):
        return False
    if len(token_lower) >= 4 and not any(v in token_lower for v in "aeiouy"):
        return True
    if REPEATED_CHAR_RE.search(token_lower):
        return True
    if _is_keyboard_smash(token_lower):
        return True
    return False


def validate_and_classify_input(
    text: str,
    history: list[dict[str, Any]] | None = None,
) -> tuple[bool, str | None]:
    """Validate visitor input before calling an LLM or triggering external tools.

    Returns (True, None) for valid inputs, or (False, calm_rejection_message) for empty
    or obvious nonsense/gibberish inputs.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return False, "Please type your question first. 🙂"

    # Allow emails or URLs immediately (e.g. "waled@gmail.com").
    if EMAIL_INLINE_RE.search(cleaned) or URL_INLINE_RE.search(cleaned):
        return True, None

    alnum_chars = [ch for ch in cleaned if ch.isalnum()]
    if not alnum_chars:
        return False, _rejection_message(cleaned)

    # Reject pure punctuation/symbol floods like "a$%^&*()_+" when symbols outnumber letters/digits.
    non_space_len = len(re.sub(r"\s+", "", cleaned))
    if non_space_len >= 4 and len(alnum_chars) / non_space_len < 0.4:
        return False, _rejection_message(cleaned)

    # Reject purely numeric spam unless it matches CV-relevant numbers.
    if re.fullmatch(r"[\d\s.,+\-/#]+", cleaned):
        digits_only = re.sub(r"\s+", "", cleaned)
        if digits_only in ALLOWED_CV_NUMBERS:
            return True, None
        return False, _rejection_message(cleaned)

    tokens = WORD_TOKEN_RE.findall(cleaned)
    if not tokens:
        return False, _rejection_message(cleaned)

    lowered_tokens = [tok.lower() for tok in tokens]

    # Reject single-character or two-identical-character noise like "a" or "aa" unless recognized.
    if len(tokens) == 1:
        tok = lowered_tokens[0]
        if tok in ALLOWED_SHORT_TECH_AND_GREETINGS or tok in ALLOWED_CV_NUMBERS:
            return True, None
        # Allow single-word visitor names in contact follow-ups (e.g. "Waled", "زياد") if not smash/repeated.
        if _is_after_contact_prompt(history) and len(tok) >= 2 and not REPEATED_CHAR_RE.search(tok) and not _is_keyboard_smash(tok):
            if ARABIC_CHAR_RE.search(tok) or any(v in tok for v in "aeiouy"):
                return True, None
        if len(tok) <= 1:
            return False, _rejection_message(cleaned)

    # Check repeated-character noise across the message (e.g. "aaaaaaa", "ههههههه").
    compact_alnum = "".join(alnum_chars).lower()
    if len(compact_alnum) >= 4 and len(set(compact_alnum)) <= 2 and compact_alnum not in ALLOWED_SHORT_TECH_AND_GREETINGS:
        return False, _rejection_message(cleaned)

    # Evaluate meaningful vs gibberish tokens.
    meaningful_count = 0
    for tok in lowered_tokens:
        if _is_keyboard_smash(tok) or REPEATED_CHAR_RE.search(tok):
            continue
        if re.fullmatch(r"[a-z]+", tok):
            if _is_gibberish_latin_word(tok):
                continue
            if len(tok) >= 2:
                meaningful_count += 1
        elif ARABIC_CHAR_RE.search(tok):
            if len(tok) >= 2 or tok in ALLOWED_SHORT_TECH_AND_GREETINGS:
                meaningful_count += 1
        elif any(ch.isalpha() for ch in tok):
            if CJK_HANGUL_KANA_RE.search(tok) or len(tok) >= 2:
                meaningful_count += 1
        elif any(ch.isdigit() for ch in tok):
            meaningful_count += 1

    if meaningful_count == 0:
        return False, _rejection_message(cleaned)

    return True, None
