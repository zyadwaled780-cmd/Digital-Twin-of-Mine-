from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from threading import Lock
from typing import Any
from pypdf import PdfReader

BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
ACTIVE_SOURCE_FILE = KNOWLEDGE_DIR / "active_source.json"
DEFAULT_INDEX_FILE = KNOWLEDGE_DIR / "index.json"
DEFAULT_SUPPLEMENTAL_FILE = KNOWLEDGE_DIR / "supplemental_profile.json"
INDEX_SCHEMA_VERSION = 4

LEGACY_BLOCKED_FILENAMES = {"cv.pdf", "cv.original.pdf", "summary.txt"}
NON_CV_KNOWLEDGE_FILENAMES = {
    "readme.md",
    "supplemental_profile.md",
    "supplemental_profile.json",
    "active_source.json",
    "index.json",
}

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_CANDIDATE_RE = re.compile(r"(?<!\w)(\+?\d[\d\s\-().]{7,}\d)(?!\w)")
URL_RE = re.compile(r"https?://\S+|linkedin\.com/\S+", re.IGNORECASE)
ARABIC_DIACRITICS_RE = re.compile(r"[\u064B-\u065F\u0670]")

SECTION_HEADINGS = {
    "CONTACT INFORMATION": "contact_information",
    "CAREER OBJECTIVE": "career_objective",
    "EDUCATION": "education",
    "EXPERIENCE": "experience",
    "COURSES AND CERTIFICATIONS": "courses_and_certifications",
    "TECHNICAL SKILLS": "technical_skills",
    "SOFT SKILLS": "soft_skills",
    "PROJECTS": "projects",
}

STOPWORDS = {
    "the", "is", "are", "was", "were", "be", "been", "being", "a", "an", "and", "or", "but",
    "of", "in", "on", "at", "to", "for", "with", "from", "by", "as", "it", "its", "this",
    "that", "these", "those", "he", "his", "him", "what", "which", "who", "whom", "whose",
    "where", "when", "why", "how", "do", "does", "did", "can", "could", "would", "should",
    "has", "have", "had", "will", "any", "me", "my", "you", "your", "please", "tell",
    "know", "knows", "listed", "mention", "mentioned",
    "في", "من", "على", "عن", "الي", "إلى", "هل", "ما", "ايه", "إيه", "هو", "هي", "ده", "دي",
    "يا", "ان", "إن", "او", "أو", "كل", "كان", "عند", "عنده", "عندي", "مع", "لي", "لـ", "لو",
    "بتاع", "اللي", "الذي", "التي", "ممكن", "قولي", "قل", "عرفني", "احكيلي",
}

ARABIC_TO_ENGLISH_HINTS: dict[str, list[str]] = {
    "مشروع": ["projects", "project", "github"],
    "مشاريع": ["projects", "project", "github"],
    "شغل": ["experience", "projects", "career_objective"],
    "عمل": ["experience", "utest", "outlier"],
    "خبره": ["experience", "utest", "outlier", "testing", "annotation"],
    "خبرات": ["experience", "utest", "outlier"],
    "مهاره": ["technical_skills", "soft_skills", "skills"],
    "مهارات": ["technical_skills", "soft_skills", "skills"],
    "اداه": ["technical_skills", "n8n", "python", "docker", "ollama"],
    "ادوات": ["technical_skills", "n8n", "python", "docker", "ollama", "sql", "supabase", "flutter"],
    "تقنيه": ["technical_skills", "rag", "hugging", "javascript", "dart"],
    "تقنيات": ["technical_skills", "rag", "hugging", "javascript", "dart"],
    "برمجه": ["technical_skills", "python", "javascript", "dart", "sql"],
    "لغات": ["technical_skills", "python", "javascript", "dart", "sql"],
    "تعليم": ["education", "university", "gpa", "graduation", "bis"],
    "دراسه": ["education", "university", "capital", "commerce", "gpa"],
    "جامعه": ["education", "capital", "university"],
    "كليه": ["education", "commerce", "capital"],
    "تخرج": ["education", "2028", "graduation"],
    "معدل": ["education", "gpa", "3.4"],
    "تقدير": ["education", "gpa", "3.4"],
    "كورس": ["courses_and_certifications", "course", "courses", "depi", "mckinsey", "ibm", "gci"],
    "كورسات": ["courses_and_certifications", "course", "courses", "depi", "mckinsey", "ibm", "gci"],
    "شهاده": ["courses_and_certifications", "certification", "certifications", "depi", "ibm", "gci"],
    "شهادات": ["courses_and_certifications", "certification", "certifications", "depi", "ibm", "gci"],
    "تدريب": ["courses_and_certifications", "depi", "eyouth"],
    "هدف": ["career_objective", "businesses", "automate"],
    "تخصص": ["header", "career_objective", "automation", "specialist"],
    "مين": ["header", "career_objective", "education", "projects", "technical_skills"],
    "نبذه": ["header", "career_objective", "soft_skills"],
    "شخصيه": ["soft_skills", "listening", "patience", "adaptability"],
    "تواصل": ["public_profile", "linkedin", "contact"],
    "لينكدان": ["public_profile", "linkedin"],
    "موقع": ["public_profile", "location", "october", "egypt"],
    "بلد": ["public_profile", "location", "egypt"],
    "مصر": ["public_profile", "egypt", "october"],
    "جيتهب": ["projects", "github"],
    "لينك": ["projects", "github", "linkedin"],
    "روابط": ["projects", "github", "linkedin"],
}

SECTION_BOOST_KEYWORDS: dict[str, set[str]] = {
    "header": {"who", "zyad", "name", "title", "role", "specialist", "intro", "overview", "about"},
    "public_profile": {"linkedin", "location", "city", "egypt", "october", "reach", "contact", "profile"},
    "career_objective": {"objective", "goal", "focus", "help", "businesses", "automate", "overview", "about"},
    "education": {"education", "degree", "university", "college", "faculty", "bis", "gpa", "graduation", "year", "2028", "3.4"},
    "experience": {"experience", "work", "job", "utest", "outlier", "tester", "testing", "annotator", "annotation", "employer"},
    "courses_and_certifications": {"course", "courses", "certification", "certifications", "depi", "eyouth", "mckinsey", "forward", "hp", "ibm", "gci", "matsuo", "2026"},
    "technical_skills": {"skill", "skills", "tech", "stack", "tool", "tools", "n8n", "sql", "supabase", "ollama", "hugging", "face", "git", "github", "docker", "rag", "python", "javascript", "dart", "flutter"},
    "soft_skills": {"soft", "listening", "patience", "detail", "adaptability", "problem", "solving"},
    "projects": {"project", "projects", "repo", "repos", "github", "url", "link", "digital", "twin", "support", "agent", "portfolio", "learning", "app"},
}

BROAD_QUERY_TOKENS = {
    "who", "about", "overview", "summary", "introduce", "background", "profile",
    "everything", "tell", "zyad", "مين", "نبذة", "احكيلي", "عرفني",
}


class CVExtractionError(ValueError):
    """Raised when a CV document cannot be reliably extracted into readable text."""


class AmbiguousCVSourceError(RuntimeError):
    """Raised when multiple candidate CV files exist without an explicit active source."""


class InvalidCVSourceError(ValueError):
    """Raised when a configured CV or supplemental source path escapes trusted roots or points to legacy backups."""


@dataclass(frozen=True)
class KnowledgeChunk:
    chunk_id: str
    source: str
    section: str
    precedence: int
    visibility: str
    content: str
    source_file: str
    source_sha256: str


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_write_text(path: Path, content: str) -> None:
    """Write text atomically using a temporary file in the same directory and os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp_name = tmp.name
            tmp.write(content)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_name, path)
        tmp_name = None
    finally:
        if tmp_name is not None:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass


def scrub_private_contact_info(text: str) -> str:
    """Redact private email addresses and phone numbers across all public content lines."""
    scrubbed_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lower = line.lower()
        if lower.startswith("phone:") or lower.startswith("tel:") or lower.startswith("email:"):
            continue
        url_spans = [m.span() for m in URL_RE.finditer(line)]

        def _in_url(start: int, end: int) -> bool:
            return any(u_start <= start and end <= u_end for u_start, u_end in url_spans)

        cleaned = EMAIL_RE.sub(
            lambda m: m.group(0) if _in_url(m.start(), m.end()) else "[REDACTED_EMAIL]",
            line,
        )

        url_spans = [m.span() for m in URL_RE.finditer(cleaned)]

        def _replace_phone(match: re.Match[str]) -> str:
            if _in_url(match.start(), match.end()):
                return match.group(0)
            digits = re.sub(r"\D", "", match.group(1))
            if len(digits) >= 9:
                return "[REDACTED_PHONE]"
            return match.group(0)

        cleaned = PHONE_CANDIDATE_RE.sub(_replace_phone, cleaned)
        scrubbed_lines.append(cleaned)
    return "\n".join(scrubbed_lines)


def normalize_cv_text(text: str) -> str:
    """Normalize extraction artifacts while preserving dates, names, URLs, and section structure."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(https?://[^\s\n]+)-\s*\n\s*([a-z0-9][a-z0-9-]*)(?=\s*(?:\n|$))", r"\1-\2", text)
    text = re.sub(r"([A-Za-z0-9])-\s*\n\s*([a-z][a-z0-9-]*)", r"\1-\2", text)
    text = text.replace("Gi tHub", "GitHub")
    text = text.replace("Gmail -Ollama", "Gmail-Ollama")
    text = text.replace("context -aware", "context-aware")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def validate_extracted_text(text: str, source_name: str) -> str:
    """Verify extracted CV text is complete enough for reliable grounding."""
    normalized = normalize_cv_text(text)
    non_ws = re.sub(r"\s+", "", normalized)
    if len(non_ws) < 100:
        raise CVExtractionError(
            f"CV extraction from '{source_name}' produced too little readable text "
            f"({len(non_ws)} non-whitespace chars); the file may be scanned or empty."
        )
    replacement_count = normalized.count("\ufffd")
    if replacement_count > max(3, int(len(non_ws) * 0.05)):
        raise CVExtractionError(
            f"CV extraction from '{source_name}' contains excessive corrupted characters "
            f"({replacement_count}); text quality is unreliable."
        )
    return normalized


def extract_document_text(path: str | Path) -> str:
    """Extract and validate text from a local .pdf, .md, or .txt CV document."""
    doc_path = Path(path)
    if not doc_path.is_file():
        raise FileNotFoundError(f"CV document not found: {doc_path.name}")

    suffix = doc_path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(doc_path))
        pages: list[str] = []
        for page in reader.pages:
            extracted = page.extract_text() or ""
            if extracted.strip():
                pages.append(extracted)
        raw_text = "\n\n".join(pages)
    elif suffix in (".md", ".txt"):
        raw_text = doc_path.read_text(encoding="utf-8")
    else:
        raise CVExtractionError(f"Unsupported CV file format: {suffix or '(no extension)'}")

    return validate_extracted_text(raw_text, doc_path.name)


def resolve_trusted_path(
    raw_path: str | Path,
    *,
    trusted_root: Path | None = None,
    block_legacy: bool = True,
) -> Path:
    """Resolve a configured source path relative to trusted_root, rejecting escapes and legacy backups."""
    root = (trusted_root or BASE_DIR).resolve()
    candidate = Path(raw_path)
    resolved = (candidate if candidate.is_absolute() else (root / candidate)).resolve()

    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise InvalidCVSourceError(
            f"Configured path escapes trusted application directory: {candidate.name}"
        ) from exc

    if block_legacy and resolved.name.lower() in LEGACY_BLOCKED_FILENAMES and resolved.parent == root:
        raise InvalidCVSourceError(
            f"Legacy backup file '{resolved.name}' cannot be used as an active knowledge source."
        )

    return resolved


def resolve_active_cv_path(
    explicit_path: str | Path | None = None,
    knowledge_dir: Path | None = None,
    active_source_file: Path | None = None,
    trusted_root: Path | None = None,
) -> Path:
    """Resolve the single authoritative CV path relative to trusted_root/application root."""
    kdir = knowledge_dir or KNOWLEDGE_DIR
    manifest_path = active_source_file or (kdir / "active_source.json")
    root = (trusted_root or (kdir.parent if knowledge_dir is not None else BASE_DIR)).resolve()

    if explicit_path is not None:
        resolved = resolve_trusted_path(explicit_path, trusted_root=root, block_legacy=True)
        if not resolved.is_file():
            raise FileNotFoundError(f"Configured CV source not found: {resolved.name}")
        return resolved

    env_path = os.getenv("CV_SOURCE_PATH", "").strip()
    if env_path:
        resolved = resolve_trusted_path(env_path, trusted_root=root, block_legacy=True)
        if not resolved.is_file():
            raise FileNotFoundError(f"CV_SOURCE_PATH target not found: {resolved.name}")
        return resolved

    if manifest_path.is_file():
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        active_rel = str(data.get("active_cv", "")).strip()
        if not active_rel:
            raise AmbiguousCVSourceError("active_source.json does not specify 'active_cv'.")
        resolved = resolve_trusted_path(active_rel, trusted_root=root, block_legacy=True)
        if not resolved.is_file():
            raise FileNotFoundError(f"Active CV listed in manifest not found: {resolved.name}")
        return resolved

    candidates = [
        p
        for p in sorted(kdir.glob("*"))
        if p.is_file()
        and p.suffix.lower() in (".pdf", ".md", ".txt")
        and p.name.lower() not in NON_CV_KNOWLEDGE_FILENAMES
        and p.name.lower() not in LEGACY_BLOCKED_FILENAMES
    ] if kdir.is_dir() else []

    if len(candidates) == 1:
        return candidates[0].resolve()
    if len(candidates) > 1:
        names = ", ".join(c.name for c in candidates)
        raise AmbiguousCVSourceError(
            f"Multiple candidate CV files found ({names}) without active_source.json or CV_SOURCE_PATH."
        )
    raise FileNotFoundError("No active CV document configured in knowledge/.")


def resolve_supplemental_path(
    explicit_path: str | Path | None = None,
    knowledge_dir: Path | None = None,
    active_source_file: Path | None = None,
    trusted_root: Path | None = None,
) -> Path | None:
    """Resolve the configured supplemental source path (if any) within trusted_root."""
    kdir = knowledge_dir or KNOWLEDGE_DIR
    manifest_path = active_source_file or (kdir / "active_source.json")
    root = (trusted_root or (kdir.parent if knowledge_dir is not None else BASE_DIR)).resolve()

    if explicit_path is not None:
        return resolve_trusted_path(explicit_path, trusted_root=root, block_legacy=True)

    if manifest_path.is_file():
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        supp_rel = str(data.get("supplemental", "")).strip()
        if supp_rel:
            return resolve_trusted_path(supp_rel, trusted_root=root, block_legacy=True)

    default_json = kdir / "supplemental_profile.json"
    if default_json.is_file():
        return default_json.resolve()
    return None


def _split_contact_section(lines: list[str]) -> tuple[list[str], list[str]]:
    """Separate private contact details (phone/email) from public profile lines (location/LinkedIn)."""
    private_lines: list[str] = []
    public_lines: list[str] = []
    for line in lines:
        lower = line.lower()
        if (
            lower.startswith("phone:")
            or lower.startswith("tel:")
            or lower.startswith("email:")
            or EMAIL_RE.search(line)
            or (PHONE_CANDIDATE_RE.search(line) and not URL_RE.search(line))
        ):
            private_lines.append(line)
        else:
            public_lines.append(line)
    return private_lines, public_lines


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "item"


def parse_cv_sections(text: str, source_name: str, source_sha256: str) -> list[KnowledgeChunk]:
    """Parse normalized CV text into stable section-level and item-level chunks with privacy scrubbing."""
    sections: dict[str, list[str]] = {"header": []}
    current_section = "header"

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        canonical_heading = SECTION_HEADINGS.get(line.upper())
        if canonical_heading:
            current_section = canonical_heading
            sections.setdefault(current_section, [])
            continue
        sections.setdefault(current_section, []).append(line)

    chunks: list[KnowledgeChunk] = []

    if sections.get("header"):
        header_public = scrub_private_contact_info("\n".join(sections["header"]))
        if header_public:
            chunks.append(
                KnowledgeChunk(
                    chunk_id="cv::header",
                    source="active_cv",
                    section="header",
                    precedence=1,
                    visibility="public",
                    content=header_public,
                    source_file=source_name,
                    source_sha256=source_sha256,
                )
            )

    contact_lines = sections.get("contact_information", [])
    if contact_lines:
        private_lines, public_lines = _split_contact_section(contact_lines)
        if private_lines:
            chunks.append(
                KnowledgeChunk(
                    chunk_id="cv::contact_information",
                    source="active_cv",
                    section="contact_information",
                    precedence=1,
                    visibility="private",
                    content="[PRIVATE CONTACT DETAILS REDACTED FROM PUBLIC INDEX]",
                    source_file=source_name,
                    source_sha256=source_sha256,
                )
            )
        if public_lines:
            public_clean = scrub_private_contact_info("\n".join(public_lines))
            if public_clean:
                chunks.append(
                    KnowledgeChunk(
                        chunk_id="cv::public_profile",
                        source="active_cv",
                        section="public_profile",
                        precedence=1,
                        visibility="public",
                        content=public_clean,
                        source_file=source_name,
                        source_sha256=source_sha256,
                    )
                )

    ordered_sections = [
        "career_objective",
        "education",
        "experience",
        "courses_and_certifications",
        "technical_skills",
        "soft_skills",
        "projects",
    ]
    for sec in ordered_sections:
        lines = sections.get(sec, [])
        if not lines:
            continue
        scrubbed = scrub_private_contact_info("\n".join(lines))
        if not scrubbed:
            continue
        chunks.append(
            KnowledgeChunk(
                chunk_id=f"cv::{sec}",
                source="active_cv",
                section=sec,
                precedence=1,
                visibility="public",
                content=scrubbed,
                source_file=source_name,
                source_sha256=source_sha256,
            )
        )

        # Add granular item-level chunks for multi-entry sections to improve targeted retrieval precision.
        if sec in ("projects", "experience", "courses_and_certifications"):
            for idx, item_line in enumerate(scrubbed.splitlines()):
                item_clean = item_line.strip()
                if not item_clean:
                    continue
                prefix = item_clean.split(":", 1)[0].split("|", 1)[0].split("-", 1)[0].strip()
                item_slug = _slugify(prefix)[:40] or f"item_{idx}"
                chunks.append(
                    KnowledgeChunk(
                        chunk_id=f"cv::{sec}::{item_slug}",
                        source="active_cv",
                        section=sec,
                        precedence=1,
                        visibility="public",
                        content=item_clean,
                        source_file=source_name,
                        source_sha256=source_sha256,
                    )
                )

    return chunks


def parse_supplemental_sections(path: Path | None) -> tuple[str, list[KnowledgeChunk]]:
    """Parse structured supplemental profile entries, including ONLY status=='approved' items."""
    if path is None or not path.is_file():
        return "", []
    raw = path.read_text(encoding="utf-8")
    sha = _sha256_text(raw)

    if path.suffix.lower() == ".json":
        payload = json.loads(raw)
        entries = payload.get("entries", []) if isinstance(payload, dict) else []
        chunks: list[KnowledgeChunk] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            status = str(entry.get("status", "")).strip().lower()
            if status != "approved":
                continue
            entry_id = re.sub(r"[^a-z0-9_]+", "_", str(entry.get("id", "entry")).lower()).strip("_")
            section = re.sub(r"[^a-z0-9_]+", "_", str(entry.get("section", entry_id)).lower()).strip("_")
            content = scrub_private_contact_info(str(entry.get("content", "")).strip())
            if not content:
                continue
            chunks.append(
                KnowledgeChunk(
                    chunk_id=f"supplemental::{entry_id}",
                    source="supplemental_profile",
                    section=section,
                    precedence=2,
                    visibility="public",
                    content=content,
                    source_file=path.name,
                    source_sha256=sha,
                )
            )
        return sha, chunks

    sections: dict[str, list[str]] = {}
    current = ""
    approved_section = False
    for raw_line in raw.splitlines():
        line = raw_line.strip()
        if line.startswith("## "):
            heading = line[3:].strip()
            approved_section = "[APPROVED]" in heading.upper()
            clean_heading = re.sub(r"\[APPROVED\]", "", heading, flags=re.IGNORECASE).strip()
            current = re.sub(r"[^a-z0-9]+", "_", clean_heading.lower()).strip("_") or "general"
            if approved_section:
                sections.setdefault(current, [])
            continue
        if not approved_section or not current or not line or line.startswith("# ") or line.startswith("<!--"):
            continue
        sections[current].append(line)

    chunks = []
    for sec, lines in sections.items():
        content = scrub_private_contact_info("\n".join(lines))
        if not content:
            continue
        chunks.append(
            KnowledgeChunk(
                chunk_id=f"supplemental::{sec}",
                source="supplemental_profile",
                section=sec,
                precedence=2,
                visibility="public",
                content=content,
                source_file=path.name,
                source_sha256=sha,
            )
        )
    return sha, chunks


def _rel_display_path(path: Path, root: Path | None = None) -> str:
    base = (root or BASE_DIR).resolve()
    try:
        return path.resolve().relative_to(base).as_posix()
    except ValueError:
        return path.name


def build_knowledge_index(
    cv_path: str | Path | None = None,
    supplemental_path: str | Path | None = None,
    output_path: str | Path | None = None,
    knowledge_dir: Path | None = None,
    active_source_file: Path | None = None,
    trusted_root: Path | None = None,
    write_index: bool = True,
) -> dict[str, Any]:
    """Ingest the active CV and approved supplemental entries and write knowledge/index.json atomically."""
    kdir = knowledge_dir or KNOWLEDGE_DIR
    root = (trusted_root or (kdir.parent if knowledge_dir is not None else BASE_DIR)).resolve()

    resolved_cv = resolve_active_cv_path(
        explicit_path=cv_path,
        knowledge_dir=kdir,
        active_source_file=active_source_file,
        trusted_root=root,
    )
    cv_bytes = resolved_cv.read_bytes()
    cv_sha = _sha256_bytes(cv_bytes)
    cv_text = extract_document_text(resolved_cv)
    cv_display = _rel_display_path(resolved_cv, root=root)

    supp_path = resolve_supplemental_path(
        explicit_path=supplemental_path,
        knowledge_dir=kdir,
        active_source_file=active_source_file,
        trusted_root=root,
    )
    supp_sha, supp_chunks = parse_supplemental_sections(supp_path)
    supp_display = _rel_display_path(supp_path, root=root) if supp_path and supp_path.is_file() else ""

    chunks = parse_cv_sections(cv_text, cv_display, cv_sha) + supp_chunks
    combined_sha = _sha256_text(
        f"{cv_display}:{cv_sha}:{supp_display}:{supp_sha}:{INDEX_SCHEMA_VERSION}"
    )

    index_data = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "active_cv_file": cv_display,
        "active_cv_sha256": cv_sha,
        "supplemental_file": supp_display,
        "supplemental_sha256": supp_sha,
        "combined_sha256": combined_sha,
        "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "chunks": [asdict(c) for c in chunks],
    }

    if write_index:
        target = Path(output_path) if output_path is not None else (kdir / "index.json")
        atomic_write_text(target, json.dumps(index_data, ensure_ascii=False, indent=2) + "\n")

    return index_data


def _normalize_arabic_token(tok: str) -> list[str]:
    """Normalize Arabic variants and strip common prefixes/suffixes for better morphological recall."""
    clean = ARABIC_DIACRITICS_RE.sub("", tok)
    clean = clean.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ى", "ي").replace("ة", "ه")
    variants = {clean}
    # Strip conjunction/preposition prefixes (و، ف، ب، ل، ك)
    no_conj = clean
    if len(no_conj) > 3 and no_conj[0] in ("و", "ف", "ب", "ل", "ك"):
        no_conj = no_conj[1:]
        variants.add(no_conj)
    # Strip definite article ال
    for base in list(variants):
        if base.startswith("ال") and len(base) > 4:
            variants.add(base[2:])
    # Strip possessive/plural suffixes (ه، ها، هم، ك، ي)
    for base in list(variants):
        if len(base) > 4 and base.endswith(("ها", "هم")):
            variants.add(base[:-2])
        if len(base) > 3 and base.endswith(("ه", "ك", "ي")):
            variants.add(base[:-1])
    return [v for v in variants if v]


def _tokenize_with_hints(text: str, *, filter_stopwords: bool = True) -> set[str]:
    """Tokenize English + Arabic text, normalize Arabic forms, and expand domain synonyms."""
    lowered = text.lower()
    raw_tokens = re.findall(r"[a-z0-9_.\-]+|[\u0621-\u064A\u0660-\u0669\u0671-\u06D3]+", lowered)
    expanded: set[str] = set()

    for tok in raw_tokens:
        if filter_stopwords and tok in STOPWORDS:
            continue
        expanded.add(tok)

        if "-" in tok:
            for part in tok.split("-"):
                if part and (not filter_stopwords or part not in STOPWORDS):
                    expanded.add(part)

        if re.search(r"[\u0600-\u06FF]", tok):
            for variant in _normalize_arabic_token(tok):
                if not filter_stopwords or variant not in STOPWORDS:
                    expanded.add(variant)
                if variant in ARABIC_TO_ENGLISH_HINTS:
                    expanded.update(ARABIC_TO_ENGLISH_HINTS[variant])
        elif tok in ARABIC_TO_ENGLISH_HINTS:
            expanded.update(ARABIC_TO_ENGLISH_HINTS[tok])

    return expanded


def _fuzzy_match_score(query_tok: str, candidate_tokens: set[str]) -> float:
    """Return bounded fuzzy match weight for minor typos in Latin terms >= 4 chars."""
    if len(query_tok) < 4 or not re.fullmatch(r"[a-z0-9]+", query_tok):
        return 0.0
    best = 0.0
    for cand in candidate_tokens:
        if len(cand) < 4 or abs(len(cand) - len(query_tok)) > 2:
            continue
        if not re.fullmatch(r"[a-z0-9]+", cand):
            continue
        ratio = SequenceMatcher(None, query_tok, cand).ratio()
        if ratio >= 0.80 and ratio > best:
            best = ratio
    return best * 1.4 if best >= 0.80 else 0.0


def _score_chunk_bm25(
    chunk: KnowledgeChunk,
    query_tokens: set[str],
    history_tokens: set[str],
    idf_map: dict[str, float],
) -> float:
    content_tokens = _tokenize_with_hints(chunk.content, filter_stopwords=True)
    section_hints = SECTION_BOOST_KEYWORDS.get(chunk.section, set())
    all_chunk_tokens = content_tokens | section_hints

    is_section_chunk = chunk.chunk_id.count("::") == 1
    hint_weight = 1.5 if is_section_chunk else 0.9

    score = 0.0
    for tok in query_tokens:
        if len(tok) < 2 and not tok.isdigit():
            continue
        idf = idf_map.get(tok, 1.5)
        if tok in content_tokens:
            score += 2.5 * idf
        elif tok in section_hints:
            score += hint_weight * idf
        else:
            fuzzy = _fuzzy_match_score(tok, all_chunk_tokens)
            if fuzzy > 0:
                score += fuzzy

    for tok in history_tokens:
        if len(tok) < 2 and not tok.isdigit():
            continue
        idf = idf_map.get(tok, 1.2)
        if tok in content_tokens:
            score += 0.65 * idf
        elif tok in section_hints:
            score += 0.45 * idf

    # Prefer full section chunk slightly over granular item chunks unless an item matched uniquely.
    if score > 0 and is_section_chunk:
        score += 0.5
    if score > 0 and chunk.precedence == 1:
        score += 0.25
    return score


class KnowledgeStore:
    """Thread-safe active knowledge cache with automatic hash-based invalidation."""

    def __init__(
        self,
        knowledge_dir: Path | None = None,
        index_file: Path | None = None,
        active_source_file: Path | None = None,
        trusted_root: Path | None = None,
    ):
        self.knowledge_dir = knowledge_dir or KNOWLEDGE_DIR
        self.index_file = index_file or (self.knowledge_dir / "index.json")
        self.active_source_file = active_source_file or (self.knowledge_dir / "active_source.json")
        self.trusted_root = (
            trusted_root or (self.knowledge_dir.parent if knowledge_dir is not None else BASE_DIR)
        ).resolve()
        self._lock = Lock()
        self._cached_index: dict[str, Any] | None = None

    def _compute_current_combined_sha(self) -> tuple[Path, Path | None, str]:
        cv_path = resolve_active_cv_path(
            knowledge_dir=self.knowledge_dir,
            active_source_file=self.active_source_file,
            trusted_root=self.trusted_root,
        )
        cv_sha = _sha256_bytes(cv_path.read_bytes())
        supp_path = resolve_supplemental_path(
            knowledge_dir=self.knowledge_dir,
            active_source_file=self.active_source_file,
            trusted_root=self.trusted_root,
        )
        supp_sha = _sha256_text(supp_path.read_text(encoding="utf-8")) if supp_path and supp_path.is_file() else ""
        cv_display = _rel_display_path(cv_path, root=self.trusted_root)
        supp_display = _rel_display_path(supp_path, root=self.trusted_root) if supp_path and supp_path.is_file() else ""
        combined = _sha256_text(
            f"{cv_display}:{cv_sha}:{supp_display}:{supp_sha}:{INDEX_SCHEMA_VERSION}"
        )
        return cv_path, supp_path, combined

    def get_index(self, force_rebuild: bool = False) -> dict[str, Any]:
        with self._lock:
            cv_path, supp_path, expected_sha = self._compute_current_combined_sha()

            if (
                not force_rebuild
                and self._cached_index is not None
                and self._cached_index.get("combined_sha256") == expected_sha
            ):
                return self._cached_index

            if not force_rebuild and self.index_file.is_file():
                try:
                    disk_data = json.loads(self.index_file.read_text(encoding="utf-8"))
                    if disk_data.get("combined_sha256") == expected_sha:
                        self._cached_index = disk_data
                        return disk_data
                except (json.JSONDecodeError, OSError):
                    pass

            build_kwargs = {
                "cv_path": cv_path,
                "supplemental_path": supp_path,
                "output_path": self.index_file,
                "knowledge_dir": self.knowledge_dir,
                "active_source_file": self.active_source_file,
                "trusted_root": self.trusted_root,
            }
            try:
                rebuilt = build_knowledge_index(**build_kwargs, write_index=True)
            except OSError:
                rebuilt = build_knowledge_index(**build_kwargs, write_index=False)
            self._cached_index = rebuilt
            return rebuilt

    def get_chunks(self) -> list[KnowledgeChunk]:
        """Return model-facing public chunks (strict CV-only by default; private contact excluded)."""
        index_data = self.get_index()
        cv_only = os.getenv("CV_ONLY_MODE", "true").strip().lower() not in ("0", "false", "no", "off")
        chunks = [KnowledgeChunk(**item) for item in index_data.get("chunks", [])]
        return [
            c
            for c in chunks
            if c.visibility != "private" and (not cv_only or c.source == "active_cv")
        ]

    def retrieve_evidence(
        self,
        query: str,
        history: list[dict[str, Any]] | None = None,
        top_k: int = 6,
    ) -> list[KnowledgeChunk]:
        chunks = self.get_chunks()
        if not chunks:
            return []

        # Compute lightweight IDF across active chunks.
        doc_freq: dict[str, int] = {}
        num_docs = len(chunks)
        for c in chunks:
            toks = _tokenize_with_hints(c.content, filter_stopwords=True) | SECTION_BOOST_KEYWORDS.get(c.section, set())
            for tok in toks:
                doc_freq[tok] = doc_freq.get(tok, 0) + 1
        idf_map = {tok: math.log((num_docs + 1) / (df + 0.5)) + 1.0 for tok, df in doc_freq.items()}

        query_tokens = _tokenize_with_hints(query or "", filter_stopwords=True)
        recent_user_texts: list[str] = []
        for entry in reversed(history or []):
            if isinstance(entry, dict) and entry.get("role") == "user":
                content = entry.get("content")
                if isinstance(content, str) and content.strip():
                    recent_user_texts.append(content)
                if len(recent_user_texts) >= 2:
                    break
        history_tokens = _tokenize_with_hints(" ".join(recent_user_texts), filter_stopwords=True)

        scored: list[tuple[float, KnowledgeChunk]] = []
        for chunk in chunks:
            s = _score_chunk_bm25(chunk, query_tokens, history_tokens, idf_map)
            if s > 0:
                scored.append((s, chunk))

        scored.sort(key=lambda pair: (-pair[0], pair[1].precedence, pair[1].chunk_id))
        selected: list[KnowledgeChunk] = []
        selected_parents: set[str] = set()
        for _, chunk in scored:
            parts = chunk.chunk_id.split("::")
            is_item = len(parts) == 3
            parent_id = f"{parts[0]}::{chunk.section}"
            if is_item and parent_id in selected_parents:
                continue
            if not is_item and len(parts) == 2:
                selected_parents.add(chunk.chunk_id)
                selected = [
                    c for c in selected
                    if not (c.chunk_id.count("::") == 2 and f"{c.chunk_id.split('::')[0]}::{c.section}" == chunk.chunk_id)
                ]
            selected.append(chunk)
            if len(selected) >= top_k:
                break

        raw_lower_tokens = set(re.findall(r"[a-z0-9]+|[\u0621-\u064A\u0660-\u0669\u0671-\u06D3]+", (query or "").lower()))
        is_broad = bool(raw_lower_tokens & BROAD_QUERY_TOKENS) or not selected
        if is_broad:
            core_ids = [
                "cv::header",
                "cv::career_objective",
                "cv::technical_skills",
                "cv::projects",
                "cv::education",
            ]
            by_id = {c.chunk_id: c for c in chunks}
            seen = {c.chunk_id for c in selected}
            for cid in core_ids:
                if cid in by_id and cid not in seen and len(selected) < max(top_k, 5):
                    selected.append(by_id[cid])
                    seen.add(cid)

        by_id = {c.chunk_id: c for c in chunks}
        if "cv::header" in by_id and all(c.chunk_id != "cv::header" for c in selected):
            selected = [by_id["cv::header"]] + selected[: max(top_k - 1, 1)]

        return selected

    def get_diagnostics_metadata(self) -> dict[str, Any]:
        index_data = self.get_index()
        public_chunks = self.get_chunks()
        section_chunks = [c for c in public_chunks if c.chunk_id.count("::") == 1]
        return {
            "active_cv_file": index_data.get("active_cv_file", ""),
            "active_cv_sha256_prefix": str(index_data.get("active_cv_sha256", ""))[:12],
            "combined_sha256_prefix": str(index_data.get("combined_sha256", ""))[:12],
            "public_chunk_count": len(public_chunks),
            "sections": [c.section for c in section_chunks],
        }


DEFAULT_STORE = KnowledgeStore()
