import argparse
import json

from knowledge_base import (
    ACTIVE_SOURCE_FILE,
    BASE_DIR,
    DEFAULT_STORE,
    KnowledgeStore,
    atomic_write_text,
    build_knowledge_index,
    resolve_trusted_path,
)


def ingest(source: str | None = None, set_active: bool = False) -> dict:
    """Validate, optionally activate, and atomically build the versioned knowledge index."""
    if set_active:
        if not source:
            raise ValueError("--set-active requires --source PATH.")
        resolved = resolve_trusted_path(source, trusted_root=BASE_DIR, block_legacy=True)
        if not resolved.is_file():
            raise FileNotFoundError(f"CV source not found: {resolved.name}")
        active_value = resolved.resolve().relative_to(BASE_DIR.resolve()).as_posix()
        manifest = json.loads(ACTIVE_SOURCE_FILE.read_text(encoding="utf-8")) if ACTIVE_SOURCE_FILE.is_file() else {}
        manifest["active_cv"] = active_value
        manifest.setdefault("supplemental", "knowledge/supplemental_profile.json")
        atomic_write_text(ACTIVE_SOURCE_FILE, json.dumps(manifest, indent=2) + "\n")

    build_knowledge_index(cv_path=source, write_index=True)
    store = DEFAULT_STORE if source is None else KnowledgeStore()
    return store.get_diagnostics_metadata()


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate CV extraction and rebuild knowledge/index.json.")
    parser.add_argument("--source", help="Optional path to a new CV file (.pdf, .md, or .txt).")
    parser.add_argument(
        "--set-active",
        action="store_true",
        help="Update knowledge/active_source.json to point to --source.",
    )
    args = parser.parse_args()
    try:
        meta = ingest(source=args.source, set_active=args.set_active)
    except Exception as exc:
        print(f"CV ingestion failed: {type(exc).__name__}: {exc}", flush=True)
        return 1

    print(
        f"Knowledge index built: active_cv={meta['active_cv_file']} "
        f"sha256={meta['active_cv_sha256_prefix']} "
        f"chunks={meta['public_chunk_count']} "
        f"sections={meta['sections']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
