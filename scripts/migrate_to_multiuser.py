"""Migrate a single-tenant ProfRAG artifacts/ tree to the multi-user layout.

What it does (idempotent — safe to re-run):
  1. Creates (or reuses) the owner account in artifacts/.users/users.db.
  2. Moves every document folder, the global index (metadata.json), jobs/, and
     chroma_db/ from the base artifacts root into artifacts/<owner_key>/.
  3. Rewrites path strings inside each moved document's metadata.json and the
     moved global index so they point at the new location.
  4. Renames Chroma collections to the <base>__u__<owner_key> format.

Notes:
  - Chroma record metadata (the per-chunk "document_folder" field) is NOT
    rewritten; deletion and retrieval fall back to "document_id", which is
    folder-relative and unaffected by the move.
  - Run with the API/UI stopped.

Usage (interactive):
    python scripts/migrate_to_multiuser.py

Usage (non-interactive):
    python scripts/migrate_to_multiuser.py --email owner@example.com \
        --password "..." --question-index 0 --answer "..."
"""

from __future__ import annotations

import argparse
import getpass
import json
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import user_store
from core.collections import user_collection_names
from core.config import load_app_config

# Base-root entries that are infrastructure, never document folders or user roots.
_INFRA_NAMES = {"jobs", "chroma_db"}


def _prompt_owner_details(args: argparse.Namespace) -> tuple[str, str, str, str]:
    email = args.email or input("Owner email: ").strip()
    password = args.password or getpass.getpass("Owner password (min 8 chars): ")
    if args.question_index is not None:
        question = user_store.SECURITY_QUESTIONS[args.question_index]
    else:
        print("Security questions:")
        for i, q in enumerate(user_store.SECURITY_QUESTIONS):
            print(f"  [{i}] {q}")
        question = user_store.SECURITY_QUESTIONS[int(input("Pick a question number: ").strip())]
    answer = args.answer or input(f"Answer to '{question}': ").strip()
    return email, password, question, answer


def _get_or_create_owner(base_root: Path, args: argparse.Namespace) -> user_store.UserRecord:
    if args.email:
        existing = user_store.get_user_by_email(base_root, args.email)
        if existing is not None:
            print(f"Reusing existing account: {existing.email_normalized} ({existing.user_key})")
            return existing
    email, password, question, answer = _prompt_owner_details(args)
    existing = user_store.get_user_by_email(base_root, email)
    if existing is not None:
        print(f"Reusing existing account: {existing.email_normalized} ({existing.user_key})")
        return existing
    record = user_store.create_user(base_root, email, password, question, answer)
    print(f"Created owner account: {record.email_normalized} ({record.user_key})")
    return record


def _is_document_folder(path: Path) -> bool:
    return path.is_dir() and (path / "metadata.json").exists()


def _rewrite_strings(value: Any, mapping: list[tuple[str, str]]) -> Any:
    if isinstance(value, str):
        for old, new in mapping:
            if old and old in value:
                value = value.replace(old, new)
        return value
    if isinstance(value, dict):
        return {k: _rewrite_strings(v, mapping) for k, v in value.items()}
    if isinstance(value, list):
        return [_rewrite_strings(v, mapping) for v in value]
    return value


def _path_mapping(base_root: Path, base_cfg_value: str, user_key: str) -> list[tuple[str, str]]:
    """All textual forms the old root can take inside stored JSON, longest first
    so absolute forms win over the bare relative prefix."""
    forms: list[tuple[str, str]] = []
    candidates = {
        str(base_root.resolve()),
        str(base_root),
        base_root.as_posix(),
        base_cfg_value,
        base_cfg_value.replace("/", "\\"),
    }
    for old in sorted(candidates, key=len, reverse=True):
        if not old:
            continue
        sep = "\\" if "\\" in old else "/"
        forms.append((old + sep, f"{old}{sep}{user_key}{sep}"))
    return forms


def _rewrite_json_file(path: Path, mapping: list[tuple[str, str]]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps(_rewrite_strings(payload, mapping), indent=2), encoding="utf-8")


def _move_documents(base_root: Path, user_root: Path, mapping: list[tuple[str, str]]) -> list[str]:
    moved: list[str] = []
    for entry in sorted(base_root.iterdir()):
        if entry.name.startswith(".") or entry.name in _INFRA_NAMES or entry == user_root:
            continue
        if not _is_document_folder(entry):
            continue  # other user roots or unknown dirs stay put
        target = user_root / entry.name
        if target.exists():
            print(f"  skip (exists at destination): {entry.name}")
            continue
        shutil.move(str(entry), str(target))
        _rewrite_json_file(target / "metadata.json", mapping)
        moved.append(entry.name)
        print(f"  moved document: {entry.name}")
    return moved


def _move_global_index(base_root: Path, user_root: Path, mapping: list[tuple[str, str]]) -> None:
    source = base_root / "metadata.json"
    if not source.exists():
        return
    target = user_root / "metadata.json"
    if target.exists():
        # Merge: keep destination entries as-is (already migrated); rewrite and
        # add only source entries so re-runs never double-apply the mapping.
        src = _rewrite_strings(json.loads(source.read_text(encoding="utf-8")), mapping)
        dst = json.loads(target.read_text(encoding="utf-8"))
        dst_docs = dst.setdefault("documents", {})
        for key, entry in (src.get("documents", {}) or {}).items():
            dst_docs.setdefault(key, entry)
        target.write_text(json.dumps(dst, indent=2), encoding="utf-8")
        source.unlink()
        print("  merged global index into existing user index")
    else:
        shutil.move(str(source), str(target))
        _rewrite_json_file(target, mapping)
        print("  moved global index")


def _move_tree(source: Path, target: Path, label: str) -> None:
    if not source.exists():
        return
    if not target.exists():
        shutil.move(str(source), str(target))
        print(f"  moved {label}")
        return
    # Merge file-by-file (jobs); leave conflicts in place.
    for item in sorted(source.iterdir()):
        dest = target / item.name
        if dest.exists():
            continue
        shutil.move(str(item), str(dest))
    if not any(source.iterdir()):
        source.rmdir()
    print(f"  merged {label}")


def _rename_chroma_collections(config: dict[str, Any], user_root: Path, user_key: str) -> None:
    persist_dir = user_root / "chroma_db"
    chroma_host = config.get("vector_db", {}).get("host")
    if not chroma_host and not persist_dir.exists():
        print("  no chroma_db directory — skipping collection rename")
        return

    import chromadb

    if chroma_host:
        client = chromadb.HttpClient(host=chroma_host, port=int(config["vector_db"].get("port", 8000)))
    else:
        client = chromadb.PersistentClient(path=str(persist_dir))

    scoped = user_collection_names(config, user_key)
    base_names = {
        "chunks": str(config["vector_db"].get("collection_name", "pdf_rag_chunks")),
        "summaries": str(config["vector_db"].get("summary_collection_name", "pdf_rag_summaries")),
        "cards": str(config["vector_db"].get("card_collection_name", "pdf_rag_cards")),
    }
    existing = {c.name for c in client.list_collections()}
    for family, base_name in base_names.items():
        new_name = scoped[family]
        if new_name in existing:
            print(f"  collection already renamed: {new_name}")
            continue
        if base_name not in existing:
            print(f"  collection not found (nothing to rename): {base_name}")
            continue
        client.get_collection(name=base_name).modify(name=new_name)
        print(f"  renamed collection: {base_name} -> {new_name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config/app_config.yaml")
    parser.add_argument("--email", default="")
    parser.add_argument("--password", default="")
    parser.add_argument("--question-index", type=int, default=None)
    parser.add_argument("--answer", default="")
    args = parser.parse_args()

    config = load_app_config(args.config)
    base_cfg_value = str(config["paths"]["artifacts_root"])
    base_root = Path(base_cfg_value)
    if not base_root.exists():
        print(f"Artifacts root does not exist: {base_root}")
        return 1

    owner = _get_or_create_owner(base_root, args)
    user_root = base_root / owner.user_key
    user_root.mkdir(parents=True, exist_ok=True)
    mapping = _path_mapping(base_root, base_cfg_value, owner.user_key)

    print(f"Migrating into: {user_root}")
    moved = _move_documents(base_root, user_root, mapping)
    _move_global_index(base_root, user_root, mapping)
    _move_tree(base_root / "jobs", user_root / "jobs", "jobs/")
    _move_tree(base_root / "chroma_db", user_root / "chroma_db", "chroma_db/")
    _rename_chroma_collections(config, user_root, owner.user_key)

    print(f"Done. Moved {len(moved)} document folder(s). Owner: {owner.email_normalized} ({owner.user_key})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
