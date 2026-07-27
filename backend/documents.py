import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

UPLOAD_DIR = Path("./data/uploads")
MANIFEST_PATH = UPLOAD_DIR / "manifest.json"

_lock = Lock()


def _ensure_dirs():
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _read_manifest():
    if not MANIFEST_PATH.exists():
        return []
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_manifest(entries):
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)


def list_documents():
    _ensure_dirs()
    with _lock:
        return _read_manifest()


def save_document(filename: str, content: bytes) -> dict:
    """Persist an uploaded PDF to disk and register it in the manifest."""
    _ensure_dirs()
    safe_name = os.path.basename(filename)
    doc_id = uuid.uuid4().hex
    stored_name = f"{doc_id}_{safe_name}"
    file_path = UPLOAD_DIR / stored_name

    with open(file_path, "wb") as f:
        f.write(content)

    entry = {
        "id": doc_id,
        "filename": safe_name,
        "stored_name": stored_name,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "size": len(content),
    }

    with _lock:
        entries = _read_manifest()
        entries.append(entry)
        _write_manifest(entries)

    return entry


def get_document(doc_id: str) -> dict | None:
    for entry in list_documents():
        if entry["id"] == doc_id:
            return entry
    return None


def get_document_path(doc_id: str) -> Path | None:
    entry = get_document(doc_id)
    if entry is None:
        return None
    path = UPLOAD_DIR / entry["stored_name"]
    return path if path.exists() else None


def delete_document(doc_id: str) -> dict | None:
    """Remove an uploaded PDF from disk and the manifest. Returns the removed entry, or None if not found."""
    with _lock:
        entries = _read_manifest()
        remaining = [e for e in entries if e["id"] != doc_id]
        if len(remaining) == len(entries):
            return None
        removed = next(e for e in entries if e["id"] == doc_id)
        _write_manifest(remaining)

    file_path = UPLOAD_DIR / removed["stored_name"]
    file_path.unlink(missing_ok=True)
    return removed
