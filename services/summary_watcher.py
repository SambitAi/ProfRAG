from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

from core.metadata import load_metadata
from services import summarize_document

logger = logging.getLogger(__name__)

_WATCHER_THREAD: threading.Thread | None = None
_STOP_EVENT = threading.Event()


def _candidate_docs(artifacts_root: Path) -> list[Path]:
    if not artifacts_root.exists():
        return []
    out: list[Path] = []
    for p in artifacts_root.iterdir():
        if not p.is_dir():
            continue
        m = p / "metadata.json"
        if not m.exists():
            continue
        meta = load_metadata(p)
        if meta.get("ready_to_chat") and meta.get("summary_status") in ("pending", "in_progress", None):
            out.append(p)
    return out


def _loop(config: dict[str, Any], stop_event: threading.Event) -> None:
    artifacts_root = Path(config["paths"]["artifacts_root"])
    interval = int(config.get("summary_watcher", {}).get("scan_interval_seconds", 60))
    max_workers = int(config.get("summary_watcher", {}).get("max_workers", 2))
    sem = threading.BoundedSemaphore(value=max_workers)

    def _run_one(folder: Path) -> None:
        if not sem.acquire(blocking=False):
            return
        try:
            summarize_document.run(folder, config)
        except Exception:
            logger.exception("Summary watcher failed for %s", folder)
        finally:
            sem.release()

    while not stop_event.is_set():
        for folder in _candidate_docs(artifacts_root):
            if stop_event.is_set():
                break
            t = threading.Thread(target=_run_one, args=(folder,), daemon=True)
            t.start()
        stop_event.wait(interval)


def start(config: dict[str, Any]) -> None:
    global _WATCHER_THREAD
    if _WATCHER_THREAD and _WATCHER_THREAD.is_alive():
        return
    _STOP_EVENT.clear()
    _WATCHER_THREAD = threading.Thread(target=_loop, args=(config, _STOP_EVENT), daemon=True)
    _WATCHER_THREAD.start()


def stop() -> None:
    _STOP_EVENT.set()

