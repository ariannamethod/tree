"""Learning tendrils for the tree engine.

This module performs asynchronous persistence of search windows.  Instead of
spawning a new thread for every :func:`learn` call, tasks are queued and
consumed by a small pool of worker threads.  This keeps the number of threads
bounded while still avoiding blocking the caller.
"""

from __future__ import annotations

import queue
import threading
from typing import Iterable, List, Tuple

import roots

# Fixed number of background workers processing tasks from ``_TASKS``.
WORKER_LIMIT = 1
_TASKS: "queue.Queue[Tuple[List[str], str]]" = queue.Queue()


def _worker() -> None:
    """Consume queued learn tasks one at a time."""
    while True:
        words, context = _TASKS.get()
        try:
            roots.add_memory(words, context)
        finally:
            _TASKS.task_done()


# Start the pool of worker threads on module import.
for i in range(WORKER_LIMIT):
    threading.Thread(
        target=_worker, daemon=True, name=f"branches-worker-{i}"
    ).start()


def learn(words: Iterable[str], context: str) -> None:
    """Asynchronously record *words* with the given *context* window."""

    # Store a concrete sequence in the queue so generators remain valid.
    # Deduplicate while preserving order to keep storage compact.
    seen = []
    for w in words:
        if w not in seen:
            seen.append(w)
    _TASKS.put((seen, context))


def wait() -> None:
    """Block until all queued learning tasks are processed."""

    _TASKS.join()
