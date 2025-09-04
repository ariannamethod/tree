from __future__ import annotations

"""Learning tendrils for the tree engine.

This module performs asynchronous persistence of search windows.  Each call to
:func:`learn` sprouts a background thread that commits the gathered context to
:mod:`roots` without blocking the caller.
"""

import threading
from typing import Iterable

import roots


def learn(words: Iterable[str], context: str) -> None:
    """Asynchronously record *words* with the given *context* window."""

    def task() -> None:
        for w in words:
            roots.add_memory(w, context)

    threading.Thread(target=task, daemon=True).start()
