from __future__ import annotations

"""Minimal resonant memory for the tree engine.

This module persists search contexts so the system can grow roots with
experience.  It stores every searched word together with the harvested
textual window.  The database is intentionally tiny and relies only on the
standard library so the engine can run on the humblest CPU.
"""

from contextlib import closing
from pathlib import Path
import sqlite3
from typing import Iterable, Tuple

DB_PATH = Path("tree.sqlite")
MEMORY_LIMIT = 1000


def initialize() -> None:
    """Ensure the sqlite database exists with the required table."""
    with closing(sqlite3.connect(DB_PATH)) as db:
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                word TEXT,
                context TEXT,
                created REAL DEFAULT (strftime('%s','now'))
            )
            """
        )
        db.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_memory_word ON memory(word)
            """
        )
        db.commit()


def prune(limit: int = MEMORY_LIMIT) -> None:
    """Trim oldest rows so no more than *limit* memories remain."""
    with closing(sqlite3.connect(DB_PATH)) as db:
        cur = db.execute(
            "SELECT id FROM memory ORDER BY id DESC LIMIT 1 OFFSET ?", (limit,)
        )
        row = cur.fetchone()
        if row:
            cutoff_id = row[0]
            db.execute("DELETE FROM memory WHERE id <= ?", (cutoff_id,))
            db.commit()


def add_memory(word: str, context: str) -> None:
    """Persist a searched *word* with its *context* window."""
    with closing(sqlite3.connect(DB_PATH)) as db:
        db.execute(
            "INSERT INTO memory(word, context) VALUES (?, ?)",
            (word, context[:2000]),
        )
        db.commit()
    prune(MEMORY_LIMIT)


def recall(limit: int = 5) -> Iterable[Tuple[str, str]]:
    """Return the most recent memories for inspection."""
    with closing(sqlite3.connect(DB_PATH)) as db:
        cur = db.execute(
            "SELECT word, context FROM memory ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        return cur.fetchall()


# Create the database on import so roots exist immediately.
initialize()
