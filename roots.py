from __future__ import annotations

"""Minimal resonant memory for the tree engine.

This module persists search contexts so the system can grow roots with
experience.  It stores every searched word together with the harvested
textual window.  The database is intentionally tiny and relies only on the
standard library so the engine can run on the humblest CPU.
"""

from contextlib import closing
from pathlib import Path
from collections import Counter
import math
import re
import sqlite3
from typing import Iterable, List, Tuple

DB_PATH = Path("tree.sqlite")


def initialize() -> None:
    """Ensure the sqlite database exists with the required tables."""
    with closing(sqlite3.connect(DB_PATH)) as db:
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                words TEXT,
                context TEXT,
                created REAL DEFAULT (strftime('%s','now'))
            )
            """
        )
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS word_df (
                word TEXT PRIMARY KEY,
                df INTEGER
            )
            """
        )
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_tf (
                memory_id INTEGER,
                word TEXT,
                weight REAL,
                FOREIGN KEY(memory_id) REFERENCES memory(id)
            )
            """
        )
        db.commit()


def _tokenize(text: str) -> List[str]:
    """Split *text* into normalized word tokens."""
    return re.findall(r"\w+", text.lower())


def add_memory(words: Iterable[str], context: str) -> None:
    """Persist *words* with their *context* window and TF-IDF weights."""
    tokens = _tokenize(context)
    if not tokens:
        return

    unique_words = " ".join(sorted(set(words)))

    with closing(sqlite3.connect(DB_PATH)) as db:
        cur = db.execute(
            "INSERT INTO memory(words, context) VALUES (?, ?)",
            (unique_words, context[:2000]),
        )
        memory_id = cur.lastrowid

        counts = Counter(tokens)
        total_terms = sum(counts.values())

        # Update document frequencies for words appearing in this context
        for w in counts:
            db.execute(
                "INSERT INTO word_df(word, df) VALUES(?, 1) "
                "ON CONFLICT(word) DO UPDATE SET df = df + 1",
                (w,),
            )

        # Total number of documents including the one just inserted
        total_docs = db.execute(
            "SELECT COUNT(*) FROM memory"
        ).fetchone()[0]

        for w, c in counts.items():
            df = db.execute(
                "SELECT df FROM word_df WHERE word = ?",
                (w,),
            ).fetchone()[0]
            tf = c / total_terms
            idf = math.log(total_docs / df) + 1.0
            weight = tf * idf
            db.execute(
                "INSERT INTO memory_tf(memory_id, word, weight) "
                "VALUES (?, ?, ?)",
                (memory_id, w, weight),
            )

        db.commit()


def recall(limit: int = 5) -> Iterable[Tuple[str, str]]:
    """Return the most recent memories for inspection."""
    with closing(sqlite3.connect(DB_PATH)) as db:
        cur = db.execute(
            "SELECT words, context FROM memory ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        return cur.fetchall()


def recall_semantic(
    query_words: Iterable[str], limit: int = 5
) -> Iterable[Tuple[str, str]]:
    """Return memories ranked by TF-IDF relevance to *query_words*."""
    words = [w.lower() for w in query_words]
    if not words:
        return []

    placeholders = ",".join("?" * len(words))
    with closing(sqlite3.connect(DB_PATH)) as db:
        cur = db.execute(
            f"""
            SELECT m.words, m.context, SUM(t.weight) AS score
            FROM memory m
            JOIN memory_tf t ON m.id = t.memory_id
            WHERE t.word IN ({placeholders})
            GROUP BY m.id
            ORDER BY score DESC
            LIMIT ?
            """,
            (*words, limit),
        )
        return [(row[0], row[1]) for row in cur.fetchall()]


# Create the database on import so roots exist immediately.
initialize()
