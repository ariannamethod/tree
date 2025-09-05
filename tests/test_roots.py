import sqlite3
from pathlib import Path

import roots


def setup_module(module):
    if Path("tree.sqlite").exists():
        Path("tree.sqlite").unlink()
    roots.initialize()


def test_initialize_creates_word_index():
    with sqlite3.connect(roots.DB_PATH) as db:
        cur = db.execute("PRAGMA index_list(memory)")
        indexes = [row[1] for row in cur.fetchall()]
    assert "idx_memory_word" in indexes


def test_prune_limits_records():
    roots.MEMORY_LIMIT = 3
    for i in range(5):
        roots.add_memory(f"w{i}", "ctx")
    rows = list(roots.recall(10))
    assert len(rows) == 3
    assert [w for w, _ in rows] == ["w4", "w3", "w2"]
