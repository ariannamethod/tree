import sqlite3
import threading

_DB_PATH = 'roots.db'
_conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
_conn.execute(
    'CREATE TABLE IF NOT EXISTS searches (word TEXT, snippet TEXT)'
)
_lock = threading.Lock()


def store(word: str, snippet: str) -> None:
    """Persist search snippet for a word."""
    with _lock:
        _conn.execute(
            'INSERT INTO searches(word, snippet) VALUES(?, ?)',
            (word, snippet),
        )
        _conn.commit()


def get_memory(word: str) -> list[str]:
    """Retrieve all snippets previously stored for a word."""
    cur = _conn.execute(
        'SELECT snippet FROM searches WHERE word=?', (word,)
    )
    return [row[0] for row in cur.fetchall()]
