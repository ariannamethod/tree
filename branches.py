import sqlite3
import threading
import collections
import re

_DB_PATH = 'branches.db'
_conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
_conn.execute(
    'CREATE TABLE IF NOT EXISTS stats (word TEXT PRIMARY KEY, count INTEGER)'
)
_lock = threading.Lock()


def _update_stats(counter: collections.Counter) -> None:
    with _lock:
        for word, c in counter.items():
            cur = _conn.execute(
                'SELECT count FROM stats WHERE word=?', (word,)
            )
            row = cur.fetchone()
            if row:
                _conn.execute(
                    'UPDATE stats SET count=? WHERE word=?',
                    (row[0] + c, word),
                )
            else:
                _conn.execute(
                    'INSERT INTO stats(word, count) VALUES(?, ?)',
                    (word, c),
                )
        _conn.commit()


def learn(text: str) -> None:
    tokens = re.findall(r"\w+", text.lower())
    counter = collections.Counter(tokens)
    _update_stats(counter)


def learn_async(text: str) -> None:
    thread = threading.Thread(target=learn, args=(text,), daemon=True)
    thread.start()


def top_words(limit: int = 10) -> list[str]:
    cur = _conn.execute(
        'SELECT word FROM stats ORDER BY count DESC LIMIT ?',
        (limit,),
    )
    return [row[0] for row in cur.fetchall()]
