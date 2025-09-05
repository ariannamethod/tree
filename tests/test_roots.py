from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

import roots  # noqa: E402


def _reset_db():
    if Path("tree.sqlite").exists():
        Path("tree.sqlite").unlink()
    roots.initialize()


def test_recall_semantic_ranks_by_tfidf():
    _reset_db()
    roots.add_memory(["alpha"], "alpha beta")
    roots.add_memory(["beta"], "beta beta beta")
    roots.add_memory(["gamma"], "gamma delta")

    results = roots.recall_semantic(["beta"], limit=2)
    contexts = [r[1] for r in results]
    assert contexts[0] == "beta beta beta"
    assert contexts[1] == "alpha beta"
