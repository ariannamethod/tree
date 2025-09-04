from pathlib import Path
import random
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

import roots  # noqa: E402
import tree  # noqa: E402


def test_responses_evolve(monkeypatch):
    # ensure clean memory
    if Path("tree.sqlite").exists():
        Path("tree.sqlite").unlink()
        roots.initialize()
    tree.NGRAMS.clear()

    context_text = "alpha beta gamma alpha delta beta theta beta"
    words = context_text.lower().split()
    fixed_ctx = tree.Context(
        raw=context_text,
        lower=context_text.lower(),
        words=words,
        unique=["gamma", "theta", "delta"],
        quality_score=1.0,
    )

    monkeypatch.setattr(tree, "_context", lambda _: fixed_ctx)
    monkeypatch.setattr(
        tree.branches,
        "learn",
        lambda w, c: [roots.add_memory(k, c) for k in w],
    )

    random.seed(0)
    first = tree.respond("alpha beta")
    random.seed(0)
    second = tree.respond("alpha beta")
    assert first != second
