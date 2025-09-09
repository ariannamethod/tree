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

    monkeypatch.setattr(tree, "_context", lambda message, words, lang='en': fixed_ctx)
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


def test_select_random_fallback():
    tree.NGRAMS.clear()
    tokens = [
        "africa",
        "arabia",
        "argentina",
        "australia",
        "austria",
        "belgium",
    ]
    random.seed(0)
    result = tree._select(tokens, ["hello"], 0.4, limit=5)
    assert result == [
        "austria",
        "argentina",
        "arabia",
        "africa",
        "belgium",
    ]


def test_short_query_no_recall(monkeypatch):
    def boom(limit):  # pragma: no cover - should not be called
        raise AssertionError(
            "roots.recall should not be invoked for short words"
        )

    monkeypatch.setattr(roots, "recall", boom)
    assert tree._recall_fragment("you") is None


def test_stopword_no_recall(monkeypatch):
    def boom(limit):  # pragma: no cover - should not be called
        raise AssertionError(
            "roots.recall should not be invoked for stopwords"
        )

    monkeypatch.setattr(roots, "recall", boom)
    assert tree._recall_fragment("the") is None
