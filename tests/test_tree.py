from pathlib import Path
import random
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

import roots  # noqa: E402
import tree  # noqa: E402
import guide  # noqa: E402


def test_responses_evolve(monkeypatch):
    # ensure clean memory
    if Path("tree.sqlite").exists():
        Path("tree.sqlite").unlink()
        roots.initialize()
    tree.NGRAMS.clear()

    context_text = "alpha beta gamma alpha delta beta theta beta epsilon zeta eta lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega all unique words"
    words = context_text.lower().split()
    fixed_ctx = tree.Context(
        raw=context_text,
        lower=context_text.lower(),
        words=words,
        unique=["gamma", "theta", "delta", "epsilon", "zeta", "eta", "lambda", "unique"],
        quality_score=1.0,
    )

    monkeypatch.setattr(tree, "_context", lambda msg, w, lang='en': fixed_ctx)
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


def test_guide_integration_how_are_you():
    """Test that guide integration works for 'how are you'."""
    response = tree.respond("how are you")
    assert response
    # Should contain one of the guide templates
    assert any(phrase in response.lower() for phrase in ["present", "listening", "fine", "steady", "quiet", "good", "mind"])


def test_guide_integration_who_are_you():
    """Test that guide integration works for 'who are you'."""
    response = tree.respond("who are you")
    assert response
    # Should contain tree-specific identity
    assert "tree" in response.lower() or "engine" in response.lower() or "echo" in response.lower()


def test_guide_integration_hello():
    """Test that guide integration works for greetings."""
    response = tree.respond("hello")
    assert response
    # Should be a greeting response
    assert any(word in response.lower() for word in ["hello", "hey", "hi", "listening"])


def test_guide_can_be_disabled(monkeypatch):
    """Test that guide can be disabled with feature flag."""
    # Disable guide
    monkeypatch.setattr(tree, "USE_ENGLISH_GUIDE", False)
    
    # Clear guide cache to ensure fresh load
    guide.clear_cache()
    
    # This would normally match the guide
    response = tree.respond("how are you")
    
    # Since guide is disabled, response should come from normal pipeline
    # It might be different from guide templates
    assert response  # Just ensure we get a response


def test_guide_fallback_on_no_match():
    """Test that non-matching inputs fall back to normal pipeline."""
    response = tree.respond("tell me about quantum physics")
    assert response
    # Should get some response even though it doesn't match guide patterns
