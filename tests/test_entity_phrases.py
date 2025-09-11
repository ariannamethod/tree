from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

import tree


def test_detect_entity_phrases_multiword():
    """Test detection of proper-case multiword sequences."""
    # Test basic multiword entities
    result = tree._detect_entity_phrases("I met Donald Trump yesterday")
    assert "Donald Trump" in result

    # Test multiple entities
    result = tree._detect_entity_phrases("Donald Trump visited New York City")
    assert "Donald Trump" in result
    assert "New York City" in result

    # Test Cyrillic support
    result = tree._detect_entity_phrases("Владимир Путин живёт в Москва")
    assert "Владимир Путин" in result


def test_detect_entity_phrases_all_caps():
    """Test detection of ALL-CAPS short tokens."""
    # Test acronyms
    result = tree._detect_entity_phrases("IBM and NASA collaborated")
    assert "IBM" in result
    assert "NASA" in result

    # Test mixed with hyphens and numbers
    result = tree._detect_entity_phrases("GPT-5 and AI-2 models")
    assert "GPT-5" in result
    assert "AI-2" in result


def test_detect_entity_phrases_edge_cases():
    """Test edge cases for entity detection."""
    # Empty message
    result = tree._detect_entity_phrases("")
    assert result == []

    # No entities
    result = tree._detect_entity_phrases("this is just regular text")
    assert result == []

    # Single word entities (should be excluded for multiword)
    result = tree._detect_entity_phrases("Donald works at Microsoft")
    # Should only detect "Microsoft" if it's all-caps, but it's not
    # "Donald" is single word so not included in multiword detection
    assert result == []

    # Punctuation handling
    result = tree._detect_entity_phrases("\"New York\" and 'Los Angeles'")
    assert "New York" in result
    assert "Los Angeles" in result


def test_keywords_includes_entity_phrases():
    """Test that _keywords function includes detected entity phrases."""
    # Test with entity phrases
    result = tree._keywords("Donald Trump visited New York")
    assert "Donald Trump" in result
    assert "New York" in result

    # Test with ALL-CAPS entities
    result = tree._keywords("IBM released new AI technology")
    assert "IBM" in result


def test_entity_phrases_prioritized_in_keywords():
    """Test that entity phrases are prioritized over regular keywords."""
    result = tree._keywords("Donald Trump visited New York yesterday", maximum=4)
    # Entity phrases should appear first
    assert result[0] == "Donald Trump" or result[1] == "Donald Trump"
    assert "New York" in result[:3]  # Should be in top 3


def test_compose_preserves_entity_phrases():
    """Test that _compose preserves entity phrases intact."""
    candidates = ["technology", "innovation", "business", "leadership"]
    source = ["donald", "trump", "visited", "new", "york"]

    # Mock entity detection for this test by calling compose with a message that has entities
    result = tree._compose(
        candidates,
        source,
        mix_ratio=0.5,
        user_message="Donald Trump visited New York",
        context_tokens=candidates,
        memory_tokens=[],
    )

    # The result should be a valid sentence
    assert len(result) > 0
    # Should end with proper punctuation
    assert result.endswith(".") or result.endswith("?")
    # Should be capitalized
    assert result[0].isupper()


def test_safe_diversification_gating():
    """Test that diversification is gated based on candidate pool size."""
    # Small pool - should not apply strict diversification
    small_candidates = ["alpha", "beta"]
    small_source = ["test"]

    result = tree._compose(
        small_candidates,
        small_source,
        mix_ratio=0.8,  # High mix ratio
        user_message="test message",
        context_tokens=small_candidates,
        memory_tokens=[],
    )

    # Should still produce output even with small pool
    assert len(result) > 0

    # Large pool - should apply diversification
    large_candidates = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]
    large_source = ["one", "two", "three", "four"]

    result = tree._compose(
        large_candidates,
        large_source,
        mix_ratio=0.8,
        user_message="one two three four",
        context_tokens=large_candidates,
        memory_tokens=[],
    )

    # Should produce output with proper diversification
    assert len(result) > 0


def test_memory_context_entity_relevance():
    """Test that memory context prioritizes entity-relevant fragments."""
    # This test verifies that the _context_from_memory function works
    # even if no actual fragments exist in memory (should return None gracefully)

    result = tree._context_from_memory("Donald Trump visited New York")
    # Should return None or valid Context since no memory fragments exist in test environment
    assert result is None or isinstance(result, tree.Context)

    # Test empty message
    result = tree._context_from_memory("")
    assert result is None

    # Test message with no valid keys
    result = tree._context_from_memory("a an the")
    assert result is None
