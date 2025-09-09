"""Tests to verify hardcoded filler phrase injection has been removed."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

import tree  # noqa: E402


def test_empty_input_returns_empty_string():
    """Empty input should return empty string, not hardcoded phrase."""
    result = tree.respond("")
    assert result == "", f"Expected empty string, got: {repr(result)}"


def test_empty_input_variations():
    """Various empty inputs should all return empty string."""
    empty_inputs = ["", "   ", "\t", "\n", "  \t  \n  "]
    for empty_input in empty_inputs:
        result = tree.respond(empty_input)
        assert result == "", f"Expected empty string for {repr(empty_input)}, got: {repr(result)}"


def test_no_hardcoded_phrases_in_output(monkeypatch):
    """Responses should not contain known hardcoded phrases that were removed."""
    # Mock network requests to return empty to ensure no external content
    monkeypatch.setattr(tree, "_fetch_reddit", lambda query: "")
    monkeypatch.setattr(tree, "_fetch_google", lambda query: "")
    
    # Mock treesoning to return empty
    async def mock_treesoning(message):
        return "", 0.0
    monkeypatch.setattr(tree.treesoning, "get_treesoning_context", mock_treesoning)
    
    # Test various inputs that previously triggered hardcoded responses
    test_inputs = [
        "hello",
        "how are you", 
        "python programming",
        "berlin city",
        "random input"
    ]
    
    forbidden_phrases = [
        "Emptiness speaks",
        "Silence responds", 
        "doing great, thanks for asking",
        "Nice to meet you",
        "What brings you here today",
        "resonates through code",
        "echoes through urban streets", 
        "resonates through conversation",
        "conversation flows naturally",
        "conversation flows through natural dialogue",
        "Silence echoes"
    ]
    
    for test_input in test_inputs:
        result = tree.respond(test_input)
        for forbidden in forbidden_phrases:
            assert forbidden not in result, f"Found forbidden phrase '{forbidden}' in response to '{test_input}': {repr(result)}"


def test_no_fabricated_filler_words(monkeypatch):
    """Responses should not contain fabricated filler words like 'within', 'through', 'inside'."""
    # Mock to return empty contexts so we can test pure source material
    monkeypatch.setattr(tree, "_fetch_reddit", lambda query: "")
    monkeypatch.setattr(tree, "_fetch_google", lambda query: "")
    
    async def mock_treesoning(message):
        return "", 0.0
    monkeypatch.setattr(tree.treesoning, "get_treesoning_context", mock_treesoning)
    
    # Test input that doesn't contain the filler words
    result = tree.respond("apple banana cherry")
    
    # These words should not appear unless they were in the input or fetched content
    fabricated_words = ["within", "through", "inside", "preview", "hear", "aug", "wanted"]
    
    result_words = result.lower().split()
    for word in fabricated_words:
        if word in result_words:
            # Only OK if it was in the original input
            assert word in "apple banana cherry".lower(), f"Fabricated word '{word}' found in response but not in input. Response: {repr(result)}"


def test_empty_context_yields_empty_response(monkeypatch):
    """When no context can be fetched, response should be empty."""
    # Mock everything to return empty
    monkeypatch.setattr(tree, "_fetch_reddit", lambda query: "")
    monkeypatch.setattr(tree, "_fetch_google", lambda query: "")
    monkeypatch.setattr(tree, "_recall_fragment", lambda word, limit=50: None)
    
    async def mock_treesoning(message):
        return "", 0.0
    monkeypatch.setattr(tree.treesoning, "get_treesoning_context", mock_treesoning)
    
    result = tree.respond("nonexistent fakejibberish unknownword")
    # Should be empty when no context is available
    assert result == "", f"Expected empty response when no context available, got: {repr(result)}"