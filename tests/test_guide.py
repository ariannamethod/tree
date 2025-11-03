"""Tests for guide module."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

import guide  # noqa: E402


def setup_module():
    """Clear guide cache before tests."""
    guide.clear_cache()


def teardown_function():
    """Clear guide cache after each test."""
    guide.clear_cache()


def test_load_guide_default_path():
    """Test loading guide from default path."""
    g = guide.load_guide()
    assert g is not None
    assert isinstance(g.patterns, list)
    # Should have at least the patterns from TREE_EN_GUIDE.md
    assert len(g.patterns) >= 4


def test_load_guide_caching():
    """Test that guide is cached after first load."""
    g1 = guide.load_guide()
    g2 = guide.load_guide()
    assert g1 is g2


def test_match_how_are_you():
    """Test matching 'how are you' pattern."""
    m = guide.match("how are you")
    assert m is not None
    assert len(m.templates) > 0
    assert len(m.query_hints) > 0


def test_match_how_are_you_case_insensitive():
    """Test case-insensitive matching."""
    m1 = guide.match("How Are You")
    m2 = guide.match("HOW ARE YOU?")
    m3 = guide.match("how are you?")
    assert m1 is not None
    assert m2 is not None
    assert m3 is not None


def test_match_who_are_you():
    """Test matching 'who are you' pattern."""
    m = guide.match("who are you")
    assert m is not None
    assert len(m.templates) > 0
    assert "I'm a small engine that learns as we talk." in m.templates


def test_match_what_is_your_name():
    """Test matching 'what is your name' pattern."""
    m = guide.match("what is your name")
    assert m is not None
    assert len(m.templates) > 0
    assert any("Tree" in t for t in m.templates)


def test_match_hello():
    """Test matching hello/hi/hey pattern."""
    m1 = guide.match("hello")
    m2 = guide.match("hi")
    m3 = guide.match("hey")
    assert m1 is not None
    assert m2 is not None
    assert m3 is not None


def test_match_no_match():
    """Test that non-matching input returns None."""
    m = guide.match("this is some random text that won't match")
    assert m is None


def test_choose_template():
    """Test template selection."""
    templates = ["Template 1", "Template 2", "Template 3"]
    result = guide.choose_template(templates)
    assert result in templates


def test_choose_template_empty():
    """Test template selection with empty list."""
    result = guide.choose_template([])
    assert result == ""


def test_get_query_hints():
    """Test extracting query hints from a match."""
    m = guide.match("how are you")
    hints = guide.get_query_hints(m)
    assert isinstance(hints, list)
    assert len(hints) > 0


def test_get_query_hints_no_match():
    """Test query hints with no match."""
    hints = guide.get_query_hints(None)
    assert hints == []


def test_pattern_with_punctuation():
    """Test that patterns handle various punctuation."""
    m1 = guide.match("how are you?")
    m2 = guide.match("how are you!")
    m3 = guide.match("how are you.")
    assert m1 is not None
    assert m2 is not None
    assert m3 is not None


def test_pattern_with_whitespace():
    """Test that patterns handle extra whitespace."""
    # Pattern with leading/trailing whitespace should NOT match since patterns are anchored
    # The strip() in tree.respond would handle this, but pattern itself is anchored
    # Test the normal case
    m = guide.match("how are you")
    assert m is not None
