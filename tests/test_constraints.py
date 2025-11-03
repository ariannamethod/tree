"""Tests for constraints module."""

import pytest
from constraints import (
    normalize_whitespace,
    capitalize_first_alpha,
    ensure_single_terminal_punct,
    limit_commas,
    finalize_text,
)


class TestNormalizeWhitespace:
    """Tests for normalize_whitespace function."""
    
    def test_collapse_multiple_spaces(self):
        """Multiple spaces should collapse to single space."""
        assert normalize_whitespace("hello  world") == "hello world"
        assert normalize_whitespace("hello   world") == "hello world"
    
    def test_collapse_tabs_and_newlines(self):
        """Tabs and newlines should be normalized to single spaces."""
        assert normalize_whitespace("hello\tworld") == "hello world"
        assert normalize_whitespace("hello\nworld") == "hello world"
        assert normalize_whitespace("hello\n\nworld") == "hello world"
    
    def test_strip_leading_trailing(self):
        """Leading and trailing whitespace should be removed."""
        assert normalize_whitespace("  hello world  ") == "hello world"
        assert normalize_whitespace("\thello world\n") == "hello world"
    
    def test_mixed_whitespace(self):
        """Mixed whitespace types should all collapse."""
        assert normalize_whitespace("hello \t\n world") == "hello world"
    
    def test_empty_string(self):
        """Empty string should remain empty."""
        assert normalize_whitespace("") == ""
        assert normalize_whitespace("   ") == ""


class TestCapitalizeFirstAlpha:
    """Tests for capitalize_first_alpha function."""
    
    def test_capitalize_first_letter(self):
        """First alphabetic character should be capitalized."""
        assert capitalize_first_alpha("hello") == "Hello"
        assert capitalize_first_alpha("world") == "World"
    
    def test_already_capitalized(self):
        """Already capitalized text should remain unchanged."""
        assert capitalize_first_alpha("Hello") == "Hello"
    
    def test_non_alpha_prefix(self):
        """Non-alphabetic characters at start should be preserved."""
        assert capitalize_first_alpha("123hello") == "123Hello"
        assert capitalize_first_alpha("...hello") == "...Hello"
        assert capitalize_first_alpha("  hello") == "  Hello"
    
    def test_unicode_support(self):
        """Should handle unicode characters."""
        assert capitalize_first_alpha("über") == "Über"
        assert capitalize_first_alpha("δοκιμή") == "Δοκιμή"
    
    def test_empty_string(self):
        """Empty string should remain empty."""
        assert capitalize_first_alpha("") == ""


class TestEnsureSingleTerminalPunct:
    """Tests for ensure_single_terminal_punct function."""
    
    def test_double_period(self):
        """Double periods should become single period."""
        assert ensure_single_terminal_punct("hello..") == "hello."
        assert ensure_single_terminal_punct("world...") == "world."
    
    def test_double_question(self):
        """Double question marks should become single question mark."""
        assert ensure_single_terminal_punct("why??") == "why?"
        assert ensure_single_terminal_punct("what???") == "what?"
    
    def test_double_exclamation(self):
        """Double exclamation marks should become single exclamation."""
        assert ensure_single_terminal_punct("wow!!") == "wow!"
        assert ensure_single_terminal_punct("amazing!!!") == "amazing!"
    
    def test_mixed_terminal_punct(self):
        """Mixed terminal punctuation should keep first type."""
        assert ensure_single_terminal_punct("hello.?") == "hello."
        assert ensure_single_terminal_punct("what?!") == "what?"
    
    def test_no_terminal_punct(self):
        """Text without terminal punctuation should get a period."""
        assert ensure_single_terminal_punct("hello world") == "hello world."
        assert ensure_single_terminal_punct("test") == "test."
    
    def test_single_punct_unchanged(self):
        """Single terminal punctuation should remain unchanged."""
        assert ensure_single_terminal_punct("hello.") == "hello."
        assert ensure_single_terminal_punct("why?") == "why?"
        assert ensure_single_terminal_punct("wow!") == "wow!"
    
    def test_empty_string(self):
        """Empty string should get a period."""
        assert ensure_single_terminal_punct("") == "."


class TestLimitCommas:
    """Tests for limit_commas function."""
    
    def test_no_commas(self):
        """Text without commas should remain unchanged."""
        assert limit_commas("hello world") == "hello world"
    
    def test_within_limit(self):
        """Text with commas within limit should remain unchanged."""
        assert limit_commas("one, two") == "one, two"
        assert limit_commas("one, two, three") == "one, two, three"
    
    def test_exceed_limit(self):
        """Excess commas should be dropped."""
        result = limit_commas("a, b, c, d, e", max_commas=2)
        # Should keep first 2 commas
        assert result.count(",") == 2
        assert result.startswith("a, b, c")
    
    def test_custom_limit(self):
        """Custom comma limit should be respected."""
        result = limit_commas("a, b, c, d", max_commas=1)
        assert result.count(",") == 1
    
    def test_default_limit(self):
        """Default limit should be 2 commas."""
        result = limit_commas("a, b, c, d, e, f")
        assert result.count(",") == 2


class TestFinalizeText:
    """Tests for finalize_text function - the full pipeline."""
    
    def test_full_pipeline(self):
        """All transformations should be applied."""
        input_text = "  hello  world..  "
        expected = "Hello world."
        assert finalize_text(input_text) == expected
    
    def test_whitespace_and_capitalization(self):
        """Whitespace normalization and capitalization."""
        assert finalize_text("  test  message  ") == "Test message."
    
    def test_double_punct_normalized(self):
        """Double punctuation should be normalized."""
        assert finalize_text("hello world??") == "Hello world?"
        assert finalize_text("amazing!!") == "Amazing!"
    
    def test_excess_commas_limited(self):
        """Excess commas should be limited to 2."""
        result = finalize_text("a, b, c, d, e, f")
        assert result.count(",") == 2
        assert result[0].isupper()  # Should be capitalized
        assert result.endswith(".")  # Should have terminal punct
    
    def test_question_preserved(self):
        """Question marks should be preserved through pipeline."""
        assert finalize_text("  why  not?? ") == "Why not?"
    
    def test_empty_input(self):
        """Empty input should produce a period."""
        assert finalize_text("") == "."
        assert finalize_text("   ") == "."
