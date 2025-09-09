from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

import tree  # noqa: E402


def test_sanitize_visible_basic():
    """Test basic sanitizer functionality."""
    # Test HTML unescaping
    text = "&quot;Hello&quot; &amp; goodbye &lt;test&gt;"
    result = tree._sanitize_visible(text)
    expected = '"Hello" & goodbye <test>'
    assert result == expected


def test_sanitize_visible_control_characters():
    """Test removal of invisible/control characters."""
    # Include various control characters (Cc/Cf/Cs/Co/Cn categories)
    text = "Hello\u0000\u0001\u0002World"  # Null, SOH, STX (Cc)
    result = tree._sanitize_visible(text)
    assert result == "HelloWorld"
    
    # Test Zero Width Space (ZWSP), Zero Width Joiner (ZWJ), etc.
    text = "Hello\u200B\u200C\u200D\u200EWorld"  # ZWSP, ZWNJ, ZWJ, LRM (Cf)
    result = tree._sanitize_visible(text)
    assert result == "HelloWorld"
    
    # Test BOM (Byte Order Mark)
    text = "\uFEFFHello World\uFEFF"  # BOM at start and end (Cf)
    result = tree._sanitize_visible(text)
    assert result == "Hello World"


def test_sanitize_visible_unicode_spaces():
    """Test conversion of Unicode space separators to standard space."""
    # Test various Unicode space characters (Zs category)
    text = "Hello\u00A0\u2000\u2001\u2002\u2003World"  # NBSP, EN QUAD, EM QUAD, etc.
    result = tree._sanitize_visible(text)
    assert result == "Hello World"  # All converted to regular spaces and collapsed
    
    # Test that individual Unicode spaces are converted
    text = "Hello\u00A0World"  # NBSP
    result = tree._sanitize_visible(text)
    assert result == "Hello World"


def test_sanitize_visible_preserve_punctuation():
    """Test that punctuation is preserved across languages."""
    # Test basic punctuation marks (P* categories)
    text = "Hello! How are you? I'm fine... Really?! Yes."
    result = tree._sanitize_visible(text)
    assert result == text  # Should be unchanged
    
    # Test that NFKC normalization converts fullwidth punctuation to ASCII equivalents
    # This is actually desirable for cleaning external content
    text = "Hello—world…testing！？"  # Em dash, ellipsis, fullwidth exclamation/question
    result = tree._sanitize_visible(text)
    expected = "Hello—world...testing!?"  # NFKC normalizes fullwidth to ASCII
    assert result == expected


def test_sanitize_visible_preserve_multilingual():
    """Test that letters/marks/numbers across languages are preserved."""
    # Test various scripts
    text = "Hello עולם Мир 世界 مرحبا"  # English, Hebrew, Russian, Chinese, Arabic
    result = tree._sanitize_visible(text)
    assert result == text  # Should preserve all text
    
    # Test with combining marks
    text = "café naïve résumé"  # Letters with combining marks
    result = tree._sanitize_visible(text)
    assert result == text  # Should preserve combining marks


def test_sanitize_visible_whitespace_collapse():
    """Test whitespace collapsing functionality."""
    text = "Hello    \t\n\r   World"
    result = tree._sanitize_visible(text)
    assert result == "Hello World"
    
    # Test leading/trailing whitespace removal
    text = "   Hello World   "
    result = tree._sanitize_visible(text)
    assert result == "Hello World"


def test_sanitize_visible_empty_and_edge_cases():
    """Test edge cases."""
    # Empty string
    assert tree._sanitize_visible("") == ""
    
    # Only whitespace
    assert tree._sanitize_visible("   \t\n  ") == ""
    
    # Only control characters
    assert tree._sanitize_visible("\u0000\u0001\u200B\u200C") == ""
    
    # Mixed valid and invalid
    text = "\u0000Hello\u200B World\u0001"
    result = tree._sanitize_visible(text)
    assert result == "Hello World"


def test_sanitize_visible_unicode_normalization():
    """Test Unicode NFKC normalization."""
    # Test that NFKC normalization is applied
    # Using decomposed vs precomposed characters
    text1 = "café"  # precomposed é
    text2 = "cafe\u0301"  # e + combining acute accent
    
    result1 = tree._sanitize_visible(text1)
    result2 = tree._sanitize_visible(text2)
    
    # Both should normalize to the same result
    assert result1 == result2 == "café"


def test_sanitize_visible_integration():
    """Test integration with mixed content like external snippets might have."""
    # Simulate content that might come from external sources
    text = "&quot;Hello\u200B\u00A0World&quot;\u0000\t\n  with\u200C\u200D mixed   content!"
    result = tree._sanitize_visible(text)
    expected = '"Hello World" with mixed content!'
    assert result == expected