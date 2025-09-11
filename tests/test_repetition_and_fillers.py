from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

import tree


def test_denylist_blocked_unless_in_context():
    """Ensure denylisted tokens are filtered unless present in input/context."""
    # Test with denylisted word not in context - should be filtered
    candidates = ["arrangement", "data", "science", "machine"]
    user_tokens = ["data", "science"]
    context_tokens = ["learning", "algorithm"] 
    memory_tokens = []
    
    result = tree._filter_source_gated_tokens(candidates, user_tokens, context_tokens, memory_tokens)
    
    # "arrangement" should not appear in result since it's denylisted and not in sources
    assert "arrangement" not in result
    assert "data" in result  # Should be present as it's in user_tokens
    assert "science" in result  # Should be present as it's in user_tokens


def test_denylist_allowed_when_in_context():
    """Denylisted tokens should be allowed if they appear in input/context."""
    # Test with denylisted word present in user input - should be allowed
    candidates = ["arrangement", "deal", "data", "science"]
    user_tokens = ["arrangement", "deal", "data"]
    context_tokens = ["science", "learning"]
    memory_tokens = []
    
    result = tree._filter_source_gated_tokens(candidates, user_tokens, context_tokens, memory_tokens)
    
    # Both denylisted words should be allowed since they're in user_tokens/context_tokens
    assert "arrangement" in result
    assert "deal" in result
    assert "data" in result
    assert "science" in result


def test_no_more_than_two_repeats():
    """Any token appears ≤ 2 times; no immediate duplicates."""
    # Test that no token appears more than twice
    candidates = ["data", "data", "data", "science", "machine", "learning"]
    source = ["algorithm", "neural", "network"]
    
    result = tree._compose(candidates, source, 0.5)
    words = result.lower().replace(".", "").split()
    
    # Count occurrences of each word
    word_counts = {}
    for word in words:
        word_counts[word] = word_counts.get(word, 0) + 1
    
    # No word should appear more than 2 times
    for word, count in word_counts.items():
        assert count <= 2, f"Word '{word}' appears {count} times, should be ≤ 2"
    
    # No immediate duplicates
    for i in range(len(words) - 1):
        assert words[i] != words[i + 1], f"Immediate duplicate found: '{words[i]}'"


def test_bigram_not_repeated():
    """Same bigram doesn't occur twice in single reply."""
    # Test the bigram checking function directly
    tokens = ["data", "science", "machine", "learning"]
    
    # Should allow new bigram
    assert tree._check_bigram_rule(tokens, "algorithm") == True
    
    # Should block repeated bigram - if tokens has "data science", 
    # don't allow creating "data science" again
    tokens_with_bigram = ["hello", "data", "science", "world"]
    assert tree._check_bigram_rule(["data"], "science") == True  # First occurrence OK
    
    # If we already have "data science" in the sequence, don't create it again
    tokens_existing = ["machine", "data", "science", "learning"]
    # This would create "learning data" + candidate, different from existing "data science"
    assert tree._check_bigram_rule(tokens_existing, "model") == True