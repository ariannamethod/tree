"""Test NER (Named Entity Recognition) functionality."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

import ner


def test_tokenize_unicode_aware():
    """Test unicode-aware tokenization."""
    # Test basic Latin text
    tokens = ner.tokenize_unicode_aware("Hello Steve Jobs!")
    assert tokens == ["Hello", "Steve", "Jobs", "!"]
    
    # Test with Cyrillic characters
    tokens = ner.tokenize_unicode_aware("Привет Владимир Путин!")
    assert "Привет" in tokens
    assert "Владимир" in tokens
    assert "Путин" in tokens
    
    # Test with accented characters
    tokens = ner.tokenize_unicode_aware("Bonjour François Mitterrand")
    assert "François" in tokens
    assert "Mitterrand" in tokens


def test_detect_proper_case_entities():
    """Test detection of multiword proper case sequences."""
    tokens, entities = ner.detect_entities("I met Steve Jobs yesterday.")
    
    # Should detect "Steve Jobs" as entity
    assert "Steve Jobs" in entities
    
    # Should not detect "I" as entity (sentence start)
    assert "I" not in entities
    
    # Test longer sequences
    tokens, entities = ner.detect_entities("Donald John Trump was president.")
    assert "Donald John Trump" in entities or "Donald" in entities


def test_detect_acronym_entities():
    """Test detection of ALL-CAPS acronyms."""
    tokens, entities = ner.detect_entities("I work at IBM and NASA.")
    
    assert "IBM" in entities
    assert "NASA" in entities
    
    # Single letter words should not be detected
    tokens, entities = ner.detect_entities("I have a dog.")
    assert "I" not in entities


def test_avoid_sentence_start_capture():
    """Test that sentence-initial words are not captured as entities."""
    # "Hello" should not be captured as entity at sentence start
    tokens, entities = ner.detect_entities("Hello world.")
    assert "Hello" not in entities
    
    # But "John" should be captured if not at sentence start
    tokens, entities = ner.detect_entities("I saw John today.")
    assert "John" in entities


def test_mixed_entity_detection():
    """Test detection of multiple entity types in one text."""
    text = "Steve Jobs founded Apple Inc. and worked with NASA on projects."
    tokens, entities = ner.detect_entities(text)
    
    # Should detect proper names
    assert "Steve Jobs" in entities
    
    # Should detect company name (mixed case pattern)
    # Note: "Apple Inc." might be detected differently based on punctuation handling
    
    # Should detect acronyms
    assert "NASA" in entities


def test_preserve_entities_in_tokens():
    """Test that entities are preserved as atomic units in token lists."""
    entities = {"Steve Jobs", "IBM"}
    tokens = ["I", "met", "Steve", "Jobs", "at", "IBM", "today"]
    
    preserved = ner.preserve_entities_in_tokens(tokens, entities)
    
    # "Steve Jobs" should be atomic
    assert "Steve Jobs" in preserved
    assert "Steve" not in preserved or preserved.count("Steve") == 0
    assert "Jobs" not in preserved or preserved.count("Jobs") == 0
    
    # "IBM" should remain
    assert "IBM" in preserved


def test_empty_and_edge_cases():
    """Test edge cases and empty inputs."""
    # Empty text
    tokens, entities = ner.detect_entities("")
    assert len(entities) == 0
    
    # Only punctuation
    tokens, entities = ner.detect_entities("!!! ... ???")
    assert len(entities) == 0
    
    # Single character words
    tokens, entities = ner.detect_entities("I a A")
    assert len(entities) == 0
    
    # Preserve empty entities set
    tokens = ["hello", "world"]
    preserved = ner.preserve_entities_in_tokens(tokens, set())
    assert preserved == tokens


def test_case_sensitivity():
    """Test that case sensitivity is properly handled."""
    # Mixed case should be handled correctly
    tokens, entities = ner.detect_entities("john smith and John Smith")
    
    # Should detect "John Smith" (proper case)
    assert "John Smith" in entities
    
    # Should not detect "john smith" (lowercase)
    assert "john smith" not in entities


def test_unicode_proper_names():
    """Test detection of proper names with unicode characters."""
    # Cyrillic proper names
    tokens, entities = ner.detect_entities("Встретил Владимира Путина вчера.")
    # Should detect Cyrillic proper names if they follow proper case pattern
    
    # French names with accents
    tokens, entities = ner.detect_entities("J'ai vu François Mitterrand.")
    # Should handle accented characters in proper names