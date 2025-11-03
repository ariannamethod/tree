"""Tests for verbgraph module."""

import pytest
from verbgraph import VerbGraph


class TestVerbGraph:
    """Tests for VerbGraph class."""
    
    def test_initialization(self):
        """VerbGraph should initialize empty."""
        vg = VerbGraph()
        assert vg.preferred_punct("unknown") == "."
    
    def test_add_single_sentence(self):
        """Adding a sentence should track verb-punctuation association."""
        vg = VerbGraph()
        vg.add_sentence("I run!")
        assert vg.preferred_punct("run") == "!"
    
    def test_add_multiple_same_verb(self):
        """Multiple observations should influence preference."""
        vg = VerbGraph()
        vg.add_sentence("I run!")
        vg.add_sentence("They run!")
        vg.add_sentence("We run.")
        # '!' appears twice, '.' appears once
        assert vg.preferred_punct("run") == "!"
    
    def test_majority_wins(self):
        """The most common punctuation should win."""
        vg = VerbGraph()
        vg.add_sentence("She walks.")
        vg.add_sentence("He walks.")
        vg.add_sentence("They walks.")
        vg.add_sentence("I walks!")
        # '.' appears 3 times, '!' appears once
        assert vg.preferred_punct("walks") == "."
    
    def test_default_fallback(self):
        """Unknown verbs should default to period."""
        vg = VerbGraph()
        vg.add_sentence("I run!")
        assert vg.preferred_punct("jump") == "."
        assert vg.preferred_punct("unknown") == "."
    
    def test_case_insensitive(self):
        """Verb matching should be case-insensitive."""
        vg = VerbGraph()
        vg.add_sentence("I RUN!")
        assert vg.preferred_punct("run") == "!"
        assert vg.preferred_punct("Run") == "!"
        assert vg.preferred_punct("RUN") == "!"
    
    def test_question_marks(self):
        """Question marks should be tracked."""
        vg = VerbGraph()
        vg.add_sentence("Do you think?")
        assert vg.preferred_punct("think") == "?"
    
    def test_mixed_punctuation_types(self):
        """Different punctuation types should be tracked separately."""
        vg = VerbGraph()
        vg.add_sentence("I jump!")
        vg.add_sentence("You jump?")
        vg.add_sentence("We jump.")
        # Each appears once, but most_common will return one
        result = vg.preferred_punct("jump")
        assert result in ".!?"
    
    def test_multiple_words_in_sentence(self):
        """All words in sentence should be tracked."""
        vg = VerbGraph()
        vg.add_sentence("The cat sleeps quietly.")
        assert vg.preferred_punct("cat") == "."
        assert vg.preferred_punct("sleeps") == "."
        assert vg.preferred_punct("quietly") == "."
    
    def test_unicode_support(self):
        """Should handle unicode text."""
        vg = VerbGraph()
        vg.add_sentence("Ich laufe schnell!")
        assert vg.preferred_punct("laufe") == "!"
        assert vg.preferred_punct("schnell") == "!"
    
    def test_no_terminal_punct(self):
        """Sentences without terminal punctuation default to period."""
        vg = VerbGraph()
        vg.add_sentence("I walk")
        assert vg.preferred_punct("walk") == "."
    
    def test_empty_sentence(self):
        """Empty sentences should be handled gracefully."""
        vg = VerbGraph()
        vg.add_sentence("")
        assert vg.preferred_punct("any") == "."
    
    def test_tokenization(self):
        """Tokenizer should handle various inputs."""
        vg = VerbGraph()
        # Test with punctuation mid-sentence
        vg.add_sentence("Hello, world!")
        assert vg.preferred_punct("hello") == "!"
        assert vg.preferred_punct("world") == "!"
