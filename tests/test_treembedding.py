"""Test treembedding functionality."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

import treembedding


def test_treembedding_initialization():
    """Test that TreeEmbedding initializes correctly."""
    embedding = treembedding.TreeEmbedding()
    
    # Should have empty initial state
    stats = embedding.get_embedding_stats()
    assert stats['total_sequences'] == 0
    assert stats['unique_tokens'] == 0
    assert stats['total_cooccurrences'] == 0


def test_update_sequence():
    """Test sequence updating functionality."""
    embedding = treembedding.TreeEmbedding()
    
    tokens = ["hello", "world", "test"]
    embedding.update_sequence(tokens)
    
    stats = embedding.get_embedding_stats()
    assert stats['total_sequences'] == 1
    assert stats['unique_tokens'] == 3
    
    # Should track co-occurrences
    assert stats['total_cooccurrences'] > 0


def test_suggest_next():
    """Test next token suggestion functionality."""
    embedding = treembedding.TreeEmbedding()
    
    # Train with some sequences
    embedding.update_sequence(["hello", "world"])
    embedding.update_sequence(["hello", "there"])
    
    candidates = ["world", "there", "everyone"]
    suggestions = embedding.suggest_next("hello", candidates)
    
    # Should return candidates (maintaining compatibility)
    assert len(suggestions) == len(candidates)
    assert all(c in suggestions for c in candidates)


def test_empty_inputs():
    """Test handling of empty inputs."""
    embedding = treembedding.TreeEmbedding()
    
    # Empty sequence
    embedding.update_sequence([])
    stats = embedding.get_embedding_stats()
    assert stats['total_sequences'] == 1  # Should still count as a sequence
    assert stats['unique_tokens'] == 0
    
    # Empty candidates
    suggestions = embedding.suggest_next("hello", [])
    assert suggestions == []


def test_multiple_sequences():
    """Test learning from multiple sequences."""
    embedding = treembedding.TreeEmbedding()
    
    sequences = [
        ["apple", "pie", "is", "delicious"],
        ["apple", "juice", "is", "sweet"],
        ["orange", "juice", "is", "tangy"]
    ]
    
    for seq in sequences:
        embedding.update_sequence(seq)
    
    stats = embedding.get_embedding_stats()
    assert stats['total_sequences'] == 3
    assert stats['unique_tokens'] > 0
    
    # Should have learned co-occurrences
    candidates = ["juice", "pie", "water"]
    suggestions = embedding.suggest_next("apple", candidates)
    assert len(suggestions) == len(candidates)


def test_global_embedding_instance():
    """Test that the global embedding instance works."""
    # Should be able to access the global instance
    embedding = treembedding.embedding_system
    
    # Should be a TreeEmbedding instance
    assert isinstance(embedding, treembedding.TreeEmbedding)
    
    # Should be usable
    embedding.update_sequence(["test", "global", "instance"])
    stats = embedding.get_embedding_stats()
    assert stats['total_sequences'] >= 1


def test_frequency_tracking():
    """Test that token frequencies are tracked correctly."""
    embedding = treembedding.TreeEmbedding()
    
    # Add sequences with repeated tokens
    embedding.update_sequence(["apple", "apple", "banana"])
    embedding.update_sequence(["apple", "cherry"])
    
    stats = embedding.get_embedding_stats()
    
    # Should track most frequent tokens
    most_frequent = stats['most_frequent_tokens']
    assert len(most_frequent) > 0
    
    # Apple should be most frequent (appears 3 times)
    top_token, top_count = most_frequent[0]
    assert top_token == "apple"
    assert top_count == 3


def test_cooccurrence_tracking():
    """Test that co-occurrence relationships are tracked."""
    embedding = treembedding.TreeEmbedding()
    
    # Add sequence where "hello" is always followed by "world"
    embedding.update_sequence(["hello", "world"])
    embedding.update_sequence(["hello", "world", "again"])
    
    # Check that co-occurrences are being tracked
    stats = embedding.get_embedding_stats()
    assert stats['total_cooccurrences'] > 0
    
    # The internal tracking should show hello->world relationship
    # (This tests the internal structure - in a real implementation,
    # we might expose this through a public method)
    assert "hello" in embedding._cooccurrence_counts
    assert "world" in embedding._cooccurrence_counts["hello"]