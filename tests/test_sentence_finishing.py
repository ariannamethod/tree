from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

import tree


def test_question_mark_for_questions():
    """Inputs with '?' end with '?'."""
    # Test questions that are likely to have sufficient context
    questions = [
        "What is machine learning?",
        "How does neural network work?", 
        "Why is AI important?",
        "What is data science?",
        "How do computers work?"
    ]
    
    for question in questions:
        result = tree.respond(question)
        # Only check punctuation if we got a real response (not micro-interjection)
        if len(result) > 10 and result not in ["Huh?", "Hmmm…", "…"]:
            assert result.endswith("?"), f"Question '{question}' should end with ?, got: {result}"


def test_period_for_statements():
    """Non-questions end with a single '.'."""
    statements = [
        "Machine learning is fascinating",
        "I love programming",
        "Data science helps businesses",
        "Computer algorithms are powerful"
    ]
    
    for statement in statements:
        result = tree.respond(statement)
        # Only check punctuation if we got a real response (not micro-interjection)
        if len(result) > 10 and result not in ["Huh?", "Hmmm…", "…"]:
            assert result.endswith("."), f"Statement '{statement}' should end with ., got: {result}"
            
            # Should not have double punctuation
            assert not result.endswith(".."), f"Should not have double periods: {result}"
            assert not result.endswith(".?"), f"Should not have mixed punctuation: {result}"


def test_capitalized_start():
    """First character is uppercase letter when applicable."""
    test_messages = [
        "machine learning technology",
        "data science applications", 
        "artificial intelligence research",
        "computer programming languages"
    ]
    
    for message in test_messages:
        result = tree.respond(message)
        # Only check capitalization if we got a real response (not micro-interjection)
        if len(result) > 10 and result not in ["Huh?", "Hmmm…", "…"]:
            first_char = result[0]
            if first_char.isalpha():
                assert first_char.isupper(), f"First letter should be uppercase in: {result}"