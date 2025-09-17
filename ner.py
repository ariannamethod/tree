"""Heuristic Named Entity Recognition (NER) module.

This module provides unicode-aware tokenization and entity detection
for preserving proper nouns and acronyms during text generation.
"""

import re
from typing import List, Set, Tuple


def _is_proper_case(word: str) -> bool:
    """Check if a word follows proper case (first letter uppercase, rest lowercase)."""
    if len(word) < 2:
        return False
    return word[0].isupper() and word[1:].islower()


def _is_all_caps(word: str) -> bool:
    """Check if a word is all uppercase letters (acronym pattern)."""
    return len(word) >= 2 and word.isupper() and word.isalpha()


def _is_sentence_start(position: int, tokens: List[str]) -> bool:
    """Check if this token position could be a sentence start."""
    if position == 0:
        return True

    # Look for sentence-ending punctuation in previous tokens
    for i in range(position - 1, -1, -1):
        token = tokens[i]
        if token.endswith((".", "!", "?")):
            return True
        # If we find a non-punctuation/non-space token, it's not sentence start
        if token and not re.match(r"^[^\w]*$", token):
            return False

    return False


def tokenize_unicode_aware(text: str) -> List[str]:
    """Unicode-aware tokenization supporting Latin, Cyrillic, and accented."""
    # Split on whitespace and punctuation, but preserve word boundaries
    # This regex captures word characters including unicode letters
    tokens = re.findall(r"\b\w+\b|[^\w\s]", text, re.UNICODE)
    return tokens


def detect_entities(text: str) -> Tuple[List[str], Set[str]]:
    """Detect named entities in text using heuristic patterns.

    Returns:
        Tuple of (tokens_list, detected_entities_set)
    """
    tokens = tokenize_unicode_aware(text)
    entities = set()

    i = 0
    while i < len(tokens):
        token = tokens[i]

        # Skip if it's punctuation or not a word
        if not re.match(r"\w+", token):
            i += 1
            continue

        # Check for ALL-CAPS acronyms
        if _is_all_caps(token):
            entities.add(token)
            i += 1
            continue

        # Check for proper case sequences (multiword proper nouns)
        if _is_proper_case(token):
            # Look ahead to build multiword proper noun
            entity_tokens = [token]
            j = i + 1

            # Continue collecting proper case words
            while j < len(tokens):
                if j >= len(tokens):
                    break

                next_token = tokens[j]
                # Skip punctuation and whitespace, but stop at sentence boundaries
                if not re.match(r"\w+", next_token):
                    j += 1
                    # Stop if we hit sentence-ending punctuation
                    if next_token in (".", "!", "?"):
                        break
                    continue

                if _is_proper_case(next_token):
                    entity_tokens.append(next_token)
                    j += 1
                else:
                    break

            # Only add as entity if:
            # 1. It's a multiword entity (len > 1), OR
            # 2. It's a single word that's not at sentence start
            if len(entity_tokens) > 1 or not _is_sentence_start(i, tokens):
                entity = " ".join(entity_tokens)
                entities.add(entity)

            # Move index to after the processed tokens
            i = j
        else:
            i += 1

    return tokens, entities


def preserve_entities_in_tokens(tokens: List[str], entities: Set[str]) -> List[str]:
    """Process token list to preserve detected entities as atomic units.

    This function ensures that detected entities are treated as single tokens
    during generation and ranking processes.
    """
    if not entities:
        return tokens

    preserved_tokens = []
    i = 0

    while i < len(tokens):
        # Try to match entities starting at this position
        matched_entity = None
        max_length = 0

        for entity in entities:
            entity_tokens = entity.split()
            if len(entity_tokens) <= len(tokens) - i:
                # Check if tokens match this entity
                if tokens[i : i + len(entity_tokens)] == entity_tokens:
                    if len(entity_tokens) > max_length:
                        matched_entity = entity
                        max_length = len(entity_tokens)

        if matched_entity:
            # Replace the matched tokens with the atomic entity
            preserved_tokens.append(matched_entity)
            i += max_length
        else:
            preserved_tokens.append(tokens[i])
            i += 1

    return preserved_tokens
