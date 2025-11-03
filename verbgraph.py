"""VerbGraph utility for tracking verb-punctuation associations.

This module provides a minimal scaffold inspired by Middle English conventions
for tracking which verbs tend to end with which punctuation marks.
This will later be integrated to steer terminal punctuation in respond().
"""

import re
from collections import defaultdict, Counter
from typing import Dict

# Minimum token length to consider for verb tracking
MIN_TOKEN_LENGTH = 2


class VerbGraph:
    """Tracks verb → punctuation mark associations.

    This class observes sentences and learns which verbs tend to appear
    with which terminal punctuation marks (. ! ?).
    """

    def __init__(self):
        """Initialize an empty VerbGraph."""
        # Maps verb (normalized) to Counter of punctuation marks
        self._verb_punct_counts: Dict[str, Counter] = defaultdict(Counter)

    def _tokenize(self, text: str) -> list[str]:
        """Unicode-safe tokenizer that extracts words from text.

        Args:
            text: Input text to tokenize.

        Returns:
            List of word tokens (alphabetic sequences).
        """
        # Extract sequences of word characters (unicode-aware)
        tokens = re.findall(r"\w+", text)
        return tokens

    def _extract_terminal_punct(self, text: str) -> str:
        """Extract the terminal punctuation from text.

        Args:
            text: Input text.

        Returns:
            The terminal punctuation mark (. ! ?) or '.' as default.
        """
        text = text.strip()
        if not text:
            return "."

        # Check last character
        if text[-1] in ".!?":
            return text[-1]

        return "."

    def add_sentence(self, text: str) -> None:
        """Add a sentence to the graph, learning verb-punctuation associations.

        This tokenizes the text, identifies potential verbs (simplified as
        all non-trivial tokens), and associates them with the terminal
        punctuation of the sentence.

        Args:
            text: A sentence to learn from.
        """
        tokens = self._tokenize(text)
        punct = self._extract_terminal_punct(text)

        # Consider all tokens as potential verbs (simplified heuristic)
        # In a real implementation, we'd use POS tagging
        for token in tokens:
            normalized_verb = token.lower()
            if len(normalized_verb) >= MIN_TOKEN_LENGTH:
                self._verb_punct_counts[normalized_verb][punct] += 1

    def preferred_punct(self, verb: str) -> str:
        """Get the preferred punctuation mark for a verb.

        Args:
            verb: The verb to query (case-insensitive).

        Returns:
            The most common punctuation mark for this verb, or '.' as fallback.
        """
        normalized_verb = verb.lower()

        if normalized_verb not in self._verb_punct_counts:
            return "."

        counts = self._verb_punct_counts[normalized_verb]
        if not counts:
            return "."

        # Return the most common punctuation mark
        return counts.most_common(1)[0][0]
