"""Guide module for Tree English playbook.

This module provides pattern matching and template selection for common
English interactions, allowing Tree to respond with curated templates
for specific patterns like greetings and identity questions.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class Match:
    """Represents a pattern match with templates and query hints."""

    pattern: str
    templates: List[str]
    query_hints: List[str]


@dataclass
class Guide:
    """Container for parsed guide patterns."""

    patterns: List[
        tuple[re.Pattern, List[str], List[str]]
    ]  # (compiled_pattern, templates, hints)


# Global cache for the loaded guide
_guide_cache: Optional[Guide] = None

# Module-level random instance for template selection
_rng = random.Random()


def load_guide(path: Optional[str] = None) -> Guide:
    """Load and parse the English guide from markdown.

    Args:
        path: Optional path to guide file. Defaults to docs/TREE_EN_GUIDE.md

    Returns:
        Parsed Guide object with compiled patterns
    """
    global _guide_cache

    if _guide_cache is not None:
        return _guide_cache

    if path is None:
        # Default to docs/TREE_EN_GUIDE.md relative to this file
        module_dir = Path(__file__).parent
        path = module_dir / "docs" / "TREE_EN_GUIDE.md"

    path = Path(path)
    if not path.exists():
        # Return empty guide if file doesn't exist
        _guide_cache = Guide(patterns=[])
        return _guide_cache

    patterns = []
    current_pattern = None
    current_templates = []
    current_hints = []
    in_templates = False
    in_hints = False

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()

            # Check for pattern header
            if line.startswith("## Pattern:"):
                # Save previous pattern if exists
                if current_pattern is not None:
                    patterns.append((current_pattern, current_templates, current_hints))

                # Extract pattern and flags
                pattern_text = line[len("## Pattern:"):].strip()

                # Check for (i) case-insensitive flag
                flags = 0
                if pattern_text.endswith(" (i)"):
                    flags = re.IGNORECASE
                    pattern_text = pattern_text[:-4].strip()

                # Compile pattern
                try:
                    current_pattern = re.compile(pattern_text, flags)
                except re.error:
                    current_pattern = None

                current_templates = []
                current_hints = []
                in_templates = False
                in_hints = False

            # Check for Templates section
            elif line.startswith("### Templates"):
                in_templates = True
                in_hints = False

            # Check for QueryHints section
            elif line.startswith("### QueryHints"):
                in_templates = False
                in_hints = True

            # Check for new section that ends current sections
            elif line.startswith("#"):
                in_templates = False
                in_hints = False

            # Parse template or hint lines (start with -)
            elif line.startswith("- ") and (in_templates or in_hints):
                content = line[2:].strip()
                if in_templates:
                    current_templates.append(content)
                elif in_hints:
                    current_hints.append(content)

    # Save last pattern
    if current_pattern is not None:
        patterns.append((current_pattern, current_templates, current_hints))

    _guide_cache = Guide(patterns=patterns)
    return _guide_cache


def match(message: str) -> Optional[Match]:
    """Match message against loaded patterns.

    Args:
        message: User input message to match

    Returns:
        Match object if pattern matches, None otherwise
    """
    guide = load_guide()

    for pattern, templates, hints in guide.patterns:
        if pattern.match(message):
            return Match(
                pattern=pattern.pattern, templates=templates, query_hints=hints
            )

    return None


def choose_template(templates: List[str]) -> str:
    """Choose a template from the list.

    Uses a module-level random instance to avoid affecting global random state.

    Args:
        templates: List of template strings

    Returns:
        Selected template string, or empty string if list is empty
    """
    if not templates:
        return ""

    # Use module-level random instance to avoid affecting global state
    return _rng.choice(templates)


def get_query_hints(match_obj: Optional[Match]) -> List[str]:
    """Extract query hints from a match.

    Args:
        match_obj: Match object from match()

    Returns:
        List of query hint strings, empty if no match
    """
    if match_obj is None:
        return []

    return match_obj.query_hints


def clear_cache():
    """Clear the cached guide. Useful for testing."""
    global _guide_cache
    _guide_cache = None
