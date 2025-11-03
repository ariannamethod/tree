"""Text constraints and normalization utilities.

This module provides functions to normalize whitespace, capitalize text,
ensure single terminal punctuation, and limit commas in generated text.
All functions are pure stdlib and CPU-only.
"""

import re


def normalize_whitespace(text: str) -> str:
    """Collapse consecutive whitespace characters into single spaces.

    Args:
        text: Input text that may contain multiple spaces, tabs, newlines, etc.

    Returns:
        Text with all whitespace sequences collapsed to single spaces and trimmed.
    """
    return re.sub(r"\s+", " ", text).strip()


def capitalize_first_alpha(text: str) -> str:
    """Capitalize the first alphabetic character in the text.

    This is unicode-aware and will capitalize letters in any language.
    Non-alphabetic characters at the beginning are preserved.

    Args:
        text: Input text to capitalize.

    Returns:
        Text with the first alphabetic character capitalized.
    """
    if not text:
        return text

    result = list(text)
    for i, char in enumerate(result):
        if char.isalpha():
            result[i] = char.upper()
            break

    return "".join(result)


def ensure_single_terminal_punct(text: str) -> str:
    """Ensure text ends with at most one terminal punctuation mark.

    Terminal marks are: . ! ?
    If text has multiple terminal marks at the end, keep only one.
    If text has no terminal mark, append a period.

    Args:
        text: Input text to normalize.

    Returns:
        Text with exactly one terminal punctuation mark at the end.
    """
    if not text:
        return "."

    # Remove all trailing terminal punctuation
    cleaned = re.sub(r"[.!?]+$", "", text)

    # Find what the last punctuation was (if any) to preserve intent
    match = re.search(r"([.!?])+$", text)
    if match:
        # Use the first punctuation character from the sequence
        punct = match.group(0)[0]
    else:
        # Default to period if no terminal punctuation found
        punct = "."

    return cleaned + punct


def limit_commas(text: str, max_commas: int = 2) -> str:
    """Limit the number of commas in text.

    If text contains more than max_commas commas, drop the excess ones.
    Commas are dropped from the end of the text first.

    Args:
        text: Input text that may contain commas.
        max_commas: Maximum number of commas to keep (default: 2).

    Returns:
        Text with at most max_commas commas.
    """
    comma_count = text.count(",")

    if comma_count <= max_commas:
        return text

    # Split by commas (this splits into all parts)
    parts = text.split(",")

    # Keep first max_commas parts with commas, join the rest without commas
    # +1 because max_commas commas means max_commas+1 parts
    result_parts = parts[: max_commas + 1]
    remaining_parts = parts[max_commas + 1 :]

    # Join the allowed parts with commas
    result = ",".join(result_parts)

    # Append remaining parts without commas (just add them with space)
    if remaining_parts:
        result += " " + " ".join(remaining_parts)

    return result


def finalize_text(text: str) -> str:
    """Apply all text constraints in order.

    This applies the full pipeline:
    1. Normalize whitespace
    2. Limit commas to at most 2
    3. Ensure single terminal punctuation
    4. Capitalize first alphabetic character

    Args:
        text: Input text to finalize.

    Returns:
        Cleaned and normalized text.
    """
    text = normalize_whitespace(text)
    text = limit_commas(text, max_commas=2)
    text = ensure_single_terminal_punct(text)
    text = capitalize_first_alpha(text)
    return text
