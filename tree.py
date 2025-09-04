from __future__ import annotations

"""true recursive echo engine

The :mod:`tree` module is a minimalist transformer inspired construct.
It selects charged words from a user prompt, harvests tiny webs of
context from the internet, and composes two resonant sentences that drift
semantically away from the original message.
"""

import random
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from difflib import SequenceMatcher
from html.parser import HTMLParser
from typing import Dict, Iterable, List

import branches

STOPWORDS = {
    "the",
    "and",
    "to",
    "a",
    "in",
    "it",
    "of",
    "for",
    "on",
    "with",
    "as",
    "is",
    "at",
    "by",
    "from",
    "or",
    "an",
    "be",
    "this",
    "that",
}


class _Extractor(HTMLParser):
    """Pull text from HTML using only the standard library."""

    def __init__(self) -> None:
        super().__init__()
        self._chunks: List[str] = []

    def handle_data(self, data: str) -> None:  # pragma: no cover - trivial
        data = data.strip()
        if data:
            self._chunks.append(data)

    @property
    def text(self) -> str:
        return " ".join(self._chunks)


def _fetch(word: str) -> str:
    """Retrieve a snippet from the web for *word*."""
    url = "https://duckduckgo.com/html/?q=" + urllib.parse.quote(word)
    with urllib.request.urlopen(url, timeout=10) as resp:
        html = resp.read().decode("utf-8", "ignore")
    parser = _Extractor()
    parser.feed(html)
    return parser.text[:500]


@dataclass
class Context:
    raw: str
    lower: str
    words: List[str]
    unique: List[str]


def _context(words: Iterable[str]) -> Context:
    """Build a layered context window from web snippets."""
    snippets = [_fetch(w) for w in words]
    raw = " \n".join(snippets)[:2000]
    lower = raw.lower()
    tokens = re.findall(r"\w+", lower)
    unique = sorted(set(tokens))
    return Context(raw=raw, lower=lower, words=tokens, unique=unique)


def _keywords(message: str, minimum: int = 4, maximum: int = 6) -> List[str]:
    """Extract charged keywords from *message* using simple heuristics."""
    words = re.findall(r"\w+", message.lower())
    candidates = [w for w in words if w not in STOPWORDS]
    if not candidates:
        candidates = words
    scores: Dict[str, float] = {}
    for w in candidates:
        freq = candidates.count(w)
        scores[w] = len(w) / freq
    ranked = sorted(scores, key=scores.get, reverse=True)
    span = max(minimum, min(maximum, len(ranked)))
    return ranked[:span]


def _select(
    tokens: Iterable[str], source: List[str], target: float
) -> List[str]:
    """Pick tokens semantically *target* distance from source words."""
    joined = " ".join(source)
    graded = []
    for t in tokens:
        if t in source:
            continue
        dist = 1 - SequenceMatcher(None, t, joined).ratio()
        graded.append((abs(dist - target), t))
    graded.sort()
    return [g[1] for g in graded]


def _compose(candidates: List[str]) -> str:
    n = random.randint(4, 8)
    chosen = candidates[:n]
    return " ".join(chosen).capitalize() + "."


def respond(message: str) -> str:
    """Generate a resonant response to *message*."""
    keys = _keywords(message)
    ctx = _context(keys)
    source_words = re.findall(r"\w+", message.lower())
    first = _compose(_select(ctx.unique, source_words, 0.5))
    second = _compose(_select(ctx.unique, source_words, 0.8))
    branches.learn(keys, ctx.raw)
    return f"{first} {second}"


if __name__ == "__main__":  # pragma: no cover - manual exercise
    import sys

    user = " ".join(sys.argv[1:]) or input("> ")
    print(respond(user))
