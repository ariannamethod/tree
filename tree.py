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
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from html.parser import HTMLParser
from typing import Dict, Iterable, List
import math
import time

import branches
import roots


# simple in-memory cache for retrieved snippets
_FETCH_CACHE: Dict[str, str] = {}


def _detect_language(text: str) -> str:
    """Very small heuristic language detection."""
    if re.search(r"[А-Яа-я]", text):
        return "ru-ru"
    return "en-us"


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
    "are",
    "was",
    "but",
    "not",
    "had",
    "have",
    "has",
    "were",
    "been",
    "their",
    "said",
    "each",
    "which",
    "she",
    "do",
    "how",
    "if",
    "will",
    "up",
    "other",
    "about",
    "out",
    "many",
    "then",
    "them",
    "these",
    "so",
    "some",
    "her",
    "would",
    "make",
    "like",
    "into",
    "him",
    "time",
    "two",
    "more",
    "very",
    "when",
    "come",
    "may",
    "its",
    "only",
    "think",
    "now",
    "people",
    "my",
    "made",
    "over",
    "did",
    "down",
    "way",
    "find",
    "use",
    "get",
    "give",
    "work",
    "life",
    "day",
    "part",
    "year",
    "back",
    "see",
    "know",
    "just",
    "first",
    "could",
    "any",
}


class _Extractor(HTMLParser):
    """Pull text from HTML using only the standard library."""

    def __init__(self) -> None:
        super().__init__()
        self._chunks: List[str] = []
        self._in_script = False
        self._in_style = False

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.lower() in ("script", "style"):
            self._in_script = True
            self._in_style = True

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in ("script", "style"):
            self._in_script = False
            self._in_style = False

    def handle_data(self, data: str) -> None:
        if not (self._in_script or self._in_style):
            data = data.strip()
            if data and len(data) > 2:  # Skip very short fragments
                self._chunks.append(data)

    @property
    def text(self) -> str:
        return " ".join(self._chunks)


def _fetch(word: str, retries: int = 2, lang: str = "en-us") -> str:
    """Retrieve a snippet for *word* using cache and local memory."""
    if word in _FETCH_CACHE:
        return _FETCH_CACHE[word]

    recalled = _recall_fragment(word)
    if recalled:
        _FETCH_CACHE[word] = recalled
        return recalled

    user_agent = "Mozilla/5.0 (compatible; TreeEngine/1.0)"

    for attempt in range(retries + 1):
        try:
            query = word if attempt == 0 else f"{word} meaning"
            url = (
                "https://duckduckgo.com/html/?kl="
                + urllib.parse.quote(lang)
                + "&q="
                + urllib.parse.quote(query)
            )

            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": user_agent,
                    "Accept-Language": lang,
                },
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                html = resp.read().decode("utf-8", "ignore")

            parser = _Extractor()
            parser.feed(html)
            text = parser.text[:600]

            if len(text) > 50:
                _FETCH_CACHE[word] = text
                return text

        except Exception:
            if attempt < retries:
                time.sleep(0.5)
                continue

    fallback = f"{word} resonates through digital space, echoing meaning"
    _FETCH_CACHE[word] = fallback
    return fallback


@dataclass
class Context:
    raw: str
    lower: str
    words: List[str]
    unique: List[str]
    quality_score: float = 0.0


NGRAMS: Dict[str, Counter[str]] = defaultdict(Counter)


def _update_ngram_with_text(text: str) -> None:
    tokens = [
        t
        for t in re.findall(r"\w+", text.lower())
        if len(t) > 2 and not t.isdigit() and t not in STOPWORDS
    ]
    for a, b in zip(tokens, tokens[1:]):
        NGRAMS[a][b] += 1


def _update_ngrams_from_roots(limit: int = 100) -> None:
    for _, ctx in roots.recall(limit):
        _update_ngram_with_text(ctx)


def _hash_ngrams(text: str, n: int = 3) -> set[int]:
    """Return a set of hashed character n-grams for *text*."""
    text = re.sub(r"\s+", " ", text.lower())
    if len(text) < n:
        return set()
    return {hash(text[i : i + n]) for i in range(len(text) - n + 1)}  # noqa: E203,E501


def _recall_fragment(word: str, limit: int = 50) -> str | None:
    """Find a relevant context for *word* from stored roots."""
    target = _hash_ngrams(word)
    if not target:
        return None

    best_score = 0.0
    best_ctx: str | None = None
    for _, ctx in roots.recall(limit):
        grams = _hash_ngrams(ctx)
        if not grams:
            continue
        score = len(target & grams) / len(target)
        if score > best_score:
            best_score, best_ctx = score, ctx
    if best_ctx and best_score > 0.1:
        return best_ctx[:600]
    return None


def _normalize(counter: Counter[str]) -> Dict[str, float]:
    if not counter:
        return {}
    norm = math.sqrt(sum(v * v for v in counter.values()))
    return {k: v / norm for k, v in counter.items()}


def _source_vector(source: List[str]) -> Dict[str, float]:
    agg: Counter[str] = Counter()
    for s in source:
        agg.update(NGRAMS.get(s, {}))
    return _normalize(agg)


def _context(words: Iterable[str]) -> Context:
    """Build a layered context window from web snippets."""
    snippets = []
    lang = _detect_language(" ".join(words))

    for w in words:
        snippet = _fetch(w, lang=lang)
        if snippet:
            snippets.append(snippet)

    if not snippets:
        # Emergency fallback
        snippets = ["consciousness flows through digital streams"]

    raw = " \n".join(snippets)[:2500]  # Slightly larger context
    lower = raw.lower()
    tokens = re.findall(r"\w+", lower)

    # Filter tokens by length and remove numbers-only
    filtered_tokens = [
        t
        for t in tokens
        if len(t) > 2 and not t.isdigit() and t not in STOPWORDS  # noqa: E501
    ]

    unique = sorted(set(filtered_tokens))

    # Calculate quality score based on diversity and length
    quality_score = len(unique) / max(len(tokens), 1) if tokens else 0

    return Context(
        raw=raw,
        lower=lower,
        words=filtered_tokens,
        unique=unique,
        quality_score=quality_score,
    )


def _keywords(message: str, minimum: int = 3, maximum: int = 7) -> List[str]:
    """Extract charged keywords from *message* using improved heuristics."""
    # Clean and normalize the message
    cleaned = re.sub(r"[^\w\s]", " ", message.lower())
    words = re.findall(r"\b\w{2,}\b", cleaned)  # At least 2 characters

    candidates = [w for w in words if w not in STOPWORDS and not w.isdigit()]

    if not candidates:
        candidates = words[:maximum]  # Take first few if no good candidates

    scores: Dict[str, float] = {}
    for w in candidates:
        freq = candidates.count(w)
        length_bonus = min(len(w) / 12, 1.0)  # Favor longer words
        rarity_bonus = 1.0 / freq if freq > 0 else 1.0
        # Slight bonus for early words
        position_bonus = 1.0 if w in words[:3] else 0.8

        scores[w] = length_bonus * rarity_bonus * position_bonus

    if not scores:
        return words[:maximum]

    ranked = sorted(scores, key=scores.get, reverse=True)
    span = max(minimum, min(maximum, len(ranked)))
    return ranked[:span]


def _select(
    tokens: Iterable[str],
    source: List[str],
    target: float,
    limit: int = 12,
) -> List[str]:
    """Pick tokens semantically *target* distance from source words."""
    tokens = list(tokens)
    if not tokens or not source:
        return []

    src_vec = _source_vector(source)
    if not src_vec:
        return tokens[:limit]

    graded = []
    for t in tokens:
        if t in source or len(t) < 3:
            continue

        vec = _normalize(NGRAMS.get(t, Counter()))
        cos = sum(vec.get(k, 0.0) * src_vec.get(k, 0.0) for k in vec)
        distance = 1.0 - cos
        graded.append((abs(distance - target), t))

    graded.sort()
    return [g[1] for g in graded[:limit]]


def _compose(
    candidates: List[str],
    user_tokens: List[str] | None = None,
    mix_ratio: float = 0.0,
    min_words: int = 3,
    max_words: int = 9,
) -> str:
    """Compose a more natural sentence from candidates and user tokens."""
    user_tokens = user_tokens or []
    if not candidates and not user_tokens:
        return "Silence echoes."

    max_pool = len(candidates) + len(user_tokens)
    n = random.randint(min_words, min(max_words, max_pool))
    if mix_ratio > 0:
        mix_count = min(len(user_tokens), int(n * mix_ratio))
    else:
        mix_count = 0
    base_count = max(0, n - mix_count)

    chosen = candidates[:base_count]
    if mix_count:
        chosen.extend(random.sample(user_tokens, mix_count))
    random.shuffle(chosen)

    sentence = " ".join(chosen)

    if len(chosen) > 4:
        mid_point = len(chosen) // 2
        chosen.insert(mid_point, random.choice(["and", "through", "within"]))
        sentence = " ".join(chosen)

    sentence = sentence.capitalize()
    if not sentence.endswith("."):
        sentence += "."

    return sentence


def respond(message: str) -> str:
    """Generate a resonant response to *message*."""
    if not message.strip():
        return "Emptiness speaks. Silence responds."

    _update_ngrams_from_roots()

    keys = _keywords(message)
    ctx = _context(keys)

    if ctx.quality_score < 0.1 or len(ctx.unique) < 3:
        fallback_keys = re.findall(r"\b\w{4,}\b", message.lower())[:3]
        if fallback_keys:
            ctx = _context(fallback_keys)

    user_tokens = re.findall(r"\b\w+\b", message.lower())
    mix = 0.3 + 0.2 * min(ctx.quality_score, 1.0)

    first_candidates = _select(ctx.unique, user_tokens, 0.4)
    second_candidates = _select(ctx.unique, user_tokens, 0.7)

    first = _compose(first_candidates, user_tokens, mix)
    second = _compose(second_candidates, user_tokens, mix)

    if SequenceMatcher(None, first, second).ratio() > 0.6:
        second_candidates = _select(ctx.unique, user_tokens, 0.9)
        second = _compose(second_candidates, user_tokens, mix)

    convo_context = f"{message}\n{first}\n{second}"
    branches.learn(keys + user_tokens, convo_context)
    branches.wait()
    _update_ngram_with_text(convo_context)

    return f"{first}\n{second}"


if __name__ == "__main__":  # pragma: no cover - manual exercise
    import sys

    user_input = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("> ")
    print(respond(user_input))
