from __future__ import annotations

"""true recursive echo engine

The :mod:`tree` module is a minimalist transformer inspired construct.
It selects charged words from a user prompt, harvests tiny webs of
context from conversational sources, and composes two resonant sentences that drift
semantically away from the original message.
"""

import asyncio
import random
import re
import urllib.parse
import urllib.request
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from html.parser import HTMLParser
from typing import Dict, Iterable, List, Tuple, Optional
import math
import hashlib

import branches
import roots
import treesoning


# Micro-interjections for true empties and cold start
MICRO_INTERJECTIONS = ["Huh?", "Hmmm…", "…"]

# simple in-memory cache for retrieved snippets keyed by language
_FETCH_CACHE: Dict[Tuple[str, str], str] = {}

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
    "all",
}

USER_AGENT = "Mozilla/5.0 (compatible; TreeEngine/1.0; +https://github.com/ariannamethod/tree)"


def _detect_language(text: str) -> str:
    """Return language code based on characters in *text*."""
    if re.search(r"[А-Яа-я]", text):
        return "ru"
    return "en"


def _get_micro_interjection(message: str) -> str:
    """Get deterministic micro-interjection for message."""
    normalized = re.sub(r"\s+", " ", message.lower().strip())
    if not normalized:
        normalized = "empty"

    # Deterministic selection based on message hash
    hash_digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
    idx = int(hash_digest[:8], 16) % len(MICRO_INTERJECTIONS)
    return MICRO_INTERJECTIONS[idx]


def _is_context_usable_for_output(context: "Context") -> bool:
    """Check if context meets minimum threshold for generating output."""
    if not context or not context.raw:
        return False

    # Must have at least 24 chars after sanitization
    clean_text = context.raw.strip()
    if len(clean_text) < 24:
        return False

    # Must have at least 3 non-stopword tokens with len >= 3
    tokens = [t for t in context.unique if len(t) >= 3 and t not in STOPWORDS]
    return len(tokens) >= 3


def _is_context_usable_for_training(context: "Context") -> bool:
    """Check if context meets minimum threshold for training/learning."""
    if not context or not context.raw:
        return False

    # Must have at least 60 chars
    clean_text = context.raw.strip()
    if len(clean_text) < 60:
        return False

    # Must have at least 5 non-stopword tokens with len >= 3
    tokens = [t for t in context.unique if len(t) >= 3 and t not in STOPWORDS]
    return len(tokens) >= 5


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
            if data and len(data) > 2:
                self._chunks.append(data)

    @property
    def text(self) -> str:
        return " ".join(self._chunks)


def _fetch_conversational(message: str, word: str, lang: str = "en", retries: int = 2) -> str:
    """Retrieve conversational context using Nicole's approach - no DuckDuckGo!"""
    cache_key = (lang, f"{message}:{word}")
    if cache_key in _FETCH_CACHE:
        return _FETCH_CACHE[cache_key]

    # First try memory
    recalled = _recall_fragment(word)
    if recalled:
        _FETCH_CACHE[cache_key] = recalled
        return recalled

    # Build conversational queries like Nicole
    results = []

    # Strategy 1: Reddit for natural responses
    reddit_query = _build_reddit_query(message, word)
    reddit_result = _fetch_reddit(reddit_query)
    if reddit_result:
        results.append(reddit_result)

    # Strategy 2: Google for conversational context (not definitions!)
    google_query = _build_google_query(message, word)
    google_result = _fetch_google(google_query)
    if google_result:
        results.append(google_result)

    # Combine results
    if results:
        combined = "\n\n".join(results)[:800]
        _FETCH_CACHE[cache_key] = combined
        return combined

    # Fallback patterns for common phrases
    fallback = _get_fallback_pattern(message, word)
    _FETCH_CACHE[cache_key] = fallback
    return fallback


def _build_reddit_query(message: str, word: str) -> str:
    """Build language-agnostic Reddit query."""
    # Use structural search with quotes and key terms
    if word:
        return f'"{word}" AND "{message.strip()}"'
    else:
        return f'"{message.strip()}"'


def _build_google_query(message: str, word: str) -> str:
    """Build language-agnostic Google query."""
    # Use structural search approach
    if word:
        return f'"{word}" AND "{message.strip()}"'
    else:
        return f'"{message.strip()}"'


def _fetch_reddit(query: str) -> str:
    """Fetch from Reddit JSON API."""
    try:
        url = (
            f"https://www.reddit.com/search.json?q={urllib.parse.quote(query)}"
            "&limit=3&sort=relevance"
        )
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        posts = data.get("data", {}).get("children", [])
        snippets = []

        for post in posts[:2]:
            post_data = post.get("data", {})
            title = post_data.get("title", "").strip()
            selftext = post_data.get("selftext", "").strip()

            if title and len(title) > 10:
                content = title
                if selftext and len(selftext) > 20:
                    content += f" - {selftext[:300]}"
                snippets.append(content[:400])

        return " ".join(snippets) if snippets else ""

    except Exception:
        return ""


def _fetch_google(query: str) -> str:
    """Simple Google search for conversational context."""
    try:
        url = f"https://www.google.com/search?q={urllib.parse.quote(query)}&num=2"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        req = urllib.request.Request(url, headers=headers)

        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", "ignore")

        # Simple parsing for titles and descriptions
        titles = re.findall(r"<h3[^>]*>([^<]+)</h3>", html)
        descriptions = re.findall(r"<span[^>]*>([^<]{30,150})</span>", html)

        snippets = []
        for i, title in enumerate(titles[:2]):
            if title and len(title) > 5:
                content = title
                if i < len(descriptions) and descriptions[i]:
                    content += f" - {descriptions[i][:200]}"
                snippets.append(content)

        return " ".join(snippets) if snippets else ""

    except Exception:
        return ""


def _get_fallback_pattern(message: str, word: str) -> str:
    """Return empty string - fallbacks now handled by micro-interjections."""
    return ""


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
    return {hash(text[i:i + n]) for i in range(len(text) - n + 1)}


def _context_from_memory(message: str) -> Optional[Context]:
    """Try to build context from roots memory."""
    # Extract keys: tokens with len>=3 not in STOPWORDS
    tokens = re.findall(r"\b\w{3,}\b", message.lower())
    keys = [t for t in tokens if t not in STOPWORDS]

    if not keys:
        return None

    # Try to recall fragments for each key
    fragments = []
    for key in keys[:3]:  # Limit to first 3 keys
        fragment = _recall_fragment(key)
        if fragment:
            fragments.append(fragment)

    if not fragments:
        # Expand keys via local NGRAMS proximity and try again
        expanded_keys = []
        for key in keys[:2]:  # Try fewer keys for expansion
            if key in NGRAMS:
                # Get top related words
                related = NGRAMS[key].most_common(3)
                expanded_keys.extend([word for word, _ in related])

        for key in expanded_keys[:3]:
            fragment = _recall_fragment(key)
            if fragment:
                fragments.append(fragment)

    if not fragments:
        return None

    # Aggregate and deduplicate fragments
    raw = " ".join(set(fragments))[:800]  # Deduplicate and limit
    lower = raw.lower()
    tokens = re.findall(r"\w+", lower)

    filtered_tokens = [t for t in tokens if len(t) > 2 and not t.isdigit() and t not in STOPWORDS]

    unique = list(dict.fromkeys(filtered_tokens))
    quality_score = len(unique) / max(len(tokens), 1) if tokens else 0

    return Context(
        raw=raw,
        lower=lower,
        words=filtered_tokens,
        unique=unique,
        quality_score=quality_score,
    )


def _recall_fragment(word: str, limit: int = 50) -> str | None:
    """Find a relevant context for *word* from stored roots."""
    word_l = word.lower()
    if len(word_l) < 4 or word_l in STOPWORDS:
        return None

    target = _hash_ngrams(word_l)
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

    threshold = 0.5 if len(word_l) <= 5 else 0.1
    if best_ctx and best_score > threshold:
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


def _context(message: str, words: Iterable[str], lang: str = "en") -> Context:
    """Build conversational context window using treesoning intelligence."""
    # First try treesoning for smart analysis
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        context_text, confidence = loop.run_until_complete(
            treesoning.get_treesoning_context(message)
        )
        loop.close()

        if confidence > 0.3 and context_text:
            # Use treesoning result
            raw = context_text[:2500]
            lower = raw.lower()
            tokens = re.findall(r"\w+", lower)

            filtered_tokens = [
                t for t in tokens if len(t) > 2 and not t.isdigit() and t not in STOPWORDS
            ]

            unique = list(dict.fromkeys(filtered_tokens))
            quality_score = confidence

            return Context(
                raw=raw,
                lower=lower,
                words=filtered_tokens,
                unique=unique,
                quality_score=quality_score,
            )
    except Exception as e:
        print(f"Treesoning failed: {e}, falling back to simple context")

    # Fallback to original method if treesoning fails
    snippets = []
    for w in words:
        snippet = _fetch_conversational(message, w, lang)
        if snippet:
            snippets.append(snippet)

    if not snippets:
        # Return empty context instead of language-bound template
        return Context(raw="", lower="", words=[], unique=[], quality_score=0.0)

    raw = " \n".join(snippets)[:2500]
    lower = raw.lower()
    tokens = re.findall(r"\w+", lower)

    filtered_tokens = [t for t in tokens if len(t) > 2 and not t.isdigit() and t not in STOPWORDS]

    unique = list(dict.fromkeys(filtered_tokens))
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
    words = re.findall(r"\b\w{2,}\b", cleaned)

    candidates = [w for w in words if w not in STOPWORDS and not w.isdigit()]

    if not candidates:
        candidates = words[:maximum]

    scores: Dict[str, float] = {}
    for w in candidates:
        freq = candidates.count(w)
        length_bonus = min(len(w) / 12, 1.0)
        rarity_bonus = 1.0 / freq if freq > 0 else 1.0
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
        random.shuffle(tokens)
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
    source: List[str],
    mix_ratio: float,
    min_words: int = 3,
    max_words: int = 9,
) -> str:
    """Compose a more natural sentence blending source words."""
    if not candidates and not source:
        return ""  # Return empty for fallback chain handling

    n = random.randint(min_words, min(max_words, max(len(candidates), len(source))))
    source_count = min(len(source), max(1, int(n * mix_ratio))) if source else 0
    cand_count = max(1, n - source_count)

    chosen: List[str] = []
    if source_count:
        chosen.extend(random.sample(source, source_count))
    chosen.extend(candidates[:cand_count])

    random.shuffle(chosen)

    # Remove the English function words insertion
    # if len(chosen) > 4:
    #     mid_point = len(chosen) // 2
    #     chosen.insert(mid_point, random.choice(["and", "through", "within"]))

    sentence = " ".join(chosen)
    sentence = sentence.capitalize()
    if not sentence.endswith("."):
        sentence += "."

    return sentence


def respond(message: str) -> str:
    """Generate a resonant response to *message*."""
    if not message.strip():
        return _get_micro_interjection(message)

    _update_ngrams_from_roots()

    lang = _detect_language(message)
    keys = _keywords(message)

    # Step 1: Try normal provider/context building
    ctx = _context(message, keys, lang)

    # Step 2: If context is empty/low-quality, attempt memory recall from roots
    if not _is_context_usable_for_output(ctx):
        memory_ctx = _context_from_memory(message)
        if memory_ctx and _is_context_usable_for_output(memory_ctx):
            ctx = memory_ctx

    # Step 3: If still empty or too short, return micro-interjection
    if not _is_context_usable_for_output(ctx):
        return _get_micro_interjection(message)

    source_words = re.findall(r"\b\w+\b", message.lower())
    mix_ratio = 0.5 - 0.2 * min(ctx.quality_score, 1.0)

    # Generate responses with different semantic distances
    first_candidates = _select(ctx.unique, source_words, 0.4)
    second_candidates = _select(ctx.unique, source_words, 0.7)

    first = _compose(first_candidates, source_words, mix_ratio)
    second = _compose(second_candidates, source_words, mix_ratio)

    # Handle empty compositions
    if not first and not second:
        return _get_micro_interjection(message)
    elif not first:
        first = second
        second = ""
    elif not second:
        second = ""

    # Ensure responses are different enough if both exist
    if first and second and SequenceMatcher(None, first, second).ratio() > 0.6:
        # Regenerate second response with different distance
        second_candidates = _select(ctx.unique, source_words, 0.9)
        second = _compose(second_candidates, source_words, mix_ratio)

    # Learning guard: only learn when context passes training threshold
    if _is_context_usable_for_training(ctx):
        branches.learn(keys, ctx.raw)
        _update_ngram_with_text(ctx.raw)
        branches.wait()

    # Return composed response
    if first and second:
        return f"{first} {second}"
    elif first:
        return first
    else:
        return _get_micro_interjection(message)


if __name__ == "__main__":  # pragma: no cover - manual exercise
    import sys

    user_input = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("> ")
    print(respond(user_input))
