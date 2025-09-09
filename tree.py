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
from typing import Dict, Iterable, List, Tuple
import math

import branches
import roots
import treesoning


# simple in-memory cache for retrieved snippets keyed by language
_FETCH_CACHE: Dict[Tuple[str, str], str] = {}

STOPWORDS = {
    "the", "and", "to", "a", "in", "it", "of", "for", "on", "with", "as", "is", "at", "by", "from",
    "or", "an", "be", "this", "that", "are", "was", "but", "not", "had", "have", "has", "were", "been",
    "their", "said", "each", "which", "she", "do", "how", "if", "will", "up", "other", "about", "out",
    "many", "then", "them", "these", "so", "some", "her", "would", "make", "like", "into", "him", "time",
    "two", "more", "very", "when", "come", "may", "its", "only", "think", "now", "people", "my", "made",
    "over", "did", "down", "way", "find", "use", "get", "give", "work", "life", "day", "part", "year",
    "back", "see", "know", "just", "first", "could", "any", "all",
}

USER_AGENT = "Mozilla/5.0 (compatible; TreeEngine/1.0; +https://github.com/ariannamethod/tree)"


def _detect_language(text: str) -> str:
    """Return language code based on characters in *text*."""
    if re.search(r"[А-Яа-я]", text):
        return "ru"
    return "en"


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
    
    # Return empty string if no results found
    _FETCH_CACHE[cache_key] = ""
    return ""


def _build_reddit_query(message: str, word: str) -> str:
    """Build Reddit query for conversational context."""
    msg_lower = message.lower()
    
    if "how are you" in msg_lower:
        return "how are you responses conversation reddit"
    elif any(qw in msg_lower for qw in ["how", "what", "why", "when", "where"]):
        return f"how to respond to {message.strip()} reddit"
    elif word:
        return f"talking about {word} conversation reddit"
    else:
        return f"response to {message.strip()} reddit"


def _build_google_query(message: str, word: str) -> str:
    """Build Google query for conversational context."""
    msg_lower = message.lower()
    
    if "how are you" in msg_lower:
        return "how to respond how are you conversation"
    elif any(qw in msg_lower for qw in ["how", "what", "why"]):
        return f"conversation about {message.strip()}"
    else:
        return f"talking about {word} smalltalk"


def _fetch_reddit(query: str) -> str:
    """Fetch from Reddit JSON API."""
    try:
        url = f"https://www.reddit.com/search.json?q={urllib.parse.quote(query)}&limit=3&sort=relevance"
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
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        req = urllib.request.Request(url, headers=headers)
        
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", "ignore")
            
        # Simple parsing for titles and descriptions
        titles = re.findall(r'<h3[^>]*>([^<]+)</h3>', html)
        descriptions = re.findall(r'<span[^>]*>([^<]{30,150})</span>', html)
        
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
    return {hash(text[i : i + n]) for i in range(len(text) - n + 1)}


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
                t for t in tokens
                if len(t) > 2 and not t.isdigit() and t not in STOPWORDS
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
        # No fallback template - just use empty context
        raw = ""
        lower = ""
        tokens = []
        filtered_tokens = []
        unique = []
        quality_score = 0.0
    else:
        raw = " \n".join(snippets)[:2500]
        lower = raw.lower()
        tokens = re.findall(r"\w+", lower)

        filtered_tokens = [
            t for t in tokens
            if len(t) > 2 and not t.isdigit() and t not in STOPWORDS
        ]

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
        return ""
    
    # Calculate available words
    available_words = len(candidates) + len(source)
    if available_words == 0:
        return ""
    
    # Ensure we don't try to select more words than available
    max_possible = min(max_words, available_words)
    min_possible = min(min_words, max_possible)
    
    if min_possible > max_possible:
        return ""

    n = random.randint(min_possible, max_possible)
    source_count = (
        min(len(source), max(1, int(n * mix_ratio))) if source else 0
    )
    cand_count = max(0, n - source_count)
    
    # Ensure we don't exceed available words
    source_count = min(source_count, len(source))
    cand_count = min(cand_count, len(candidates))

    chosen: List[str] = []
    if source_count > 0:
        chosen.extend(random.sample(source, source_count))
    if cand_count > 0:
        chosen.extend(candidates[:cand_count])

    if not chosen:
        return ""

    random.shuffle(chosen)

    sentence = " ".join(chosen)
    sentence = sentence.capitalize()
    if not sentence.endswith("."):
        sentence += "."

    return sentence


def respond(message: str) -> str:
    """Generate a resonant response to *message*."""
    if not message.strip():
        return ""

    _update_ngrams_from_roots()

    lang = _detect_language(message)
    keys = _keywords(message)
    ctx = _context(message, keys, lang)  # Pass full message for context

    # Fallback if context is poor quality
    if ctx.quality_score < 0.1 or len(ctx.unique) < 3:
        # Try with a broader search
        fallback_keys = re.findall(r"\b\w{4,}\b", message.lower())[:3]
        if fallback_keys:
            ctx = _context(message, fallback_keys, lang)

    source_words = re.findall(r"\b\w+\b", message.lower())

    mix_ratio = 0.5 - 0.2 * min(ctx.quality_score, 1.0)

    # Generate responses with different semantic distances
    first_candidates = _select(ctx.unique, source_words, 0.4)
    second_candidates = _select(ctx.unique, source_words, 0.7)

    # Only use source words if we have meaningful context
    use_source_words = source_words if ctx.quality_score > 0.1 and ctx.unique else []

    first = _compose(first_candidates, use_source_words, mix_ratio)
    second = _compose(second_candidates, use_source_words, mix_ratio)

    # Ensure responses are different enough
    if SequenceMatcher(None, first, second).ratio() > 0.6:
        # Regenerate second response with different distance
        second_candidates = _select(ctx.unique, use_source_words, 0.9)
        second = _compose(second_candidates, use_source_words, mix_ratio)

    # Learn asynchronously - store the conversational context
    branches.learn(keys, ctx.raw)
    _update_ngram_with_text(ctx.raw)
    branches.wait()

    # Combine responses, filtering out empty ones
    results = [resp for resp in [first, second] if resp.strip()]
    return " ".join(results)


if __name__ == "__main__":  # pragma: no cover - manual exercise
    import sys

    user_input = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("> ")
    print(respond(user_input))
