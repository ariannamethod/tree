from __future__ import annotations

"""true recursive echo engine

The :mod:`tree` module is a minimalist transformer inspired construct.
It selects charged words from a user prompt, harvests tiny webs of
context from conversational sources, and composes two resonant sentences that drift
semantically away from the original message.
"""

import asyncio
import html
import random
import re
import unicodedata
import urllib.parse
import urllib.request
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from html.parser import HTMLParser
from typing import Dict, Iterable, List, Tuple
import math
import time

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


def _sanitize_visible(text: str) -> str:
    """Strip invisible/control characters and HTML artifacts from text.
    
    - HTML-unescapes text
    - Applies Unicode NFKC normalization
    - Removes invisible/control/format/surrogate/private-use code points
      (categories: Cc/Cf/Cs/Co/Cn)
    - Converts Unicode space separators (Zs) to standard space (U+0020)
    - Collapses runs of whitespace to a single space
    - Preserves all punctuation (P*), letters/marks/numbers across all languages
    """
    if not text:
        return ""

    # HTML-unescape the text
    text = html.unescape(text)

    # Apply Unicode NFKC normalization
    text = unicodedata.normalize('NFKC', text)

    # Filter out invisible/control/format/surrogate/private-use code points
    # Keep only visible characters: letters (L*), marks (M*), numbers (N*),
    # punctuation (P*), symbols (S*), and spaces (Zs->U+0020)
    filtered_chars = []
    for char in text:
        category = unicodedata.category(char)
        if category.startswith('C'):  # Control/Other categories - remove these
            continue
        elif category == 'Zs':  # Space separator - convert to standard space
            filtered_chars.append(' ')
        else:  # Keep all other visible characters (L*, M*, N*, P*, S*, Zl, Zp)
            filtered_chars.append(char)

    text = ''.join(filtered_chars)

    # Collapse runs of whitespace to a single space
    text = re.sub(r'\s+', ' ', text)

    # Strip leading/trailing whitespace
    return text.strip()


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
                # Sanitize each snippet before appending
                sanitized_content = _sanitize_visible(content[:400])
                if sanitized_content:
                    snippets.append(sanitized_content)
        
        # Sanitize the final joined string before return
        result = " ".join(snippets) if snippets else ""
        return _sanitize_visible(result)
        
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
                # Sanitize title before composing
                sanitized_title = _sanitize_visible(title)
                if sanitized_title:
                    content = sanitized_title
                    if i < len(descriptions) and descriptions[i]:
                        # Sanitize description before composing
                        sanitized_desc = _sanitize_visible(descriptions[i][:200])
                        if sanitized_desc:
                            content += f" - {sanitized_desc}"
                    snippets.append(content)
        
        # Sanitize the final joined string before return
        result = " ".join(snippets) if snippets else ""
        return _sanitize_visible(result)
        
    except Exception:
        return ""


def _get_fallback_pattern(message: str, word: str) -> str:
    """Fallback conversational patterns like Nicole."""
    msg_lower = message.lower()
    
    if "how are you" in msg_lower:
        patterns = [
            "I'm doing great, thanks for asking! How about you?",
            "Pretty good today, yourself?", 
            "Not bad at all, how are things with you?",
            "I'm fine, thank you for asking"
        ]
        return random.choice(patterns)
    
    if "hello" in msg_lower or "hi" in msg_lower:
        return "Hello there! Nice to meet you. What brings you here today?"
    
    if any(tech in msg_lower for tech in ["python", "code", "programming"]):
        return f"{word} resonates through code, building digital bridges and solving complex puzzles"
    
    if any(city in msg_lower for city in ["berlin", "paris", "tokyo", "london", "moscow"]):
        return f"{word} echoes through urban streets, where culture and history interweave"
    
    # Generic conversational fallback
    return f"{word} resonates through conversation, connecting thoughts and sparking dialogue"


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
            # Use treesoning result - sanitize context_text before tokenization and truncation
            raw = _sanitize_visible(context_text)[:2500]
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
        snippets = [f"conversation flows naturally, connecting {' and '.join(words)} through dialogue"]

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
        return "Silence echoes."

    n = random.randint(
        min_words, min(max_words, max(len(candidates), len(source)))
    )
    source_count = (
        min(len(source), max(1, int(n * mix_ratio))) if source else 0
    )
    cand_count = max(1, n - source_count)

    chosen: List[str] = []
    if source_count:
        chosen.extend(random.sample(source, source_count))
    chosen.extend(candidates[:cand_count])

    random.shuffle(chosen)

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

    first = _compose(first_candidates, source_words, mix_ratio)
    second = _compose(second_candidates, source_words, mix_ratio)

    # Ensure responses are different enough
    if SequenceMatcher(None, first, second).ratio() > 0.6:
        # Regenerate second response with different distance
        second_candidates = _select(ctx.unique, source_words, 0.9)
        second = _compose(second_candidates, source_words, mix_ratio)

    # Learn asynchronously - store the conversational context
    branches.learn(keys, ctx.raw)
    _update_ngram_with_text(ctx.raw)
    branches.wait()

    # Sanitize the final response before returning
    return _sanitize_visible(f"{first} {second}")


if __name__ == "__main__":  # pragma: no cover - manual exercise
    import sys

    user_input = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("> ")
    print(respond(user_input))
