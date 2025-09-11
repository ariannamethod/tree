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

# Denylist for problematic filler tokens
DENYLIST_TOKENS = {"arrangement", "deal", "around", "base"}


def _is_question(message: str) -> bool:
    """Detect if message is a question based on markers."""
    # Check for explicit question mark
    if "?" in message:
        return True

    # Check for interrogative words (English and some Russian)
    message_lower = message.lower()
    interrogative_words = {
        # English
        "who",
        "what",
        "when",
        "where",
        "why",
        "how",
        "which",
        "whose",
        "will",
        "can",
        "could",
        "would",
        "should",
        "is",
        "are",
        "do",
        "does",
        "did",
        # Russian
        "кто",
        "что",
        "как",
        "почему",
        "зачем",
        "где",
        "когда",
        "какой",
        "чей",
    }

    words = re.findall(r"\b\w+\b", message_lower)
    if words and words[0] in interrogative_words:
        return True

    return False


def _normalize_token(token: str) -> str:
    """Normalize token for comparison (lowercase, basic cleanup)."""
    return token.lower().strip()


def _unicode_capitalize(text: str) -> str:
    """Unicode-aware capitalization of first letter."""
    if not text:
        return text

    # Find first alphabetic character and capitalize it
    result = list(text)
    for i, char in enumerate(result):
        if char.isalpha():
            result[i] = char.upper()
            break

    return "".join(result)


def _detect_language(text: str) -> str:
    """Return language code based on characters in *text*."""
    if re.search(r"[А-Яа-я]", text):
        return "ru"
    return "en"


def _detect_entity_phrases(message: str) -> List[str]:
    """Detect entity-like phrases using stdlib-only heuristics.

    Returns list of entity phrases like ["Donald Trump", "New York", "IBM"].
    Detects:
    - Proper-case multiword sequences: consecutive Capitalized tokens
    - ALL-CAPS short tokens (2-6 chars): acronyms like IBM, GPT-5
    - Unicode-aware for Latin/Cyrillic scripts
    """
    if not message.strip():
        return []

    entities = []

    # Split into tokens while preserving spaces for phrase reconstruction
    tokens = re.findall(r"\S+", message)

    # Pattern 1: ALL-CAPS short tokens (2-6 chars, may include numbers/hyphens)
    for token in tokens:
        # Remove punctuation from edges for checking
        clean_token = re.sub(r"^[^\w]+|[^\w]+$", "", token)
        if len(clean_token) >= 2 and len(clean_token) <= 6:
            # Must be mostly uppercase letters (allow numbers, hyphens)
            if re.match(r"^[A-Z]+(?:[0-9-][A-Z]*)*$", clean_token):
                entities.append(clean_token)

    # Pattern 2: Proper-case multiword sequences
    # Find sequences of consecutive capitalized words
    i = 0
    while i < len(tokens):
        # Look for capitalized word (unicode-aware)
        token = re.sub(r"^[^\w]+|[^\w]+$", "", tokens[i])  # Clean punctuation
        if len(token) >= 2 and token[0].isupper():
            # Check if it's a Latin or Cyrillic letter
            first_char = token[0]
            if (first_char >= "A" and first_char <= "Z") or (
                first_char >= "А" and first_char <= "Я"
            ):
                # Start of potential multi-word entity
                phrase_tokens = [token]
                j = i + 1

                # Look for consecutive capitalized tokens
                while j < len(tokens):
                    next_token = re.sub(r"^[^\w]+|[^\w]+$", "", tokens[j])
                    if (
                        len(next_token) >= 2
                        and next_token[0].isupper()
                        and (
                            (next_token[0] >= "A" and next_token[0] <= "Z")
                            or (next_token[0] >= "А" and next_token[0] <= "Я")
                        )
                    ):
                        phrase_tokens.append(next_token)
                        j += 1
                    else:
                        break

                # Only add multi-word phrases (2+ words)
                if len(phrase_tokens) >= 2:
                    entities.append(" ".join(phrase_tokens))

                i = j  # Continue from where we left off
            else:
                i += 1
        else:
            i += 1

    return entities


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


@dataclass
class TokenWithSource:
    """Token with its source information for diversification tracking."""

    token: str
    source: str  # "input", "memory_fragment_N", "snippet"


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
    return {hash(text[i: i + n]) for i in range(len(text) - n + 1)}


def _context_from_memory(message: str) -> Context | None:
    """Try to build context from roots memory, prioritizing entity-relevant fragments."""
    # Detect entity phrases for relevance filtering
    entity_phrases = _detect_entity_phrases(message)

    # Extract keys: tokens with len>=3 not in STOPWORDS
    tokens = re.findall(r"\b\w{3,}\b", message.lower())
    keys = [t for t in tokens if t not in STOPWORDS]

    if not keys:
        return None

    # Try to recall fragments for each key, with entity relevance scoring
    fragments = []
    entity_fragments = []  # Prioritized fragments containing entity phrases

    for key in keys[:3]:  # Limit to first 3 keys
        fragment = _recall_fragment(key)
        if fragment:
            # Check if fragment contains any entity phrase
            contains_entity = any(entity.lower() in fragment.lower() for entity in entity_phrases)
            if contains_entity:
                entity_fragments.append(fragment)
            else:
                fragments.append(fragment)

    # Prioritize entity-relevant fragments
    prioritized_fragments = entity_fragments + fragments

    if not prioritized_fragments:
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
                # Again check for entity relevance
                contains_entity = any(
                    entity.lower() in fragment.lower() for entity in entity_phrases
                )
                if contains_entity:
                    entity_fragments.append(fragment)
                else:
                    fragments.append(fragment)

        prioritized_fragments = entity_fragments + fragments

    if not prioritized_fragments:
        return None

    # Aggregate and deduplicate fragments, using prioritized list
    raw = " ".join(set(prioritized_fragments))[:800]  # Deduplicate and limit
    lower = raw.lower()
    tokens = re.findall(r"\w+", lower)

    filtered_tokens = [t for t in tokens if len(t) > 2 and not t.isdigit() and t not in STOPWORDS]

    unique = list(dict.fromkeys(filtered_tokens))

    # Boost quality score if entity phrases are present in memory
    base_quality = len(unique) / max(len(tokens), 1) if tokens else 0
    entity_boost = 0.2 if entity_fragments else 0  # Boost quality for entity-relevant memories
    quality_score = min(1.0, base_quality + entity_boost)

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
    """Extract charged keywords from *message* with entity phrase locking."""
    # First, detect and prioritize entity phrases
    entity_phrases = _detect_entity_phrases(message)

    # Clean and normalize the message for regular keyword extraction
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
        regular_keywords = words[:maximum]
    else:
        ranked = sorted(scores, key=scores.get, reverse=True)
        span = max(minimum, min(maximum, len(ranked)))
        regular_keywords = ranked[:span]

    # Combine entity phrases with regular keywords, prioritizing entities
    result = []

    # Add entity phrases first (these are locked and prioritized)
    for entity in entity_phrases:
        result.append(entity)

    # Add regular keywords up to the limit, avoiding duplicates
    for keyword in regular_keywords:
        if len(result) >= maximum:
            break
        # Avoid adding keywords that are already part of entity phrases
        if not any(keyword.lower() in entity.lower() for entity in entity_phrases):
            result.append(keyword)

    # Ensure we meet minimum requirements
    if len(result) < minimum:
        # Add more regular keywords if needed
        for word in words:
            if len(result) >= maximum:
                break
            if word not in result and word not in STOPWORDS:
                result.append(word)

    return result[:maximum]


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


def _filter_source_gated_tokens(
    candidates: List[str],
    user_tokens: List[str],
    context_tokens: List[str],
    memory_tokens: List[str],
) -> List[str]:
    """Filter candidates to only include tokens from valid sources, applying denylist."""
    # Collect all allowed tokens from sources
    allowed_tokens = set()

    # Add tokens from all valid sources (normalized for comparison)
    for token in user_tokens:
        allowed_tokens.add(_normalize_token(token))
    for token in context_tokens:
        allowed_tokens.add(_normalize_token(token))
    for token in memory_tokens:
        allowed_tokens.add(_normalize_token(token))

    # Filter candidates
    filtered = []
    for candidate in candidates:
        normalized = _normalize_token(candidate)

        # Check denylist - if token is denylisted, only allow if in sources
        if normalized in DENYLIST_TOKENS:
            if normalized in allowed_tokens:
                filtered.append(candidate)
            # else: skip denylisted token not in sources
        else:
            # Non-denylisted token - only allow if from valid sources
            if normalized in allowed_tokens:
                filtered.append(candidate)

    return filtered


def _check_repetition_rules(tokens: List[str], candidate: str) -> bool:
    """Check if adding candidate would violate repetition rules."""
    normalized_candidate = _normalize_token(candidate)

    # Count current occurrences
    count = sum(1 for token in tokens if _normalize_token(token) == normalized_candidate)

    # Rule 1: No more than 2 occurrences total
    if count >= 2:
        return False

    # Rule 2: No immediate duplicates
    if tokens and _normalize_token(tokens[-1]) == normalized_candidate:
        return False

    return True


def _check_bigram_rule(tokens: List[str], candidate: str) -> bool:
    """Check if adding candidate would create a repeated bigram."""
    if len(tokens) < 1:
        return True

    # Form the potential new bigram
    new_bigram = (_normalize_token(tokens[-1]), _normalize_token(candidate))

    # Check if this bigram already exists in the token sequence
    for i in range(len(tokens) - 1):
        existing_bigram = (_normalize_token(tokens[i]), _normalize_token(tokens[i + 1]))
        if existing_bigram == new_bigram:
            return False

    return True


def _compose(
    candidates: List[str],
    source: List[str],
    mix_ratio: float,
    min_words: int = 3,
    max_words: int = 9,
    user_message: str = "",
    context_tokens: List[str] = None,
    memory_tokens: List[str] = None,
) -> str:
    """Compose a more natural sentence blending source words with filtering and anti-repetition.

    Now includes entity phrase locking to preserve named entities intact.
    """
    if context_tokens is None:
        context_tokens = []
    if memory_tokens is None:
        memory_tokens = []

    if not candidates and not source:
        return ""  # Return empty for fallback chain handling

    # Extract user input tokens
    user_tokens = re.findall(r"\b\w+\b", user_message.lower()) if user_message else []

    # Detect entity phrases for preservation
    entity_phrases = _detect_entity_phrases(user_message) if user_message else []

    # Apply source-gated filtering
    filtered_candidates = _filter_source_gated_tokens(
        candidates, user_tokens, context_tokens, memory_tokens
    )

    # If all candidates were filtered out, return empty for fallback
    if not filtered_candidates and not source and not entity_phrases:
        return ""

    # Determine safe diversification - enable only with sufficient sources
    total_candidate_pool = len(filtered_candidates) + len(source)
    safe_diversification = total_candidate_pool >= 8  # Minimum pool size for diversification

    # Choose target length, accounting for entity phrases
    base_max = max(len(filtered_candidates), len(source))
    adjusted_max_words = max(max_words, base_max) if base_max > 0 else max_words
    base_n = random.randint(min_words, min(adjusted_max_words, max(min_words, base_max)))
    # Reserve space for entity phrases
    entity_space = min(len(entity_phrases), 2)  # Limit entity phrases to prevent domination
    n = max(base_n, entity_space + 1)  # Ensure room for at least one other token

    source_count = min(len(source), max(1, int(n * mix_ratio))) if source else 0
    cand_count = max(1, n - source_count - entity_space) if filtered_candidates else 0

    # Prepare tokens with source tracking for diversification
    available_tokens: List[TokenWithSource] = []

    # Add entity phrases first (highest priority)
    for entity in entity_phrases[:entity_space]:
        available_tokens.append(TokenWithSource(entity, "entity_phrase"))

    # Add source tokens (from user input)
    if source_count > 0 and source:
        source_sample = random.sample(source, source_count)
        for token in source_sample:
            # Skip tokens that are part of entity phrases
            if not any(token.lower() in entity.lower() for entity in entity_phrases):
                available_tokens.append(TokenWithSource(token, "input"))

    # Add candidate tokens (from context/memory)
    if cand_count > 0 and filtered_candidates:
        # Simple sampling for now - can be enhanced to prefer certain sources
        cand_sample = filtered_candidates[:cand_count]
        for token in cand_sample:
            # Skip tokens that are part of entity phrases
            if not any(token.lower() in entity.lower() for entity in entity_phrases):
                # Determine source type - simplified for now
                source_type = "context"  # Can be enhanced with more detailed tracking
                available_tokens.append(TokenWithSource(token, source_type))

    if not available_tokens:
        return ""

    # Now build the sentence with anti-repetition and diversification
    chosen_tokens: List[str] = []
    source_counts: Dict[str, int] = {}

    # Prioritize entity phrases first, then shuffle others
    entity_tokens = [t for t in available_tokens if t.source == "entity_phrase"]
    other_tokens = [t for t in available_tokens if t.source != "entity_phrase"]
    random.shuffle(other_tokens)

    # Combine with entity phrases first
    prioritized_tokens = entity_tokens + other_tokens

    for token_with_source in prioritized_tokens:
        token = token_with_source.token
        source_type = token_with_source.source

        # Entity phrases are always included (they're pre-validated)
        if source_type == "entity_phrase":
            chosen_tokens.append(token)
            source_counts[source_type] = source_counts.get(source_type, 0) + 1
            continue

        # Check repetition rules for non-entity tokens
        if not _check_repetition_rules(chosen_tokens, token):
            continue

        # Check bigram rule for non-entity tokens
        if not _check_bigram_rule(chosen_tokens, token):
            continue

        # Check diversification quota (40% max from single source) - but only if safe
        total_tokens = len(chosen_tokens)
        if safe_diversification and total_tokens > 0:
            source_count = source_counts.get(source_type, 0)
            if source_count / total_tokens >= 0.4 and len(source_counts) > 1:
                continue

        # Token passed all checks - add it
        chosen_tokens.append(token)
        source_counts[source_type] = source_counts.get(source_type, 0) + 1

        # Stop if we have enough tokens
        if len(chosen_tokens) >= n:
            break

    if not chosen_tokens:
        return ""

    # Join tokens and apply sentence finishing rules
    sentence = " ".join(chosen_tokens)

    # Apply unicode-aware capitalization
    sentence = _unicode_capitalize(sentence)

    # Determine appropriate ending punctuation
    is_question_msg = _is_question(user_message) if user_message else False

    if is_question_msg:
        if not sentence.endswith("?"):
            sentence += "?"
    else:
        if not sentence.endswith("."):
            sentence += "."

    # Ensure no double punctuation
    sentence = re.sub(r"[.?]+$", "?" if is_question_msg else ".", sentence)

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

    # Prepare context and memory tokens for source tracking
    context_tokens = ctx.unique if ctx else []
    memory_tokens = []

    # If we used memory context, extract memory tokens
    memory_ctx = _context_from_memory(message)
    if memory_ctx:
        memory_tokens = memory_ctx.unique

    # Generate responses with different semantic distances
    first_candidates = _select(ctx.unique, source_words, 0.4)
    second_candidates = _select(ctx.unique, source_words, 0.7)

    first = _compose(
        first_candidates,
        source_words,
        mix_ratio,
        user_message=message,
        context_tokens=context_tokens,
        memory_tokens=memory_tokens,
    )
    second = _compose(
        second_candidates,
        source_words,
        mix_ratio,
        user_message=message,
        context_tokens=context_tokens,
        memory_tokens=memory_tokens,
    )

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
        second = _compose(
            second_candidates,
            source_words,
            mix_ratio,
            user_message=message,
            context_tokens=context_tokens,
            memory_tokens=memory_tokens,
        )

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
