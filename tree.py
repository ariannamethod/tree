from __future__ import annotations

"""true recursive echo engine

The :mod:`tree` module is a minimalist transformer inspired construct.
It selects charged words from a user prompt, harvests tiny webs of
context from conversational sources, and composes two resonant sentences that drift
semantically away from the original message.
"""

import asyncio
import logging
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
import ner
import treembedding
import constraints
import guide

# Set up logging for debugging entity detection and candidate selection
logger = logging.getLogger(__name__)


# Feature flag for English Guide
USE_ENGLISH_GUIDE = True

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
    return {hash(text[i : i + n]) for i in range(len(text) - n + 1)}


def _context_from_memory(message: str) -> Context | None:
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
    logger.debug(f"_select: input pool size={len(tokens)}, target distance={target:.2f}, limit={limit}")
    
    if not tokens or not source:
        logger.debug("_select: empty tokens or source, returning empty")
        return []

    src_vec = _source_vector(source)
    if not src_vec:
        logger.debug("_select: no source vector, returning random selection")
        random.shuffle(tokens)
        result = tokens[:limit]
        logger.debug(f"_select: random selection returned {len(result)} tokens")
        return result

    graded = []
    for t in tokens:
        if t in source or len(t) < 3:
            continue

        vec = _normalize(NGRAMS.get(t, Counter()))
        cos = sum(vec.get(k, 0.0) * src_vec.get(k, 0.0) for k in vec)
        distance = 1.0 - cos
        graded.append((abs(distance - target), t))

    graded.sort()
    result = [g[1] for g in graded[:limit]]
    logger.debug(f"_select: graded {len(graded)} tokens, returning {len(result)} tokens")
    return result


def _filter_source_gated_tokens(
    candidates: List[str],
    user_tokens: List[str],
    context_tokens: List[str],
    memory_tokens: List[str],
) -> List[str]:
    """Filter candidates to only include tokens from valid sources, applying denylist."""
    logger.debug(f"_filter_source_gated_tokens: filtering {len(candidates)} candidates")
    logger.debug(f"Source tokens - user: {len(user_tokens)}, context: {len(context_tokens)}, memory: {len(memory_tokens)}")
    
    # Collect all allowed tokens from sources
    allowed_tokens = set()

    # Add tokens from all valid sources (normalized for comparison)
    for token in user_tokens:
        allowed_tokens.add(_normalize_token(token))
    for token in context_tokens:
        allowed_tokens.add(_normalize_token(token))
    for token in memory_tokens:
        allowed_tokens.add(_normalize_token(token))

    logger.debug(f"Total allowed tokens from sources: {len(allowed_tokens)}")

    # Filter candidates
    filtered = []
    denied_count = 0
    not_in_sources_count = 0
    
    for candidate in candidates:
        normalized = _normalize_token(candidate)

        # Check denylist - if token is denylisted, only allow if in sources
        if normalized in DENYLIST_TOKENS:
            if normalized in allowed_tokens:
                filtered.append(candidate)
            else:
                denied_count += 1
            # else: skip denylisted token not in sources
        else:
            # Non-denylisted token - only allow if from valid sources
            if normalized in allowed_tokens:
                filtered.append(candidate)
            else:
                not_in_sources_count += 1

    logger.debug(f"Filtering results - kept: {len(filtered)}, denied: {denied_count}, not in sources: {not_in_sources_count}")
    
    if len(filtered) < len(candidates) * 0.1:  # If we filtered out more than 90%
        logger.debug("FALLBACK WARNING: POOL_TOO_SMALL after filtering")
    
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
    """Compose a more natural sentence blending source words with filtering and anti-repetition."""
    if context_tokens is None:
        context_tokens = []
    if memory_tokens is None:
        memory_tokens = []

    if not candidates and not source:
        return ""  # Return empty for fallback chain handling

    # Extract user input tokens
    user_tokens = re.findall(r"\b\w+\b", user_message.lower()) if user_message else []

    # Apply source-gated filtering
    filtered_candidates = _filter_source_gated_tokens(
        candidates, user_tokens, context_tokens, memory_tokens
    )

    # If all candidates were filtered out, return empty for fallback
    if not filtered_candidates and not source:
        return ""

    # Choose target length
    n = random.randint(min_words, min(max_words, max(len(filtered_candidates), len(source))))
    source_count = min(len(source), max(1, int(n * mix_ratio))) if source else 0
    cand_count = max(1, n - source_count) if filtered_candidates else 0

    # Prepare tokens with source tracking for diversification
    available_tokens: List[TokenWithSource] = []

    # Add source tokens (from user input)
    if source_count > 0 and source:
        source_sample = random.sample(source, source_count)
        for token in source_sample:
            available_tokens.append(TokenWithSource(token, "input"))

    # Add candidate tokens (from context/memory)
    if cand_count > 0 and filtered_candidates:
        # Simple sampling for now - can be enhanced to prefer certain sources
        cand_sample = filtered_candidates[:cand_count]
        for token in cand_sample:
            # Determine source type - simplified for now
            source_type = "context"  # Can be enhanced with more detailed tracking
            available_tokens.append(TokenWithSource(token, source_type))

    if not available_tokens:
        return ""

    # Now build the sentence with anti-repetition and diversification
    chosen_tokens: List[str] = []
    source_counts: Dict[str, int] = {}

    random.shuffle(available_tokens)

    for token_with_source in available_tokens:
        token = token_with_source.token
        source_type = token_with_source.source

        # Check repetition rules
        if not _check_repetition_rules(chosen_tokens, token):
            continue

        # Check bigram rule
        if not _check_bigram_rule(chosen_tokens, token):
            continue

        # Check diversification quota (40% max from single source)
        total_tokens = len(chosen_tokens)
        if total_tokens > 0:
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
    logger.debug(f"=== RESPOND START ===")
    logger.debug(f"Input text: '{message}'")
    
    if not message.strip():
        logger.debug("Empty input - returning micro-interjection")
        return _get_micro_interjection(message)

    # Check English Guide for pattern matches (when enabled)
    if USE_ENGLISH_GUIDE:
        matched = guide.match(message.strip())
        if matched:
            logger.debug(f"Matched guide pattern: {matched.pattern}")
            template = guide.choose_template(matched.templates)
            if template:
                logger.debug(f"Using guide template: {template}")
                # Apply constraints to finalize the text
                final_response = constraints.finalize_text(template)
                logger.debug(f"Final response from guide: '{final_response}'")
                logger.debug(f"=== RESPOND END (via guide) ===")
                return final_response

    _update_ngrams_from_roots()

    # Detect entities using NER system
    tokens, detected_entities = ner.detect_entities(message)
    logger.debug(f"Detected entities: {list(detected_entities)}")
    logger.debug(f"Tokenized input: {tokens}")
    
    # Update treembedding system with input sequence
    treembedding.embedding_system.update_sequence(tokens)

    lang = _detect_language(message)
    keys = _keywords(message)
    logger.debug(f"Language: {lang}, Keywords: {keys}")

    # Step 1: Try normal provider/context building
    ctx = _context(message, keys, lang)
    logger.debug(f"Initial context quality score: {ctx.quality_score if ctx else 0}")
    logger.debug(f"Initial candidate pool size: {len(ctx.unique) if ctx else 0}")

    # Step 2: If context is empty/low-quality, attempt memory recall from roots
    if not _is_context_usable_for_output(ctx):
        logger.debug("Context not usable for output - attempting memory recall")
        memory_ctx = _context_from_memory(message)
        if memory_ctx and _is_context_usable_for_output(memory_ctx):
            logger.debug(f"Memory context found - quality: {memory_ctx.quality_score}, candidates: {len(memory_ctx.unique)}")
            ctx = memory_ctx
        else:
            logger.debug("Memory context not available or not usable")

    # Step 3: If still empty or too short, return micro-interjection
    if not _is_context_usable_for_output(ctx):
        logger.debug("FALLBACK TRIGGERED: NO_VALID_CONTEXT - returning micro-interjection")
        return _get_micro_interjection(message)

    source_words = re.findall(r"\b\w+\b", message.lower())
    mix_ratio = 0.5 - 0.2 * min(ctx.quality_score, 1.0)
    logger.debug(f"Source words from input: {len(source_words)}, Mix ratio: {mix_ratio:.2f}")

    # Prepare context and memory tokens for source tracking
    context_tokens = ctx.unique if ctx else []
    memory_tokens = []

    # If we used memory context, extract memory tokens
    memory_ctx = _context_from_memory(message)
    if memory_ctx:
        memory_tokens = memory_ctx.unique

    # Preserve entities in context tokens for atomic handling
    if detected_entities:
        context_tokens = ner.preserve_entities_in_tokens(context_tokens, detected_entities)
        logger.debug(f"Context tokens after entity preservation: {len(context_tokens)}")

    # Generate responses with different semantic distances
    first_candidates = _select(ctx.unique, source_words, 0.4)
    second_candidates = _select(ctx.unique, source_words, 0.7)
    
    logger.debug(f"First candidates pool size: {len(first_candidates)}")
    logger.debug(f"Second candidates pool size: {len(second_candidates)}")

    # Use treembedding to enhance candidate selection
    if first_candidates and source_words:
        first_candidates = treembedding.embedding_system.suggest_next(
            source_words[-1] if source_words else "", first_candidates
        )
    if second_candidates and source_words:
        second_candidates = treembedding.embedding_system.suggest_next(
            source_words[-1] if source_words else "", second_candidates
        )

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
    
    logger.debug(f"Generated first response: '{first}'")
    logger.debug(f"Generated second response: '{second}'")

    # Handle empty compositions
    if not first and not second:
        logger.debug("FALLBACK TRIGGERED: EMPTY_COMPOSITIONS - returning micro-interjection")
        return _get_micro_interjection(message)
    elif not first:
        first = second
        second = ""
        logger.debug("First composition empty, using second as first")
    elif not second:
        second = ""
        logger.debug("Second composition empty")

    # Ensure responses are different enough if both exist
    if first and second and SequenceMatcher(None, first, second).ratio() > 0.6:
        logger.debug("Responses too similar, regenerating second with different distance")
        # Regenerate second response with different distance
        second_candidates = _select(ctx.unique, source_words, 0.9)
        logger.debug(f"Regenerated second candidates pool size: {len(second_candidates)}")
        second = _compose(
            second_candidates,
            source_words,
            mix_ratio,
            user_message=message,
            context_tokens=context_tokens,
            memory_tokens=memory_tokens,
        )
        logger.debug(f"Regenerated second response: '{second}'")

    # Learning guard: only learn when context passes training threshold
    if _is_context_usable_for_training(ctx):
        logger.debug("Context usable for training - learning from interaction")
        branches.learn(keys, ctx.raw)
        _update_ngram_with_text(ctx.raw)
        
        # Update treembedding system with successful generation
        if first:
            response_tokens = ner.tokenize_unicode_aware(first)
            treembedding.embedding_system.update_sequence(response_tokens)
        if second:
            response_tokens = ner.tokenize_unicode_aware(second)
            treembedding.embedding_system.update_sequence(response_tokens)
        
        branches.wait()
    else:
        logger.debug("Context not suitable for training - skipping learning")

    # Return composed response
    final_response = ""
    if first and second:
        final_response = f"{first} {second}"
    elif first:
        final_response = first
    else:
        logger.debug("FALLBACK TRIGGERED: NO_COMPOSITIONS - returning micro-interjection")
        final_response = _get_micro_interjection(message)
    
    # Apply constraints to finalize the text
    final_response = constraints.finalize_text(final_response)
    
    logger.debug(f"Final response: '{final_response}'")
    logger.debug(f"=== RESPOND END ===")
    return final_response


if __name__ == "__main__":  # pragma: no cover - manual exercise
    import sys

    user_input = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("> ")
    print(respond(user_input))
