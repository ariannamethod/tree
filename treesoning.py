"""treesoning.py - Grammar-aware context builder for tree engine

Builds intelligent search queries by analyzing grammatical structure:
- Capitalized words (proper nouns) get highest priority
- Verbs get action-oriented queries
- Pronouns get inverted
- Final query combines all branches intelligently
"""

import re
import asyncio
import urllib.parse
import urllib.request
import json
from typing import List, Tuple, Optional
from dataclasses import dataclass

USER_AGENT = "Mozilla/5.0 (compatible; TreeEngine/1.0; +https://github.com/ariannamethod/tree)"

# Common verb endings to detect verbs
VERB_ENDINGS = {
    "ed": "past",  # fixed, walked, talked
    "ing": "present",  # walking, talking, fixing
    "s": "present",  # walks, talks, fixes (3rd person)
}

# Pronoun inversion map
PRONOUN_MAP = {
    "you": "me",
    "your": "my",
    "yours": "mine",
    "yourself": "myself",
    "i": "you",
    "me": "you",
    "my": "your",
    "mine": "yours",
    "myself": "yourself",
    "we": "you",
    "us": "you",
    "our": "your",
    "ours": "yours",
}


@dataclass
class TreeBranch:
    word: str
    word_type: str  # "proper_noun", "verb", "pronoun", "content"
    priority: int  # 1=highest, 3=lowest
    query: str
    inverted_form: Optional[str] = None


@dataclass
class TreesoningResult:
    branches: List[TreeBranch]
    final_query: str
    confidence: float


def _detect_word_type(word: str, position: int, total_words: int) -> Tuple[str, int]:
    """Detect grammatical type and assign priority."""
    word_lower = word.lower()

    # Skip if too short or common
    if len(word) < 2:
        return "skip", 10

    # Highest priority: Proper nouns (capitalized, not sentence start)
    if word[0].isupper() and position > 0:
        return "proper_noun", 1

    # High priority: Pronouns (for inversion)
    if word_lower in PRONOUN_MAP:
        return "pronoun", 1

    # Medium priority: Verbs (by endings)
    for ending, tense in VERB_ENDINGS.items():
        if word_lower.endswith(ending) and len(word) > len(ending) + 2:
            return "verb", 2

    # Lower priority: Content words (nouns, adjectives, etc.)
    if len(word) >= 4 and word_lower not in {
        "this",
        "that",
        "with",
        "from",
        "they",
        "them",
        "have",
        "been",
        "were",
        "will",
        "would",
        "could",
        "should",
    }:
        return "content", 3

    return "skip", 10


def _build_branch_query(word: str, word_type: str) -> str:
    """Build language-agnostic structural query for specific word type."""
    if word_type == "proper_noun":
        return f'"{word}"'  # Use exact quoted search
    elif word_type == "verb":
        # Remove common verb endings to get root
        root = word.lower()
        for ending in ["ed", "ing", "s"]:
            if root.endswith(ending) and len(root) > len(ending) + 2:
                root = root[: -len(ending)]
                break
        return f'"{root}"'  # Exact search for verb root
    elif word_type == "content":
        return f'"{word}"'  # Exact search
    else:
        return f'"{word}"'  # Default to exact search


async def _fetch_branch_context(query: str) -> str:
    """Fetch context for a single branch query."""
    try:
        # Try Reddit first for conversational context
        reddit_url = (
            f"https://www.reddit.com/search.json?q={urllib.parse.quote(query)}"
            "&limit=2&sort=relevance"
        )
        req = urllib.request.Request(reddit_url, headers={"User-Agent": USER_AGENT})

        with urllib.request.urlopen(req, timeout=6) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        posts = data.get("data", {}).get("children", [])
        snippets = []

        for post in posts[:1]:  # Just one result per branch
            post_data = post.get("data", {})
            title = post_data.get("title", "").strip()
            if title and len(title) > 10:
                snippets.append(title[:200])

        if snippets:
            return " ".join(snippets)

    except Exception:
        pass

    # Fallback: simple Google search
    try:
        google_url = f"https://www.google.com/search?q={urllib.parse.quote(query)}&num=1"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        req = urllib.request.Request(google_url, headers=headers)

        with urllib.request.urlopen(req, timeout=8) as resp:
            html = resp.read().decode("utf-8", "ignore")

        # Extract first meaningful snippet
        titles = re.findall(r"<h3[^>]*>([^<]+)</h3>", html)
        if titles and len(titles[0]) > 5:
            return titles[0][:150]

    except Exception:
        pass

    return ""


def _invert_pronouns(message: str) -> str:
    """Invert pronouns in message for perspective flip."""
    words = re.findall(r"\b\w+\b|\W+", message)
    result = []

    for word in words:
        if re.match(r"\w+", word):
            lower_word = word.lower()
            if lower_word in PRONOUN_MAP:
                # Preserve capitalization
                inverted = PRONOUN_MAP[lower_word]
                if word[0].isupper():
                    inverted = inverted.capitalize()
                result.append(inverted)
            else:
                result.append(word)
        else:
            result.append(word)

    return "".join(result)


async def analyze_message(message: str) -> TreesoningResult:
    """Main treesoning function - analyze message and build intelligent queries."""
    words = re.findall(r"\b\w+\b", message)
    if not words:
        return TreesoningResult([], "", 0.0)

    # Analyze each word
    branches = []
    for i, word in enumerate(words):
        word_type, priority = _detect_word_type(word, i, len(words))

        if word_type == "skip":
            continue

        query = _build_branch_query(word, word_type)

        branch = TreeBranch(word=word, word_type=word_type, priority=priority, query=query)

        # Handle pronoun inversion
        if word_type == "pronoun":
            branch.inverted_form = PRONOUN_MAP.get(word.lower())

        branches.append(branch)

    # Sort by priority (1=highest priority first)
    branches.sort(key=lambda b: b.priority)

    # Take top 3 branches for context gathering
    top_branches = branches[:3]

    # Fetch context for each branch asynchronously
    tasks = [_fetch_branch_context(branch.query) for branch in top_branches]
    contexts = await asyncio.gather(*tasks)

    # Build final query using structural approach with pronoun inversion
    inverted_message = _invert_pronouns(message)

    # Issue 2-3 parallel queries using structural, language-agnostic approach:
    # 1) Exact quoted input
    # 2) Pronoun-inverted variant
    # 3) AND combination of key terms
    proper_nouns = [b.word for b in branches if b.word_type == "proper_noun"]

    if proper_nouns:
        # Use proper nouns for structured search
        final_query = (
            f'"{inverted_message}" AND {" AND ".join(f'"{noun}"' for noun in proper_nouns[:2])}'
        )
    else:
        # Use exact quoted search or pronoun-inverted variant
        final_query = f'"{inverted_message}"'

    # Calculate confidence based on successful context retrieval
    successful_contexts = sum(1 for ctx in contexts if ctx.strip())
    confidence = successful_contexts / max(len(top_branches), 1)

    return TreesoningResult(branches=branches, final_query=final_query, confidence=confidence)


def _build_combined_context(result: TreesoningResult, branch_contexts: List[str]) -> str:
    """Combine contexts from different branches into coherent window."""
    valid_contexts = [ctx for ctx in branch_contexts if ctx.strip()]

    if not valid_contexts:
        return ""

    # Limit each context and combine
    limited_contexts = [ctx[:200] for ctx in valid_contexts]
    combined = " ".join(limited_contexts)

    return combined[:800]  # Overall limit


async def get_treesoning_context(message: str) -> Tuple[str, float]:
    """
    Public interface for tree.py to get intelligent context.
    Returns (context_text, confidence_score)
    """
    result = await analyze_message(message)

    if not result.branches:
        return "", 0.1

    # Fetch context for final query
    try:
        final_context = await _fetch_branch_context(result.final_query)
        if final_context:
            return final_context, result.confidence
    except Exception:
        pass

    # Return empty string on failure as per requirements
    return "", 0.0


# For testing
if __name__ == "__main__":
    import sys

    async def test_treesoning():
        test_message = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Claude just fixed you"

        print(f"Analyzing: '{test_message}'")
        result = await analyze_message(test_message)

        print("\nBranches found:")
        for branch in result.branches:
            print(f"  {branch.word} ({branch.word_type}) -> {branch.query}")
            if branch.inverted_form:
                print(f"    Inverted: {branch.inverted_form}")

        print(f"\nFinal query: {result.final_query}")
        print(f"Confidence: {result.confidence:.2f}")

        context, conf = await get_treesoning_context(test_message)
        print(f"\nFinal context: {context}")

    asyncio.run(test_treesoning())
