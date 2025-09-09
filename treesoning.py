

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
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

USER_AGENT = "Mozilla/5.0 (compatible; TreeEngine/1.0; +https://github.com/ariannamethod/tree)"

# Common verb endings to detect verbs
VERB_ENDINGS = {
    "ed": "past",    # fixed, walked, talked
    "ing": "present", # walking, talking, fixing
    "s": "present",   # walks, talks, fixes (3rd person)
}

# Pronoun inversion map
PRONOUN_MAP = {
    "you": "me", "your": "my", "yours": "mine", "yourself": "myself",
    "i": "you", "me": "you", "my": "your", "mine": "yours", "myself": "yourself",
    "we": "you", "us": "you", "our": "your", "ours": "yours"
}

@dataclass
class TreeBranch:
    word: str
    word_type: str  # "proper_noun", "verb", "pronoun", "content"
    priority: int   # 1=highest, 3=lowest
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
    if len(word) >= 4 and word_lower not in {"this", "that", "with", "from", "they", "them", "have", "been", "were", "will", "would", "could", "should"}:
        return "content", 3
        
    return "skip", 10

def _build_branch_query(word: str, word_type: str) -> str:
    """Build search query for specific word type."""
    if word_type == "proper_noun":
        return f"what is {word}"
    elif word_type == "verb":
        # Remove common verb endings to get root
        root = word.lower()
        for ending in ["ed", "ing", "s"]:
            if root.endswith(ending) and len(root) > len(ending) + 2:
                root = root[:-len(ending)]
                break
        # Use "how to {root}" but fallback to "{root} meaning" if root is too short
        if len(root) >= 3:
            return f"how to {root}"
        else:
            return f"{root} meaning"
    elif word_type == "pronoun":
        # Queries that reflect perspective shift
        inverted = PRONOUN_MAP.get(word.lower())
        if inverted:
            return f"about {inverted}"
        else:
            return "about you"
    elif word_type == "content":
        return f"about {word.lower()}"
    else:
        # Default fallback
        return word.lower()

async def _fetch_branch_context(query: str) -> str:
    """Fetch context for a single branch query."""
    try:
        # Use DuckDuckGo HTML search as specified
        ddg_url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
        req = urllib.request.Request(ddg_url, headers={"User-Agent": USER_AGENT})
        
        with urllib.request.urlopen(req, timeout=8) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
        
        # Extract text from HTML using simple regex-based approach
        # Remove script and style content
        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
        
        # Extract visible text using regex
        text_content = re.sub(r'<[^>]+>', ' ', html)
        text_content = re.sub(r'\s+', ' ', text_content).strip()
        
        # Return first 500-800 chars of visible text as specified
        if text_content and len(text_content) > 50:
            return text_content[:800]
            
    except Exception:
        pass
    
    return ""

def _invert_pronouns(message: str) -> str:
    """Invert pronouns in message for perspective flip."""
    words = re.findall(r'\b\w+\b|\W+', message)
    result = []
    
    for word in words:
        if re.match(r'\w+', word):
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
    words = re.findall(r'\b\w+\b', message)
    if not words:
        return TreesoningResult([], "conversation flows naturally", 0.0)
    
    # Analyze each word
    branches = []
    for i, word in enumerate(words):
        word_type, priority = _detect_word_type(word, i, len(words))
        
        if word_type == "skip":
            continue
            
        query = _build_branch_query(word, word_type)
        
        branch = TreeBranch(
            word=word,
            word_type=word_type,
            priority=priority,
            query=query
        )
        
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
    
    # Build final intelligent query
    inverted_message = _invert_pronouns(message)
    
    # Strategy: if we have proper nouns or clear context, build specific query
    proper_nouns = [b.word for b in branches if b.word_type == "proper_noun"]
    
    if proper_nouns:
        final_query = f"what it means {inverted_message} and how to respond"
    elif any(x in message.lower() for x in ("how", "what", "why")):
        final_query = f"how to answer {inverted_message}"
    else:
        final_query = f"conversation about {inverted_message}"
    
    # Calculate confidence based on successful context retrieval
    successful_contexts = sum(1 for ctx in contexts if ctx.strip())
    confidence = successful_contexts / max(len(top_branches), 1)
    
    return TreesoningResult(
        branches=branches,
        final_query=final_query,
        confidence=confidence
    )

def _build_combined_context(result: TreesoningResult, branch_contexts: List[str]) -> str:
    """Combine contexts from different branches into coherent window."""
    valid_contexts = [ctx for ctx in branch_contexts if ctx.strip()]
    
    if not valid_contexts:
        return "conversation flows through natural dialogue"
    
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
        return "conversation flows naturally through dialogue", 0.1
    
    # Fetch context for final query
    try:
        final_context = await _fetch_branch_context(result.final_query)
        if final_context:
            return final_context, result.confidence
    except Exception:
        pass
    
    # Fallback
    fallback = f"conversation about {message.strip()}"
    return fallback, 0.2

# For testing
if __name__ == "__main__":
    import sys
    
    async def test_treesoning():
        test_message = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Claude just fixed you"
        
        print(f"Analyzing: '{test_message}'")
        result = await analyze_message(test_message)
        
        print(f"\nBranches found:")
        for branch in result.branches:
            print(f"  {branch.word} ({branch.word_type}) -> {branch.query}")
            if branch.inverted_form:
                print(f"    Inverted: {branch.inverted_form}")
        
        print(f"\nFinal query: {result.final_query}")
        print(f"Confidence: {result.confidence:.2f}")
        
        context, conf = await get_treesoning_context(test_message)
        print(f"\nFinal context: {context}")
    
    asyncio.run(test_treesoning())

