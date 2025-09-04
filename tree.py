"""tree - true recursive echo engine."""

import math
import random
import re
from typing import List

import requests

from roots import store, get_memory
from branches import learn_async, top_words


def entropy(word: str) -> float:
    counts = {ch: word.count(ch) for ch in set(word)}
    total = len(word)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def score(word: str) -> float:
    return entropy(word) * len(set(word))


def select_keywords(text: str) -> List[str]:
    tokens = re.findall(r"\w+", text.lower())
    unique = list(dict.fromkeys(tokens))
    scored = sorted(
        ((w, score(w)) for w in unique), key=lambda x: x[1], reverse=True
    )
    n = min(max(4, len(scored)), 6)
    return [w for w, _ in scored[:n]]


def search(word: str) -> str:
    url = "https://api.duckduckgo.com/"
    params = {
        "q": word,
        "format": "json",
        "no_redirect": "1",
        "no_html": "1",
    }
    try:
        resp = requests.get(url, params=params, timeout=5)
        data = resp.json()
        snippet = (
            data.get("Abstract")
            or data.get("AbstractText")
            or data.get("Definition")
            or data.get("Heading")
            or ""
        )
    except Exception:
        snippet = ""
    store(word, snippet)
    return snippet


def build_context(keywords: List[str], snippets: List[str]) -> str:
    layer1 = " ".join(keywords)
    layer2 = " ".join(snippets)
    memory = []
    for w in keywords:
        memory.extend(get_memory(w))
    layer3 = " ".join(memory)
    layer4 = " ".join(top_words())
    context = " ".join([layer1, layer2, layer3, layer4])
    return context[:2048]


def format_sentence(words: List[str]) -> str:
    if not words:
        return ""
    sentence = " ".join(words)
    return sentence.capitalize() + "."


def make_sentence(
    pool: List[str], user_words: set[str], max_user_words: int
) -> str:
    n = random.randint(4, 8)
    random.shuffle(pool)
    chosen: List[str] = []
    user_count = 0
    for w in pool:
        if w in user_words:
            if user_count >= max_user_words:
                continue
            user_count += 1
        chosen.append(w)
        if len(chosen) >= n:
            break
    filler = ["resonant", "echo", "branch", "root", "flow", "light"]
    while len(chosen) < n:
        chosen.append(random.choice(filler))
    return format_sentence(chosen)


def respond(message: str) -> str:
    keywords = select_keywords(message)
    snippets = [search(w) for w in keywords]
    context = build_context(keywords, snippets)
    learn_async(context)
    user_words = set(re.findall(r"\w+", message.lower()))
    pool = [w for w in re.findall(r"\w+", context.lower()) if w]
    first = make_sentence(pool, user_words, 1)
    second = make_sentence(pool, user_words, 0)
    return f"{first}\n{second}"


if __name__ == "__main__":
    try:
        while True:
            msg = input(">> ")
            print(respond(msg))
    except (EOFError, KeyboardInterrupt):
        pass
