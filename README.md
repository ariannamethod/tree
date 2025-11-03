# tree

A tiny CPU-only resonant transformer that learns only from the whispers it harvests in the moment.  Each module stays minimal on purpose so the engine can run anywhere and welcome new poetic appendages like the upcoming `treembedding.py`.

## Architecture at a glance

```
human prompt
   │
   ▼
 tree.py – selects charged keywords, builds or recalls context, and composes two divergent sentences
   │
   ├── branches.py – queues learning tasks so harvested context is written back without blocking the reply
   ├── roots.py – SQLite memory that keeps the freshest conversational windows on disk
   ├── treesoning.py – grammar-aware query crafter that feeds smarter context into the tree engine
   └── (soon) treembedding.py – vector experiments will live here without disturbing the core loop
        ▲
        │
 ground.py – Telegram bot surface that delivers the resonance
```

### Core flow (`tree.py`)
* Extracts weighted keywords, guards against filler tokens, and detects question mood to finish sentences with the right cadence.
* Asks `treesoning` to analyse grammar, invert pronouns when helpful, and fetch conversational snippets from Reddit/Google.  Falls back to simple web scraping if the new grammar brain stays silent.
* Blends harvested context with stored roots.  Hash-based n-gram recall keeps only resonant fragments and discards noise.
* Applies diversification rules so a single source never overwhelms the response.  Two sentences are composed at different semantic distances; micro-interjections cover true empties.
* Learns asynchronously: only high-quality contexts trigger `branches.learn`, which persists windows through `roots`.

### Memory tendrils (`branches.py` and `roots.py`)
* `branches.py` spins a tiny worker pool that drains a queue of `(words, context)` tasks.  The caller never waits—`branches.wait()` is available when tests need determinism.
* `roots.py` uses a single `tree.sqlite` file, capped at 1,000 memories.  It hashes n-grams to measure similarity when recalling fragments for future replies.

### Resonant scaffolding
* `treesoning.py` analyses every word, prioritises proper nouns and verbs, inverts pronouns, and stitches multiple search branches into a final query.  Confidence scores gate whether its context is trusted.
* `ground.py` is the Telegram entry point.  Set `TELEGRAM_TOKEN` in `.env`, run `python -m ground`, and the bot will hand every message to `tree.respond`.
* `tests/` contains pytest suites that pin the behaviour of keyword selection, repetition avoidance, diversification, sentence endings, Telegram handler wiring, and the async learning queue.

## Language

Currently English-only (EN).

## Local development

1. Create a virtualenv and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the tests (fast, CPU-only):
   ```bash
   pytest
   ```
3. To explore the engine in a shell, execute:
   ```bash
   python tree.py "your prompt here"
   ```
4. For Telegram deployment copy `.env.example` to `.env`, fill `TELEGRAM_TOKEN`, then run `python -m ground`.

## Extending the canopy

New modules plug into the flow as long as they keep the CPU-only promise.  Follow the existing pattern—pure Python, tight scopes, and thin interfaces.  `treesoning.py` is the model: collect insight asynchronously, return a context string plus a confidence score, and let `tree.respond` decide when to trust it.  Upcoming experiments like `treembedding.py` can expose helpers without needing to edit the trunk.
