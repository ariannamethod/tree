# tree

A tiny CPU-only resonant transformer.  The engine lives in three modules:

* `tree.py` – the true recursive echo engine
* `branches.py` – asynchronous learners that grow the model
* `roots.py` – sqlite memory for harvested contexts

The system starts with no weights or dataset.  For every user message the
engine searches the web for a handful of charged keywords, stores the gathered
text and answers with two semantically distant sentences.

## Deployment

Copy `.env.example` to `.env` and set your Telegram bot token in the `TELEGRAM_TOKEN` variable before running the application.
