from __future__ import annotations

import logging
import os
import signal

from telegram import Update
from telegram.ext import CallbackContext, Filters, MessageHandler, Updater

import tree

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def handle_message(update: Update, context: CallbackContext) -> None:
    """Handle incoming messages by responding via ``tree.respond``."""
    try:
        text = update.message.text or ""
        response = tree.respond(text)
        update.message.reply_text(response)
    except Exception:  # pragma: no cover - defensive
        logger.exception("error handling message")


def main() -> None:
    """Run the Telegram bot."""
    token = os.environ.get("TELEGRAM_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_TOKEN not set")

    updater = Updater(token, use_context=True)
    dispatcher = updater.dispatcher
    dispatcher.add_handler(
        MessageHandler(Filters.text & ~Filters.command, handle_message)
    )

    def _shutdown(signum, frame):
        logger.info("shutting down")
        updater.stop()
        updater.is_idle = False

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _shutdown)

    updater.start_polling()
    updater.idle()


if __name__ == "__main__":  # pragma: no cover - manual run
    main()
