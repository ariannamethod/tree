import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("telegram")

import ground  # noqa: E402


def test_message_triggers_tree_response():
    update = MagicMock()
    context = MagicMock()
    update.message.text = "hello"

    with patch.object(
        ground.tree, "respond", return_value="hi"
    ) as mock_respond:
        ground.handle_message(update, context)

    mock_respond.assert_called_once_with("hello")
    update.message.reply_text.assert_called_once_with("hi")
