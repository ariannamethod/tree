import tree


def dummy_search(word: str) -> str:
    return f"snippet {word}"


def test_respond(monkeypatch):
    monkeypatch.setattr(tree, 'search', dummy_search)
    reply = tree.respond('meaning of life')
    lines = [line for line in reply.strip().split('\n') if line]
    assert len(lines) == 2
    for line in lines:
        words = line.split()
        assert 4 <= len(words) <= 8
