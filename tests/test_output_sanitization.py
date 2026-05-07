from mlxserve.api.app import _sanitize_completion_text


def test_sanitize_truncates_roleplay_markers():
    text = "Bonjour! assistant: Je continue user: encore"
    assert _sanitize_completion_text(text) == "Bonjour!"


def test_sanitize_truncates_leading_assistant_marker():
    text = "assistant: Bonjour, je peux aider."
    assert _sanitize_completion_text(text) == "Bonjour, je peux aider."


def test_sanitize_preserves_code_fences():
    text = "```python\nprint(1)\n```"
    assert "```" in _sanitize_completion_text(text)
    assert "print(1)" in _sanitize_completion_text(text)


def test_sanitize_strips_trailing_incomplete_assistant_word():
    text = "Voici les composants d'un ordinateur portable. Ass"
    assert _sanitize_completion_text(text) == "Voici les composants d'un ordinateur portable."


def test_sanitize_strips_trailing_assistant_word():
    text = "Réponse courte. assistant"
    assert _sanitize_completion_text(text) == "Réponse courte."


def test_sanitize_does_not_truncate_code_with_username_line():
    text = "Voici du code.\nusername = 'x'\nprint(username)"
    assert _sanitize_completion_text(text) == text
