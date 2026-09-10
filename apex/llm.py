"""Compatibility LLM shim delegating to :mod:`apex.providers`."""
from apex.providers import get_provider


def gemini_complete(prompt: str, *, api_key: str) -> dict:
    """Complete one prompt through the configured provider."""
    return get_provider(api_key=api_key).complete(prompt)
