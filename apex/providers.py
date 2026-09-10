"""Provider abstraction layer for Gemini and Ollama."""
import os
import time
from typing import Protocol, runtime_checkable

_RETRY_DELAYS = (1, 2, 4, 8)
_DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"


@runtime_checkable
class Provider(Protocol):
    def complete(self, prompt: str) -> dict:
        """Return {text: str|None, tokens: int, error: str|None}."""
        ...


def _ok(text: str, tokens: int) -> dict:
    return {"text": text, "tokens": tokens, "error": None}


def _err(message: str) -> dict:
    return {"text": None, "tokens": 0, "error": message}


class GeminiProvider:
    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._model = os.environ.get("GEMINI_MODEL", _DEFAULT_GEMINI_MODEL)
        self.total_tokens = 0

    def complete(self, prompt: str) -> dict:
        import google.genai as genai

        last_error = "unknown error"
        for delay in (0, *_RETRY_DELAYS):
            if delay:
                time.sleep(delay)
            try:
                client = genai.Client(api_key=self._api_key or None)
                response = client.models.generate_content(
                    model=self._model,
                    contents=prompt,
                    config={"max_output_tokens": 8192},
                )
                usage = getattr(response, "usage_metadata", None)
                tokens = int(getattr(usage, "total_token_count", 0) or 0)
                text = response.text or ""
                self.total_tokens += tokens
                return _ok(text, tokens)
            except Exception as exc:
                last_error = str(exc)
        return _err(
            f"GeminiProvider failed after {len(_RETRY_DELAYS) + 1} attempts: {last_error}"
        )


class OllamaProvider:
    def __init__(self) -> None:
        self._base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        self._model = os.environ.get("OLLAMA_MODEL", "llama3")
        self.total_tokens = 0

    def complete(self, prompt: str) -> dict:
        import json
        import urllib.request

        last_error = "unknown error"
        for delay in (0, *_RETRY_DELAYS):
            if delay:
                time.sleep(delay)
            try:
                payload = json.dumps(
                    {
                        "model": self._model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.2},
                    }
                ).encode()
                request = urllib.request.Request(
                    f"{self._base}/api/generate",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=300) as response:
                    data = json.loads(response.read())
                text = data.get("response", "")
                prompt_tokens = int(data.get("prompt_eval_count") or 0)
                output_tokens = int(data.get("eval_count") or 0)
                tokens = prompt_tokens + output_tokens
                if tokens == 0:
                    tokens = len(prompt.split()) + len(text.split())
                self.total_tokens += tokens
                return _ok(text, tokens)
            except Exception as exc:
                last_error = str(exc)
        return _err(
            f"OllamaProvider failed after {len(_RETRY_DELAYS) + 1} attempts: {last_error}"
        )


def get_provider(api_key: str = "") -> GeminiProvider | OllamaProvider:
    provider = os.environ.get("LLM_PROVIDER", "gemini").lower()
    if provider == "ollama":
        return OllamaProvider()
    if provider == "gemini":
        return GeminiProvider(api_key=api_key)
    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")
