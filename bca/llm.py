"""LLM client abstraction and disk cache for the language channel.

This isolates every OpenAI dependency behind a small :class:`LLMClient` protocol so
that generators and appraisers can be unit-tested with fakes (no network in tests).
The API key is loaded lazily from ``.env`` only when a real call is made, and is
never printed, logged, or stored in ``repr``.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable

Message = dict  # {"role": str, "content": str}


@dataclass
class ChatResult:
    """A single chat completion result."""

    text: str
    usage: Optional[dict] = None
    raw: Any = field(default=None, repr=False)


def token_counts(usage: Optional[dict]) -> dict:
    """Normalise an OpenAI ``usage`` dict to the three standard token counts.

    Returns zeros for any missing field (e.g. a cached or mocked call with no usage),
    so logs always carry numeric token columns.
    """
    usage = usage or {}
    return {
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
    }


@runtime_checkable
class LLMClient(Protocol):
    """Anything that can turn chat messages into a :class:`ChatResult`."""

    def chat(self, model: str, messages: "list[Message]", *,
             temperature: Optional[float] = 0.0, **kwargs: Any) -> ChatResult:
        ...


class OpenAIClient:
    """A thin, lazily-authenticated wrapper over the OpenAI chat API.

    The key is read from the environment (via ``.env``) on first use only, so the
    object can be constructed in environments without a key (e.g. tests, which never
    call :meth:`chat` on a real client). The key is never logged or exposed.
    """

    def __init__(
        self,
        api_key_env: str = "OPENAI_API_KEY",
        env_path: Optional["str | Path"] = None,
        max_calls: Optional[int] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self._api_key_env = api_key_env
        self._env_path = env_path
        self.max_calls = max_calls
        self.base_url = base_url  # None = default OpenAI endpoint
        self.n_calls = 0
        self._client = None  # constructed lazily

    @classmethod
    def from_spec(cls, spec: Any, max_calls: Optional[int] = None,
                  env_path: Optional["str | Path"] = None) -> "OpenAIClient":
        """Build a client for a :class:`~bca.models.ModelSpec`'s endpoint.

        Reads the key from the env var named on the spec (never hardcodes
        ``OPENAI_API_KEY``) and points at the spec's ``base_url`` (``None`` = OpenAI).
        """
        return cls(api_key_env=spec.api_key_env, base_url=spec.base_url,
                   max_calls=max_calls, env_path=env_path)

    def __repr__(self) -> str:  # never leak the key
        return (f"OpenAIClient(endpoint={self.base_url or 'openai-default'}, "
                f"n_calls={self.n_calls}, max_calls={self.max_calls})")

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        from dotenv import load_dotenv

        if self._env_path:
            load_dotenv(self._env_path)
        else:
            load_dotenv()  # cwd / parent search
            # Also load the .env shipped alongside the package, without overriding.
            package_env = Path(__file__).resolve().parent / ".env"
            if package_env.exists():
                load_dotenv(package_env, override=False)
        key = os.environ.get(self._api_key_env)
        if not key:
            raise RuntimeError(
                f"environment variable {self._api_key_env} is not set; cannot call the API"
            )
        from openai import OpenAI

        if self.base_url:
            self._client = OpenAI(api_key=key, base_url=self.base_url)
        else:
            self._client = OpenAI(api_key=key)
        return self._client

    def chat(self, model: str, messages: "list[Message]", *,
             temperature: Optional[float] = 0.0, **kwargs: Any) -> ChatResult:
        if self.max_calls is not None and self.n_calls >= self.max_calls:
            raise RuntimeError(f"max_calls ({self.max_calls}) exceeded")
        client = self._ensure_client()
        # A ``None`` temperature means "do not send this sampling param" (for endpoints
        # that reject it); everything else is forwarded verbatim (e.g. extra_body).
        if temperature is not None:
            kwargs = {"temperature": temperature, **kwargs}
        response = client.chat.completions.create(
            model=model, messages=messages, **kwargs
        )
        self.n_calls += 1
        text = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        usage_dict = usage.model_dump() if hasattr(usage, "model_dump") else None
        return ChatResult(text=text, usage=usage_dict, raw=response)


class JSONCache:
    """A persistent JSON-file cache keyed by ``(model, kind, payload)``."""

    def __init__(self, path: "str | Path", enabled: bool = True) -> None:
        self.path = Path(path)
        self.enabled = enabled
        self._store: dict = {}
        if self.enabled and self.path.exists():
            try:
                self._store = json.loads(self.path.read_text())
            except json.JSONDecodeError:
                self._store = {}

    @staticmethod
    def make_key(model: str, kind: str, payload: dict) -> str:
        """A deterministic content hash over the model, call kind, and input payload."""
        blob = json.dumps(
            {"model": model, "kind": kind, "payload": payload}, sort_keys=True
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[dict]:
        if not self.enabled:
            return None
        return self._store.get(key)

    def set(self, key: str, value: dict) -> None:
        if not self.enabled:
            return
        self._store[key] = value

    def save(self) -> None:
        if not self.enabled:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._store, indent=2))

    def __len__(self) -> int:
        return len(self._store)
