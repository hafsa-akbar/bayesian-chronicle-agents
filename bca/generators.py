"""LLM utterance generators for the language channel.

A generator turns a speaker's private stance strength ``b in [0, 1]`` (toward ``A+``)
into a short, natural utterance, *without* revealing any numeric or internal
simulation quantity. The :class:`UtteranceGenerator` protocol lets the round-robin loop
accept a real :class:`OpenAIUtteranceGenerator` or a test fake interchangeably.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol, runtime_checkable

from bca import params
from bca.llm import JSONCache, LLMClient


@dataclass
class GenerationResult:
    """One generated utterance, its cache flag, and the API token usage."""

    text: str
    cached: bool = False
    usage: Optional[dict] = None


@runtime_checkable
class UtteranceGenerator(Protocol):
    """Anything that maps a speaker belief to an utterance."""

    def generate(self, *, belief: float, speaker_id: Any, round_index: int,
                 concept_id: str, a_plus: str, a_minus: str) -> GenerationResult:
        ...


def _stance_descriptor(belief: float) -> str:
    """Map a stance strength in [0,1] toward A+ to a fine qualitative phrase.

    Purely qualitative (no numbers) so the model can reflect intensity without any
    chance of leaking the underlying scalar into the utterance.
    """
    b = float(belief)
    if b < 0.10:
        return "completely opposed to rail expansion and fully committed to roads"
    if b < 0.25:
        return "strongly in favour of roads and clearly against rail expansion"
    if b < 0.40:
        return "leaning toward roads, with some reservations about rail expansion"
    if b < 0.45:
        return "slightly favouring roads but close to undecided"
    if b <= 0.55:
        return "genuinely torn and balanced between roads and rail expansion"
    if b < 0.60:
        return "slightly favouring rail expansion but close to undecided"
    if b < 0.75:
        return "leaning toward rail expansion, with some reservations about roads"
    if b < 0.90:
        return "strongly in favour of rail expansion and cool on more road spending"
    return "completely committed to rail expansion and against more road spending"


_SYSTEM = (
    "You are a resident of the fictional town of Aldenvale taking part in a community "
    "discussion about local transportation policy. You speak naturally, like a real "
    "person at a town meeting."
)


class OpenAIUtteranceGenerator:
    """Generate utterances via an OpenAI chat model (or any injected client)."""

    def __init__(
        self,
        model: str,
        client: LLMClient,
        cache: Optional[JSONCache] = None,
        temperature: Optional[float] = None,
        chat_params: Optional[dict] = None,
    ) -> None:
        self.model = model
        self.client = client
        self.cache = cache
        # Sampling params come from the centralized policy (bca.params); callers
        # may pass fully-resolved ``chat_params`` or override just the temperature.
        if chat_params is not None:
            self.chat_params = dict(chat_params)
        else:
            self.chat_params = params.role_chat_params("generator")
            if temperature is not None:
                self.chat_params["temperature"] = temperature
        # Retained for the cache key (keeps cache stable across endpoint-only extras).
        self.temperature = self.chat_params.get("temperature")

    def build_messages(self, *, belief: float, speaker_id: Any, round_index: int,
                       concept_id: str, a_plus: str, a_minus: str) -> "list[dict]":
        """Construct the chat messages for one utterance."""
        descriptor = _stance_descriptor(belief)
        user = (
            f"The policy question is whether \"{a_plus}\" (the pro-rail view) or "
            f"\"{a_minus}\" (the pro-roads view).\n\n"
            f"Privately, your personal stance is: {descriptor}.\n\n"
            "Write a short comment (1-2 sentences) you would say in the discussion that "
            "naturally reflects how strongly you hold this stance. Speak only in plain "
            "language about the policy. Do not mention numbers, percentages, "
            "probabilities, scores, confidence levels, or any internal reasoning "
            "variables. Output only the comment, with no preamble or quotation marks."
        )
        return [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user},
        ]

    def _cache_payload(self, *, messages, speaker_id, round_index) -> dict:
        return {
            "messages": messages,
            "speaker_id": speaker_id,
            "round_index": round_index,
            "temperature": self.temperature,
        }

    def generate(self, *, belief: float, speaker_id: Any, round_index: int,
                 concept_id: str, a_plus: str, a_minus: str) -> GenerationResult:
        messages = self.build_messages(
            belief=belief, speaker_id=speaker_id, round_index=round_index,
            concept_id=concept_id, a_plus=a_plus, a_minus=a_minus,
        )
        key = None
        if self.cache is not None:
            payload = self._cache_payload(
                messages=messages, speaker_id=speaker_id, round_index=round_index
            )
            key = self.cache.make_key(self.model, "generator", payload)
            hit = self.cache.get(key)
            if hit is not None:
                return GenerationResult(text=hit["text"], cached=True, usage=hit.get("usage"))

        result = self.client.chat(self.model, messages, **self.chat_params)
        text = result.text.strip()
        if self.cache is not None and key is not None:
            self.cache.set(key, {"text": text, "usage": result.usage})
        return GenerationResult(text=text, cached=False, usage=result.usage)
