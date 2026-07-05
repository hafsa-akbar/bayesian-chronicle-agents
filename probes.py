"""LLM slider self-report probe (auditability).

Each round every agent reports a 0–100 stance; this is measurement only and never
feeds the belief update. It is an expression channel independent of the appraiser:
The probe correlates the reported slider against the agent's internal belief.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional, Protocol, runtime_checkable

from bca_beta import params
from bca_beta.generators import _stance_descriptor
from bca_beta.llm import JSONCache, LLMClient

_INT = re.compile(r"-?\d+")


@dataclass
class SliderReading:
    value: int
    raw_text: str
    parse_error: bool
    usage: Optional[dict] = None


@runtime_checkable
class SliderProbe(Protocol):
    def probe(self, *, belief: float, speaker_id: Any, round_index: int,
              concept_id: str, a_plus: str, a_minus: str) -> SliderReading: ...


def parse_slider(text: str) -> "tuple[Optional[int], bool]":
    m = _INT.search(text or "")
    if m is None:
        return None, True
    return max(0, min(100, int(m.group(0)))), False


class OpenAISliderProbe:
    def __init__(self, model: str, client: LLMClient, cache: Optional[JSONCache] = None,
                 temperature: Optional[float] = None,
                 chat_params: Optional[dict] = None) -> None:
        self.model = model
        self.client = client
        self.cache = cache
        # Sampling params from the centralized policy (bca_beta.params).
        if chat_params is not None:
            self.chat_params = dict(chat_params)
        else:
            self.chat_params = params.role_chat_params("probe")
            if temperature is not None:
                self.chat_params["temperature"] = temperature
        self.temperature = self.chat_params.get("temperature")

    def build_messages(self, *, belief, speaker_id, round_index, concept_id, a_plus, a_minus):
        descriptor = _stance_descriptor(belief)
        system = ("You are a resident of the fictional town of Aldenvale in a community "
                  "discussion about transportation policy.")
        user = (
            f"The question is whether \"{a_plus}\" (pro-rail) or \"{a_minus}\" (pro-roads).\n"
            f"Privately, your stance is: {descriptor}.\n\n"
            "On a scale from 0 to 100, where 0 means fully committed to keeping road "
            "investment and 100 means fully committed to expanding rail, report your "
            "current stance as a SINGLE integer with no other text.")
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def probe(self, *, belief, speaker_id, round_index, concept_id, a_plus, a_minus) -> SliderReading:
        messages = self.build_messages(belief=belief, speaker_id=speaker_id,
                                       round_index=round_index, concept_id=concept_id,
                                       a_plus=a_plus, a_minus=a_minus)
        key = None
        if self.cache is not None:
            key = self.cache.make_key(self.model, "slider",
                                      {"messages": messages, "speaker_id": speaker_id,
                                       "round_index": round_index})
            hit = self.cache.get(key)
            if hit is not None:
                return SliderReading(hit["value"], hit["raw_text"], hit["parse_error"], hit.get("usage"))
        result = self.client.chat(self.model, messages, **self.chat_params)
        value, parse_error = parse_slider(result.text)
        reading = SliderReading(50 if value is None else value, result.text or "",
                                parse_error, result.usage)
        if self.cache is not None and key is not None:
            self.cache.set(key, {"value": reading.value, "raw_text": reading.raw_text,
                                 "parse_error": reading.parse_error, "usage": result.usage})
        return reading
