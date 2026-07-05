"""LLM stance appraisers for the language channel.

An appraiser maps an utterance to ``p_plus`` -- the probability it supports ``A+``
versus ``A-`` -- and (optionally) applies a fitted temperature calibration. It is the
inverse of the generator: ``speaker belief -> utterance -> appraiser -> evidence``.
The :class:`StanceAppraiser` protocol lets the round-robin loop swap in a test fake.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

from bca_beta import params
from bca_beta.calibration import Calibration
from bca_beta.llm import JSONCache, LLMClient

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class AppraisalResult:
    """The appraiser's verdict on one utterance, plus the API token usage."""

    p_plus_raw: float
    p_plus_calibrated: float
    rationale: str
    parse_error: bool
    usage: Optional[dict] = None


@runtime_checkable
class StanceAppraiser(Protocol):
    """Anything that maps an utterance to a stance probability."""

    def appraise(self, *, utterance: str, a_plus: str, a_minus: str) -> AppraisalResult:
        ...


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def parse_appraiser_response(text: str) -> "tuple[Optional[float], str, bool]":
    """Extract ``(p_plus_raw, rationale, parse_error)`` from a raw model response."""
    match = _JSON_BLOCK.search(text or "")
    if match is None:
        return None, "", True
    try:
        data = json.loads(match.group(0))
        p_plus = float(data["p_plus"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None, "", True
    rationale = str(data.get("rationale", "")) if isinstance(data, dict) else ""
    return _clamp01(p_plus), rationale, False


class OpenAIStanceAppraiser:
    """Appraise utterances via an OpenAI chat model (or any injected client).

    The raw appraiser output is what gets cached; the calibration temperature is
    applied *after* the (possibly cached) raw output, so re-running with a different
    ``tau`` never triggers a new API call.
    """

    def __init__(
        self,
        model: str,
        client: LLMClient,
        cache: Optional[JSONCache] = None,
        calibration: Optional[Calibration] = None,
        chat_params: Optional[dict] = None,
    ) -> None:
        self.model = model
        self.client = client
        self.cache = cache
        self.calibration = calibration
        # Sampling params from the centralized policy (deterministic appraisal).
        self.chat_params = (
            dict(chat_params) if chat_params is not None
            else params.role_chat_params("appraiser")
        )

    def build_messages(self, *, utterance: str, a_plus: str, a_minus: str) -> "list[dict]":
        """Construct the chat messages for one appraisal."""
        system = (
            "You are an impartial stance classifier. You rate how strongly a statement "
            "supports one policy stance over another, judging the evidence in the text "
            "rather than your own opinion."
        )
        user = (
            f"Stance A+ : {a_plus}\n"
            f"Stance A- : {a_minus}\n\n"
            f"Statement: \"{utterance}\"\n\n"
            "Estimate the probability that the statement supports A+ rather than A-. "
            "Respond with ONLY a JSON object of the form "
            '{"p_plus": <number between 0 and 1>, "rationale": "<one short sentence>"}.'
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def _raw_appraisal(self, *, utterance: str, a_plus: str, a_minus: str) -> dict:
        """Return the (possibly cached) raw appraisal dict."""
        messages = self.build_messages(utterance=utterance, a_plus=a_plus, a_minus=a_minus)
        key = None
        if self.cache is not None:
            key = self.cache.make_key(self.model, "appraiser", {"messages": messages})
            hit = self.cache.get(key)
            if hit is not None:
                return hit

        result = self.client.chat(self.model, messages, **self.chat_params)
        p_plus_raw, rationale, parse_error = parse_appraiser_response(result.text)
        record = {
            "p_plus_raw": 0.5 if p_plus_raw is None else p_plus_raw,
            "rationale": rationale,
            "parse_error": parse_error,
            "usage": result.usage,
        }
        if self.cache is not None and key is not None:
            self.cache.set(key, record)
        return record

    def appraise(self, *, utterance: str, a_plus: str, a_minus: str) -> AppraisalResult:
        raw = self._raw_appraisal(utterance=utterance, a_plus=a_plus, a_minus=a_minus)
        p_plus_raw = float(raw["p_plus_raw"])
        if self.calibration is not None:
            p_plus_calibrated = self.calibration.calibrate(p_plus_raw=p_plus_raw)
        else:
            p_plus_calibrated = p_plus_raw
        return AppraisalResult(
            p_plus_raw=p_plus_raw,
            p_plus_calibrated=p_plus_calibrated,
            rationale=str(raw["rationale"]),
            parse_error=bool(raw["parse_error"]),
            usage=raw.get("usage"),
        )
