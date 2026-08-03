"""One place that wires a :class:`~bca.models.ModelSpec` into the language channel.

Every experiment builds its generator / appraiser / slider-probe through
:func:`build_channel`, so there is no per-model, per-experiment copy-paste: switching the
model key swaps the endpoint and model id while the sampling policy (``bca.params``)
stays identical. :func:`provenance_block` records exactly what was used, for run metrics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from bca import params
from bca.appraisers import OpenAIStanceAppraiser
from bca.calibration import Calibration
from bca.generators import OpenAIUtteranceGenerator
from bca.llm import JSONCache, OpenAIClient
from bca.models import ModelSpec
from bca.probes import OpenAISliderProbe


@dataclass
class Channel:
    """The LLM-backed language channel for one model, plus its shared client."""

    client: OpenAIClient
    generator: OpenAIUtteranceGenerator
    appraiser: OpenAIStanceAppraiser
    slider_probe: OpenAISliderProbe


def build_channel(
    spec: ModelSpec,
    *,
    cache: Optional[JSONCache] = None,
    calibration: Optional[Calibration] = None,
    max_calls: Optional[int] = None,
    appraiser_spec: Optional[ModelSpec] = None,
) -> Channel:
    """Build the generator/appraiser/probe for ``spec`` on a shared client.

    All three roles are driven with the centralized sampling params for ``spec`` (so the
    only difference between models is the endpoint + model id). ``appraiser_spec`` may be
    given to appraise on a different model; by default the appraiser uses ``spec``.
    """
    appraiser_spec = appraiser_spec or spec
    client = OpenAIClient.from_spec(spec, max_calls=max_calls)
    generator = OpenAIUtteranceGenerator(
        model=spec.model_id, client=client, cache=cache,
        chat_params=params.role_chat_params("generator", spec),
    )
    appraiser = OpenAIStanceAppraiser(
        model=appraiser_spec.model_id, client=client, cache=cache, calibration=calibration,
        chat_params=params.role_chat_params("appraiser", appraiser_spec),
    )
    slider_probe = OpenAISliderProbe(
        model=spec.model_id, client=client, cache=cache,
        chat_params=params.role_chat_params("probe", spec),
    )
    return Channel(client=client, generator=generator, appraiser=appraiser,
                   slider_probe=slider_probe)


def provenance_block(
    spec: ModelSpec,
    *,
    appraiser_spec: Optional[ModelSpec] = None,
    gamma: Optional[float] = None,
    calibration_path: Optional[Any] = None,
) -> "dict[str, Any]":
    """A JSON-serializable record of the model, endpoint, and exact sampling params."""
    appraiser_spec = appraiser_spec or spec
    block: "dict[str, Any]" = {
        "model_key": spec.key,
        "generator_model_id": spec.model_id,
        "appraiser_model_id": appraiser_spec.model_id,
        "endpoint": spec.endpoint_label,
        "api_key_env": spec.api_key_env,
        "sampling": params.sampling_provenance(spec),
    }
    if gamma is not None:
        block["gamma"] = gamma
    if calibration_path is not None:
        block["calibration_json"] = str(calibration_path)
    return block
