"""Model registry: one source of truth for the LLMs the experiments can run on.

Each entry maps a short, stable KEY (used on the command line and for namespacing
outputs and calibration files) to the concrete ``model_id``, the OpenAI-compatible
endpoint, and the environment variable holding that endpoint's API key. Adding a new
model is a single entry in :data:`REGISTRY`; nothing else needs to change.

The sampling parameters are deliberately *not* stored here -- they live in
:mod:`bca_beta.params` so that every model is driven with identical settings. The only
per-model call knobs a registry entry may carry are ``extra_body`` (params an endpoint
requires, e.g. an explicit ``max_tokens`` for the Hugging Face router) and
``unsupported_params`` (params an endpoint rejects, dropped with a warning).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

_PKG = Path(__file__).resolve().parent
CALIBRATION_DIR = _PKG / "experiments" / "calibration"
OUTPUTS_DIR = _PKG / "experiments" / "outputs"


@dataclass(frozen=True)
class ModelSpec:
    """A single model + endpoint the experiments can be pointed at.

    Attributes:
        key: short stable identifier (CLI ``--model-key``; also namespaces outputs and
            the per-model calibration file).
        model_id: the provider's model name sent to the chat API.
        api_key_env: environment variable holding the API key for this endpoint.
        base_url: OpenAI-compatible endpoint; ``None`` means the default OpenAI base URL.
        extra_body: params this endpoint requires, forwarded verbatim into every chat
            request (e.g. ``{"max_tokens": 512}`` for the HF router). Applied identically
            to every role, so it never changes the sampling policy.
        unsupported_params: sampling params this endpoint rejects; dropped with a warning
            rather than silently (see :func:`bca_beta.params.role_chat_params`).
    """

    key: str
    model_id: str
    api_key_env: str = "OPENAI_API_KEY"
    base_url: Optional[str] = None
    extra_body: Mapping[str, Any] = field(default_factory=dict)
    unsupported_params: "tuple[str, ...]" = ()

    @property
    def endpoint_label(self) -> str:
        """A short human-readable endpoint name for logs and provenance."""
        return self.base_url or "openai-default"


REGISTRY: "dict[str, ModelSpec]" = {
    # OpenAI models use the defaults: OPENAI_API_KEY, base_url=None.
    "gpt-5.4-mini": ModelSpec(key="gpt-5.4-mini", model_id="gpt-5.4-mini"),
    "gpt-5.4": ModelSpec(key="gpt-5.4", model_id="gpt-5.4"),
    "llama4-scout": ModelSpec(
        key="llama4-scout",
        # The chat-usable variant is the "-Instruct" model, not the base checkpoint.
        model_id="meta-llama/Llama-4-Scout-17B-16E-Instruct",
        api_key_env="HF_API_KEY",
        base_url="https://router.huggingface.co/v1",
        # The HF router wants an explicit token cap; applied identically to every role.
        extra_body={"max_tokens": 512},
    ),
}


def available_keys() -> "list[str]":
    """Sorted list of registered model keys (for error messages and help text)."""
    return sorted(REGISTRY)


def get_model_spec(key: str) -> ModelSpec:
    """Return the registered :class:`ModelSpec` for ``key`` or raise a clear error."""
    try:
        return REGISTRY[key]
    except KeyError:
        raise KeyError(
            f"unknown model key {key!r}; available: {', '.join(available_keys())}"
        ) from None


def spec_for_model_id(model_id: str) -> ModelSpec:
    """Wrap a raw ``model_id`` (legacy ``--model``) as a default-OpenAI-endpoint spec.

    Its ``key`` is the ``model_id`` itself, so outputs and calibration still namespace
    cleanly for models that are not in the registry.
    """
    return ModelSpec(key=model_id, model_id=model_id)


def resolve_spec(model_key: Optional[str], model: Optional[str] = None) -> ModelSpec:
    """Resolve the model to run: a registry ``model_key`` wins, else a raw ``model`` id.

    ``--model-key`` selects a registered endpoint; if it is absent we fall back to the
    legacy raw ``--model`` id on the default OpenAI endpoint.
    """
    if model_key:
        return get_model_spec(model_key)
    if model:
        return spec_for_model_id(model)
    raise ValueError("resolve_spec requires either model_key or model")


def default_calibration_path(spec: ModelSpec) -> Path:
    """Per-model appraiser calibration file, e.g. ``.../calibration/llama4-scout_tau.json``."""
    return CALIBRATION_DIR / f"{spec.key}_tau.json"


def default_output_root(spec: ModelSpec) -> Path:
    """Per-model output root, e.g. ``.../outputs/llama4-scout``."""
    return OUTPUTS_DIR / spec.key
