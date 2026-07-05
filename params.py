"""Centralized sampling policy -- the single source of truth for chat parameters.

Every model is driven with identical sampling settings, so a difference in results
reflects the model, not the knobs: the per-role temperatures live here, once, and every
generator / appraiser / probe reads them from here. The only per-model variation is what
an endpoint requires or rejects, carried on the :class:`~bca_beta.models.ModelSpec`
(``extra_body`` / ``unsupported_params``) and merged in by :func:`role_chat_params`; a
dropped param is warned about, never silent.
"""
from __future__ import annotations

import sys
import warnings
from typing import Any, Optional

from bca_beta.models import ModelSpec

# The ONLY sampling knob, per role. Everything else is left at provider defaults
# (top_p = 1, no max_tokens, no seed, no penalties) -- the "no other sampling params"
# policy, applied identically to every model.
ROLE_TEMPERATURE: "dict[str, float]" = {
    "generator": 0.9,   # belief -> utterance: some diversity of phrasing
    "appraiser": 0.0,   # utterance -> stance probability: deterministic
    "probe": 0.0,       # slider self-report: deterministic
}

ROLES = tuple(ROLE_TEMPERATURE)


def _warn_dropped(spec: ModelSpec, role: str, param: str, value: Any) -> None:
    msg = (
        f"model {spec.key!r} ({spec.endpoint_label}) rejects sampling param "
        f"{param!r}; dropping it for role {role!r} (requested {param}={value!r})"
    )
    warnings.warn(msg, UserWarning, stacklevel=2)
    print(f"WARNING: {msg}", file=sys.stderr)


def role_chat_params(role: str, spec: Optional[ModelSpec] = None) -> "dict[str, Any]":
    """The exact kwargs to send to ``client.chat`` for ``role`` on ``spec``.

    Identical for every model except for what the endpoint requires (``extra_body``) or
    rejects (``unsupported_params``). A rejected param is set to ``None`` -- the client
    omits ``None`` sampling params -- and a warning is emitted.
    """
    if role not in ROLE_TEMPERATURE:
        raise KeyError(f"unknown role {role!r}; expected one of {ROLES}")

    params: "dict[str, Any]" = {"temperature": ROLE_TEMPERATURE[role]}

    if spec is not None:
        for name in ("temperature",):
            if name in spec.unsupported_params:
                _warn_dropped(spec, role, name, params[name])
                params[name] = None  # client omits None-valued sampling params
        if spec.extra_body:
            params["extra_body"] = dict(spec.extra_body)

    return params


def sampling_provenance(spec: Optional[ModelSpec] = None) -> "dict[str, Any]":
    """A record of the exact sampling params used per role, for run metrics.json."""
    return {role: role_chat_params(role, spec) for role in ROLES}
