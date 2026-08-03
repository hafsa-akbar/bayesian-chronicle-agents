"""bca — a structured, controllable belief-update layer.

A Beta belief-update rule that is *exactly* a Friedkin--Johnsen (FJ) convex-blend
step with a time-varying susceptibility, plus a calibrated appraiser, language-model
generation, and classical baselines. See ``README.md`` for the model definition and
the Beta-update derivation.
"""

__all__ = ["belief", "agent", "analysis"]
