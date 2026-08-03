"""Agent holding one Beta belief per concept node.

The agent is a thin wrapper around its beliefs: it receives appraised evidence
and forwards it to the relevant ``BetaBelief``. The ``generator`` and ``appraiser``
slots are optional; when set they supply a language-model utterance generator and a
calibrated stance appraiser without changing the belief-update core.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Hashable, Optional

from bca.belief import BetaBelief

DEFAULT_CONCEPT = "transit_priority"


@dataclass
class Agent:
    """A generative agent whose opinions are explicit Beta credences.

    Attributes:
        agent_id: Identifier of the agent (used as the network node key).
        beliefs: Mapping ``concept_key -> BetaBelief`` (typically a single concept).
        generator: Optional language-model utterance generator. Given the agent's
            credence it produces the natural-language utterance the agent broadcasts.
        appraiser: Optional calibrated stance appraiser. It maps a heard utterance
            to evidence likelihoods ``(l+, l-)``.
        committed: if True the agent expresses but never updates (≡ κ → ∞).
    """

    agent_id: Hashable
    beliefs: dict[str, BetaBelief] = field(default_factory=dict)
    generator: Optional[Any] = None
    appraiser: Optional[Any] = None
    committed: bool = False

    @classmethod
    def single_concept(
        cls,
        agent_id: Hashable,
        mean: float,
        kappa: float,
        weight: float = 1.0,
        concept: str = DEFAULT_CONCEPT,
        committed: bool = False,
        gamma: float = 1.0,
    ) -> "Agent":
        """Build a single-concept agent.

        Args:
            agent_id: Identifier of the agent.
            mean: Starting credence ``b0`` in ``(0, 1)``.
            kappa: Prior strength (stubbornness) for the concept.
            weight: Evidence weight per observation (default ``1.0``).
            concept: Concept-node key (default ``"transit_priority"``).
            committed: if True the agent expresses but never updates (default ``False``).
            gamma: Forgetting factor in ``(0, 1]`` (default ``1.0`` = static latent).
        """
        belief = BetaBelief.from_mean(mean=mean, kappa=kappa, weight=weight, gamma=gamma)
        return cls(agent_id=agent_id, beliefs={concept: belief}, committed=committed)

    def _resolve_concept(self, concept: Optional[str]) -> str:
        """Return an explicit concept key, defaulting to the sole concept if unique."""
        if concept is not None:
            if concept not in self.beliefs:
                raise KeyError(f"agent {self.agent_id!r} has no concept {concept!r}")
            return concept
        if len(self.beliefs) != 1:
            raise ValueError(
                f"agent {self.agent_id!r} holds {len(self.beliefs)} concepts; "
                "pass an explicit `concept`"
            )
        return next(iter(self.beliefs))

    def belief_for(self, concept: Optional[str] = None) -> BetaBelief:
        """Return the ``BetaBelief`` for ``concept`` (the sole concept by default)."""
        return self.beliefs[self._resolve_concept(concept)]

    def credence(self, concept: Optional[str] = None) -> float:
        """Return the current credence ``b`` for ``concept`` (sole concept by default)."""
        return self.belief_for(concept).mean

    def apply_evidence(
        self,
        evidence_pos: float,
        evidence_neg: float,
        concept: Optional[str] = None,
    ) -> float:
        """Forward appraised evidence ``(l+, l-)`` to the relevant belief.

        Args:
            evidence_pos: Likelihood ``l+`` for the ``A+`` stance.
            evidence_neg: Likelihood ``l-`` for the ``A-`` stance.
            concept: Concept to update; defaults to the sole concept when the
                agent holds exactly one.

        Returns:
            The agent's updated credence for that concept.
        """
        return self.belief_for(concept).update(evidence_pos, evidence_neg)
