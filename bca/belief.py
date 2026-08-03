"""Beta belief and its Friedkin--Johnsen reading.

A concept node of an agent carries a Beta belief over a binary proposition
``A+`` versus ``A-``. The credence is the Beta mean ``b = alpha / (alpha + beta)``
(see "BCA redesign", Architecture section 2.1 and the Proof Setup, section 6).

Update rule
-----------
On hearing one utterance the appraiser returns stance likelihoods
``(evidence_pos, evidence_neg) = (l+, l-)``. We summarise them by the normalised
evidence ``e = l+ / (l+ + l-) in [0, 1]`` and apply one conjugate Beta step with a
fixed evidence weight ``w`` (default ``w = 1``):

    alpha <- alpha + w * e
    beta  <- beta  + w * (1 - e)

Friedkin--Johnsen correspondence
--------------------------------
Writing ``n = alpha + beta`` for the total pseudo-count before the step, the
credence obeys the *exact* convex blend (Prop. 1 of the redesign):

    b(t+1) = (1 - eta(t)) * b(t) + eta(t) * e,    eta(t) = w / (n + w).

This is an FJ-style update: the next opinion is a convex combination of the
retained opinion and the incoming social signal ``e``, with susceptibility
``eta``. Because each step adds exactly ``w`` pseudo-counts, ``n = kappa + w*c``
(Lemma 1), where ``c`` is the number of observations already received (in
a single concept this is the per-agent event count, not a global round index),
so

    eta(c) = w / (kappa + w (c + 1)),

strictly decreasing in both the prior strength ``kappa`` and the count ``c``. The
prior strength ``kappa`` is the agent's *stubbornness*; it is **not** itself an FJ
susceptibility -- it *induces* the time-varying susceptibility curve ``eta(c)``.

Forgetting factor ``gamma`` (non-stationary latent)
---------------------------------------------------
The static-latent update above (``gamma = 1``, the default) is exact conjugate Bayes
for an i.i.d. world: the count ``n = alpha + beta`` grows without bound, so the
susceptibility ``eta`` decays to 0 and the prior's weight washes out -- a population
*always* converges and FJ "persistent disagreement" holds only transiently.

For a *non-stationary* latent we discount old evidence by a forgetting factor
``gamma in (0, 1]`` while keeping the fixed prior ``(alpha0, beta0)``:

    alpha <- alpha0 + gamma (alpha - alpha0) + w e
    beta  <- beta0  + gamma (beta  - beta0)  + w (1 - e).

Then the count *saturates* at ``n* = kappa + w / (1 - gamma)``, so ``eta`` tends to a
positive constant, and the prior keeps a *permanent* weight
``rho* = kappa (1 - gamma) / (kappa (1 - gamma) + w)``. The steady-state credence is
``b* = rho* b0 + (1 - rho*) ebar`` -- exactly classical Friedkin--Johnsen with
constant susceptibility. ``gamma = 1`` gives ``rho* = 0`` and reproduces the
static-latent model byte-for-byte. The two knobs are thus ``kappa`` (prior strength /
stubbornness) and ``gamma`` (world-persistence / how slowly evidence is forgotten).
``eta()`` (= ``w / (n + w)``) stays exact for any ``gamma``.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BetaBelief:
    """A Beta(alpha, beta) credence over a binary concept with an FJ reading.

    Attributes:
        alpha: Pseudo-count mass on the ``A+`` (affirmative) stance. Grows with
            evidence; must stay strictly positive.
        beta: Pseudo-count mass on the ``A-`` stance. Must stay strictly positive.
        kappa: Prior strength (stubbornness) ``kappa = alpha0 + beta0`` fixed at
            initialization. High ``kappa`` is stubborn, low ``kappa`` is pliable.
            Stored separately from ``alpha + beta`` (which grows over time) so the
            analytic susceptibility ``eta(t)`` can be evaluated against time.
        weight: Evidence weight ``w in (0, 1]`` of a single observation. One
            utterance's worth of pseudo-count. Fixed constant (default ``w = 1``).
        gamma: Forgetting factor ``in (0, 1]`` (world-persistence). ``1.0`` (default)
            is the static-latent conjugate update (count grows unbounded, prior washes
            out); ``< 1`` discounts old evidence so the count saturates and the prior
            keeps a permanent weight -- classical FJ persistent disagreement.
    """

    alpha: float
    beta: float
    kappa: float
    weight: float = 1.0
    gamma: float = 1.0

    def __post_init__(self) -> None:
        if self.alpha <= 0.0 or self.beta <= 0.0:
            raise ValueError(
                f"alpha and beta must be positive, got alpha={self.alpha}, "
                f"beta={self.beta}"
            )
        if self.kappa <= 0.0:
            raise ValueError(f"kappa (prior strength) must be positive, got {self.kappa}")
        if self.weight <= 0.0:
            raise ValueError(f"weight must be positive, got {self.weight}")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError(f"gamma (forgetting factor) must lie in (0, 1], got {self.gamma}")
        # Capture the fixed prior counts so gamma<1 can discount only the *evidence*
        # accumulated since construction, never the prior.
        self._alpha0 = self.alpha
        self._beta0 = self.beta

    @classmethod
    def from_mean(
        cls, mean: float, kappa: float, weight: float = 1.0, gamma: float = 1.0
    ) -> "BetaBelief":
        """Build a belief from a starting credence ``b0`` and prior strength ``kappa``.

        Implements section 6 of the redesign: ``alpha0 = kappa * b0`` and
        ``beta0 = kappa * (1 - b0)``, so that ``alpha0 + beta0 = kappa`` and the
        initial mean is exactly ``b0``.

        Args:
            mean: Starting credence ``b0`` in the open interval ``(0, 1)``.
            kappa: Prior strength (stubbornness), strictly positive.
            weight: Evidence weight per observation (default ``1.0``).
            gamma: Forgetting factor in ``(0, 1]`` (default ``1.0`` = static latent).
        """
        if not 0.0 < mean < 1.0:
            raise ValueError(f"mean (starting credence b0) must lie in (0, 1), got {mean}")
        return cls(
            alpha=kappa * mean, beta=kappa * (1.0 - mean), kappa=kappa,
            weight=weight, gamma=gamma,
        )

    @property
    def mean(self) -> float:
        """The credence ``b = alpha / (alpha + beta)``: the single opinion value."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def strength(self) -> float:
        """Current total pseudo-count ``n = alpha + beta``.

        At ``gamma = 1`` this equals ``kappa + w*t`` after ``t`` unit-weight updates;
        at ``gamma < 1`` it saturates at ``kappa + w / (1 - gamma)``.
        """
        return self.alpha + self.beta

    def eta(self, t: int | None = None) -> float:
        """Instantaneous FJ susceptibility ``eta = w / (n + w)``.

        With no argument this uses the *current* state ``n = alpha + beta`` and is
        exact for any ``gamma`` -- the susceptibility that governs the next update.
        Passing an explicit round index ``t`` returns the ``gamma = 1`` analytic value
        ``w / (kappa + w (t + 1))``; at ``gamma < 1`` pass no argument (the closed form
        in ``t`` does not hold once evidence is discounted).

        Args:
            t: Optional round index. If ``None``, evaluate from the current
                Beta parameters (correct for any ``gamma``); otherwise use the
                ``gamma = 1`` closed form in ``kappa`` and ``t``.
        """
        n = (self.alpha + self.beta) if t is None else (self.kappa + self.weight * t)
        return self.weight / (n + self.weight)

    def update(self, evidence_pos: float, evidence_neg: float) -> float:
        """Apply one (possibly discounted) conjugate Beta step; return the new credence.

        The likelihoods are normalised to ``e = evidence_pos / (evidence_pos +
        evidence_neg)``. With the forgetting factor ``gamma`` the *evidence* accumulated
        since construction is discounted while the fixed prior is kept:

            alpha <- alpha0 + gamma (alpha - alpha0) + w e
            beta  <- beta0  + gamma (beta  - beta0)  + w (1 - e).

        At ``gamma = 1`` this is exactly ``alpha += w*e``, ``beta += w*(1-e)`` -- the FJ
        convex blend ``b <- (1 - eta) b + eta e`` with ``eta`` evaluated before the step.

        Args:
            evidence_pos: Likelihood ``l+ >= 0`` for the ``A+`` stance.
            evidence_neg: Likelihood ``l- >= 0`` for the ``A-`` stance.

        Returns:
            The updated credence ``mean`` after incorporating the evidence.

        Raises:
            ValueError: If ``evidence_pos`` or ``evidence_neg`` is negative, or if
                their sum is not strictly positive (no informative evidence).
        """
        if evidence_pos < 0.0 or evidence_neg < 0.0:
            raise ValueError(
                f"likelihoods must be non-negative, got l+={evidence_pos}, l-={evidence_neg}"
            )
        total = evidence_pos + evidence_neg
        if total <= 0.0:
            raise ValueError("evidence_pos + evidence_neg must be strictly positive")

        e = evidence_pos / total
        self.alpha = self._alpha0 + self.gamma * (self.alpha - self._alpha0) + self.weight * e
        self.beta = self._beta0 + self.gamma * (self.beta - self._beta0) + self.weight * (1.0 - e)
        return self.mean
