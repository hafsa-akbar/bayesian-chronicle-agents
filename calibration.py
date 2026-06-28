"""Temperature calibration for the stance appraiser.

A stance appraiser maps an utterance to a probability ``p_plus`` that it supports
``A+``. RLHF-trained chat models are systematically over/under-confident, so we fit a
single temperature ``tau`` on held-out soft-labelled data and apply it at inference.

Two calibration modes (both reduce to ``p_calibrated = sigmoid(z / tau)`` for a raw
logit ``z``):

* ``logit_diff`` (preferred when token logprobs for ``A+``/``A-`` are available):
  ``z = logit_plus - logit_minus``.
* ``raw_prob`` (fallback when the appraiser only returns a probability):
  ``z = logit(p_plus_raw)``.

``tau`` is fitted by minimising the soft-label binary cross-entropy
``-mean(y log p + (1-y) log(1-p))`` on the calibration split.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
from scipy import optimize, stats

ArrayLike = Union[float, np.ndarray]


def sigmoid(x: ArrayLike) -> ArrayLike:
    """Numerically stable logistic sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def logit(p: ArrayLike, eps: float = 1e-6) -> ArrayLike:
    """Inverse sigmoid with clamping to ``[eps, 1-eps]`` to avoid infinities."""
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def calibrated_prob(z: ArrayLike, tau: float) -> ArrayLike:
    """Temperature-scaled probability ``sigmoid(z / tau)``."""
    if tau <= 0.0:
        raise ValueError(f"tau must be positive, got {tau}")
    return sigmoid(np.asarray(z, dtype=float) / tau)


def bce_loss(y: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> float:
    """Soft-label binary cross-entropy ``-mean(y log p + (1-y) log(1-p))``."""
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def fit_temperature(
    z: np.ndarray,
    y: np.ndarray,
    bounds: "tuple[float, float]" = (1e-3, 1e3),
    eps: float = 1e-12,
) -> float:
    """Fit the temperature ``tau`` that minimises soft-label BCE.

    Args:
        z: Raw logits (``logit_plus - logit_minus`` or ``logit(p_plus_raw)``).
        y: Target soft labels in ``[0, 1]``.
        bounds: Search interval for ``tau`` (bounded scalar minimisation).
        eps: Clamp for the cross-entropy.

    Returns:
        The fitted positive temperature ``tau``.
    """
    z = np.asarray(z, dtype=float)
    y = np.asarray(y, dtype=float)
    if z.shape != y.shape:
        raise ValueError(f"z and y must share shape, got {z.shape} and {y.shape}")
    if z.size == 0:
        raise ValueError("need at least one example to fit a temperature")

    result = optimize.minimize_scalar(
        lambda tau: bce_loss(y, sigmoid(z / tau), eps),
        bounds=bounds,
        method="bounded",
    )
    return float(result.x)


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error: |mean(p) - mean(y)| per equal-width bin of ``p``.

    Generalised to soft labels: each bin compares the mean prediction to the mean
    soft label, weighted by bin occupancy.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # bin index in [0, n_bins-1]; values exactly 1.0 fall in the last bin.
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    n = p.size
    for b in range(n_bins):
        mask = idx == b
        count = int(mask.sum())
        if count:
            ece += (count / n) * abs(p[mask].mean() - y[mask].mean())
    return float(ece)


def _safe_corr(fn, y: np.ndarray, p: np.ndarray) -> float:
    """Correlation that returns nan rather than raising on constant input."""
    if y.size < 2 or np.std(p) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    return float(fn(y, p).statistic)


def calibration_metrics(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> dict:
    """Compute the full set of soft-label calibration metrics.

    Returns a dict with ``mae``, ``rmse``, ``brier``, ``nll``, ``ece``, ``pearson``,
    ``spearman``.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    err = p - y
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "brier": float(np.mean(err ** 2)),
        "nll": bce_loss(y, p),
        "ece": expected_calibration_error(y, p, n_bins),
        "pearson": _safe_corr(stats.pearsonr, y, p),
        "spearman": _safe_corr(stats.spearmanr, y, p),
    }


@dataclass
class Calibration:
    """A fitted appraiser calibration: a mode and a temperature ``tau``.

    Attributes:
        mode: ``"raw_prob"`` (calibrate from a probability) or ``"logit_diff"``
            (calibrate from two stance logits).
        tau: The fitted temperature (``tau = 1`` is the identity in logit space).
        eps: Clamp used by :func:`logit` when converting a probability.
    """

    mode: str
    tau: float
    eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.mode not in ("raw_prob", "logit_diff"):
            raise ValueError(f"unknown calibration mode {self.mode!r}")
        if self.tau <= 0.0:
            raise ValueError(f"tau must be positive, got {self.tau}")

    def raw_logit(
        self,
        *,
        p_plus_raw: Optional[float] = None,
        logit_plus: Optional[float] = None,
        logit_minus: Optional[float] = None,
    ) -> float:
        """Compute the raw logit ``z`` for this mode from appraiser outputs."""
        if self.mode == "raw_prob":
            if p_plus_raw is None:
                raise ValueError("raw_prob mode requires p_plus_raw")
            return float(logit(p_plus_raw, self.eps))
        if logit_plus is None or logit_minus is None:
            raise ValueError("logit_diff mode requires logit_plus and logit_minus")
        return float(logit_plus - logit_minus)

    def calibrate(
        self,
        *,
        p_plus_raw: Optional[float] = None,
        logit_plus: Optional[float] = None,
        logit_minus: Optional[float] = None,
    ) -> float:
        """Apply the temperature to produce the calibrated probability ``p_plus``."""
        z = self.raw_logit(
            p_plus_raw=p_plus_raw, logit_plus=logit_plus, logit_minus=logit_minus
        )
        return float(calibrated_prob(z, self.tau))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Calibration":
        return cls(mode=data["mode"], tau=float(data["tau"]), eps=float(data.get("eps", 1e-6)))

    def save(self, path: "str | Path") -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: "str | Path") -> "Calibration":
        return cls.from_dict(json.loads(Path(path).read_text()))
