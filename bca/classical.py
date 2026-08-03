"""Closed-form classical opinion-dynamics references (pure NumPy, no API).

Overlaid on the LLM runs as yardsticks; nothing here is simulated.
For the committed-minority curve see the BCA redesign Prop. 3. The engine updates
per event (N-1 per round); per the redesign footnote, N-1 unit-weight steps equal one
aggregate-weight round step, so `eta_per_round` here is the per-round effective
susceptibility (aggregate weight), not the per-event value.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def eta_schedule(kappa: float, weight: float, n_steps: int) -> np.ndarray:
    """Induced susceptibility η[t] = w / (κ + w(t+1)) for t = 0..n_steps-1."""
    t = np.arange(n_steps)
    return weight / (kappa + weight * (t + 1))


def degroot_consensus(initial: np.ndarray, W: Optional[np.ndarray] = None) -> float:
    """Consensus value. On a complete graph with uniform weights this is the mean."""
    initial = np.asarray(initial, dtype=float)
    if W is None:
        return float(initial.mean())
    # Left Perron eigenvector (stationary distribution) weighted average.
    vals, vecs = np.linalg.eig(np.asarray(W, dtype=float).T)
    idx = int(np.argmin(np.abs(vals - 1.0)))
    w = np.real(vecs[:, idx])
    w = w / w.sum()
    return float(w @ initial)


def degroot_trajectory(initial: np.ndarray, W: np.ndarray, n_steps: int) -> np.ndarray:
    """Iterate x(t+1) = W x(t); return shape (n_steps+1, n)."""
    x = np.asarray(initial, dtype=float)
    W = np.asarray(W, dtype=float)
    traj = [x.copy()]
    for _ in range(n_steps):
        x = W @ x
        traj.append(x.copy())
    return np.array(traj)


def fj_fixed_point(initial: np.ndarray, susceptibility: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Friedkin--Johnsen fixed point x* = (I - ΛW)^{-1} (I - Λ) x0, Λ = diag(susceptibility)."""
    initial = np.asarray(initial, dtype=float)
    Lam = np.diag(np.asarray(susceptibility, dtype=float))
    W = np.asarray(W, dtype=float)
    n = initial.size
    I = np.eye(n)
    return np.linalg.solve(I - Lam @ W, (I - Lam) @ initial)


def committed_minority_curve(f_grid: np.ndarray, eta_per_round: np.ndarray) -> np.ndarray:
    """Prop. 3 free-agent credence after the horizon: b_F = 1 - ∏_t (1 - η[t] f).

    Assumes committed agents at 1 and free agents starting at 0.
    """
    f_grid = np.asarray(f_grid, dtype=float)
    eta = np.asarray(eta_per_round, dtype=float)
    # outer product → (n_f, n_t); product over t.
    factors = 1.0 - np.outer(f_grid, eta)
    return 1.0 - np.prod(factors, axis=1)
