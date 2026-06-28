"""Analysis for the language-channel runs: recovery, alignment, and regime metrics.

The prescribed prior strength ``kappa`` (stubbornness) is *not* an FJ susceptibility;
it *induces* a time-varying one,

    eta_i(c) = w / (kappa_i + n_c + w),   n_c = w (1 - gamma^c) / (1 - gamma),

with ``n_c`` the discounted observation count for the concept. The recovery routines
invert this *live* susceptibility (reading ``gamma`` from an optional per-event column,
default ``1.0``), so they recover ``kappa`` at any ``gamma``: at ``gamma = 1`` the count
grows unboundedly and ``eta`` decays as ``~1/c``; at ``gamma < 1`` it saturates and
``eta`` tends to a positive constant (classical FJ).

A single agent's update is the FJ convex blend ``b <- (1 - eta) b + eta e`` with ``e``
the evidence; see ``README.md`` for the model definition and the Beta-update derivation.
This module also computes appraiser alignment, trajectory deviation, the DeGroot / FJ /
committed-minority regime metrics, and belief--slider auditability.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# --- Susceptibility inversion (gamma-aware) ---


def _discounted_evidence_count(c: np.ndarray, weight: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """Accumulated (forgetting-discounted) evidence pseudo-count after ``c`` events.

    ``n_c - kappa = w (1 - gamma^c) / (1 - gamma)``, which tends to ``w * c`` as
    ``gamma -> 1`` (the static-latent count). Evaluated elementwise; the ``gamma = 1``
    rows use the ``w * c`` limit directly to avoid a 0/0.
    """
    c = np.asarray(c, dtype=float)
    weight = np.asarray(weight, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    static = weight * c
    one_minus_g = 1.0 - gamma
    with np.errstate(divide="ignore", invalid="ignore"):
        discounted = weight * (1.0 - gamma ** c) / one_minus_g
    return np.where(gamma >= 1.0, static, discounted)


def _kappa_from_eta(eta: np.ndarray, c: np.ndarray, weight: np.ndarray,
                    gamma: np.ndarray) -> np.ndarray:
    """Invert the live susceptibility to the prior strength, gamma-aware.

    ``eta = w / (n + w)`` so ``n = w/eta - w``; the prior strength is
    ``kappa = n - (n_c - kappa)`` with the discounted count above. At ``gamma = 1`` this
    reduces to ``kappa = w (1/eta - (c + 1))``.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        n = weight / eta - weight
    kappa = n - _discounted_evidence_count(c, weight, gamma)
    return np.where(np.isfinite(eta) & (eta > 0.0), kappa, np.nan)


# --- Language-channel recovery and alignment ---
def recover_eta_from(
    df: pd.DataFrame,
    evidence_col: str,
    before_col: str = "listener_belief_before",
    after_col: str = "listener_belief_after",
    eps: float = 1e-9,
) -> pd.Series:
    """Recover per-event ``eta`` using an arbitrary evidence column.

    ``eta = (b_after - b_before) / (evidence - b_before)``. Two uses:
    ``evidence_col="evidence_used_for_update"`` gives the *mechanical* invariant
    (must equal the predicted eta); ``evidence_col="speaker_belief_snapshot"`` gives
    the *oracle-aligned* recovery (the scientific result -- it degrades as the
    appraised evidence drifts from the latent speaker belief).
    """
    denom = df[evidence_col].to_numpy() - df[before_col].to_numpy()
    delta = df[after_col].to_numpy() - df[before_col].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        eta = delta / denom
    eta = np.where(np.abs(denom) < eps, np.nan, eta)
    return pd.Series(eta, index=df.index, name="eta_recovered")


def recover_kappa_from(
    df: pd.DataFrame,
    evidence_col: str,
    before_col: str = "listener_belief_before",
    after_col: str = "listener_belief_after",
    c_col: str = "c_before",
    weight_col: str = "weight",
    gamma_col: str = "gamma",
    eps: float = 1e-9,
) -> pd.Series:
    """Recover ``kappa`` from an evidence column, gamma-aware.

    Inverts the live susceptibility with the discounted count; at ``gamma = 1`` (or when
    the ``gamma`` column is absent) this reduces to ``kappa = w (1/eta - (c+1))``.
    """
    eta = recover_eta_from(df, evidence_col, before_col, after_col, eps).to_numpy()
    weight = df[weight_col].to_numpy() if weight_col in df.columns else np.ones(len(df))
    gamma = df[gamma_col].to_numpy() if gamma_col in df.columns else np.ones(len(df))
    c = df[c_col].to_numpy()
    kappa = _kappa_from_eta(eta, c, weight, gamma)
    return pd.Series(kappa, index=df.index, name="kappa_recovered")


def appraiser_alignment_metrics(
    df: pd.DataFrame,
    appraised_col: str = "appraised_evidence_calibrated",
    latent_col: str = "speaker_belief_snapshot",
) -> dict:
    """How well the appraised evidence tracks the latent speaker belief."""
    a = df[appraised_col].to_numpy(dtype=float)
    latent = df[latent_col].to_numpy(dtype=float)
    err = a - latent
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "pearson": _safe_corr_pair(latent, a, "pearson"),
        "spearman": _safe_corr_pair(latent, a, "spearman"),
        "n": int(len(df)),
    }


def _safe_corr_pair(x: np.ndarray, y: np.ndarray, kind: str) -> float:
    from scipy import stats

    if x.size < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    fn = stats.pearsonr if kind == "pearson" else stats.spearmanr
    return float(fn(x, y).statistic)


def alignment_by_bin(
    df: pd.DataFrame,
    latent_col: str = "speaker_belief_snapshot",
    appraised_col: str = "appraised_evidence_calibrated",
    n_bins: int = 10,
) -> pd.DataFrame:
    """Bin the latent speaker belief into ``n_bins`` and report mean appraised evidence."""
    latent = df[latent_col].to_numpy(dtype=float)
    appraised = df[appraised_col].to_numpy(dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(latent, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        count = int(mask.sum())
        rows.append(
            {
                "bin": b,
                "bin_low": float(edges[b]),
                "bin_high": float(edges[b + 1]),
                "count": count,
                "latent_mean": float(latent[mask].mean()) if count else float("nan"),
                "appraised_mean": float(appraised[mask].mean()) if count else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def summarize_oracle_aligned_kappa(
    df: pd.DataFrame,
    kappa_recovered_col: str = "kappa_recovered_oracle_aligned",
    eta_col: str = "eta_recovered_oracle_aligned_using_speaker_belief",
    group_cols: "tuple[str, ...]" = ("model", "kappa_condition", "seed"),
    min_denominator: float = 0.0,
) -> pd.DataFrame:
    """Aggregate oracle-aligned recovered kappa, grouped by model/condition/seed.

    ``min_denominator`` skips ill-conditioned events where the latent speaker belief is
    within that distance of the listener's belief (the appraiser-noise regime where the
    eta inversion is unstable).
    """
    work = df.copy()
    skipped = ~np.isfinite(work[eta_col].to_numpy())
    if min_denominator > 0.0 and "speaker_belief_snapshot" in work.columns:
        den = (work["speaker_belief_snapshot"] - work["listener_belief_before"]).abs().to_numpy()
        skipped = skipped | (den <= min_denominator)
    work["_skipped"] = skipped
    work["_kappa_abs_err"] = (work[kappa_recovered_col] - work["kappa_condition"]).abs()
    keys = [c for c in group_cols if c in work.columns]
    group_by = keys[0] if len(keys) == 1 else keys
    rows = []
    for key_vals, g in work.groupby(group_by, dropna=False):
        if not isinstance(key_vals, tuple):
            key_vals = (key_vals,)
        rec = dict(zip(keys, key_vals))
        valid = g[~g["_skipped"]]
        rec["kappa_prescribed"] = rec.get("kappa_condition", float("nan"))
        rec["n_events"] = len(g)
        rec["n_skipped"] = int(g["_skipped"].sum())
        if len(valid):
            rec["kappa_recovered_mean"] = float(valid[kappa_recovered_col].mean())
            rec["kappa_recovered_median"] = float(valid[kappa_recovered_col].median())
            rec["kappa_recovery_mae"] = float(valid["_kappa_abs_err"].mean())
            rec["kappa_recovery_relative_error"] = float(
                (valid["_kappa_abs_err"] / valid["kappa_condition"]).mean()
            )
        else:
            rec["kappa_recovered_mean"] = float("nan")
            rec["kappa_recovered_median"] = float("nan")
            rec["kappa_recovery_mae"] = float("nan")
            rec["kappa_recovery_relative_error"] = float("nan")
        rows.append(rec)
    sort_key = "kappa_condition" if "kappa_condition" in keys else keys[0]
    return pd.DataFrame(rows).sort_values(sort_key).reset_index(drop=True)


def trajectory_deviation(
    llm_by_round: np.ndarray,
    oracle_by_round: np.ndarray,
) -> dict:
    """Per-round and final mean-absolute belief deviation: LLM run vs oracle baseline.

    Both inputs have shape ``(n_rounds + 1, n_agents)`` (start-of-round snapshots plus
    the final beliefs), recorded under an identical event schedule.
    """
    llm = np.asarray(llm_by_round, dtype=float)
    oracle = np.asarray(oracle_by_round, dtype=float)
    if llm.shape != oracle.shape:
        raise ValueError(f"shape mismatch: {llm.shape} vs {oracle.shape}")
    dev = np.abs(llm - oracle)
    by_round = dev.mean(axis=1)
    return {
        "mean_abs_dev_by_round": by_round.tolist(),
        "final_mean_abs_dev": float(by_round[-1]),
        "overall_mean_abs_dev": float(dev.mean()),
    }


def plot_appraised_vs_speaker_belief(df: pd.DataFrame, ax: Optional["object"] = None):
    """Scatter latent speaker belief vs appraised evidence with the identity line."""
    import matplotlib.pyplot as plt

    latent = df["speaker_belief_snapshot"].to_numpy(dtype=float)
    appraised = df["appraised_evidence_calibrated"].to_numpy(dtype=float)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1, label="identity (y = x)")
    ax.scatter(latent, appraised, s=8, alpha=0.3, color="tab:blue", label="utterances")
    ax.set_xlabel("latent speaker belief")
    ax.set_ylabel("appraised evidence (calibrated)")
    ax.set_title("appraiser alignment")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best")
    return ax


def plot_oracle_aligned_kappa_recovery(summary_df: pd.DataFrame, ax: Optional["object"] = None):
    """Prescribed vs oracle-aligned recovered kappa per condition (log-log)."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter, ScalarFormatter

    recovered_col = (
        "kappa_recovered_median" if "kappa_recovered_median" in summary_df.columns
        else "kappa_recovered_mean"
    )
    grouped = summary_df.groupby("kappa_prescribed")[recovered_col].median()
    x = grouped.index.to_numpy(dtype=float)
    y = grouped.to_numpy(dtype=float)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    lo = float(min(x.min(), np.nanmin(y)))
    hi = float(max(x.max(), np.nanmax(y)))
    ax.plot([lo, hi], [lo, hi], "--", color="grey", linewidth=1, label="identity (y = x)")
    ax.scatter(x, y, color="tab:red", zorder=3, label="conditions")
    # Log-log only when all values are positive; a real
    # appraiser can drive recovered kappa <= 0, so fall back to linear to show all points.
    if np.nanmin(x) > 0 and np.nanmin(y) > 0:
        ax.set_xscale("log")
        ax.set_yscale("log")
        for axis in (ax.xaxis, ax.yaxis):
            fmt = ScalarFormatter(useMathText=False)
            fmt.set_scientific(False)
            axis.set_major_formatter(fmt)
            axis.set_minor_formatter(NullFormatter())
        ax.set_xticks(x)
        ax.set_yticks(x)
        ax.set_aspect("equal", adjustable="box")
    else:
        ax.axhline(0.0, color="lightgrey", linewidth=0.8, zorder=1)
    ax.set_xlabel("prescribed kappa")
    ax.set_ylabel("oracle-aligned recovered kappa (median)")
    ax.set_title("oracle-aligned kappa recovery")
    ax.legend(loc="best")
    return ax


def plot_trajectory_deviation(
    deviation_by_kappa: "dict[float, list[float]]",
    ax: Optional["object"] = None,
):
    """Plot mean-absolute belief deviation by round, one line per kappa condition."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    for kappa in sorted(deviation_by_kappa):
        series = deviation_by_kappa[kappa]
        ax.plot(range(len(series)), series, marker="o", markersize=3, label=f"kappa={kappa:g}")
    ax.set_xlabel("round")
    ax.set_ylabel("mean abs belief deviation vs oracle")
    ax.set_title("trajectory fidelity")
    ax.legend(loc="best", fontsize=8)
    return ax


# --- Regime metrics (DeGroot / FJ / committed) + slider auditability ---
def variance_by_round(belief_by_round: np.ndarray) -> np.ndarray:
    """Cross-agent variance at each round (length n_rounds+1)."""
    return np.var(np.asarray(belief_by_round, dtype=float), axis=1)


def degroot_metrics(belief_by_round: np.ndarray) -> dict:
    from bca_beta import classical
    bbr = np.asarray(belief_by_round, dtype=float)
    v = variance_by_round(bbr)
    consensus = float(bbr[-1].mean())
    reference = classical.degroot_consensus(bbr[0])
    return {"final_variance": float(v[-1]), "consensus_value": consensus,
            "consensus_reference": reference, "consensus_abs_error": abs(consensus - reference)}


def fj_metrics(belief_by_round: np.ndarray, tol: float = 1e-4) -> dict:
    v = variance_by_round(np.asarray(belief_by_round, dtype=float))
    return {"initial_variance": float(v[0]), "final_variance": float(v[-1]),
            "persists": bool(v[-1] > tol)}


def fj_reference_metrics(belief_by_round, kappas, weight: float = 1.0, gamma: float = 1.0) -> dict:
    """Compare FJ terminal beliefs to the classical FJ fixed point of the *discounted* update.

    The steady state of the forgetting-factor Beta update is exactly an FJ fixed point
    ``x* = (I - ΛW)^{-1} (I - Λ) x0`` with social susceptibility
    ``λ_i(γ) = w / (κ_i (1 - γ) + w) = 1 - ρ*_i`` and neighbour-averaging
    ``W_ij = 1/(N-1)`` (zero diagonal) -- the complete graph the engine actually runs.

    * ``γ = 1``: ``λ_i = 1`` (no anchoring) -> the solve is singular and the FJ
      prediction is *consensus*; we return the mean of the initial beliefs, matching the
      static-latent dynamics that always converge.
    * ``γ < 1``: ``λ_i < 1`` -> a genuine interior fixed point with persistent
      disagreement, which the discounted dynamics actually reach. ``R²`` of the observed
      terminal beliefs against this reference is then high.
    """
    from bca_beta import classical
    bbr = np.asarray(belief_by_round, dtype=float)
    kappas = np.asarray(kappas, dtype=float)
    n = bbr.shape[1]
    x0, obs = bbr[0], bbr[-1]
    if gamma >= 1.0:
        # Fully pliable limit: FJ predicts consensus = mean of initial opinions.
        ref = np.full(n, float(x0.mean()))
    else:
        lam = weight / (kappas * (1.0 - gamma) + weight)
        W = (np.ones((n, n)) - np.eye(n)) / (n - 1)  # neighbour averaging, zero self-weight
        ref = classical.fj_fixed_point(x0, lam, W)
    ss_res = float(np.sum((obs - ref) ** 2))
    ss_tot = float(np.sum((obs - obs.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"r2": r2, "reference_mean": float(ref.mean()),
            "observed_terminal_mean": float(obs.mean()), "gamma": float(gamma)}


def committed_dose_response(belief_by_round: np.ndarray, committed_mask: np.ndarray,
                            n_committed: int) -> dict:
    bbr = np.asarray(belief_by_round, dtype=float)
    free = ~np.asarray(committed_mask, dtype=bool)
    final_free = bbr[-1][free]
    return {"n_committed": int(n_committed),
            "converted_fraction": float(np.mean(final_free > 0.5)) if final_free.size else float("nan"),
            "mean_free_belief": float(final_free.mean()) if final_free.size else float("nan")}


def belief_slider_correlation(df: pd.DataFrame, belief_col: str = "belief",
                              slider_col: str = "slider_unit_scaled") -> dict:
    b = df[belief_col].to_numpy(dtype=float)
    s = df[slider_col].to_numpy(dtype=float)
    return {"pearson": _safe_corr_pair(b, s, "pearson"),
            "spearman": _safe_corr_pair(b, s, "spearman"),
            "mae": float(np.mean(np.abs(b - s))), "n": int(len(df))}


def slider_reliability_bins(df: pd.DataFrame, n_bins: int = 10, belief_col: str = "belief",
                            slider_col: str = "slider_unit_scaled") -> pd.DataFrame:
    b = df[belief_col].to_numpy(dtype=float)
    s = df[slider_col].to_numpy(dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(b, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for k in range(n_bins):
        m = idx == k
        rows.append({"bin_low": float(edges[k]), "bin_high": float(edges[k + 1]),
                     "count": int(m.sum()),
                     "belief_mean": float(b[m].mean()) if m.any() else float("nan"),
                     "slider_mean": float(s[m].mean()) if m.any() else float("nan")})
    return pd.DataFrame(rows)
