"""Exact kappa recovery from logged events (no API calls).

Inverts the exact one-step identity (paper App. F.2)

    b' = b + (w/n')(e - b) + (kappa(1-gamma)/n')(b0 - b),   n' = kappa + gamma*m + w,

for kappa per event, with m = w(1-gamma^c)/(1-gamma) the listener's discounted
evidence count and b0 its initial belief (read from the c=0 event rows). Unlike
the gamma=1 shortcut kappa = w/eta - m - w, this inversion is algebraically exact
at any gamma: with the consumed evidence it recovers the prescribed kappa to numerical
precision (validated below), so oracle-aligned deviations (substituting the
speaker's latent belief s for e) are attributable to the language channel alone.

Reads  experiments/outputs/<key>/kappa_recovery/<gamma>/events.csv and writes
       exact_kappa_summary.csv  (per-condition, per-seed medians)
       exact_recovery.json      (condition medians, Spearman, rel. error, MAE)

    python -m bca.experiments.exact_kappa_reanalysis
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

OUT = Path(__file__).resolve().parent / "outputs"
MODELS = ["gpt-5.4-mini", "gpt-5.4", "llama4-scout", "claude-sonnet-4-6"]
GAMMA_DIR = "0.7"
MIN_GAP = 0.05      # |s - b| conditioning threshold (same as the run-time analysis)
MIN_DEN = 1e-6      # numerical floor for the inversion denominator


def exact_kappa(df: pd.DataFrame, evidence_col: str) -> "tuple[np.ndarray, np.ndarray]":
    g, w = df["gamma"].values, df["weight"].values
    b, b2 = df["listener_belief_before"].values, df["listener_belief_after"].values
    ev, c, b0 = df[evidence_col].values, df["c_before"].values, df["b0"].values
    m = w * (1 - g**c) / (1 - g)
    num = w * (ev - b) - (b2 - b) * (g * m + w)
    den = (b2 - b) - (1 - g) * (b0 - b)
    with np.errstate(divide="ignore", invalid="ignore"):
        return num / den, den


def reanalyze(key: str) -> dict:
    d = OUT / key / "kappa_recovery" / GAMMA_DIR
    df = pd.read_csv(d / "events.csv")
    first = df[df.c_before == 0].drop_duplicates(["kappa_condition", "seed", "listener_id"])
    first = first.assign(b0=first.listener_alpha_before
                         / (first.listener_alpha_before + first.listener_beta_before))
    df = df.merge(first[["kappa_condition", "seed", "listener_id", "b0"]],
                  on=["kappa_condition", "seed", "listener_id"])

    # validation: inversion with the consumed evidence must recover kappa exactly
    km, dm = exact_kappa(df, "evidence_used_for_update")
    ok = np.abs(dm) > 1e-9
    mech_err = float(np.nanmax(np.abs(km[ok] - df["kappa_condition"].values[ok])))

    ko, den = exact_kappa(df, "speaker_belief_snapshot")
    s, b = df["speaker_belief_snapshot"].values, df["listener_belief_before"].values
    keep = (np.abs(s - b) > MIN_GAP) & (np.abs(den) > MIN_DEN)
    d2 = df.loc[keep].assign(kappa_exact=ko[keep])

    med = d2.groupby("kappa_condition")["kappa_exact"].median()
    ks = med.index.values
    summary = dict(
        estimator="exact F.2 inversion (oracle-aligned)",
        min_gap=MIN_GAP,
        mechanical_max_abs_err=mech_err,
        kept_fraction=float(keep.mean()),
        condition_median_recovery={float(k): float(v) for k, v in med.items()},
        spearman=float(spearmanr(ks, med.values).statistic),
        relative_error=float(np.mean(np.abs(med.values - ks) / ks)),
        mae=float(np.mean(np.abs(med.values - ks))),
    )
    d2.groupby(["kappa_condition", "seed"])["kappa_exact"].median().reset_index() \
        .to_csv(d / "exact_kappa_summary.csv", index=False)
    (d / "exact_recovery.json").write_text(json.dumps(summary, indent=1))
    return summary


def main() -> None:
    for key in MODELS:
        if not (OUT / key / "kappa_recovery" / GAMMA_DIR / "events.csv").exists():
            print(f"{key}: no events.csv under kappa_recovery/{GAMMA_DIR} "
                  f"(run kappa_recovery first) — skipped")
            continue
        s = reanalyze(key)
        print(f"{key}: mech_err={s['mechanical_max_abs_err']:.1e} "
              f"spearman={s['spearman']:.3f} rel_err={s['relative_error']:.2f} "
              f"kept={s['kept_fraction']:.2f}")


if __name__ == "__main__":
    main()
