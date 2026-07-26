"""Paper figures, regenerated from logged outputs (no API calls).

Written to ``experiments/outputs/figures/``:

  fig1_regimes.png       one-knob arc: DeGroot consensus, FJ persistence + gamma=1
                         collapse, committed-minority dose-response (headline model)
  fig2_channel.png       appraiser transfer functions, DeGroot drift per model, and
                         pliable-vs-anchored sensitivity to channel bias
  fig3_kappa.png         oracle-aligned kappa recovery with per-seed spread
  fig4_gamma_oracle.png  FJ vs DeGroot final variance across the oracle gamma sweep
                         (run ``gamma_sweep_oracle`` first; it is free)

    python -m bca_beta.experiments.make_figures
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bca_beta import classical
from bca_beta.experiments.classical_regimes import WEIGHT, build_regime_agents

OUT = Path(__file__).resolve().parent / "outputs"
FIG = OUT / "figures"
GAMMA = 0.7
HEADLINE = "gpt-5.4-mini"

# key, display label, colour, linestyle
MODELS = [
    ("gpt-5.4-mini", "gpt-5.4-mini", "#0173B2", "-"),
    ("gpt-5.4", "gpt-5.4", "#029E73", "-"),
    ("llama4-scout", "Llama-4-Scout", "#D55E00", "-"),
    ("claude-sonnet-4-6", "Claude-Sonnet-4.6", "#CC78BC", "-"),
]

STUBBORN_C = "#c0392b"
PLIABLE_C = "#2980b9"

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "legend.fontsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
})


def _regime_path(key: str, regime: str, name: str, gamma: float = GAMMA) -> Path:
    return OUT / key / "classical_regimes" / f"{gamma:.1f}" / regime / name


def _beliefs(key: str, regime: str, gamma: float = GAMMA) -> np.ndarray:
    return pd.read_csv(_regime_path(key, regime, "belief_by_round.csv", gamma)).to_numpy()


def _fj_reference(bbr: np.ndarray, gamma: float) -> np.ndarray:
    n = bbr.shape[1]
    if gamma >= 1.0:
        return np.full(n, float(bbr[0].mean()))
    agents, _ = build_regime_agents("fj", n, seed=0, gamma=gamma)
    kappas = np.array([a.belief_for().kappa for a in agents], dtype=float)
    lam = WEIGHT / (kappas * (1.0 - gamma) + WEIGHT)
    W = (np.ones((n, n)) - np.eye(n)) / (n - 1)
    return classical.fj_fixed_point(bbr[0], lam, W)


def _plot_fj_panel(ax, bbr: np.ndarray, gamma: float, title: str) -> None:
    ref = _fj_reference(bbr, gamma)
    stubborn = np.isin(np.round(bbr[0], 1), (0.1, 0.9))
    rounds = np.arange(bbr.shape[0])
    for r in np.unique(np.round(ref, 3)):
        ax.axhline(r, color="0.35", ls=":", lw=1, zorder=0)
    for j in range(bbr.shape[1]):
        ax.plot(rounds, bbr[:, j], color=STUBBORN_C if stubborn[j] else PLIABLE_C,
                lw=0.9, alpha=0.55)
    ax.set_xlim(0, rounds[-1])
    ax.set_ylim(0, 1)
    ax.set_xlabel("round")
    ax.set_title(title)


def fig1_regimes() -> None:
    fig, axes = plt.subplots(1, 4, figsize=(11.6, 2.45))

    # (a) DeGroot consensus: per-agent trajectories coloured by initial belief.
    ax = axes[0]
    bbr = _beliefs(HEADLINE, "degroot")
    rounds = np.arange(bbr.shape[0])
    cmap = plt.get_cmap("Blues")
    for j in np.argsort(bbr[0]):
        ax.plot(rounds, bbr[:, j], color=cmap(0.45 + 0.4 * bbr[0, j]), lw=0.9,
                alpha=0.9)
    ax.axhline(bbr[0].mean(), color="k", ls=":", lw=1)
    ax.text(rounds[-1], bbr[0].mean() + 0.03, "initial mean", ha="right", fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_xlabel("round")
    ax.set_ylabel("belief $b$")
    ax.set_title("DeGroot ($\\kappa\\to 0$): consensus")

    # (b, c) FJ persistence at gamma=0.7 and collapse at gamma=1.
    _plot_fj_panel(axes[1], _beliefs(HEADLINE, "fj"), GAMMA,
                   "FJ (mixed $\\kappa$): disagreement persists")
    _plot_fj_panel(axes[2], _beliefs(HEADLINE, "fj", 1.0), 1.0,
                   "$\\gamma=1$ ablation: collapse")
    axes[2].legend(handles=[
        plt.Line2D([], [], color=STUBBORN_C, lw=1.4,
                   label="stubborn ($\\kappa=16$)"),
        plt.Line2D([], [], color=PLIABLE_C, lw=1.4, label="pliable ($\\kappa=1$)"),
        plt.Line2D([], [], color="0.35", ls=":", lw=1,
                   label="FJ fixed points (theory)"),
    ], loc="upper right", frameon=False)

    # (d) Committed minority: per-seed dose-response + closed-form reference.
    ax = axes[3]
    dr = pd.read_csv(_regime_path(HEADLINE, "committed", "dose_response.csv"))
    ref = dr.groupby("f")["prop3_reference"].mean()
    ax.plot(ref.index, ref.values, color="k", ls="--", lw=1,
            label="stylised full-following reference")
    mean = dr.groupby("f")["mean_free_belief"].mean()
    ax.scatter(dr["f"], dr["mean_free_belief"], s=9, color="#0173B2", alpha=0.4,
               label="seeds")
    ax.plot(mean.index, mean.values, color="#0173B2", lw=1.6, marker="o", ms=3.5,
            label="mean")
    ax.set_ylim(0, 0.55)
    ax.set_xlabel("committed fraction $f$")
    ax.set_ylabel("mean free-agent belief")
    ax.set_title("Committed ($\\kappa\\to\\infty$): no tipping")
    ax.legend(loc="upper left", frameon=False)

    fig.tight_layout(w_pad=1.2)
    fig.savefig(FIG / "fig1_regimes.png", bbox_inches="tight")
    plt.close(fig)



def fig2_channel() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 2.6),
                             gridspec_kw={"width_ratios": [1.1, 1.25, 0.9]})

    # (a) Appraiser transfer function per model.
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color="gray", ls=":", lw=1, label="identity (faithful)")
    for key, label, color, ls in MODELS:
        b = pd.read_csv(OUT / key / "kappa_recovery" / f"{GAMMA:.1f}" /
                        "alignment_by_bin.csv")
        b = b[b["count"] >= 50]
        ax.plot(b["latent_mean"], b["appraised_mean"], color=color, ls=ls,
                marker="o", ms=3, lw=1.3, label=label)
    ax.set_xlabel("latent speaker belief")
    ax.set_ylabel("mean appraised evidence")
    ax.set_title("(a) Each channel's transfer function")
    ax.legend(loc="upper left", frameon=False)

    # (b) DeGroot population-mean drift per model.
    ax = axes[1]
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    for key, label, color, ls in MODELS:
        mean_traj = _beliefs(key, "degroot").mean(axis=1)
        ax.plot(np.arange(len(mean_traj)), mean_traj, color=color, ls=ls, lw=1.5,
                marker="o", ms=2.5)
        ax.annotate(f"{mean_traj[-1]:.2f}", (len(mean_traj) - 1, mean_traj[-1]),
                    textcoords="offset points", xytext=(6, -2), fontsize=7,
                    color=color)
    ax.text(0.5, 0.515, "initial mean 0.50", fontsize=7, color="gray")
    ax.set_xlim(0, 23.5)
    ax.set_ylim(0, 0.72)
    ax.set_xlabel("round")
    ax.set_ylabel("population mean belief")
    ax.set_title("(b) Pliable population: bias compounds")

    # (c) Displacement from reference: pliable regime vs prior-anchored regime.
    ax = axes[2]
    xs = np.arange(len(MODELS))
    dg, fj = [], []
    for key, _, _, _ in MODELS:
        m = json.loads((_regime_path(key, "degroot", "metrics.json")).read_text())
        dg.append(m["consensus_abs_error"])
        m = json.loads((_regime_path(key, "fj", "metrics.json")).read_text())
        fj.append(abs(m["observed_terminal_mean"] - m["reference_mean"]))
    w = 0.38
    ax.bar(xs - w / 2, dg, width=w, color=[c for _, _, c, _ in MODELS], alpha=0.9,
           label="DeGroot ($\\kappa\\to0$)")
    ax.bar(xs + w / 2, fj, width=w, color=[c for _, _, c, _ in MODELS], alpha=0.35,
           label="FJ (anchored)")
    ax.set_xticks(xs)
    ax.set_xticklabels([lbl.replace("Claude-Sonnet", "Sonnet") for _, lbl, _, _
                        in MODELS], rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("|final $-$ reference|")
    ax.set_title("(c) Priors absorb channel bias")
    leg = ax.legend(loc="upper left", frameon=False)
    for h in leg.legend_handles:
        h.set_color("#555555")

    fig.tight_layout(w_pad=1.4)
    fig.savefig(FIG / "fig2_channel.png", bbox_inches="tight")
    plt.close(fig)


def fig3_kappa() -> None:
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    ax.axvspan(0.4, 2.0, color="0.92", zorder=0)
    ax.text(0.95, 34, "noisy zone: pliable listeners\nend up near speakers, so the\n"
            "inversion divides by small $|s-b|$", fontsize=6.5, ha="center",
            color="0.35")
    lims = [0.3, 70]
    ax.plot(lims, lims, color="gray", ls=":", lw=1, label="identity")
    for key, label, color, ls in MODELS:
        df = pd.read_csv(OUT / key / "kappa_recovery" / f"{GAMMA:.1f}" /
                         "exact_kappa_summary.csv")
        pres = sorted(df["kappa_condition"].unique())
        med = np.array([df[df["kappa_condition"] == k]
                        ["kappa_exact"].median() for k in pres])
        # medians below the log-axis floor are unplottable; break the line there
        med = np.where(med >= lims[0], med, np.nan)
        seeds = df[df["kappa_exact"] > 0]
        ax.scatter(seeds["kappa_condition"], seeds["kappa_exact"],
                   s=7, color=color, alpha=0.30, zorder=2)
        ax.plot(pres, med, color=color, ls=ls, lw=1.5,
                marker="o", ms=4, zorder=3, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_xlabel("prescribed $\\kappa$")
    ax.set_ylabel("recovered $\\hat{\\kappa}$ (per-condition median)")
    ax.set_title("$\\kappa$ recovery through the language channel")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_kappa.png", bbox_inches="tight")
    plt.close(fig)


def fig4_gamma_oracle() -> None:
    summary = pd.read_csv(OUT / "gamma_oracle" / "gamma_oracle_summary.csv")
    var = summary[summary["metric"] == "final_variance"]
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for regime, label, color, marker in [
        ("fj", "FJ (heterogeneous $\\kappa$)", "#c0392b", "o"),
        ("degroot", "DeGroot ($\\kappa\\to0$)", "#2980b9", "s"),
    ]:
        d = var[var["regime"] == regime].sort_values("gamma", ascending=False)
        ax.plot(d["gamma"], d["value"], color=color, marker=marker, ms=4,
                lw=1.5, label=label)
    ax.invert_xaxis()
    ax.axvline(0.7, color="0.8", ls=":", lw=1)
    ax.text(0.7, ax.get_ylim()[1] * 0.95, " operating point", fontsize=7,
            color="0.4", va="top")
    ax.set_xlabel("forgetting factor $\\gamma$")
    ax.set_ylabel("final cross-agent variance")
    ax.set_title("Disagreement persists only for $\\gamma<1$")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_gamma_oracle.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig1_regimes()
    fig2_channel()
    fig3_kappa()
    fig4_gamma_oracle()
    print(f"wrote figures to {FIG}")


if __name__ == "__main__":
    main()
