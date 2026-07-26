"""Free (no-API) oracle sweep over the forgetting factor gamma.

This runs only the *belief mechanism* -- oracle evidence, where each listener hears the
speaker's current credence directly (no language model) -- across the three classical
regimes and a grid of gamma values. Because it makes no OpenAI calls it is essentially
free, and it is how we *choose* gamma before spending on the LLM channel.

For each gamma it reports:

* DeGroot   -- final cross-agent variance (consensus check; should stay ~0).
* FJ        -- final variance, plus the stubborn-group (kappa=16) and pliable-group
               (kappa=1) terminal variances. At gamma=1 the population converges
               (variance -> 0); at gamma<1 the stubborn extremes hold apart and a
               genuine persistent-disagreement variance survives.
* Committed -- the dose-response (mean free-agent belief vs committed fraction f).

Run it with::

    python -m bca_beta.experiments.gamma_sweep_oracle

Outputs (under ``experiments/outputs/gamma_oracle/``): ``gamma_oracle_summary.csv``
(tidy: gamma, regime, metric, value) and ``gamma_oracle_committed_dose.csv``
(gamma, f, mean_free_belief, converted_fraction).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import networkx as nx
import numpy as np
import pandas as pd

from bca_beta import analysis
from bca_beta.engine import run_round_robin
from bca_beta.experiments.classical_regimes import (
    A_MINUS,
    A_PLUS,
    CONCEPT_ID,
    DEFAULT_F_GRID,
    WEIGHT,
    build_regime_agents,
)

DEFAULT_GAMMAS = (1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5)
DEFAULT_OUT = Path(__file__).resolve().parent / "outputs" / "gamma_oracle"
STUBBORN_KAPPA = 16.0
PLIABLE_KAPPA = 1.0


def _oracle_belief_by_round(agents, n_rounds: int, seed: int) -> np.ndarray:
    """Run the no-API oracle mechanism and return ``belief_by_round`` (n_rounds+1, N)."""
    res = run_round_robin(
        agents=agents,
        graph=nx.complete_graph(len(agents)),
        rng=np.random.default_rng(seed),
        n_rounds=n_rounds,
        weight=WEIGHT,
        generator=None,
        appraiser=None,
        a_plus=A_PLUS,
        a_minus=A_MINUS,
        concept_id=CONCEPT_ID,
        run_meta={"seed": seed},
        evidence_mode="oracle",
    )
    return res.belief_by_round


def run_gamma_oracle_sweep(
    gammas: Sequence[float] = DEFAULT_GAMMAS,
    n_agents: int = 20,
    n_rounds: int = 20,
    seed: int = 0,
    f_grid: Sequence[float] = DEFAULT_F_GRID,
) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """Sweep gamma over the three regimes via the oracle mechanism (no API).

    Returns:
        ``(summary_df, dose_df)``. ``summary_df`` is tidy with columns
        ``gamma, regime, metric, value``; ``dose_df`` has ``gamma, f,
        mean_free_belief, converted_fraction``.
    """
    summary_rows: list[dict] = []
    dose_rows: list[dict] = []

    for gamma in gammas:
        # --- DeGroot: pliable population should reach consensus at any gamma ---
        dg_agents, _ = build_regime_agents("degroot", n_agents, seed, gamma=gamma)
        dg_bbr = _oracle_belief_by_round(dg_agents, n_rounds, seed)
        summary_rows.append(
            {"gamma": gamma, "regime": "degroot", "metric": "final_variance",
             "value": float(analysis.variance_by_round(dg_bbr)[-1])}
        )

        # --- FJ: stubborn extremes vs pliable middle; disagreement grows as gamma falls ---
        fj_agents, _ = build_regime_agents("fj", n_agents, seed, gamma=gamma)
        fj_bbr = _oracle_belief_by_round(fj_agents, n_rounds, seed)
        terminal = fj_bbr[-1]
        kappas = np.array([a.belief_for().kappa for a in fj_agents], dtype=float)
        stubborn = terminal[kappas == STUBBORN_KAPPA]
        pliable = terminal[kappas == PLIABLE_KAPPA]
        for metric, value in [
            ("final_variance", float(np.var(terminal))),
            ("stubborn_terminal_variance", float(np.var(stubborn)) if stubborn.size else float("nan")),
            ("pliable_terminal_variance", float(np.var(pliable)) if pliable.size else float("nan")),
        ]:
            summary_rows.append({"gamma": gamma, "regime": "fj", "metric": metric, "value": value})

        # --- Committed: smooth dose-response of mean free-agent belief vs f ---
        last_dose_belief = float("nan")
        for f in f_grid:
            cm_agents, mask = build_regime_agents("committed", n_agents, seed, f=f, gamma=gamma)
            cm_bbr = _oracle_belief_by_round(cm_agents, n_rounds, seed)
            dr = analysis.committed_dose_response(cm_bbr, mask, n_committed=int(mask.sum()))
            dose_rows.append({"gamma": gamma, "f": f,
                              "mean_free_belief": dr["mean_free_belief"],
                              "converted_fraction": dr["converted_fraction"]})
            last_dose_belief = dr["mean_free_belief"]
        # A single scalar handle on the committed effect for the summary table.
        summary_rows.append(
            {"gamma": gamma, "regime": "committed", "metric": f"mean_free_belief_at_f={max(f_grid):g}",
             "value": last_dose_belief}
        )

    return pd.DataFrame(summary_rows), pd.DataFrame(dose_rows)


def _print_summary(summary: pd.DataFrame) -> None:
    """Print the gamma-selection table: FJ final variance is the headline column."""
    fj = summary[(summary.regime == "fj") & (summary.metric == "final_variance")]
    dg = summary[(summary.regime == "degroot") & (summary.metric == "final_variance")]
    fj_stub = summary[(summary.regime == "fj") & (summary.metric == "stubborn_terminal_variance")]
    print("Oracle gamma sweep (no API) -- choosing gamma")
    print("-" * 64)
    print(f"{'gamma':>7} {'DeGroot var':>13} {'FJ var':>12} {'FJ stubborn var':>17}")
    for g in sorted(summary["gamma"].unique(), reverse=True):
        dgv = float(dg[dg.gamma == g]["value"].iloc[0])
        fjv = float(fj[fj.gamma == g]["value"].iloc[0])
        stv = float(fj_stub[fj_stub.gamma == g]["value"].iloc[0])
        print(f"{g:>7.2f} {dgv:>13.5f} {fjv:>12.5f} {stv:>17.5f}")
    print("-" * 64)
    print("Pick gamma where FJ var is clearly > 0 (persistent disagreement) while "
          "DeGroot stays ~0 (consensus).")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point for the free oracle gamma sweep."""
    parser = argparse.ArgumentParser(description="BCA free oracle gamma sweep (no API)")
    parser.add_argument("--gammas", type=float, nargs="+", default=list(DEFAULT_GAMMAS))
    parser.add_argument("--n-agents", type=int, default=20)
    parser.add_argument("--n-rounds", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--f-grid", type=float, nargs="+", default=list(DEFAULT_F_GRID))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    summary, dose = run_gamma_oracle_sweep(
        gammas=tuple(args.gammas), n_agents=args.n_agents, n_rounds=args.n_rounds,
        seed=args.seed, f_grid=tuple(args.f_grid),
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "gamma_oracle_summary.csv", index=False)
    dose.to_csv(out_dir / "gamma_oracle_committed_dose.csv", index=False)
    _print_summary(summary)
    print(f"wrote oracle gamma-sweep outputs to {out_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
