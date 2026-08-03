"""Classical-regime experiments through the LLM engine.

Runs three classical-regime experiments (DeGroot consensus, FJ persistent
disagreement, committed-minority dose-response) through the LLM engine,
computes regime metrics, overlays closed-form references, and writes
per-regime outputs + a combined slider log for the auditability check.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence

import networkx as nx
import numpy as np
import pandas as pd

from bca import analysis, classical
from bca.agent import Agent
from bca.calibration import Calibration
from bca.engine import build_initial_means, run_round_robin

CONCEPT_ID = "transit_priority"
A_PLUS = "Aldenvale should expand its rail-transit network"
A_MINUS = "Aldenvale should keep investing in roads"
DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_F_GRID = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40)
DEFAULT_OUT = Path(__file__).resolve().parent / "outputs" / "classical_regimes"

WEIGHT = 1.0


def build_regime_agents(
    regime: str,
    n_agents: int,
    seed: int,
    f: Optional[float] = None,
    gamma: float = 1.0,
) -> "tuple[list[Agent], np.ndarray]":
    """Build agents and a boolean committed_mask for the given regime.

    Args:
        regime: One of ``"degroot"``, ``"fj"``, ``"committed"``.
        n_agents: Population size.
        seed: RNG seed for initial means and FJ middle values.
        f: Committed fraction (required for ``"committed"`` regime).
        gamma: Forgetting factor in ``(0, 1]`` shared by all agents (default ``1.0``).
            At ``gamma < 1`` the FJ regime sustains genuine persistent disagreement
            instead of converging.

    Returns:
        ``(agents, committed_mask)`` where ``committed_mask`` is a boolean
        array of length ``n_agents``.

    Raises:
        ValueError: If ``regime`` is unknown.
    """
    rng = np.random.default_rng(seed)
    means = build_initial_means(n_agents, seed=seed)  # spread [0.1,0.9] + jitter

    if regime == "degroot":
        agents = [
            Agent.single_concept(i, float(means[i]), kappa=0.1, gamma=gamma)
            for i in range(n_agents)
        ]
        return agents, np.zeros(n_agents, dtype=bool)

    if regime == "fj":
        agents = []
        for i in range(n_agents):
            if i % 2 == 0:  # stubborn at extremes
                agents.append(
                    Agent.single_concept(
                        i, 0.1 if i % 4 == 0 else 0.9, kappa=16.0, gamma=gamma
                    )
                )
            else:  # pliable near middle
                agents.append(
                    Agent.single_concept(
                        i,
                        float(np.clip(0.5 + rng.uniform(-0.05, 0.05), 0.05, 0.95)),
                        kappa=1.0,
                        gamma=gamma,
                    )
                )
        return agents, np.zeros(n_agents, dtype=bool)

    if regime == "committed":
        n_c = int(round(f * n_agents))
        mask = np.array([i < n_c for i in range(n_agents)], dtype=bool)
        agents = [
            Agent.single_concept(
                i,
                1.0 - 1e-9 if mask[i] else 1e-9,
                kappa=4.0,
                committed=bool(mask[i]),
                gamma=gamma,
            )
            for i in range(n_agents)
        ]
        return agents, mask

    raise ValueError(f"unknown regime {regime!r}")


def estimate_calls(
    regime_grid: Sequence[str],
    n_seeds: int,
    n_rounds: int,
    n_agents: int,
    f_grid: Optional[Sequence[float]] = None,
) -> dict:
    """Project LLM call counts: 3 calls/agent/round (utterance + slider + appraisal).

    Args:
        regime_grid: List of regimes to include (``"degroot"``, ``"fj"``, ``"committed"``).
        n_seeds: Number of random seeds per condition.
        n_rounds: Rounds per run.
        n_agents: Agents per run.
        f_grid: Committed fraction grid (used when ``"committed"`` in ``regime_grid``).

    Returns:
        Dict with ``n_generator_calls``, ``n_slider_calls``, ``n_appraiser_calls``.
    """
    single_regimes = [r for r in regime_grid if r != "committed"]
    n_f = len(f_grid) if (f_grid is not None and "committed" in regime_grid) else (
        1 if "committed" in regime_grid else 0
    )
    n_runs = len(single_regimes) * n_seeds + n_f * n_seeds
    calls = n_runs * n_rounds * n_agents
    return {
        "n_generator_calls": calls,
        "n_slider_calls": calls,
        "n_appraiser_calls": calls,
    }


def run_classical_regimes(
    *,
    generator: Any,
    appraiser: Any,
    slider_probe: Any,
    n_agents: int,
    n_rounds: int,
    n_seeds: int,
    f_grid: Sequence[float],
    model: str,
    appraiser_model: str,
    out_dir: "str | Path",
    regimes: Sequence[str] = ("degroot", "fj", "committed"),
    make_plots: bool = True,
    max_workers: int = 1,
    cache: Optional[Any] = None,
    gamma: float = 1.0,
    provenance: Optional[dict] = None,
) -> dict:
    """Run the selected classical regimes through the LLM engine.

    Per run: calls ``run_round_robin`` with ``evidence_mode="appraised"`` and the
    provided ``slider_probe``. Aggregates per-regime ``belief_by_round``, computes
    regime metrics, overlays classical references, and writes outputs. Only the
    regimes listed in ``regimes`` are run (the committed f-sweep only when
    ``"committed"`` is selected).

    Args:
        generator: Utterance generator.
        appraiser: Stance appraiser.
        slider_probe: Slider probe for auditability logging.
        n_agents: Population size.
        n_rounds: Rounds per run.
        n_seeds: Seeds per condition.
        f_grid: Committed fraction values to sweep.
        model: Generator model identifier (stamped in run_meta).
        appraiser_model: Appraiser model identifier.
        out_dir: Directory for outputs.
        regimes: Which regimes to run; default all three.
        make_plots: If False skip matplotlib (useful in tests).
        max_workers: Parallelism for per-speaker LLM calls within a round.
        cache: Optional JSONCache; ``cache.save()`` is called after each run.

    Returns:
        Metrics dict keyed by regime (``"degroot"``, ``"fj"``, ``"committed"``).
    """
    out_dir = Path(out_dir)
    graph = nx.complete_graph(n_agents)

    all_slider_rows: list[dict] = []
    metrics_out: dict = {}
    run_id = 0

    # ------------------------------------------------------------------ #
    # Helper to execute one run and return the RoundRobinResult           #
    # ------------------------------------------------------------------ #
    def _one_run(regime: str, seed: int, f: Optional[float] = None):
        nonlocal run_id
        agents, committed_mask = build_regime_agents(regime, n_agents, seed, f=f, gamma=gamma)
        meta: dict = {
            "run_id": run_id,
            "seed": seed,
            "model": model,
            "appraiser_model": appraiser_model,
            "regime": regime,
            "f": f if f is not None else float("nan"),
        }
        res = run_round_robin(
            agents=agents,
            graph=graph,
            rng=np.random.default_rng(seed),
            n_rounds=n_rounds,
            weight=WEIGHT,
            generator=generator,
            appraiser=appraiser,
            slider_probe=slider_probe,
            a_plus=A_PLUS,
            a_minus=A_MINUS,
            concept_id=CONCEPT_ID,
            run_meta=meta,
            evidence_mode="appraised",
            max_workers=max_workers,
        )
        run_id += 1
        if cache is not None:
            cache.save()
        return res, committed_mask

    # Bound matplotlib axes lazily so unselected regimes don't import it.
    plt = None
    if make_plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

    # ================================================================== #
    # DeGroot                                                             #
    # ================================================================== #
    if "degroot" in regimes:
        dg_bbr_list: list[np.ndarray] = []
        dg_events: list[dict] = []
        dg_utterances: list[dict] = []

        for seed in range(n_seeds):
            res, _ = _one_run("degroot", seed)
            dg_bbr_list.append(res.belief_by_round)
            dg_events.extend(res.events)
            dg_utterances.extend(res.utterances)
            all_slider_rows.extend(res.sliders)

        dg_bbr = np.mean(np.stack(dg_bbr_list, axis=0), axis=0)  # (n_rounds+1, n_agents)
        dg_m = analysis.degroot_metrics(dg_bbr)
        v = analysis.variance_by_round(dg_bbr)
        dg_m["initial_variance"] = float(v[0])

        # classical reference already in dg_m["consensus_reference"]
        dg_dir = out_dir / "degroot"
        dg_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(dg_events).to_csv(dg_dir / "events.csv", index=False)
        pd.DataFrame(dg_utterances).to_csv(dg_dir / "utterances.csv", index=False)
        pd.DataFrame(dg_bbr).to_csv(dg_dir / "belief_by_round.csv", index=False)
        (dg_dir / "metrics.json").write_text(json.dumps(dg_m, indent=2))

        metrics_out["degroot"] = dg_m

        if make_plots:
            fig, ax = plt.subplots()
            ax.plot(analysis.variance_by_round(dg_bbr), marker="o")
            ax.set_xlabel("round")
            ax.set_ylabel("cross-agent variance")
            ax.set_title("DeGroot regime: variance by round")
            fig.savefig(dg_dir / "variance_by_round.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

    # ================================================================== #
    # FJ                                                                  #
    # ================================================================== #
    if "fj" in regimes:
        fj_bbr_list: list[np.ndarray] = []
        fj_events: list[dict] = []
        fj_utterances: list[dict] = []

        for seed in range(n_seeds):
            res, _ = _one_run("fj", seed)
            fj_bbr_list.append(res.belief_by_round)
            fj_events.extend(res.events)
            fj_utterances.extend(res.utterances)
            all_slider_rows.extend(res.sliders)

        fj_bbr = np.mean(np.stack(fj_bbr_list, axis=0), axis=0)
        fj_m = analysis.fj_metrics(fj_bbr)

        # Per-agent prescribed kappa (FJ assignment depends only on agent index).
        fj_agents, _ = build_regime_agents("fj", n_agents, seed=0, gamma=gamma)
        fj_kappas = np.array([a.belief_for().kappa for a in fj_agents], dtype=float)

        # Analytical FJ fixed-point reference (gamma-aware; consensus at gamma=1,
        # persistent disagreement at gamma<1).
        fj_m.update(analysis.fj_reference_metrics(fj_bbr, fj_kappas, weight=WEIGHT, gamma=gamma))

        # Per-kappa-group terminal spread: stubborn (kappa=16) hold apart while
        # pliable (kappa=1) converge.
        terminal = fj_bbr[-1]
        stubborn = fj_kappas == 16.0
        pliable = fj_kappas == 1.0
        fj_m["stubborn_terminal_var"] = (
            float(np.var(terminal[stubborn])) if stubborn.any() else float("nan")
        )
        fj_m["pliable_terminal_var"] = (
            float(np.var(terminal[pliable])) if pliable.any() else float("nan")
        )

        fj_dir = out_dir / "fj"
        fj_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(fj_events).to_csv(fj_dir / "events.csv", index=False)
        pd.DataFrame(fj_utterances).to_csv(fj_dir / "utterances.csv", index=False)
        pd.DataFrame(fj_bbr).to_csv(fj_dir / "belief_by_round.csv", index=False)
        (fj_dir / "metrics.json").write_text(json.dumps(fj_m, indent=2))

        metrics_out["fj"] = fj_m

        if make_plots:
            fig, ax = plt.subplots()
            ax.plot(analysis.variance_by_round(fj_bbr), marker="o")
            ax.set_xlabel("round")
            ax.set_ylabel("cross-agent variance")
            ax.set_title("FJ regime: variance by round")
            fig.savefig(fj_dir / "variance_by_round.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

            # Observed terminal vs FJ fixed-point reference scatter.
            lam = WEIGHT / (fj_kappas + WEIGHT)
            W = np.full((n_agents, n_agents), 1.0 / n_agents)
            ref = classical.fj_fixed_point(fj_bbr[0], lam, W)
            fig, ax = plt.subplots()
            ax.scatter(ref, terminal, s=12)
            lo = float(min(ref.min(), terminal.min()))
            hi = float(max(ref.max(), terminal.max()))
            ax.plot([lo, hi], [lo, hi], "--", color="grey", linewidth=1, label="identity")
            ax.set_xlabel("FJ fixed-point reference")
            ax.set_ylabel("observed terminal belief")
            ax.set_title(f"FJ shape match (R^2={fj_m['r2']:.3f})")
            ax.legend()
            fig.savefig(fj_dir / "fj_reference_scatter.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

    # ================================================================== #
    # Committed minority (dose-response over f_grid)                     #
    # ================================================================== #
    if "committed" in regimes:
        cm_dir = out_dir / "committed"
        cm_dir.mkdir(parents=True, exist_ok=True)

        dr_rows: list[dict] = []
        bbr_rows: list[dict] = []  # tidy long: f, seed, round, agent_id, belief
        cm_events: list[dict] = []
        cm_utterances: list[dict] = []

        # Prop. 3 closed-form reference curve
        eta_sched = classical.eta_schedule(kappa=4.0, weight=WEIGHT, n_steps=n_rounds)
        f_arr = np.asarray(list(f_grid), dtype=float)
        prop3_ref = classical.committed_minority_curve(f_arr, eta_sched)

        for fi, f_val in enumerate(f_grid):
            for seed in range(n_seeds):
                res, committed_mask = _one_run("committed", seed, f=float(f_val))
                cm_events.extend(res.events)
                cm_utterances.extend(res.utterances)
                all_slider_rows.extend(res.sliders)

                # tidy belief_by_round rows for this (f, seed)
                bbr = np.asarray(res.belief_by_round, dtype=float)
                for r in range(bbr.shape[0]):
                    for aid_idx, aid in enumerate(res.agent_order):
                        bbr_rows.append({
                            "f": float(f_val),
                            "seed": seed,
                            "round": r,
                            "agent_id": aid,
                            "belief": float(bbr[r, aid_idx]),
                        })

                # per-seed dose-response row
                dr = analysis.committed_dose_response(
                    res.belief_by_round, committed_mask, n_committed=int(committed_mask.sum())
                )
                dr_rows.append({
                    "f": float(f_val),
                    "seed": seed,
                    "converted_fraction": dr["converted_fraction"],
                    "mean_free_belief": dr["mean_free_belief"],
                    "prop3_reference": float(prop3_ref[fi]),
                })

        pd.DataFrame(cm_events).to_csv(cm_dir / "events.csv", index=False)
        pd.DataFrame(cm_utterances).to_csv(cm_dir / "utterances.csv", index=False)
        pd.DataFrame(bbr_rows).to_csv(cm_dir / "belief_by_round.csv", index=False)
        dr_df = pd.DataFrame(dr_rows)
        dr_df.to_csv(cm_dir / "dose_response.csv", index=False)

        # aggregate committed metrics across f values and seeds
        cm_metrics = {
            "dose_response": dr_df.groupby("f")[["converted_fraction", "mean_free_belief", "prop3_reference"]]
            .mean()
            .reset_index()
            .to_dict(orient="records")
        }
        (cm_dir / "metrics.json").write_text(json.dumps(cm_metrics, indent=2))

        metrics_out["committed"] = {
            "dose_response": dr_df.to_dict(orient="records"),
        }

        if make_plots:
            mean_dr = dr_df.groupby("f")[
                ["converted_fraction", "mean_free_belief", "prop3_reference"]
            ].mean()

            # HEADLINE: continuous effect size (mean free-agent belief) vs f, with
            # the Prop. 3 closed-form reference. Smooth & monotone (no tipping point).
            fig, ax = plt.subplots()
            ax.plot(mean_dr.index, mean_dr["mean_free_belief"], marker="o", label="LLM")
            ax.plot(mean_dr.index, mean_dr["prop3_reference"], linestyle="--",
                    label="Prop. 3 ref.")
            ax.set_xlabel("committed fraction f")
            ax.set_ylabel("mean free-agent belief")
            ax.set_title("Committed-minority dose-response")
            ax.legend()
            fig.savefig(cm_dir / "dose_response.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

            # SECONDARY (thresholded/binarized view at belief>0.5). The 0->1 jump is
            # a binarization artifact, NOT a tipping point; the smooth effect size is
            # the headline above.
            fig, ax = plt.subplots()
            ax.plot(mean_dr.index, mean_dr["converted_fraction"], marker="o", color="tab:red")
            ax.set_xlabel("committed fraction f")
            ax.set_ylabel("converted fraction (belief > 0.5)")
            ax.set_title("Committed minority: thresholded view (artifact, not a tipping point)")
            fig.savefig(cm_dir / "converted_fraction.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

    # ================================================================== #
    # Sliders.csv — combined auditability input                                #
    # ================================================================== #
    pd.DataFrame(all_slider_rows).to_csv(out_dir / "sliders.csv", index=False)

    if provenance is not None:
        metrics_out["provenance"] = provenance
        (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))

    return metrics_out


def _print_summary(metrics: dict) -> None:
    print("regime experiments")
    print("-" * 60)
    dg = metrics.get("degroot", {})
    print(f"DeGroot: initial_var={dg.get('initial_variance', float('nan')):.4f}  "
          f"final_var={dg.get('final_variance', float('nan')):.4f}  "
          f"consensus={dg.get('consensus_value', float('nan')):.4f}")
    fj = metrics.get("fj", {})
    print(f"FJ:      initial_var={fj.get('initial_variance', float('nan')):.4f}  "
          f"final_var={fj.get('final_variance', float('nan')):.4f}  "
          f"persists={fj.get('persists', '?')}")
    cm = metrics.get("committed", {})
    dr = cm.get("dose_response", [])
    print(f"Committed: {len(dr)} (f, seed) rows in dose-response")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point for the regime experiments."""
    parser = argparse.ArgumentParser(description="BCA regime experiments")
    parser.add_argument(
        "--regime", choices=["degroot", "fj", "committed", "all"], default="all",
        help="which regime(s) to run",
    )
    parser.add_argument("--n-agents", type=int, default=20)
    parser.add_argument("--n-rounds", type=int, default=10)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--f-grid", type=float, nargs="+", default=list(DEFAULT_F_GRID))
    parser.add_argument("--model-key", default=None,
                        help="registry model key (e.g. gpt-5.4, llama4-scout); "
                             "overrides --model and selects the endpoint")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="raw generator model id (legacy)")
    parser.add_argument("--appraiser-model", default=None)
    parser.add_argument("--calibration-json", type=Path, default=None)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="print call estimates and exit without API calls")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="forgetting factor in (0,1]; <1 sustains FJ persistent disagreement")
    args = parser.parse_args(argv)

    appraiser_model = args.appraiser_model or args.model
    regimes = (
        ("degroot", "fj", "committed")
        if args.regime == "all"
        else (args.regime,)
    )

    if args.dry_run:
        est = estimate_calls(
            regime_grid=list(regimes),
            n_seeds=args.n_seeds,
            n_rounds=args.n_rounds,
            n_agents=args.n_agents,
            f_grid=args.f_grid,
        )
        print("regime dry run")
        print(f"  generator calls : {est['n_generator_calls']}")
        print(f"  slider calls    : {est['n_slider_calls']}")
        print(f"  appraiser calls : {est['n_appraiser_calls']}")
        return 0

    # Build real LLM-backed objects only for a live run.
    from bca import channel, models
    from bca.llm import JSONCache

    spec = models.resolve_spec(model_key=args.model_key, model=args.model)
    appraiser_spec = (
        models.resolve_spec(model_key=None, model=args.appraiser_model)
        if args.appraiser_model else spec
    )
    out_dir = Path(args.out) if args.out != DEFAULT_OUT else models.default_output_root(spec) / "classical_regimes" / f"{args.gamma:.1f}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = None
    if args.cache:
        cache = JSONCache(out_dir / "cache.json")
        print(f"cache: {len(cache)} calls preloaded")

    cal_path = args.calibration_json or models.default_calibration_path(spec)
    if cal_path and Path(cal_path).exists():
        calibration = Calibration.load(cal_path)
        print(f"calibration: loaded {cal_path}")
    else:
        calibration = None
        print(f"WARNING: no calibration found for {spec.key} at {cal_path}; "
              f"running the appraiser UNCALIBRATED", flush=True)

    ch = channel.build_channel(spec, cache=cache, calibration=calibration,
                               appraiser_spec=appraiser_spec)
    provenance = channel.provenance_block(
        spec, appraiser_spec=appraiser_spec, gamma=args.gamma, calibration_path=cal_path,
    )

    metrics = run_classical_regimes(
        generator=ch.generator,
        appraiser=ch.appraiser,
        slider_probe=ch.slider_probe,
        n_agents=args.n_agents,
        n_rounds=args.n_rounds,
        n_seeds=args.n_seeds,
        f_grid=args.f_grid,
        model=spec.model_id,
        appraiser_model=appraiser_spec.model_id,
        out_dir=out_dir,
        regimes=regimes,
        make_plots=True,
        max_workers=args.concurrency,
        cache=cache,
        gamma=args.gamma,
        provenance=provenance,
    )
    _print_summary(metrics)
    print(f"wrote regime outputs to {out_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
