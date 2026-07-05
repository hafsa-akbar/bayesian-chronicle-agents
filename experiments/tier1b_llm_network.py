"""The LLM language channel on a single-concept network.

The prescribed prior strength ``kappa`` induces a time-varying susceptibility
``eta(c)``. This experiment routes evidence through the language channel

    speaker belief -> generated utterance -> calibrated appraiser -> evidence,

and asks whether the kappa-controlled dynamics survive. The non-tautological test is
**oracle-aligned recovery**: we recover ``eta``/``kappa`` using the *latent* speaker
belief, not the appraised evidence the Beta update actually consumed. (Recovering
with the consumed evidence is exact by construction -- kept only as a debug invariant.)

Efficient event loop (per kappa, seed, round):
  1. snapshot all speaker beliefs at the start of the round;
  2. generate one utterance per speaker from its snapshot belief;
  3. appraise each utterance once;
  4. broadcast the appraised evidence to all listeners (no per-listener appraisal);
  5. for each listener-speaker pair, update the listener with the calibrated evidence.

No OpenAI dependency is required to *use* :func:`simulate_run` /
:func:`run_tier1b_experiment` -- they accept any generator/appraiser satisfying the
protocols, so the science is unit-tested with deterministic fakes.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import networkx as nx
import numpy as np
import pandas as pd

from bca_beta import analysis
from bca_beta.agent import Agent
from bca_beta.calibration import Calibration
from bca_beta.engine import build_initial_means, run_round_robin

CONCEPT_ID = "transit_priority"
A_PLUS = "Aldenvale should expand its rail-transit network"
A_MINUS = "Aldenvale should keep investing in roads"
DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_KAPPAS = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
DEFAULT_OUT = Path(__file__).resolve().parent / "outputs" / "tier1b"

# Rough per-call token estimates for --dry-run cost projection.
GEN_TOKENS_PER_CALL = 200
APPR_TOKENS_PER_CALL = 160

ETA_MECH_COL = "eta_recovered_mechanical_using_appraised_evidence"
ETA_ORACLE_COL = "eta_recovered_oracle_aligned_using_speaker_belief"
KAPPA_ORACLE_COL = "kappa_recovered_oracle_aligned"


@dataclass
class Tier1bRunResult:
    """One (kappa, seed) run: event log, utterance log, and per-round trajectory."""

    events: list[dict]
    utterances: list[dict]
    belief_by_round: np.ndarray  # (n_rounds + 1, n_agents)
    agent_order: list


def simulate_run(
    *,
    kappa: float,
    seed: int,
    n_agents: int,
    n_rounds: int,
    weight: float,
    generator: Optional[Any],
    appraiser: Optional[Any],
    a_plus: str,
    a_minus: str,
    concept_id: str,
    model: str,
    appraiser_model: str,
    run_id: Any,
    evidence_mode: str = "appraised",
    graph: Optional[nx.Graph] = None,
    max_workers: int = 1,
    gamma: float = 1.0,
) -> Tier1bRunResult:
    """Run one (kappa, seed) round-robin and return its logs.

    ``evidence_mode="appraised"`` uses the generator+appraiser channel;
    ``evidence_mode="oracle"`` uses the speaker's snapshot belief directly (the
    no-API matched baseline, sharing the same init and event schedule for a given
    seed). ``gamma`` is the shared forgetting factor (``1.0`` = static latent).
    """
    means = build_initial_means(n_agents, seed=seed)
    agents = [
        Agent.single_concept(i, mean=float(means[i]), kappa=kappa, weight=weight,
                             concept=concept_id, gamma=gamma)
        for i in range(n_agents)
    ]
    _graph = graph if graph is not None else nx.complete_graph(n_agents)
    res = run_round_robin(
        agents=agents, graph=_graph, rng=np.random.default_rng(seed),
        n_rounds=n_rounds, weight=weight, generator=generator, appraiser=appraiser,
        a_plus=a_plus, a_minus=a_minus, concept_id=concept_id,
        run_meta={"run_id": run_id, "seed": seed, "model": model,
                  "appraiser_model": appraiser_model, "kappa_condition": kappa},
        evidence_mode=evidence_mode, max_workers=max_workers)
    return Tier1bRunResult(events=res.events, utterances=res.utterances,
                           belief_by_round=res.belief_by_round, agent_order=res.agent_order)


def estimate_calls(
    kappas: Sequence[float],
    n_seeds: int,
    n_rounds: int,
    n_agents: int,
    price_per_1m_tokens: Optional[float] = None,
) -> dict:
    """Project generator/appraiser call counts, tokens, and (optional) cost.

    One generator and one appraiser call per speaker per round per run, with one run
    per ``(kappa, seed)``.
    """
    n_runs = len(kappas) * n_seeds
    calls = n_runs * n_rounds * n_agents
    approx_tokens = calls * GEN_TOKENS_PER_CALL + calls * APPR_TOKENS_PER_CALL
    out = {
        "n_runs": n_runs,
        "n_generator_calls": calls,
        "n_appraiser_calls": calls,
        "approx_tokens": int(approx_tokens),
    }
    if price_per_1m_tokens is not None:
        out["approx_cost_usd"] = approx_tokens / 1e6 * price_per_1m_tokens
    return out


def _add_recovery_columns(events_df: pd.DataFrame) -> pd.DataFrame:
    """Append the mechanical (debug) and oracle-aligned (science) recovery columns."""
    events_df[ETA_MECH_COL] = analysis.recover_eta_from(events_df, "evidence_used_for_update")
    events_df[ETA_ORACLE_COL] = analysis.recover_eta_from(events_df, "speaker_belief_snapshot")
    events_df[KAPPA_ORACLE_COL] = analysis.recover_kappa_from(events_df, "speaker_belief_snapshot")
    return events_df


def _build_metrics(
    events_df: pd.DataFrame,
    utterances_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    deviation_by_kappa: "dict[float, list[float]]",
    model: str,
    appraiser_model: str,
    min_denominator: float = 0.05,
) -> dict:
    eta_pred = events_df["eta_predicted"].to_numpy()
    mech = events_df[ETA_MECH_COL].to_numpy()
    mech_finite = np.isfinite(mech)
    oracle = events_df[ETA_ORACLE_COL].to_numpy()
    oracle_finite = np.isfinite(oracle)

    # Oracle-aligned kappa recovery. Per-event kappa estimates invert eta, so they
    # blow up once a listener's belief is within the appraiser noise floor of the
    # speaker's (`|speaker - listener|` small). At gamma=1 the population converges, so
    # these ill-conditioned events dominate (measured well-conditioned kept-fraction
    # ~0.12); a forgetting factor gamma<1 keeps beliefs spread and lifts the kept
    # fraction sharply (~0.77 at gamma=0.7). The threshold is therefore RETAINED -- even
    # at gamma=0.7 ~23% of events are within the noise floor -- and the honest headline
    # restricts to *well-conditioned* events (denominator above `min_denominator`) and
    # takes the per-condition median; the all-events numbers are kept for transparency.
    ko = events_df[KAPPA_ORACLE_COL].to_numpy()
    kc = events_df["kappa_condition"].to_numpy()
    den = np.abs(
        events_df["speaker_belief_snapshot"].to_numpy() - events_df["listener_belief_before"].to_numpy()
    )

    def _cond_median(mask: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
        m = mask & np.isfinite(ko)
        if not m.any():
            return np.array([]), np.array([])
        s = pd.DataFrame({"k": kc[m], "kh": ko[m]}).groupby("k")["kh"].median()
        return s.index.to_numpy(dtype=float), s.to_numpy(dtype=float)

    def _recovery_block(mask: np.ndarray) -> dict:
        p, r = _cond_median(mask)
        err = np.abs(r - p)
        per_event = np.abs(ko[mask & np.isfinite(ko)] - kc[mask & np.isfinite(ko)])
        return {
            "kept_fraction": float(mask.mean()),
            "mae": float(np.mean(err)) if err.size else float("nan"),
            "relative_error": float(np.mean(err / p)) if err.size else float("nan"),
            "pearson": analysis._safe_corr_pair(p, r, "pearson"),
            "spearman": analysis._safe_corr_pair(p, r, "spearman"),
            "per_event_mae": float(np.mean(per_event)) if per_event.size else float("nan"),
            "condition_median_recovery": {str(pp): float(rr) for pp, rr in zip(p, r)},
        }

    well_conditioned = _recovery_block(den > min_denominator)
    all_events = _recovery_block(np.ones(len(events_df), dtype=bool))

    return {
        "model": model,
        "appraiser_model": appraiser_model,
        "n_events": int(len(events_df)),
        "n_utterances": int(len(utterances_df)),
        "n_parse_errors": int(utterances_df["parse_error"].sum()),
        "token_usage": {
            "generator_total_tokens": int(utterances_df["generator_total_tokens"].sum()),
            "generator_prompt_tokens": int(utterances_df["generator_prompt_tokens"].sum()),
            "generator_completion_tokens": int(utterances_df["generator_completion_tokens"].sum()),
            "appraiser_total_tokens": int(utterances_df["appraiser_total_tokens"].sum()),
            "appraiser_prompt_tokens": int(utterances_df["appraiser_prompt_tokens"].sum()),
            "appraiser_completion_tokens": int(utterances_df["appraiser_completion_tokens"].sum()),
            "total_tokens": int(
                utterances_df["generator_total_tokens"].sum()
                + utterances_df["appraiser_total_tokens"].sum()
            ),
        },
        "alignment": analysis.appraiser_alignment_metrics(utterances_df),
        "oracle_aligned_kappa": {
            "min_denominator": min_denominator,
            "aggregation": "per-condition median of per-event estimates, well-conditioned events",
            "mae": well_conditioned["mae"],
            "relative_error": well_conditioned["relative_error"],
            "pearson": well_conditioned["pearson"],
            "spearman": well_conditioned["spearman"],
            "kept_fraction": well_conditioned["kept_fraction"],
            "condition_median_recovery": well_conditioned["condition_median_recovery"],
            "per_event_mae": well_conditioned["per_event_mae"],
            "n_skipped_near_zero_denominator": int((~oracle_finite).sum()),
            "all_events": all_events,
        },
        "oracle_aligned_eta": {
            "mae": float(np.mean(np.abs(oracle[oracle_finite] - eta_pred[oracle_finite]))),
        },
        "mechanical": {
            "eta_mae": float(np.mean(np.abs(mech[mech_finite] - eta_pred[mech_finite]))),
        },
        "trajectory": {
            "final_mean_abs_dev_by_kappa": {
                str(k): (v[-1] if v else float("nan")) for k, v in deviation_by_kappa.items()
            },
            "overall_mean_abs_dev": float(
                np.mean([np.mean(v) for v in deviation_by_kappa.values()]) if deviation_by_kappa else float("nan")
            ),
        },
    }


def run_tier1b_experiment(
    *,
    generator: Any,
    appraiser: Any,
    kappas: Sequence[float],
    n_seeds: int,
    n_rounds: int,
    n_agents: int,
    weight: float = 1.0,
    model: str,
    appraiser_model: str,
    a_plus: str = A_PLUS,
    a_minus: str = A_MINUS,
    concept_id: str = CONCEPT_ID,
    out_dir: "str | Path" = DEFAULT_OUT,
    make_plots: bool = True,
    run_oracle_baseline: bool = True,
    min_denominator: float = 0.05,
    max_workers: int = 1,
    cache: Optional[Any] = None,
    gamma: float = 1.0,
    provenance: Optional[dict] = None,
) -> dict:
    """Run the full kappa x seed sweep, analyse it, and write all outputs.

    ``min_denominator`` restricts the oracle-aligned kappa recovery to well-conditioned
    events (``|speaker - listener| > min_denominator``), above the appraiser noise floor.
    ``max_workers`` parallelises the per-speaker LLM calls; ``cache`` (if given) is saved
    after each run so a long sweep is resumable.
    """
    all_events: list[dict] = []
    all_utterances: list[dict] = []
    deviation_by_kappa: dict[float, list[float]] = {}

    total_runs = len(kappas) * n_seeds
    run_id = 0
    for kappa in kappas:
        per_kappa_dev = []
        for seed in range(n_seeds):
            print(f"[run] starting run {run_id + 1}/{total_runs}  kappa={kappa:g} seed={seed}",
                  flush=True)
            llm = simulate_run(
                kappa=float(kappa), seed=seed, n_agents=n_agents, n_rounds=n_rounds, weight=weight,
                generator=generator, appraiser=appraiser, a_plus=a_plus, a_minus=a_minus,
                concept_id=concept_id, model=model, appraiser_model=appraiser_model,
                run_id=run_id, evidence_mode="appraised", max_workers=max_workers, gamma=gamma,
            )
            all_events.extend(llm.events)
            all_utterances.extend(llm.utterances)
            if cache is not None:
                cache.save()  # persist after each run so the sweep is resumable
            if run_oracle_baseline:
                oracle = simulate_run(
                    kappa=float(kappa), seed=seed, n_agents=n_agents, n_rounds=n_rounds, weight=weight,
                    generator=None, appraiser=None, a_plus=a_plus, a_minus=a_minus,
                    concept_id=concept_id, model="oracle", appraiser_model="oracle",
                    run_id=run_id, evidence_mode="oracle", gamma=gamma,
                )
                dev = analysis.trajectory_deviation(llm.belief_by_round, oracle.belief_by_round)
                per_kappa_dev.append(dev["mean_abs_dev_by_round"])
            run_id += 1
        if per_kappa_dev:
            deviation_by_kappa[float(kappa)] = np.mean(np.asarray(per_kappa_dev), axis=0).tolist()

    events_df = _add_recovery_columns(pd.DataFrame(all_events))
    utterances_df = pd.DataFrame(all_utterances)
    summary_df = analysis.summarize_oracle_aligned_kappa(events_df, min_denominator=min_denominator)
    alignment_bins = analysis.alignment_by_bin(utterances_df)
    metrics = _build_metrics(
        events_df, utterances_df, summary_df, deviation_by_kappa, model, appraiser_model,
        min_denominator=min_denominator,
    )
    if provenance is not None:
        metrics["provenance"] = provenance

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    events_df.to_csv(out_dir / "tier1b_events.csv", index=False)
    utterances_df.to_csv(out_dir / "tier1b_utterances.csv", index=False)
    summary_df.to_csv(out_dir / "tier1b_kappa_summary.csv", index=False)
    alignment_bins.to_csv(out_dir / "tier1b_alignment_by_bin.csv", index=False)
    (out_dir / "tier1b_metrics.json").write_text(json.dumps(metrics, indent=2))

    if make_plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ax = analysis.plot_appraised_vs_speaker_belief(utterances_df)
        ax.figure.savefig(out_dir / "plot_appraised_vs_speaker_belief.png", dpi=150, bbox_inches="tight")
        plt.close(ax.figure)

        ax = analysis.plot_oracle_aligned_kappa_recovery(summary_df)
        ax.figure.savefig(out_dir / "plot_oracle_aligned_kappa_recovery.png", dpi=150, bbox_inches="tight")
        plt.close(ax.figure)

        if deviation_by_kappa:
            ax = analysis.plot_trajectory_deviation(deviation_by_kappa)
            ax.figure.savefig(out_dir / "plot_trajectory_deviation.png", dpi=150, bbox_inches="tight")
            plt.close(ax.figure)

    return metrics


def _print_metrics(metrics: dict) -> None:
    a = metrics["alignment"]
    k = metrics["oracle_aligned_kappa"]
    print("LLM-network kappa recovery")
    print("-" * 60)
    print(f"model / appraiser   = {metrics['model']} / {metrics['appraiser_model']}")
    print(f"events / utterances = {metrics['n_events']} / {metrics['n_utterances']}"
          f"  (parse errors: {metrics['n_parse_errors']})")
    print(f"appraiser alignment : MAE={a['mae']:.4f} RMSE={a['rmse']:.4f} "
          f"r={a['pearson']:.4f} rho={a['spearman']:.4f}")
    print(f"oracle-aligned kappa (well-conditioned, |den|>{k['min_denominator']}, "
          f"{k['kept_fraction']:.0%} of events):")
    print(f"   MAE={k['mae']:.4f} rel.err={k['relative_error']:.4f} "
          f"r={k['pearson']:.4f} rho={k['spearman']:.4f}")
    print(f"   (all events: rho={k['all_events']['spearman']:.4f}, "
          f"rel.err={k['all_events']['relative_error']:.2f})")
    print(f"mechanical eta MAE  = {metrics['mechanical']['eta_mae']:.3e} (debug invariant)")
    print(f"trajectory overall  = {metrics['trajectory']['overall_mean_abs_dev']:.4f} mean abs dev")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point for the LLM-network experiment."""
    parser = argparse.ArgumentParser(description="BCA LLM-network experiment")
    parser.add_argument("--model-key", default=None,
                        help="registry model key (e.g. gpt-5.4, llama4-scout); "
                             "overrides --model and selects the endpoint")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="raw generator model id (legacy)")
    parser.add_argument("--appraiser-model", default=None, help="appraiser model (defaults to --model)")
    parser.add_argument("--n-agents", type=int, default=20)
    parser.add_argument("--n-rounds", type=int, default=10, help="pilot default 10; full 20")
    parser.add_argument("--n-seeds", type=int, default=3, help="pilot default 3; full 5")
    parser.add_argument("--kappas", type=float, nargs="+", default=list(DEFAULT_KAPPAS))
    parser.add_argument("--full", action="store_true", help="full run: n_rounds=20, n_seeds=5")
    parser.add_argument("--weight", type=float, default=1.0)
    parser.add_argument("--recovery-min-denominator", type=float, default=0.05,
                        help="skip oracle-aligned recovery events with |speaker-listener| below this")
    parser.add_argument("--concurrency", type=int, default=8,
                        help="parallel per-speaker LLM calls within a round")
    parser.add_argument("--calibration-json", type=Path, default=None, help="fitted appraiser calibration")
    parser.add_argument("--max-calls", type=int, default=None, help="hard cap on API calls")
    parser.add_argument("--dry-run", action="store_true", help="print call/token/cost estimate and exit")
    parser.add_argument("--resume", action="store_true", help="reuse the persistent cache to skip done calls")
    parser.add_argument("--cache", action="store_true", help="persist a prompt-keyed call cache")
    parser.add_argument("--price-per-1m-tokens", type=float, default=None, help="USD per 1M tokens for --dry-run")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="forgetting factor in (0,1]; 1.0 = static latent (default)")
    args = parser.parse_args(argv)

    n_rounds, n_seeds = (20, 5) if args.full else (args.n_rounds, args.n_seeds)
    appraiser_model = args.appraiser_model or args.model

    if args.dry_run:
        est = estimate_calls(args.kappas, n_seeds, n_rounds, args.n_agents, args.price_per_1m_tokens)
        print("LLM-network dry run")
        print(f"  runs             : {est['n_runs']}")
        print(f"  generator calls  : {est['n_generator_calls']}")
        print(f"  appraiser calls  : {est['n_appraiser_calls']}")
        print(f"  approx tokens    : {est['approx_tokens']:,}")
        if "approx_cost_usd" in est:
            print(f"  approx cost (USD): {est['approx_cost_usd']:.2f}")
        return 0

    # Build the real LLM-backed generator/appraiser only for a live run.
    from bca_beta import channel, models
    from bca_beta.llm import JSONCache

    spec = models.resolve_spec(model_key=args.model_key, model=args.model)
    appraiser_spec = (
        models.resolve_spec(model_key=None, model=args.appraiser_model)
        if args.appraiser_model else spec
    )
    out_dir = Path(args.out) if args.out != DEFAULT_OUT else models.default_output_root(spec) / "tier1b"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = None
    if args.cache or args.resume:
        cache = JSONCache(out_dir / "tier1b_cache.json")
        if args.resume:
            print(f"resume: {len(cache)} cached calls preloaded")

    # Load the selected model's calibration automatically when not given explicitly.
    cal_path = args.calibration_json or models.default_calibration_path(spec)
    if cal_path and Path(cal_path).exists():
        calibration = Calibration.load(cal_path)
        print(f"calibration: loaded {cal_path}")
    else:
        calibration = None
        print(f"WARNING: no calibration found for {spec.key} at {cal_path}; "
              f"running the appraiser UNCALIBRATED", flush=True)

    ch = channel.build_channel(spec, cache=cache, calibration=calibration,
                               max_calls=args.max_calls, appraiser_spec=appraiser_spec)
    provenance = channel.provenance_block(
        spec, appraiser_spec=appraiser_spec, gamma=args.gamma, calibration_path=cal_path,
    )

    metrics = run_tier1b_experiment(
        generator=ch.generator, appraiser=ch.appraiser, kappas=args.kappas,
        n_seeds=n_seeds, n_rounds=n_rounds, n_agents=args.n_agents, weight=args.weight,
        model=spec.model_id, appraiser_model=appraiser_spec.model_id, out_dir=out_dir,
        make_plots=not args.no_plot, min_denominator=args.recovery_min_denominator,
        max_workers=args.concurrency, cache=cache, gamma=args.gamma, provenance=provenance,
    )
    if cache is not None:
        cache.save()
    _print_metrics(metrics)
    print(f"wrote LLM-network outputs to {out_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
