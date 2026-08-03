"""Fit the appraiser's temperature calibration on soft-labelled data.

Runs the stance appraiser on every labelled utterance, fits one temperature ``tau``
on the calibration split by minimising soft-label binary cross-entropy, and evaluates
the raw vs calibrated predictions on the held-out validation split.

The numerical core (:func:`compute_raw_logit`, :func:`fit_and_evaluate`) is pure and
unit-tested; only :func:`main` touches the OpenAI API.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from bca import calibration as cal

DEFAULT_LABELS = Path(__file__).resolve().parents[1] / "appraiser_labels.csv"
DEFAULT_OUT = Path(__file__).resolve().parent / "outputs" / "appraiser_calibration"
DEFAULT_MODEL = "gpt-5.4-mini"
APPR_TOKENS_PER_CALL = 160


def compute_raw_logit(df: pd.DataFrame, mode: str, eps: float = 1e-6) -> np.ndarray:
    """Compute the raw logit ``z`` per row for the chosen calibration mode."""
    if mode == "raw_prob":
        return cal.logit(df["p_plus_raw"].to_numpy(dtype=float), eps)
    if mode == "logit_diff":
        return df["logit_plus"].to_numpy(dtype=float) - df["logit_minus"].to_numpy(dtype=float)
    raise ValueError(f"unknown calibration mode {mode!r}")


def fit_and_evaluate(
    raw_df: pd.DataFrame,
    mode: str = "raw_prob",
    n_bins: int = 10,
) -> "tuple[cal.Calibration, dict, pd.DataFrame]":
    """Fit ``tau`` on the calibration split and evaluate raw vs calibrated predictions.

    Args:
        raw_df: Rows with ``split``, ``soft_label_p_plus``, and the appraiser output
            (``p_plus_raw``, or ``logit_plus``/``logit_minus`` for ``logit_diff`` mode).
        mode: ``"raw_prob"`` or ``"logit_diff"``.
        n_bins: ECE bins.

    Returns:
        ``(calibration, metrics, validation_predictions_df)``.
    """
    z = compute_raw_logit(raw_df, mode)
    y = raw_df["soft_label_p_plus"].to_numpy(dtype=float)
    is_cal = (raw_df["split"] == "calibration").to_numpy()
    is_val = (raw_df["split"] == "validation").to_numpy()

    tau = cal.fit_temperature(z[is_cal], y[is_cal])
    calibration = cal.Calibration(mode=mode, tau=tau)

    p_raw = cal.sigmoid(z)            # uncalibrated (tau = 1)
    p_calibrated = cal.sigmoid(z / tau)

    def split_metrics(mask: np.ndarray) -> dict:
        return {
            "raw": cal.calibration_metrics(y[mask], p_raw[mask], n_bins),
            "calibrated": cal.calibration_metrics(y[mask], p_calibrated[mask], n_bins),
        }

    metrics = {
        "mode": mode,
        "tau": tau,
        "n_calibration": int(is_cal.sum()),
        "n_validation": int(is_val.sum()),
        "validation": split_metrics(is_val),
        "calibration_split": split_metrics(is_cal),
    }

    val_pred = pd.DataFrame({
        "item_id": raw_df.loc[is_val, "item_id"].to_numpy(),
        "soft_label_p_plus": y[is_val],
        "p_raw": p_raw[is_val],
        "p_calibrated": p_calibrated[is_val],
    })
    return calibration, metrics, val_pred


def appraise_labels(df: pd.DataFrame, appraiser: Any) -> pd.DataFrame:
    """Run the appraiser on every labelled utterance; return a raw-outputs frame."""
    from bca.llm import token_counts

    rows = []
    for _, row in df.iterrows():
        result = appraiser.appraise(
            utterance=row["utterance"], a_plus=row["a_plus"], a_minus=row["a_minus"]
        )
        tc = token_counts(result.usage)
        rows.append({
            "item_id": row["item_id"],
            "split": row["split"],
            "concept_id": row["concept_id"],
            "utterance": row["utterance"],
            "soft_label_p_plus": float(row["soft_label_p_plus"]),
            "p_plus_raw": result.p_plus_raw,
            "rationale": result.rationale,
            "parse_error": result.parse_error,
            "prompt_tokens": tc["prompt_tokens"],
            "completion_tokens": tc["completion_tokens"],
            "total_tokens": tc["total_tokens"],
        })
    return pd.DataFrame(rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point: appraise the labels, fit tau, evaluate, and save."""
    parser = argparse.ArgumentParser(description="BCA appraiser temperature calibration")
    parser.add_argument("--model-key", default=None,
                        help="registry model key (e.g. gpt-5.4, llama4-scout); "
                             "overrides --model and selects the endpoint")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="raw appraiser model id (legacy)")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--mode", choices=("raw_prob", "logit_diff"), default="raw_prob")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-calls", type=int, default=None)
    parser.add_argument("--cache", action="store_true", help="persist a prompt-keyed call cache")
    parser.add_argument("--dry-run", action="store_true", help="print call/token estimate and exit")
    parser.add_argument("--price-per-1m-tokens", type=float, default=None)
    args = parser.parse_args(argv)

    labels = pd.read_csv(args.labels)

    if args.dry_run:
        n = len(labels)
        approx_tokens = n * APPR_TOKENS_PER_CALL
        print("appraiser calibration dry run")
        print(f"  appraiser calls : {n}")
        print(f"  approx tokens   : {approx_tokens:,}")
        if args.price_per_1m_tokens is not None:
            print(f"  approx cost(USD): {approx_tokens / 1e6 * args.price_per_1m_tokens:.2f}")
        return 0

    from bca import channel, models
    from bca.llm import JSONCache

    spec = models.resolve_spec(model_key=args.model_key, model=args.model)
    out_dir = Path(args.out) if args.out != DEFAULT_OUT else models.default_output_root(spec) / "appraiser_calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = JSONCache(out_dir / "cache.json") if args.cache else None
    # Fit on the *raw* appraiser output, so no calibration is applied here. The channel
    # points the appraiser at the selected model's endpoint with the centralized params.
    ch = channel.build_channel(spec, cache=cache, calibration=None, max_calls=args.max_calls)
    appraiser = ch.appraiser

    raw_df = appraise_labels(labels, appraiser)
    if cache is not None:
        cache.save()
    raw_df.to_csv(out_dir / "appraiser_calibration_raw_outputs.csv", index=False)

    calibration, metrics, val_pred = fit_and_evaluate(raw_df, mode=args.mode)
    metrics["model"] = spec.model_id
    metrics["provenance"] = channel.provenance_block(spec)
    if "total_tokens" in raw_df.columns:
        metrics["token_usage"] = {
            "total_tokens": int(raw_df["total_tokens"].sum()),
            "prompt_tokens": int(raw_df["prompt_tokens"].sum()),
            "completion_tokens": int(raw_df["completion_tokens"].sum()),
            "n_calls": int(len(raw_df)),
        }

    calibration.save(out_dir / "appraiser_calibration_tau.json")
    # Canonical per-model calibration file, auto-loaded by downstream runs.
    canonical = models.default_calibration_path(spec)
    canonical.parent.mkdir(parents=True, exist_ok=True)
    calibration.save(canonical)
    (out_dir / "appraiser_calibration_metrics.json").write_text(json.dumps(metrics, indent=2))
    val_pred.to_csv(out_dir / "appraiser_validation_predictions.csv", index=False)

    v = metrics["validation"]
    print(f"model = {spec.key} ({spec.model_id}) @ {spec.endpoint_label}")
    print(f"fitted tau = {calibration.tau:.4f} (mode={args.mode})")
    print(f"validation NLL   raw={v['raw']['nll']:.4f}  calibrated={v['calibrated']['nll']:.4f}")
    print(f"validation ECE   raw={v['raw']['ece']:.4f}  calibrated={v['calibrated']['ece']:.4f}")
    print(f"validation MAE   raw={v['raw']['mae']:.4f}  calibrated={v['calibrated']['mae']:.4f}")
    print(f"wrote calibration outputs to {out_dir}")
    print(f"canonical per-model calibration -> {canonical}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
