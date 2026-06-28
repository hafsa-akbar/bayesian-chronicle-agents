from __future__ import annotations
import argparse
import json
from pathlib import Path

import pandas as pd

from bca_beta import analysis


def run_tier2(sliders_csv: str | Path, out_dir: str | Path, make_plots: bool = True) -> dict:
    """
    Read a combined slider log and compute belief↔slider correlation metrics.

    Writes:
    - belief_vs_slider.csv: input rows
    - metrics.json: overall and per-regime correlation metrics
    - reliability.csv: binned reliability table
    - belief_vs_slider.png (optional): scatter plot

    Args:
        sliders_csv: path to CSV with columns [regime, belief, slider_unit_scaled, ...]
        out_dir: directory to write output files
        make_plots: if True, generate scatter plot with Agg backend

    Returns:
        dict with keys 'overall' and 'by_regime' containing correlation metrics
    """
    df = pd.read_csv(sliders_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Compute metrics
    metrics = {
        "overall": analysis.belief_slider_correlation(df),
        "by_regime": {r: analysis.belief_slider_correlation(g)
                      for r, g in df.groupby("regime")} if "regime" in df else {}
    }

    # Write outputs
    df.to_csv(out_dir / "belief_vs_slider.csv", index=False)
    analysis.slider_reliability_bins(df).to_csv(out_dir / "reliability.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Optional plot
    if make_plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1, label="identity")
        ax.scatter(df["belief"], df["slider_unit_scaled"], s=8, alpha=0.3)
        ax.set_xlabel("internal belief b")
        ax.set_ylabel("expressed slider (0-1)")
        ax.set_title("belief-slider auditability")
        ax.legend()
        fig.savefig(out_dir / "belief_vs_slider.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    return metrics


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Belief-slider auditability: correlation from regime-run output"
    )
    parser.add_argument("--sliders", required=True, help="Path to combined slider CSV")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--no-plot", action="store_true", help="Skip plot generation")

    args = parser.parse_args(argv)
    run_tier2(args.sliders, args.out, make_plots=not args.no_plot)
    return 0


if __name__ == "__main__":
    exit(main())
