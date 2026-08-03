# Bayesian Chronicle Agents (Beta belief layer)

A small, inspectable belief-update layer for generative agents: *what* an agent
believes follows a transparent probabilistic rule; *how* it speaks is left to an LLM.
The experiments prescribe a stubbornness parameter κ, route all evidence through a
real generate–appraise language round-trip, and test whether κ and the classical
opinion-dynamics regimes (DeGroot, Friedkin–Johnsen, committed minority) survive the
channel.

## The model

Each agent holds a `Beta(α, β)` credence with mean `b = α/(α+β)`. Evidence `e ∈ [0,1]`
arrives with weight `w` and updates the belief as a convex blend

```
b ← (1 − η)·b + η·e ,    η = w / (n + w) ,    n = α + β
```

which is a Friedkin–Johnsen-style update with time-varying susceptibility η.
Two per-agent knobs control the dynamics:

- **κ — stubbornness**: prior strength `κ = α₀ + β₀`. High κ = stubborn, low κ = pliable.
- **γ — forgetting** `∈ (0, 1]`: how fast old evidence is discounted. `γ = 1` is exact
  conjugate Bayes (the population always converges); `γ < 1` keeps a permanent prior
  weight, giving FJ-style persistent disagreement. Reported runs use **γ = 0.7**.

In LLM runs, evidence flows through the language channel:
speaker belief → generated utterance → calibrated appraiser → evidence.

## Setup

```bash
git clone https://github.com/hafsa-akbar/bayesian-chronicle-agents.git
cd bayesian-chronicle-agents
pip install -r requirements.txt
```

Put API keys in a `.env` at the repo root (`OPENAI_API_KEY`, `HF_API_KEY`,
`ANTHROPIC_API_KEY`). Run all commands from the repo root; `python -m pytest` runs
the test suite.

## Reproducibility

The per-run aggregates behind every figure and table in the paper are committed under
`bca/experiments/outputs/`. With no
API key you can regenerate the figures and the γ-selection sweep:

```bash
python -m bca.experiments.gamma_sweep_oracle   # bare mechanism, no LLM — how γ = 0.7 was chosen
python -m bca.experiments.make_figures         # figures → bca/experiments/outputs/figures/
```

## Running the experiments

To re-run the pipeline through the live LLM channel, pick a model key (table below;
`gpt-5.4-mini` is cheapest) and run the steps in order. Every paid command accepts
`--dry-run` (prints a cost estimate and exits — run it first), plus `--cache`,
`--resume`, and `--max-calls`.

**1 — Calibrate the appraiser.** The appraiser turns each utterance into evidence: it
reads the text and states the probability `p₊` that it supports stance A⁺. LLM
appraisers are overconfident, so a single temperature τ (`p_cal = σ(logit(p₊)/τ)`) is
fit on `bca/appraiser_labels.csv` — 100 utterances with graded soft labels
`y ∈ [0,1]` for stance strength, split 80 fit / 20 held-out (~100 calls). The fit is
saved to `bca/experiments/calibration/<model_key>_tau.json` and auto-loaded
downstream; τ is model-specific, frozen before any simulation, independent of κ and γ.

```bash
python -m bca.experiments.calibrate_appraiser --model-key gpt-5.4-mini --cache
```

**2 — Classical regimes.** The three classical baselines — DeGroot consensus (κ → 0),
Friedkin–Johnsen persistent disagreement (heterogeneous κ), committed-minority
dose-response (frozen κ → ∞ minority) — each scored against its closed-form reference
(`classical.py`). Also logs the slider self-reports that step 4 audits.

```bash
python -m bca.experiments.classical_regimes --model-key gpt-5.4-mini --regime all \
  --gamma 0.7 --n-seeds 5 --n-rounds 20 --cache
```

**3 — κ recovery.** Prescribes κ ∈ {0.5, …, 32}, then recovers it from the observed
dynamics *oracle-aligned* — against the speaker's latent belief. The second (offline)
command inverts the exact one-step identity per logged event and writes the canonical
`exact_kappa_summary.csv` / `exact_recovery.json`; with the consumed evidence the
inversion recovers κ exactly, so oracle-aligned deviations are due to the channel alone.

```bash
python -m bca.experiments.kappa_recovery --model-key gpt-5.4-mini --gamma 0.7 \
  --n-seeds 5 --n-rounds 20 --cache
python -m bca.experiments.exact_kappa_reanalysis
```

**4 — Belief–expression audit.** Correlates each agent's independent 0–100 slider
self-report (from step 2) with its hidden belief, checking that what agents *say*
tracks the belief layer.

```bash
python -m bca.experiments.slider_audit --model-key gpt-5.4-mini \
  --sliders bca/experiments/outputs/gpt-5.4-mini/classical_regimes/0.7/sliders.csv \
  --out bca/experiments/outputs/gpt-5.4-mini/slider_audit/0.7
```

**5 — Figures.** Re-run `python -m bca.experiments.make_figures` on the fresh outputs.

Outputs land under `bca/experiments/outputs/<model_key>/<experiment>/<γ>/`; each run
records model, endpoint, and sampling params in `metrics.json` / `provenance.json`.

## Models

Select a model per run with `--model-key` (registry in `models.py`):

| key | endpoint | key env |
|---|---|---|
| `gpt-5.4-mini` | OpenAI | `OPENAI_API_KEY` |
| `gpt-5.4` | OpenAI | `OPENAI_API_KEY` |
| `llama4-scout` | HF router | `HF_API_KEY` |
| `claude-sonnet-4-6` | Anthropic (OpenAI-compat) | `ANTHROPIC_API_KEY` |

Sampling params are identical across models (`params.py`); adding a model is one
`ModelSpec` entry.

## Layout

```
bca/
  belief.py        BetaBelief: the update rule, κ and γ
  agent.py         Agent: one BetaBelief per concept
  engine.py        shared round-robin event loop
  classical.py     closed-form DeGroot / FJ / committed-minority references
  llm.py           LLM client + cache
  generators.py    belief → utterance
  appraisers.py    utterance → evidence (calibrated)
  probes.py        belief → 0–100 slider self-report
  calibration.py   appraiser temperature calibration
  analysis.py      recovery, alignment, and regime metrics + plots
  models.py        model registry
  channel.py       ModelSpec → (client, generator, appraiser, probe)
  experiments/     runnable experiments (see above) + committed outputs
```
