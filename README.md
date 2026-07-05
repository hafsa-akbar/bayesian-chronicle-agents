# bayesian chronicle agents (bca) — controllable Beta belief-update layer

`bca` is a small, inspectable belief-update layer that sits between a generative
agent's persona and the text it produces, so that *what* an agent believes is governed
by a transparent probabilistic rule while *how* it speaks is left to a language model.

## Two knobs — κ (stubbornness) and γ (world-persistence/forgetfulness)

The belief layer has two per-agent knobs:

- **κ — prior strength / stubbornness**, fixed at initialization (`κ = α₀ + β₀`).
  High κ = stubborn, low κ = pliable. κ is *not* itself an FJ susceptibility; it
  **induces** one, `η_i(c) = w / (n_i + w)`, where `n_i = α_i + β_i` is the running
  pseudo-count and `c` the per-agent observation count.
- **γ — forgetting factor / world-persistence** `∈ (0, 1]`, how slowly old evidence is
  discounted. The update is
  `α ← α₀ + γ(α−α₀) + w·e`,  `β ← β₀ + γ(β−β₀) + w·(1−e)`.
  - **γ = 1 (static latent):** exact conjugate Bayes for an i.i.d. world. The count
    `n = κ + w·c` grows without bound, so `η(c) = w/(κ + w(c+1))` decays to 0 and the
    prior washes out — the population always converges (FJ persistent disagreement holds
    only transiently).
  - **γ < 1 (non-stationary latent):** the count saturates at `n* = κ + w/(1−γ)`, so η
    tends to a positive constant and the prior keeps a permanent weight
    `ρ* = κ(1−γ)/(κ(1−γ)+w)`. Steady state `b* = ρ*·b₀ + (1−ρ*)·ē` — classical FJ with 
    persistent disagreement.

The reported runs operate at a constant **γ = 0.7**; `γ = 1` reproduces the static-latent model
and serves as the no-forgetting ablation. Both LLM drivers accept `--gamma`.

## The mechanism

Each agent holds a `Beta(α, β)` credence with mean `b = α/(α+β)`. On hearing evidence
`(ℓ⁺, ℓ⁻)` it normalizes `e = ℓ⁺/(ℓ⁺+ℓ⁻)` and applies one conjugate step `α += w·e`,
`β += w·(1−e)`. Writing `n = α+β`, the mean follows the exact convex blend

```
b(c+1) = (1 − η(c))·b(c) + η(c)·e ,     η(c) = w / (n + w) .
```

That is an FJ-style update with a time-varying susceptibility η(c). See the *Polar
Opinion Dynamics* paper for the FJ model (`x(t+1) = A W x(t) + (I−A) x(0)`; diagonal of
`A` = susceptibility, `I−A` = stubbornness) and the *BCA redesign* document (§2.4 "The
update", §2.5 "Why κ behaves like Friedkin–Johnsen stubbornness", proofs in §6–§8).

All runs use one concept (`transit_priority`: "expand rail" vs "keep roads") and `N = 20`
agents on a complete graph. The model is selected per run with `--model-key` (registry in
`models.py`; see [Running on multiple models](#running-on-multiple-models)) — the same
experiments run unchanged on `gpt-5.4-mini`, `gpt-5.4`, and `llama4-scout`. API keys are
read from `.env` (via `python-dotenv`, never logged).

## Appraiser calibration

`experiments/tier1a_fit_appraiser_calibration.py` runs the appraiser on 100 soft-labelled
utterances (`tier1a_appraiser_labels.csv`), fits one temperature τ on the 80-example
calibration split by minimising soft-label BCE, and evaluates on the 20-example validation
split. τ is model-specific (live fits: **1.89** for `gpt-5.4-mini`, **2.15** for
`gpt-5.4`, **1.51** for `llama4-scout`; all overconfident) — for `gpt-5.4-mini` it
improves validation ECE 0.100 → 0.052 and NLL 0.568 → 0.493. Each fit is saved to
`experiments/calibration/<model_key>_tau.json` and auto-loaded by every downstream run
of that model; it is independent of κ and γ.

```bash
python -m bca_beta.experiments.tier1a_fit_appraiser_calibration --model-key gpt-5.4-mini --cache
```

## The LLM language channel and κ recovery

`experiments/tier1b_llm_network.py` routes evidence through the real language channel

```
speaker belief  →  generated utterance  →  calibrated appraiser  →  evidence
```

and asks whether the κ-controlled dynamics survive. The main result is non-tautological:
recovery is measured *oracle-aligned* — using the latent `speaker_belief_snapshot`, not the
appraised evidence the Beta update consumed — alongside the alignment between appraised
evidence and latent belief. (Recovering with the consumed evidence is exact by construction
and kept only as a `mechanical` debug invariant.) Per round: snapshot speaker beliefs,
generate one utterance per speaker, appraise each once, broadcast the appraised evidence to
all listeners. The κ-recovery inversion is γ-aware (it uses the saturating count
`n_c = κ + w(1−γ^c)/(1−γ)`, reducing to `w(1/η−(c+1))` at γ = 1).

```bash
python -m bca_beta.experiments.tier1b_llm_network --model-key gpt-5.4-mini --gamma 0.7 --cache
# cost controls: --dry-run, --max-calls, --cache, --resume, --price-per-1m-tokens
```

At γ = 0.7 (5 seeds × 20 rounds) the appraiser tracks the latent belief faithfully
(Pearson ≈ 0.96) and κ is recovered oracle-aligned with Spearman ≈ 0.96 / relative error
≈ 0.28, with ≈ 76% of events well-conditioned (vs ≈ 12% at γ = 1, where the population
converges). Per-event κ estimates
are heavy-tailed (inverting η amplifies appraiser error when listener ≈ speaker), so the
headline uses the per-condition median; `tier1b_metrics.json` keeps the full transparency
set (`condition_median_recovery`, `per_event_mae`, `n_skipped_near_zero_denominator`,
trajectory deviation, alignment MAE) and the real API token usage from `response.usage`.

Outputs (`experiments/outputs/<model_key>/tier1b/`): `tier1b_events.csv`, `tier1b_utterances.csv`,
`tier1b_metrics.json`, `tier1b_kappa_summary.csv`, `tier1b_alignment_by_bin.csv`, and
`plot_appraised_vs_speaker_belief.png` / `plot_oracle_aligned_kappa_recovery.png` /
`plot_trajectory_deviation.png`.

## Classical regimes

`experiments/tier1_5_regimes.py` runs three classical baselines through the LLM channel,
each a setting of (κ, γ):

- **DeGroot consensus** (κ → 0, pliable): all agents converge to a shared opinion.
- **Friedkin–Johnsen persistent disagreement** (heterogeneous κ, γ < 1): pliable and
  stubborn agents settle into a stable spread determined by their initial opinions and κ.
- **Committed-minority dose-response** (κ → ∞ frozen agents; free-agent κ = 4): a frozen
  minority pulls the free majority, and the effect (mean final belief of free agents vs
  committed fraction f) is a **smooth, monotone, threshold-free** dose-response — a
  deliberate contrast to the sharp ~25% critical-mass tipping of the empirical literature
  (Centola 2018), which this affine update cannot and should not reproduce.

Each regime is judged against its closed-form reference in `classical.py`; `engine.py` is
the shared round-robin loop. A per-round slider probe logs an independent expressed opinion
per agent for the auditability check.

```bash
python -m bca_beta.experiments.tier1_5_regimes --model-key gpt-5.4-mini --regime all \
  --gamma 0.7 --n-seeds 5 --n-rounds 20 --cache
```

Outputs (`experiments/outputs/<model_key>/tier1_5/`): regime-wise events and metrics;
`sliders.csv` with per-agent-per-round slider responses.

## Auditability

`experiments/tier2_auditability.py` correlates the slider probe (independent expressed
opinion) against the agent's hidden belief at each round, checking that expression is a
faithful signal of the belief-update dynamics rather than an artifact of generation or
appraisal (Pearson ≈ 0.99).

```bash
python -m bca_beta.experiments.tier2_auditability --model-key gpt-5.4-mini \
  --sliders bca_beta/experiments/outputs/gpt-5.4-mini/tier1_5/sliders.csv --out <dir>
```

## Choosing γ — the forgetting-factor sweep

`experiments/gamma_sweep_oracle.py` runs only the belief mechanism (oracle evidence, no
language model, so it is free) across the three regimes and a grid of γ — how we pick γ
before spending on the LLM channel:

```bash
python -m bca_beta.experiments.gamma_sweep_oracle   # gammas {1.0,…,0.5}, writes outputs/gamma_oracle/
```

The signature finding: at γ = 1 the FJ population converges (final variance ≈ 1e-4); as
soon as γ < 1 persistent disagreement appears and grows (≈ 0.056 at γ = 0.7, saturating
around γ ≈ 0.7), while DeGroot stays at consensus for every γ.

## Package layout

```
bca_beta/
  belief.py        BetaBelief: mean, update(ℓ⁺, ℓ⁻), eta(c), forgetting factor γ
  agent.py         Agent: one BetaBelief per concept; generator/appraiser slots
  calibration.py   temperature calibration (fit τ, metrics, save/load)
  llm.py           LLMClient protocol, OpenAIClient (lazy .env key), JSONCache
  generators.py    OpenAIUtteranceGenerator (belief → utterance) + protocol
  appraisers.py    OpenAIStanceAppraiser (utterance → p_plus, calibrated) + protocol
  probes.py        OpenAISliderProbe (belief → 0–100 slider self-report) + protocol
  engine.py        shared round-robin event loop + initial-belief helper
  analysis.py      γ-aware κ/η recovery, alignment, trajectory, regime + slider metrics, plots
  classical.py     closed-form DeGroot / FJ / committed-minority (Prop 3) references
  models.py        model registry: KEY → {model_id, base_url, api_key_env, extra_body}
  params.py        centralized sampling policy (role → temperature), identical per model
  channel.py       one wiring path: ModelSpec → (client, generator, appraiser, probe)
  experiments/
    tier1a_fit_appraiser_calibration.py  fit the appraiser temperature
    tier1b_llm_network.py              LLM-channel κ-recovery experiment
    tier1_5_regimes.py                 classical-regime arc (DeGroot/FJ/committed)
    tier2_auditability.py              belief–expression audit
    gamma_sweep_oracle.py              free (no-API) γ sweep — the γ-selection artifact
    calibration/<model_key>_tau.json     per-model appraiser fit (τ); downstream runs auto-load it
  tier1a_appraiser_labels.csv          100 soft-labelled calibration utterances
  tests/           unit tests (all LLM calls mocked — no API in tests)
```

## Running on multiple models

All experiments select a model by a short **key** with `--model-key`, resolved through the
registry in `models.py`:

| key | model_id | endpoint | key env |
|---|---|---|---|
| `gpt-5.4-mini` | `gpt-5.4-mini` | default OpenAI | `OPENAI_API_KEY` |
| `gpt-5.4` | `gpt-5.4` | default OpenAI | `OPENAI_API_KEY` |
| `llama4-scout` | `meta-llama/Llama-4-Scout-17B-16E-Instruct` | `https://router.huggingface.co/v1` | `HF_API_KEY` |

Adding a model is a single `ModelSpec` entry in `models.REGISTRY`. The sampling params are
**identical across models** — the generator runs at temperature 0.9 and the appraiser/probe
at 0.0 for *every* model (centralized in `params.py`); switching the key only changes the
model id and endpoint. Anything an endpoint requires (e.g. the HF router's `max_tokens`) or
rejects is carried on the spec and applied identically, with a printed warning if a param
can't be honored. Each run records the model key, resolved id, endpoint, and exact sampling
params in its `metrics.json` (`provenance`). The legacy `--model` / `--appraiser-model`
(raw ids, default OpenAI endpoint) still work when `--model-key` is omitted.

**Per-model calibration.** τ is model-specific, so re-fit Tier 1a per model; it is saved to
`experiments/calibration/<model_key>_tau.json` and auto-loaded by downstream runs of the
same key (override with `--calibration-json`).

**Output layout.** With `--model-key` and no explicit `--out`, each experiment writes under a
per-model root:

```
experiments/
  calibration/<model_key>_tau.json          # per-model appraiser calibration
  outputs/<model_key>/
    tier1a/                                  # calibration fit artifacts
    tier1b/    | tier1b_gamma/<γ>/           # κ-recovery (metrics.json carries provenance)
    tier1_5/   | tier1_5_gamma/<γ>/          # DeGroot / FJ / committed (+ provenance.json)
    tier2_gamma/<γ>/                         # belief–slider audit
```

### Running the cross-model comparison

Run the same arc once per key. **Fit calibration first** (each is a paid ~100-call
run), then the three-regime arc at the operating point γ = 0.7. Every command prints a
`--dry-run` estimate first — inspect it and only then drop `--dry-run`.

```bash
CAL=bca_beta/experiments/calibration            # per-model τ lands here automatically
for KEY in gpt-5.4-mini gpt-5.4 llama4-scout; do
  # 1) appraiser calibration (per model)  -> $CAL/${KEY}_tau.json
  python -m bca_beta.experiments.tier1a_fit_appraiser_calibration --model-key $KEY --cache --dry-run

  # 2) three opinion-dynamics regimes at the operating point (auto-loads $CAL/${KEY}_tau.json)
  python -m bca_beta.experiments.tier1_5_regimes --model-key $KEY --regime all \
    --gamma 0.7 --n-seeds 5 --n-rounds 20 --concurrency 8 --cache --dry-run

  # 3) (optional) κ-recovery sweep, same model
  python -m bca_beta.experiments.tier1b_llm_network --model-key $KEY \
    --gamma 0.7 --kappas 0.5 1 2 4 8 16 32 --n-seeds 5 --n-rounds 20 --concurrency 8 --cache --dry-run
done
```

Drop `--dry-run` per command once the estimate is approved. `llama4-scout` reads `HF_API_KEY`
and routes through the HF OpenAI-compatible endpoint automatically — no other change.

## Running the tests

```bash
python -m pytest bca_beta        # no network access (all LLM calls mocked)
```