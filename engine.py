"""Shared round-robin belief-update engine.

Each round, every agent hears each neighbour once and applies one Beta update.
Reused across the calibrated-appraiser, committed-agent, and slider-probe runs.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import networkx as nx
import numpy as np

from bca_beta.agent import Agent
from bca_beta.llm import token_counts


def build_initial_means(
    n_agents: int,
    seed: int,
    low: float = 0.1,
    high: float = 0.9,
    jitter: float = 0.02,
) -> np.ndarray:
    """Starting credences spread across ``[low, high]`` with small seeded jitter.

    The jitter breaks exact symmetry so no two agents start identical, while keeping
    a real spread of opinions to move.
    """
    rng = np.random.default_rng(seed)
    base = np.linspace(low, high, n_agents)
    perturbed = base + rng.uniform(-jitter, jitter, size=n_agents)
    return np.clip(perturbed, 0.05, 0.95)


@dataclass
class RoundRobinResult:
    """Output of one round-robin simulation."""

    events: list[dict]
    utterances: list[dict]
    sliders: list[dict]
    belief_by_round: np.ndarray  # (n_rounds + 1, n_agents)
    agent_order: list


def run_round_robin(
    *,
    agents: Sequence[Agent],
    graph: nx.Graph,
    rng: np.random.Generator,
    n_rounds: int,
    weight: float,
    generator,
    appraiser,
    slider_probe=None,
    a_plus: str,
    a_minus: str,
    concept_id: str,
    run_meta: dict,
    evidence_mode: str = "appraised",
    max_workers: int = 1,
) -> RoundRobinResult:
    """Run one round-robin and return full event/utterance logs.

    Every emitted row begins with ``**run_meta`` so the caller controls which
    run-level fields appear (e.g. ``run_id``, ``seed``, ``model``).

    Args:
        agents: Sequence of :class:`~bca_beta.agent.Agent` objects.
        graph: Network topology; neighbours of a listener are its speakers.
        rng: Seeded RNG for the pair-update permutation.
        n_rounds: Number of full rounds to simulate.
        weight: Evidence weight per observation.
        generator: Utterance generator (protocol: ``generate(...)``).
        appraiser: Stance appraiser (protocol: ``appraise(...)``).
        slider_probe: Optional slider probe (protocol: ``probe(...)``). When
            given, each agent is probed once per round and a row is appended to
            ``sliders`` (including committed agents); when ``None`` no sliders
            are logged.
        a_plus: ``A+`` stance label.
        a_minus: ``A-`` stance label.
        concept_id: Concept node key for belief lookups.
        run_meta: Dict of run-level metadata prepended to every row.
        evidence_mode: ``"appraised"`` (language channel) or ``"oracle"``
            (speaker snapshot passed directly as evidence).
        max_workers: Parallel per-speaker LLM calls within a round.

    Returns:
        :class:`RoundRobinResult` with events, utterances, sliders (one row per
        agent per round when ``slider_probe`` is given, else empty), belief
        trajectory, and agent order.
    """
    agent_map = {a.agent_id: a for a in agents}
    order = [a.agent_id for a in agents]
    events: list[dict] = []
    utterances: list[dict] = []
    sliders: list[dict] = []
    belief_by_round = np.empty((n_rounds + 1, len(order)), dtype=float)
    c_counts = {aid: 0 for aid in order}
    event_id = 0

    for r in range(n_rounds):
        snapshot = {aid: agent_map[aid].credence(concept_id) for aid in order}
        belief_by_round[r] = [snapshot[aid] for aid in order]

        def _slider_reading(aid):
            if slider_probe is None:
                return None
            return slider_probe.probe(belief=snapshot[aid], speaker_id=aid, round_index=r,
                                      concept_id=concept_id, a_plus=a_plus, a_minus=a_minus)

        def _speaker_bundle(aid):
            uid = f"{run_meta.get('run_id', 0)}:r{r}:s{aid}"
            if evidence_mode == "oracle":
                b = snapshot[aid]
                return dict(utterance_id=uid, text="", raw=b, cal=b, rationale="oracle",
                            parse_error=False, gen_usage=None, appr_usage=None,
                            slider_reading=_slider_reading(aid))
            gen = generator.generate(belief=snapshot[aid], speaker_id=aid, round_index=r,
                                     concept_id=concept_id, a_plus=a_plus, a_minus=a_minus)
            appr = appraiser.appraise(utterance=gen.text, a_plus=a_plus, a_minus=a_minus)
            return dict(utterance_id=uid, text=gen.text, raw=appr.p_plus_raw,
                        cal=appr.p_plus_calibrated, rationale=appr.rationale,
                        parse_error=appr.parse_error, gen_usage=gen.usage, appr_usage=appr.usage,
                        slider_reading=_slider_reading(aid))

        if evidence_mode == "appraised" and max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                computed = dict(zip(order, pool.map(_speaker_bundle, order)))
        else:
            computed = {aid: _speaker_bundle(aid) for aid in order}

        bundle = {}
        for aid in order:
            b = bundle[aid] = computed[aid]
            gtc, atc = token_counts(b["gen_usage"]), token_counts(b["appr_usage"])
            utterances.append({**run_meta, "round_index": r, "speaker_id": aid,
                               "speaker_belief_snapshot": snapshot[aid], "utterance_text": b["text"],
                               "appraised_evidence_raw": b["raw"],
                               "appraised_evidence_calibrated": b["cal"],
                               "appraiser_rationale": b["rationale"],
                               "abs_error_vs_speaker_belief": abs(b["cal"] - snapshot[aid]),
                               "parse_error": b["parse_error"],
                               "generator_prompt_tokens": gtc["prompt_tokens"],
                               "generator_completion_tokens": gtc["completion_tokens"],
                               "generator_total_tokens": gtc["total_tokens"],
                               "appraiser_prompt_tokens": atc["prompt_tokens"],
                               "appraiser_completion_tokens": atc["completion_tokens"],
                               "appraiser_total_tokens": atc["total_tokens"]})
            reading = b["slider_reading"]
            if reading is not None:
                sliders.append({**run_meta, "round_index": r, "agent_id": aid,
                                "committed": agent_map[aid].committed, "belief": snapshot[aid],
                                "slider_value": reading.value,
                                "slider_unit_scaled": reading.value / 100.0,
                                "parse_error": reading.parse_error})

        pairs = [(l, s) for l in order if not agent_map[l].committed for s in graph.neighbors(l)]
        for k in rng.permutation(len(pairs)):
            listener, spk = pairs[int(k)]
            b = bundle[spk]
            evidence_used = b["cal"]
            belief = agent_map[listener].belief_for(concept_id)
            c_before = c_counts[listener]
            a_before, be_before, bel_before = belief.alpha, belief.beta, belief.mean
            # The susceptibility that governs THIS update, read from the live state
            # before applying it: eta = w / (n + w). Exact for any gamma; at gamma=1
            # it equals the closed form w / (kappa + w (c_before + 1)).
            eta_pred = belief.eta()
            bel_after = agent_map[listener].apply_evidence(
                evidence_pos=evidence_used, evidence_neg=1.0 - evidence_used, concept=concept_id)
            events.append({**run_meta, "round_index": r, "event_id": event_id,
                           "listener_id": listener, "speaker_id": spk, "concept_id": concept_id,
                           "c_before": c_before, "listener_belief_before": bel_before,
                           "speaker_belief_snapshot": snapshot[spk],
                           "utterance_id": b["utterance_id"], "utterance_text": b["text"],
                           "appraised_evidence_raw": b["raw"],
                           "appraised_evidence_calibrated": b["cal"],
                           "evidence_used_for_update": evidence_used, "eta_predicted": eta_pred,
                           "listener_alpha_before": a_before, "listener_beta_before": be_before,
                           "listener_alpha_after": belief.alpha, "listener_beta_after": belief.beta,
                           "listener_belief_after": bel_after, "parse_error": b["parse_error"],
                           "weight": weight, "gamma": belief.gamma})
            c_counts[listener] += 1
            event_id += 1

    belief_by_round[n_rounds] = [agent_map[aid].credence(concept_id) for aid in order]
    return RoundRobinResult(events, utterances, sliders, belief_by_round, order)
