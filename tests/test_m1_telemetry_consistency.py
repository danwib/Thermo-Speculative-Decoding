# SPDX-License-Identifier: Apache-2.0
"""Telemetry consistency checks for M1 synthetic bigram."""

from __future__ import annotations

import numpy as np

from tsd.psi import logq_full
from tsd.targets.m1_bigram import craft_psi_from_row, make_bigram_synthetic, row_logprobs
from tsd.tsu_iface import SimTSU
from tsd.verifier import accept_correct_step


def _normalise_logp(logp_row: np.ndarray) -> np.ndarray:
    max_log = float(np.max(logp_row))
    shifted = logp_row - max_log
    log_total = max_log + np.log(np.sum(np.exp(shifted), dtype=np.float64))
    return logp_row - log_total


def test_acceptance_matches_overlap_and_ce_gap() -> None:
    """Mean acceptance should track overlap and CE gap remain finite."""

    vocab = 128
    steps = 15_000
    P, logP = make_bigram_synthetic(V=vocab, seed=21, c=60.0)

    prev_id = 5
    logp_row = np.array(row_logprobs(logP, prev_id), copy=True)
    normalized_logp = _normalise_logp(logp_row)
    p_row = np.exp(normalized_logp)

    psi = craft_psi_from_row(normalized_logp, K=64, tau=0.95, epsilon="auto")
    logq_all = logq_full(psi)
    q_row = np.exp(logq_all)

    overlap_mass = float(np.minimum(p_row, q_row).sum())
    ce_gap = float(np.sum(p_row * (normalized_logp - logq_all), dtype=np.float64))

    tsu = SimTSU()
    rng = np.random.Generator(np.random.PCG64(90210))

    acceptances = np.empty(steps, dtype=np.float64)
    overlaps = np.full(steps, overlap_mass, dtype=np.float64)
    ce_gaps = np.full(steps, ce_gap, dtype=np.float64)

    for step in range(steps):
        tokens, logq = tsu.sample_categorical(psi, batch_size=1, seed=4000 + step)
        proposed_token = int(tokens[0])
        proposed_logq = float(logq[0])

        _, accepted = accept_correct_step(
            logp=normalized_logp,
            psi=psi,
            proposed_token=proposed_token,
            proposed_logq=proposed_logq,
            rng=rng,
        )

        acceptances[step] = float(accepted)

    accept_mean = float(np.mean(acceptances))
    overlap_mean = float(np.mean(overlaps))
    ce_gap_mean = float(np.mean(ce_gaps))

    assert abs(accept_mean - overlap_mean) < 0.03
    assert np.isfinite(ce_gap_mean)
    assert ce_gap_mean >= -1e-9
