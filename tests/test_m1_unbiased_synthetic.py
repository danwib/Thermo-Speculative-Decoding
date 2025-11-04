# SPDX-License-Identifier: Apache-2.0
"""Unbiasedness test for M1 synthetic bigram with fixed contexts."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import chisquare

from tsd.targets.m1_bigram import craft_psi_from_row, make_bigram_synthetic, row_logprobs
from tsd.tsu_iface import SimTSU
from tsd.verifier import accept_correct_step


def _normalize_logp(logp_row: np.ndarray) -> np.ndarray:
    max_log = float(np.max(logp_row))
    shifted = logp_row - max_log
    log_total = max_log + np.log(np.sum(np.exp(shifted), dtype=np.float64))
    return logp_row - log_total


def test_m1_unbiased_fixed_contexts() -> None:
    """SimTSU + accept/correct reproduces p(y|x) when context is held fixed."""

    vocab = 64
    P, logP = make_bigram_synthetic(V=vocab, seed=13, c=40.0)
    contexts = [0, 1, 2, 3]
    samples_per_ctx = 20_000

    simulator = SimTSU()
    rng = np.random.Generator(np.random.PCG64(777))

    successes = 0

    for prev_id in contexts:
        logp_row = np.array(row_logprobs(logP, prev_id), copy=True)
        normalized_logp = _normalize_logp(logp_row)

        psi = craft_psi_from_row(logp_row=normalized_logp, K=32, tau=1.0, epsilon="auto")

        proposed, logq = simulator.sample_categorical(
            psi, batch_size=samples_per_ctx, seed=10_000 + prev_id
        )

        emitted = np.empty(samples_per_ctx, dtype=np.int32)
        for idx in range(samples_per_ctx):
            emitted_token, _ = accept_correct_step(
                logp=normalized_logp,
                psi=psi,
                proposed_token=int(proposed[idx]),
                proposed_logq=float(logq[idx]),
                rng=rng,
            )
            emitted[idx] = emitted_token

        counts = np.bincount(emitted, minlength=vocab)
        expected = P[prev_id] * samples_per_ctx

        chi2, pvalue = chisquare(counts, expected)
        if pvalue > 0.05:
            successes += 1

    assert successes >= 3  # allow one borderline context due to randomness
