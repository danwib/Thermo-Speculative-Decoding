# SPDX-License-Identifier: Apache-2.0
"""Tests for the software TSU simulator."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import chisquare

from tsd.psi import PsiTopK, logq_for_ids
from tsd.tsu_iface import SimTSU


def _build_random_psi(seed: int) -> PsiTopK:
    rng = np.random.default_rng(seed)
    vocab_size = 32
    k = 6

    ids = np.sort(rng.choice(vocab_size, size=k, replace=False).astype(np.int32))
    scores = rng.normal(loc=0.0, scale=0.5, size=k).astype(np.float32)
    scale = np.float32(0.02)
    quantised = np.clip(
        np.rint(scores / scale),
        -128,
        127,
    ).astype(np.int8)

    return PsiTopK(
        ids=ids,
        scores_q8=quantised,
        scale=scale,
        zero_point=np.int8(0),
        tau=np.float16(1.1),
        epsilon=np.float16(1e-3),
        vocab_size=np.int32(vocab_size),
    )


def test_sim_tsu_matches_reference_distribution() -> None:
    """Empirical frequencies drawn from SimTSU must match F(ψ)."""

    psi = _build_random_psi(seed=1234)
    simulator = SimTSU()

    num_samples = 50_000
    tokens, logq = simulator.sample_categorical(psi, batch_size=num_samples, seed=2024)

    all_ids = np.arange(int(psi.vocab_size), dtype=np.int32)
    logq_all = logq_for_ids(psi, all_ids)
    probs = np.exp(logq_all)

    counts = np.bincount(tokens, minlength=int(psi.vocab_size))
    expected = probs * num_samples

    chi2, pvalue = chisquare(counts, expected)

    assert not np.isnan(chi2)
    assert pvalue > 0.05

    logq_expected = logq_for_ids(psi, tokens)
    assert np.allclose(logq, logq_expected, atol=1e-10)

