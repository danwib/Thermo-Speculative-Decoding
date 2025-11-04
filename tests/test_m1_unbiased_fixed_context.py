# SPDX-License-Identifier: Apache-2.0
"""Fixed-context unbiasedness test for the M1 synthetic bigram target."""

from __future__ import annotations

import numpy as np
from scipy.stats import chisquare

from tsd.targets.m1_bigram import craft_psi_from_row, make_bigram_synthetic
from tsd.tsu_iface import SimTSU
from tsd.verifier import accept_correct_step


def test_m1_unbiased_fixed_context() -> None:
    """Chi-square checks per-context unbiasedness for the synthetic bigram."""

    vocab = 128
    P, logP = make_bigram_synthetic(V=vocab, seed=13)
    contexts = [0, 1, 2, 3]
    samples_per_ctx = 20_000

    tsu = SimTSU()
    rng = np.random.Generator(np.random.PCG64(4242))

    successes = 0

    for ctx in contexts:
        logp_row = np.array(logP[ctx], copy=True)
        psi = craft_psi_from_row(
            logp_row=logp_row,
            K=64,
            tau=1.0,
            epsilon="auto",
        )

        proposed, logq = tsu.sample_categorical(
            psi,
            batch_size=samples_per_ctx,
            seed=50_000 + ctx,
        )

        emitted = np.empty(samples_per_ctx, dtype=np.int32)
        for idx in range(samples_per_ctx):
            emitted_token, _ = accept_correct_step(
                logp=logp_row,
                psi=psi,
                proposed_token=int(proposed[idx]),
                proposed_logq=float(logq[idx]),
                rng=rng,
            )
            emitted[idx] = emitted_token

        counts = np.bincount(emitted, minlength=vocab)
        expected = P[ctx] * samples_per_ctx

        chi2, pvalue = chisquare(counts, expected)
        if pvalue > 0.05:
            successes += 1

    assert successes >= 3
