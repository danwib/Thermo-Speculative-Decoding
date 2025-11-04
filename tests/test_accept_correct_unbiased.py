# SPDX-License-Identifier: Apache-2.0
"""End-to-end unbiasedness test for accept/correct."""

from __future__ import annotations

import numpy as np
from scipy.stats import chisquare

from tsd.targets.m0_categorical import craft_psi_from_p, make_p
from tsd.tsu_iface import SimTSU
from tsd.verifier import accept_correct_step


def test_accept_correct_unbiased() -> None:
    """Accept/correct should reproduce the target distribution."""

    vocab_size = 1_000
    k = 16
    p, logp = make_p(vocab_size=vocab_size, seed=2024)
    psi = craft_psi_from_p(p, k=k, tau=0.8, epsilon=1e-8)

    simulator = SimTSU()
    num_samples = 30_000
    proposed_tokens, proposed_logq = simulator.sample_categorical(
        psi, batch_size=num_samples, seed=7
    )

    rng = np.random.default_rng(90210)
    outputs = np.empty(num_samples, dtype=np.int32)
    accept_mask = np.empty(num_samples, dtype=bool)

    for idx in range(num_samples):
        final_token, accepted = accept_correct_step(
            logp=logp,
            psi=psi,
            proposed_token=int(proposed_tokens[idx]),
            proposed_logq=float(proposed_logq[idx]),
            rng=rng,
        )
        outputs[idx] = final_token
        accept_mask[idx] = accepted

    counts = np.bincount(outputs, minlength=vocab_size)
    expected = p * num_samples

    chi2, pvalue = chisquare(counts, expected)
    assert not np.isnan(chi2)
    assert pvalue > 0.05

    accept_rate = float(np.mean(accept_mask))
    assert 0.05 <= accept_rate <= 0.95
    assert np.all(outputs >= 0)
    assert np.all(outputs < vocab_size)
