# SPDX-License-Identifier: Apache-2.0
"""Slow end-to-end test for milestone M0."""

from __future__ import annotations

import numpy as np
import pytest

from tsd.targets import craft_psi_from_p, make_p
from tsd.telemetry.metrics import chi2_test_counts
from tsd.tsu_iface import SimTSU
from tsd.verifier import accept_correct_step


@pytest.mark.slow
def test_m0_end_to_end_unbiased() -> None:
    """Full pipeline smoke test at realistic scale."""

    vocab_size = 1_000
    k = 64
    steps = 30_000
    seed = 13

    p, logp = make_p(vocab_size=vocab_size, seed=seed)
    psi = craft_psi_from_p(p, k=k, tau=1.0, epsilon=1e-6)
    simulator = SimTSU()

    proposed_tokens, proposed_logq = simulator.sample_categorical(
        psi, batch_size=steps, seed=seed + 1
    )

    rng = np.random.default_rng(seed + 10_000)
    emitted = np.empty(steps, dtype=np.int32)
    accepted = np.empty(steps, dtype=bool)

    for idx in range(steps):
        token = int(proposed_tokens[idx])
        logq = float(proposed_logq[idx])
        out_token, is_accepted = accept_correct_step(
            logp=logp,
            psi=psi,
            proposed_token=token,
            proposed_logq=logq,
            rng=rng,
        )
        emitted[idx] = out_token
        accepted[idx] = is_accepted

    counts = np.bincount(emitted, minlength=vocab_size)
    chi2_stat, p_value = chi2_test_counts(counts, p)
    accept_rate = float(np.mean(accepted))

    assert p_value > 0.05
    assert 0.2 <= accept_rate <= 0.95
