# SPDX-License-Identifier: Apache-2.0
"""Golden ψ contract tests ensuring sampler/verifier agreement."""

from __future__ import annotations

import numpy as np

from tsd.psi import PsiTopK, logq_for_ids
from tsd.targets import craft_psi_from_p, make_p
from tsd.tsu_iface import SimTSU


def _generate_random_psi(rng: np.random.Generator) -> tuple[PsiTopK, np.ndarray]:
    vocab_size = int(rng.integers(64, 512))
    k = int(rng.integers(4, min(128, vocab_size) + 1))
    tau = float(rng.uniform(0.5, 1.5))
    epsilon = float(rng.uniform(1e-6, 1e-3))

    p, _ = make_p(vocab_size=vocab_size, seed=int(rng.integers(0, 1 << 31)))
    psi = craft_psi_from_p(p, k=k, tau=tau, epsilon=epsilon)
    return psi, p


def test_simtsu_matches_reference_logq() -> None:
    rng = np.random.default_rng(1234)
    simulator = SimTSU()

    num_samples = 10_000

    for idx in range(50):
        psi, _ = _generate_random_psi(rng)
        tokens, logq = simulator.sample_categorical(
            psi, batch_size=num_samples, seed=idx + 2024
        )

        logq_recomputed = logq_for_ids(psi, tokens)
        assert np.array_equal(logq, logq_recomputed)

        counts = np.bincount(tokens, minlength=int(psi.vocab_size))
        inset_counts = counts[psi.ids]
        inset_mask = inset_counts > 0
        assert inset_mask.any()

        empirical_probs = inset_counts[inset_mask] / num_samples
        theoretical_probs = np.exp(logq_for_ids(psi, psi.ids[inset_mask]))
        assert np.allclose(empirical_probs, theoretical_probs, atol=0.02)
