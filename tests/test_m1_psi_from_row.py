# SPDX-License-Identifier: Apache-2.0
"""Tests for per-context ψ crafting from bigram rows."""

from __future__ import annotations

import numpy as np
import pytest

from tsd.psi import psi_size_bytes
from tsd.targets.m1_bigram import craft_psi_from_row


def _random_logp_row(vocab: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    probs = rng.dirichlet(alpha=np.ones(vocab, dtype=np.float64))
    return np.log(probs)


def test_craft_psi_auto_epsilon_and_size() -> None:
    """Auto epsilon matches tail mass and payload stays compact."""

    vocab = 200
    K = 64
    logp_row = _random_logp_row(vocab, seed=123)
    psi = craft_psi_from_row(logp_row, K=K, tau=0.9, epsilon="auto")

    assert psi.ids.shape[0] == K
    assert int(psi.vocab_size) == vocab
    assert psi_size_bytes(psi) <= 1024

    max_log = float(np.max(logp_row))
    shifted = logp_row - max_log
    log_total = max_log + np.log(np.sum(np.exp(shifted), dtype=np.float64))
    probs = np.exp(logp_row - log_total)

    topk = np.argsort(-probs, kind="stable")[:K]
    topk_mass = float(probs[topk].sum())
    tail_mass = max(0.0, 1.0 - topk_mass)
    denom = max(1, vocab - K)
    expected_eps = 0.0 if denom == 0 else tail_mass / denom

    assert float(psi.epsilon) == pytest.approx(expected_eps, rel=1e-3, abs=1e-6)


def test_topk_indices_preserve_order() -> None:
    """Chosen Top-K ids must match the highest probabilities."""

    vocab = 80
    K = 32
    logp_row = _random_logp_row(vocab, seed=321)
    psi = craft_psi_from_row(logp_row, K=K, tau=1.0)

    probs = np.exp(logp_row)
    expected_ids = np.argsort(-probs, kind="stable")[:K].astype(np.int32)
    assert np.array_equal(psi.ids, expected_ids)
