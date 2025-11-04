# SPDX-License-Identifier: Apache-2.0
"""Golden ψ semantics tests."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pytest

from tsd.psi import PsiTopK, dequantize_scores, logq_for_ids, logq_full


VOCAB_SIZE = 16
TOPK_IDS = np.array([1, 4, 7, 10], dtype=np.int32)
BASE_SCORES = np.array([0.42, 0.18, -0.05, -0.32], dtype=np.float32)
SCALE = np.float32(0.01)
ZERO_POINT = np.int8(0)


def build_payload(tau: float, epsilon: float) -> Tuple[PsiTopK, np.ndarray]:
    """Helper to construct a ψ payload for tests."""

    quantised = np.clip(
        np.rint(BASE_SCORES / SCALE) + ZERO_POINT,
        -128,
        127,
    ).astype(np.int8)
    psi = PsiTopK(
        ids=TOPK_IDS,
        scores_q8=quantised,
        scale=SCALE,
        zero_point=ZERO_POINT,
        tau=np.float16(tau),
        epsilon=np.float16(epsilon),
        vocab_size=np.int32(VOCAB_SIZE),
    )
    return psi, BASE_SCORES


def test_probs_sum_to_one() -> None:
    """The induced distribution normalises across the full vocabulary."""

    psi, _ = build_payload(tau=1.2, epsilon=1e-3)
    logq = logq_for_ids(psi, np.arange(VOCAB_SIZE, dtype=np.int32))
    probs = np.exp(logq)
    assert np.isclose(np.sum(probs), 1.0, rtol=1e-6, atol=1e-9)


def test_tau_monotonicity() -> None:
    """Increasing tau flattens the distribution (entropy grows)."""

    psi_cold, _ = build_payload(tau=0.6, epsilon=1e-3)
    psi_hot, _ = build_payload(tau=2.0, epsilon=1e-3)

    full_vocab = np.arange(VOCAB_SIZE, dtype=np.int32)
    logq_cold = logq_for_ids(psi_cold, full_vocab)
    logq_hot = logq_for_ids(psi_hot, full_vocab)

    probs_cold = np.exp(logq_cold)
    probs_hot = np.exp(logq_hot)

    entropy_cold = -np.sum(probs_cold * logq_cold)
    entropy_hot = -np.sum(probs_hot * logq_hot)

    assert entropy_hot >= entropy_cold


def test_epsilon_tail_mass() -> None:
    """Larger epsilon increases out-of-set probability mass."""

    psi_small, _ = build_payload(tau=1.0, epsilon=1e-6)
    psi_large, _ = build_payload(tau=1.0, epsilon=5e-3)

    vocab = np.arange(VOCAB_SIZE, dtype=np.int32)
    mask = ~np.isin(vocab, TOPK_IDS)

    tail_mass_small = np.sum(np.exp(logq_for_ids(psi_small, vocab))[mask])
    tail_mass_large = np.sum(np.exp(logq_for_ids(psi_large, vocab))[mask])

    assert tail_mass_large > tail_mass_small


def test_quant_roundtrip() -> None:
    """De-quantisation reconstructs proposer scores within tolerance."""

    psi, original_scores = build_payload(tau=1.0, epsilon=1e-3)
    reconstructed = dequantize_scores(psi)
    assert np.allclose(reconstructed, original_scores, atol=float(SCALE))


def test_logq_full_matches_logq_for_ids() -> None:
    """logq_full should agree with logq_for_ids over the full vocabulary."""

    psi, _ = build_payload(tau=0.9, epsilon=2e-3)
    logq_all = logq_full(psi)
    logq_enum = logq_for_ids(psi, np.arange(VOCAB_SIZE, dtype=np.int32))
    assert np.allclose(logq_all, logq_enum)
