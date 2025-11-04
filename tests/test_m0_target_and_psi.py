# SPDX-License-Identifier: Apache-2.0
"""Tests for M0 categorical target and ψ crafting."""

from __future__ import annotations

import numpy as np

from tsd.psi import psi_size_bytes
from tsd.targets.m0_categorical import craft_psi_from_p, make_p


def test_make_p_reproducible_and_normalised() -> None:
    """make_p should yield reproducible Dirichlet samples."""

    p1, logp1 = make_p(vocab_size=32, seed=123)
    p2, logp2 = make_p(vocab_size=32, seed=123)

    assert np.allclose(p1, p2)
    assert np.allclose(logp1, logp2)
    assert np.isclose(np.sum(p1), 1.0)
    assert np.all(p1 > 0.0)


def test_craft_psi_matches_topk_order_and_size() -> None:
    """Crafted ψ should preserve Top-K order and remain compact."""

    vocab_size = 256
    k = 64
    p, _ = make_p(vocab_size=vocab_size, seed=321)
    psi = craft_psi_from_p(p, k=k, tau=1.0, epsilon=1e-6)

    expected_ids = np.argsort(-p, kind="stable")[:k].astype(np.int32)
    assert np.array_equal(psi.ids, expected_ids)

    assert psi_size_bytes(psi) <= 1024
    assert int(psi.vocab_size) == vocab_size
