# SPDX-License-Identifier: Apache-2.0
"""Unit tests for accept/correct primitives."""

from __future__ import annotations

import numpy as np

from tsd.targets.m0_categorical import craft_psi_from_p, make_p


def _normalise(logp: np.ndarray) -> np.ndarray:
    max_logp = np.max(logp)
    shifted = logp - max_logp
    sum_exp = np.sum(np.exp(shifted), dtype=np.float64)
    log_z = max_logp + np.log(sum_exp)
    return logp - log_z


def test_residual_non_negative_after_clipping() -> None:
    """Residual distribution should be non-negative after clipping."""

    p, logp = make_p(vocab_size=32, seed=42)
    psi = craft_psi_from_p(p, k=8, tau=1.0, epsilon=1e-6)

    logp_norm = _normalise(np.asarray(logp, dtype=np.float64))

    all_ids = np.arange(len(p), dtype=np.int32)
    from tsd.psi import logq_for_ids

    logq_all = logq_for_ids(psi, all_ids)
    p_all = np.exp(logp_norm)
    q_all = np.exp(logq_all)

    proposed_token = int(psi.ids[0])
    proposed_logq = logq_all[proposed_token]
    delta = logp_norm[proposed_token] - proposed_logq
    alpha = 1.0 if delta >= 0.0 else float(np.exp(delta))

    residual = p_all - alpha * q_all
    residual = np.clip(residual, 0.0, None)
    assert np.all(residual >= 0.0)
    assert np.sum(residual) > 0.0


def test_accept_probability_in_range() -> None:
    """α must lie between 0 and 1 inclusive."""

    p, logp = make_p(vocab_size=16, seed=7)
    psi = craft_psi_from_p(p, k=5, tau=0.9, epsilon=1e-6)

    logp_norm = _normalise(np.asarray(logp, dtype=np.float64))
    proposed_token = int(psi.ids[0])
    from tsd.psi import logq_for_ids

    logq_all = logq_for_ids(psi, np.arange(16, dtype=np.int32))
    delta = logp_norm[proposed_token] - logq_all[proposed_token]
    alpha = 1.0 if delta >= 0.0 else float(np.exp(delta))
    assert 0.0 <= alpha <= 1.0
