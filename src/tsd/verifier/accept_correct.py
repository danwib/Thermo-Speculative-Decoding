# SPDX-License-Identifier: Apache-2.0
"""Speculative accept/correct utilities."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ..psi import PsiTopK, logq_for_ids

__all__ = ["accept_correct_step"]


def _normalise_log_probs(logp: np.ndarray) -> Tuple[np.ndarray, float]:
    """Normalise log probabilities to avoid drift."""

    max_logp = np.max(logp)
    shifted = logp - max_logp
    sum_exp = np.sum(np.exp(shifted), dtype=np.float64)
    log_z = max_logp + np.log(sum_exp)
    return logp - log_z, log_z


def accept_correct_step(
    logp: np.ndarray,
    psi: PsiTopK,
    proposed_token: int,
    proposed_logq: float,
    rng: np.random.Generator,
) -> Tuple[int, bool]:
    """Perform a single-token accept/correct step.

    Parameters
    ----------
    logp:
        Target log probabilities of shape ``(V,)``.
    psi:
        ψ payload used to parameterise the proposer distribution.
    proposed_token:
        Token sampled by the TSU.
    proposed_logq:
        ``log q`` corresponding to the proposed token (**must** be the value
        returned by the TSU; do not recompute from ``ψ`` externally).
    rng:
        Random number generator for acceptance decisions and residual sampling.

    Returns
    -------
    Tuple[int, bool]
        Final token identifier and a flag indicating whether the proposal was
        accepted (`True`) or corrected (`False`).
    """

    logp = np.asarray(logp, dtype=np.float64)
    vocab_size = logp.shape[0]
    if proposed_token < 0 or proposed_token >= vocab_size:
        raise ValueError("proposed_token must lie within the vocabulary range.")

    normalized_logp, _ = _normalise_log_probs(logp)
    logp_token = normalized_logp[proposed_token]
    # Use the log q provided by the TSU. Recomputing it here risks diverging
    # from the hardware/software sampler contract.
    delta = logp_token - proposed_logq
    alpha = 1.0 if delta >= 0.0 else float(np.exp(delta))

    if rng.random() < alpha:
        return int(proposed_token), True

    all_ids = np.arange(vocab_size, dtype=np.int32)
    logq_all = logq_for_ids(psi, all_ids)
    q_all = np.exp(logq_all)
    p_all = np.exp(normalized_logp)

    residual = p_all - alpha * q_all
    residual = np.clip(residual, 0.0, None)
    residual_mass = float(np.sum(residual, dtype=np.float64))

    if residual_mass <= 0.0 or not np.isfinite(residual_mass):
        residual = p_all.copy()
        residual_mass = 1.0

    residual /= residual_mass

    final_token = rng.choice(all_ids, p=residual)
    return int(final_token), False
