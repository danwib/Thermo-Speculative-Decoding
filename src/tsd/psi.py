# SPDX-License-Identifier: Apache-2.0
"""ψ schema definitions and reference sampling utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


__all__ = [
    "PsiTopK",
    "dequantize_scores",
    "logq_for_ids",
    "logq_full",
    "psi_size_bytes",
]


@dataclass(slots=True)
class PsiTopK:
    """Compact ψ payload for Top-K sampling.

    Attributes
    ----------
    ids:
        Array of token identifiers sorted by descending proposer score.
    scores_q8:
        Quantised int8 proposer scores aligned with ``ids``.
    scale:
        Quantisation scale applied during de-quantisation.
    zero_point:
        Quantisation zero-point for ``scores_q8``.
    tau:
        Sampling temperature applied to the de-quantised scores.
    epsilon:
        Floor probability mass allocated to out-of-set tokens.
    vocab_size:
        Size of the full vocabulary used by the sampler.
    """

    ids: np.ndarray
    scores_q8: np.ndarray
    scale: np.float32
    zero_point: np.int8
    tau: np.float16
    epsilon: np.float16
    vocab_size: np.int32

    def __post_init__(self) -> None:
        """Validate shapes and normalise dtypes."""

        self.ids = np.asarray(self.ids, dtype=np.int32)
        self.scores_q8 = np.asarray(self.scores_q8, dtype=np.int8)

        if self.ids.ndim != 1:
            raise ValueError("ids must be a 1D array.")
        if self.scores_q8.ndim != 1:
            raise ValueError("scores_q8 must be a 1D array.")
        if self.ids.shape[0] != self.scores_q8.shape[0]:
            raise ValueError("ids and scores_q8 must have matching length.")

        self.scale = np.float32(self.scale)
        if self.scale <= 0:
            raise ValueError("scale must be positive.")

        self.zero_point = np.int8(self.zero_point)

        self.tau = np.float16(self.tau)
        if float(self.tau) <= 0.0:
            raise ValueError("tau must be greater than 0.")

        self.epsilon = np.float16(self.epsilon)
        if float(self.epsilon) < 0.0:
            raise ValueError("epsilon must be non-negative.")

        self.vocab_size = np.int32(self.vocab_size)
        if int(self.vocab_size) < self.ids.shape[0]:
            raise ValueError("vocab_size must be at least as large as K.")


def dequantize_scores(psi: PsiTopK) -> np.ndarray:
    """De-quantise proposer scores.

    Parameters
    ----------
    psi:
        ψ payload with quantised scores.

    Returns
    -------
    np.ndarray
        Array of shape ``(K,)`` containing float32 scores.
    """

    scores = psi.scores_q8.astype(np.float32)
    zero_point = np.float32(psi.zero_point)
    scale = np.float32(psi.scale)
    return (scores - zero_point) * scale


def _logq_components(psi: PsiTopK) -> Tuple[np.ndarray, float, float, float, int]:
    scores = dequantize_scores(psi).astype(np.float64)
    tau = float(np.float64(psi.tau))
    epsilon = float(np.float64(psi.epsilon))
    vocab_size = int(psi.vocab_size)
    k = psi.ids.shape[0]

    scaled_scores = scores / tau
    max_scaled = np.max(scaled_scores) if scaled_scores.size else 0.0
    sum_exp = np.sum(np.exp(scaled_scores - max_scaled), dtype=np.float64)
    log_topk_sum = max_scaled + np.log(sum_exp)

    tail_count = vocab_size - k
    tail_mass = tail_count * epsilon
    if tail_count > 0 and tail_mass < 0:
        raise ValueError("Tail mass cannot be negative.")

    if tail_mass > 0:
        log_tail_mass = np.log(tail_mass)
        log_z = np.logaddexp(log_topk_sum, log_tail_mass)
        log_tail_prob = np.log(epsilon) - log_z
    else:
        log_z = log_topk_sum
        log_tail_prob = -np.inf

    return scaled_scores, log_z, log_tail_prob, epsilon, vocab_size


def logq_for_ids(psi: PsiTopK, query_ids: np.ndarray) -> np.ndarray:
    """Compute ``log q`` for requested token identifiers.

    Parameters
    ----------
    psi:
        ψ payload containing quantised Top-K scores.
    query_ids:
        Array of token identifiers to evaluate. Duplicates are permitted.

    Returns
    -------
    np.ndarray
        Array matching ``query_ids.shape`` with log-probabilities in float64.

    Raises
    ------
    ValueError
        If ``query_ids`` contains identifiers outside ``[0, vocab_size)``.
    """

    query = np.asarray(query_ids, dtype=np.int64)
    if np.any((query < 0) | (query >= int(psi.vocab_size))):
        raise ValueError("query_ids must lie within the vocabulary range.")

    scaled_scores, log_z, log_tail_prob, _, _ = _logq_components(psi)
    id_map: Dict[int, int] = {int(token_id): idx for idx, token_id in enumerate(psi.ids.tolist())}

    log_probs = np.full(query.shape, log_tail_prob, dtype=np.float64)
    for position, token_id in enumerate(query.tolist()):
        match = id_map.get(int(token_id))
        if match is not None:
            log_probs[position] = scaled_scores[match] - log_z

    return log_probs


def logq_full(psi: PsiTopK) -> np.ndarray:
    """Return log ``q`` values for the full vocabulary."""

    scaled_scores, log_z, log_tail_prob, _, vocab_size = _logq_components(psi)
    log_probs = np.full(vocab_size, log_tail_prob, dtype=np.float64)
    log_probs[psi.ids] = scaled_scores - log_z
    return log_probs


def psi_size_bytes(psi: PsiTopK) -> int:
    """Estimate the size of ``ψ`` in bytes.

    Parameters
    ----------
    psi:
        ψ payload to size.

    Returns
    -------
    int
        Total number of bytes required to serialise ``psi``.
    """

    size = int(psi.ids.nbytes + psi.scores_q8.nbytes)
    size += np.dtype(np.float32).itemsize  # scale
    size += np.dtype(np.int8).itemsize  # zero_point
    size += np.dtype(np.float16).itemsize * 2  # tau and epsilon
    size += np.dtype(np.int32).itemsize  # vocab_size
    return size
