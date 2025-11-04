# SPDX-License-Identifier: Apache-2.0
"""Fixed categorical target helpers for milestone M0."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ..psi import PsiTopK

__all__ = ["make_p", "craft_psi_from_p", "entropy"]


def make_p(vocab_size: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Draw a categorical target distribution ``p``.

    Parameters
    ----------
    vocab_size:
        Vocabulary cardinality. Must be positive.
    seed:
        Random seed fed to a PCG64 generator.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        ``p`` and ``log p`` arrays, both float64 with shape ``(vocab_size,)``.

    Raises
    ------
    ValueError
        If ``vocab_size`` is not positive.
    """

    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive.")

    rng = np.random.Generator(np.random.PCG64(seed))
    concentration = np.ones(vocab_size, dtype=np.float64)
    p = rng.dirichlet(concentration).astype(np.float64)
    logp = np.log(p)
    return p, logp


def craft_psi_from_p(
    p: np.ndarray,
    k: int,
    tau: float = 1.0,
    epsilon: float = 1e-6,
) -> PsiTopK:
    """Construct a ψ payload aligned with the target distribution.

    Parameters
    ----------
    p:
        Target categorical distribution of shape ``(V,)``.
    k:
        Number of Top-K entries retained in ψ.
    tau:
        Temperature value copied into the payload.
    epsilon:
        Floor mass assigned to out-of-set tokens.

    Returns
    -------
    PsiTopK
        ψ payload whose Top-K ordering matches the target ordering.

    Raises
    ------
    ValueError
        If inputs are invalid or ``k`` exceeds the vocabulary size.
    """

    probs = np.asarray(p, dtype=np.float64)
    if probs.ndim != 1:
        raise ValueError("p must be a 1D array.")
    if not np.all(probs >= 0.0):
        raise ValueError("p must contain non-negative probabilities.")

    vocab_size = probs.shape[0]
    total_mass = np.sum(probs, dtype=np.float64)
    if not np.isfinite(total_mass) or total_mass <= 0.0:
        raise ValueError("p must sum to a positive finite value.")

    probs = probs / total_mass

    if k <= 0 or k > vocab_size:
        raise ValueError("k must be between 1 and vocab_size inclusive.")
    if tau <= 0.0:
        raise ValueError("tau must be positive.")
    if epsilon < 0.0:
        raise ValueError("epsilon must be non-negative.")

    sorted_indices = np.argsort(-probs, kind="stable")
    topk_indices = sorted_indices[:k].astype(np.int32)

    scores = np.log(probs[topk_indices])
    scores = scores.astype(np.float32)

    max_abs = float(np.max(np.abs(scores)))
    scale = np.float32(max(max_abs / 127.0, 1e-6))
    zero_point = np.int8(0)
    quantised = np.clip(
        np.round(scores / scale),
        -128,
        127,
    ).astype(np.int8)

    return PsiTopK(
        ids=topk_indices,
        scores_q8=quantised,
        scale=scale,
        zero_point=zero_point,
        tau=np.float16(tau),
        epsilon=np.float16(epsilon),
        vocab_size=np.int32(vocab_size),
    )


def entropy(p: np.ndarray) -> float:
    """Compute entropy of a categorical distribution in nats.

    Parameters
    ----------
    p:
        Probability vector of shape ``(V,)``. It will be normalised internally.

    Returns
    -------
    float
        Entropy value in natural units.

    Raises
    ------
    ValueError
        If ``p`` does not sum to a positive value.
    """

    probs = np.asarray(p, dtype=np.float64)
    if probs.ndim != 1:
        raise ValueError("p must be a 1D array.")
    total = np.sum(probs, dtype=np.float64)
    if total <= 0.0:
        raise ValueError("p must sum to a positive value.")

    probs = probs / total
    mask = probs > 0.0
    return float(-np.sum(probs[mask] * np.log(probs[mask]), dtype=np.float64))
