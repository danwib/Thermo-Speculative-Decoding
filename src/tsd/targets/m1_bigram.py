# SPDX-License-Identifier: Apache-2.0
"""Bigram target utilities for milestone M1."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

__all__ = [
    "build_vocab_from_corpus",
    "make_bigram_from_corpus",
    "make_bigram_synthetic",
    "row_logprobs",
]


def build_vocab_from_corpus(path: str) -> List[str]:
    """Return a sorted character vocabulary extracted from ``path``.

    Parameters
    ----------
    path:
        File path to a plain-text corpus.

    Returns
    -------
    list[str]
        Sorted list of unique characters.
    """

    corpus_path = Path(path)
    text = corpus_path.read_text(encoding="utf-8")
    unique_chars = sorted(set(text))
    if not unique_chars:
        raise ValueError("Corpus must contain at least one character.")
    return unique_chars


def make_bigram_from_corpus(
    path: str,
    laplace: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int], List[str]]:
    """Build a bigram matrix from a character-level corpus.

    Parameters
    ----------
    path:
        File path to the corpus data.
    laplace:
        Additive smoothing mass added to every transition count.

    Returns
    -------
    tuple
        ``(P, logP, stoi, itos)`` where ``P`` is the conditional probability
        matrix, ``logP`` its natural logarithm, ``stoi`` maps characters to
        indices, and ``itos`` is the inverse mapping.
    """

    if laplace < 0.0:
        raise ValueError("laplace must be non-negative.")

    corpus_path = Path(path)
    text = corpus_path.read_text(encoding="utf-8")
    if len(text) == 0:
        raise ValueError("Corpus must contain at least one character.")

    itos = build_vocab_from_corpus(path)
    stoi = {ch: idx for idx, ch in enumerate(itos)}
    vocab_size = len(itos)

    counts = np.zeros((vocab_size, vocab_size), dtype=np.float64)
    if len(text) >= 2:
        prev_char = text[0]
        for current_char in text[1:]:
            prev_idx = stoi[prev_char]
            curr_idx = stoi[current_char]
            counts[prev_idx, curr_idx] += 1.0
            prev_char = current_char

    if laplace > 0.0:
        counts += laplace
    elif not np.all(counts.sum(axis=1) > 0.0):
        raise ValueError(
            "laplace must be positive when some characters lack outgoing counts."
        )

    row_sums = counts.sum(axis=1, keepdims=True)
    if np.any(row_sums == 0.0):
        raise ValueError("Encountered a row with zero total mass after smoothing.")

    P = counts / row_sums
    logP = np.log(P)
    return P, logP, stoi, itos


def make_bigram_synthetic(
    V: int,
    seed: int = 7,
    c: float = 50.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample a synthetic bigram distribution with Dirichlet structure.

    Parameters
    ----------
    V:
        Vocabulary size.
    seed:
        RNG seed used for reproducibility.
    c:
        Coupling strength between the stationary unigram and each row.

    Returns
    -------
    tuple
        ``(P, logP)`` where rows of ``P`` sum to 1 and ``logP`` is ``np.log(P)``.
    """

    if V <= 0:
        raise ValueError("Vocabulary size must be positive.")
    if c < 0.0:
        raise ValueError("c must be non-negative.")

    rng = np.random.default_rng(seed)
    base_alpha = np.ones(V, dtype=np.float64)
    unigram = rng.dirichlet(base_alpha)

    P = np.empty((V, V), dtype=np.float64)
    row_alpha = 1.0 + c * unigram
    for token_idx in range(V):
        P[token_idx] = rng.dirichlet(row_alpha)

    logP = np.log(P)
    return P, logP


def row_logprobs(logP: np.ndarray, prev_id: int) -> np.ndarray:
    """Return a view of the conditional log-probabilities for ``prev_id``."""

    if logP.ndim != 2:
        raise ValueError("logP must be a 2D array.")
    if not (0 <= prev_id < logP.shape[0]):
        raise ValueError("prev_id out of range for provided logP.")
    return logP[prev_id]
