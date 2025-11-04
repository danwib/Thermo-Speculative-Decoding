# SPDX-License-Identifier: Apache-2.0
"""Tests for bigram target construction utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tsd.targets.m1_bigram import (
    build_vocab_from_corpus,
    make_bigram_from_corpus,
    make_bigram_synthetic,
    row_logprobs,
)


def test_bigram_from_corpus(tmp_path: Path) -> None:
    """Corpus-derived bigram should normalise rows and expose vocab metadata."""

    corpus_path = tmp_path / "tiny.txt"
    corpus_path.write_text("ababa\n", encoding="utf-8")

    vocab = build_vocab_from_corpus(str(corpus_path))
    assert sorted(vocab) == vocab
    assert len(vocab) == 3  # "a", "b", "\n"

    P, logP, stoi, itos = make_bigram_from_corpus(str(corpus_path), laplace=1.0)
    assert itos == vocab
    assert len(stoi) == len(itos)
    assert P.shape == (len(itos), len(itos))
    assert logP.shape == P.shape

    row_sums = P.sum(axis=1)
    assert np.allclose(row_sums, 1.0)
    assert np.allclose(np.log(P), logP)

    row = row_logprobs(logP, prev_id=stoi["a"])
    assert row.shape == (len(itos),)
    assert np.isclose(np.exp(row).sum(), 1.0)


def test_bigram_synthetic_reproducible() -> None:
    """Synthetic bigram should be deterministic for identical seeds."""

    P1, logP1 = make_bigram_synthetic(V=4, seed=123, c=25.0)
    P2, logP2 = make_bigram_synthetic(V=4, seed=123, c=25.0)
    assert np.array_equal(P1, P2)
    assert np.array_equal(logP1, logP2)
    assert np.allclose(P1.sum(axis=1), 1.0)
    assert np.allclose(np.log(P1), logP1)
