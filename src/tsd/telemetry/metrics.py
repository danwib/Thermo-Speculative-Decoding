# SPDX-License-Identifier: Apache-2.0
"""Telemetry schema definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.stats import chisquare


@dataclass(slots=True)
class TelemetryEvent:
    """Event emitted by TSD runs."""

    run_id: str
    step: int
    mode: str
    psi_bytes: int
    tsu_latency_ms: float
    verifier_latency_ms: float
    accepted_prefix: int
    first_token_reject: bool
    ce_gap_nats: float
    q_temp: float
    q_floor: float
    version_hash: str
    psi_semver: str
    sampler_semver: str
    seed: Optional[int] = None
    L: Optional[int] = None
    B: Optional[int] = None


def chi2_test_counts(emp_counts: np.ndarray, p: np.ndarray) -> Tuple[float, float]:
    """Perform a χ² goodness-of-fit test between empirical counts and ``p``.

    Parameters
    ----------
    emp_counts:
        Observed counts for each vocabulary item.
    p:
        Target distribution probabilities. Will be normalised internally.

    Returns
    -------
    Tuple[float, float]
        χ² statistic and associated p-value.
    """

    counts = np.asarray(emp_counts, dtype=np.float64)
    probs = np.asarray(p, dtype=np.float64)
    if counts.shape != probs.shape:
        raise ValueError("emp_counts and p must share the same shape.")

    total = np.sum(counts)
    if total <= 0.0:
        raise ValueError("emp_counts must sum to a positive value.")

    probs = probs / np.sum(probs)
    expected = probs * total
    stat, pvalue = chisquare(counts, expected)
    return float(stat), float(pvalue)
