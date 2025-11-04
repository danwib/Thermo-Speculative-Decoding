# SPDX-License-Identifier: Apache-2.0
"""ψ schema definitions and reference sampling utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(slots=True)
class PsiPayload:
    """Compact representation of proposer parameters.

    Args:
        ids: Top-K token identifiers sorted by descending proposer score.
        scores: Quantized proposer scores aligned with ``ids``.
        scale: De-quantization scale applied to ``scores``.
        zero_point: Zero-point offset for de-quantization.
        tau: Sampling temperature applied in the TSU.
        epsilon: Floor probability applied before normalization.
        mask: Optional vocabulary mask encoded as a bitset.
    """

    ids: np.ndarray
    scores: np.ndarray
    scale: np.float16
    zero_point: np.int8
    tau: np.float16
    epsilon: np.float16
    mask: Optional[np.ndarray] = None


def reference_sampler(payload: PsiPayload, rng: np.random.Generator) -> np.ndarray:
    """Sample from the distribution encoded by ``payload``.

    Args:
        payload: ψ payload emitted by the proposer.
        rng: Random number generator used for sampling.

    Returns:
        Sampled token identifiers.

    Raises:
        NotImplementedError: Until the reference sampler is implemented.
    """

    raise NotImplementedError("reference_sampler will be implemented in M0.")

