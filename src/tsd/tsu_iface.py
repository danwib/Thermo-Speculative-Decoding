# SPDX-License-Identifier: Apache-2.0
"""Adapters for Thermodynamic Sampling Units (TSU)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from .psi import PsiTopK


class TSUAdapter(Protocol):
    """Common interface for TSU implementations."""

    def sample(self, payload: PsiTopK, *, rng: np.random.Generator) -> "TSUResult":
        """Draw samples from the TSU.

        Args:
            payload: ψ payload describing the sampling distribution.
            rng: NumPy RNG for reproducibility.

        Returns:
            A ``TSUResult`` containing sampled tokens and associated ``log_q``.
        """


@dataclass(slots=True)
class TSUResult:
    """Container for TSU sampling results."""

    tokens: Sequence[int]
    log_q: Sequence[float]


@dataclass(slots=True)
class SimTSU:
    """Reference implementation backed by the ψ sampler."""

    max_block_length: int = 1

    def sample(self, payload: PsiTopK, *, rng: np.random.Generator) -> TSUResult:
        """Sample via the reference ψ sampler.

        Args:
            payload: ψ payload provided by the proposer.
            rng: Random number generator used for sampling.

        Returns:
            TSUResult with placeholder values.

        Raises:
            NotImplementedError: Until simulator logic is implemented.
        """

        raise NotImplementedError("SimTSU.sample will be implemented in M0.")
