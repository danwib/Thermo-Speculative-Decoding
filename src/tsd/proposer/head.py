# SPDX-License-Identifier: Apache-2.0
"""Proposer head stub emitting ψ payloads."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..psi import PsiPayload


class ProposerHead:
    """Maps context into ψ payloads."""

    def emit(self, context: Sequence[int], *, rng: np.random.Generator) -> PsiPayload:
        """Produce ψ for the given context.

        Args:
            context: Sequence of token identifiers provided by the verifier.
            rng: Random number generator for stochastic heads.

        Returns:
            ψ payload compatible with the TSU.

        Raises:
            NotImplementedError: Until proposer logic is implemented.
        """

        raise NotImplementedError("ProposerHead.emit will be implemented in M1.")

