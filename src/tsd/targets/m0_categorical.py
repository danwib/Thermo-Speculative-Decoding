# SPDX-License-Identifier: Apache-2.0
"""Fixed categorical target distribution for M0."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(slots=True)
class CategoricalTarget:
    """Static categorical distribution with explicit probabilities."""

    logits: np.ndarray

    def log_prob(self, token: int) -> float:
        """Return log probability for ``token``.

        Args:
            token: Token identifier.

        Raises:
            NotImplementedError: Until the target is implemented.
        """

        raise NotImplementedError("CategoricalTarget.log_prob will be implemented in M0.")


def categorical_target(logits: Sequence[float]) -> CategoricalTarget:
    """Instantiate the M0 categorical target."""

    raise NotImplementedError("categorical_target will be implemented in M0.")

