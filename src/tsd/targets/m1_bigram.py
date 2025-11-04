# SPDX-License-Identifier: Apache-2.0
"""Bigram target distribution for M1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np


@dataclass(slots=True)
class BigramTarget:
    """Bigram distribution conditioned on the previous token."""

    transitions: Dict[int, np.ndarray]

    def log_prob(self, prev_token: int, token: int) -> float:
        """Return log probability ``p(token | prev_token)``."""

        raise NotImplementedError("BigramTarget.log_prob will be implemented in M1.")


def bigram_target(
    transitions: Dict[int, Sequence[float]],
) -> BigramTarget:
    """Instantiate the M1 bigram target."""

    raise NotImplementedError("bigram_target will be implemented in M1.")

