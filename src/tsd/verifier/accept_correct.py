# SPDX-License-Identifier: Apache-2.0
"""Speculative accept/correct utilities."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def accept_correct(
    log_q: Sequence[float],
    log_p: Sequence[float],
    *,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Perform accept/correct on speculative proposals.

    Args:
        log_q: Log probabilities under the proposer distribution.
        log_p: Log probabilities under the verifier distribution.
        rng: Random number generator for acceptance tests.

    Returns:
        Tuple of accepted token mask and correction decisions.

    Raises:
        NotImplementedError: Until accept/correct is implemented.
    """

    raise NotImplementedError("accept_correct will be implemented in M0.")

