# SPDX-License-Identifier: Apache-2.0
"""Proposer state management stubs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ProposerState:
    """Container for proposer recurrent state.

    Args:
        payload: Placeholder for learnable state tensors.
    """

    payload: Any = None

    def reset(self) -> None:
        """Reset the proposer state.

        Raises:
            NotImplementedError: Until state reset is implemented.
        """

        raise NotImplementedError("ProposerState.reset will be implemented in M1.")

