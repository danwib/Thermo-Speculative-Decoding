# SPDX-License-Identifier: Apache-2.0
"""Golden ψ semantics tests."""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="Implement ψ sampler for M0.")
def test_reference_sampler_matches_simulator() -> None:
    """Ensure reference ψ sampler matches simulator outputs."""

