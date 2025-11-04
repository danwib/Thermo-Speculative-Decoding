# SPDX-License-Identifier: Apache-2.0
"""Acceptance correctness tests."""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="Implement accept/correct flow for M0.")
def test_accept_correct_unbiasedness() -> None:
    """Validate acceptance-correctness against the target distribution."""

