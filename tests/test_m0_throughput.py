# SPDX-License-Identifier: Apache-2.0
"""Throughput regression guard for M0."""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="Benchmark harness pending M0 implementation.")
def test_m0_throughput_budget() -> None:
    """Track throughput metrics for the M0 pipeline."""

