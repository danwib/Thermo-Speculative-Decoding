# SPDX-License-Identifier: Apache-2.0
"""Telemetry schema definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class TelemetryEvent:
    """Event emitted by TSD runs."""

    run_id: str
    step: int
    mode: str
    psi_bytes: int
    tsu_latency_ms: float
    verifier_latency_ms: float
    accepted_prefix: int
    first_token_reject: bool
    ce_gap_nats: float
    q_temp: float
    q_floor: float
    version_hash: str
    psi_semver: str
    sampler_semver: str
    seed: Optional[int] = None
    L: Optional[int] = None
    B: Optional[int] = None

