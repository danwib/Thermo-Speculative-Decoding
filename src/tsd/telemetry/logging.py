# SPDX-License-Identifier: Apache-2.0
"""Telemetry logging helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .metrics import TelemetryEvent


class TelemetryLogger:
    """Writes telemetry events to JSONL and CSV outputs."""

    def __init__(self, root: Path) -> None:
        """Create a logger rooted at ``root``.

        Args:
            root: Directory where telemetry files will be created.
        """

        self._root = root

    def write_events(self, events: Iterable[TelemetryEvent]) -> None:
        """Persist events to disk."""

        raise NotImplementedError("TelemetryLogger.write_events will be implemented in M0.")

