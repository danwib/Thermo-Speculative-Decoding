# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for the M1 contextual CLI."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict


def _run_cli(repo_root: Path, extra_args: list[str]) -> str:
    env = os.environ.copy()
    cmd = [sys.executable, "-m", "scripts.run_m1", "run", *extra_args]
    result = subprocess.run(
        cmd,
        cwd=repo_root,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def _parse_metrics(stdout: str) -> Dict[str, float]:
    pattern = re.compile(r"^([a-zA-Z0-9_]+)=([0-9.eE+-]+)$", re.MULTILINE)
    return {match.group(1): float(match.group(2)) for match in pattern.finditer(stdout)}


def test_m1_cli_smoke_synthetic() -> None:
    """Synthetic bigram run should emit stable telemetry metrics."""

    repo_root = Path(__file__).resolve().parents[1]
    stdout = _run_cli(
        repo_root,
        [
            "--vocab",
            "64",
            "--steps",
            "2000",
            "--seed",
            "11",
            "--K",
            "32",
            "--tau",
            "0.9",
        ],
    )
    metrics = _parse_metrics(stdout)
    for key in ("accept_rate", "mean_overlap", "mean_ce_gap", "chi2_stat", "p_value"):
        assert key in metrics
        assert metrics[key] == metrics[key]  # not NaN


def test_m1_cli_smoke_corpus(tmp_path: Path) -> None:
    """Corpus mode should run on tiny data sets."""

    repo_root = Path(__file__).resolve().parents[1]
    corpus_path = tmp_path / "toy_corpus.txt"
    corpus_path.write_text("abbaab\nabbaab\n", encoding="utf-8")

    stdout = _run_cli(
        repo_root,
        [
            "--corpus",
            str(corpus_path),
            "--steps",
            "1500",
            "--seed",
            "5",
            "--K",
            "8",
        ],
    )
    metrics = _parse_metrics(stdout)
    for key in ("accept_rate", "mean_overlap", "mean_ce_gap", "chi2_stat", "p_value"):
        assert key in metrics
        assert metrics[key] == metrics[key]
