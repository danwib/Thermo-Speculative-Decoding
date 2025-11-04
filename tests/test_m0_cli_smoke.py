# SPDX-License-Identifier: Apache-2.0
"""Smoke test for the M0 CLI."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List


def _run_cli(repo_root: Path, extra_args: List[str]) -> str:
    env = os.environ.copy()
    cmd = [sys.executable, "-m", "scripts.run_m0", "run", *extra_args]
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


def _parse_metric(stdout: str, name: str) -> float:
    match = re.search(rf"{name}=([0-9.eE+-]+)", stdout)
    assert match is not None, f"{name} missing in output"
    return float(match.group(1))


def test_m0_cli_smoke(tmp_path: Path) -> None:
    """Run M0 CLI in numeric and auto epsilon modes."""

    repo_root = Path(__file__).resolve().parents[1]

    stdout_numeric = _run_cli(
        repo_root,
        [
            "--vocab",
            "500",
            "--K",
            "32",
            "--steps",
            "4000",
            "--seed",
            "11",
            "--eps",
            "1e-6",
        ],
    )
    p_value_numeric = _parse_metric(stdout_numeric, "p_value")
    accept_numeric = _parse_metric(stdout_numeric, "accept_rate")
    assert p_value_numeric > 0.01
    assert 0.1 < accept_numeric < 1.0

    stdout_auto = _run_cli(
        repo_root,
        [
            "--vocab",
            "500",
            "--K",
            "32",
            "--steps",
            "4000",
            "--seed",
            "11",
            "--eps",
            "auto",
        ],
    )
    p_value_auto = _parse_metric(stdout_auto, "p_value")
    accept_auto = _parse_metric(stdout_auto, "accept_rate")
    eps_used_auto = _parse_metric(stdout_auto, "eps_used")
    overlap_auto = _parse_metric(stdout_auto, "overlap_mass")

    assert 0.0 <= p_value_auto <= 1.0
    assert 0.1 < accept_auto < 1.0
    assert accept_auto >= accept_numeric
    assert eps_used_auto >= 0.0
    assert 0.0 <= overlap_auto <= 1.0

    stdout_zipf = _run_cli(
        repo_root,
        [
            "--vocab",
            "400",
            "--K",
            "24",
            "--steps",
            "3000",
            "--seed",
            "19",
            "--pgen",
            "zipf",
            "--alpha",
            "1.2",
        ],
    )
    overlap_zipf = _parse_metric(stdout_zipf, "overlap_mass")
    assert 0.0 <= overlap_zipf <= 1.0
