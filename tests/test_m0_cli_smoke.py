# SPDX-License-Identifier: Apache-2.0
"""Smoke test for the M0 CLI."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path


def test_m0_cli_smoke(tmp_path: Path) -> None:
    """Run a small M0 experiment and validate summary statistics."""

    env = os.environ.copy()
    # Ensure runs directory is created within the repository.
    repo_root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        "-m",
        "scripts.run_m0",
        "run",
        "--vocab",
        "500",
        "--K",
        "32",
        "--steps",
        "5000",
        "--seed",
        "11",
    ]
    result = subprocess.run(
        cmd,
        cwd=repo_root,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    stdout = result.stdout

    p_value_match = re.search(r"p_value=([0-9.eE+-]+)", stdout)
    assert p_value_match is not None
    p_value = float(p_value_match.group(1))
    assert p_value > 0.01

    accept_match = re.search(r"accept_rate=([0-9.]+)", stdout)
    assert accept_match is not None
    accept_rate = float(accept_match.group(1))
    assert 0.1 < accept_rate < 1.0
