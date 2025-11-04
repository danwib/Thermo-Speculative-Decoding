# SPDX-License-Identifier: Apache-2.0
"""Utility to generate an M0 artifact bundle (metrics + summary)."""

from __future__ import annotations

import datetime as _dt
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tsd.psi import psi_size_bytes
from tsd.targets import craft_psi_from_p, make_p
from tsd.telemetry.metrics import chi2_test_counts
from tsd.tsu_iface import SimTSU
from tsd.verifier import accept_correct_step


def main() -> None:
    """Run a deterministic M0 experiment and persist artifacts."""

    vocab = 1_000
    k = 64
    tau = 1.0
    epsilon = 1e-6
    steps = 20_000
    seed = 13

    p, logp = make_p(vocab_size=vocab, seed=seed)
    psi = craft_psi_from_p(p, k=k, tau=tau, epsilon=epsilon)
    psi_bytes = psi_size_bytes(psi)
    simulator = SimTSU()

    runs_root = Path("runs")
    run_id = f"m0_artifact_{_dt.datetime.utcnow().strftime('%Y%m%dT%H%M%S')}"
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    telemetry_path = run_dir / "m0_metrics.jsonl"
    summary_path = run_dir / "summary.json"

    rng = np.random.default_rng(seed + 10_000)
    emitted_counts = np.zeros(vocab, dtype=np.int64)
    accept_count = 0

    with telemetry_path.open("w", encoding="utf-8") as handle:
        for step in range(steps):
            tokens, logq = simulator.sample_categorical(
                psi, batch_size=1, seed=seed + step
            )
            proposed = int(tokens[0])
            proposed_logq = float(logq[0])
            emitted, accepted = accept_correct_step(
                logp=logp,
                psi=psi,
                proposed_token=proposed,
                proposed_logq=proposed_logq,
                rng=rng,
            )
            emitted_counts[emitted] += 1
            accept_count += int(accepted)

            row = {
                "step": step,
                "proposed": proposed,
                "emitted": int(emitted),
                "accepted": bool(accepted),
                "psi_bytes": int(psi_bytes),
                "K": int(k),
                "tau": float(tau),
                "eps": float(epsilon),
            }
            handle.write(json.dumps(row) + "\n")

    chi2_stat, p_value = chi2_test_counts(emitted_counts, p)
    accept_rate = accept_count / steps

    summary = {
        "accept_rate": accept_rate,
        "chi2_stat": chi2_stat,
        "chi2_p": p_value,
        "psi_bytes_mean": float(psi_bytes),
        "steps": steps,
        "seed": seed,
        "vocab": vocab,
        "K": k,
    }

    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"artifact_dir={run_dir}")
    print(f"accept_rate={accept_rate:.4f}")
    print(f"chi2_p={p_value:.4f}")
    print(f"psi_bytes_mean={psi_bytes:.1f}")


if __name__ == "__main__":
    main()
