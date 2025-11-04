# SPDX-License-Identifier: Apache-2.0
"""Entry point for the M0 smoke test."""

from __future__ import annotations

import datetime as _dt
import json
import sys
from pathlib import Path

import numpy as np
import typer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tsd.psi import psi_size_bytes
from tsd.targets import craft_psi_from_p, make_p
from tsd.telemetry.metrics import chi2_test_counts
from tsd.tsu_iface import SimTSU
from tsd.verifier import accept_correct_step

app = typer.Typer(add_completion=False)


@app.callback()
def main() -> None:
    """CLI bootstrap for M0 experiments."""


@app.command()
def run(
    vocab: int = typer.Option(1000, "--vocab", min=1, help="Vocabulary size."),
    k: int = typer.Option(64, "--K", min=1, help="Top-K payload size."),
    tau: float = typer.Option(1.0, "--tau", help="Proposer temperature."),
    eps: float = typer.Option(1e-6, "--eps", help="Out-of-set floor mass."),
    steps: int = typer.Option(100_000, "--steps", min=1, help="Number of samples."),
    seed: int = typer.Option(7, "--seed", help="Base seed for RNGs."),
) -> None:
    """Run the M0 smoke test with deterministic settings."""

    if k > vocab:
        raise typer.BadParameter("K must be <= vocab size.")

    p, logp = make_p(vocab_size=vocab, seed=seed)
    psi = craft_psi_from_p(p, k=k, tau=tau, epsilon=eps)
    psi_bytes = psi_size_bytes(psi)

    tsu = SimTSU()

    runs_root = Path("runs")
    run_id = _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = run_dir / "m0_metrics.jsonl"

    rng = np.random.default_rng(seed + 10_000)
    emitted_counts = np.zeros(vocab, dtype=np.int64)
    accept_count = 0

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for step in range(steps):
            batch_seed = seed + step
            tokens, logq = tsu.sample_categorical(psi, batch_size=1, seed=batch_seed)
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
                "eps": float(eps),
            }
            handle.write(json.dumps(row) + "\n")

    chi2_stat, p_value = chi2_test_counts(emitted_counts, p)
    accept_rate = accept_count / steps

    typer.echo(f"metrics_path={jsonl_path}")
    typer.echo(f"accept_rate={accept_rate:.4f}")
    typer.echo(f"chi2_stat={chi2_stat:.4f}")
    typer.echo(f"p_value={p_value:.4f}")
    typer.echo(f"psi_bytes_mean={psi_bytes:.1f}")


if __name__ == "__main__":
    app()
