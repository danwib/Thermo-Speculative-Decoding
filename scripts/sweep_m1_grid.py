# SPDX-License-Identifier: Apache-2.0
"""Grid sweep over K and τ for the M1 synthetic bigram target."""

from __future__ import annotations

import csv
import datetime as _dt
import sys
from itertools import product
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import typer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tsd.psi import logq_full
from tsd.targets.m1_bigram import craft_psi_from_row, make_bigram_synthetic
from tsd.tsu_iface import SimTSU
from tsd.verifier import accept_correct_step

app = typer.Typer(add_completion=False)


def _parse_eps(eps_opt: Optional[str]) -> Optional[Union[float, str]]:
    if eps_opt is None:
        return "auto"
    candidate = eps_opt.strip().lower()
    if candidate == "auto":
        return "auto"
    try:
        return float(candidate)
    except ValueError as exc:  # pragma: no cover - defensive
        raise typer.BadParameter("`--eps` must be a float or 'auto'") from exc


@app.command()
def sweep(
    vocab: int = typer.Option(256, "--vocab", min=2, help="Synthetic vocabulary size."),
    seeds: List[int] = typer.Option([7, 11], "--seeds", help="Base seeds to evaluate."),
    steps: int = typer.Option(15_000, "--steps", min=1, help="Samples per configuration."),
    Ks: List[int] = typer.Option(
        [32, 64, 128], "--Ks", help="Top-K sizes to explore.", show_default=True
    ),
    taus: List[float] = typer.Option(
        [0.9, 1.0, 1.1], "--taus", help="τ values to explore.", show_default=True
    ),
    eps: Optional[str] = typer.Option(
        "auto", "--eps", help="Tail floor mass ('auto' to match tail)."
    ),
    c: float = typer.Option(50.0, "--c", help="Synthetic Dirichlet coupling strength."),
) -> None:
    """Explore acceptance, overlap, and CE gap over (K, τ, seed) grid."""

    eps_val = _parse_eps(eps)
    Ks = [k for k in Ks if k > 0]
    if not Ks:
        raise typer.BadParameter("--Ks must contain positive integers.")
    taus = list(taus)
    if not taus:
        raise typer.BadParameter("--taus must contain at least one value.")

    runs_root = Path("runs")
    run_id = _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "m1_sweep.csv"

    header = ["K", "tau", "seed", "accept_rate", "mean_overlap", "mean_ce_gap"]
    typer.echo("\t".join(header))

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)

        for K, tau, seed in product(Ks, taus, seeds):
            K_eff = min(K, vocab)
            P, logP = make_bigram_synthetic(V=vocab, seed=seed, c=c)
            prev_id = 0
            logp_row = np.array(logP[prev_id], copy=True)
            psi = craft_psi_from_row(
                logp_row=logp_row,
                K=K_eff,
                tau=tau,
                epsilon=eps_val,
            )

            tsu = SimTSU()
            proposed, logq = tsu.sample_categorical(
                psi, batch_size=steps, seed=seed + 1
            )
            rng = np.random.Generator(np.random.PCG64(seed + 10_000))

            accept_count = 0
            for idx in range(steps):
                emitted_token, accepted = accept_correct_step(
                    logp=logp_row,
                    psi=psi,
                    proposed_token=int(proposed[idx]),
                    proposed_logq=float(logq[idx]),
                    rng=rng,
                )
                accept_count += int(accepted)

            accept_rate = accept_count / steps

            logq_all = logq_full(psi)
            q_row = np.exp(logq_all)
            p_row = np.exp(logp_row)

            overlap = float(np.minimum(p_row, q_row).sum())
            ce_gap = float(np.sum(p_row * (logp_row - logq_all), dtype=np.float64))

            row = [K_eff, tau, seed, accept_rate, overlap, ce_gap]
            typer.echo(
                f"{K_eff}\t{tau:.3f}\t{seed}\t{accept_rate:.4f}\t{overlap:.4f}\t{ce_gap:.4f}"
            )
            writer.writerow(row)

    typer.echo(f"csv_path={csv_path}")


if __name__ == "__main__":
    app()
