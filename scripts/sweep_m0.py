# SPDX-License-Identifier: Apache-2.0
"""Seed/steps sweep utility for M0 stability analysis."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import typer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts.run_m0 import run as run_m0


app = typer.Typer(add_completion=False)


@app.command()
def sweep(
    seeds: List[int] = typer.Option(
        [7, 11, 13, 17], "--seeds", help="Seeds to evaluate."
    ),
    steps: int = typer.Option(20_000, "--steps", min=1, help="Steps per run."),
    vocab: int = typer.Option(1000, "--vocab", min=1),
    k: int = typer.Option(64, "--K", min=1),
    tau: float = typer.Option(1.0, "--tau"),
    eps: Optional[str] = typer.Option("auto", "--eps"),
    pgen: str = typer.Option("dirichlet", "--pgen"),
    alpha: float = typer.Option(1.1, "--alpha"),
) -> None:
    """Run M0 for multiple seeds and print summary statistics."""

    typer.echo("seed\taccept_rate\toverlap_mass\tp_value")
    for seed in seeds:
        metrics = run_m0(
            vocab=vocab,
            k=k,
            tau=tau,
            eps=eps,
            steps=steps,
            seed=seed,
            pgen=pgen,
            alpha=alpha,
            return_metrics=True,
        )
        if metrics is None:
            continue
        typer.echo(
            f"{seed}\t{metrics['accept_rate']:.4f}\t{metrics['overlap_mass']:.4f}\t{metrics['p_value']:.4f}"
        )


if __name__ == "__main__":
    app()
