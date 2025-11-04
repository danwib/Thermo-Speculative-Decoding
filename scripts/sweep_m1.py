# SPDX-License-Identifier: Apache-2.0
"""Seed/steps sweep utility for M1 stability analysis."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import typer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts.run_m1 import run as run_m1


app = typer.Typer(add_completion=False)


@app.command()
def sweep(
    seeds: List[int] = typer.Option(
        [7, 11, 13, 17], "--seeds", help="Seeds to evaluate."
    ),
    steps: int = typer.Option(20_000, "--steps", min=1, help="Steps per run."),
    corpus: Optional[Path] = typer.Option(
        None, "--corpus", help="Corpus path for char-level bigram."
    ),
    vocab: int = typer.Option(256, "--vocab", min=2, help="Vocabulary when synthetic."),
    c: float = typer.Option(50.0, "--c", help="Synthetic Dirichlet coupling strength."),
    k: int = typer.Option(64, "--K", min=1),
    tau: float = typer.Option(1.0, "--tau"),
    eps: Optional[str] = typer.Option("auto", "--eps"),
) -> None:
    """Run M1 synthetic/corpus bigram across multiple seeds."""

    typer.echo("seed\taccept_rate\tmean_overlap\tmean_ce_gap")
    for seed in seeds:
        metrics = run_m1(
            corpus=corpus,
            vocab=vocab,
            c=c,
            seed=seed,
            K=k,
            tau=tau,
            eps=eps,
            steps=steps,
            return_metrics=True,
        )
        if metrics is None:
            continue
        typer.echo(
            f"{seed}\t{metrics['accept_rate']:.4f}\t{metrics['mean_overlap']:.4f}\t{metrics['mean_ce_gap']:.4f}"
        )


if __name__ == "__main__":
    app()
