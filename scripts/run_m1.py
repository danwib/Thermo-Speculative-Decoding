# SPDX-License-Identifier: Apache-2.0
"""Entry point for the M1 bigram experiment."""

from __future__ import annotations

import typer


app = typer.Typer(add_completion=False)


@app.callback()
def main() -> None:
    """CLI bootstrap for M1 experiments."""


@app.command()
def run(seed: int = 0) -> None:
    """Run the M1 bigram experiment."""

    raise NotImplementedError("M1 run script will be implemented after proposer stubs land.")


if __name__ == "__main__":
    app()

