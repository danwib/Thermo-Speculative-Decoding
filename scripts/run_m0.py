# SPDX-License-Identifier: Apache-2.0
"""Entry point for the M0 smoke test."""

from __future__ import annotations

import typer


app = typer.Typer(add_completion=False)


@app.callback()
def main() -> None:
    """CLI bootstrap for M0 experiments."""


@app.command()
def run(seed: int = 0) -> None:
    """Run the M0 smoke test."""

    raise NotImplementedError("M0 run script will be implemented alongside tests.")


if __name__ == "__main__":
    app()

