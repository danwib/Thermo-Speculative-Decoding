# SPDX-License-Identifier: Apache-2.0
"""Entry point for the M1 bigram experiment with contextual proposals."""

from __future__ import annotations

import datetime as _dt
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import typer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tsd.psi import logq_full, psi_size_bytes
from tsd.targets import (
    craft_psi_from_row,
    make_bigram_from_corpus,
    make_bigram_synthetic,
    row_logprobs,
)
from tsd.telemetry.metrics import chi2_test_counts
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


def _stationary_from_matrix(P: np.ndarray) -> np.ndarray:
    """Approximate the stationary unigram by averaging conditional rows."""

    unigram = np.mean(P, axis=0)
    unigram /= np.sum(unigram, dtype=np.float64)
    return unigram.astype(np.float64)


@app.callback()
def main() -> None:
    """CLI bootstrap for M1 experiments."""


@app.command()
def run(
    corpus: Optional[Path] = typer.Option(
        None, "--corpus", help="Path to corpus file for char-level bigram."
    ),
    vocab: int = typer.Option(256, "--vocab", min=2, help="Vocabulary size when synthetic."),
    c: float = typer.Option(50.0, "--c", help="Synthetic Dirichlet coupling strength."),
    seed: int = typer.Option(7, "--seed", help="Base seed for RNGs."),
    K: int = typer.Option(64, "--K", min=1, help="Top-K payload size."),
    tau: float = typer.Option(1.0, "--tau", help="Proposer temperature."),
    eps: Optional[str] = typer.Option("auto", "--eps", help="Tail mass floor ('auto' to match)."),
    steps: int = typer.Option(50_000, "--steps", min=1, help="Number of decoding steps."),
    return_metrics: bool = False,
) -> Optional[Dict[str, float]]:
    """Run the contextual M1 bigram experiment."""

    eps_val = _parse_eps(eps)

    if tau <= 0.0:
        raise typer.BadParameter("--tau must be positive.")

    if corpus is not None:
        P, logP, _, _ = make_bigram_from_corpus(str(corpus))
        vocab_size = P.shape[0]
    else:
        P, logP = make_bigram_synthetic(V=vocab, seed=seed, c=c)
        vocab_size = P.shape[0]

    effective_K = min(K, vocab_size)

    stationary = _stationary_from_matrix(P)
    sampler_rng = np.random.Generator(np.random.PCG64(seed))
    prev_id = int(sampler_rng.choice(vocab_size, p=stationary))

    accept_rng = np.random.Generator(np.random.PCG64(seed + 10_000))
    tsu = SimTSU()

    runs_root = Path("runs")
    run_id = _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = run_dir / "m1_metrics.jsonl"
    summary_path = run_dir / "summary.json"

    emitted_counts = np.zeros(vocab_size, dtype=np.int64)
    accept_count = 0
    overlap_samples = []
    ce_gap_samples = []

    eps_request = "auto" if isinstance(eps_val, str) else float(eps_val)

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for step in range(steps):
            logp_row = np.array(row_logprobs(logP, prev_id), copy=True)

            psi = craft_psi_from_row(
                logp_row=logp_row,
                K=effective_K,
                tau=tau,
                epsilon=eps_val,
            )
            psi_bytes = psi_size_bytes(psi)
            eps_used = float(psi.epsilon)

            tokens, logq = tsu.sample_categorical(
                psi, batch_size=1, seed=seed + step
            )
            proposed_token = int(tokens[0])
            proposed_logq = float(logq[0])

            emitted_token, accepted = accept_correct_step(
                logp=logp_row,
                psi=psi,
                proposed_token=proposed_token,
                proposed_logq=proposed_logq,
                rng=accept_rng,
            )

            accept_count += int(accepted)
            emitted_counts[emitted_token] += 1

            logq_all = logq_full(psi)
            q_row = np.exp(logq_all)
            p_row = np.exp(logp_row)

            overlap_mass = float(np.minimum(p_row, q_row).sum())
            topk_mass = float(p_row[psi.ids].sum())
            ce_gap = float(np.sum(p_row * (logp_row - logq_all), dtype=np.float64))

            overlap_samples.append(overlap_mass)
            ce_gap_samples.append(ce_gap)

            row = {
                "step": step,
                "prev_token": int(prev_id),
                "proposed_token": proposed_token,
                "emitted_token": int(emitted_token),
                "accepted": bool(accepted),
                "K": int(psi.ids.shape[0]),
                "tau": float(tau),
                "eps_request": eps_request,
                "eps_used": eps_used,
                "psi_bytes": int(psi_bytes),
                "topk_mass": topk_mass,
                "overlap_mass": overlap_mass,
                "ce_gap_nats": ce_gap,
            }
            handle.write(json.dumps(row) + "\n")

            prev_id = emitted_token

    accept_rate = accept_count / steps
    mean_overlap = float(np.mean(overlap_samples))
    mean_ce_gap = float(np.mean(ce_gap_samples))
    chi2_stat, p_value = chi2_test_counts(emitted_counts, stationary)

    summary: Dict[str, float] = {
        "accept_rate": accept_rate,
        "mean_overlap": mean_overlap,
        "mean_ce_gap": mean_ce_gap,
        "chi2_stat": chi2_stat,
        "p_value": p_value,
    }

    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if return_metrics:
        summary_with_path: Dict[str, float] = dict(summary)
        summary_with_path["metrics_path"] = str(jsonl_path)
        return summary_with_path

    typer.echo(f"metrics_path={jsonl_path}")
    typer.echo(f"accept_rate={accept_rate:.6f}")
    typer.echo(f"mean_overlap={mean_overlap:.6f}")
    typer.echo(f"mean_ce_gap={mean_ce_gap:.6f}")
    typer.echo(f"chi2_stat={chi2_stat:.6f}")
    typer.echo(f"p_value={p_value:.6f}")

    return None


if __name__ == "__main__":
    app()
