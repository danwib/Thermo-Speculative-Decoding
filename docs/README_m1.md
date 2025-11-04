# Milestone M1 Guide

SPDX-License-Identifier: Apache-2.0

## Goal

Extend Thermo-Speculative Decoding to a contextual bigram target `p(y_t | x_{t-1})`
while reusing the ψ/TSU contract validated in M0. The proposer now emits ψ per
context, and telemetry tracks overlap and cross-entropy gaps for every step.

## Running M1

Synthetic bigram (default configuration):

```bash
python -m scripts.run_m1 run --vocab 256 --steps 50000 --seed 7 --K 64 --tau 1.0 --eps auto
```

Character corpus (Tiny Shakespeare or your own file):

```bash
python -m scripts.run_m1 run --corpus data/tiny_shakespeare.txt --steps 50000 --K 64 --tau 1.0 --eps auto
```

Key options:

- `--corpus PATH`: switch to char-level counts with Laplace smoothing (`+1`).
- `--vocab`, `--c`: control synthetic Dirichlet sampling when no corpus is
  provided (`V=256`, `c=50` by default).
- `--K`, `--tau`, `--eps`: ψ payload hyperparameters per context (auto-ε matches
  tail mass uniformly).
- `--steps`, `--seed`: end-to-end sample count and RNG seed.

## Telemetry & Summary

Each run writes a JSONL stream to `runs/<timestamp>/m1_metrics.jsonl` with:

- `step`, `prev_token`, `proposed_token`, `emitted_token`, `accepted`
- ψ metadata (`K`, `tau`, `eps_used`, `psi_bytes`)
- Context metrics (`topk_mass`, `overlap_mass`, `ce_gap_nats`)

At completion, `summary.json` captures aggregated statistics:

- Acceptance rate
- Mean overlap mass
- Mean cross-entropy gap
- χ² p-value vs the stationary unigram (smoke-check)

CLI output echoes the summary for quick inspection. Use the JSONL for deeper
analysis (acceptance traces, CE gap histograms, etc.).
