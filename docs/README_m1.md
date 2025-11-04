# Milestone M1 Guide

SPDX-License-Identifier: Apache-2.0

## Goal

Extend Thermo-Speculative Decoding to a contextual bigram target `p(y_t | x_{t-1})`
while reusing the ψ/TSU contract validated in M0.

What changes from M0:

- Target distribution is now conditional on the previous token (bigram table).
- The proposer crafts a fresh ψ payload per context using the same Top-K schema.
- Telemetry focuses on overlap mass and cross-entropy gap in addition to acceptance.

What stays the same:

- `SimTSU` still samples from `F(ψ)` and returns the exact `log q`.
- `accept_correct_step` provides unbiased acceptance with the residual
  `p - min(p, q)`.
- ψ quantisation, payload sizing, and tail-mass handling remain unchanged.

## Running M1

Synthetic bigram (default configuration):

```bash
python -m scripts.run_m1 run --vocab 256 --K 64 --eps auto --steps 50000 --seed 7
```

Character corpus (Tiny Shakespeare or your own file):

```bash
python -m scripts.run_m1 run --corpus data/tinyshakespeare.txt --K 64 --eps auto --steps 50000 --seed 7
```

Key options:

- `--corpus PATH`: switch to char-level counts with Laplace smoothing (`+1`).
- `--vocab`, `--c`: control synthetic Dirichlet sampling when no corpus is
  provided (`V=256`, `c=50` by default).
- `--K`, `--tau`, `--eps`: ψ payload hyperparameters per context (auto-ε matches
  tail mass uniformly).
- `--steps`, `--seed`: end-to-end sample count and RNG seed.

## Telemetry & Summary

Each run writes a JSONL stream to `runs/<timestamp>/m1_metrics.jsonl` capturing
per-context telemetry:

- `step`, `prev_token`, `proposed_token`, `emitted_token`, `accepted`
- ψ metadata (`K`, `tau`, `eps_used`, `psi_bytes`)
- Context metrics (`topk_mass`, `overlap_mass`, `ce_gap_nats`)

At completion, `summary.json` captures aggregated statistics:

- `accept_rate` – fraction of proposals accepted across the run.
- `mean_overlap` – average of the overlap mass `Σ min(p, q)` per context.
- `mean_ce_gap` – mean cross-entropy gap `H(p, q) - H(p)` in natural units.
- `chi2_stat`, `p_value` – χ² goodness-of-fit vs the stationary unigram
  (smoke-check that marginal frequencies remain stable).

CLI output echoes the summary for quick inspection. Use the JSONL for deeper
analysis (acceptance traces, CE gap histograms, etc.).
