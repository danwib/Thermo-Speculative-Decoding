# Milestone M0 Guide

SPDX-License-Identifier: Apache-2.0

## Goal

Validate Thermo-Speculative Decoding on the simplest setting: a fixed categorical
target distribution `p(y)` with no context (`L = 1`). We check that the simulated
Thermodynamic Sampling Unit (TSU) plus verifier accept/correct loop produces
unbiased samples and exposes lightweight telemetry for analysis.

## Components

- **Target distribution `p`** – sampled once via `make_p` (Dirichlet(1)) and
  cached as `(p, log p)`; see `src/tsd/targets/m0_categorical.py`.
- **Proposer payload `ψ`** – `PsiTopK` encoding (`ids`, quantised scores, `τ`,
  `ε`, `V`) produced by `craft_psi_from_p`.
- **Reference map `F(ψ)`** – converts `ψ` into probabilities via
  `logq_for_ids`; guarantees sampler/verifier agreement.
- **TSU simulator** – `SimTSU.sample_categorical` draws tokens and returns the
  exact `log q` for each proposal.
- **Verifier** – `accept_correct_step` consumes TSU-provided `log q`, performs
  accept/correct, and emits unbiased tokens.

## Running M0

From the repository root (inside a virtual environment):

```bash
python -m scripts.run_m0 run --steps 5000 --vocab 500 --K 32 --seed 11
```

Output looks like:

```
metrics_path=runs/<timestamp>/m0_metrics.jsonl
accept_rate=0.72
chi2_stat=518.93
p_value=0.33
psi_bytes_mean=332.0
```

Success criteria:

- `p_value` comfortably above 0.05 (χ² goodness-of-fit vs. `p`).
- Acceptance rate between roughly 0.2 and 0.9 (depends on `K`, `τ`, `ε`).
- JSONL file populated with per-step telemetry for offline analysis.

## Common Gotchas

- **Always use returned `log q`:** the verifier must consume the exact TSU
  output. Recomputing from `ψ` can break unbiasedness when hardware differs.
- **Ensure RNG determinism:** `SimTSU` reseeds per-step (`seed + step`) while the
  verifier uses a separate generator for acceptance draws. Misaligned seeds will
  affect statistical tests.
- **Payload size:** `psi_size_bytes(ψ)` should stay under 1 KB for `K ≤ 128`. A
  much larger value suggests quantisation parameters need tuning.
- **Runs directory:** the CLI writes to `runs/<timestamp>/m0_metrics.jsonl`.
  Clean old runs before reusing seeds to avoid confusion.

