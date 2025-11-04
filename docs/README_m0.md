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
# auto tail-mass epsilon (recommended)
python -m scripts.run_m0 run --vocab 1000 --K 64 --eps auto --steps 50000 --seed 7

# fixed epsilon (advanced)
python -m scripts.run_m0 run --vocab 1000 --K 64 --eps 1e-6 --steps 50000 --seed 7

# Zipf target with alpha=1.2
python -m scripts.run_m0 run --vocab 1000 --K 64 --pgen zipf --alpha 1.2 --eps auto --steps 50000 --seed 7
```

`--eps auto` distributes the true tail mass of `p` uniformly over the out-of-set
tokens:

```
ε = (1 - Σ_{y∈topK(p)} p(y)) / max(V - K, 1)
```

This typically raises acceptance rates versus a tiny fixed `ε`, especially when
the proposer already captures most of the target mass.

`--pgen zipf` reshapes the synthetic target to follow a Zipf law with exponent
`alpha`, useful for stress-testing long-tail behaviours.

Output looks like:

```
metrics_path=runs/<timestamp>/m0_metrics.jsonl
accept_rate=0.72
chi2_stat=518.93
p_value=0.33
psi_bytes_mean=332.0
topk_mass=0.98
eps_used=2.1e-04
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
