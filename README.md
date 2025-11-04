# Thermo-Speculative Decoding (TSD) Prototype

SPDX-License-Identifier: Apache-2.0

Thermo-Speculative Decoding (TSD) explores hybrid proposer–verifier sampling
for fast language model generation. This repository scaffolds the prototype,
including a simulated Thermodynamic Sampling Unit (TSU), verifier, and telemetry
plumbing across two milestones:

- **M0:** Smoke test with a fixed categorical target distribution.
- **M1:** Bigram contextual target with a lightweight proposer emitting compact
  ψ payloads for the TSU simulator.

## Project Layout

```
src/tsd/
  psi.py               # ψ schema and reference F(ψ) sampler
  tsu_iface.py         # Abstract TSU adapter and simulator implementation
  proposer/            # Lightweight proposer scaffolding
  verifier/            # Speculative accept/correct utilities
  targets/             # Target distributions for milestones
  train/               # Distillation and acceptance-aware training stubs
  telemetry/           # Metrics and logging helpers
docs/
  ARCHITECTURE.md
  DESIGN_TSD.md
  DEV_NOTES.md
scripts/
  run_m0.py
  run_m1.py
tests/
  ...
```

## Getting Started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The initial milestones focus on validating ψ semantics and unbiased sampling via
pytest. CLI entry points under `scripts/` will provide reproducible experiments.

## Roadmap

- [ ] Implement ψ schema and TSU simulator parity tests (M0).
- [ ] Wire up proposer and bigram target with acceptance telemetry (M1).
- [ ] Integrate Extropic `thrml` backend via a dedicated adapter (post-M1).
- [ ] Extend to block drafting (L = 2–4) and acceptance-aware training.

See `docs/ARCHITECTURE.md` for detailed design notes and open questions.

