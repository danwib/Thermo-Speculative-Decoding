# Developer Notes

SPDX-License-Identifier: Apache-2.0

## Milestone Checklist

- [ ] M0
  - [ ] Implement ψ schema and reference sampler.
  - [ ] Wire `SimTSU` through the verifier for accept/correct.
  - [ ] Add pytest coverage for ψ semantics and unbiased sampling.
- [ ] M1
  - [ ] Introduce contextual bigram target and proposer stub.
  - [ ] Extend telemetry (JSONL + CSV).
  - [ ] Provide CLI scripts for reproducible experiments.

## Open Questions

1. Finalize `thrml` adapter surface and fidelity tests.
2. Decide on proposer architecture (SSM vs. GRU) and training data flow.
3. Define acceptance-aware fine-tuning objectives and telemetry dashboards.

Update this file as decisions land or new risks emerge.

