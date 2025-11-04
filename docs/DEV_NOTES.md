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

## Notes

- Probability handshake: the verifier must consume the `log q` emitted by the TSU
  (simulated or hardware) to maintain unbiased acceptance decisions. Do not
  recompute proposer probabilities inside the verifier.

## Golden-ψ Test Plan & Probability Handshake

- `tests/test_golden_psi_contract.py` validates that the simulated TSU returns
  `log q` values exactly matching `F(ψ)` and that empirical samples align with
  theoretical probabilities.
- Any TSU implementation (including `thrml`) must pass the same contract to
  guarantee verifier correctness.

## How We Validate M0

- `tests/test_m0_end_to_end.py` (`@pytest.mark.slow`): χ² p-value > 0.05,
  acceptance in `[0.4, 0.95]` for `(V=1000, K=64, steps=30k)`.
- `tests/test_m0_cli_smoke.py`: CLI subprocess run with `steps=5000`, p-value >
  0.01, acceptance within `[0.1, 1.0]`.
- `tests/test_m0_target_and_psi.py`: schema sanity and payload sizing.
- CLI `scripts/run_m0.py` and artifact script `scripts/make_m0_artifact.py` are
  expected to emit JSONL metrics and summary stats per run.

## Open Questions

1. Finalize `thrml` adapter surface and fidelity tests.
2. Decide on proposer architecture (SSM vs. GRU) and training data flow.
3. Define acceptance-aware fine-tuning objectives and telemetry dashboards.

Update this file as decisions land or new risks emerge.
