# TSD Design Notes

SPDX-License-Identifier: Apache-2.0

## ψ Payload

- `ids`: `np.ndarray` of shape `(K,)`, dtype `int32`.
- `scores`: `np.ndarray` of shape `(K,)`, dtype `int8`, paired with global scale
  and zero-point metadata.
- `scale`: `float16`, `zero_point`: `int8`.
- `tau`: `float16` (temperature), `epsilon`: `float16` (floor).
- `mask`: optional bitset over the full vocabulary for exclusion.

`F(ψ)` deterministically expands to categorical parameters used by the TSU.
Simulator and production adapters must share semantics.

## Sampling Loop

1. Proposer consumes context and emits ψ.
2. TSU samples tokens and returns `(tokens, log_q)` tuples.
3. Verifier scores proposed tokens under target `p` and performs accept/correct.
4. Telemetry records cadence statistics and acceptance outcomes.

## Future Extensions

- Block drafts (`L = 2–4`) with tree-batched verification.
- Acceptance-aware tuning loops leveraging `tsd.train`.
- GPU verifier kernels for large targets.

Contributions should reference this document when updating major design choices.

