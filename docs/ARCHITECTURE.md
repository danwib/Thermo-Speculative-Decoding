# TSD Architecture

SPDX-License-Identifier: Apache-2.0

## Overview

The prototype is organized around three cooperating components:

- **Proposer (`q`)** emits compact ψ payloads that parameterize the Thermodynamic
  Sampling Unit (TSU).
- **TSU** performs sampling according to ψ, returning sampled tokens and the
  corresponding `log q`.
- **Verifier (`p`)** re-scores the proposed tokens, performing speculative
  accept/correct to ensure unbiased draws from the target distribution.

The system targets milestone-driven development:

1. **M0 (Smoke Test):** Fixed categorical target `p(y)` with no context. Validate
   accept/correct flow, TSU API plumbing, and telemetry recording.
2. **M1 (Bigram Target):** Contextual bigram `p(y_t | x_{t-1})` with a small
   proposer that produces ψ for single-token drafts. Extend telemetry, integrate
   proposer state stubs, and exercise TSU simulator control flow.

## Module Responsibilities

- `tsd.psi`: Defines the ψ schema (Top-K IDs + int8 scores, scaling metadata,
  temperature/floor parameters, optional mask) and a reference `F(ψ)` sampler used
  both in tests and simulator implementations.
- `tsd.tsu_iface`: Contains the `TSUAdapter` protocol, shared sampler utilities,
  and a `SimTSU` implementation that uses the reference sampler.
- `tsd.proposer`: Houses lightweight state and head modules that map verifier
  context into ψ payloads. Initial milestones use deterministic stubs with seeded
  randomness.
- `tsd.verifier`: Implements accept/correct for speculative decoding.
- `tsd.targets`: Encapsulates milestone-specific target distributions.
- `tsd.train`: Future modules for distillation and acceptance-aware fine-tuning.
- `tsd.telemetry`: Aggregates metrics, logging, and JSONL output.

## Integration with `thrml`

The TSU abstraction intentionally decouples simulator logic from the Extropic
`thrml` backend. A dedicated adapter (`ThrmlTSU`) will wrap `thrml` once the
package is available.

> TODO: Lock to `thrml >= X.Y` when available; run golden-ψ fidelity tests before
> enabling the backend.

## Telemetry

Metrics are streamed to JSONL files under `runs/<date>/tsd_metrics.jsonl` with the
following schema:

- `run_id`, `seed`, `step`, `mode` (e.g., `M0`, `M1`), `L`, `B`
- `psi_bytes`, `tsu_latency_ms`, `verifier_latency_ms`
- `accepted_prefix`, `first_token_reject`
- `ce_gap_nats`, `q_temp`, `q_floor`
- `version_hash`, `psi_semver`, `sampler_semver`

A per-run CSV summary aggregates mean/variance statistics for quick inspection.

## Testing Strategy

- `test_psi_semantics.py`: Golden ψ tests to ensure simulator parity.
- `test_accept_correct.py`: Acceptance unbiasedness against the target `p`.
- `test_m0_throughput.py`: Latency and ψ size regression safeguards.
- `test_m1_acceptance.py`: Validate acceptance rates and telemetry for contextual
  targets.

All tests should seed RNGs for reproducibility and avoid reliance on GPU-specific
determinism.

