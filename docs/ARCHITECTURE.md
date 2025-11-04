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

## ψ v1 Schema

The Top-K ψ payload is represented by `PsiTopK` with the following fields:

- `ids (int32[K])`: Token identifiers ordered by descending proposer score.
- `scores_q8 (int8[K])`: Symmetric int8 quantised scores aligned with `ids`.
- `scale (float32)` and `zero_point (int8)`: Quantisation metadata used during
  de-quantisation.
- `tau (float16)`: Temperature applied to the de-quantised scores.
- `epsilon (float16)`: Floor probability allocated to tokens outside the Top-K
  set.
- `vocab_size (int32)`: Cardinality of the target vocabulary.

The induced categorical distribution `F(ψ)` is defined as:

```
Z = Σ_k exp(s_k / τ) + (V - K) * ε
p(i) = exp(s_i / τ) / Z       if i ∈ ids
p(o) = ε / Z                  otherwise
```

where `s_k` are the de-quantised scores, `τ` is the temperature, `ε` is the
floor probability, `K` is the payload size, and `V` is the vocabulary size. The
software reference computes `log q` in float64 for numerical stability and uses a
single normalisation pass compatible with simulator implementations.

### Numerical considerations

- De-quantised scores are accumulated in `float64` during normalisation to avoid
  catastrophic cancellation for sharp distributions.
- Log-space arithmetic (`logq_for_ids`, `accept_correct_step`) prevents underflow
  when forming acceptance ratios or residuals.

## M0 Target Construction

- `make_p(vocab_size, seed)` samples a fixed categorical target by drawing from a
  Dirichlet(1) prior using PCG64 for reproducibility. It returns both `p` and
  `log p` for downstream evaluation.
- `craft_psi_from_p(p, K, tau, epsilon)` selects the Top-K entries of `p` (stable
  order), encodes their log probabilities into quantised scores, and emits a
  `PsiTopK` payload. The floor mass `ε` keeps residual probability consistent
  with `F(ψ)`, ensuring high acceptance rates in the smoke test.

References:
- `src/tsd/psi.py` – ψ schema, `logq_for_ids`, and payload sizing.
- `src/tsd/tsu_iface.py` – `SimTSU.sample_categorical` (reference TSU).
- `src/tsd/verifier/accept_correct.py` – single-token accept/correct.
- `src/tsd/targets/m0_categorical.py` – target and ψ crafting utilities.
- `scripts/run_m0.py` – CLI entry point writing telemetry.

```
  +---------------+     +-----------------+     +---------------------+
  | make_p / ψ    | --> | SimTSU (F(ψ))    | --> | accept_correct_step |
  | crafting      |     | sample + log q  |     | accept / residual   |
  +---------------+     +-----------------+     +---------------------+
          ^                       |                          |
          |                       v                          v
        params                ψ telemetry              telemetry JSONL
```

### Accept/Correct (L = 1)

Acceptance ratio:

```
α(x*) = min(1, p(x*) / q(x*))
```

Residual distribution (after rejection):

```
r(y) ∝ p(y) - min{p(y), q(y)}
```

Normalise by ``1 - Σ_y min{p(y), q(y)}``. This construction is always
non-negative, unlike the earlier `p - α q` heuristic.

## Bigram Targets (M1)

- `build_vocab_from_corpus(path)` extracts a sorted character vocabulary from an
  on-disk corpus. Newlines and whitespace remain as standard tokens so the
  transition matrix stays faithful to the source.
- `make_bigram_from_corpus(path, laplace)` counts adjacent character pairs,
  applies Laplace smoothing (`+laplace` everywhere), and normalises each row to
  obtain conditional probabilities `P(y | x)` alongside the matching `log P`.
- `make_bigram_synthetic(V, seed, c)` samples a stationary unigram
  `u ~ Dirichlet(1)` and, for every context, a row distribution from
  `Dirichlet(1 + c * u)` to inject mild structure without an external corpus.
- `row_logprobs(logP, prev_id)` returns the view consumed by the proposer and
  verifier when conditioning on `x_{t-1}`.

All helpers operate in `float64` and seed `numpy.random.default_rng` for
deterministic experiments. The `(stoi, itos)` vocabulary metadata emitted by the
corpus path keeps proposer and verifier token IDs aligned.

Per-context ψ payloads reuse the M0 Top-K schema: `craft_psi_from_row` selects
the largest `K` transitions, embeds their log-probabilities into quantised scores,
and relies on the same `F(ψ)` semantics for both TSU sampling and verifier
correction.

The downstream components remain identical to M0: `SimTSU` still invokes the
reference `F(ψ)` sampler, the verifier runs `accept_correct_step`, and the
residual distribution `p - min(p, q)` guarantees unbiasedness regardless of the
context.

## Accept/Correct (L = 1)

- The verifier receives a proposed token `x` and its ``log q(x)`` from the TSU.
  The acceptance ratio is computed as ``α = min(1, exp(log p(x) - log q(x)))``
  using log-space arithmetic for stability.
- On acceptance (`u < α`), the proposed token is emitted directly.
- On rejection, we sample from
  ``r(y) ∝ p(y) - min{p(y), q(y)}``, normalised by the remaining mass ``1 - Σ_y min{p(y), q(y)}``.
  Reconstruct ``q`` via `F(ψ)` and use the TSU-returned ``log q(x*)`` in the acceptance ratio.

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
