# Changelog

All notable changes to this project will be documented in this file.

## [M1] - 2025-11-04

- Added contextual bigram targets with corpus + synthetic builders and Laplace smoothing.
- Crafted per-context ψ payloads using the M0 Top-K format for proposer stubs.
- Introduced the `scripts/run_m1.py` CLI, telemetry (overlap, CE gap), and summary outputs.
- Added unbiasedness tests on fixed-context slices and telemetry consistency checks.
- Provided seed sweep helper and docs covering M1 workflow.

## [M0] - 2025-11-04

- Implemented ψ schema, TSU simulator, and accept/correct verification loop.
- Added statistical tests: golden-ψ contract, M0 end-to-end, CLI smoke, TPS guards.
- Introduced `scripts/run_m0.py` CLI and artifact generator for reproducible runs.
- Expanded architecture docs and onboarding notes for the milestone.
- Documented the M0 success checklist and published release tag `m0-green`.
