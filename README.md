# VEPS DAG Engine Prototype

This repository contains the single-process Python prototype, CPU benchmark
harnesses, conformance campaigns, and video analytics application used to
evaluate lease-protected publication of compiled DAG plans. An admitted frame
selects one immutable plan and holds its lease through execution. Stateless
publication can overlap an old-plan frame, while preservation of mutable state
requires a bounded pre-publication drain. Superseded processors are retired
only after leases that can reference them have ended.

## Development checks

Run the unit and integration tests with `uv run pytest`. Run the type checker
with `uv run pyright`. Core concurrency tests can also be run under
`uv run python -Xgil=0 -m pytest tests/test_grace_period.py
tests/test_correctness_repairs.py`. These checks do not produce publication
measurements.

## Benchmark evidence

`python -m benchmarks.cli run --help` lists the current raw-evidence CLI
options. Publication runs require a clean Git tree, locked dependencies,
recorded inputs and hardware provenance, and a disabled GIL after production
imports. The `conformance` suite writes per-campaign JSONL event traces. The
`validate-gate` subcommand replays those traces and validates raw artifacts.
Reporting is a separate offline command, and benchmark execution does not
write manuscript tables or figures.

Historical artifacts already present in this repository remain in place for
recordkeeping. They are not part of the current schema-2 publication evidence
and must not be mixed with a new validated campaign. A failed run retains
`failure.json` and must never be represented as completed evidence.

See [SPEC.md](SPEC.md) for the intended runtime contract. The concrete
implementation and tests are authoritative where that specification describes
future or optional components.
