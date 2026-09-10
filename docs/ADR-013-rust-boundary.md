# ADR-013: Rust Boundary Convention - Line Protocol, Zero Crates, One Bridge

## Status: Accepted (Week 14)

## Context
Two competing A/B harnesses existed: a JSON-protocol single-neuron probe
(quarantined as rust-core/_quarantine/rust_bridge.py.json-probe) and the
canonical line-protocol population harness (benchmarks/ab_bridge.py).
A crate-name clobber (hybra_core vs hybrid_brain_core) broke the build
and caused a silent gate skip.

## Decision
1. Rust/Python boundary = stdin/stdout LINE PROTOCOL, zero external
   crates in default build. serde/JSON rejected: build weight + float
   formatting control.
2. ONE bridge (benchmarks/ab_bridge.py). The Python side tests the
   production core/neuron.py LifPopulation, not a reimplementation.
3. Bit-exactness: spike masks identical + fracs within 1e-12.
   Mismatch fixes in Rust first, never Python.
4. PyO3 bindings stay feature-gated (python-bindings = ["dep:pyo3"]);
   default build never compiles pyo3.
5. Known discrepancy to reconcile in maturin week: harness v_reset=-70.0
   vs brain_config.yaml -65mV.

## Incident record
Cargo.toml overwritten without audit -> E0433 -> build_rust() False ->
gate SKIPPED (looked green). Rules added: audit before write; a skipped
gate on a machine with the toolchain is an open bug; cargo must finish
zero-warning; SilentlyContinue banned in recovery commands.

## Executable proof
python benchmarks/ab_bridge.py -> "A/B ACCEPTED"
tests/test_rust_bridge.py::test_rust_python_bit_equivalent -> PASSED
