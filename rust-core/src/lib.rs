//! Hybrid-Brain core - Rust acceleration layer (ADR-002 ARM64-friendly).
pub mod neurons;
pub mod synapses;

#[cfg(feature = "python-bindings")]
pub mod python_bindings;

pub const CONTRACT_VERSION: &str = "0.4.0";
