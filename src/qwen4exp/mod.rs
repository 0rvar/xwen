//! Qwen3.8-Flash-Next (`qwen4exp`) — the three subsystems the other checkpoints
//! don't have: hyper-connections, the QSA indexer, and the PLE n-gram table.
//!
//! The `ref_*` modules are frozen CPU f32 correctness oracles, tested against
//! `tests/fixtures/qwen4exp/` (transformers-generated). They are never
//! optimized; the Metal paths are graded against them. See
//! docs/qwen4exp-port.md (D5).

pub mod ref_hc;
pub mod ref_ple;
pub mod ref_qsa;

pub mod hc;
pub mod indexer;
pub mod iq4nl;
pub mod ple;
pub mod stack;

#[cfg(test)]
pub mod tiny_gguf;
