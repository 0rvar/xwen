//! Qwen3 dense (`qwen3`) — the Qwen3-4B family, loaded from HF BF16 safetensors
//! rather than from a GGUF.
//!
//! Three checkpoints share this one graph: `Qwen/Qwen3-4B`,
//! `Qwen/Qwen3-4B-Instruct-2507` and the Z-Image-Turbo text encoder, whose
//! `text_encoder/` directory is a byte-identical copy of the base model's first
//! two shards. They differ only in `rope_theta`, `max_position_embeddings` and
//! the chat template.
//!
//! Two of the three pieces here are independent of any device: the explicit
//! config ([`config`]) and the safetensors loader ([`safetensors`]). Both run
//! on the CPU and are unit-tested without a Metal device, which is the point —
//! a shape or a dtype that is wrong should fail before a single byte reaches
//! the GPU. The third, the layer stack ([`stack`]), is the Metal graph
//! `XwenModel` runs over the loaded weights.

pub mod config;
pub mod safetensors;
pub mod stack;

pub use config::{HfQwen3Config, NormVariant, QWEN3_EOG, Qwen3Config, RopeSpec};
pub use safetensors::{
    Qwen3LayerWeights, Qwen3Set, Qwen3Weights, RangeScan, TensorSet, ZERO_RUN_LIMIT, ZeroRun,
};
pub use stack::{AttnImpl, Qwen3Parts};
