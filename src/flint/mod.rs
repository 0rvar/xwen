//! Flint (`flint`, ggml-org/Flint-GGUF): the attention block that the rest of
//! the trunk does not already have.
//!
//! Ground truth is llama.cpp `src/models/flint.cpp` at a37cc4610 (PR 30928).
//! Per layer, activations f32 `[t, hidden]`:
//!
//! ```text
//!   n  = rms_norm(x, attn_norm)
//!   q  = attn_q(n) -> [t, 32, 128] -> rms_norm(attn_q_norm) -> rope if roped
//!   k  = attn_k(n) -> [t,  4, 128] -> rms_norm(attn_k_norm) -> rope if roped
//!   v  = attn_v(n) -> [t,  4, 128]
//!   a  = softmax([q kᵀ / √128 + mask | sink_h]) [v | 0]           (GQA 8)
//!   h  = x + attn_output(a)                                        (ffn_inp)
//!   m  = rms_norm(h, ffn_norm)                                     (ffn_norm)
//!   x' = h + ffn(m)                                                (l_out)
//! ```
//!
//! where `ffn` is a dense SwiGLU on layers `0..leading_dense_block_count` and
//! on every later layer the MoE block: a SIGMOID router over 256 experts, top-6,
//! renormalized, plus a shared expert added in full (no gate). The layer, its
//! norms and its FFN are the trunk's own (`model::Layer`, `moe::MoeBlock`,
//! `moe::DenseMlp`); only the attention is new, and it is [`FlintAttn`].
//!
//! The layer pattern, from the file's two per-layer bool arrays (which agree):
//! a sliding layer (window 513, the current position included) is roped, a
//! full-attention layer has no positional encoding at all. On the shipped file
//! the full layers are 0, 1, 5, 9, ..., 45 and 48.

pub mod attention;

pub use attention::FlintAttn;
