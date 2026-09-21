//! Qwen-Image 2.1: a text-to-image and image-editing diffusion pipeline.
//!
//! Written from diffusers at 6256aa7 (`transformer_qwenimage21.py`,
//! `pipeline_qwenimage21.py`, `autoencoder_kl_qwenimage21.py`), not vendored
//! from candle, which has no module for this model.

pub mod conditioning;
pub mod scheduler;
pub mod transformer;
pub mod vae;
