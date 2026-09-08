// Vendored from candle (https://github.com/huggingface/candle) at rev
// 21cca0b196f00784ba3e7b12832694f4e83b5c18, `candle-transformers/src/models/z_image/`,
// added upstream in PR #3261 by SpenserCai. Dual-licensed MIT / Apache-2.0 by the
// candle authors; xwen carries the copy so it can be corrected against the
// reference implementation and tuned for this machine.
/*
 * @Author: SpenserCai
 * @Date: 2026-01-02 11:35:48
 * @version:
 * @LastEditors: SpenserCai
 * @LastEditTime: 2026-01-02 11:48:26
 * @Description: file content
 */
//! Z-Image-Turbo, the text-to-image pipeline: a 6.15B single-stream diffusion
//! transformer (30 main layers, 2 noise-refiner and 2 context-refiner blocks,
//! dim 3840, 30 heads of 128), the Flux VAE (16 latent channels, 8x spatial),
//! and a flow-matching Euler scheduler with a static shift of 3.0.
//!
//! - [Hugging Face model](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
//! - [Official implementation](https://github.com/Tongyi-MAI/Z-Image)
//!
//! The text encoder is NOT here: Z-Image conditions on the `hidden_states[35]`
//! of Qwen3-4B, and that is [`crate::XwenModel::encode`] on the
//! [`crate::hub::Model::ZImageTurboEncoder`] entry, verified against a torch
//! fp32 reference (docs/zimage.md). [`pipeline::ZImagePipeline`] takes that
//! tensor and returns pixels.
//!
//! Ground truth for every form here is the official `src/zimage/*.py` and
//! diffusers' `transformer_z_image.py` / `pipeline_z_image.py`; the shapes
//! and the traps are written up in docs/zimage.md. Where the two references
//! disagree, which is the sigma grid alone, this follows DIFFUSERS
//! ([`scheduler`] says by how much and why).
//!
//! Three environment switches, all read once when the pipeline loads.
//! [`ATTN_ENV`] (`XWEN_ZIMAGE_ATTN=basic`) swaps candle's fused Metal SDPA
//! for an explicit matmul-softmax-matmul chain that shares no kernel with it,
//! and [`LINEAR_ENV`] (`XWEN_ZIMAGE_LINEAR=candle`) swaps xwen's tensor gemm
//! for candle's: both are for bisecting, not for speed.
//! [`PROFILE_ENV`] (`XWEN_ZIMAGE_PROFILE=1`) prints a per-stage table for
//! the transformer and the VAE.

pub mod conditioning;
pub mod controlnet;
pub mod inputs;
pub mod linear;
pub mod lora;
pub mod pipeline;
pub mod preprocess;
pub mod profile;
pub mod sampling;
pub mod scheduler;
pub mod transformer;
pub mod vae;

pub use linear::{LINEAR_ENV, LinearImpl};
pub use pipeline::{ImageOptions, Rendered, Timings, ZImagePipeline, encode_png, write_png};
pub use profile::{PROFILE_ENV, Profiler};
pub use sampling::{postprocess_image, seeded_noise};
pub use scheduler::{FlowMatchEulerDiscreteScheduler, SchedulerConfig};
pub use transformer::{ATTN_ENV, AttnImpl, Config, ZImageTransformer2DModel};
pub use vae::{AutoEncoderKL, VAE_ENV, VaeConfig, VaeImpl};
