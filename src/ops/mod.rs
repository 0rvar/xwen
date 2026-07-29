pub mod attn_glue;
pub mod bf16;
pub mod combine;
pub mod delta;
mod dispatch;
pub mod f16;
pub mod flash;
pub mod mm_id;
pub mod moe_glue;
pub mod mv_id;
mod pipelines;
pub mod q8;
pub mod silu_mul;

pub use attn_glue::{attn_gate, cast_f16, cast_f32, permute_01, permute_01_f16, rope_neox};
pub use bf16::matmul_bf16;
pub use combine::combine;
pub use delta::{DELTA_HEAD_DIM, delta_ba, delta_conv, delta_gnorm, delta_scan};
pub use dispatch::mv_vendored_supported;
pub use f16::matmul_f16;
pub use flash::flash_attn;
pub use mm_id::mul_mm_id;
pub use moe_glue::{moe_epilogue, moe_router, moe_router_supported};
pub use mv_id::{mul_mv, mul_mv_id, mv_classic};
pub use q8::matmul_q8;
pub use silu_mul::silu_mul;

pub use crate::gguf::ExpertStack;

use std::sync::OnceLock;

/// Prefill token-count threshold at/above which the fused MoE switches from
/// per-token mv_id to the mm_id two-pass matmul (ggml's mm_id break-even point).
/// Single source of truth: `moe` gates the prefill branch on it, and
/// `logits-dump` records it in each dump's provenance so the parity gate can
/// tell whether a dump actually exercised the mm_id path (do NOT re-hardcode 32).
pub const MM_ID_MIN_SEQ: usize = 32;

/// Effective mm_id threshold: `XWEN_MM_ID_MIN_SEQ=<n>` overrides the default
/// (probe/bench knob — e.g. forcing mm_id onto short speculative verify spans
/// to measure the mv_id/mm_id crossover). Value-parsed, read once and cached;
/// unset or unparsable falls back to `MM_ID_MIN_SEQ`. Dump provenance records
/// this effective value, so an overridden run can never masquerade as default.
pub fn mm_id_min_seq() -> usize {
    static V: OnceLock<usize> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("XWEN_MM_ID_MIN_SEQ")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(MM_ID_MIN_SEQ)
    })
}

/// `XWEN_NO_MM_ID=1` forces the per-token mv_id path everywhere (prefill
/// included), as a fallback / parity-debug switch. Read once and cached — it is
/// consulted per MoE layer on the hot path.
///
/// PRESENCE-BASED, like the `XWEN_MM_ID_*` variant toggles below: any value
/// (even `XWEN_NO_MM_ID=0`) enables it — only unset disables it.
pub(crate) fn no_mm_id() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var_os("XWEN_NO_MM_ID").is_some())
}

/// Public view of the cached `XWEN_NO_MM_ID` switch, for dump provenance.
pub fn no_mm_id_forced() -> bool {
    no_mm_id()
}

/// `XWEN_COMBINE_CLASSIC=1` reverts the routed-expert weighted combine from the
/// vendored fused kernel (`ops::combine`) back to candle's broadcast/affine/sum
/// chain. The fused kernel is bit-identical to that chain by construction, so this
/// is a safety kill-switch and provenance anchor, not a correctness tier.
///
/// PRESENCE-BASED and cached (read once), like the sibling MoE switches
/// (`no_mm_id`, `mv_classic`): any value enables it — only leaving it unset keeps
/// the fused path.
pub fn combine_classic() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var_os("XWEN_COMBINE_CLASSIC").is_some())
}

/// The active mm_id kernel variant's provenance name (e.g. `"tensor"`), for
/// dump provenance. Reflects the same cached `mm_id_variant()` the hot path uses.
pub fn active_mm_variant_name() -> &'static str {
    mm_id_variant().name()
}

/// Whether `CANDLE_METAL_ENABLE_FAST_MATH` is set FALSY — the one environment
/// under which candle compiles its Metal kernels Relaxed/Precise instead of its
/// default Fast/Fast. The vendored libraries are pinned `math_mode(fast)` at
/// the source level (see the .metal headers), so under that env every
/// bitwise-identity contract (combine, attn_glue, rope) would break SILENTLY —
/// `pipelines::compiled_pipeline` therefore hard-fails on the first vendored
/// kernel use rather than let a mixed-mode process run. Truthiness mirrors
/// candle's own parse (candle-metal-kernels utils.rs `is_truthy`: exactly
/// "true"/"t"/"yes"/"y"/"1" are truthy; anything else — "0", "false", "" — is
/// falsy). Cached (read once), like the sibling env switches.
pub(crate) fn candle_fast_math_disabled() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| match std::env::var("CANDLE_METAL_ENABLE_FAST_MATH") {
        Ok(v) => !matches!(v.as_str(), "true" | "t" | "yes" | "y" | "1"),
        Err(_) => false,
    })
}

/// `XWEN_ACT_CLASSIC=1` reverts the routed-expert SwiGLU activation from the
/// vendored fused kernel (`ops::silu_mul`) back to candle's `silu(gate) * up`
/// two-op chain. The fused kernel is bit-identical to that chain by construction,
/// so this is a safety kill-switch and provenance anchor, not a correctness tier.
///
/// PRESENCE-BASED and cached (read once), like the sibling MoE switches
/// (`combine_classic`, `no_mm_id`): any value enables it — only leaving it unset
/// keeps the fused path.
pub fn act_classic() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var_os("XWEN_ACT_CLASSIC").is_some())
}

/// `XWEN_MOE_GLUE_CLASSIC=1` reverts the MoE block glue — the fused routing
/// decision (`ops::moe_router`), the fused block epilogue (`ops::moe_epilogue`)
/// and the shared expert's fused SwiGLU activation (`ops::silu_mul`) — back to
/// the candle chains they replace. ONE switch covers all three: each fused
/// kernel is bit-identical to its candle chain by construction, so this is a
/// safety kill-switch and provenance anchor, not a correctness tier. The routed
/// experts' own activation and combine keep their older, narrower switches
/// (`XWEN_ACT_CLASSIC`, `XWEN_COMBINE_CLASSIC`), which still apply on the
/// classic branch.
///
/// PRESENCE-BASED and cached (read once), like the sibling switches
/// (`combine_classic`, `attn_glue_classic`): any value enables it — only leaving
/// it unset keeps the fused path.
pub fn moe_glue_classic() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var_os("XWEN_MOE_GLUE_CLASSIC").is_some())
}

/// `XWEN_MOE_DUAL=1` opts the routed experts' gate|up gather INTO the
/// dual-weight kernel (`ops::mul_mv_id_dual`), which computes both projections
/// and their SwiGLU activation in one dispatch instead of three. It is
/// bit-identical to the split chain, so this is a perf switch, not a
/// correctness tier — and it is OFF by default because it measured SLOWER:
/// interleaved A/B on the 35B-A3B put decode at 102.8 tok/s with the split
/// chain against 99.5 tok/s with the dual kernel, 5 reps of 5 apart (see
/// docs/decisions.md "The dual-weight expert gather"). Kept because it is a
/// measured, gated artifact worth re-pricing on a different device or after the
/// gather's occupancy changes — not because anything runs it today.
///
/// PRESENCE-BASED and cached (read once), like the sibling switches: any value
/// enables it — only leaving it unset keeps the split chain.
pub fn moe_dual() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var_os("XWEN_MOE_DUAL").is_some())
}

/// `XWEN_ATTN_GLUE_CLASSIC=1` reverts the attention glue — the fused softplus
/// gate (`ops::attn_gate`), the fused permute/cast copies
/// (`ops::permute_01`/`cast_*`), and the fused partial-rotary rope
/// (`ops::rope_neox`) — back to the candle chains they replace
/// (softplus + broadcast_mul, transpose().contiguous() + to_dtype, and the
/// narrow/contiguous/rope/cat rope path). ONE switch covers all three: each
/// fused kernel is bit-identical to its candle chain by construction, so this
/// is a safety kill-switch and provenance anchor, not a correctness tier.
///
/// PRESENCE-BASED and cached (read once), like the sibling switches
/// (`combine_classic`, `no_mm_id`): any value enables it — only leaving it
/// unset keeps the fused path.
pub fn attn_glue_classic() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var_os("XWEN_ATTN_GLUE_CLASSIC").is_some())
}

/// `XWEN_FLASH_CLASSIC=1` reverts the prefill (seq > 1) attention from the
/// vendored flash kernel (`ops::flash_attn` — in-kernel masking, no
/// materialized mask tensor) back to the candle sdpa chain (f16 cast +
/// materialized `PrefillMask` + `candle_nn::ops::sdpa` + f32 cast) —
/// byte-for-byte the pre-flash behavior, including the `XWEN_SDPA_F32`
/// experiment hook. Decode (seq == 1) always runs the sdpa vector path and is
/// unaffected. The parity gates pin provenance `flash` to "classic" on
/// references and the strict tier (parity-gate.ts referenceEnv()).
///
/// PRESENCE-BASED and cached (read once), like the sibling switches
/// (`attn_glue_classic`, `combine_classic`): any value enables it — only
/// leaving it unset keeps the fused flash path.
pub fn flash_classic() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var_os("XWEN_FLASH_CLASSIC").is_some())
}

/// `XWEN_DELTA_CLASSIC=1` reverts the gated-DeltaNet layers from the vendored
/// fused kernels (`ops::delta_*` — conv+silu+state, beta/decay head, recurrent
/// scan, gated output norm, plus the load-time fused beta|alpha projection)
/// back to `LinearAttnBlock::forward_classic`, the frozen reference scan: the
/// composed-candle recurrent form, one step per token.
///
/// Unlike the combine/act/glue switches this one is NOT a bit-identity anchor.
/// The scan kernel partitions the k- and q-contractions across threads where
/// the reference runs a candle gemm, and folds the q/k L2 norm through
/// simd_sum, so its result is bounded-close rather than bitwise — which is why
/// the parity gate pins this switch on BOTH sides of the strict tier and lets
/// the mm / decode / ppl tiers carry the real signal (docs/parity.md).
///
/// PRESENCE-BASED and cached (read once), like the sibling switches
/// (`combine_classic`, `flash_classic`): any value enables it — only leaving it
/// unset keeps the fused path.
pub fn delta_classic() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var_os("XWEN_DELTA_CLASSIC").is_some())
}

/// `XWEN_SDPA_F32` runs the sdpa attention kernel in f32 instead of the
/// shipped f16: q skips its f16 cast, the cached f16 k/v are widened exactly,
/// and candle's Metal sdpa dispatches its float32 kernels (supported at the
/// pinned rev for head_dim 128 + GQA, full and vector). An EXPERIMENT hook for
/// numerics work (e.g. isolating sdpa-precision drift), NOT a shipping path —
/// the parity gates pin provenance `sdpa` to "f16" unless the run opts in via
/// `XWEN_PARITY_EXPECT_SDPA` (see docs/parity.md §3b).
///
/// PRESENCE-BASED and cached (read once), like the sibling switches
/// (`no_mm_id`, `combine_classic`): any value enables it — only leaving it
/// unset keeps the f16 default.
pub fn sdpa_f32() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var_os("XWEN_SDPA_F32").is_some())
}

/// `XWEN_ATTN_MM_CLASSIC` reverts the attention PREFILL gemm — the mm branch
/// (ne11 > 8) of `matmul_f16` — from the DEFAULT Metal-4 cooperative-tensor
/// kernel (`f16_t.metal`'s `kernel_mul_mm_f16_f32_t`) back to the classic
/// simdgroup kernel (`f16.metal`'s `kernel_mul_mm_f16_f32_v`). The tensor
/// kernel stages the activation as f16 — one extra rounding over the classic
/// float tiles, the same precision class as the fork's own prefill (see
/// docs/parity.md §3b) — so this kill-switch exists for A/B numerics work. The
/// decode gemv branch (ne11 <= 8) is unaffected — it always runs the classic mv.
/// Orthogonal to `XWEN_ATTN_F32`, which bypasses the whole f16 library for
/// the legacy dequant-f32 QMatMul path; when that is set this switch is moot
/// (the mm branch never runs).
///
/// PRESENCE-BASED and cached (read once), like the sibling switches (`no_mm_id`,
/// `combine_classic`, `flash_classic`): any value enables it — only leaving it
/// unset keeps the tensor default.
pub(crate) fn attn_mm_classic() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var_os("XWEN_ATTN_MM_CLASSIC").is_some())
}

/// `XWEN_ATTN_DEQUANT` disables the q8_0 attention DECODE gemv (`ops::matmul_q8`)
/// for a q8_0-quantized checkpoint, sending the decode projections back through
/// the dequantized f16 dense plane (`ops::matmul_f16`) — byte-identical to the
/// pre-fast-path fallback the prefill/mm branch already uses. A kill-switch and
/// provenance anchor for a q8_0-attention checkpoint's decode (the current
/// official file — so this gemv is the production default; the unsloth UD file
/// that introduced it is deleted); on an f16-attention checkpoint (the retired
/// original) it is a no-op (there is no q8_0 alias, so decode
/// always ran the f16 gemv). Orthogonal to `XWEN_ATTN_F32`, which bypasses the
/// f16/q8 libraries entirely for the legacy dequant-f32 QMatMul path and takes
/// precedence (its `AttnWeights::DequantF32` never builds a q8_0 alias).
///
/// PRESENCE-BASED and cached (read once), like the sibling switches
/// (`attn_mm_classic`, `flash_classic`): any value enables it — only leaving it
/// unset keeps the q8_0 decode gemv.
pub(crate) fn attn_dequant() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var_os("XWEN_ATTN_DEQUANT").is_some())
}

/// The mm_id prefill kernel family. Runtime-selectable via env; the single
/// source of truth (both the kernel selection in dispatch and the rescale
/// decision in moe read the cached value here).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MmVariant {
    /// The fork's cooperative-tensor path (`_t`): Metal-4 `matmul2d`, f16 operand
    /// tiles. Default. Casts the activation to f16, so it needs the L2 rescale
    /// guard, and its parity sits in the fork-equivalent mm tier.
    Tensor,
    /// Cooperative-tensor path with f32 operand tiles (`_t_hp`): `matmul2d` on
    /// float cooperative tensors. No f16 cast (no rescale); still tiled, so mm
    /// tier. Only instantiated for q4_K/q6_K (covers the current official
    /// file's all-q4_K experts; q6_K was the retired original's expert-down
    /// dtype).
    TensorHp,
    /// Classic simdgroup tiles in f32 (`_hp`): no f16 cast, no rescale.
    ClassicHp,
    /// Classic simdgroup tiles in f16 (base name): f16 operand cast, needs rescale.
    ClassicF16,
}

impl MmVariant {
    /// Kernel host-name suffix appended to `kernel_mul_mm_id_<dtype>_f32`.
    pub(crate) fn suffix(self) -> &'static str {
        match self {
            MmVariant::Tensor => "_t",
            MmVariant::TensorHp => "_t_hp",
            MmVariant::ClassicHp => "_hp",
            MmVariant::ClassicF16 => "",
        }
    }

    /// Threadgroup tile bytes: the f32-tile variants (`_hp`, `_t_hp`) need 12288,
    /// the half-tile variants 8192 (sa+sb+float store-back tile).
    pub(crate) fn tile_smem(self) -> usize {
        match self {
            MmVariant::ClassicHp | MmVariant::TensorHp => 12288,
            MmVariant::Tensor | MmVariant::ClassicF16 => 8192,
        }
    }

    /// Whether this variant casts the down-projection activation to f16 (so the
    /// L2 rescale guard is required). The f32-tile variants (`_hp`, `_t_hp`) do not.
    pub(crate) fn casts_activation_f16(self) -> bool {
        matches!(self, MmVariant::Tensor | MmVariant::ClassicF16)
    }

    /// Stable provenance name (distinct from `suffix`, which is a kernel-name
    /// fragment and empty for `ClassicF16`).
    pub(crate) fn name(self) -> &'static str {
        match self {
            MmVariant::Tensor => "tensor",
            MmVariant::TensorHp => "tensor_hp",
            MmVariant::ClassicHp => "classic_hp",
            MmVariant::ClassicF16 => "classic_f16",
        }
    }
}

/// Which mm_id variant to run, cached (read once). Precedence: `XWEN_MM_ID_F16`
/// → classic f16 tiles; `XWEN_MM_ID_CLASSIC` → classic f32 (`_hp`) tiles;
/// `XWEN_MM_ID_TENSOR_HP` → f32 tensor tiles (`_t_hp`); else the f16 tensor
/// path (default). The tensor kernels compile on this device (the mm_id.metal
/// probe test gates that); the other variants remain for A/B.
///
/// PRESENCE-BASED toggles: each is enabled by the env var merely being SET, whatever
/// its value — `XWEN_MM_ID_F16=0` still selects the f16 classic tiles. To disable a
/// variant, UNSET its var (do not set it to `0`/`false`). First set var in the
/// precedence order above wins.
pub(crate) fn mm_id_variant() -> MmVariant {
    static V: OnceLock<MmVariant> = OnceLock::new();
    *V.get_or_init(|| {
        if std::env::var_os("XWEN_MM_ID_F16").is_some() {
            MmVariant::ClassicF16
        } else if std::env::var_os("XWEN_MM_ID_CLASSIC").is_some() {
            MmVariant::ClassicHp
        } else if std::env::var_os("XWEN_MM_ID_TENSOR_HP").is_some() {
            MmVariant::TensorHp
        } else {
            MmVariant::Tensor
        }
    })
}

/// Which expert-FFN implementation a model is built with.
/// Fused dispatches candle's kernel_mul_mv_id_*/mm_id_* Metal kernels over the
/// stacked quantized tensors (ids stay on GPU); Reference slices the stack into
/// per-expert QTensors with a CPU id readback — slow, but the correctness oracle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertRunner {
    Fused,
    Reference,
}
