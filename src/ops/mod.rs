pub mod attn_glue;
pub mod bf16;
pub mod combine;
pub mod delta;
pub mod dense_mm;
mod dispatch;
pub mod f16;
pub mod flash;
pub mod hc;
pub mod mm_id;
pub mod moe_glue;
pub mod mv_ext;
pub mod mv_id;
mod pipelines;
pub mod q8;
pub mod qsa_gather;
pub mod silu_mul;

pub use attn_glue::{attn_gate, cast_f16, cast_f32, permute_01, permute_01_f16, rope_neox};
pub use bf16::matmul_bf16;
pub use combine::combine;
pub use delta::{
    DELTA_HEAD_DIM, delta_ba, delta_ba_fused, delta_ba_fused_applies, delta_conv, delta_gnorm,
    delta_l2norm, delta_scan, delta_scan_with_trail,
};
pub use dense_mm::{dense_mm_supported, matmul_dense_q};
pub use dispatch::mv_vendored_supported;
pub use f16::matmul_f16;
pub use flash::flash_attn;
pub use hc::{hc_mix, hc_norm, hc_norm_supported, hc_silu_quarter, hc_write};
pub use mm_id::mul_mm_id;
pub use moe_glue::{moe_epilogue, moe_router, moe_router_supported};
pub use mv_ext::{matmul_mv_ext, mv_ext_supported};
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

/// Token count the dense checkpoint's SwiGLU FFN must EXCEED for prefill to
/// switch from candle's `QMatMul` to the vendored dense cooperative-tensor gemm
/// (`ops::matmul_dense_q`). Exclusive, like `F16_MM_MIN_SEQ` and ggml's own
/// `ne11_mm_min`, not inclusive like `MM_ID_MIN_SEQ`.
///
/// Set from the measured crossover at the 27B's production FFN shapes, and it
/// falls exactly on a tile boundary rather than a round number: candle's kernel
/// tiles tokens 32 wide, the vendored one 128 wide, so up to 32 tokens both fit
/// a single token tile and run at the same launch-latency floor (measured
/// 1.01-1.05x, i.e. a wash), while at 33 candle takes a second tile and the
/// vendored gemm does not — 1.20x there, rising to 2.4-3.0x at a 512-token
/// chunk (docs/decisions.md, "The dense-FFN prefill gemm"). Below the boundary
/// there is no throughput reason to take the vendored kernel, and a positive
/// reason not to: it is the less accurate of the two. Decode is untouched.
///
/// Single source of truth: `DenseMlp` gates on it and `logits-dump` records the
/// effective value in dump provenance.
pub const DENSE_MM_MIN_SEQ: usize = 32;

/// Effective dense-FFN prefill threshold: `XWEN_DENSE_MM_MIN_SEQ=<n>` overrides
/// the default (probe/bench knob — e.g. forcing the vendored gemm onto short
/// spans to re-measure the crossover). Value-parsed, read once and cached;
/// unset or unparsable falls back to `DENSE_MM_MIN_SEQ`.
pub fn dense_mm_min_seq() -> usize {
    static V: OnceLock<usize> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("XWEN_DENSE_MM_MIN_SEQ")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DENSE_MM_MIN_SEQ)
    })
}

/// `XWEN_DENSE_MM_CLASSIC=1` reverts the dense checkpoint's SwiGLU FFN prefill
/// from the vendored Metal-4 cooperative-tensor gemm (`ops::matmul_dense_q`,
/// dense_mm.metal) back to candle's `QMatMul` chain — the path every token took
/// before, and the one decode still takes at every seq.
///
/// Like `XWEN_ATTN_MM_CLASSIC` and unlike the combine/act/glue switches, this is
/// NOT a bit-identity anchor, and it is not even neutral: the vendored kernel
/// runs matmul2d's reduced-precision tensor-core path, which is where its
/// throughput comes from, so it sits ~4.1e-4 rel_l2 from the f32 oracle at the
/// 27B FFN shapes where the `QMatMul` chain sits ~1.9e-4 (the two differ by
/// ~3.7e-4). That is the fork's own prefill precision class — llama.cpp sets the
/// same descriptor flag for its dense FFN prefill, and the attention prefill
/// gemm already made the identical trade — which is why the parity gate pins
/// this switch on BOTH sides of the strict tier and lets the mm / decode / ppl
/// tiers carry the signal (docs/parity.md).
///
/// PRESENCE-BASED and cached (read once), like the sibling switches
/// (`attn_mm_classic`, `flash_classic`): any value enables it — only leaving it
/// unset keeps the vendored gemm.
pub fn dense_mm_classic() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var_os("XWEN_DENSE_MM_CLASSIC").is_some())
}

/// `XWEN_STACK_PROFILE` turns on the in-situ per-stage timing of the forward
/// stack (`stack_profile`): every stage of `run_stack` plus the lm-head tail is
/// bracketed by device syncs, each chunk's wall clock is measured the same way,
/// and the difference between the two is reported as unaccounted time. It is
/// DIAGNOSIS ONLY — no arithmetic changes — but the syncs serialize the whole
/// pipeline, so a profiled run's absolute throughput means nothing.
///
/// PRESENCE-BASED and cached (read once), like the sibling switches
/// (`dense_mm_classic`, `combine_classic`): any value enables it. Unset — the
/// normal case — costs one `Option` check at each instrumented site.
pub fn stack_profile() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var_os("XWEN_STACK_PROFILE").is_some())
}

/// `XWEN_PLE_PROFILE` breaks ONE `Stage::Ple` bracket into its sub-steps and
/// prints a line per forward: the n-gram hash, the mmap row gather + IQ4_NL
/// dequant, the embedding upload, the device projections, the device→host
/// readback, the host gate, the host conv, the speculative trail, and the
/// addend upload.
///
/// `stack_profile` prices the PLE layer as one number; this says which half of
/// the D17 hybrid that number is. The device sub-steps are bracketed by
/// `Device::synchronize` on the same contract `stack_profile` uses — a device
/// step's figure is completed GPU work, not enqueue time — and the forward
/// opens with a sync too, so an inherited backlog is charged to the caller's
/// bracket rather than to whichever sub-step first drains it.
///
/// The line also carries the row count fetched from the table and how many of
/// those rows were DISTINCT, which is the difference between "the gather is
/// arithmetic" and "the gather is page faults": a decode step's 16 rows are
/// 16 unrelated 90-byte reads scattered over a 28.8 GB mapping.
///
/// PRESENCE-BASED and cached (read once), like the sibling switches
/// (`stack_profile`, `dense_mm_classic`): any value enables it. Unset — the
/// normal case — costs one `Option` check per sub-step and reads no clock at
/// all. A forward that errors prints nothing, so a failed run's last PLE call
/// is invisible here.
pub fn ple_profile() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var_os("XWEN_PLE_PROFILE").is_some())
}

/// `XWEN_GDN_PROFILE` breaks the `Stage::MixerDelta` bracket into the steps a
/// gated-DeltaNet block actually runs — the three big projections, the fused
/// conv, the beta/decay head, the recurrent scan, the gated output norm — and
/// prints one line per forward with every GDN layer's steps folded together
/// (`gdn_profile`).
///
/// `stack_profile` says the DeltaNet mixer costs N ms; this says which kernel
/// inside it does. The device steps are bracketed by `Device::synchronize` on
/// the same contract `stack_profile` uses — a step's figure is completed GPU
/// work, not enqueue time — and each block opens with a sync, so a backlog
/// inherited from the caller is charged to the caller rather than to whichever
/// step first drains it.
///
/// The line also carries each step's DECLARED bytes as an achieved GB/s, which
/// is the difference between "this step moves the bytes it must and is done"
/// and "this step is nowhere near its own byte floor". It is a floor, not a
/// hardware peak: this machine's peak bandwidth has never been measured.
///
/// PRESENCE-BASED and cached (read once), like the sibling switches
/// (`stack_profile`, `ple_profile`): any value enables it. Unset — the normal
/// case — costs one `Option` check per step and reads no clock at all. The
/// syncs serialize the block, so a profiled run's absolute throughput means
/// nothing.
pub fn gdn_profile() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var_os("XWEN_GDN_PROFILE").is_some())
}

/// `XWEN_GDN_REPS=N` makes each `XWEN_GDN_PROFILE` step run its work N times
/// inside its own bracket, and the line divide by N.
///
/// The reason is the floor: a command buffer that carries work costs ~0.2 ms to
/// commit and wait for on this machine, and at decode most of a DeltaNet
/// block's steps do LESS GPU work than that, so a one-shot bracket measures the
/// round trip and reports the kernel. Repeating amortizes it — the floor is
/// paid once per bracket and divided by N — which is the benching rule this
/// repo already holds elsewhere (CLAUDE.md: amortized rates, never
/// per-dispatch).
///
/// Every repeated step is a pure function of tensors the block already holds,
/// so the extra rounds recompute and discard; nothing that advances the cache
/// is ever repeated. Default 1 (one-shot, floor-dominated at decode). Read once
/// and cached; inert unless `XWEN_GDN_PROFILE` is also set.
pub fn gdn_reps() -> usize {
    static V: OnceLock<usize> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("XWEN_GDN_REPS")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|n| *n >= 1)
            .unwrap_or(1)
    })
}

/// `XWEN_PLE_NO_RANDOM` keeps the PLE n-gram table's byte range on the
/// mapping's default (sequential-ish) readahead instead of tagging it
/// `MADV_RANDOM`.
///
/// The A/B knob for that hint, and the reason it is a knob: the table is read
/// 16 unrelated 90-byte rows per token over 28.8 GB, which is the textbook case
/// for `MADV_RANDOM` — but readahead that misses is only wasted bandwidth,
/// while readahead that hits is a fault the gather never takes, and the two
/// cannot be told apart without measuring. Set this and compare the `gather`
/// figure under `XWEN_PLE_PROFILE`.
///
/// PRESENCE-BASED and cached (read once), like the sibling switches: any value
/// enables it.
pub fn ple_no_random() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var_os("XWEN_PLE_NO_RANDOM").is_some())
}

/// `XWEN_PLE_NO_PREFETCH` disables the PLE table's background page-touching
/// thread, so every gather takes its own faults on the forward's own thread.
///
/// The other half of the A/B: with it set the layer is exactly what it was
/// before the prefetcher existed. Unset, each `PleTable` — i.e. each PLE layer,
/// of which the shipped checkpoint has exactly one — spawns one thread on its
/// first hint and never spawns another.
///
/// PRESENCE-BASED and cached (read once), like the sibling switches: any value
/// enables it.
pub fn ple_no_prefetch() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var_os("XWEN_PLE_NO_PREFETCH").is_some())
}

/// `XWEN_CHUNK_SYNC` makes the plain prefill loop wait for each chunk's forward
/// to complete before enqueueing the next, instead of letting the chunks
/// pipeline.
///
/// It is an A/B probe for prefill cost that accumulates ACROSS chunks rather
/// than inside them. A sync is the only thing that clears candle-metal's fence
/// map and encoder barrier history and prunes its buffer pool, so state that
/// grows chunk over chunk — a pool that only ever adds entries because each
/// chunk's mask upload asks for a fresh exact-size buffer, and the barrier
/// storms that recycled pool pointers trigger — is present in a pipelined run
/// and absent in a synced one. The difference between the two runs is that
/// state's cost; neither run's arithmetic differs, and the sync itself is the
/// only thing being added.
///
/// PRESENCE-BASED and cached (read once), like the sibling switches
/// (`stack_profile`, `dense_mm_classic`): any value enables it.
pub fn chunk_sync() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var_os("XWEN_CHUNK_SYNC").is_some())
}

/// `XWEN_PREFILL_CHUNK=<usize>` overrides the prefill chunk — how many prompt
/// tokens every prefill path (`generate`/`chat`/`batch`, serve, the ppl pass)
/// feeds the model per forward. The default is per architecture
/// (`Arch::prefill_chunk_default`, read through `XwenModel::prefill_chunk`);
/// this is the A/B knob over it. Cached (read once); unset, unparseable or
/// zero means "no override".
pub fn prefill_chunk_override() -> Option<usize> {
    static V: OnceLock<Option<usize>> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("XWEN_PREFILL_CHUNK")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .filter(|&n| n > 0)
    })
}

/// Largest token count routed to the vendored small-batch mat-vec
/// (`ops::matmul_mv_ext`). Inclusive, and it is ggml's own tested envelope: its
/// host dispatches `mul_mv_ext` for ne11 in 2..=8 and the tiled gemm above that
/// (ggml-metal-ops.cpp:2120-2223). Below 2 there is nothing to amortize — one
/// token is the plain gemv's job.
pub const MV_EXT_MAX_SEQ: usize = 8;

/// Effective small-batch ceiling: `XWEN_MV_EXT_MAX_SEQ=<n>` overrides the
/// default. A PROBE knob — the window's upper edge was inherited from ggml's
/// tuning, not measured on this device, and the next question about this kernel
/// is whether it also beats the gemm above 8. Value-parsed, read once and
/// cached; unset or unparsable falls back to `MV_EXT_MAX_SEQ`.
pub fn mv_ext_max_seq() -> usize {
    static V: OnceLock<usize> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("XWEN_MV_EXT_MAX_SEQ")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(MV_EXT_MAX_SEQ)
    })
}

/// The small-batch plan for a `seq`-token matmul: `Some(r1ptg)` to route to
/// `ops::matmul_mv_ext` with that many src1 rows per threadgroup, `None` to
/// leave the call site on the path it had. Every routing site asks this and
/// nothing else, so the kill-switch cannot be forgotten at one of them.
///
/// This is the SINGLE authority on the plan: the width it returns is handed to
/// `matmul_mv_ext` verbatim and never re-derived downstream, so a token count
/// admitted here can never be one the dispatcher rejects.
///
/// Inside ggml's 2..=8 window the width is ggml's own choice
/// (`dispatch::mv_ext_r1ptg`). Above it — reachable only by raising
/// `XWEN_MV_EXT_MAX_SEQ`, since the default ceiling is 8 — the width is 4, the
/// widest ggml uses for a full batch and a divisor of the counts a probe would
/// try; the kernel's ragged-token guard covers the remainder. That extension is
/// xwen's, not ggml's, and exists to make the "does this beat the gemm higher
/// up?" question measurable without a rebuild.
pub fn mv_ext_window(seq: usize) -> Option<usize> {
    if mv_ext_classic() || seq < 2 || seq > mv_ext_max_seq() {
        return None;
    }
    Some(mv_ext_plan(seq))
}

/// The width alone, with neither the ceiling nor the kill-switch applied:
/// ggml's table inside its window, xwen's r1ptg-4 extension above it. Split out
/// of `mv_ext_window` so tests can drive the extension without depending on the
/// process-wide cached env ceiling — and so the extension's default lives in
/// exactly one place.
pub(crate) fn mv_ext_plan(seq: usize) -> usize {
    dispatch::mv_ext_r1ptg(seq).unwrap_or(4)
}

/// `XWEN_MV_EXT_CLASSIC=1` reverts every small-batch routing decision — the
/// dense FFN below the dense-mm threshold, the q8_0 `QLinear` projections, the
/// lm_head, and the q8_0 attention/DeltaNet projections (`Proj::DenseF16Q8`,
/// which covers attn_q/k/v/output, attn_qkv, attn_gate and ssm_out) — from the
/// vendored `mul_mv_ext` kernels (`ops::matmul_mv_ext`, mv_ext.metal) back to
/// the path each site had before: candle's `QMatMul` (which runs its `mul_mm`
/// branch at these token counts), or the plain vendored gemv where that was
/// already in use.
///
/// Like `XWEN_DENSE_MM_CLASSIC` and `XWEN_DELTA_CLASSIC`, and unlike the
/// combine/act/glue switches, this is NOT a bit-identity anchor: the kernel
/// changes the K reduction's summation order (eight partial sums per row
/// instead of one lane-32 fold), so it is bounded-close rather than bitwise.
/// Unlike those two it is bounded-close on the CLOSER side — nothing is
/// narrowed, so it sits 4e-7..8e-6 from the f32 oracle where the `QMatMul`
/// chain it replaces sits ~1.8e-4 (candle stages its dequantized tile as half).
/// A non-bitwise change either way, which is why the parity gate pins this
/// switch on BOTH sides of the strict tier and lets the mm / decode / ppl tiers
/// carry the signal (docs/parity.md).
///
/// PRESENCE-BASED and cached (read once), like the sibling switches
/// (`dense_mm_classic`, `flash_classic`): any value enables it — only leaving
/// it unset keeps the vendored kernels.
pub fn mv_ext_classic() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var_os("XWEN_MV_EXT_CLASSIC").is_some())
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

/// `XWEN_HC_CLASSIC=1` reverts the qwen4exp hyper-connection gates — the
/// carrier's grouped norm and injection head (`ops::hc_norm`), the bottleneck
/// activation (`ops::hc_silu_quarter`), the stream mix (`ops::hc_mix`) and the
/// write-back (`ops::hc_write`) — to the candle chains they replace. ONE switch
/// covers all four, because they are one fused read gate and its write side.
///
/// The activation and the write-back are bit-identical to their chains by
/// construction; the norm and the mix partition reductions those chains run in
/// one order, so they are bounded rather than bitwise (hc.metal, and the
/// tolerances the hc.rs tests grade at). The two Q8_0 bottleneck matmuls are
/// outside the switch — both paths run the same `QLinear`.
///
/// PRESENCE-BASED and cached (read once), like the sibling switches
/// (`combine_classic`, `attn_glue_classic`): any value enables it — only leaving
/// it unset keeps the fused path.
pub fn hc_classic() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var_os("XWEN_HC_CLASSIC").is_some())
}

/// Kill switch (`XWEN_QSA_CLASSIC`) for the QSA indexer's fast decode path on
/// a qwen4exp checkpoint: the cached block-key plane (every complete block's
/// pooled+normed+roped key built once, `indexer::IndexerCache`) and the fused
/// K/V row gather (`ops::qsa_gather`). Set, `QsaIndexer::select` recomputes
/// every block key from the raw rows on each call and the attention gathers
/// its selected rows through the per-head `index_select` chain. Both arms are
/// BIT-IDENTICAL by construction (the pool replays candle's strided-reduce
/// order, the gather is a copy), so this is a fallback, never a parity row.
///
/// PRESENCE-BASED and cached (read once), like the sibling switches
/// (`hc_classic`, `attn_glue_classic`): any value enables it — only leaving it
/// unset keeps the fast path.
pub fn qsa_classic() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var_os("XWEN_QSA_CLASSIC").is_some())
}

/// Token count the carrier's grouped norm must be BELOW for `ops::hc_norm` to
/// take the split launch — `kernel_hc_norm_split` plus, on a gated block,
/// `kernel_hc_inject`, one threadgroup per (token, stream) each — instead of
/// the single-threadgroup-per-token `kernel_hc_norm[_inject]`.
///
/// The single kernel reads the carrier once and folds every reduction in one
/// threadgroup, which is the cheaper shape as soon as the token grid alone
/// fills the machine. Below that it leaves the GPU idle: a decode step is 97
/// launches of ONE 256-thread threadgroup, each walking a 10240-wide carrier
/// twice and, on a gated block, the whole [hc_count, width] injection head.
/// The split pair costs an extra read of `normed` and buys `hc_count` times the
/// parallelism.
///
/// Exclusive, and both arms compute the same bits, so this is purely a launch
/// -shape choice — moving it can never change a result.
pub const HC_SPLIT_MAX_N: usize = 32;

/// Effective split-launch ceiling: `XWEN_HC_SPLIT_MAX_N=<n>` overrides the
/// default (an A/B knob for the threshold — 0 pins every batch to the single
/// kernel, a large value pins every batch to the split pair). Value-parsed,
/// read once and cached; unset or unparsable falls back to
/// [`HC_SPLIT_MAX_N`].
pub fn hc_split_max_n() -> usize {
    static V: OnceLock<usize> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("XWEN_HC_SPLIT_MAX_N")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(HC_SPLIT_MAX_N)
    })
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

/// `XWEN_DELTA_DECODE_KERNEL=1` routes the seq == 1 gated-DeltaNet step to
/// `kernel_delta_scan_decode`, the decode-specialized scan, instead of the
/// general `kernel_delta_scan` every length takes by default. The two run the
/// same math on the same operands and leave the same single state plane — the
/// decode kernel drops the timestep loop, moves the state as float4 and folds
/// its row slices inside a simdgroup — so this is an A/B knob, not a numerics
/// anchor: both arms are BOUNDED against the reference in the same class, and
/// `XWEN_DELTA_CLASSIC` remains the switch that takes a run all the way back to
/// the frozen reference scan.
///
/// OPT-IN because it is a measured WASH end to end (44.6-44.7 tok/s against the
/// general kernel's 44.7-44.8 on Flash-Next, 105.4 against 105.5 on the
/// 35B-A3B, byte-identical greedy text either way), kept for the bench arm and
/// the refutation it carries rather than for a speedup — docs/decisions.md, "A
/// decode-specialized scan kernel is a WASH". Its `#[ignore]`d bench is
/// `delta_scan_decode_timing` (src/ops/delta.rs), which calls both kernels
/// directly and needs no switch.
///
/// Prefill and any multi-token verify chunk are unaffected — they never reach
/// the decode kernel — so a run that sets this differs only in the fold order
/// of its DECODE arithmetic.
///
/// PRESENCE-BASED and cached (read once), like the sibling switches: any value
/// enables it — only leaving it unset keeps the general kernel everywhere.
pub fn delta_decode_kernel() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var_os("XWEN_DELTA_DECODE_KERNEL").is_some())
}

/// `XWEN_DELTA_BA_CLASSIC=1` splits the fused beta|alpha projection back into
/// the two dispatches it replaced: a candle f32 gemv over the `[hidden, 2 *
/// v_heads]` weight, then `kernel_delta_ba` over its output. Only the shape of
/// that one step changes — both arms run the same `LinearAttnBlock` fused path,
/// and a prefill chunk takes the gemv either way (the fused kernel is confined
/// to small token counts, `dispatch::DELTA_BA_MAX_SEQ`).
///
/// Like `XWEN_DELTA_CLASSIC` and unlike the combine/act/glue switches, this is
/// NOT a bit-identity anchor: the fused kernel sums each dot product as
/// per-thread partials folded in a tree where candle's gemv sums in its own
/// order, so the two arms agree to ~1e-6 rather than bitwise. The epilogue is
/// the same Metal helper on both sides.
///
/// PRESENCE-BASED and cached (read once), like the sibling switches
/// (`delta_classic`, `flash_classic`): any value enables it — only leaving it
/// unset keeps the fused kernel.
pub fn delta_ba_classic() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var_os("XWEN_DELTA_BA_CLASSIC").is_some())
}

/// `XWEN_DELTA_SCAN_V2=1` runs the gated-DeltaNet recurrence through
/// `kernel_delta_scan_v2` and the `ops::delta_l2norm` dispatch it needs, instead
/// of the shipped single-dispatch `kernel_delta_scan`. That pair is llama.cpp's
/// Metal decomposition of the same recurrence, adapted to our layouts: a
/// SIMDGROUP owns each state value-column end to end, both key-dim contractions
/// collapse to `simd_sum`, no barrier appears anywhere in the timestep loop, and
/// the grid is eight times as many threadgroups.
///
/// NOT the default, on measurement: giving every simdgroup its own column also
/// gives it its own copy of the per-timestep q and k reads, and that traffic
/// costs more than the barriers and the redundant in-register norm it removes
/// (27B geometry, seq 4096: 14.81 ms inclusive of its norm dispatch — ~1.80 ms
/// of it — against the shipped kernel's 8.56 ms; see docs/decisions.md, "The
/// DeltaNet scan decomposition").
/// Kept as a measured artifact rather than deleted, like `XWEN_MOE_DUAL`,
/// because llama.cpp's kernel invites the same proposal again.
///
/// PRESENCE-BASED and cached (read once), like the sibling switches: any value
/// enables it — only leaving it unset keeps the shipped kernel.
pub fn delta_scan_v2() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var_os("XWEN_DELTA_SCAN_V2").is_some())
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
