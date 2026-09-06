pub mod attn_glue;
pub mod bandwidth;
pub mod bf16;
pub mod combine;
pub mod delta;
pub mod dense_mm;
mod dispatch;
pub mod f16;
pub mod f32_mv;
pub mod flash;
pub mod hc;
pub mod mm_id;
pub mod moe_glue;
pub mod mv_ext;
pub mod mv_id;
mod pipelines;
pub mod ple;
pub mod q8;
pub mod qsa_gather;
pub mod qsa_select;
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
pub(crate) use dispatch::view_offset_aligned_16;
pub use f16::matmul_f16;
pub use f32_mv::{matmul_f32, matmul_f32_supported};
pub use flash::flash_attn;
pub use hc::{
    hc_gate_down, hc_gate_fused_supported, hc_gate_up_mix, hc_mix, hc_norm, hc_norm_supported,
    hc_silu_quarter, hc_write,
};
pub use mm_id::mul_mm_id;
pub use moe_glue::{
    moe_epilogue, moe_epilogue_shexp, moe_router, moe_router_supported, moe_shexp_fused_supported,
    moe_shexp_gate_up, moe_shexp_plane_bindable,
};
pub use mv_ext::{matmul_mv_ext, mv_ext_supported};
pub use mv_id::{mul_mv, mul_mv_id, mv_classic};
pub use q8::matmul_q8;
pub use silu_mul::{silu_mul, silu_mul_l2, silu_mul_l2_supported};

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

/// Restore PLE decode's three independent device-to-host transfers instead of
/// one staging buffer and one wait. Multi-token prefill always stays classic.
/// Both paths copy the same f32 values.
pub fn ple_readback_classic() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var_os("XWEN_PLE_READBACK_CLASSIC").is_some())
}

/// `XWEN_PLE_TAIL_CLASSIC` keeps PLE's gate and conv on the host for
/// multi-token Metal forwards instead of the device kernels (`ops::ple`,
/// the default since 2026-09-05). Decode always runs the host tail with its
/// batched readback. Presence-based, like the other kill switches.
pub fn ple_tail_classic() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var_os("XWEN_PLE_TAIL_CLASSIC").is_some())
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

/// A stage the duplicate-dispatch probe can re-encode (`ops::dup`).
///
/// Each variant names a group of kernel launches inside one forward, not a
/// source module: `Experts` covers the three expert gemms, `ExpertsDown` the
/// down projection alone, and so on. The nesting is deliberate — a call site
/// may belong to two stages so that one prices a whole group and the other
/// isolates a member of it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DupStage {
    /// The three routed-expert gemms (gate, up, down).
    Experts,
    /// The routed-expert down projection alone.
    ExpertsDown,
    /// The MoE router projection (`ffn_gate_inp`) alone. In no other stage:
    /// `MoeGlue` starts after it, at the routing kernel.
    RouterProj,
    /// The MoE glue around the experts: routing kernel, SwiGLU activation,
    /// combine and block epilogue — not the router projection.
    MoeGlue,
    /// The shared expert's projections.
    Shexp,
    /// Every hyper-connection kernel: grouped norm, both bottleneck gemms, the
    /// scaled silu, the mix, and the carrier write.
    Hc,
    /// The two hyper-connection bottleneck gemms alone.
    HcGemm,
    /// Every gated-DeltaNet device step: conv, beta/decay head, scan, gated norm.
    Gdn,
    /// The gated-DeltaNet recurrent scan alone.
    GdnScan,
}

impl DupStage {
    /// The `XWEN_DUP_STAGE` token for each stage, in declaration order.
    const NAMES: [(&'static str, DupStage); 9] = [
        ("experts", DupStage::Experts),
        ("experts_down", DupStage::ExpertsDown),
        ("router_proj", DupStage::RouterProj),
        ("moe_glue", DupStage::MoeGlue),
        ("shexp", DupStage::Shexp),
        ("hc", DupStage::Hc),
        ("hc_gemm", DupStage::HcGemm),
        ("gdn", DupStage::Gdn),
        ("gdn_scan", DupStage::GdnScan),
    ];

    fn from_name(name: &str) -> Option<DupStage> {
        Self::NAMES
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, s)| *s)
    }
}

/// Parses an `XWEN_DUP_STAGE` value: a comma-separated list of stage names,
/// whitespace around each ignored, empty entries skipped, duplicates collapsed.
/// An unrecognized name is an error naming the known set rather than a silently
/// ignored entry — a misspelled stage would otherwise report a plain run's wall
/// clock as the stage's cost, which reads as "this stage is free".
fn parse_dup_stages(spec: &str) -> Result<Vec<DupStage>, String> {
    let mut out = Vec::new();
    for name in spec.split(',') {
        let name = name.trim();
        if name.is_empty() {
            continue;
        }
        match DupStage::from_name(name) {
            Some(stage) => {
                if !out.contains(&stage) {
                    out.push(stage);
                }
            }
            None => {
                let known: Vec<&str> = DupStage::NAMES.iter().map(|(n, _)| *n).collect();
                return Err(format!(
                    "XWEN_DUP_STAGE: unknown stage {name:?}; known stages are {}",
                    known.join(", ")
                ));
            }
        }
    }
    Ok(out)
}

/// The stages `XWEN_DUP_STAGE` selects, VALUE-parsed and read once. Unset is an
/// empty list, which is the no-op configuration. A malformed value is a
/// startup-class error surfaced at the first probed dispatch, in the shape
/// `mm_id_nr1_env` uses.
fn dup_stages() -> anyhow::Result<&'static [DupStage]> {
    static V: OnceLock<Result<Vec<DupStage>, String>> = OnceLock::new();
    match V.get_or_init(|| match std::env::var("XWEN_DUP_STAGE") {
        Ok(spec) => parse_dup_stages(&spec),
        Err(_) => Ok(Vec::new()),
    }) {
        Ok(stages) => Ok(stages.as_slice()),
        Err(e) => Err(anyhow::anyhow!(e.clone())),
    }
}

/// `XWEN_DUP_REPS=N`: how many EXTRA copies of a selected dispatch to encode.
/// Value-parsed and read once; unset, unparsable or below 1 falls back to 1, so
/// the default probe doubles the selected work. Inert unless `XWEN_DUP_STAGE`
/// names a stage.
fn dup_reps() -> usize {
    static V: OnceLock<usize> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("XWEN_DUP_REPS")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|n| *n >= 1)
            .unwrap_or(1)
    })
}

/// Runs `f` once for its result, then `XWEN_DUP_REPS` more times with the
/// results dropped when `stage` is selected by `XWEN_DUP_STAGE` and `n > 1`,
/// or `n == 1` too under `XWEN_DUP_DECODE`. `n` is the token count of the
/// chunk at the call site. Decode is opt-in because its dispatches are
/// latency-bound: a copy that writes a fresh buffer overlaps the original
/// wherever the chain leaves the GPU idle, so a decode delta is the LOWER
/// bound below with more room under it than a prefill one.
///
/// The instrument this exists for: run a prefill twice, once plain and once
/// with a stage selected, and the wall-clock difference is that stage's GPU
/// time IN SITU — inside the real dependency chain, with no added syncs, which
/// is what separates it from `stack_profile` and `gdn_profile` (both sync per
/// step and so price the round trip alongside the kernel). Divide the delta by
/// `XWEN_DUP_REPS`.
///
/// Read the delta as a LOWER BOUND for a stage that does not saturate the
/// machine: candle's concurrent encoder barriers only on buffer hazards, so a
/// copy that writes a fresh buffer may overlap the original wherever the stage
/// leaves the GPU idle. A stage that does saturate it gets its true in-situ
/// time.
///
/// Only ever wrapped around a launcher that is a pure function of tensors the
/// caller already holds and that allocates its own output, so the copies are
/// discardable: nothing that advances a cache, mutates host state, or is read
/// through a shared scratch buffer goes inside. The FIRST result is the one
/// returned, so the copies cannot influence the math.
///
/// With `XWEN_GDN_PROFILE` set, the GDN steps sit inside `gdn_profile::rep` as
/// well and the two repeat counts multiply.
///
/// Unset — the normal case — this is one atomic load and one call.
pub fn dup<T>(
    stage: DupStage,
    n: usize,
    f: impl FnMut() -> anyhow::Result<T>,
) -> anyhow::Result<T> {
    let stages = dup_stages()?;
    dup_with(stages, dup_reps(), dup_decode(), stage, n, f)
}

/// `XWEN_DUP_DECODE`: presence lets [`dup`] repeat single-token calls as well,
/// so the probe can price a decode stage (a launch-count fusion's before and
/// after, say) by the same wall-clock difference. Read once. Inert unless
/// `XWEN_DUP_STAGE` names a stage.
fn dup_decode() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var_os("XWEN_DUP_DECODE").is_some())
}

/// [`dup`] over an explicit configuration, so the repeat rule is testable
/// without touching the process environment (the env is read once per process
/// and would make the tests order-dependent).
fn dup_with<T>(
    stages: &[DupStage],
    reps: usize,
    decode: bool,
    stage: DupStage,
    n: usize,
    mut f: impl FnMut() -> anyhow::Result<T>,
) -> anyhow::Result<T> {
    if n == 0 || (n == 1 && !decode) || !stages.contains(&stage) {
        return f();
    }
    let out = f()?;
    for _ in 0..reps {
        let _ = f()?;
    }
    Ok(out)
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

/// `XWEN_HOST_MASK` restores the HOST-built causal prefill mask: the scalar
/// double loop over `seq x (pos + seq)` plus an upload, which the device build
/// (`PrefillMask::causal_on_device`) replaced because both halves grow with
/// absolute position and neither computes anything but `k > q`.
///
/// It is kept for two jobs. It is the control arm the Flash-Next replay check
/// runs against (`bun scripts/flashnext-replay.ts --control XWEN_HOST_MASK=1`),
/// the two paths being required to produce identical masks. And it is what any
/// non-Metal device takes, so the fallback is exercised rather than assumed.
///
/// PRESENCE-BASED and cached (read once), like the sibling switches
/// (`stack_profile`, `chunk_sync`): any value enables it.
pub fn host_mask() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var_os("XWEN_HOST_MASK").is_some())
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

/// `XWEN_ACT_L2_CLASSIC=1` reverts the routed experts' f16-tile L2 rescale — the
/// `silu(gate)*up` activation, its per-row L2 norm, the clamp and the headroom
/// scale, one vendored pass (`ops::silu_mul_l2`) — back to the candle chain
/// (`ops::silu_mul` then sqr → sum_keepdim → sqrt → clamp → affine →
/// broadcast_div, six more dispatches over the activation). The fold is BOUNDED
/// against that chain, not bitwise: its sum-of-squares runs in a fixed
/// sequential-then-tree order where candle's `sum_keepdim` runs in its own, so
/// the two agree to accumulation-order noise (3.574e-7 max relative measured by
/// `l2_fold_matches_candle_chain`). It only ever runs on the rescale branch
/// (mm_id with an f16-staged activation), which the strict parity tier never
/// takes, and mm / decode / ppl grade it. `XWEN_ACT_CLASSIC` also disables it
/// (the fold contains the activation the older switch reverts).
///
/// PRESENCE-BASED and cached (read once), like the sibling MoE switches.
pub fn act_l2_classic() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var_os("XWEN_ACT_L2_CLASSIC").is_some())
}

/// `XWEN_SHEXP_QMATMUL=1` keeps the MoE shared expert's three projections on
/// candle's `QMatMul` at every token count, instead of routing them onto the
/// vendored dense cooperative-tensor gemm (`ops::matmul_dense_q`, via
/// `QLinear::forward_gemm`) above `dense_mm_min_seq()`. That gemm's precision
/// class is the 27B FFN's (~4e-4 rel_l2 from the f32 oracle, docs/parity.md
/// §3b), so like the 27B FFN it is also off under `XWEN_DENSE_MM_CLASSIC`,
/// which the strict tier pins on both sides.
///
/// PRESENCE-BASED and cached (read once).
pub fn shexp_qmatmul() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var_os("XWEN_SHEXP_QMATMUL").is_some())
}

/// Which of the hyper-connection bottleneck's two projections (`hc_*_down`,
/// `[low_rank, hc_count*hidden]`, and `hc_*_up`, `[hc_count*hidden, low_rank]`)
/// stay on candle's `QMatMul` at prefill instead of the vendored dense gemm.
/// `XWEN_HC_GEMM_QMATMUL=down` / `=up` / `=both` (any other value reads as
/// `both`); unset routes both onto the gemm above `dense_mm_min_seq()`. Two
/// arms because they are different shapes: `down` has k = 10240 and `up` only
/// k = 320 (ten NK steps), and whether the gemm wins at ten steps is a
/// measurement, not a given. Off under `XWEN_DENSE_MM_CLASSIC` like every
/// dense-gemm route. Decode (one token) is untouched either way.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum HcGemmQmatmul {
    Neither,
    Down,
    Up,
    Both,
}

impl HcGemmQmatmul {
    pub fn down_on_qmatmul(self) -> bool {
        matches!(self, Self::Down | Self::Both)
    }
    pub fn up_on_qmatmul(self) -> bool {
        matches!(self, Self::Up | Self::Both)
    }
}

pub fn hc_gemm_qmatmul() -> HcGemmQmatmul {
    static V: OnceLock<HcGemmQmatmul> = OnceLock::new();
    *V.get_or_init(|| match std::env::var("XWEN_HC_GEMM_QMATMUL") {
        Err(_) => HcGemmQmatmul::Neither,
        Ok(v) if v == "down" => HcGemmQmatmul::Down,
        Ok(v) if v == "up" => HcGemmQmatmul::Up,
        Ok(_) => HcGemmQmatmul::Both,
    })
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

/// `XWEN_MOE_SHEXP_CLASSIC=1` reverts the FUSED SHARED EXPERT —
/// `ops::moe_shexp_gate_up` and `ops::moe_epilogue_shexp`, which fold the gate
/// gemv, the up gemv, the SwiGLU activation, the `ffn_gate_inp_shexp` logit and
/// the down gemv into one dispatch plus a shexp-aware epilogue — back to the
/// five-dispatch chain: two `QLinear` matmuls, `ops::silu_mul`, a third matmul
/// and a candle f32 gemv, feeding the plain `ops::moe_epilogue`.
///
/// Counted per MoE LAYER that is four launches against zero (the fused pair's
/// second kernel replaces an epilogue the block dispatched anyway), which on
/// Flash-Next is 192 of a decode token's ~1356 and on the 35B-A3B 160.
///
/// This is a real kill switch and NOT a bit-identity anchor, unlike the rest of
/// the moe_glue family: all three fused dot products reassociate (the gate/up
/// rows fold per-thread `simd_sum` partials where `QMatMul`'s gemv folds its
/// own partition, and the shexp-aware epilogue's routed combine folds over a
/// full simdgroup where `kernel_moe_epilogue` folds over `next_pow2(top_k/2)`
/// lanes). `kernel_moe_epilogue` itself is untouched and still runs, bitwise,
/// wherever this path does not — which is what keeps the strict parity tier's
/// anchor intact. Bounded at rel_l2 <= 1e-5 against an f32 host reference
/// (`shexp_fused_matches_reference`); the strict tier pins this switch on both
/// sides and mm/decode/ppl grade the fused path.
///
/// `XWEN_MOE_GLUE_CLASSIC` disables it too, and not merely by convention: that
/// switch takes the whole block off the epilogue path, and the fused shared
/// expert lives inside it. [`moe_shexp_fused_enabled`] reads both.
///
/// PRESENCE-BASED and cached (read once), like the sibling switches
/// (`moe_glue_classic`, `hc_gate_classic`): any value enables it — only leaving
/// it unset keeps the fused pair.
pub fn moe_shexp_classic() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var_os("XWEN_MOE_SHEXP_CLASSIC").is_some())
}

/// Token count an MoE block must be at or below for its shared expert to take
/// the FUSED PAIR instead of the five-dispatch chain.
///
/// Same trade as [`HC_GATE_FUSED_MAX_N`], for the same reason: the fused pair
/// re-reads the token's activation row in every threadgroup of its first
/// kernel, and its second kernel widens the epilogue's threadgroups, so it only
/// pays where launch latency dominates. A decode step is one token; the ceiling
/// sits at the small-batch window's width so a short speculative or ragged
/// batch takes it too.
///
/// INCLUSIVE, like [`HC_GATE_FUSED_MAX_N`] and unlike [`HC_SPLIT_MAX_N`]: the
/// pair covers `n == 1`, which an exclusive bound of 1 would exclude.
pub const MOE_SHEXP_FUSED_MAX_N: usize = 8;

/// Effective fused-shared-expert ceiling: `XWEN_MOE_SHEXP_FUSED_MAX_N=<n>`
/// overrides the default (an A/B knob for the threshold — `0` keeps every batch
/// on the five-dispatch chain, which is what the kill switch does and is why
/// [`moe_shexp_fused_enabled`] reads both; a large value pushes the pair into
/// prefill token counts it was not shaped for, where the projections take the
/// dense cooperative-tensor gemm instead). Value-parsed, read once and cached;
/// unset or unparsable falls back to [`MOE_SHEXP_FUSED_MAX_N`].
pub fn moe_shexp_fused_max_n() -> usize {
    static V: OnceLock<usize> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("XWEN_MOE_SHEXP_FUSED_MAX_N")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(MOE_SHEXP_FUSED_MAX_N)
    })
}

/// Whether the fused shared expert can run in this process AT ALL: neither kill
/// switch set, and a ceiling that admits at least one token. `MoeBlock::forward`
/// asks this and then compares its own token count against
/// [`moe_shexp_fused_max_n`]; the parity dump's `moe_shexp` provenance label
/// asks it alone.
///
/// One function because the two must agree — `XWEN_MOE_SHEXP_FUSED_MAX_N=0`
/// disables the pair exactly as either kill switch does, so a label that read
/// only the switches would stamp "fused" on a dump that never dispatched the
/// kernels.
pub fn moe_shexp_fused_enabled() -> bool {
    !moe_glue_classic() && !moe_shexp_classic() && moe_shexp_fused_max_n() >= 1
}

/// `XWEN_ROUTER_MV_CLASSIC=1` reverts the MoE ROUTER PROJECTION — the vendored
/// f32 gemv (`ops::matmul_f32`, `kernel_mul_mv_f32_f32_v`) over the
/// `[n_expert, hidden]` `ffn_gate_inp` plane — to candle's `Tensor::matmul`
/// over the `[hidden, n_expert]` transpose held alongside it.
///
/// What it buys: candle's mlx `gemv_t` tile pick gives a one-token router
/// product EIGHT threadgroups for the whole 5.24 MB plane on Flash-Next (the
/// 35B-A3B's 2048 x 256 also lands on 8), each streaming ~655 KB serially; the
/// vendored gemv runs `ceil(n_expert/2) x t` threadgroups over the same bytes,
/// 256 at 512 experts. Per decode token that is 252 MB of router weight (48
/// layers on Flash-Next, 4.0% of the token's 6.33 GB) moved on an occupied
/// kernel instead of an idle one.
///
/// A real kill switch and NOT a bit-identity anchor. Both arms multiply the
/// same f32 operands and accumulate in f32 — the only difference is SUMMATION
/// ORDER (the gemv folds four simdgroups' partials through threadgroup memory;
/// mlx's gemv folds an 8-lane shuffle ladder over 320-term serial partials) —
/// so the logits are bounded-close, not bitwise. Routing is top-k over those
/// logits and top-k is DISCRETE, so a near-tie between two experts can flip
/// which one is selected. That is why the strict parity tier pins this switch
/// on both sides and mm/decode/ppl grade the gemv.
///
/// PRESENCE-BASED and cached (read once), like the sibling switches
/// (`moe_shexp_classic`, `moe_glue_classic`): any value enables it — only
/// leaving it unset keeps the gemv.
pub fn router_mv_classic() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var_os("XWEN_ROUTER_MV_CLASSIC").is_some())
}

/// Token count an MoE block must be at or below for its ROUTER PROJECTION to
/// take the vendored f32 gemv instead of candle's `matmul`.
///
/// Same trade as [`MOE_SHEXP_FUSED_MAX_N`]: the gemv re-reads the whole router
/// plane once per token row (its grid's second axis IS the token count), which
/// is free at one token and pointless at a prefill chunk, where candle's tiled
/// gemm reads the plane once for all rows. The ceiling sits at the small-batch
/// window's width so a short speculative or ragged batch takes it too — and at
/// 8 it is exactly the gemv's own hard limit, above which `ops::matmul_f32`
/// errors rather than falling back.
///
/// INCLUSIVE, like [`MOE_SHEXP_FUSED_MAX_N`]: the gemv covers `n == 1`, which
/// an exclusive bound of 1 would exclude.
pub const ROUTER_MV_MAX_N: usize = 8;

/// Effective router-gemv ceiling: `XWEN_ROUTER_MV_MAX_N=<n>` overrides the
/// default (an A/B knob for the threshold — `0` keeps every batch on candle,
/// which is what the kill switch does and is why [`router_mv_enabled`] reads
/// both; a value above [`ROUTER_MV_MAX_N`] cannot widen the window, since
/// `ops::matmul_f32_supported` refuses past 8 and the block then keeps candle).
/// Value-parsed, read once and cached; unset or unparsable falls back to
/// [`ROUTER_MV_MAX_N`].
pub fn router_mv_max_n() -> usize {
    static V: OnceLock<usize> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("XWEN_ROUTER_MV_MAX_N")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(ROUTER_MV_MAX_N)
    })
}

/// Whether the router gemv can run in this process AT ALL: the kill switch
/// unset, and a ceiling that admits at least one token. `MoeBlock::router_mv`
/// asks this and then compares its own token count against
/// [`router_mv_max_n`]; the parity dump's `router_mv` provenance label asks it
/// alone.
///
/// One function because the two must agree — `XWEN_ROUTER_MV_MAX_N=0` disables
/// the gemv exactly as the kill switch does, so a label that read only the
/// switch would stamp "mv" on a dump that never dispatched the kernel.
///
/// Unlike [`moe_shexp_fused_enabled`] this does NOT read `moe_glue_classic`:
/// the router projection runs before the routing decision and is dispatched
/// whether or not the fused glue kernels follow it.
pub fn router_mv_enabled() -> bool {
    !router_mv_classic() && router_mv_max_n() >= 1
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
/// write-back (`ops::hc_write`) — to the candle chains they replace. It is the
/// OUTERMOST of the two hc switches: reverting the read gate takes the fused
/// decode gate (`ops::hc_gate_down` / `ops::hc_gate_up_mix`) with it, since that
/// is a branch inside the fused read. `XWEN_HC_GATE_CLASSIC` reverts the decode
/// gate alone, back to the five kernels named above plus the two `QLinear`
/// matmuls.
///
/// The activation and the write-back are bit-identical to their chains by
/// construction; the norm and the mix partition reductions those chains run in
/// one order, and the decode gate reassociates both bottleneck dot products, so
/// those are bounded rather than bitwise (hc.metal, and the tolerances the hc.rs
/// tests grade at). Which paths the two Q8_0 bottleneck matmuls are inside
/// depends on the token count: above the fused gate's ceiling both arms run the
/// same `QLinear` and the matmuls are outside this switch entirely, while at or
/// below it (decode, and the small-batch window by default) the fused gate
/// computes them itself and this switch is what puts them back on `QLinear`.
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

/// Kill switch (`XWEN_QSA_HOST_TOPK`) for the QSA indexer's device-side block
/// selection at decode (`ops::qsa_select`, `kernel_qsa_select`): set, a
/// single-token `QsaIndexer::select` above budget reads the block scores back
/// to the host and runs `top_blocks` + `expand_into` there, as every decode
/// step did before the kernel — one pipeline drain per QSA layer per step.
/// Both arms produce the SAME ROWS by construction (the kernel implements the
/// host's total order over the score bits;
/// `device_select_matches_host_top_blocks_bitwise`), so this is a fallback,
/// never a parity row. `XWEN_QSA_CLASSIC` implies it.
///
/// PRESENCE-BASED and cached (read once), like the sibling switches: any value
/// enables it — only leaving it unset keeps the device path.
pub fn qsa_host_topk() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var_os("XWEN_QSA_HOST_TOPK").is_some())
}

/// Kill switch (`XWEN_QSA_HOST_MASK`) for the QSA indexer's device-side
/// prefill selection and mask (`ops::qsa_select::select_mask`,
/// `kernel_qsa_select_mask`): set, an above-budget prefill chunk reads its
/// score plane back, selects on the host and uploads a host-filled mask, as
/// every prefill did before 2026-09-06 — the arm `XWEN_QSA_TIMER` prices.
/// `XWEN_QSA_CLASSIC` implies it.
pub fn qsa_host_mask() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var_os("XWEN_QSA_HOST_MASK").is_some())
}

/// Instrument (`XWEN_QSA_TIMER`) for the QSA indexer's prefill host round
/// trip: each above-budget prefill chunk on each QSA layer drains the device,
/// then times the score readback, the host selection, the mask fill and the
/// upload separately, and `XwenModel::dump_stack_profile` prints the totals
/// after the prefill. Measurement only: the selection is unchanged, and the
/// extra sync is one the readback would have made anyway.
pub fn qsa_timer() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var_os("XWEN_QSA_TIMER").is_some())
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

/// Token count a hyper-connection gate must be at or below for the read to take
/// the FUSED DECODE GATE — `kernel_hc_gate_down` and `kernel_hc_gate_up_mix`,
/// three dispatches per gate with the write-back — instead of the seven-dispatch
/// split path (norm, head, down gemv, activation, up gemv, mix, write).
///
/// The fused pair trades bytes for launches: each of its threadgroups re-reads
/// the carrier row and the norm weight for its token, and it swallows the two
/// q8_0 projections, so it only pays where launch latency dominates. A decode
/// step is one token; the ceiling is set at the small-batch window's width
/// rather than at 1 so a short speculative or ragged batch takes it too.
///
/// Inclusive, unlike [`HC_SPLIT_MAX_N`]: the fused gate covers `n == 1`, which
/// an exclusive bound of 1 would exclude.
pub const HC_GATE_FUSED_MAX_N: usize = 8;

/// Effective fused-gate ceiling: `XWEN_HC_GATE_FUSED_MAX_N=<n>` overrides the
/// default (an A/B knob for the threshold — `0` keeps every batch on the split
/// path, which is what the kill switch does and is why
/// [`hc_gate_fused_enabled`] reads both, a large value pushes the fused gate
/// into prefill token counts it was not shaped for). Value-parsed, read once and
/// cached; unset or unparsable falls back to [`HC_GATE_FUSED_MAX_N`].
pub fn hc_gate_fused_max_n() -> usize {
    static V: OnceLock<usize> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("XWEN_HC_GATE_FUSED_MAX_N")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(HC_GATE_FUSED_MAX_N)
    })
}

/// `XWEN_HC_GATE_CLASSIC=1` reverts the fused decode gate — `ops::hc_gate_down`
/// and `ops::hc_gate_up_mix`, which fold the grouped norm, the injection head,
/// both q8_0 bottleneck projections, the activation and the mix into two
/// dispatches — back to the split path's six: `ops::hc_norm`'s split pair, the
/// two `QLinear` matmuls, `ops::hc_silu_quarter` and `ops::hc_mix`.
///
/// Counted per GATE, the write-back included, that is seven dispatches against
/// three — the count hc.metal and docs/parity.md use, and the one the ledger's
/// "672 per token" figure is built from. `ops::hc_write` itself is identical on
/// both arms and outside this switch; it is the seventh either way, and the
/// tail mixer runs the same arms one dispatch shorter with no head and no
/// write.
///
/// This is a real kill switch and NOT a bit-identity anchor: both fused kernels
/// reassociate reductions the split path runs in another order (the down rows
/// fold per-thread `simd_sum` partials where the gemv folds its own partition,
/// the stream mean runs as a simd-shuffle butterfly, and the per-stream
/// statistics are partitioned per q8_0 block), so they are bounded-close, the
/// same class as `XWEN_HC_CLASSIC`. Flash-Next only, and no parity tier grades
/// it: `scripts/flashnext-replay.ts --control XWEN_HC_GATE_CLASSIC=1` is the
/// check. `XWEN_HC_CLASSIC` disables it too — the fused gate is the fused read
/// path's small-batch arm, and the older switch promises the candle chains.
///
/// PRESENCE-BASED and cached (read once), like the sibling switches
/// (`hc_classic`, `attn_glue_classic`): any value enables it — only leaving it
/// unset keeps the fused gate.
pub fn hc_gate_classic() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var_os("XWEN_HC_GATE_CLASSIC").is_some())
}

/// Whether the fused decode gate can run in this process AT ALL: neither kill
/// switch set, and a ceiling that admits at least one token. The read path asks
/// this and then compares its own token count against [`hc_gate_fused_max_n`];
/// the parity dump's `hc_gate` provenance label asks it alone.
///
/// One function because the two must agree. `XWEN_HC_GATE_FUSED_MAX_N=0`
/// disables the gate exactly as the kill switch does, so a label that read only
/// the switch would stamp "fused" on a dump that never dispatched the kernels —
/// which is the failure mode the `delta` field exists to prevent.
pub fn hc_gate_fused_enabled() -> bool {
    !hc_classic() && !hc_gate_classic() && hc_gate_fused_max_n() >= 1
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

    /// Kernel host-name suffix for a given pass-2 token-tile width: the tensor
    /// family is instantiated at 32 (`_t`) and 64 (`_t64`); every other family
    /// only at 32.
    pub(crate) fn suffix_nr1(self, nr1: usize) -> anyhow::Result<&'static str> {
        match (self, nr1) {
            (MmVariant::Tensor, 64) => Ok("_t64"),
            (_, 32) => Ok(self.suffix()),
            (v, n) => anyhow::bail!("mm_id variant {} has no NR1={n} kernel", v.name()),
        }
    }

    /// Threadgroup tile bytes at token-tile width `nr1`:
    /// `max(sa + sb, NR0*nr1*4)` with sa = 64*32*sizeof(S0), sb = nr1*32*sizeof(S1)
    /// and the float store-back tile aliasing the front. The f32-tile variants
    /// (`_hp`, `_t_hp`) need 12288 at 32, the half-tile variants 8192; the
    /// half-tile tensor kernel at 64 needs 16384 (the store-back tile dominates).
    pub(crate) fn tile_smem(self, nr1: usize) -> usize {
        let elem = match self {
            MmVariant::ClassicHp | MmVariant::TensorHp => 4,
            MmVariant::Tensor | MmVariant::ClassicF16 => 2,
        };
        let sa = 64 * 32 * elem;
        let sb = nr1 * 32 * elem;
        (sa + sb).max(64 * nr1 * 4)
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    /// Counts calls so a test can assert how many times `dup` ran the closure.
    fn counting<'a>(calls: &'a Cell<usize>) -> impl FnMut() -> anyhow::Result<usize> + 'a {
        move || {
            calls.set(calls.get() + 1);
            Ok(calls.get())
        }
    }

    #[test]
    fn dup_repeats_only_a_selected_stage_over_more_than_one_token() {
        let stages = [DupStage::Experts];

        // Selected, multi-token: the real call plus `reps` discarded copies.
        let calls = Cell::new(0);
        let out = dup_with(&stages, 2, false, DupStage::Experts, 64, counting(&calls)).unwrap();
        assert_eq!(calls.get(), 3);
        // The FIRST result is what the caller gets; the copies never replace it.
        assert_eq!(out, 1);

        // A stage nobody selected runs once whatever the token count.
        let calls = Cell::new(0);
        dup_with(&stages, 2, false, DupStage::GdnScan, 64, counting(&calls)).unwrap();
        assert_eq!(calls.get(), 1);

        // Decode (n == 1) runs once even for a selected stage, unless the
        // decode opt-in is set, when it repeats like any other token count.
        let calls = Cell::new(0);
        dup_with(&stages, 2, false, DupStage::Experts, 1, counting(&calls)).unwrap();
        assert_eq!(calls.get(), 1);
        let calls = Cell::new(0);
        dup_with(&stages, 2, true, DupStage::Experts, 1, counting(&calls)).unwrap();
        assert_eq!(calls.get(), 3);

        // The empty configuration — the unset environment — never repeats.
        let calls = Cell::new(0);
        dup_with(&[], 4, true, DupStage::Experts, 64, counting(&calls)).unwrap();
        assert_eq!(calls.get(), 1);
    }

    #[test]
    fn dup_stage_names_parse_and_an_unknown_one_is_an_error() {
        assert_eq!(
            parse_dup_stages("experts,gdn_scan").unwrap(),
            vec![DupStage::Experts, DupStage::GdnScan]
        );
        // Whitespace, empty entries and repeats are all tolerated.
        assert_eq!(
            parse_dup_stages(" hc , , hc_gemm , hc ").unwrap(),
            vec![DupStage::Hc, DupStage::HcGemm]
        );
        assert_eq!(parse_dup_stages("").unwrap(), vec![]);
        // Every documented name resolves.
        for (name, stage) in DupStage::NAMES {
            assert_eq!(parse_dup_stages(name).unwrap(), vec![stage]);
        }

        let err = parse_dup_stages("experts,ffn").unwrap_err();
        assert!(err.contains("ffn"), "{err}");
        assert!(err.contains("experts_down"), "{err}");
    }
}
