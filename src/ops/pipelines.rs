//! Runtime compilation and caching of xwen's vendored Metal kernels.
//!
//! candle's baked metallib is fixed at its build; our vendored kernel sources
//! (`src/ops/*.metal`) are compiled against the live device once each and the
//! resulting pipelines cached by name. `ComputePipeline`/`Library` are
//! `Send + Sync + Clone`, so a process-global cache keyed by the device's
//! registry id serves every dispatch.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use anyhow::{Context, Result};
use candle_metal_kernels::metal::{ComputePipeline, Device, Library};

/// Vendored kernel source, compiled at runtime (candle's metallib carries no
/// Rust wiring for these and cannot be extended at build time). This is the
/// DEFAULT prefill library; it instantiates every mm_id variant EXCEPT the
/// float-operand cooperative-tensor `_t_hp` one (split into `mm_id_t_hp.metal`),
/// so a toolchain that rejects float `matmul2d` operands cannot break it.
const MM_ID_SOURCE: &str = include_str!("mm_id.metal");
/// The `_t_hp` (float-operand cooperative tensor) instantiations, split out of
/// `mm_id.metal` so the default library carries no float-cooperative-tensor
/// code. Not a standalone translation unit — it references that file's template,
/// typedef and dequant definitions, so it is compiled by concatenating it onto
/// `MM_ID_SOURCE` (see `mm_id_t_hp_source`), and only on first TensorHp dispatch.
const MM_ID_T_HP_INSTANTIATIONS: &str = include_str!("mm_id_t_hp.metal");
/// Vendored ggml-geometry mat-vec kernels (decode expert gather + lm_head).
/// Deliberately separate from `mm_id.metal` so it carries no Metal-4 tensor
/// dependency.
const MV_SOURCE: &str = include_str!("mv.metal");
/// Vendored ggml small-batch mat-vec (`mul_mv_ext`) — the 2..8-token window
/// between `mv.metal`'s gemv and a tiled gemm, one weight pass per 2..5 token
/// rows. Its own library for the same isolation reason as every other split:
/// `mv.metal` is decode-critical and must not be able to fail to compile
/// because of a kernel only the small-batch window uses. Under the
/// `XWEN_MV_EXT_CLASSIC` kill-switch nothing asks for this library, so it never
/// compiles.
const MV_EXT_SOURCE: &str = include_str!("mv_ext.metal");
/// Vendored f16-weight x f32-activation matmul kernels (attention projections).
/// Separate from both `mm_id.metal` (no Metal-4 tensor dependency) and
/// `mv.metal` (attention-critical vs MoE-decode-critical: neither library can
/// break the other).
const F16_SOURCE: &str = include_str!("f16.metal");
/// Vendored q8_0-weight x f32-activation mat-vec kernel (the attention DECODE
/// gemv of a q8_0-quantized checkpoint). Separate from `mv.metal` (which carries
/// the byte-identical MoE-decode q8_0 gather kernel) exactly as `f16.metal` is:
/// attention-critical vs MoE-decode-critical, so neither can break the other.
const Q8_SOURCE: &str = include_str!("q8.metal");
/// Vendored bf16-weight x f32-activation matmul kernels (the DFlash drafter's
/// mmap-aliased BF16 planes) — the bf16 twin of `f16.metal` (gemv + classic
/// gemm). Separate from `f16.metal` for the same isolation reason as ever:
/// drafter-critical vs attention-critical, so a bfloat compile problem cannot
/// break attention. Compiled lazily on first bf16 dispatch.
const BF16_SOURCE: &str = include_str!("bf16.metal");
/// Vendored Metal-4 cooperative-tensor bf16-weight prefill gemm — the bf16 twin
/// of `f16_t.metal` (bfloat device loads staged to the half A tile, so matmul2d
/// never sees a bfloat operand). Own lazily compiled library, mirroring the
/// `f16.metal`/`f16_t.metal` split: `bf16.metal` stays Metal-4-free and the
/// attention libraries stay bfloat-free.
const BF16_T_SOURCE: &str = include_str!("bf16_t.metal");
/// Vendored f32-weight x f32-activation mat-vec kernel — the f32 twin of
/// `f16.metal`'s decode gemv, dispatched for the MoE ROUTER projection at 1..8
/// tokens (candle's mlx gemv gives that plane only 8 threadgroups). Separate
/// from `f16.metal` for the same isolation reason `bf16.metal` and `mv.metal`
/// are separate from it: router-critical vs attention-critical, so neither can
/// break the other. Compiled lazily on first router-gemv dispatch.
const F32_SOURCE: &str = include_str!("f32.metal");
/// Vendored Metal-4 cooperative-tensor attention prefill gemm (the tensor
/// analogue of `f16.metal`'s classic `kernel_mul_mm_f16_f32_v`). Separate from
/// `f16.metal` so that file stays Metal-4-free: this library is compiled lazily,
/// only on the first tensor-path dispatch (`f16_t_pipeline`), mirroring the
/// `mm_id_t_hp` split. The tensor path is the shipped default; under the
/// `XWEN_ATTN_MM_CLASSIC` kill-switch the mm branch never asks for this
/// library, so it never compiles.
const F16_T_SOURCE: &str = include_str!("f16_t.metal");
/// PROBE: the mixed-operand (half weight tile x FLOAT activation tile)
/// cooperative-tensor variant of `f16_t.metal`. Test-only reachability
/// (`run_matmul_f16_variant`'s `TensorMixed` arm); its own lazily compiled
/// library — the `mm_id_t_hp` isolation pattern — so a toolchain that rejects
/// mixed-operand `matmul2d` fails only this probe, never the default or the
/// half-tile tensor library.
const F16_T_MIXED_SOURCE: &str = include_str!("f16_t_mixed.metal");
/// Vendored Metal-4 cooperative-tensor DENSE quantized-weight prefill gemm (the
/// 27B's SwiGLU FFN projections) — `f16_t.metal`'s kernel with an in-kernel
/// block-quant tile dequant in place of the half widen-copy. Its own lazily
/// compiled library, mirroring every other Metal-4 split: `mm_id.metal` (whose
/// dequant helpers it re-vendors) is MoE-prefill-critical, this one is
/// dense-FFN-prefill-critical, and neither can break the other. Under the
/// `XWEN_DENSE_MM_CLASSIC` kill-switch nothing asks for this library, so it
/// never compiles.
const DENSE_MM_SOURCE: &str = include_str!("dense_mm.metal");
/// Vendored fused MoE weighted-combine kernels (the routed-expert combine tail).
/// Own library (no Metal-4 dependency); compiled with FP contraction disabled so
/// its per-op rounding stays bit-identical to the candle broadcast/affine/sum
/// chain it replaces (see combine.metal).
const COMBINE_SOURCE: &str = include_str!("combine.metal");
/// Vendored fused MoE SwiGLU-activation kernel (the routed-expert silu*mul glue).
/// Own library (no Metal-4 dependency); compiled with FP contraction/reassociation
/// disabled so its per-op rounding stays bit-identical to the candle silu + mul
/// chain it replaces (see silu_mul.metal).
const SILU_MUL_SOURCE: &str = include_str!("silu_mul.metal");
/// Vendored fused MoE glue kernels (the routing decision and the block tail).
/// Own library (no Metal-4 dependency). Its FP pragmas are at BLOCK scope, not
/// file scope: the epilogue pins contraction/reassociation off to keep candle's
/// rounding boundaries, while the router must compile under the same fast-math
/// latitude candle's own softmax/sort kernels do (see moe_glue.metal).
const MOE_GLUE_SOURCE: &str = include_str!("moe_glue.metal");
/// Vendored fused attention-glue kernels (softplus gate + permute/cast copies).
/// Own library (no Metal-4 dependency); compiled with FP contraction disabled so
/// its per-op rounding stays bit-identical to the candle elementwise/copy chains
/// it replaces (see attn_glue.metal).
const ATTN_GLUE_SOURCE: &str = include_str!("attn_glue.metal");
/// Vendored NEOX rope kernel with internal partial rotary. A SEPARATE library
/// from attn_glue.metal because that file's fp pragmas are file-scoped and the
/// rope rotation must instead compile under the same default math mode as
/// candle's own rope kernel to stay bit-identical (see rope.metal).
const ROPE_SOURCE: &str = include_str!("rope.metal");
/// Vendored flash-attention prefill kernel (the modified copy of candle's MLX
/// steel attention: float Q/O, half K/V, in-kernel causal+sliding-window
/// masking). Own library (no Metal-4 dependency), compiled fast-math like
/// candle's metallib with no contract pragmas — see flash.metal's header.
const FLASH_SOURCE: &str = include_str!("flash.metal");
/// Vendored fused gated-DeltaNet kernels (conv+silu+state, beta/decay head,
/// recurrent scan, gated output norm). Own library (no Metal-4 dependency).
/// The conv and beta/decay kernels pin FP contraction/reassociation off at
/// BLOCK scope so their rounding stays bit-identical to the candle chains they
/// replace; the gated norm and the scan are bounded instead — each partitions a
/// reduction across threads — and the scan stays free to contract (see
/// delta.metal).
const DELTA_SOURCE: &str = include_str!("delta.metal");
/// Vendored fused hyper-connection kernels (the qwen4exp carrier's grouped norm
/// with its injection head, the bottleneck activation, the stream mix and the
/// write-back). Own library (no Metal-4 dependency). Its FP pragmas are at BLOCK
/// scope: the activation and the write-back pin contraction/reassociation off to
/// stay bit-identical to the candle chains they replace, while the norm and the
/// mix are bounded instead — each partitions a reduction the reference runs in
/// one order (see hc.metal). Under the `XWEN_HC_CLASSIC` kill-switch nothing
/// asks for this library, so it never compiles.
const HC_SOURCE: &str = include_str!("hc.metal");
const PLE_SOURCE: &str = include_str!("ple.metal");
/// Achievable-bandwidth probes (bench only, `ops::bandwidth`). Never on a model
/// path, so it compiles only when the bench asks for it.
const BW_SOURCE: &str = include_str!("bandwidth.metal");
/// Vendored QSA decode row gather (the K/V rows a selection names, packed
/// into one contiguous plane per head). Own library; a pure copy, so it has
/// no rounding contract to state. Under `XWEN_QSA_CLASSIC` nothing asks for
/// it, so it never compiles.
const QSA_GATHER_SOURCE: &str = include_str!("qsa_gather.metal");
/// Vendored QSA decode block selection (device-side top-k of the block scores,
/// expanded into the gather's row list). Own library; integer work over
/// canonicalized score bits, so it has no rounding contract to state. Under
/// `XWEN_QSA_CLASSIC` or `XWEN_QSA_HOST_TOPK` nothing asks for it, so it never
/// compiles.
const QSA_SELECT_SOURCE: &str = include_str!("qsa_select.metal");

/// The concatenated source for the TensorHp library: the shared mm_id template
/// portion plus the split-out `_t_hp` instantiations. Built once on first use,
/// so the (potentially unsupported) float-cooperative-tensor code is only ever
/// handed to the Metal compiler when TensorHp is actually selected.
fn mm_id_t_hp_source() -> &'static str {
    static SRC: OnceLock<String> = OnceLock::new();
    SRC.get_or_init(|| format!("{MM_ID_SOURCE}\n{MM_ID_T_HP_INSTANTIATIONS}"))
}

struct Cache {
    /// One compiled library per (device registry id, source key).
    libraries: HashMap<(u64, &'static str), Library>,
    /// Pipelines keyed by (device registry id, function name). Function names
    /// are unique across our sources, so the source key is not part of the key.
    pipelines: HashMap<(u64, String), ComputePipeline>,
    /// `max_total_threads_per_threadgroup` per (device registry id, function
    /// name) — what a DISPATCH PREDICATE needs to know before it commits to a
    /// kernel, so a device that derates a pipeline below the width its grid
    /// assumes is a "take the other path", not an error at encode time. Kept
    /// beside the pipelines rather than read off one per call: the predicates
    /// run per layer per token at decode, and this is a map lookup where
    /// cloning the pipeline to ask it would be an ObjC round trip.
    max_threads: HashMap<(u64, String), usize>,
}

fn cache() -> &'static Mutex<Cache> {
    static CACHE: OnceLock<Mutex<Cache>> = OnceLock::new();
    CACHE.get_or_init(|| {
        Mutex::new(Cache {
            libraries: HashMap::new(),
            pipelines: HashMap::new(),
            max_threads: HashMap::new(),
        })
    })
}

/// Fetch (compiling and caching on first use) the compute pipeline for `name`
/// from vendored `source` (labelled `source_key` for the library cache) on
/// `device`.
fn compiled_pipeline(
    device: &Device,
    source: &str,
    source_key: &'static str,
    name: &str,
) -> Result<ComputePipeline> {
    // The vendored kernels' bit-identity contracts assume candle's kernels
    // compile in candle's default fast-math mode; our sources are pinned
    // `math_mode(fast)`. A falsy CANDLE_METAL_ENABLE_FAST_MATH would move
    // candle to Relaxed/Precise while ours stay fast — a silent break of every
    // bitwise contract, so refuse to run rather than warn (the same
    // fail-closed stance the parity provenance system takes).
    if crate::ops::candle_fast_math_disabled() {
        anyhow::bail!(
            "CANDLE_METAL_ENABLE_FAST_MATH is set falsy: candle would compile its Metal kernels \
             Relaxed/Precise while xwen's vendored libraries are pinned math_mode(fast), \
             silently breaking their bitwise-identity contracts (combine/attn_glue/rope). \
             Unset the variable — the two compile modes cannot be mixed."
        );
    }
    let key = device.registry_id();
    let mut cache = cache().lock().unwrap();

    if let Some(p) = cache.pipelines.get(&(key, name.to_string())) {
        return Ok(p.clone());
    }

    let lib_key = (key, source_key);
    if !cache.libraries.contains_key(&lib_key) {
        let lib = device
            .new_library_with_source(source, None)
            .map_err(|e| anyhow::anyhow!("compiling vendored {source_key}.metal: {e}"))?;
        cache.libraries.insert(lib_key, lib);
    }
    let lib = &cache.libraries[&lib_key];

    let func = lib
        .get_function(name, None)
        .map_err(|e| anyhow::anyhow!("locating `{name}` in vendored {source_key}.metal: {e}"))?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&func)
        .with_context(|| format!("building pipeline for `{name}`"))?;

    cache
        .pipelines
        .insert((key, name.to_string()), pipeline.clone());
    Ok(pipeline)
}

/// Pipeline for a `mm_id.metal` kernel (two-pass indexed matmul).
///
/// The float-operand cooperative-tensor `_t_hp` kernels live in a separate
/// library, compiled lazily by concatenating `mm_id_t_hp.metal` onto the shared
/// source. Routing on the kernel name keeps the default library free of
/// float-cooperative-tensor code: a `_t_hp` compile failure surfaces only here,
/// on the TensorHp path, and never touches the default prefill library.
pub(crate) fn mm_id_pipeline(device: &Device, name: &str) -> Result<ComputePipeline> {
    if name.ends_with("_t_hp") {
        compiled_pipeline(device, mm_id_t_hp_source(), "mm_id_t_hp", name)
    } else {
        compiled_pipeline(device, MM_ID_SOURCE, "mm_id", name)
    }
}

/// Pipeline for a `mv.metal` kernel (vendored ggml-geometry mat-vec).
pub(crate) fn mv_pipeline(device: &Device, name: &str) -> Result<ComputePipeline> {
    compiled_pipeline(device, MV_SOURCE, "mv", name)
}

/// Pipeline for an `mv_ext.metal` kernel (vendored ggml small-batch mat-vec).
/// Its own library, compiled lazily on the first small-batch dispatch.
pub(crate) fn mv_ext_pipeline(device: &Device, name: &str) -> Result<ComputePipeline> {
    compiled_pipeline(device, MV_EXT_SOURCE, "mv_ext", name)
}

/// Pipeline for an `f16.metal` kernel (vendored f16-weight attention matmul).
pub(crate) fn f16_pipeline(device: &Device, name: &str) -> Result<ComputePipeline> {
    compiled_pipeline(device, F16_SOURCE, "f16", name)
}

/// Pipeline for a `q8.metal` kernel (vendored q8_0-weight attention decode gemv).
pub(crate) fn q8_pipeline(device: &Device, name: &str) -> Result<ComputePipeline> {
    compiled_pipeline(device, Q8_SOURCE, "q8", name)
}

/// Pipeline for a `bf16.metal` kernel (vendored bf16-weight drafter matmul —
/// gemv + classic gemm).
pub(crate) fn bf16_pipeline(device: &Device, name: &str) -> Result<ComputePipeline> {
    compiled_pipeline(device, BF16_SOURCE, "bf16", name)
}

/// Pipeline for a `bf16_t.metal` kernel (Metal-4 cooperative-tensor bf16-weight
/// drafter prefill gemm). Its own library, compiled lazily on first use so
/// `bf16.metal` carries no Metal-4 dependency.
pub(crate) fn bf16_t_pipeline(device: &Device, name: &str) -> Result<ComputePipeline> {
    compiled_pipeline(device, BF16_T_SOURCE, "bf16_t", name)
}

/// Pipeline for an `f32.metal` kernel (vendored f32-weight MoE router gemv).
pub(crate) fn f32_pipeline(device: &Device, name: &str) -> Result<ComputePipeline> {
    compiled_pipeline(device, F32_SOURCE, "f32", name)
}

/// Pipeline for an `f16_t.metal` kernel (Metal-4 cooperative-tensor attention
/// prefill gemm). Its own library, compiled lazily on first use so the classic
/// `f16.metal` library carries no Metal-4 dependency.
pub(crate) fn f16_t_pipeline(device: &Device, name: &str) -> Result<ComputePipeline> {
    compiled_pipeline(device, F16_T_SOURCE, "f16_t", name)
}

/// Pipeline for an `f16_t_mixed.metal` kernel (the mixed-operand matmul2d
/// probe). Own library, compiled lazily on first (test-only) dispatch.
pub(crate) fn f16_t_mixed_pipeline(device: &Device, name: &str) -> Result<ComputePipeline> {
    compiled_pipeline(device, F16_T_MIXED_SOURCE, "f16_t_mixed", name)
}

/// Pipeline for a `dense_mm.metal` kernel (Metal-4 cooperative-tensor dense
/// quantized-weight prefill gemm). Its own library, compiled lazily on first
/// dense-FFN prefill dispatch.
pub(crate) fn dense_mm_pipeline(device: &Device, name: &str) -> Result<ComputePipeline> {
    compiled_pipeline(device, DENSE_MM_SOURCE, "dense_mm", name)
}

/// Pipeline for a `combine.metal` kernel (vendored fused MoE weighted combine).
pub(crate) fn combine_pipeline(device: &Device, name: &str) -> Result<ComputePipeline> {
    compiled_pipeline(device, COMBINE_SOURCE, "combine", name)
}

/// Pipeline for a `silu_mul.metal` kernel (vendored fused MoE SwiGLU activation).
pub(crate) fn silu_mul_pipeline(device: &Device, name: &str) -> Result<ComputePipeline> {
    compiled_pipeline(device, SILU_MUL_SOURCE, "silu_mul", name)
}

/// Pipeline for a `moe_glue.metal` kernel (fused MoE router / block epilogue).
pub(crate) fn moe_glue_pipeline(device: &Device, name: &str) -> Result<ComputePipeline> {
    compiled_pipeline(device, MOE_GLUE_SOURCE, "moe_glue", name)
}

/// Pipeline for an `attn_glue.metal` kernel (fused softplus gate / permute-cast).
pub(crate) fn attn_glue_pipeline(device: &Device, name: &str) -> Result<ComputePipeline> {
    compiled_pipeline(device, ATTN_GLUE_SOURCE, "attn_glue", name)
}

/// Pipeline for a `rope.metal` kernel (vendored partial-rotary NEOX rope).
pub(crate) fn rope_pipeline(device: &Device, name: &str) -> Result<ComputePipeline> {
    compiled_pipeline(device, ROPE_SOURCE, "rope", name)
}

/// Pipeline for a `flash.metal` kernel (vendored flash-attention prefill).
pub(crate) fn flash_pipeline(device: &Device, name: &str) -> Result<ComputePipeline> {
    compiled_pipeline(device, FLASH_SOURCE, "flash", name)
}

/// Pipeline for a `delta.metal` kernel (vendored fused gated-DeltaNet ops).
pub(crate) fn delta_pipeline(device: &Device, name: &str) -> Result<ComputePipeline> {
    compiled_pipeline(device, DELTA_SOURCE, "delta", name)
}

/// The threadgroup width `name`'s pipeline admits on `device`, compiling it on
/// first ask and caching the number thereafter.
///
/// For the predicates that must decide BEFORE the dispatch whether a fused
/// delta kernel can run at all: a pipeline whose register pressure derates it
/// below the width its grid assumes has to route the block to the fallback
/// path, and a predicate cannot report an error. An `Err` here is a compile or
/// lookup failure, which the caller reads the same way — this kernel is not
/// available — while every other delta dispatch reports it loudly.
pub(crate) fn delta_max_threads(device: &Device, name: &str) -> Result<usize> {
    let key = (device.registry_id(), name.to_string());
    if let Some(&width) = cache().lock().unwrap().max_threads.get(&key) {
        return Ok(width);
    }
    // Built outside the lock: `delta_pipeline` takes it itself.
    let width = delta_pipeline(device, name)?.max_total_threads_per_threadgroup();
    cache().lock().unwrap().max_threads.insert(key, width);
    Ok(width)
}

/// Pipeline for an `hc.metal` kernel (vendored fused hyper-connection gates).
/// Its own library, compiled lazily on the first carrier read or write.
pub(crate) fn hc_pipeline(device: &Device, name: &str) -> Result<ComputePipeline> {
    compiled_pipeline(device, HC_SOURCE, "hc", name)
}

/// Pipeline for a `qsa_gather.metal` kernel (the QSA decode row gather).
pub(crate) fn qsa_gather_pipeline(device: &Device, name: &str) -> Result<ComputePipeline> {
    compiled_pipeline(device, QSA_GATHER_SOURCE, "qsa_gather", name)
}

/// Pipeline for the `qsa_select.metal` kernel (the QSA decode block selection).
pub(crate) fn qsa_select_pipeline(device: &Device, name: &str) -> Result<ComputePipeline> {
    compiled_pipeline(device, QSA_SELECT_SOURCE, "qsa_select", name)
}

/// Pipeline for the PLE grouped gate and dilated convolution tail.
pub(crate) fn ple_pipeline(device: &Device, name: &str) -> Result<ComputePipeline> {
    compiled_pipeline(device, PLE_SOURCE, "ple", name)
}

/// Pipeline for the bandwidth probes.
pub(crate) fn bandwidth_pipeline(device: &Device, name: &str) -> Result<ComputePipeline> {
    compiled_pipeline(device, BW_SOURCE, "bandwidth", name)
}
