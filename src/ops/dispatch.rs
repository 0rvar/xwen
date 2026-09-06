//! Host-side dispatch plumbing shared by the indexed-MoE matvec/matmul kernels.
//!
//! candle's metallib ships `kernel_mul_mv_id_*` / `kernel_mul_mm_id_*` (the
//! quantized gather matmuls used for the expert FFN) but wires no Rust host code
//! to them. These helpers encode those kernels directly, mirroring the geometry
//! candle uses for its non-indexed `call_quantized_matmul_mv_t` / `_mm_t` and the
//! ggml-metal reference encode functions.

use std::sync::{Arc, OnceLock};

use anyhow::{Result, bail, ensure};
use candle_core::quantized::GgmlDType;
use candle_core::{DType, Device, MetalDevice, MetalStorage, Shape, Storage, Tensor};
use candle_metal_kernels::metal::{Buffer, ComputeCommandEncoder, ComputePipeline};
use candle_metal_kernels::source::Source;
use candle_metal_kernels::utils::EncoderProvider;

use crate::config::ZGate;
use crate::gguf::{ExpertStack, QuantPlane};
use crate::ops::{MmVariant, pipelines};

/// A grid or threadgroup size, spelled once so every dispatch below reads the
/// same way.
///
/// This was a macro that built the value by calling candle's `get_block_dims`
/// and overwriting all three of its fields, on the stated grounds that
/// `objc2_metal::MTLSize` could not be named here. It can: the objc2 crates are
/// direct dependencies, `=`-pinned to exactly what the pinned candle rev
/// resolves (Cargo.toml), so this is the same type candle's own signatures use
/// and a plain struct literal builds it. The round trip through `get_block_dims`
/// computed nothing that survived — every field it returned was overwritten —
/// so nothing about any dispatch changed when it went.
const fn mtl_size(width: usize, height: usize, depth: usize) -> objc2_metal::MTLSize {
    objc2_metal::MTLSize {
        width,
        height,
        depth,
    }
}

/// Everything a single indexed dispatch needs, resolved to raw device buffers.
/// Shapes follow the seam contract: weights `[n_expert, n_out, k]`, activations
/// `x` `[t, x_per_row, k]`, ids `[t, top_k]`, output `[t, top_k, n_out]`.
pub(crate) struct IdDispatch<'a> {
    pub weights: &'a Buffer,
    /// Byte offset of the expert stack's first block inside `weights`
    /// (`ExpertStack.base_off`): 0 for a dedicated classic allocation, the
    /// sub-page file offset for an mmap-aliased page-floored view. EVERY
    /// encode that binds `weights` must pass this as the buffer offset.
    pub w_off: usize,
    pub x: &'a Buffer,
    pub x_off: usize,
    pub ids: &'a Buffer,
    pub ids_off: usize,
    pub dst: &'a Buffer,
    pub n_expert: usize,
    pub n_out: usize,
    pub k: usize,
    pub t: usize,
    pub top_k: usize,
    pub x_per_row: usize,
    /// Byte stride between rows of one expert = (k / block_size) * type_size.
    pub bytes_per_row: usize,
    /// Byte stride between experts = n_out * bytes_per_row.
    pub per_expert: usize,
}

/// Threadgroup geometry for the matvec kernels, per quantized dtype. Copied from
/// candle's `call_quantized_matmul_mv_t` table — the id kernels dispatch the same
/// per-dtype `impl_fn`, so they want the same `(nth0, nth1)` threadgroup shape and
/// the same `align` row-block grouping over the output dimension.
struct MvGeom {
    nth0: usize,
    nth1: usize,
    align: usize,
}

fn mv_geom(dt: GgmlDType) -> Result<MvGeom> {
    let g = match dt {
        GgmlDType::Q4_0 | GgmlDType::Q4_1 | GgmlDType::Q5_0 | GgmlDType::Q5_1 | GgmlDType::Q8_0 => {
            MvGeom {
                nth0: 8,
                nth1: 8,
                align: 8,
            }
        }
        GgmlDType::Q2K => MvGeom {
            nth0: 2,
            nth1: 32,
            align: 4,
        },
        GgmlDType::Q4K => MvGeom {
            nth0: 4,
            nth1: 8,
            align: 4,
        },
        GgmlDType::Q3K | GgmlDType::Q5K => MvGeom {
            nth0: 2,
            nth1: 32,
            align: 4,
        },
        GgmlDType::Q6K => MvGeom {
            nth0: 2,
            nth1: 32,
            align: 2,
        },
        GgmlDType::F16 | GgmlDType::F32 => MvGeom {
            nth0: 32,
            nth1: 1,
            align: 8,
        },
        other => bail!("no kernel_mul_mv_id kernel for dtype {other:?}"),
    };
    Ok(g)
}

fn mv_kernel_name(dt: GgmlDType) -> Result<&'static str> {
    let n = match dt {
        GgmlDType::Q4_0 => "kernel_mul_mv_id_q4_0_f32",
        GgmlDType::Q4_1 => "kernel_mul_mv_id_q4_1_f32",
        GgmlDType::Q5_0 => "kernel_mul_mv_id_q5_0_f32",
        GgmlDType::Q5_1 => "kernel_mul_mv_id_q5_1_f32",
        GgmlDType::Q8_0 => "kernel_mul_mv_id_q8_0_f32",
        GgmlDType::Q2K => "kernel_mul_mv_id_q2_K_f32",
        GgmlDType::Q3K => "kernel_mul_mv_id_q3_K_f32",
        GgmlDType::Q4K => "kernel_mul_mv_id_q4_K_f32",
        GgmlDType::Q5K => "kernel_mul_mv_id_q5_K_f32",
        GgmlDType::Q6K => "kernel_mul_mv_id_q6_K_f32",
        GgmlDType::F16 => "kernel_mul_mv_id_f16_f32",
        GgmlDType::F32 => "kernel_mul_mv_id_f32_f32",
        other => bail!("no kernel_mul_mv_id kernel for dtype {other:?}"),
    };
    Ok(n)
}

/// The vendored two-pass `kernel_mul_mm_id_<dtype>_f32` (src/ops/mm_id.metal)
/// is only instantiated for the dtypes the tests and the shipped checkpoints'
/// experts use (q5_1 is Qwen3.8-Flash-Next's `ffn_down_exps` dtype on most of its
/// layers); other dtypes stay on the mv_id path.
pub(crate) fn mm_kernel_name(dt: GgmlDType) -> Result<&'static str> {
    let n = match dt {
        GgmlDType::Q5_1 => "kernel_mul_mm_id_q5_1_f32",
        GgmlDType::Q8_0 => "kernel_mul_mm_id_q8_0_f32",
        GgmlDType::Q4K => "kernel_mul_mm_id_q4_K_f32",
        GgmlDType::Q5K => "kernel_mul_mm_id_q5_K_f32",
        GgmlDType::Q6K => "kernel_mul_mm_id_q6_K_f32",
        other => bail!("no vendored kernel_mul_mm_id kernel for dtype {other:?}; use mul_mv_id"),
    };
    Ok(n)
}

/// Whether the vendored `kernel_mul_mm_id_<dtype>_f32<variant-suffix>` kernel is
/// actually instantiated in mm_id.metal for this (dtype, variant) pair. The base
/// dtype matrix (q5_1/q8_0/q4_K/q5_K/q6_K) is `mm_kernel_name`; ON TOP of that, the
/// `_t_hp` (`TensorHp`) variant is instantiated ONLY for q4_K/q6_K (covers the
/// current official file's all-q4_K experts; q6_K was the retired original's
/// expert-down dtype) — the other three variants cover the full base matrix. A combo outside
/// this matrix has no pipeline, so `moe` must fall back to mv_id rather than fault
/// the pipeline lookup. Keep in lockstep with the `template [[host_name(...)]]`
/// instantiations in mm_id.metal (the `mm_id::tests::instantiation_matrix_matches_metal`
/// test cross-checks this against the source).
pub(crate) fn mm_kernel_instantiated(dt: GgmlDType, variant: MmVariant) -> bool {
    if mm_kernel_name(dt).is_err() {
        return false;
    }
    match variant {
        MmVariant::TensorHp => matches!(dt, GgmlDType::Q4K | GgmlDType::Q6K),
        MmVariant::Tensor | MmVariant::ClassicHp | MmVariant::ClassicF16 => true,
    }
}

/// The `kernel_mul_mm_id_map0` template is instantiated for these top_k values
/// in mm_id.metal; a top_k outside the set has no map0 pass.
pub(crate) fn map0_kernel_name(top_k: usize) -> Result<String> {
    match top_k {
        1 | 2 | 4 | 5 | 6 | 8 | 10 => Ok(format!("kernel_mul_mm_id_map0_ne20_{top_k}")),
        other => bail!("no kernel_mul_mm_id_map0 instantiation for top_k={other}; use mul_mv_id"),
    }
}

// The 64xNR1-tile threadgroup memory each mm_id variant reserves (fixed
// regardless of token count — the two-pass row map lives in device scratch, not
// threadgroup memory) is `MmVariant::tile_smem(nr1)`: at NR1 32, 8192 B for the
// half-tile variants (sa 4096 + sb 2048, store-back float tile reuses the region
// up to NR0*NR1*4 = 8192) and 12288 B for the f32 `_hp` tiles (sa 8192 + sb
// 4096); at NR1 64 (tensor family only) 16384 B, the store-back tile dominating.
/// Apple-silicon threadgroup memory ceiling; we refuse a launch that would exceed
/// it rather than let the GPU fault.
const MAX_THREADGROUP_SMEM: usize = 32768;

/// `ggml_metal_kargs_mul_mm_id_map0` (ggml-metal-impl.h). `#[repr(C)]` matches
/// the Metal `constant` struct layout byte-for-byte.
#[repr(C)]
#[derive(Clone, Copy)]
struct Map0Args {
    ne02: i32,
    ne10: i32,
    ne11: i32,
    nb11: u64,
    nb12: u64,
    ne21: i32,
    ne20: i32,
    nb21: u64,
    /// xwen extension: the pass-2 token-tile width the work list is built for.
    nr1: i32,
}

/// `ggml_metal_kargs_mul_mm_id` (ggml-metal-impl.h).
#[repr(C)]
#[derive(Clone, Copy)]
struct MmIdArgs {
    ne00: i32,
    ne02: i32,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne11: i32,
    nb10: u64,
    nb11: u64,
    nb12: u64,
    nb13: u64,
    ne20: i32,
    ne21: i32,
    ne0: i32,
    ne1: i32,
    r2: i16,
    r3: i16,
    /// xwen extension: 1 = grid.x indexes map0's (expert, tile) work list,
    /// 0 = the full ggml grid. Occupies the verbatim struct's trailing pad.
    work_list: i32,
}

/// `ggml_metal_kargs_mul_mv` (ggml-metal-impl.h). Written to buffer(0) of the
/// vendored plain mat-vec kernels (`kernel_mul_mv_<dtype>_f32_v`). `#[repr(C)]`
/// matches the Metal `constant` struct layout byte-for-byte.
#[repr(C)]
#[derive(Clone, Copy)]
struct MvArgs {
    ne00: i32,
    ne01: i32,
    ne02: i32,
    nb00: u64,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne10: i32,
    ne11: i32,
    ne12: i32,
    nb10: u64,
    nb11: u64,
    nb12: u64,
    nb13: u64,
    ne0: i32,
    ne1: i32,
    nr0: i32,
    r2: i16,
    r3: i16,
}

/// `ggml_metal_kargs_mul_mm` (ggml-metal-impl.h). Written to buffer(0) of the
/// vendored `kernel_mul_mm_f16_f32_v` prefill gemm. `#[repr(C)]` matches the
/// Metal `constant` struct layout byte-for-byte.
#[repr(C)]
#[derive(Clone, Copy)]
struct MmArgs {
    ne00: i32,
    ne02: i32,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne12: i32,
    nb10: u64,
    nb11: u64,
    nb12: u64,
    nb13: u64,
    ne0: i32,
    ne1: i32,
    r2: i16,
    r3: i16,
}

/// The `mv_ext_args` struct of src/ops/mv_ext.metal — ggml's
/// `ggml_metal_kargs_mul_mv_ext` minus the batch/broadcast fields, which are
/// all 1 at batch 1. `#[repr(C)]` matches the Metal `constant` struct layout
/// byte-for-byte.
#[repr(C)]
#[derive(Clone, Copy)]
struct MvExtArgs {
    ne00: i32,
    ne01: i32,
    nb01: u64,
    ne11: i32,
    nb11: u64,
    ne0: i32,
}

/// `ggml_metal_kargs_mul_mv_id` (ggml-metal-impl.h). Written to buffer(0) of the
/// vendored indexed mat-vec kernels (`kernel_mul_mv_id_<dtype>_f32_v`).
#[repr(C)]
#[derive(Clone, Copy)]
struct MvIdArgs {
    nei0: i32,
    nei1: i32,
    nbi1: u64,
    ne00: i32,
    ne01: i32,
    ne02: i32,
    nb00: u64,
    nb01: u64,
    nb02: u64,
    ne10: i32,
    ne11: i32,
    ne12: i32,
    ne13: i32,
    nb10: u64,
    nb11: u64,
    nb12: u64,
    ne0: i32,
    ne1: i32,
    nb1: u64,
    nr0: i32,
}

/// Matches the Metal `combine_args` struct (src/ops/combine.metal). `#[repr(C)]`
/// pins the layout byte-for-byte.
#[repr(C)]
#[derive(Clone, Copy)]
struct CombineArgs {
    top_k: i32,
    n_out: i32,
}

/// Matches the Metal `silu_mul_args` struct (src/ops/silu_mul.metal).
/// `#[repr(C)]` pins the layout byte-for-byte.
#[repr(C)]
#[derive(Clone, Copy)]
struct SiluMulArgs {
    n: i32,
}

/// Matches the Metal `silu_mul_l2_args` struct (src/ops/silu_mul.metal).
#[repr(C)]
#[derive(Clone, Copy)]
struct SiluMulL2Args {
    ff: i32,
    n_rows: i32,
    scale: f32,
    clamp_min: f32,
    clamp_max: f32,
}

/// `kernel_moe_silu_mul_l2`'s launch shape: one 256-thread threadgroup per row,
/// holding the row's activation in a 1024-float threadgroup array. Rows wider
/// than that are the caller's problem (the candle chain).
pub(crate) const SILU_MUL_L2_THREADS: usize = 256;
pub(crate) const SILU_MUL_L2_MAX_FF: usize = 1024;

/// candle's `fast_sum` threadgroup width for a `top_k`-wide reduction:
/// `min(pipeline_max, next_pow2(top_k/2))`. The combine kernels reproduce it so
/// the simd_sum lane partition matches candle's reduction order bit-for-bit, but
/// they fold only ONE 32-lane simdgroup (see combine.metal), so a width above 32
/// would silently drop lanes 32.. — `run_combine` bails when this exceeds 32.
fn combine_reduction_width(top_k: usize) -> usize {
    (top_k / 2).next_power_of_two()
}

/// Whether the combine kernel's i32 index math (`down_base = s*top_k*n_out + c`,
/// plus the strided `k*n_out` loads) stays within i32 for the whole grid. The
/// largest flat index into `down` approaches `seq*top_k*n_out`; computed in i64
/// so the check itself cannot overflow. `run_combine` bails when this is false
/// rather than let the kernel wrap to a negative offset.
fn combine_index_fits_i32(seq: usize, top_k: usize, n_out: usize) -> bool {
    (seq as i64) * (top_k as i64) * (n_out as i64) <= i32::MAX as i64
}

/// The fork's host-side mv/mm break-even for the float-family mul_mat path
/// (ggml-metal-ops.cpp `ne11_mm_min`): the tiled matmul kernel is dispatched
/// when the token count EXCEEDS this, the gemv otherwise. `run_matmul_f16`
/// mirrors it. ggml's small-batch `mul_mv_ext` kernels cover exactly the 2..8
/// range that rides the gemv here; xwen vendors them for the QUANTIZED dtypes
/// only (mv_ext.metal), so the f16 attention projections still re-read their
/// weights per token in that window — ledgered in TODO.md.
const F16_MM_MIN_SEQ: usize = 8;

/// Fork host constants for `kernel_mul_mv_f16_f32_v` at our shapes: nr0 = 2
/// src0 rows per threadgroup (the only case in ggml's disp switch) and
/// nsg = min(4, ceil(ne00/128)) = 4 simdgroups splitting the K reduction
/// (every attention K is >= 3072). Baked into the kernels as well — the whole
/// dense mv family (`f16.metal`, `bf16.metal`, `f32.metal`) shares this one
/// grid, so `f32_mv::tests::mv_geometry_matches_metal` holds all three sources'
/// `#define MV_NR0` / `#define MV_NSG` equal to these.
pub(crate) const MV_F16_NR0: usize = 2;
pub(crate) const MV_F16_NSG: usize = 4;

/// Tile geometry of the dense cooperative-tensor quantized gemm
/// (dense_mm.metal's `kernel_mul_mm_q_f32_t`), which the host must agree with to
/// size the grid and the threadgroup allocation: NR0 = 64 out rows, NR1 = 128
/// tokens, and an A tile of NR0 x NK(32) halves — the only threadgroup memory
/// the kernel uses. `dense_mm::tests::geometry_matches_metal` cross-checks these
/// against the kernel source.
pub(crate) const DENSE_MM_NR0: usize = 64;
pub(crate) const DENSE_MM_NR1: usize = 128;
pub(crate) const DENSE_MM_A_TILE_SMEM: usize = DENSE_MM_NR0 * 32 * 2;

/// Geometry of the vendored small-batch mat-vec (mv_ext.metal), baked into the
/// kernel and mirrored here for the grid: `NSG` simdgroups per threadgroup and
/// `NXPSG` threads along each row, hence `NYPSG = 32/NXPSG` rows per simdgroup
/// and `R0PTG = NYPSG*NSG` weight rows per threadgroup. These are the values
/// ggml's host picks for our shapes (ggml-metal-ops.cpp:2160-2172: nsg is always
/// 2, nxpsg is 8 whenever ne00 % 128 == 0).
/// `mv_ext::tests::geometry_matches_metal` cross-checks them against the source.
pub(crate) const MV_EXT_NSG: usize = 2;
pub(crate) const MV_EXT_NXPSG: usize = 8;
pub(crate) const MV_EXT_NYPSG: usize = 32 / MV_EXT_NXPSG;
pub(crate) const MV_EXT_R0PTG: usize = MV_EXT_NYPSG * MV_EXT_NSG;

/// K must be a whole multiple of this for the baked `nxpsg = 8` geometry: a
/// thread walks `chpt*nxpsg` chunks per pass (4*8 float4 chunks = 128 elements
/// for the 32-element block types, 1*8 float4x4 chunks = 128 for the K-quants),
/// and the kernel has no tail path. It is also ggml's own condition for
/// choosing this nxpsg.
pub(crate) const MV_EXT_K_MULTIPLE: usize = 128;

/// ggml's N_R0 / N_SG for the vendored q8_0 attention decode gemv
/// (kernel_mul_mv_q8_0_f32_attn, q8.metal): N_R0_Q8_0 = 2 rows per threadgroup,
/// N_SG_Q8_0 = 4 simdgroups splitting the K reduction (ggml-metal-impl.h). Same
/// f16-style geometry as the attention f16 gemv above, and held equal to
/// `q8.metal`'s own `#define MV_NR0` / `#define MV_NSG` by
/// `f32_mv::tests::mv_geometry_matches_metal`.
pub(crate) const MV_Q8_NR0: usize = 2;
pub(crate) const MV_Q8_NSG: usize = 4;

/// ggml's per-dtype (N_R0, N_SG) for the vendored mat-vec kernels
/// (ggml-metal-impl.h) plus the grid geometry the fork's host dispatch selects
/// (ggml-metal-ops.cpp). Two shapes coexist:
///   * K-quants (q4_K/q5_K/q6_K, `row_split=false`): each simdgroup owns its own
///     `nr0` output rows, so grid.x covers `ceil(ne01/(nr0*nsg))` blocks and the
///     kernel needs no threadgroup memory (bare simd_sum per row).
///   * q8_0 (`row_split=true`, like the f16 gemv): the `nsg` simdgroups split the
///     K reduction over the SAME `nr0` rows and combine through shmem, so grid.x
///     covers `ceil(ne01/nr0)` blocks and the kernel reserves `nr0*32` floats.
struct MvVendoredGeom {
    nr0: usize,
    nsg: usize,
    row_split: bool,
}

impl MvVendoredGeom {
    /// Threadgroup memory the kernel reads (`helper_mv_reduce_and_write`): `nr0`
    /// rows of one 32-lane simdgroup partial each. Zero for the K-quant kernels.
    fn smem_bytes(&self) -> usize {
        if self.row_split {
            self.nr0 * 32 * std::mem::size_of::<f32>()
        } else {
            0
        }
    }

    /// grid.x row-block count for `n_out` output rows.
    fn row_blocks(&self, n_out: usize) -> usize {
        if self.row_split {
            n_out.div_ceil(self.nr0)
        } else {
            n_out.div_ceil(self.nr0 * self.nsg)
        }
    }
}

fn mv_vendored_geom(dt: GgmlDType) -> Result<MvVendoredGeom> {
    let g = match dt {
        GgmlDType::Q4K | GgmlDType::Q6K => MvVendoredGeom {
            nr0: 2,
            nsg: 2,
            row_split: false,
        },
        GgmlDType::Q5K => MvVendoredGeom {
            nr0: 1,
            nsg: 2,
            row_split: false,
        },
        GgmlDType::Q8_0 => MvVendoredGeom {
            nr0: 2,
            nsg: 4,
            row_split: true,
        },
        other => bail!("no vendored ggml-geometry mat-vec for dtype {other:?}"),
    };
    Ok(g)
}

/// True iff the vendored ggml-geometry mat-vec kernels exist for `dt`. The
/// current official Q4_K_M touches q4_K (routed experts) and q8_0 (shared
/// experts / lm_head); q6_K and q5_K date from the retired checkpoints (the
/// original's q6_K expert-downs + lm_head, the unsloth UD's q5_K experts) and
/// stay supported for any file that carries them. Other dtypes stay on the
/// candle baked path.
pub fn mv_vendored_supported(dt: GgmlDType) -> bool {
    matches!(
        dt,
        GgmlDType::Q4K | GgmlDType::Q5K | GgmlDType::Q6K | GgmlDType::Q8_0
    )
}

fn mv_vendored_id_kernel_name(dt: GgmlDType) -> Result<&'static str> {
    match dt {
        GgmlDType::Q4K => Ok("kernel_mul_mv_id_q4_K_f32_v"),
        GgmlDType::Q5K => Ok("kernel_mul_mv_id_q5_K_f32_v"),
        GgmlDType::Q6K => Ok("kernel_mul_mv_id_q6_K_f32_v"),
        GgmlDType::Q8_0 => Ok("kernel_mul_mv_id_q8_0_f32_v"),
        other => bail!("no vendored kernel_mul_mv_id kernel for dtype {other:?}"),
    }
}

fn mv_vendored_plain_kernel_name(dt: GgmlDType) -> Result<&'static str> {
    match dt {
        GgmlDType::Q4K => Ok("kernel_mul_mv_q4_K_f32_v"),
        GgmlDType::Q5K => Ok("kernel_mul_mv_q5_K_f32_v"),
        GgmlDType::Q6K => Ok("kernel_mul_mv_q6_K_f32_v"),
        GgmlDType::Q8_0 => Ok("kernel_mul_mv_q8_0_f32_v"),
        other => bail!("no vendored kernel_mul_mv kernel for dtype {other:?}"),
    }
}

/// Encode the vendored `kernel_mul_mv_id_<dtype>_f32_v` (decode path). Same seam
/// contract as `encode_mul_mv_id`, but dispatches our ggml-geometry kernel with
/// the per-dtype `MvVendoredGeom`: the K-quant kernels give each simdgroup its
/// own `nr0` rows (`ceil(n_out/(nr0*nsg))` row-blocks), the q8_0 kernel splits
/// the K reduction over `nsg` simdgroups (`ceil(n_out/nr0)` row-blocks + shmem).
/// grid.z enumerates every (token, slot) pair (the id wrapper decodes z). The
/// argument struct goes to buffer(0) (ggml layout), matching the kernel signature.
pub(crate) fn encode_mul_mv_id_vendored(
    device: &MetalDevice,
    ep: impl EncoderProvider,
    dt: GgmlDType,
    d: &IdDispatch,
) -> Result<()> {
    let name = mv_vendored_id_kernel_name(dt)?;
    let geom = mv_vendored_geom(dt)?;
    let pipeline = pipelines::mv_pipeline(device.device(), name)?;

    let args = MvIdArgs {
        nei0: d.top_k as i32,
        nei1: d.t as i32,
        nbi1: (d.top_k * DType::U32.size_in_bytes()) as u64,
        ne00: d.k as i32,
        ne01: d.n_out as i32,
        ne02: d.n_expert as i32,
        nb00: 0,
        nb01: d.bytes_per_row as u64,
        nb02: d.per_expert as u64,
        ne10: d.k as i32,
        ne11: d.x_per_row as i32,
        ne12: d.t as i32,
        ne13: 1,
        nb10: DType::F32.size_in_bytes() as u64,
        nb11: (d.k * DType::F32.size_in_bytes()) as u64,
        nb12: (d.x_per_row * d.k * DType::F32.size_in_bytes()) as u64,
        ne0: d.n_out as i32,
        ne1: d.top_k as i32,
        nb1: (d.n_out * DType::F32.size_in_bytes()) as u64,
        nr0: geom.nr0 as i32,
    };

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_bytes(0, &args);
    encoder.set_input_buffer(1, Some(d.weights), d.w_off);
    encoder.set_input_buffer(2, Some(d.x), d.x_off);
    encoder.set_output_buffer(3, Some(d.dst), 0);
    encoder.set_input_buffer(4, Some(d.ids), d.ids_off);
    // The q8_0 kernel's cross-simdgroup reduce reads threadgroup memory; the
    // K-quant kernels declare none (smem_bytes == 0).
    let smem = geom.smem_bytes();
    if smem > 0 {
        encoder.set_threadgroup_memory_length(0, smem);
    }

    // grid.z walks every (token, slot) pair; threads are `nsg` simdgroups of 32.
    let grid = mtl_size(geom.row_blocks(d.n_out), 1, d.top_k * d.t);
    let threads = mtl_size(32, geom.nsg, 1);
    encoder.dispatch_thread_groups(grid, threads);
    Ok(())
}

/// Encode `kernel_mul_mv_id_<dtype>_f32` (decode path). Each threadgroup along z
/// handles one (token, expert-slot) pair; the kernel reads the expert id from the
/// ids buffer and offsets `weights` by `expert * per_expert`.
pub(crate) fn encode_mul_mv_id(
    device: &MetalDevice,
    ep: impl EncoderProvider,
    dt: GgmlDType,
    d: &IdDispatch,
) -> Result<()> {
    let geom = mv_geom(dt)?;
    let name = mv_kernel_name(dt)?;

    // Kernel argument order mirrors kernel_mul_mv_id's signature exactly.
    let nei0 = d.top_k as i64;
    let nei1 = d.t as i64;
    let nbi1 = (d.top_k * DType::U32.size_in_bytes()) as u64;
    let ne00 = d.k as i64;
    let ne01 = d.n_out as i64;
    let ne02 = d.n_expert as i64;
    let nb00 = 0u64;
    let nb01 = d.bytes_per_row as u64;
    let nb02 = d.per_expert as u64;
    let ne10 = d.k as i64;
    let ne11 = d.x_per_row as i64;
    let ne12 = d.t as i64;
    let ne13 = 1i64;
    let nb10 = DType::F32.size_in_bytes() as u64;
    let nb11 = (d.k * DType::F32.size_in_bytes()) as u64;
    let nb12 = (d.x_per_row * d.k * DType::F32.size_in_bytes()) as u64;
    let ne0 = d.n_out as i64;
    let ne1 = d.top_k as i64;
    let nb1 = (d.n_out * DType::F32.size_in_bytes()) as u64;

    let pipeline = device
        .kernels()
        .load_pipeline(device.device(), Source::Quantized, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    candle_metal_kernels::set_params!(
        encoder,
        (
            (d.weights, d.w_off),
            (d.x, d.x_off),
            candle_metal_kernels::Output::new(d.dst),
            (d.ids, d.ids_off),
            nei0,
            nei1,
            nbi1,
            ne00,
            ne01,
            ne02,
            nb00,
            nb01,
            nb02,
            ne10,
            ne11,
            ne12,
            ne13,
            nb10,
            nb11,
            nb12,
            ne0,
            ne1,
            nb1
        )
    );

    // grid.x groups the n_out rows into `align`-wide blocks; grid.z walks all
    // top_k*t (token, slot) pairs (the id wrapper decodes z into token+slot).
    let grid = mtl_size(d.n_out.div_ceil(geom.align), 1, d.top_k * d.t);
    let threads = mtl_size(geom.nth0, geom.nth1, 1);
    encoder.dispatch_thread_groups(grid, threads);
    Ok(())
}

/// Count of 4-byte scratch slots `run` over-allocates on the dst buffer's tail
/// for the mm_id two-pass: the per-expert token count (`tpe`, n_expert i32) then
/// the per-expert compacted token-slot list (`ids-map`, n_expert*t i32). The dst
/// buffer is f32 and these entries are i32 (both 4 bytes), so one slot == one
/// dst element. Living in the dst allocation, the scratch shares its lifetime
/// (the returned tensor keeps it resident) instead of racing the buffer pool.
pub(crate) fn mm_scratch_elems(n_expert: usize, t: usize, top_k: usize, nr1: usize) -> usize {
    n_expert + n_expert * t + mm_work_list_elems(n_expert, t, top_k, nr1)
}

/// 4-byte slots of the pass-2 work list map0 appends after the ids-map: one
/// count then the packed (expert, tile) pairs. Bounded WITHOUT a readback:
/// rows sum to `t*top_k` and each expert adds at most one partial tile, so
/// `ceil(t*top_k/nr1) + n_expert` pairs suffice.
pub(crate) fn mm_work_list_elems(n_expert: usize, t: usize, top_k: usize, nr1: usize) -> usize {
    1 + mm_tiles_max(n_expert, t, top_k, nr1)
}

/// Upper bound on the number of (expert, tile) pairs in the work list; also the
/// work-list grid's x extent.
fn mm_tiles_max(n_expert: usize, t: usize, top_k: usize, nr1: usize) -> usize {
    (t * top_k).div_ceil(nr1) + n_expert
}

/// Byte offset of the work list inside a scratch region whose tpe is at 0.
fn mm_work_off(n_expert: usize, t: usize) -> usize {
    (n_expert + n_expert * t) * MM_SCRATCH_ENTRY_BYTES
}

/// Presence switch `XWEN_MM_ID_FULL_GRID`: launch pass 2 on the full ggml grid
/// (one column of token tiles per expert, sized for the whole chunk) instead of
/// map0's work list. Kill switch for the work-list grid; read once.
pub(crate) fn mm_id_full_grid() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var_os("XWEN_MM_ID_FULL_GRID").is_some())
}

/// `XWEN_MM_ID_NR1=32|64`: force the pass-2 token-tile width. Read once; any
/// other value is a startup-class error surfaced at the first mm_id dispatch.
fn mm_id_nr1_env() -> Result<Option<usize>> {
    static V: OnceLock<Result<Option<usize>, String>> = OnceLock::new();
    V.get_or_init(|| match std::env::var("XWEN_MM_ID_NR1") {
        Ok(v) if v == "32" => Ok(Some(32)),
        Ok(v) if v == "64" => Ok(Some(64)),
        Ok(v) => Err(format!("XWEN_MM_ID_NR1 must be 32 or 64, got {v:?}")),
        Err(_) => Ok(None),
    })
    .clone()
    .map_err(|e| anyhow::anyhow!(e))
}

/// Per-dispatch overrides for the pass-2 launch shape, for A/B tests and
/// benches that must not touch the env. `None` = the production rule.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct MmTuning {
    pub nr1: Option<usize>,
    pub full_grid: Option<bool>,
}

/// The pass-2 token-tile width for this geometry. Only the tensor (`_t`)
/// family is instantiated at 64; there the width follows the mean routed rows
/// per expert (`t*top_k/n_expert >= 24` → 64: an expert with 40-64 rows then
/// dequantizes its weight tile once instead of twice). Other families stay at 32.
pub(crate) fn mm_nr1(
    variant: MmVariant,
    t: usize,
    top_k: usize,
    n_expert: usize,
    tuning: MmTuning,
) -> Result<usize> {
    // Validate the env value on every family, so a typo errors even where the
    // width cannot change.
    let forced = mm_id_nr1_env()?;
    if variant != MmVariant::Tensor {
        return Ok(32);
    }
    if let Some(nr1) = tuning.nr1 {
        return Ok(nr1);
    }
    if let Some(nr1) = forced {
        return Ok(nr1);
    }
    Ok(if t * top_k >= 24 * n_expert { 64 } else { 32 })
}

fn mm_full_grid(tuning: MmTuning) -> bool {
    tuning.full_grid.unwrap_or_else(mm_id_full_grid)
}

/// The live-field subset needed to encode the map0 pass. map0's output (per-expert
/// token count + compacted token-slot list) depends ONLY on the ids and
/// t/top_k/n_expert — the expert count comes from the dispatched thread count and
/// the other `Map0Args` fields (ne10/ne11/nb11/nb12) are not read by the kernel —
/// so ONE map0 pass serves every projection of a MoE block regardless of each
/// projection's k / x_per_row (they differ between gate/up and down).
struct Map0Dispatch<'a> {
    ids: &'a Buffer,
    ids_off: usize,
    n_expert: usize,
    top_k: usize,
    t: usize,
    /// Pass-2 token-tile width the work list is built for.
    nr1: usize,
}

/// Byte width of one scratch entry (tpe counts and ids-map slots are both i32).
const MM_SCRATCH_ENTRY_BYTES: usize = 4;

/// Encode the map0 pass: one thread per expert builds that expert's compacted
/// token-slot list (`ids-map`, written at `ids_map_off`) and its token-slot count
/// (`tpe`, written at `tpe_off`) into `scratch`, then the pass-2 work list
/// (count + packed (expert, tile) pairs) at `work_off`. `tpe` is `n_expert` i32;
/// `ids-map` is `n_expert*t` i32; the work list is `mm_work_list_elems` u32 (see
/// `mm_scratch_elems`). The dead `Map0Args` fields (ne10/ne11/nb11/nb12) are
/// zeroed — the kernel never reads them.
fn encode_map0(
    device: &MetalDevice,
    ep: impl EncoderProvider,
    m: &Map0Dispatch,
    scratch: &Buffer,
    tpe_off: usize,
    ids_map_off: usize,
    work_off: usize,
) -> Result<()> {
    let map0_name = map0_kernel_name(m.top_k)?;
    let map0 = pipelines::mm_id_pipeline(device.device(), &map0_name)?;
    check_mm_id_bounds(m.n_expert, m.t, m.nr1)?;

    // map0 runs one thread per expert, rounded up to whole simdgroups so the
    // work-list scan's simd_* ops are defined (phantom threads contribute zero
    // and write nothing). The ids scratch holds one u16 per (thread, slot), and
    // the scan reuses it for one u32 per simdgroup (at most 32 for a 1024-thread
    // group).
    let ntg = m.n_expert.div_ceil(32) * 32;
    let map0_smem = (ntg * m.top_k * std::mem::size_of::<u16>()).max(32 * 4);
    if map0_smem > MAX_THREADGROUP_SMEM {
        bail!(
            "kernel_mul_mm_id_map0 needs {map0_smem} bytes of threadgroup memory for \
             n_expert={} top_k={}, over the {MAX_THREADGROUP_SMEM}-byte limit",
            m.n_expert,
            m.top_k
        );
    }
    if ntg > map0.max_total_threads_per_threadgroup() {
        bail!(
            "kernel_mul_mm_id_map0 dispatches {ntg} threads/threadgroup (n_expert={} rounded \
             up to whole simdgroups), over the pipeline max {}",
            m.n_expert,
            map0.max_total_threads_per_threadgroup()
        );
    }

    let map0_args = Map0Args {
        ne02: m.n_expert as i32,
        ne10: 0,
        ne11: 0,
        nb11: 0,
        nb12: 0,
        ne21: m.t as i32,
        ne20: m.top_k as i32,
        nb21: (m.top_k * DType::U32.size_in_bytes()) as u64,
        nr1: m.nr1 as i32,
    };
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    // buffers: 0=args, 1=ids, 2=tpe out, 3=ids-map out, 4=work list out.
    encoder.set_compute_pipeline_state(&map0);
    encoder.set_bytes(0, &map0_args);
    encoder.set_input_buffer(1, Some(m.ids), m.ids_off);
    encoder.set_output_buffer(2, Some(scratch), tpe_off);
    encoder.set_output_buffer(3, Some(scratch), ids_map_off);
    encoder.set_output_buffer(4, Some(scratch), work_off);
    encoder.set_threadgroup_memory_length(0, map0_smem);
    encoder.dispatch_thread_groups(mtl_size(1, 1, 1), mtl_size(ntg, 1, 1));
    Ok(())
}

/// Index-width limits of the mm_id kernels, checked on both passes. The kernels
/// address tokens and rows with `short` (the ggml port's choice), so a prefill
/// chunk above 32768 tokens — reachable only through `XWEN_PREFILL_CHUNK` —
/// would wrap; the packed work-list entry holds the expert and the tile index
/// in 16 bits each.
fn check_mm_id_bounds(n_expert: usize, t: usize, nr1: usize) -> Result<()> {
    ensure!(
        t <= 32768,
        "mm_id: a {t}-token chunk exceeds the kernels' 16-bit token indexing (max 32768); \
         lower XWEN_PREFILL_CHUNK"
    );
    ensure!(
        n_expert < 65536,
        "mm_id: n_expert={n_expert} does not fit the 16-bit work-list expert field"
    );
    ensure!(
        t.div_ceil(nr1) < 65536,
        "mm_id: {t} tokens at NR1 {nr1} exceed the 16-bit work-list tile field"
    );
    Ok(())
}

/// Encode the token-grouped matmul pass: each expert's threadgroups cover only
/// its own rows, read from the `tpe`/`ids-map` regions of `scratch` that a prior
/// `encode_map0` wrote. Writes the `[t, top_k, n_out]` result to `d.dst`.
///
/// `variant` picks the mm_id kernel family (tensor `_t` / classic `_hp` / classic
/// f16), threaded in from the single cached read in `ops::mm_id_variant`, never
/// re-read here. It sets the kernel host-name suffix and the tile smem.
///
/// Ordering: `encode_map0` marked tpe/ids-map as outputs and this pass reads them
/// as inputs on the same buffer, so candle inserts the RAW barrier automatically
/// (its Output-mark hazard tracking within an encoder, or the per-encoder fence
/// wait across encoders when the two passes are submitted separately).
///
/// `nr1` is the token-tile width (`mm_nr1`; 64 selects the `_t64` kernels) and
/// must match what the map0 pass built its work list for. `full_grid` launches
/// the ggml grid instead of walking that list.
#[allow(clippy::too_many_arguments)]
fn encode_mm(
    device: &MetalDevice,
    ep: impl EncoderProvider,
    dt: GgmlDType,
    d: &IdDispatch,
    variant: MmVariant,
    nr1: usize,
    full_grid: bool,
    scratch: &Buffer,
    tpe_off: usize,
    ids_map_off: usize,
    work_off: usize,
) -> Result<()> {
    let mm_name = format!("{}{}", mm_kernel_name(dt)?, variant.suffix_nr1(nr1)?);
    let mm = pipelines::mm_id_pipeline(device.device(), &mm_name)?;
    check_mm_id_bounds(d.n_expert, d.t, nr1)?;
    let tile_smem = variant.tile_smem(nr1);
    ensure!(
        tile_smem <= MAX_THREADGROUP_SMEM,
        "{mm_name} needs {tile_smem} bytes of threadgroup memory, over the \
         {MAX_THREADGROUP_SMEM}-byte limit"
    );

    let nb11 = (d.k * DType::F32.size_in_bytes()) as u64;
    let nb12 = (d.x_per_row * d.k * DType::F32.size_in_bytes()) as u64;

    let mm_args = MmIdArgs {
        ne00: d.k as i32,
        ne02: d.n_expert as i32,
        nb01: d.bytes_per_row as u64,
        nb02: d.per_expert as u64,
        nb03: 0,
        ne11: d.x_per_row as i32,
        nb10: DType::F32.size_in_bytes() as u64,
        nb11,
        nb12,
        nb13: 0,
        ne20: d.top_k as i32,
        ne21: d.t as i32,
        ne0: d.n_out as i32,
        ne1: d.top_k as i32,
        r2: 1,
        r3: 1,
        work_list: if full_grid { 0 } else { 1 },
    };
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    // buffers: 0=args, 1=weights, 2=x, 3=tpe, 4=ids-map, 5=dst, 6=work list.
    encoder.set_compute_pipeline_state(&mm);
    encoder.set_bytes(0, &mm_args);
    encoder.set_input_buffer(1, Some(d.weights), d.w_off);
    encoder.set_input_buffer(2, Some(d.x), d.x_off);
    encoder.set_input_buffer(3, Some(scratch), tpe_off);
    encoder.set_input_buffer(4, Some(scratch), ids_map_off);
    encoder.set_output_buffer(5, Some(d.dst), 0);
    encoder.set_input_buffer(6, Some(scratch), work_off);
    encoder.set_threadgroup_memory_length(0, tile_smem);

    // grid: 64-wide n_out rows on y; on x either every (expert, tile) pair of
    // map0's work list (bounded by mm_tiles_max, the tail early-returns) or,
    // on the full grid, nr1-wide token tiles with one z-slab per expert.
    let grid = if full_grid {
        mtl_size(d.t.div_ceil(nr1), d.n_out.div_ceil(64), d.n_expert)
    } else {
        mtl_size(
            mm_tiles_max(d.n_expert, d.t, d.top_k, nr1),
            d.n_out.div_ceil(64),
            1,
        )
    };
    encoder.dispatch_thread_groups(grid, mtl_size(128, 1, 1));
    Ok(())
}

/// Encode the self-contained two-pass indexed matmul (standalone prefill path):
/// map0 then mm, with both scratch regions living at the tail of `d.dst` (offsets
/// past the `[t, top_k, n_out]` output). The returned tensor keeps the whole
/// allocation resident, so the scratch shares its lifetime instead of racing the
/// buffer pool. The shared-map0 production path (`prepare_mm_id_map0` +
/// `run_mm_shared`) uses a dedicated scratch buffer instead.
pub(crate) fn encode_mul_mm_id(
    device: &MetalDevice,
    ep: impl EncoderProvider + Copy,
    dt: GgmlDType,
    d: &IdDispatch,
    variant: MmVariant,
    tuning: MmTuning,
) -> Result<()> {
    let nr1 = mm_nr1(variant, d.t, d.top_k, d.n_expert, tuning)?;
    let tpe_off = d.t * d.top_k * d.n_out * DType::F32.size_in_bytes();
    let ids_map_off = tpe_off + d.n_expert * MM_SCRATCH_ENTRY_BYTES;
    let work_off = tpe_off + mm_work_off(d.n_expert, d.t);
    let m = Map0Dispatch {
        ids: d.ids,
        ids_off: d.ids_off,
        n_expert: d.n_expert,
        top_k: d.top_k,
        t: d.t,
        nr1,
    };
    encode_map0(device, ep, &m, d.dst, tpe_off, ids_map_off, work_off)?;
    encode_mm(
        device,
        ep,
        dt,
        d,
        variant,
        nr1,
        mm_full_grid(tuning),
        d.dst,
        tpe_off,
        ids_map_off,
        work_off,
    )?;
    Ok(())
}

/// Wrap a freshly written f32 device buffer as an owned output `Tensor`.
pub(crate) fn output_tensor(
    dst: Arc<Buffer>,
    device: &MetalDevice,
    count: usize,
    shape: impl Into<Shape>,
) -> Tensor {
    let storage = MetalStorage::new(dst, device.clone(), count, DType::F32);
    Tensor::from_storage(
        Storage::Metal(storage),
        shape,
        candle_core::op::BackpropOp::none(),
        false,
    )
}

/// Which id-kernel family to dispatch.
#[derive(Clone, Copy)]
pub(crate) enum Mode {
    /// candle's baked `kernel_mul_mv_id` — one matvec per (token, slot); decode
    /// path, older geometry. Kept as the `XWEN_MV_CLASSIC` fallback.
    Mv,
    /// Vendored ggml-geometry `kernel_mul_mv_id_<dtype>_f32_v` — decode path,
    /// current geometry (default for the `mv_vendored_supported` dtypes).
    MvVendored,
    /// `kernel_mul_mm_id` — token-grouped matmul; prefill path.
    Mm,
}

/// A shared map0 scratch (`prepare_mm_id_map0`) plus the geometry it was laid out
/// for. The producer wrote `tpe` (n_expert i32 @ 0) then the `ids-map` at
/// `n_expert * MM_SCRATCH_ENTRY_BYTES` using ITS n_expert/t/top_k; a consumer that
/// recomputed that offset from a different stack's n_expert would read the wrong
/// region. Carrying the geometry lets `run_mm_shared` validate each consuming
/// projection against the producer before it reads the map.
pub(crate) struct Map0Scratch {
    buffer: Arc<Buffer>,
    n_expert: usize,
    t: usize,
    top_k: usize,
    /// The token-tile width the work list was built for; a consumer dispatching
    /// at a different width would walk a list of the wrong tiles.
    nr1: usize,
}

/// Where `Mode::Mm` reads its map0 scratch from.
enum MmScratch<'a> {
    /// Self-contained: map0 runs here and both scratch regions live at the tail of
    /// the freshly allocated dst (the returned tensor keeps them resident).
    Owned,
    /// Shared: map0 already ran into this dedicated scratch (`prepare_mm_id_map0`),
    /// so only the mm pass runs here, reading tpe @ 0 and ids-map @ n_expert*4 —
    /// after validating this projection's geometry against the producer's.
    Shared(&'a Map0Scratch),
}

/// Validate the seam shapes, resolve every operand to a device buffer, and encode
/// the requested id kernel. Returns the `[t, top_k, n_out]` output tensor.
/// `variant` is only consulted for `Mode::Mm` (which mm_id kernel family);
/// callers pass the cached `ops::mm_id_variant()` in production and an explicit
/// value in A/B tests. `Mode::Mv` ignores it.
pub(crate) fn run(
    stack: &ExpertStack,
    x: &Tensor,
    ids: &Tensor,
    mode: Mode,
    variant: MmVariant,
) -> Result<Tensor> {
    run_inner(
        stack,
        x,
        ids,
        mode,
        variant,
        MmTuning::default(),
        MmScratch::Owned,
    )
}

/// `run` for `Mode::Mm` with an explicit pass-2 launch shape (tile width /
/// grid kind), for A/B tests and benches. Production goes through `run`.
#[cfg(test)]
pub(crate) fn run_mm_tuned(
    stack: &ExpertStack,
    x: &Tensor,
    ids: &Tensor,
    variant: MmVariant,
    tuning: MmTuning,
) -> Result<Tensor> {
    if let Some(nr1) = tuning.nr1 {
        ensure!(
            nr1 == 32 || nr1 == 64,
            "MmTuning.nr1 must be 32 or 64, got {nr1}"
        );
    }
    run_inner(stack, x, ids, Mode::Mm, variant, tuning, MmScratch::Owned)
}

/// Run one `Mode::Mm` projection against a shared map0 scratch (`prepare_mm_id_map0`),
/// skipping the map0 pass. Used by `FusedExperts::forward` so the block's three
/// projections build the token-slot map once. `scratch` must stay alive until this
/// dispatch is submitted (the caller holds it across gate/up/down).
pub(crate) fn run_mm_shared(
    stack: &ExpertStack,
    x: &Tensor,
    ids: &Tensor,
    variant: MmVariant,
    scratch: &Map0Scratch,
) -> Result<Tensor> {
    run_inner(
        stack,
        x,
        ids,
        Mode::Mm,
        variant,
        MmTuning::default(),
        MmScratch::Shared(scratch),
    )
}

fn run_inner(
    stack: &ExpertStack,
    x: &Tensor,
    ids: &Tensor,
    mode: Mode,
    variant: MmVariant,
    tuning: MmTuning,
    scratch: MmScratch,
) -> Result<Tensor> {
    let cdev = x.device().clone();
    let Device::Metal(mdev) = &cdev else {
        bail!("mul_*_id requires x on a Metal device");
    };

    let (t, x_per_row, kx) = x
        .dims3()
        .map_err(|e| anyhow::anyhow!("x must be rank-3 [t, x_per_row, k]: {e}"))?;
    let (t_ids, top_k) = ids
        .dims2()
        .map_err(|e| anyhow::anyhow!("ids must be rank-2 [t, top_k]: {e}"))?;

    if x.dtype() != DType::F32 {
        bail!("x must be f32, got {:?}", x.dtype());
    }
    if ids.dtype() != DType::U32 {
        bail!("ids must be u32, got {:?}", ids.dtype());
    }
    if !x.is_contiguous() {
        bail!("x must be contiguous");
    }
    if !ids.is_contiguous() {
        bail!("ids must be contiguous");
    }
    if kx != stack.k {
        bail!("x k ({kx}) does not match expert stack k ({})", stack.k);
    }
    if t_ids != t {
        bail!("ids t ({t_ids}) does not match x t ({t})");
    }
    if x_per_row != 1 && x_per_row != top_k {
        bail!("x_per_row ({x_per_row}) must be 1 (shared row) or top_k ({top_k}) (per-slot row)");
    }

    let dt = stack.dtype;
    let block_size = dt.block_size();
    if !stack.k.is_multiple_of(block_size) {
        bail!(
            "expert stack k ({}) is not a multiple of {dt:?} block size {block_size}",
            stack.k
        );
    }
    let bytes_per_row = stack.k / block_size * dt.type_size();
    let per_expert = stack.n_out * bytes_per_row;

    let Some(w_buf) = stack.buffer.as_deref() else {
        bail!(
            "expert stack has no device buffer (not on a Metal device); fused MoE requires Metal"
        );
    };

    let out_count = t * top_k * stack.n_out;
    // Owned Mm over-allocates the dst buffer to hold the two-pass scratch (tpe +
    // ids-map) at its tail; the returned tensor keeps the whole allocation
    // resident, so the scratch shares its lifetime and the pool reuses it once
    // the tensor drops. Shared Mm and the Mv paths write no scratch tail.
    let alloc_count = match (mode, &scratch) {
        (Mode::Mm, MmScratch::Owned) => {
            let nr1 = mm_nr1(variant, t, top_k, stack.n_expert, tuning)?;
            out_count + mm_scratch_elems(stack.n_expert, t, top_k, nr1)
        }
        _ => out_count,
    };
    let dst = mdev.new_buffer(alloc_count, DType::F32, "mul_id")?;

    let (x_guard, x_layout) = x.storage_and_layout();
    let Storage::Metal(x_storage) = &*x_guard else {
        bail!("x is not on a Metal device");
    };
    let x_buf = x_storage.buffer();
    let x_off = x_layout.start_offset() * DType::F32.size_in_bytes();

    let (ids_guard, ids_layout) = ids.storage_and_layout();
    let Storage::Metal(ids_storage) = &*ids_guard else {
        bail!("ids is not on a Metal device");
    };
    let ids_buf = ids_storage.buffer();
    let ids_off = ids_layout.start_offset() * DType::U32.size_in_bytes();

    let d = IdDispatch {
        weights: w_buf,
        w_off: stack.base_off,
        x: x_buf,
        x_off,
        ids: ids_buf,
        ids_off,
        dst: &dst,
        n_expert: stack.n_expert,
        n_out: stack.n_out,
        k: stack.k,
        t,
        top_k,
        x_per_row,
        bytes_per_row,
        per_expert,
    };
    {
        let cmd = mdev.command_encoder()?;
        match (mode, &scratch) {
            (Mode::Mv, _) => encode_mul_mv_id(mdev, &cmd, dt, &d)?,
            (Mode::MvVendored, _) => encode_mul_mv_id_vendored(mdev, &cmd, dt, &d)?,
            (Mode::Mm, MmScratch::Owned) => encode_mul_mm_id(mdev, &cmd, dt, &d, variant, tuning)?,
            (Mode::Mm, MmScratch::Shared(s)) => {
                // The producer laid the ids-map out at `s.n_expert *
                // MM_SCRATCH_ENTRY_BYTES` and sized `tpe`/`ids-map` for its
                // t/top_k; a projection with a different geometry would read the
                // wrong region. Validate before using the producer's n_expert for
                // the offset (guaranteed == stack.n_expert once this passes).
                let nr1 = mm_nr1(variant, t, top_k, stack.n_expert, tuning)?;
                if s.n_expert != stack.n_expert || s.t != t || s.top_k != top_k || s.nr1 != nr1 {
                    bail!(
                        "shared map0 scratch geometry (n_expert={}, t={}, top_k={}, nr1={}) does not match \
                         this projection (n_expert={}, t={}, top_k={}, nr1={}); the ids-map offset or the work list would be wrong",
                        s.n_expert,
                        s.t,
                        s.top_k,
                        s.nr1,
                        stack.n_expert,
                        t,
                        top_k,
                        nr1
                    );
                }
                encode_mm(
                    mdev,
                    &cmd,
                    dt,
                    &d,
                    variant,
                    nr1,
                    mm_full_grid(tuning),
                    &s.buffer,
                    0,
                    s.n_expert * MM_SCRATCH_ENTRY_BYTES,
                    mm_work_off(s.n_expert, s.t),
                )?
            }
        }
    }
    drop(x_guard);
    drop(ids_guard);

    Ok(output_tensor(dst, mdev, out_count, (t, top_k, stack.n_out)))
}

/// Whether the dual-weight gate|up gather kernel covers these two stacks: both
/// q4_K (the only dtype `kernel_mul_mv_id_q4_K_f32_dual` is written for — the
/// official Q4_K_M's routed experts), same geometry, both resident on Metal.
pub(crate) fn mv_id_dual_supported(gate: &ExpertStack, up: &ExpertStack) -> bool {
    gate.dtype == GgmlDType::Q4K
        && up.dtype == GgmlDType::Q4K
        && gate.n_expert == up.n_expert
        && gate.n_out == up.n_out
        && gate.k == up.k
        && gate.buffer.is_some()
        && up.buffer.is_some()
}

/// Gate and up expert matvecs plus their SwiGLU activation in ONE dispatch
/// (`kernel_mul_mv_id_q4_K_f32_dual`, mv.metal), replacing two
/// `kernel_mul_mv_id_q4_K_f32_v` launches and the `kernel_moe_silu_mul` pass
/// between them. `x` is `[t, x_per_row, k]` f32, `ids` `[t, top_k]` u32; returns
/// `[t, top_k, n_out]` f32 — the ACTIVATION, not the two projections.
/// Bit-identical to that trio (mv_id.rs `dual_matches_split_bitwise` proves it).
/// Callers must check `mv_id_dual_supported` first.
pub(crate) fn run_mv_id_dual(
    gate: &ExpertStack,
    up: &ExpertStack,
    x: &Tensor,
    ids: &Tensor,
) -> Result<Tensor> {
    let cdev = x.device().clone();
    let Device::Metal(mdev) = &cdev else {
        bail!("mul_mv_id_dual requires x on a Metal device");
    };
    if !mv_id_dual_supported(gate, up) {
        bail!(
            "no dual-weight gather kernel for gate {:?}[{},{},{}] / up {:?}[{},{},{}]",
            gate.dtype,
            gate.n_expert,
            gate.n_out,
            gate.k,
            up.dtype,
            up.n_expert,
            up.n_out,
            up.k
        );
    }

    let (t, x_per_row, kx) = x
        .dims3()
        .map_err(|e| anyhow::anyhow!("x must be rank-3 [t, x_per_row, k]: {e}"))?;
    let (t_ids, top_k) = ids
        .dims2()
        .map_err(|e| anyhow::anyhow!("ids must be rank-2 [t, top_k]: {e}"))?;
    if x.dtype() != DType::F32 {
        bail!("x must be f32, got {:?}", x.dtype());
    }
    if ids.dtype() != DType::U32 {
        bail!("ids must be u32, got {:?}", ids.dtype());
    }
    if !x.is_contiguous() {
        bail!("x must be contiguous");
    }
    if !ids.is_contiguous() {
        bail!("ids must be contiguous");
    }
    if kx != gate.k {
        bail!("x k ({kx}) does not match expert stack k ({})", gate.k);
    }
    if t_ids != t {
        bail!("ids t ({t_ids}) does not match x t ({t})");
    }
    if x_per_row != 1 && x_per_row != top_k {
        bail!("x_per_row ({x_per_row}) must be 1 (shared row) or top_k ({top_k}) (per-slot row)");
    }

    let dt = GgmlDType::Q4K;
    let block_size = dt.block_size();
    if !gate.k.is_multiple_of(block_size) {
        bail!(
            "expert stack k ({}) is not a multiple of {dt:?} block size {block_size}",
            gate.k
        );
    }
    let bytes_per_row = gate.k / block_size * dt.type_size();
    let per_expert = gate.n_out * bytes_per_row;
    let geom = mv_vendored_geom(dt)?;

    let (Some(gate_buf), Some(up_buf)) = (gate.buffer.as_deref(), up.buffer.as_deref()) else {
        bail!("expert stacks have no device buffer; the fused MoE requires Metal");
    };

    let out_count = t * top_k * gate.n_out;
    let dst = mdev.new_buffer(out_count, DType::F32, "mul_mv_id_dual")?;

    let (x_guard, x_layout) = x.storage_and_layout();
    let Storage::Metal(x_storage) = &*x_guard else {
        bail!("x is not on a Metal device");
    };
    let (ids_guard, ids_layout) = ids.storage_and_layout();
    let Storage::Metal(ids_storage) = &*ids_guard else {
        bail!("ids is not on a Metal device");
    };

    let args = MvIdArgs {
        nei0: top_k as i32,
        nei1: t as i32,
        nbi1: (top_k * DType::U32.size_in_bytes()) as u64,
        ne00: gate.k as i32,
        ne01: gate.n_out as i32,
        ne02: gate.n_expert as i32,
        nb00: 0,
        nb01: bytes_per_row as u64,
        nb02: per_expert as u64,
        ne10: gate.k as i32,
        ne11: x_per_row as i32,
        ne12: t as i32,
        ne13: 1,
        nb10: DType::F32.size_in_bytes() as u64,
        nb11: (gate.k * DType::F32.size_in_bytes()) as u64,
        nb12: (x_per_row * gate.k * DType::F32.size_in_bytes()) as u64,
        ne0: gate.n_out as i32,
        ne1: top_k as i32,
        nb1: (gate.n_out * DType::F32.size_in_bytes()) as u64,
        nr0: geom.nr0 as i32,
    };

    let pipeline = pipelines::mv_pipeline(mdev.device(), "kernel_mul_mv_id_q4_K_f32_dual")?;
    {
        let cmd = mdev.command_encoder()?;
        let ep = &cmd;
        let encoder = ep.encoder();
        let encoder: &ComputeCommandEncoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_bytes(0, &args);
        encoder.set_input_buffer(1, Some(gate_buf), gate.base_off);
        encoder.set_input_buffer(
            2,
            Some(x_storage.buffer()),
            x_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder.set_output_buffer(3, Some(&dst), 0);
        encoder.set_input_buffer(
            4,
            Some(ids_storage.buffer()),
            ids_layout.start_offset() * DType::U32.size_in_bytes(),
        );
        encoder.set_input_buffer(5, Some(up_buf), up.base_off);

        let grid = mtl_size(geom.row_blocks(gate.n_out), 1, top_k * t);
        let threads = mtl_size(32, geom.nsg, 1);
        encoder.dispatch_thread_groups(grid, threads);
    }
    drop(x_guard);
    drop(ids_guard);

    Ok(output_tensor(dst, mdev, out_count, (t, top_k, gate.n_out)))
}

/// Allocate the shared map0 scratch for one MoE block and encode the single map0
/// pass from `ids`. The returned buffer holds `tpe` (n_expert i32 @ 0) then the
/// `ids-map` (n_expert*t i32 @ n_expert*4); all three projections read it via
/// `run_mm_shared`. map0's output depends only on ids/t/top_k/n_expert, so one
/// pass serves gate/up/down despite their differing k / x_per_row. The caller
/// keeps the returned buffer alive until the down projection's mm is submitted;
/// candle's per-encoder fences order the mm reads after this write.
/// `variant` decides the pass-2 tile width the work list is built for (`mm_nr1`);
/// every consumer must dispatch the same variant.
pub(crate) fn prepare_mm_id_map0(
    n_expert: usize,
    ids: &Tensor,
    variant: MmVariant,
) -> Result<Map0Scratch> {
    let cdev = ids.device().clone();
    let Device::Metal(mdev) = &cdev else {
        bail!("prepare_mm_id_map0 requires ids on a Metal device");
    };
    let (t, top_k) = ids
        .dims2()
        .map_err(|e| anyhow::anyhow!("ids must be rank-2 [t, top_k]: {e}"))?;
    if ids.dtype() != DType::U32 {
        bail!("ids must be u32, got {:?}", ids.dtype());
    }
    if !ids.is_contiguous() {
        bail!("ids must be contiguous");
    }

    let nr1 = mm_nr1(variant, t, top_k, n_expert, MmTuning::default())?;
    let scratch = mdev.new_buffer(
        mm_scratch_elems(n_expert, t, top_k, nr1),
        DType::F32,
        "mm_id_map0",
    )?;

    let (ids_guard, ids_layout) = ids.storage_and_layout();
    let Storage::Metal(ids_storage) = &*ids_guard else {
        bail!("ids is not on a Metal device");
    };
    let ids_buf = ids_storage.buffer();
    let ids_off = ids_layout.start_offset() * DType::U32.size_in_bytes();

    let m = Map0Dispatch {
        ids: ids_buf,
        ids_off,
        n_expert,
        top_k,
        t,
        nr1,
    };
    {
        let cmd = mdev.command_encoder()?;
        encode_map0(
            mdev,
            &cmd,
            &m,
            &scratch,
            0,
            n_expert * MM_SCRATCH_ENTRY_BYTES,
            mm_work_off(n_expert, t),
        )?;
    }
    drop(ids_guard);
    Ok(Map0Scratch {
        buffer: scratch,
        n_expert,
        t,
        top_k,
        nr1,
    })
}

/// Plain (non-indexed) quantized mat-vec against the vendored ggml-geometry
/// kernel — the lm_head bypass at seq==1. `weight` is a rank-2 `[n_out, k]`
/// quantized tensor's raw device buffer; `x` is `[t, k]` f32 (t small, typically
/// 1). Returns `[t, n_out]` f32. Supports the vendored dtypes (`mv_vendored_supported`
/// — q4_K/q5_K/q6_K/q8_0): the current official lm_head is q8_0, the retired
/// original's was q6_K.
/// Callers gate on `mv_vendored_supported` and fall back to QMatMul otherwise.
pub(crate) fn run_plain_mv(
    weight: &Buffer,
    dt: GgmlDType,
    n_out: usize,
    k: usize,
    x: &Tensor,
) -> Result<Tensor> {
    let cdev = x.device().clone();
    let Device::Metal(mdev) = &cdev else {
        bail!("mul_mv requires x on a Metal device");
    };
    let (t, kx) = x
        .dims2()
        .map_err(|e| anyhow::anyhow!("x must be rank-2 [t, k]: {e}"))?;
    if x.dtype() != DType::F32 {
        bail!("x must be f32, got {:?}", x.dtype());
    }
    if !x.is_contiguous() {
        bail!("x must be contiguous");
    }
    if kx != k {
        bail!("x k ({kx}) does not match weight k ({k})");
    }
    if !mv_vendored_supported(dt) {
        bail!("no vendored plain mv kernel for dtype {dt:?}");
    }
    let geom = mv_vendored_geom(dt)?;

    let block_size = dt.block_size();
    if !k.is_multiple_of(block_size) {
        bail!("weight k ({k}) is not a multiple of {dt:?} block size {block_size}");
    }
    let bytes_per_row = k / block_size * dt.type_size();

    let out_count = t * n_out;
    let dst = mdev.new_buffer(out_count, DType::F32, "mul_mv")?;

    let (x_guard, x_layout) = x.storage_and_layout();
    let Storage::Metal(x_storage) = &*x_guard else {
        bail!("x is not on a Metal device");
    };
    let x_buf = x_storage.buffer();
    let x_off = x_layout.start_offset() * DType::F32.size_in_bytes();

    let args = MvArgs {
        ne00: k as i32,
        ne01: n_out as i32,
        ne02: 1,
        nb00: 0,
        nb01: bytes_per_row as u64,
        nb02: (n_out * bytes_per_row) as u64,
        nb03: (n_out * bytes_per_row) as u64,
        ne10: k as i32,
        ne11: t as i32,
        ne12: 1,
        nb10: DType::F32.size_in_bytes() as u64,
        nb11: (k * DType::F32.size_in_bytes()) as u64,
        nb12: (t * k * DType::F32.size_in_bytes()) as u64,
        nb13: (t * k * DType::F32.size_in_bytes()) as u64,
        ne0: n_out as i32,
        ne1: t as i32,
        nr0: geom.nr0 as i32,
        r2: 1,
        r3: 1,
    };

    let name = mv_vendored_plain_kernel_name(dt)?;
    let pipeline = pipelines::mv_pipeline(mdev.device(), name)?;
    {
        let cmd = mdev.command_encoder()?;
        let ep = &cmd;
        let encoder = ep.encoder();
        let encoder: &ComputeCommandEncoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_bytes(0, &args);
        encoder.set_input_buffer(1, Some(weight), 0);
        encoder.set_input_buffer(2, Some(x_buf), x_off);
        encoder.set_output_buffer(3, Some(&dst), 0);
        // The q8_0 kernel's cross-simdgroup reduce reads threadgroup memory; the
        // K-quant kernels declare none (smem_bytes == 0).
        let smem = geom.smem_bytes();
        if smem > 0 {
            encoder.set_threadgroup_memory_length(0, smem);
        }

        // grid.x per MvVendoredGeom (K-quant: ceil(n_out/(nr0*nsg)); q8_0:
        // ceil(n_out/nr0)); grid.y = one column per token row (nr1 == 1 for the
        // quant mv path); threads `nsg` simdgroups.
        let grid = mtl_size(geom.row_blocks(n_out), t, 1);
        let threads = mtl_size(32, geom.nsg, 1);
        encoder.dispatch_thread_groups(grid, threads);
    }
    drop(x_guard);

    Ok(output_tensor(dst, mdev, out_count, (t, n_out)))
}

/// Dense f16-weight x f32-activation matmul against the vendored ggml-geometry
/// kernels (f16.metal) — the attention projections. `weight` is a rank-2
/// `[n_out, k]` dense f16 tensor, `x` is `[t, k]` f32; returns `[t, n_out]` f32
/// with no activation cast and no output rounding (the stored f16 weights are
/// the only f16 in the chain). Dispatches per the fork's host split: the gemv
/// for t <= 8 tokens, the tiled gemm above (`F16_MM_MIN_SEQ`).
pub(crate) fn run_matmul_f16(weight: &Tensor, x: &Tensor) -> Result<Tensor> {
    let kernel = if crate::ops::attn_mm_classic() {
        F16MmKernel::Classic
    } else {
        F16MmKernel::Tensor
    };
    run_matmul_f16_variant(weight, x, kernel)
}

/// Dense bf16-weight x f32-activation matmul against the vendored bf16 twin
/// kernels (bf16.metal / bf16_t.metal) — the DFlash drafter's mmap-aliased
/// BF16 planes. Same contract and host split as `run_matmul_f16` (`weight`
/// rank-2 `[n_out, k]` BF16, `x` `[t, k]` f32, `[t, n_out]` f32 out; gemv at
/// t <= 8, tiled gemm above), same `XWEN_ATTN_MM_CLASSIC` kill-switch for
/// the gemm branch — the two families differ only in the weight element type
/// the kernels load.
pub(crate) fn run_matmul_bf16(weight: &Tensor, x: &Tensor) -> Result<Tensor> {
    let kernel = if crate::ops::attn_mm_classic() {
        F16MmKernel::Classic
    } else {
        F16MmKernel::Tensor
    };
    run_matmul_bf16_variant(weight, x, kernel)
}

/// Which prefill (ne11 > 8) mm-branch kernel `run_matmul_f16_variant`
/// dispatches. Production only ever selects the first two (`run_matmul_f16`);
/// `TensorMixed` is reachable exclusively from the f16.rs probe tests.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum F16MmKernel {
    /// Classic simdgroup kernel, float tiles (`f16.metal`) — the
    /// `XWEN_ATTN_MM_CLASSIC` kill-switch.
    Classic,
    /// Metal-4 cooperative-tensor kernel, half operand tiles (`f16_t.metal`) —
    /// shipped default.
    Tensor,
    /// Mixed-operand cooperative-tensor probe: half weight tile x FLOAT
    /// activation tile (`f16_t_mixed.metal`). No env switch, no production
    /// selection — constructed only by the f16.rs tests (hence dead in
    /// non-test builds).
    #[allow(dead_code)]
    TensorMixed,
}

/// `run_matmul_f16` with the prefill (ne11 > 8) mm-branch kernel chosen
/// explicitly. Production derives the kernel from the cached
/// `XWEN_ATTN_MM_CLASSIC` kill-switch (`run_matmul_f16`); the f16.rs tests
/// call this with an explicit kernel because the switch is a process-global
/// `OnceLock`. The decode gemv branch is identical for every kernel choice
/// (classic mv).
pub(crate) fn run_matmul_f16_variant(
    weight: &Tensor,
    x: &Tensor,
    kernel: F16MmKernel,
) -> Result<Tensor> {
    run_matmul_dense_variant(weight, x, kernel, DenseWeight::F16)
}

/// `run_matmul_bf16` with the mm-branch kernel chosen explicitly (the bf16.rs
/// tests, same `OnceLock` reason as `run_matmul_f16_variant`). The
/// `TensorMixed` probe exists only for f16 and is rejected here.
pub(crate) fn run_matmul_bf16_variant(
    weight: &Tensor,
    x: &Tensor,
    kernel: F16MmKernel,
) -> Result<Tensor> {
    run_matmul_dense_variant(weight, x, kernel, DenseWeight::Bf16)
}

/// Dense f32-weight x f32-activation mat-vec — `ops::matmul_f32`'s launcher.
/// GEMV ONLY: `run_matmul_dense_variant` hard-errors on the F32 family above
/// `F16_MM_MIN_SEQ` tokens (there is no `kernel_mul_mm_f32_f32_v`), because the
/// caller — the MoE router projection — routes larger batches back to candle.
pub(crate) fn run_matmul_f32(weight: &Tensor, x: &Tensor) -> Result<Tensor> {
    run_matmul_dense_variant(weight, x, F16MmKernel::Classic, DenseWeight::F32)
}

/// Shape half of `ops::matmul_f32`'s admission test: the token window the gemv
/// covers, plus the two vector-load divisibility rules
/// `run_matmul_dense_variant` enforces (`k % 32 == 0` for the float4 K walk with
/// no tail, `n_out % 4 == 0` shared with the f16 family). Dtype, device and
/// contiguity are the CALLER's to check — they are tensor properties, and the
/// MoE block asks them next to its own the way `MoeBlock::fused_shexp` does.
pub(crate) fn matmul_f32_shape_supported(t: usize, k: usize, n_out: usize) -> bool {
    t >= 1 && t <= F16_MM_MIN_SEQ && k.is_multiple_of(32) && n_out.is_multiple_of(4)
}

/// Whether `t`'s view starts at a 16-byte-aligned BYTE offset — the binding
/// rule `run_matmul_dense_variant` hard-errors on for BOTH of its operands,
/// because the kernels read weight and activation through 16-byte vector
/// device pointers (float4 in the f32 gemv).
///
/// A property of the TENSOR rather than of the geometry, so it sits outside
/// `matmul_f32_shape_supported` and is asked separately: `MoeBlock::router_mv`
/// asks it of the router plane and of the activation, so a view that lands off
/// the boundary keeps candle's matmul instead of erroring the forward. A
/// freshly allocated tensor starts at offset zero, and so does every plane the
/// GGUF loader hands over (32-byte tensor alignment); a view into a larger
/// buffer is the only thing that can land off the boundary.
pub(crate) fn view_offset_aligned_16(t: &Tensor) -> bool {
    let (_guard, layout) = t.storage_and_layout();
    (layout.start_offset() * t.dtype().size_in_bytes()).is_multiple_of(16)
}

/// The weight element type of the dense mixed-dtype matmul family: which dtype
/// the input tensor must carry and which kernel library the dispatch selects.
/// Both are 2-byte types consumed by structurally identical kernels (the bf16
/// sources are line-for-line twins of the f16 ones), so every argument,
/// stride, grid and smem computation below is family-independent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DenseWeight {
    F16,
    Bf16,
    /// The MoE router plane. GEMV-ONLY — there is no f32 tiled gemm in the
    /// family, so the mm branch bails; `ops::matmul_f32`'s contract says so and
    /// its caller keeps larger batches on candle.
    F32,
}

impl DenseWeight {
    fn dtype(self) -> DType {
        match self {
            DenseWeight::F16 => DType::F16,
            DenseWeight::Bf16 => DType::BF16,
            DenseWeight::F32 => DType::F32,
        }
    }

    /// The public-facing op name, for error messages and buffer labels.
    fn op_name(self) -> &'static str {
        match self {
            DenseWeight::F16 => "matmul_f16",
            DenseWeight::Bf16 => "matmul_bf16",
            DenseWeight::F32 => "matmul_f32",
        }
    }
}

/// Shared body of `run_matmul_f16_variant` / `run_matmul_bf16_variant` /
/// `run_matmul_f32`: the family picks the required weight dtype and the kernel
/// library; everything else (args, offsets, grids, smem) is identical across
/// the three, the gemv kernels being line-for-line twins. The F32 family has a
/// gemv only, so it hard-errors in the mm branch instead of selecting a
/// pipeline there.
fn run_matmul_dense_variant(
    weight: &Tensor,
    x: &Tensor,
    kernel: F16MmKernel,
    family: DenseWeight,
) -> Result<Tensor> {
    let name = family.op_name();
    let cdev = x.device().clone();
    let Device::Metal(mdev) = &cdev else {
        bail!("{name} requires x on a Metal device");
    };
    let (n_out, k) = weight
        .dims2()
        .map_err(|e| anyhow::anyhow!("weight must be rank-2 [n_out, k]: {e}"))?;
    let (t, kx) = x
        .dims2()
        .map_err(|e| anyhow::anyhow!("x must be rank-2 [t, k]: {e}"))?;
    if weight.dtype() != family.dtype() {
        bail!(
            "{name} weight must be {:?}, got {:?}",
            family.dtype(),
            weight.dtype()
        );
    }
    if x.dtype() != DType::F32 {
        bail!("x must be f32, got {:?}", x.dtype());
    }
    if !weight.is_contiguous() {
        bail!("weight must be contiguous");
    }
    if !x.is_contiguous() {
        bail!("x must be contiguous");
    }
    if kx != k {
        bail!("x k ({kx}) does not match weight k ({k})");
    }
    // Both kernels stream K through vector types (half4/float4 in the gemv,
    // half4x4/float2x4 tiles in the gemm) and skip the fork's bc_inp/K-tail
    // handling, and the gemm's float4 output copy needs 16-byte-aligned dst
    // rows. Every attention shape satisfies these (K multiple of 1024, out
    // dims all multiples of 4).
    if !k.is_multiple_of(32) {
        bail!("{name} requires k % 32 == 0, got {k}");
    }
    if !n_out.is_multiple_of(4) {
        bail!("{name} requires n_out % 4 == 0, got {n_out}");
    }

    let out_count = t * n_out;
    let dst = mdev.new_buffer(out_count, DType::F32, name)?;

    let (w_guard, w_layout) = weight.storage_and_layout();
    let Storage::Metal(w_storage) = &*w_guard else {
        bail!("weight is not on a Metal device");
    };
    let w_buf = w_storage.buffer();
    let w_off = w_layout.start_offset() * family.dtype().size_in_bytes();
    // The kernels read the weight through 4/16-element vector device pointers
    // (half4/half4x4 in the f16 family, bfloat4 in the bf16 one, float4 in the
    // f32 router gemv), which Metal requires 16-byte aligned; rows are
    // (k % 8 == 0 checked above), so only a misaligned view start could break
    // it. The mmap alias paths produce exactly such offset views (GGUF's
    // 32-byte tensor alignment satisfies this); a hand-sliced view lands here.
    if !w_off.is_multiple_of(16) {
        bail!("{name} requires a 16-byte-aligned weight view, got byte offset {w_off}");
    }

    let (x_guard, x_layout) = x.storage_and_layout();
    let Storage::Metal(x_storage) = &*x_guard else {
        bail!("x is not on a Metal device");
    };
    let x_buf = x_storage.buffer();
    let x_off = x_layout.start_offset() * DType::F32.size_in_bytes();
    // The kernels read x through float4/float2x4 device pointers, which Metal
    // requires 16-byte aligned; rows are (k % 4 == 0 checked above), so only a
    // misaligned view start could break it.
    if !x_off.is_multiple_of(16) {
        bail!("{name} requires a 16-byte-aligned x view, got byte offset {x_off}");
    }

    let nb01 = (k * family.dtype().size_in_bytes()) as u64;
    let nb11 = (k * DType::F32.size_in_bytes()) as u64;

    {
        let cmd = mdev.command_encoder()?;
        let ep = &cmd;
        let encoder = ep.encoder();
        let encoder: &ComputeCommandEncoder = encoder.as_ref();
        if t > F16_MM_MIN_SEQ {
            // Tiled gemm: 128 threads / 4 simdgroups. Default is the Metal-4
            // cooperative-tensor kernel (ggml's dense geometry: 64(out) x
            // 128(token) tiles, A-tile-only 4096 B smem, the f32 activation read
            // directly from device as a cooperative tensor — no staging); it uses
            // a 128-wide token tile, so grid.x covers ceil(t/128). The classic
            // simdgroup kernel (XWEN_ATTN_MM_CLASSIC, float tiles, 12288 B; the
            // store-back tile reuses the region) and the test-only TensorMixed
            // probe (half sa 4096 + float sb 4096 = 8192 B) keep 64(out) x
            // 32(token) tiles, so grid.x covers ceil(t/32). Same MmArgs and 128
            // threads for all — only the kernel, its library, the tile smem, and
            // the token-tile width differ.
            let args = MmArgs {
                ne00: k as i32,
                ne02: 1,
                nb01,
                nb02: n_out as u64 * nb01,
                nb03: n_out as u64 * nb01,
                ne12: 1,
                nb10: DType::F32.size_in_bytes() as u64,
                nb11,
                nb12: t as u64 * nb11,
                nb13: t as u64 * nb11,
                ne0: n_out as i32,
                ne1: t as i32,
                r2: 1,
                r3: 1,
            };
            // (pipeline, threadgroup smem bytes, token-tile width for grid.x).
            // The bf16 kernels are geometry-identical twins, so smem/tile
            // widths match per kernel kind; only the library differs.
            let (pipeline, smem, n_tile) = match (kernel, family) {
                (F16MmKernel::Classic, DenseWeight::F16) => (
                    pipelines::f16_pipeline(mdev.device(), "kernel_mul_mm_f16_f32_v")?,
                    12288,
                    32,
                ),
                (F16MmKernel::Classic, DenseWeight::Bf16) => (
                    pipelines::bf16_pipeline(mdev.device(), "kernel_mul_mm_bf16_f32_v")?,
                    12288,
                    32,
                ),
                (F16MmKernel::Tensor, DenseWeight::F16) => (
                    pipelines::f16_t_pipeline(mdev.device(), "kernel_mul_mm_f16_f32_t")?,
                    4096,
                    128,
                ),
                (F16MmKernel::Tensor, DenseWeight::Bf16) => (
                    pipelines::bf16_t_pipeline(mdev.device(), "kernel_mul_mm_bf16_f32_t")?,
                    4096,
                    128,
                ),
                (F16MmKernel::TensorMixed, DenseWeight::F16) => (
                    pipelines::f16_t_mixed_pipeline(
                        mdev.device(),
                        "kernel_mul_mm_f16_f32_t_mixed",
                    )?,
                    8192,
                    32,
                ),
                (F16MmKernel::TensorMixed, DenseWeight::Bf16) => {
                    bail!("the TensorMixed probe exists only for f16 weights")
                }
                // f32.metal carries the gemv alone: the MoE router is the only
                // f32 matmul plane and it stays on candle above this token
                // count, so no f32 tiled gemm was ever written.
                (_, DenseWeight::F32) => bail!(
                    "matmul_f32 is a gemv only: t ({t}) must be <= {F16_MM_MIN_SEQ}, \
                     larger batches belong on candle's matmul"
                ),
            };
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_bytes(0, &args);
            encoder.set_input_buffer(1, Some(w_buf), w_off);
            encoder.set_input_buffer(2, Some(x_buf), x_off);
            encoder.set_output_buffer(3, Some(&dst), 0);
            encoder.set_threadgroup_memory_length(0, smem);
            let grid = mtl_size(t.div_ceil(n_tile), n_out.div_ceil(64), 1);
            // threadsPerThreadgroup is (32, 4, 1) — ggml's (SIMD width, nsg, 1)
            // shape for every mul_mm dispatch. Thread linearization is identical
            // to a flat (128, 1, 1) (tiitg = x + 32*y; simdgroup packing from the
            // linear index is probe-verified on the M5 target, not an MSL-spec
            // universal), so outputs are bit-identical either way, but the 2D
            // shape measures a consistent few percent faster on the large
            // projection shapes (amortized bench, 9216x3072: ~1.79 -> ~1.71
            // ms/dispatch).
            encoder.dispatch_thread_groups(grid, mtl_size(32, 4, 1));
        } else {
            // gemv: NR0 rows per threadgroup, NSG simdgroups splitting K, one
            // grid.y column per token; smem is the cross-simdgroup reduce
            // scratch (NR0 * 32 floats).
            let args = MvArgs {
                ne00: k as i32,
                ne01: n_out as i32,
                ne02: 1,
                nb00: 0,
                nb01,
                nb02: n_out as u64 * nb01,
                nb03: n_out as u64 * nb01,
                ne10: k as i32,
                ne11: t as i32,
                ne12: 1,
                nb10: DType::F32.size_in_bytes() as u64,
                nb11,
                nb12: t as u64 * nb11,
                nb13: t as u64 * nb11,
                ne0: n_out as i32,
                ne1: t as i32,
                nr0: MV_F16_NR0 as i32,
                r2: 1,
                r3: 1,
            };
            let pipeline = match family {
                DenseWeight::F16 => {
                    pipelines::f16_pipeline(mdev.device(), "kernel_mul_mv_f16_f32_v")?
                }
                DenseWeight::Bf16 => {
                    pipelines::bf16_pipeline(mdev.device(), "kernel_mul_mv_bf16_f32_v")?
                }
                DenseWeight::F32 => {
                    pipelines::f32_pipeline(mdev.device(), "kernel_mul_mv_f32_f32_v")?
                }
            };
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_bytes(0, &args);
            encoder.set_input_buffer(1, Some(w_buf), w_off);
            encoder.set_input_buffer(2, Some(x_buf), x_off);
            encoder.set_output_buffer(3, Some(&dst), 0);
            encoder.set_threadgroup_memory_length(0, MV_F16_NR0 * 32 * DType::F32.size_in_bytes());
            let grid = mtl_size(n_out.div_ceil(MV_F16_NR0), t, 1);
            encoder.dispatch_thread_groups(grid, mtl_size(32, MV_F16_NSG, 1));
        }
    }
    drop(w_guard);
    drop(x_guard);

    Ok(output_tensor(dst, mdev, out_count, (t, n_out)))
}

/// q8_0-weight x f32-activation mat-vec against the vendored q8.metal kernel —
/// the attention DECODE gemv (seq <= 8) of a q8_0-quantized checkpoint. `weight`
/// is the rank-2 `[n_out, k]` q8_0 tensor's raw device buffer, bound at `w_off`
/// (the mmap alias's `base_off`, 0 for the classic private copy); `x` is `[t, k]`
/// f32 (t small — decode/short-verify spans). Returns `[t, n_out]` f32 with no
/// activation cast and no output rounding (the stored q8_0 weights are the only
/// quantized values in the chain). Mirrors `run_matmul_f16`'s gemv branch: the
/// q8_0 kernel's `N_SG` simdgroups split the K reduction over `N_R0` rows and
/// combine through `N_R0*32` floats of shmem.
pub(crate) fn run_matmul_q8(
    weight: &Buffer,
    w_off: usize,
    n_out: usize,
    k: usize,
    x: &Tensor,
) -> Result<Tensor> {
    let cdev = x.device().clone();
    let Device::Metal(mdev) = &cdev else {
        bail!("matmul_q8 requires x on a Metal device");
    };
    let (t, kx) = x
        .dims2()
        .map_err(|e| anyhow::anyhow!("x must be rank-2 [t, k]: {e}"))?;
    if x.dtype() != DType::F32 {
        bail!("x must be f32, got {:?}", x.dtype());
    }
    if !x.is_contiguous() {
        bail!("x must be contiguous");
    }
    if kx != k {
        bail!("x k ({kx}) does not match weight k ({k})");
    }
    // The kernel walks whole q8_0 (32-element) blocks with no K tail; every
    // attention K (the hidden dim) is a multiple of 32.
    let block = GgmlDType::Q8_0.block_size();
    if !k.is_multiple_of(block) {
        bail!("matmul_q8 requires k % {block} == 0, got {k}");
    }

    let bytes_per_row = k / block * GgmlDType::Q8_0.type_size();

    let out_count = t * n_out;
    let dst = mdev.new_buffer(out_count, DType::F32, "matmul_q8")?;

    let (x_guard, x_layout) = x.storage_and_layout();
    let Storage::Metal(x_storage) = &*x_guard else {
        bail!("x is not on a Metal device");
    };
    let x_buf = x_storage.buffer();
    let x_off = x_layout.start_offset() * DType::F32.size_in_bytes();

    let nb11 = (k * DType::F32.size_in_bytes()) as u64;
    let args = MvArgs {
        ne00: k as i32,
        ne01: n_out as i32,
        ne02: 1,
        nb00: 0,
        nb01: bytes_per_row as u64,
        nb02: (n_out * bytes_per_row) as u64,
        nb03: (n_out * bytes_per_row) as u64,
        ne10: k as i32,
        ne11: t as i32,
        ne12: 1,
        nb10: DType::F32.size_in_bytes() as u64,
        nb11,
        nb12: t as u64 * nb11,
        nb13: t as u64 * nb11,
        ne0: n_out as i32,
        ne1: t as i32,
        nr0: MV_Q8_NR0 as i32,
        r2: 1,
        r3: 1,
    };

    let pipeline = pipelines::q8_pipeline(mdev.device(), "kernel_mul_mv_q8_0_f32_attn")?;
    {
        let cmd = mdev.command_encoder()?;
        let ep = &cmd;
        let encoder = ep.encoder();
        let encoder: &ComputeCommandEncoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_bytes(0, &args);
        encoder.set_input_buffer(1, Some(weight), w_off);
        encoder.set_input_buffer(2, Some(x_buf), x_off);
        encoder.set_output_buffer(3, Some(&dst), 0);
        encoder.set_threadgroup_memory_length(0, MV_Q8_NR0 * 32 * DType::F32.size_in_bytes());

        // grid.x = ceil(n_out / N_R0); grid.y = one column per token; threads are
        // N_SG simdgroups of 32 splitting the K reduction.
        let grid = mtl_size(n_out.div_ceil(MV_Q8_NR0), t, 1);
        encoder.dispatch_thread_groups(grid, mtl_size(32, MV_Q8_NSG, 1));
    }
    drop(x_guard);

    Ok(output_tensor(dst, mdev, out_count, (t, n_out)))
}

/// The r1ptg (src1 rows per threadgroup) ggml's host picks for a `t`-token
/// batch, or `None` outside ITS window (2..=8). Not the routing decision —
/// `ops::mv_ext_window` owns that and is what callers ask; this is the vendored
/// table it consults inside ggml's range.
///
/// The mapping is ggml's own (ggml-metal-ops.cpp:2176-2190) — it divides the
/// batch as evenly as the four widths allow (6 = 2x3 rather than 4+2, 8 = 2x4
/// rather than 5+3), so no threadgroup runs mostly-masked rows.
pub(crate) fn mv_ext_r1ptg(t: usize) -> Option<usize> {
    let r = match t {
        2 => 2,
        3 | 6 => 3,
        4 | 7 | 8 => 4,
        5 => 5,
        _ => return None,
    };
    Some(r)
}

/// The vendored small-batch `kernel_mul_mv_ext_<dtype>_f32_r1_<r1ptg>`
/// (src/ops/mv_ext.metal). Instantiated for the three dtypes the shipped files
/// present to a small-batch matmul; a weight in any other dtype keeps whatever
/// path it had. Keep in lockstep with the entry-point list in mv_ext.metal
/// (`mv_ext::tests::instantiation_matrix_matches_metal` cross-checks it against
/// the source).
pub(crate) fn mv_ext_kernel_name(dt: GgmlDType, r1ptg: usize) -> Result<&'static str> {
    let n = match (dt, r1ptg) {
        (GgmlDType::Q4K, 2) => "kernel_mul_mv_ext_q4_K_f32_r1_2",
        (GgmlDType::Q4K, 3) => "kernel_mul_mv_ext_q4_K_f32_r1_3",
        (GgmlDType::Q4K, 4) => "kernel_mul_mv_ext_q4_K_f32_r1_4",
        (GgmlDType::Q4K, 5) => "kernel_mul_mv_ext_q4_K_f32_r1_5",
        (GgmlDType::Q6K, 2) => "kernel_mul_mv_ext_q6_K_f32_r1_2",
        (GgmlDType::Q6K, 3) => "kernel_mul_mv_ext_q6_K_f32_r1_3",
        (GgmlDType::Q6K, 4) => "kernel_mul_mv_ext_q6_K_f32_r1_4",
        (GgmlDType::Q6K, 5) => "kernel_mul_mv_ext_q6_K_f32_r1_5",
        (GgmlDType::Q8_0, 2) => "kernel_mul_mv_ext_q8_0_f32_r1_2",
        (GgmlDType::Q8_0, 3) => "kernel_mul_mv_ext_q8_0_f32_r1_3",
        (GgmlDType::Q8_0, 4) => "kernel_mul_mv_ext_q8_0_f32_r1_4",
        (GgmlDType::Q8_0, 5) => "kernel_mul_mv_ext_q8_0_f32_r1_5",
        (other, r) => bail!("no vendored mul_mv_ext kernel for dtype {other:?} at r1ptg {r}"),
    };
    Ok(n)
}

/// Whether a `[n_out, k]` quantized weight of this dtype can take the
/// small-batch mat-vec: the kernel must be instantiated for the dtype, and `k`
/// must be a whole multiple of BOTH the baked 128-element pass width and the
/// dtype's super-block (the kernel has no K tail). Every production shape
/// satisfies both; anything else keeps the path it had.
pub(crate) fn mv_ext_supported(dt: GgmlDType, k: usize) -> bool {
    mv_ext_kernel_name(dt, 2).is_ok()
        && k.is_multiple_of(MV_EXT_K_MULTIPLE)
        && k.is_multiple_of(dt.block_size())
}

/// Quantized-weight x f32-activation matmul against the vendored small-batch
/// mat-vec (mv_ext.metal). `weight` is the rank-2 `[n_out, k]` quantized
/// tensor's raw device buffer bound at `w_off`; `x` is `[t, k]` f32. Returns
/// `[t, n_out]` f32.
///
/// `r1ptg` — the src1 rows a threadgroup covers — is the CALLER'S, not
/// re-derived here: `ops::mv_ext_window` is the single authority on the plan, so
/// a token count it admits can never be one this function rejects. Any `t` works
/// for a valid `r1ptg`; the grid covers `ceil(t/r1ptg)` threadgroups and the
/// kernel masks the ragged remainder.
///
/// Every operand stays f32: the weight chunk is dequantized to f32 registers,
/// the activation is read as f32, and the accumulation is f32. What differs
/// from the gemv and from candle's gemm is the summation ORDER of the K
/// reduction (8 lanes per row here, each holding a different chunk interleave),
/// so results are bounded-close to both rather than bit-identical to either —
/// see the mv_ext.metal header.
pub(crate) fn run_matmul_mv_ext(
    weight: &Buffer,
    w_off: usize,
    dtype: GgmlDType,
    n_out: usize,
    k: usize,
    x: &Tensor,
    r1ptg: usize,
) -> Result<Tensor> {
    let cdev = x.device().clone();
    let Device::Metal(mdev) = &cdev else {
        bail!("matmul_mv_ext requires x on a Metal device");
    };
    let (t, kx) = x
        .dims2()
        .map_err(|e| anyhow::anyhow!("x must be rank-2 [t, k]: {e}"))?;
    if x.dtype() != DType::F32 {
        bail!("x must be f32, got {:?}", x.dtype());
    }
    if !x.is_contiguous() {
        bail!("x must be contiguous");
    }
    if kx != k {
        bail!("x k ({kx}) does not match weight k ({k})");
    }
    if t == 0 {
        bail!("matmul_mv_ext needs at least one token");
    }
    // Rejects any r1ptg with no instantiated kernel, which is the only way this
    // can be wrong now that the plan arrives from the caller.
    let name = mv_ext_kernel_name(dtype, r1ptg)?;
    if !mv_ext_supported(dtype, k) {
        bail!(
            "matmul_mv_ext requires k a multiple of {MV_EXT_K_MULTIPLE} and of the {dtype:?} \
             block size {}, got {k}",
            dtype.block_size()
        );
    }

    let bytes_per_row = k / dtype.block_size() * dtype.type_size();
    // The kernel reads the weight through `device const block_q *`, whose
    // alignment is that of its widest member (the `half` scale, 2 bytes). Rows
    // are whole blocks, so only the bound base offset could break it — GGUF's
    // 32-byte tensor alignment satisfies this; a hand-sliced view lands here.
    if !w_off.is_multiple_of(2) || !bytes_per_row.is_multiple_of(2) {
        bail!(
            "matmul_mv_ext requires a 2-byte-aligned weight view and row stride, \
             got offset {w_off} and stride {bytes_per_row}"
        );
    }

    let (x_guard, x_layout) = x.storage_and_layout();
    let Storage::Metal(x_storage) = &*x_guard else {
        bail!("x is not on a Metal device");
    };
    let x_buf = x_storage.buffer();
    let x_off = x_layout.start_offset() * DType::F32.size_in_bytes();
    // The activation is read as `float4` / `float4x4`, both of which Metal
    // requires 16-byte aligned. Row starts are fine by construction (the row
    // stride is a multiple of 128 floats), so only an offset view could break
    // it; such a caller falls back rather than reading misaligned.
    if !x_off.is_multiple_of(16) {
        bail!("matmul_mv_ext requires a 16-byte-aligned x view, got offset {x_off}");
    }

    let out_count = t * n_out;
    let dst = mdev.new_buffer(out_count, DType::F32, "matmul_mv_ext")?;

    let args = MvExtArgs {
        ne00: k as i32,
        ne01: n_out as i32,
        nb01: bytes_per_row as u64,
        ne11: t as i32,
        nb11: (k * DType::F32.size_in_bytes()) as u64,
        ne0: n_out as i32,
    };

    let pipeline = pipelines::mv_ext_pipeline(mdev.device(), name)?;
    {
        let cmd = mdev.command_encoder()?;
        let ep = &cmd;
        let encoder = ep.encoder();
        let encoder: &ComputeCommandEncoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_bytes(0, &args);
        encoder.set_input_buffer(1, Some(weight), w_off);
        encoder.set_input_buffer(2, Some(x_buf), x_off);
        encoder.set_output_buffer(3, Some(&dst), 0);
        // No threadgroup memory: the reduction is a shuffle ladder inside each
        // simdgroup. grid.x covers the weight rows R0PTG at a time, grid.y the
        // token rows r1ptg at a time; threads are ggml's (SIMD width, nsg, 1).
        let grid = mtl_size(n_out.div_ceil(MV_EXT_R0PTG), t.div_ceil(r1ptg), 1);
        encoder.dispatch_thread_groups(grid, mtl_size(32, MV_EXT_NSG, 1));
    }
    drop(x_guard);

    Ok(output_tensor(dst, mdev, out_count, (t, n_out)))
}

/// The vendored dense cooperative-tensor `kernel_mul_mm_<dtype>_f32_t`
/// (src/ops/dense_mm.metal) is instantiated for the same dtype set as the MoE
/// gather; a checkpoint storing its FFN weights in any other dtype stays on the
/// `QMatMul` path. Keep in lockstep with the `template [[host_name(...)]]` lines
/// in dense_mm.metal (`dense_mm::tests::instantiation_matrix_matches_metal`
/// cross-checks this against the source).
pub(crate) fn dense_mm_kernel_name(dt: GgmlDType) -> Result<&'static str> {
    let n = match dt {
        GgmlDType::Q8_0 => "kernel_mul_mm_q8_0_f32_t",
        GgmlDType::Q4K => "kernel_mul_mm_q4_K_f32_t",
        GgmlDType::Q5K => "kernel_mul_mm_q5_K_f32_t",
        GgmlDType::Q6K => "kernel_mul_mm_q6_K_f32_t",
        other => bail!("no vendored dense kernel_mul_mm kernel for dtype {other:?}"),
    };
    Ok(n)
}

/// Whether a `[n_out, k]` quantized weight of this dtype can take the dense
/// cooperative-tensor prefill gemm: the kernel must be instantiated, and `k`
/// must be a whole multiple of BOTH the NK = 32 step and the dtype's super-block
/// (so an A-tile chunk never straddles a super-block and ggml's K-tail zero-pad
/// stays unreachable). Every production FFN shape satisfies both; a checkpoint
/// that does not falls back to `QMatMul`.
pub(crate) fn dense_mm_supported(dt: GgmlDType, k: usize) -> bool {
    dense_mm_kernel_name(dt).is_ok() && k.is_multiple_of(32) && k.is_multiple_of(dt.block_size())
}

/// Dense quantized-weight x f32-activation matmul against the vendored Metal-4
/// cooperative-tensor gemm (dense_mm.metal) — the 27B's SwiGLU FFN projections
/// at prefill. `weight` is the rank-2 `[n_out, k]` quantized tensor's raw device
/// buffer bound at `w_off` (0 for the shared private allocation the companion
/// `QLinear` also uses); `x` is `[t, k]` f32. Returns `[t, n_out]` f32.
///
/// Weights are dequantized to half in the A tile and the f32 activation is read
/// straight from device memory, with the reduced-precision matmul2d descriptor —
/// the same precision class as the attention prefill gemm (`run_matmul_f16`'s
/// tensor branch), not f32 accumulation-order noise. Caller-gated on token count
/// (`ops::dense_mm_min_seq`); there is no gemv branch here, because decode keeps
/// the `QMatMul` path.
pub(crate) fn run_matmul_dense_q_mm(
    weight: &Buffer,
    w_off: usize,
    dtype: GgmlDType,
    n_out: usize,
    k: usize,
    x: &Tensor,
) -> Result<Tensor> {
    let cdev = x.device().clone();
    let Device::Metal(mdev) = &cdev else {
        bail!("matmul_dense_q requires x on a Metal device");
    };
    let (t, kx) = x
        .dims2()
        .map_err(|e| anyhow::anyhow!("x must be rank-2 [t, k]: {e}"))?;
    if x.dtype() != DType::F32 {
        bail!("x must be f32, got {:?}", x.dtype());
    }
    if !x.is_contiguous() {
        bail!("x must be contiguous");
    }
    if kx != k {
        bail!("x k ({kx}) does not match weight k ({k})");
    }
    let name = dense_mm_kernel_name(dtype)?;
    if !dense_mm_supported(dtype, k) {
        bail!(
            "matmul_dense_q requires k a multiple of 32 and of the {dtype:?} block size {}, got {k}",
            dtype.block_size()
        );
    }

    let bytes_per_row = k / dtype.block_size() * dtype.type_size();
    // The kernel reads the weight through `device const block_q *`, whose
    // alignment is that of its widest member (the `half` scale, 2 bytes). Rows
    // are whole blocks, so only the bound base offset could break it — GGUF's
    // 32-byte tensor alignment satisfies this; a hand-sliced view lands here.
    if !w_off.is_multiple_of(2) || !bytes_per_row.is_multiple_of(2) {
        bail!(
            "matmul_dense_q requires a 2-byte-aligned weight view and row stride, \
             got offset {w_off} and stride {bytes_per_row}"
        );
    }

    let (x_guard, x_layout) = x.storage_and_layout();
    let Storage::Metal(x_storage) = &*x_guard else {
        bail!("x is not on a Metal device");
    };
    let x_buf = x_storage.buffer();
    let x_off = x_layout.start_offset() * DType::F32.size_in_bytes();

    let out_count = t * n_out;
    let dst = mdev.new_buffer(out_count, DType::F32, "matmul_dense_q")?;

    let nb11 = (k * DType::F32.size_in_bytes()) as u64;
    let args = MmArgs {
        ne00: k as i32,
        ne02: 1,
        nb01: bytes_per_row as u64,
        nb02: (n_out * bytes_per_row) as u64,
        nb03: (n_out * bytes_per_row) as u64,
        ne12: 1,
        nb10: DType::F32.size_in_bytes() as u64,
        nb11,
        nb12: t as u64 * nb11,
        nb13: t as u64 * nb11,
        ne0: n_out as i32,
        ne1: t as i32,
        r2: 1,
        r3: 1,
    };

    let pipeline = pipelines::dense_mm_pipeline(mdev.device(), name)?;
    {
        let cmd = mdev.command_encoder()?;
        let ep = &cmd;
        let encoder = ep.encoder();
        let encoder: &ComputeCommandEncoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_bytes(0, &args);
        encoder.set_input_buffer(1, Some(weight), w_off);
        encoder.set_input_buffer(2, Some(x_buf), x_off);
        encoder.set_output_buffer(3, Some(&dst), 0);
        // The A tile only: NR0(64) x NK(32) half. There is no B or C tile — the
        // activation is a device-memory cooperative tensor and the result stores
        // straight to device.
        encoder.set_threadgroup_memory_length(0, DENSE_MM_A_TILE_SMEM);
        // 64(out) x 128(token) tiles; (32, 4, 1) threads is ggml's (SIMD width,
        // nsg, 1) shape for every mul_mm dispatch.
        let grid = mtl_size(t.div_ceil(DENSE_MM_NR1), n_out.div_ceil(DENSE_MM_NR0), 1);
        encoder.dispatch_thread_groups(grid, mtl_size(32, 4, 1));
    }
    drop(x_guard);

    Ok(output_tensor(dst, mdev, out_count, (t, n_out)))
}

/// Fused MoE weighted combine against the vendored `combine.metal` kernels —
/// the routed-expert combine tail of `FusedExperts::forward`. Reads `down`
/// (`[seq, top_k, n_out]` f32 contiguous) once and returns `[seq, n_out]` f32:
///   - `col_l2` = `None`  (rescale-free): `dst[s,c] = Σ_k down[s,k,c] * w[s,k]`
///   - `col_l2` = `Some`  (`[seq, top_k, 1]` f32): the per-column L2 rescale is
///     undone in the same pass — `dst[s,c] = Σ_k down[s,k,c]*col_l2[s,k]*2^-15*w[s,k]`.
/// `weights` is `[seq, top_k]` f32. The launch geometry and per-op rounding
/// mirror candle's strided `sum(1)` exactly, so the result is bit-identical to
/// the candle broadcast/affine/sum chain (see combine.metal / the combine.rs test).
pub(crate) fn run_combine(
    down: &Tensor,
    col_l2: Option<&Tensor>,
    weights: &Tensor,
) -> Result<Tensor> {
    let cdev = down.device().clone();
    let Device::Metal(mdev) = &cdev else {
        bail!("combine requires down on a Metal device");
    };

    let (seq, top_k, n_out) = down
        .dims3()
        .map_err(|e| anyhow::anyhow!("down must be rank-3 [seq, top_k, n_out]: {e}"))?;
    if down.dtype() != DType::F32 {
        bail!("down must be f32, got {:?}", down.dtype());
    }
    if !down.is_contiguous() {
        bail!("down must be contiguous");
    }
    if weights
        .dims2()
        .map_err(|e| anyhow::anyhow!("weights must be rank-2 [seq, top_k]: {e}"))?
        != (seq, top_k)
    {
        bail!(
            "weights shape {:?} must be [seq, top_k] = [{seq}, {top_k}]",
            weights.dims()
        );
    }
    if weights.dtype() != DType::F32 {
        bail!("weights must be f32, got {:?}", weights.dtype());
    }
    if !weights.is_contiguous() {
        bail!("weights must be contiguous");
    }
    if let Some(l2) = col_l2 {
        if l2
            .dims3()
            .map_err(|e| anyhow::anyhow!("col_l2 must be rank-3 [seq, top_k, 1]: {e}"))?
            != (seq, top_k, 1)
        {
            bail!(
                "col_l2 shape {:?} must be [seq, top_k, 1] = [{seq}, {top_k}, 1]",
                l2.dims()
            );
        }
        if l2.dtype() != DType::F32 {
            bail!("col_l2 must be f32, got {:?}", l2.dtype());
        }
        if !l2.is_contiguous() {
            bail!("col_l2 must be contiguous");
        }
    }

    // The reduction is a single simd_sum over one 32-lane simdgroup, so the
    // candle-matching threadgroup width must not exceed 32 (see combine.metal);
    // a wider width would leave lanes 32.. in a second simdgroup whose partials
    // are never folded in. This is an error, not a fallback — production top_k is
    // 10 (width 8); a top_k needing width > 32 (i.e. >= 66) is out of contract.
    let width_hint = combine_reduction_width(top_k);
    if width_hint > 32 {
        bail!(
            "combine top_k={top_k} needs threadgroup width {width_hint} > 32; the single-simdgroup \
             simd_sum reduction would silently drop lanes 32.."
        );
    }
    // The kernels address `down` with i32 index math; a grid whose flat element
    // count exceeds i32::MAX would wrap to a negative offset.
    if !combine_index_fits_i32(seq, top_k, n_out) {
        bail!(
            "combine index math overflows i32: seq={seq} top_k={top_k} n_out={n_out} \
             (seq*top_k*n_out = {} exceeds i32::MAX)",
            (seq as i64) * (top_k as i64) * (n_out as i64)
        );
    }

    let name = if col_l2.is_some() {
        "kernel_moe_combine_rescale"
    } else {
        "kernel_moe_combine"
    };
    let pipeline = pipelines::combine_pipeline(mdev.device(), name)?;

    let out_length = seq * n_out;
    let dst = mdev.new_buffer(out_length, DType::F32, "combine")?;

    // Resolve operand buffers. `storage_and_layout` guards must outlive the encode.
    let (down_guard, down_layout) = down.storage_and_layout();
    let Storage::Metal(down_storage) = &*down_guard else {
        bail!("down is not on a Metal device");
    };
    let down_buf = down_storage.buffer();
    let down_off = down_layout.start_offset() * DType::F32.size_in_bytes();

    let (w_guard, w_layout) = weights.storage_and_layout();
    let Storage::Metal(w_storage) = &*w_guard else {
        bail!("weights is not on a Metal device");
    };
    let w_buf = w_storage.buffer();
    let w_off = w_layout.start_offset() * DType::F32.size_in_bytes();

    // The optional col_l2 guard is bound for the whole encode when present.
    let l2_resolved = match col_l2 {
        Some(l2) => {
            let (guard, layout) = l2.storage_and_layout();
            let off = layout.start_offset() * DType::F32.size_in_bytes();
            Some((guard, off))
        }
        None => None,
    };

    let args = CombineArgs {
        top_k: top_k as i32,
        n_out: n_out as i32,
    };
    // candle's `fast_sum_f32_strided` launch: out_length threadgroups, block_dim
    // = min(pipeline max, next_pow2(top_k/2)); reproduced so the simd_sum lane
    // partition (and thus the reduction order) is identical. The width guard
    // above pins `combine_reduction_width(top_k)` <= 32, so this stays within one
    // simdgroup.
    let width = std::cmp::min(
        pipeline.max_total_threads_per_threadgroup(),
        combine_reduction_width(top_k),
    );
    {
        let cmd = mdev.command_encoder()?;
        let ep = &cmd;
        let encoder = ep.encoder();
        let encoder: &ComputeCommandEncoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_bytes(0, &args);
        encoder.set_input_buffer(1, Some(down_buf), down_off);
        if let Some((l2_guard, l2_off)) = &l2_resolved {
            let Storage::Metal(l2_storage) = &**l2_guard else {
                bail!("col_l2 is not on a Metal device");
            };
            encoder.set_input_buffer(2, Some(l2_storage.buffer()), *l2_off);
            encoder.set_input_buffer(3, Some(w_buf), w_off);
            encoder.set_output_buffer(4, Some(&dst), 0);
        } else {
            encoder.set_input_buffer(2, Some(w_buf), w_off);
            encoder.set_output_buffer(3, Some(&dst), 0);
        }
        encoder.dispatch_thread_groups(mtl_size(out_length, 1, 1), mtl_size(width, 1, 1));
    }
    drop(down_guard);
    drop(w_guard);
    drop(l2_resolved);

    Ok(output_tensor(dst, mdev, out_length, (seq, n_out)))
}

/// Fused SwiGLU activation against the `kernel_moe_silu_mul` kernel
/// (silu_mul.metal): `act = silu(gate) * up`, one pass over the two operands.
/// `gate` and `up` are same-shape, same-length f32 contiguous tensors (the MoE
/// up/gate expert-matvec outputs, `[seq, top_k, expert_ff]`); the result has the
/// same shape and dtype. Bit-identical to the candle `silu(gate) * up` chain it
/// replaces (silu_mul.rs `fused_matches_candle_bitwise` proves it), so the fused
/// path is safe under every parity tier. Metal only; the caller's kill-switch is
/// the candle chain (`XWEN_ACT_CLASSIC`).
pub(crate) fn run_silu_mul(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    let cdev = gate.device().clone();
    let Device::Metal(mdev) = &cdev else {
        bail!("silu_mul requires gate on a Metal device");
    };

    if gate.dtype() != DType::F32 {
        bail!("gate must be f32, got {:?}", gate.dtype());
    }
    if up.dtype() != DType::F32 {
        bail!("up must be f32, got {:?}", up.dtype());
    }
    if gate.dims() != up.dims() {
        bail!(
            "gate shape {:?} must equal up shape {:?}",
            gate.dims(),
            up.dims()
        );
    }
    if !gate.is_contiguous() {
        bail!("gate must be contiguous");
    }
    if !up.is_contiguous() {
        bail!("up must be contiguous");
    }
    if !gate.device().same_device(up.device()) {
        bail!("gate and up must live on the same Metal device");
    }
    let shape = gate.shape().clone();
    let n = checked_elems(shape.dims(), "silu_mul")?;
    glue_index_fits_i32(n)?;

    let pipeline = pipelines::silu_mul_pipeline(mdev.device(), "kernel_moe_silu_mul")?;
    let dst = mdev.new_buffer(n, DType::F32, "silu_mul")?;

    let (gate_guard, gate_layout) = gate.storage_and_layout();
    let Storage::Metal(gate_storage) = &*gate_guard else {
        bail!("gate is not on a Metal device");
    };
    let (up_guard, up_layout) = up.storage_and_layout();
    let Storage::Metal(up_storage) = &*up_guard else {
        bail!("up is not on a Metal device");
    };

    let args = SiluMulArgs { n: n as i32 };
    {
        let cmd = mdev.command_encoder()?;
        let ep = &cmd;
        let encoder = ep.encoder();
        let encoder: &ComputeCommandEncoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_bytes(0, &args);
        encoder.set_input_buffer(
            1,
            Some(gate_storage.buffer()),
            gate_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder.set_input_buffer(
            2,
            Some(up_storage.buffer()),
            up_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder.set_output_buffer(3, Some(&dst), 0);
        dispatch_linear(encoder, &pipeline, n);
    }
    drop(gate_guard);
    drop(up_guard);

    Ok(output_tensor(dst, mdev, n, shape))
}

/// Fused SwiGLU activation PLUS the f16-tile L2 rescale against
/// `kernel_moe_silu_mul_l2` (silu_mul.metal): from the `[seq, top_k, expert_ff]`
/// f32 `gate`/`up` pair, returns `(act_s, col_l2)` — `act_s` the same shape,
/// `col_l2` `[seq, top_k, 1]` — as `FusedExperts::project_inner`'s candle chain
/// defines them: `act = silu(gate) * up`, `col_l2 = clamp(sqrt(Σ act²), clamp_min,
/// clamp_max)`, `act_s = (act * scale) / col_l2`. One threadgroup per row;
/// bails (never falls back) for `expert_ff > SILU_MUL_L2_MAX_FF`, so the caller
/// must ask `silu_mul_l2_supported` first. Bounded, not bitwise, against the
/// chain (the sum's order differs — see the kernel header).
pub(crate) fn run_silu_mul_l2(
    gate: &Tensor,
    up: &Tensor,
    scale: f32,
    clamp_min: f32,
    clamp_max: f32,
) -> Result<(Tensor, Tensor)> {
    let cdev = gate.device().clone();
    let Device::Metal(mdev) = &cdev else {
        bail!("silu_mul_l2 requires gate on a Metal device");
    };
    if gate.dtype() != DType::F32 {
        bail!("gate must be f32, got {:?}", gate.dtype());
    }
    if up.dtype() != DType::F32 {
        bail!("up must be f32, got {:?}", up.dtype());
    }
    if gate.dims() != up.dims() {
        bail!(
            "gate shape {:?} must equal up shape {:?}",
            gate.dims(),
            up.dims()
        );
    }
    if !gate.is_contiguous() || !up.is_contiguous() {
        bail!("gate and up must be contiguous");
    }
    if !gate.device().same_device(up.device()) {
        bail!("gate and up must live on the same Metal device");
    }
    let (seq, top_k, ff) = gate
        .dims3()
        .map_err(|e| anyhow::anyhow!("gate must be rank-3 [seq, top_k, expert_ff]: {e}"))?;
    if ff == 0 || ff > SILU_MUL_L2_MAX_FF {
        bail!("silu_mul_l2 supports 1..={SILU_MUL_L2_MAX_FF} columns per row, got {ff}");
    }
    let n_rows = checked_elems(&[seq, top_k], "silu_mul_l2 rows")?;
    if n_rows == 0 {
        bail!("silu_mul_l2 requires a non-empty activation, got [{seq}, {top_k}, {ff}]");
    }
    let n = checked_elems(&[n_rows, ff], "silu_mul_l2")?;
    glue_index_fits_i32(n)?;

    let pipeline = pipelines::silu_mul_pipeline(mdev.device(), "kernel_moe_silu_mul_l2")?;
    let act_s = mdev.new_buffer(n, DType::F32, "silu_mul_l2 act")?;
    let col_l2 = mdev.new_buffer(n_rows, DType::F32, "silu_mul_l2 col_l2")?;

    let (gate_guard, gate_layout) = gate.storage_and_layout();
    let Storage::Metal(gate_storage) = &*gate_guard else {
        bail!("gate is not on a Metal device");
    };
    let (up_guard, up_layout) = up.storage_and_layout();
    let Storage::Metal(up_storage) = &*up_guard else {
        bail!("up is not on a Metal device");
    };

    let args = SiluMulL2Args {
        ff: ff as i32,
        n_rows: n_rows as i32,
        scale,
        clamp_min,
        clamp_max,
    };
    {
        let cmd = mdev.command_encoder()?;
        let ep = &cmd;
        let encoder = ep.encoder();
        let encoder: &ComputeCommandEncoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_bytes(0, &args);
        encoder.set_input_buffer(
            1,
            Some(gate_storage.buffer()),
            gate_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder.set_input_buffer(
            2,
            Some(up_storage.buffer()),
            up_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder.set_output_buffer(3, Some(&act_s), 0);
        encoder.set_output_buffer(4, Some(&col_l2), 0);
        encoder.dispatch_thread_groups(mtl_size(n_rows, 1, 1), mtl_size(SILU_MUL_L2_THREADS, 1, 1));
    }
    drop(gate_guard);
    drop(up_guard);

    Ok((
        output_tensor(act_s, mdev, n, (seq, top_k, ff)),
        output_tensor(col_l2, mdev, n_rows, (seq, top_k, 1)),
    ))
}

/// Matches the Metal `moe_router_args` struct (src/ops/moe_glue.metal).
/// `#[repr(C)]` pins the layout byte-for-byte.
#[repr(C)]
#[derive(Clone, Copy)]
struct MoeRouterArgs {
    n_expert: i32,
    n_expert_pad: i32,
    top_k: i32,
    softmax_width: i32,
    sum_width: i32,
    sum_floor: f32,
}

/// Matches the Metal `moe_epilogue_args` struct (src/ops/moe_glue.metal).
/// `#[repr(C)]` pins the layout byte-for-byte.
#[repr(C)]
#[derive(Clone, Copy)]
struct MoeEpilogueArgs {
    top_k: i32,
    n_out: i32,
}

/// Threadgroup-array bounds baked into `moe_glue.metal`'s router kernel. The
/// dispatch refuses any geometry that would overrun one of them; the
/// `moe_glue.rs` geometry test parses the `#define`s and holds these equal.
pub(crate) const MOE_ROUTER_MAX_EXPERTS: usize = 512;
pub(crate) const MOE_ROUTER_MAX_SOFTMAX: usize = 256;
pub(crate) const MOE_ROUTER_MAX_TOP_K: usize = 32;

/// candle's threadgroup width for a `work_per_threadgroup`-element reduction:
/// `min(pipeline_max, next_pow2(work/2))` (call_reduce_contiguous /
/// call_last_softmax, which share the formula). The router kernel reproduces
/// both the softmax and the top-k sum with it, so the lane partitions — and thus
/// the reduction orders — match candle's bit-for-bit.
///
/// The `pipeline_max` term is candle's OWN pipeline limit, which is not
/// reachable from here; every width this is used for is far below any plausible
/// limit (n_expert 256 gives 128, top_k 8 gives 4), and the bitwise tests would
/// fail loudly if a device ever clamped candle lower.
fn candle_reduction_width(work: usize) -> usize {
    (work / 2).next_power_of_two()
}

/// Fused MoE routing decision against `kernel_moe_router` (moe_glue.metal):
/// softmax over the full expert set, descending bitonic arg-sort, top-k gather,
/// floor-clamped sum, renormalize — one threadgroup per token, replacing seven
/// candle dispatches. `logits` is `[seq, n_expert]` f32 contiguous (the router
/// matmul output, which stays a candle dispatch). Returns
/// `(ids [seq, top_k] u32, weights [seq, top_k] f32)`. Bit-identical to the
/// candle chain it replaces (moe_glue.rs `router_matches_candle_bitwise` proves
/// it); the caller's kill-switch is that chain (`XWEN_MOE_GLUE_CLASSIC`).
pub(crate) fn run_moe_router(
    logits: &Tensor,
    top_k: usize,
    sum_floor: f32,
) -> Result<(Tensor, Tensor)> {
    let cdev = logits.device().clone();
    let Device::Metal(mdev) = &cdev else {
        bail!("moe_router requires logits on a Metal device");
    };

    let (seq, n_expert) = logits
        .dims2()
        .map_err(|e| anyhow::anyhow!("logits must be rank-2 [seq, n_expert]: {e}"))?;
    if logits.dtype() != DType::F32 {
        bail!("logits must be f32, got {:?}", logits.dtype());
    }
    if !logits.is_contiguous() {
        bail!("logits must be contiguous");
    }
    if seq == 0 || n_expert == 0 {
        bail!("moe_router needs at least one token and one expert");
    }
    if top_k == 0 || top_k > n_expert {
        bail!("moe_router top_k={top_k} must be in 1..={n_expert}");
    }
    if top_k > MOE_ROUTER_MAX_TOP_K {
        bail!(
            "moe_router top_k={top_k} exceeds the kernel's {MOE_ROUTER_MAX_TOP_K}-wide selection \
             buffer (its sum folds one simdgroup)"
        );
    }
    let n_expert_pad = n_expert.next_power_of_two();
    if n_expert_pad > MOE_ROUTER_MAX_EXPERTS {
        bail!(
            "moe_router n_expert={n_expert} pads to {n_expert_pad}, over the kernel's \
             {MOE_ROUTER_MAX_EXPERTS}-wide threadgroup arrays"
        );
    }
    let softmax_width = candle_reduction_width(n_expert);
    if softmax_width > MOE_ROUTER_MAX_SOFTMAX {
        bail!(
            "moe_router softmax width {softmax_width} for n_expert={n_expert} exceeds the \
             kernel's {MOE_ROUTER_MAX_SOFTMAX}-pair reduction array"
        );
    }
    glue_index_fits_i32(checked_elems(&[seq, n_expert], "moe_router")?)?;

    let pipeline = pipelines::moe_glue_pipeline(mdev.device(), "kernel_moe_router")?;
    // The bitonic network needs exactly one thread per padded column, and the
    // softmax phase reuses the low `softmax_width` of them (next_pow2(n) is
    // always >= next_pow2(n/2), so the network is the wider of the two).
    if n_expert_pad > pipeline.max_total_threads_per_threadgroup() {
        bail!(
            "moe_router needs {n_expert_pad} threads per threadgroup, over this device's \
             pipeline limit {}",
            pipeline.max_total_threads_per_threadgroup()
        );
    }

    let n_sel = checked_elems(&[seq, top_k], "moe_router selection")?;
    let ids = mdev.new_buffer(n_sel, DType::U32, "moe_router_ids")?;
    let weights = mdev.new_buffer(n_sel, DType::F32, "moe_router_weights")?;

    let (logits_guard, logits_layout) = logits.storage_and_layout();
    let Storage::Metal(logits_storage) = &*logits_guard else {
        bail!("logits is not on a Metal device");
    };

    let args = MoeRouterArgs {
        n_expert: n_expert as i32,
        n_expert_pad: n_expert_pad as i32,
        top_k: top_k as i32,
        softmax_width: softmax_width as i32,
        sum_width: candle_reduction_width(top_k) as i32,
        sum_floor,
    };
    {
        let cmd = mdev.command_encoder()?;
        let ep = &cmd;
        let encoder = ep.encoder();
        let encoder: &ComputeCommandEncoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_bytes(0, &args);
        encoder.set_input_buffer(1, Some(logits_storage.buffer()), f32_off(logits_layout));
        encoder.set_output_buffer(2, Some(&ids), 0);
        encoder.set_output_buffer(3, Some(&weights), 0);
        encoder.dispatch_thread_groups(mtl_size(seq, 1, 1), mtl_size(n_expert_pad, 1, 1));
    }
    drop(logits_guard);

    let ids = Tensor::from_storage(
        Storage::Metal(MetalStorage::new(ids, mdev.clone(), n_sel, DType::U32)),
        (seq, top_k),
        candle_core::op::BackpropOp::none(),
        false,
    );
    Ok((ids, output_tensor(weights, mdev, n_sel, (seq, top_k))))
}

/// Fused MoE block epilogue against `kernel_moe_epilogue` (moe_glue.metal):
/// `dst[s,c] = Σ_k down[s,k,c] * w[s,k] + shexp[s,c] * sigmoid(gate[s])`, one
/// pass over `down`, replacing the weighted combine, the shared-expert gate
/// sigmoid, its broadcast multiply and the routed+shared add. `down` is
/// `[seq, top_k, n_out]` f32 contiguous, `w` `[seq, top_k]` f32, `shexp`
/// `[seq, n_out]` f32, `gate` `[seq, 1]` f32 (the RAW pre-sigmoid gate logit).
/// Bit-identical to the candle chain it replaces (moe_glue.rs
/// `epilogue_matches_candle_bitwise` proves it); the caller's kill-switch is
/// that chain (`XWEN_MOE_GLUE_CLASSIC`).
pub(crate) fn run_moe_epilogue(
    down: &Tensor,
    weights: &Tensor,
    shexp: &Tensor,
    gate: &Tensor,
) -> Result<Tensor> {
    let cdev = down.device().clone();
    let Device::Metal(mdev) = &cdev else {
        bail!("moe_epilogue requires down on a Metal device");
    };

    let (seq, top_k, n_out) = down
        .dims3()
        .map_err(|e| anyhow::anyhow!("down must be rank-3 [seq, top_k, n_out]: {e}"))?;
    check_f32(down, &[seq, top_k, n_out], "down")?;
    check_f32(weights, &[seq, top_k], "weights")?;
    check_f32(shexp, &[seq, n_out], "shexp")?;
    check_f32(gate, &[seq, 1], "gate")?;
    for (name, t) in [("weights", weights), ("shexp", shexp), ("gate", gate)] {
        if !down.device().same_device(t.device()) {
            bail!("{name} must live on the same Metal device as down");
        }
    }
    if seq == 0 || top_k == 0 || n_out == 0 {
        bail!("moe_epilogue needs a non-empty down projection");
    }

    // Same single-simdgroup reduction contract as the combine kernels: candle's
    // width for this top_k must fold inside one 32-lane simdgroup, or lanes 32..
    // would be silently dropped. An error, not a fallback.
    let width_hint = combine_reduction_width(top_k);
    if width_hint > 32 {
        bail!(
            "moe_epilogue top_k={top_k} needs threadgroup width {width_hint} > 32; the \
             single-simdgroup simd_sum reduction would silently drop lanes 32.."
        );
    }
    if !combine_index_fits_i32(seq, top_k, n_out) {
        bail!(
            "moe_epilogue index math overflows i32: seq={seq} top_k={top_k} n_out={n_out} \
             (seq*top_k*n_out = {} exceeds i32::MAX)",
            (seq as i64) * (top_k as i64) * (n_out as i64)
        );
    }

    let pipeline = pipelines::moe_glue_pipeline(mdev.device(), "kernel_moe_epilogue")?;
    let out_length = seq * n_out;
    let dst = mdev.new_buffer(out_length, DType::F32, "moe_epilogue")?;

    let (down_guard, down_layout) = down.storage_and_layout();
    let Storage::Metal(down_storage) = &*down_guard else {
        bail!("down is not on a Metal device");
    };
    let (w_guard, w_layout) = weights.storage_and_layout();
    let Storage::Metal(w_storage) = &*w_guard else {
        bail!("weights is not on a Metal device");
    };
    let (sh_guard, sh_layout) = shexp.storage_and_layout();
    let Storage::Metal(sh_storage) = &*sh_guard else {
        bail!("shexp is not on a Metal device");
    };
    let (g_guard, g_layout) = gate.storage_and_layout();
    let Storage::Metal(g_storage) = &*g_guard else {
        bail!("gate is not on a Metal device");
    };

    let args = MoeEpilogueArgs {
        top_k: top_k as i32,
        n_out: n_out as i32,
    };
    let width = std::cmp::min(
        pipeline.max_total_threads_per_threadgroup(),
        combine_reduction_width(top_k),
    );
    {
        let cmd = mdev.command_encoder()?;
        let ep = &cmd;
        let encoder = ep.encoder();
        let encoder: &ComputeCommandEncoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_bytes(0, &args);
        encoder.set_input_buffer(1, Some(down_storage.buffer()), f32_off(down_layout));
        encoder.set_input_buffer(2, Some(w_storage.buffer()), f32_off(w_layout));
        encoder.set_input_buffer(3, Some(sh_storage.buffer()), f32_off(sh_layout));
        encoder.set_input_buffer(4, Some(g_storage.buffer()), f32_off(g_layout));
        encoder.set_output_buffer(5, Some(&dst), 0);
        encoder.dispatch_thread_groups(mtl_size(out_length, 1, 1), mtl_size(width, 1, 1));
    }
    drop(down_guard);
    drop(w_guard);
    drop(sh_guard);
    drop(g_guard);

    Ok(output_tensor(dst, mdev, out_length, (seq, n_out)))
}

/// Threads per threadgroup in `kernel_moe_shexp_gate_up`: four simdgroups. It
/// does NOT have to divide the row's q8_0 block count — the leftover threads
/// carry a `+0.0` accumulator through the reduction — so the only width bound is
/// the register one below.
pub(crate) const MOE_SHEXP_THREADS: usize = 128;

/// q8_0 blocks of the activation row ONE thread of `kernel_moe_shexp_gate_up`
/// stages in registers. It bounds the kernel's register footprint (32 floats per
/// block), so a `hidden` wider than
/// `MOE_SHEXP_THREADS * MOE_SHEXP_MAX_BLK_PER_THREAD` blocks is refused rather
/// than run at whatever occupancy it would spill to.
pub(crate) const MOE_SHEXP_MAX_BLK_PER_THREAD: usize = 2;

/// Bottleneck rows one threadgroup of `kernel_moe_shexp_gate_up` computes,
/// sharing one staged pass over the token's activation. The per-row accumulator
/// PAIRS are registers, which is what bounds it.
pub(crate) const MOE_SHEXP_ROWS_PER_TG: usize = 4;

/// q8_0 blocks of one `ffn_down_shexp` row a single lane of
/// `kernel_moe_epilogue_shexp` folds. Its reduction is one simdgroup, so this
/// bounds `inner` at `32 * MOE_SHEXP_MAX_BLK_PER_LANE`.
pub(crate) const MOE_SHEXP_MAX_BLK_PER_LANE: usize = 4;

/// Threads per threadgroup in `kernel_moe_epilogue_shexp`: exactly one
/// simdgroup, keeping it inside `kernel_moe_epilogue`'s single-`simd_sum`
/// contract.
pub(crate) const MOE_SHEXP_EPILOGUE_THREADS: usize = 32;

/// Whether `kernel_moe_shexp_gate_up` covers this half of the geometry: q8_0
/// planes, and a `hidden` that is a whole number of q8_0 blocks no wider than
/// the staged register array. `inner` only has to be positive here — the row
/// tiling handles a ragged last tile — but the epilogue half bounds it, and
/// callers ask [`moe_shexp_fused_supported`], which is the conjunction.
fn moe_shexp_gate_up_supported(hidden: usize, inner: usize, dtype: GgmlDType) -> bool {
    if dtype != GgmlDType::Q8_0 {
        return false;
    }
    let block = GgmlDType::Q8_0.block_size();
    if hidden == 0 || !hidden.is_multiple_of(block) {
        return false;
    }
    inner > 0 && hidden / block <= MOE_SHEXP_THREADS * MOE_SHEXP_MAX_BLK_PER_THREAD
}

/// Whether `kernel_moe_epilogue_shexp` covers this half of the geometry: a q8_0
/// down plane, an `inner` that is a whole number of q8_0 blocks its single
/// simdgroup can fold, and a `top_k` that fits that same simdgroup.
fn moe_shexp_epilogue_supported(inner: usize, top_k: usize, dtype: GgmlDType) -> bool {
    if dtype != GgmlDType::Q8_0 {
        return false;
    }
    let block = GgmlDType::Q8_0.block_size();
    if inner == 0 || !inner.is_multiple_of(block) {
        return false;
    }
    if inner / block > MOE_SHEXP_EPILOGUE_THREADS * MOE_SHEXP_MAX_BLK_PER_LANE {
        return false;
    }
    top_k > 0 && top_k <= MOE_SHEXP_EPILOGUE_THREADS
}

/// Whether the fused shared-expert decode pair covers this block's geometry and
/// weight dtypes. The bounds are the kernels', so a block outside them keeps the
/// five-dispatch chain rather than failing; both launchers ask this too and
/// `bail!` when it says no.
pub(crate) fn moe_shexp_fused_supported(
    hidden: usize,
    inner: usize,
    top_k: usize,
    gate_dtype: GgmlDType,
    up_dtype: GgmlDType,
    down_dtype: GgmlDType,
) -> bool {
    gate_dtype == up_dtype
        && up_dtype == down_dtype
        && moe_shexp_gate_up_supported(hidden, inner, gate_dtype)
        && moe_shexp_epilogue_supported(inner, top_k, down_dtype)
}

/// Matches the Metal `moe_shexp_gate_up_args` struct (src/ops/moe_glue.metal).
#[repr(C)]
#[derive(Clone, Copy)]
struct MoeShexpGateUpArgs {
    hidden: i32,
    inner: i32,
    nblk: i32,
    n_row_tg: i32,
}

/// Matches the Metal `moe_epilogue_shexp_args` struct (src/ops/moe_glue.metal).
#[repr(C)]
#[derive(Clone, Copy)]
struct MoeEpilogueShexpArgs {
    top_k: i32,
    n_out: i32,
    inner: i32,
    nblk_inner: i32,
}

/// The first half of the fused shared expert against
/// `kernel_moe_shexp_gate_up` (moe_glue.metal): the gate and up q8_0
/// projections, their SwiGLU activation and the scalar gate logit, in ONE
/// dispatch where the classic chain spends four.
///
/// `x` is the block's normed input `[n, hidden]` f32, `gate` and `up` the
/// `[inner, hidden]` q8_0 projections' raw bytes, and `gate_inp` the
/// `ffn_gate_inp_shexp` row as the `[hidden, 1]` f32 tensor `SharedExpert`
/// already holds pre-transposed. Returns `(h [n, inner], logit [n, 1])` — the
/// ungated bottleneck and the RAW pre-sigmoid gate logit, which is what
/// [`run_moe_epilogue_shexp`] takes (the sigmoid stays in the epilogue, where it
/// was before this path existed).
///
/// Geometry outside [`moe_shexp_fused_supported`] is an error here: the caller
/// asks that predicate first and keeps the classic chain when it says no.
pub(crate) fn run_moe_shexp_gate_up(
    x: &Tensor,
    gate: &QuantPlane,
    up: &QuantPlane,
    gate_inp: &Tensor,
    hidden: usize,
    inner: usize,
) -> Result<(Tensor, Tensor)> {
    let cdev = x.device().clone();
    let Device::Metal(mdev) = &cdev else {
        bail!("moe_shexp_gate_up requires the activation on a Metal device");
    };
    if !moe_shexp_gate_up_supported(hidden, inner, gate.dtype) || up.dtype != gate.dtype {
        bail!(
            "moe_shexp_gate_up does not cover hidden {hidden}, inner {inner}, {:?}/{:?} weights",
            gate.dtype,
            up.dtype
        );
    }
    let (n, row) = x
        .dims2()
        .map_err(|e| anyhow::anyhow!("the activation must be rank-2 [n, hidden]: {e}"))?;
    if n == 0 {
        bail!("moe_shexp_gate_up needs at least one token");
    }
    if row != hidden {
        bail!("the activation is {row} wide, expected hidden = {hidden}");
    }
    for (what, plane) in [("gate", gate), ("up", up)] {
        if plane.out_dim != inner || plane.in_dim != hidden {
            bail!(
                "the shared expert's {what} projection is [{}, {}], expected [{inner}, {hidden}]",
                plane.out_dim,
                plane.in_dim
            );
        }
        // The kernel indexes the weight through `device const moe_block_q8_0 *`,
        // whose alignment is that of its `half` scale. Rows are whole blocks, so
        // only the bound base offset could break it.
        if !plane.base_off.is_multiple_of(2) {
            bail!(
                "moe_shexp_gate_up needs a 2-byte-aligned {what} view, got offset {}",
                plane.base_off
            );
        }
        check_plane_fits(plane, &format!("shared expert {what} projection"))?;
    }
    check_f32(x, &[n, hidden], "shared expert activation")?;
    check_f32(gate_inp, &[hidden, 1], "shared expert gate row")?;
    if !x.device().same_device(gate_inp.device()) {
        bail!("the shared expert gate row must live on the same Metal device as the activation");
    }
    let n_h = checked_elems(&[n, inner], "moe_shexp bottleneck")?;
    glue_index_fits_i32(checked_elems(&[n, hidden], "moe_shexp activation")?)?;
    glue_index_fits_i32(n_h)?;

    let pipeline = pipelines::moe_glue_pipeline(mdev.device(), "kernel_moe_shexp_gate_up")?;
    if pipeline.max_total_threads_per_threadgroup() < MOE_SHEXP_THREADS {
        bail!(
            "kernel_moe_shexp_gate_up needs {MOE_SHEXP_THREADS} threads per threadgroup, the \
             pipeline allows {}",
            pipeline.max_total_threads_per_threadgroup()
        );
    }
    check_delta_simd_width(&pipeline, "kernel_moe_shexp_gate_up")?;

    let h = mdev.new_buffer(n_h, DType::F32, "moe_shexp_h")?;
    let logit = mdev.new_buffer(n, DType::F32, "moe_shexp_logit")?;

    let (x_guard, x_layout) = x.storage_and_layout();
    let Storage::Metal(x_storage) = &*x_guard else {
        bail!("the shared expert activation is not on a Metal device");
    };
    let (r_guard, r_layout) = gate_inp.storage_and_layout();
    let Storage::Metal(r_storage) = &*r_guard else {
        bail!("the shared expert gate row is not on a Metal device");
    };

    let n_row_tg = inner.div_ceil(MOE_SHEXP_ROWS_PER_TG);
    let args = MoeShexpGateUpArgs {
        hidden: hidden as i32,
        inner: inner as i32,
        nblk: (hidden / GgmlDType::Q8_0.block_size()) as i32,
        n_row_tg: n_row_tg as i32,
    };
    {
        let cmd = mdev.command_encoder()?;
        let ep = &cmd;
        let encoder = ep.encoder();
        let encoder: &ComputeCommandEncoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_bytes(0, &args);
        encoder.set_input_buffer(1, Some(x_storage.buffer()), f32_off(x_layout));
        encoder.set_input_buffer(2, Some(&gate.buffer), gate.base_off);
        encoder.set_input_buffer(3, Some(&up.buffer), up.base_off);
        encoder.set_input_buffer(4, Some(r_storage.buffer()), f32_off(r_layout));
        encoder.set_output_buffer(5, Some(&h), 0);
        encoder.set_output_buffer(6, Some(&logit), 0);
        // One threadgroup per (row tile of the two projections, token), plus one
        // for the gate logit.
        encoder.dispatch_thread_groups(
            mtl_size(n_row_tg + 1, n, 1),
            mtl_size(MOE_SHEXP_THREADS, 1, 1),
        );
    }
    drop(x_guard);
    drop(r_guard);

    let h = output_tensor(h, mdev, n_h, (n, inner));
    let logit = output_tensor(logit, mdev, n, (n, 1));
    Ok((h, logit))
}

/// The second half of the fused shared expert against
/// `kernel_moe_epilogue_shexp` (moe_glue.metal): `run_moe_epilogue`'s block tail
/// with the shared expert's q8_0 DOWN projection folded into the same pass, so
/// the classic chain's separate down gemv disappears.
///
/// `down` is the uncombined routed projection `[seq, top_k, n_out]` f32, `w` the
/// routing weights `[seq, top_k]`, `h` [`run_moe_shexp_gate_up`]'s bottleneck
/// `[seq, inner]`, `down_shexp` the `[n_out, inner]` q8_0 projection's raw bytes
/// and `gate` the RAW `[seq, 1]` gate logit. Returns `[seq, n_out]`.
///
/// BOUNDED, not bitwise, against `run_moe_epilogue` — the routed combine folds
/// over 32 lanes here where that kernel folds over `next_pow2(top_k/2)` — which
/// is why this is a second kernel and `kernel_moe_epilogue` is left alone as the
/// strict tier's anchor.
pub(crate) fn run_moe_epilogue_shexp(
    down: &Tensor,
    weights: &Tensor,
    h: &Tensor,
    down_shexp: &QuantPlane,
    gate: &Tensor,
    inner: usize,
) -> Result<Tensor> {
    let cdev = down.device().clone();
    let Device::Metal(mdev) = &cdev else {
        bail!("moe_epilogue_shexp requires down on a Metal device");
    };

    let (seq, top_k, n_out) = down
        .dims3()
        .map_err(|e| anyhow::anyhow!("down must be rank-3 [seq, top_k, n_out]: {e}"))?;
    if seq == 0 || top_k == 0 || n_out == 0 {
        bail!("moe_epilogue_shexp needs a non-empty down projection");
    }
    if !moe_shexp_epilogue_supported(inner, top_k, down_shexp.dtype) {
        bail!(
            "moe_epilogue_shexp does not cover inner {inner}, top_k {top_k}, {:?} weights",
            down_shexp.dtype
        );
    }
    if down_shexp.out_dim != n_out || down_shexp.in_dim != inner {
        bail!(
            "the shared expert's down projection is [{}, {}], expected [{n_out}, {inner}]",
            down_shexp.out_dim,
            down_shexp.in_dim
        );
    }
    if !down_shexp.base_off.is_multiple_of(2) {
        bail!(
            "moe_epilogue_shexp needs a 2-byte-aligned down view, got offset {}",
            down_shexp.base_off
        );
    }
    check_plane_fits(down_shexp, "shared expert down projection")?;
    check_f32(down, &[seq, top_k, n_out], "down")?;
    check_f32(weights, &[seq, top_k], "weights")?;
    check_f32(h, &[seq, inner], "shared expert bottleneck")?;
    check_f32(gate, &[seq, 1], "gate")?;
    for (name, t) in [("weights", weights), ("bottleneck", h), ("gate", gate)] {
        if !down.device().same_device(t.device()) {
            bail!("{name} must live on the same Metal device as down");
        }
    }
    if !combine_index_fits_i32(seq, top_k, n_out) {
        bail!(
            "moe_epilogue_shexp index math overflows i32: seq={seq} top_k={top_k} n_out={n_out} \
             (seq*top_k*n_out = {} exceeds i32::MAX)",
            (seq as i64) * (top_k as i64) * (n_out as i64)
        );
    }
    glue_index_fits_i32(checked_elems(&[seq, inner], "moe_shexp bottleneck")?)?;

    let pipeline = pipelines::moe_glue_pipeline(mdev.device(), "kernel_moe_epilogue_shexp")?;
    if pipeline.max_total_threads_per_threadgroup() < MOE_SHEXP_EPILOGUE_THREADS {
        bail!(
            "kernel_moe_epilogue_shexp needs {MOE_SHEXP_EPILOGUE_THREADS} threads per \
             threadgroup, the pipeline allows {}",
            pipeline.max_total_threads_per_threadgroup()
        );
    }
    check_delta_simd_width(&pipeline, "kernel_moe_epilogue_shexp")?;

    let out_length = checked_elems(&[seq, n_out], "moe_epilogue_shexp output")?;
    let dst = mdev.new_buffer(out_length, DType::F32, "moe_epilogue_shexp")?;

    let (down_guard, down_layout) = down.storage_and_layout();
    let Storage::Metal(down_storage) = &*down_guard else {
        bail!("down is not on a Metal device");
    };
    let (w_guard, w_layout) = weights.storage_and_layout();
    let Storage::Metal(w_storage) = &*w_guard else {
        bail!("weights is not on a Metal device");
    };
    let (h_guard, h_layout) = h.storage_and_layout();
    let Storage::Metal(h_storage) = &*h_guard else {
        bail!("the shared expert bottleneck is not on a Metal device");
    };
    let (g_guard, g_layout) = gate.storage_and_layout();
    let Storage::Metal(g_storage) = &*g_guard else {
        bail!("gate is not on a Metal device");
    };

    let args = MoeEpilogueShexpArgs {
        top_k: top_k as i32,
        n_out: n_out as i32,
        inner: inner as i32,
        nblk_inner: (inner / GgmlDType::Q8_0.block_size()) as i32,
    };
    {
        let cmd = mdev.command_encoder()?;
        let ep = &cmd;
        let encoder = ep.encoder();
        let encoder: &ComputeCommandEncoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_bytes(0, &args);
        encoder.set_input_buffer(1, Some(down_storage.buffer()), f32_off(down_layout));
        encoder.set_input_buffer(2, Some(w_storage.buffer()), f32_off(w_layout));
        encoder.set_input_buffer(3, Some(h_storage.buffer()), f32_off(h_layout));
        encoder.set_input_buffer(4, Some(&down_shexp.buffer), down_shexp.base_off);
        encoder.set_input_buffer(5, Some(g_storage.buffer()), f32_off(g_layout));
        encoder.set_output_buffer(6, Some(&dst), 0);
        // kernel_moe_epilogue's grid, unchanged: one threadgroup per output
        // element, now a full simdgroup wide.
        encoder.dispatch_thread_groups(
            mtl_size(out_length, 1, 1),
            mtl_size(MOE_SHEXP_EPILOGUE_THREADS, 1, 1),
        );
    }
    drop(down_guard);
    drop(w_guard);
    drop(h_guard);
    drop(g_guard);

    Ok(output_tensor(dst, mdev, out_length, (seq, n_out)))
}

/// Matches the Metal `attn_gate_args` struct (src/ops/attn_glue.metal).
/// `#[repr(C)]` pins the layout byte-for-byte.
#[repr(C)]
#[derive(Clone, Copy)]
struct AttnGateArgs {
    n_head: i32,
    seq: i32,
    head_dim: i32,
}

/// Matches the Metal `permute_args` struct (src/ops/attn_glue.metal).
#[repr(C)]
#[derive(Clone, Copy)]
struct PermuteArgs {
    d0: i32,
    d1: i32,
    d2: i32,
}

/// Matches the Metal `rope_args` struct (src/ops/rope.metal).
#[repr(C)]
#[derive(Clone, Copy)]
struct RopeArgs {
    heads: i32,
    seq: i32,
    head_dim: i32,
    n_rot: i32,
    pos: i32,
}

/// Matches the Metal `flash_attn_params` struct (src/ops/flash.metal).
/// `#[repr(C)]` pins the layout byte-for-byte: twelve 4-byte fields (48 bytes,
/// a multiple of 8) followed by eight `i64` element strides — no implicit
/// padding on either side.
#[repr(C)]
#[derive(Clone, Copy)]
struct FlashAttnArgs {
    gqa_factor: i32,
    scale: f32,
    nk: i32,
    nq_aligned: i32,
    nk_aligned: i32,
    ql_rem: i32,
    kl_rem: i32,
    kl: i32,
    q_off: i32,
    k_off: i32,
    window: i32,
    disable_skip: i32,
    q_stride_h: i64,
    q_stride_r: i64,
    k_stride_h: i64,
    k_stride_r: i64,
    v_stride_h: i64,
    v_stride_r: i64,
    o_stride_h: i64,
    o_stride_r: i64,
}

/// Linear one-thread-per-element launch shared by the attention-glue kernels:
/// `n` threads in threadgroups of up to 256 (bounds-checked in the kernels, so
/// the rounded-up tail is harmless).
fn dispatch_linear(
    encoder: &ComputeCommandEncoder,
    pipeline: &candle_metal_kernels::metal::ComputePipeline,
    n: usize,
) {
    let width = pipeline.max_total_threads_per_threadgroup().min(256);
    let grid = mtl_size(n.div_ceil(width), 1, 1);
    let threads = mtl_size(width, 1, 1);
    encoder.dispatch_thread_groups(grid, threads);
}

/// The attention-glue kernels address their tensors with i32 index math; refuse
/// a launch whose flat element count would wrap. (Production maxima are ~5M.)
fn glue_index_fits_i32(n: usize) -> Result<()> {
    if n > i32::MAX as usize {
        bail!("attn-glue index math overflows i32: {n} elements exceed i32::MAX");
    }
    Ok(())
}

/// Overflow-checked product of size components. The glue-op guards
/// (`glue_index_fits_i32`, table-extent bounds) must see the TRUE mathematical
/// value: an unchecked usize product wraps in release builds, which could carry
/// a wrapped (small) count past the guard. Not reachable from real tensors
/// (candle cannot hold one that large), but the guards should not be
/// circumventable in principle.
fn checked_elems(parts: &[usize], what: &str) -> Result<usize> {
    let mut n = 1usize;
    for &p in parts {
        n = n
            .checked_mul(p)
            .ok_or_else(|| anyhow::anyhow!("{what}: element count {parts:?} overflows usize"))?;
    }
    Ok(n)
}

/// Fused softplus output gate against the `kernel_attn_gate_*` pair
/// (attn_glue.metal): `dst[h,s,d] = attn[h,s,d] * softplus_chain(gate[s,h])`,
/// replacing the 10-dispatch candle chain (softplus + transpose/reshape +
/// broadcast_mul) with one pass over `attn`. `attn` is `[n_head, seq,
/// head_dim]` contiguous, f32 or f16 (the decode path's raw sdpa output — the
/// f16 variant widens in-kernel, exact, so it is bit-identical to `cast_f32` +
/// the f32 variant); `gate` is `[seq, n_head]` f32 contiguous (the g_proj
/// output layout). Output is always f32. The per-op rounding mirrors candle's
/// chain exactly, so the result is bit-identical (see attn_glue.metal / the
/// attn_glue.rs tests).
pub(crate) fn run_attn_gate(attn: &Tensor, gate: &Tensor) -> Result<Tensor> {
    let cdev = attn.device().clone();
    let Device::Metal(mdev) = &cdev else {
        bail!("attn_gate requires attn on a Metal device");
    };

    let (n_head, seq, head_dim) = attn
        .dims3()
        .map_err(|e| anyhow::anyhow!("attn must be rank-3 [n_head, seq, head_dim]: {e}"))?;
    let kernel_name = match attn.dtype() {
        DType::F32 => "kernel_attn_gate_f32",
        DType::F16 => "kernel_attn_gate_f16",
        dt => bail!("attn must be f32 or f16, got {dt:?}"),
    };
    if !attn.is_contiguous() {
        bail!("attn must be contiguous");
    }
    if gate
        .dims2()
        .map_err(|e| anyhow::anyhow!("gate must be rank-2 [seq, n_head]: {e}"))?
        != (seq, n_head)
    {
        bail!(
            "gate shape {:?} must be [seq, n_head] = [{seq}, {n_head}]",
            gate.dims()
        );
    }
    if gate.dtype() != DType::F32 {
        bail!("gate must be f32, got {:?}", gate.dtype());
    }
    if !gate.is_contiguous() {
        bail!("gate must be contiguous");
    }
    if !attn.device().same_device(gate.device()) {
        bail!("attn and gate must live on the same Metal device");
    }
    let n = checked_elems(&[n_head, seq, head_dim], "attn_gate")?;
    glue_index_fits_i32(n)?;

    let pipeline = pipelines::attn_glue_pipeline(mdev.device(), kernel_name)?;
    let dst = mdev.new_buffer(n, DType::F32, "attn_gate")?;

    let (attn_guard, attn_layout) = attn.storage_and_layout();
    let Storage::Metal(attn_storage) = &*attn_guard else {
        bail!("attn is not on a Metal device");
    };
    let (gate_guard, gate_layout) = gate.storage_and_layout();
    let Storage::Metal(gate_storage) = &*gate_guard else {
        bail!("gate is not on a Metal device");
    };

    let args = AttnGateArgs {
        n_head: n_head as i32,
        seq: seq as i32,
        head_dim: head_dim as i32,
    };
    {
        let cmd = mdev.command_encoder()?;
        let ep = &cmd;
        let encoder = ep.encoder();
        let encoder: &ComputeCommandEncoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_bytes(0, &args);
        encoder.set_input_buffer(
            1,
            Some(attn_storage.buffer()),
            attn_layout.start_offset() * attn.dtype().size_in_bytes(),
        );
        encoder.set_input_buffer(
            2,
            Some(gate_storage.buffer()),
            gate_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder.set_output_buffer(3, Some(&dst), 0);
        dispatch_linear(encoder, &pipeline, n);
    }
    drop(attn_guard);
    drop(gate_guard);

    Ok(output_tensor(dst, mdev, n, (n_head, seq, head_dim)))
}

/// Fused transpose(0,1)+contiguous with optional dtype conversion, against the
/// `kernel_permute_cast_*` family (attn_glue.metal): `x` `[d0, d1, d2]`
/// contiguous becomes `[d1, d0, d2]` contiguous in ONE pass, converting per
/// `out_dtype` (f32→f32 copy, f32→f16 RTNE, f16→f32 exact — candle's cast
/// scalar). `d0 == 1` degenerates to a plain (optionally casting) copy, which
/// is how the shape-preserving `cast_*` wrappers use it.
pub(crate) fn run_permute_cast(x: &Tensor, out_dtype: DType) -> Result<Tensor> {
    let cdev = x.device().clone();
    let Device::Metal(mdev) = &cdev else {
        bail!("permute_cast requires x on a Metal device");
    };

    let (d0, d1, d2) = x
        .dims3()
        .map_err(|e| anyhow::anyhow!("x must be rank-3 [d0, d1, d2]: {e}"))?;
    if !x.is_contiguous() {
        bail!("x must be contiguous");
    }
    let name = match (x.dtype(), out_dtype) {
        (DType::F32, DType::F32) => "kernel_permute_cast_f32_f32",
        (DType::F32, DType::F16) => "kernel_permute_cast_f32_f16",
        (DType::F16, DType::F32) => "kernel_permute_cast_f16_f32",
        (from, to) => bail!("no permute_cast kernel for {from:?} -> {to:?}"),
    };
    let n = checked_elems(&[d0, d1, d2], "permute_cast")?;
    glue_index_fits_i32(n)?;

    let pipeline = pipelines::attn_glue_pipeline(mdev.device(), name)?;
    let dst = mdev.new_buffer(n, out_dtype, "permute_cast")?;

    let (x_guard, x_layout) = x.storage_and_layout();
    let Storage::Metal(x_storage) = &*x_guard else {
        bail!("x is not on a Metal device");
    };

    let args = PermuteArgs {
        d0: d0 as i32,
        d1: d1 as i32,
        d2: d2 as i32,
    };
    {
        let cmd = mdev.command_encoder()?;
        let ep = &cmd;
        let encoder = ep.encoder();
        let encoder: &ComputeCommandEncoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_bytes(0, &args);
        encoder.set_input_buffer(
            1,
            Some(x_storage.buffer()),
            x_layout.start_offset() * x.dtype().size_in_bytes(),
        );
        encoder.set_output_buffer(2, Some(&dst), 0);
        dispatch_linear(encoder, &pipeline, n);
    }
    drop(x_guard);

    let storage = MetalStorage::new(dst, mdev.clone(), n, out_dtype);
    Ok(Tensor::from_storage(
        Storage::Metal(storage),
        (d1, d0, d2),
        candle_core::op::BackpropOp::none(),
        false,
    ))
}

/// Vendored partial-rotary NEOX rope against the `kernel_rope_neox_*` pair
/// (rope.metal): rotates the first `n_rot` dims of `x` `[heads, seq, head_dim]`
/// f32 with candle's by-halves rope math and passes the rest through, in ONE
/// read+write of `x` — replacing the narrow/contiguous/rope/cat chain.
/// `out_dtype` picks the store width (f32, or f16 — the rotation still runs in
/// f32 and only the final store rounds, one RTNE rounding, bit-identical to
/// f32-rope + `cast_f16`; pass-through dims round the same way). `cos`/`sin`
/// are the full `[max_ctx, n_rot/2]` f32 tables; `pos` selects the starting
/// row. Bit-identical to the candle chain (see rope.metal / the attn_glue.rs
/// tests).
pub(crate) fn run_rope(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    pos: usize,
    n_rot: usize,
    out_dtype: DType,
) -> Result<Tensor> {
    let cdev = x.device().clone();
    let Device::Metal(mdev) = &cdev else {
        bail!("rope requires x on a Metal device");
    };

    let (heads, seq, head_dim) = x
        .dims3()
        .map_err(|e| anyhow::anyhow!("x must be rank-3 [heads, seq, head_dim]: {e}"))?;
    if x.dtype() != DType::F32 {
        bail!("x must be f32, got {:?}", x.dtype());
    }
    if !x.is_contiguous() {
        bail!("x must be contiguous");
    }
    if n_rot == 0 || n_rot % 2 != 0 || n_rot > head_dim {
        bail!("n_rot ({n_rot}) must be even and in 2..=head_dim ({head_dim})");
    }
    let kernel_name = match out_dtype {
        DType::F32 => "kernel_rope_neox_f32",
        DType::F16 => "kernel_rope_neox_f16",
        dt => bail!("rope output dtype must be f32 or f16, got {dt:?}"),
    };
    let half = n_rot / 2;
    // Checked: a caller-supplied pos near usize::MAX must not wrap the row
    // bound (release builds wrap unchecked usize adds).
    let end = pos
        .checked_add(seq)
        .ok_or_else(|| anyhow::anyhow!("rope pos + seq ({pos} + {seq}) overflows usize"))?;
    for (name, t) in [("cos", cos), ("sin", sin)] {
        let (rows, cols) = t
            .dims2()
            .map_err(|e| anyhow::anyhow!("{name} must be rank-2 [max_ctx, n_rot/2]: {e}"))?;
        if cols != half {
            bail!("{name} has {cols} columns, expected n_rot/2 = {half}");
        }
        if end > rows {
            bail!("{name} has {rows} rows, need pos + seq = {end}");
        }
        if t.dtype() != DType::F32 {
            bail!("{name} must be f32, got {:?}", t.dtype());
        }
        if !t.is_contiguous() {
            bail!("{name} must be contiguous");
        }
        if !x.device().same_device(t.device()) {
            bail!("{name} must live on the same Metal device as x");
        }
    }
    let n = checked_elems(&[heads, seq, head_dim], "rope")?;
    glue_index_fits_i32(n)?;
    glue_index_fits_i32(checked_elems(&[end, half], "rope tables")?)?;

    let pipeline = pipelines::rope_pipeline(mdev.device(), kernel_name)?;
    let dst = mdev.new_buffer(n, out_dtype, "rope")?;

    let (x_guard, x_layout) = x.storage_and_layout();
    let Storage::Metal(x_storage) = &*x_guard else {
        bail!("x is not on a Metal device");
    };
    let (cos_guard, cos_layout) = cos.storage_and_layout();
    let Storage::Metal(cos_storage) = &*cos_guard else {
        bail!("cos is not on a Metal device");
    };
    let (sin_guard, sin_layout) = sin.storage_and_layout();
    let Storage::Metal(sin_storage) = &*sin_guard else {
        bail!("sin is not on a Metal device");
    };

    let args = RopeArgs {
        heads: heads as i32,
        seq: seq as i32,
        head_dim: head_dim as i32,
        n_rot: n_rot as i32,
        pos: pos as i32,
    };
    {
        let cmd = mdev.command_encoder()?;
        let ep = &cmd;
        let encoder = ep.encoder();
        let encoder: &ComputeCommandEncoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_bytes(0, &args);
        encoder.set_input_buffer(
            1,
            Some(x_storage.buffer()),
            x_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder.set_input_buffer(
            2,
            Some(cos_storage.buffer()),
            cos_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder.set_input_buffer(
            3,
            Some(sin_storage.buffer()),
            sin_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder.set_output_buffer(4, Some(&dst), 0);
        dispatch_linear(encoder, &pipeline, n);
    }
    drop(x_guard);
    drop(cos_guard);
    drop(sin_guard);

    let storage = MetalStorage::new(dst, mdev.clone(), n, out_dtype);
    Ok(Tensor::from_storage(
        Storage::Metal(storage),
        (heads, seq, head_dim),
        candle_core::op::BackpropOp::none(),
        false,
    ))
}

/// Matches the Metal `delta_conv_args` struct (src/ops/delta.metal).
#[repr(C)]
#[derive(Clone, Copy)]
struct DeltaConvArgs {
    seq: i32,
    conv_dim: i32,
    taps: i32,
    tail: i32,
}

/// Matches the Metal `delta_ba_args` struct (src/ops/delta.metal).
#[repr(C)]
#[derive(Clone, Copy)]
struct DeltaBaArgs {
    seq: i32,
    v_heads: i32,
}

/// Matches the Metal `delta_ba_fused_args` struct (src/ops/delta.metal).
#[repr(C)]
#[derive(Clone, Copy)]
struct DeltaBaFusedArgs {
    seq: i32,
    hidden: i32,
    v_heads: i32,
}

/// Matches the Metal `delta_gnorm_args` struct (src/ops/delta.metal).
#[repr(C)]
#[derive(Clone, Copy)]
struct DeltaGnormArgs {
    v_heads: i32,
    head_dim: i32,
    eps: f32,
}

/// Matches the Metal `delta_l2norm_args` struct (src/ops/delta.metal).
#[repr(C)]
#[derive(Clone, Copy)]
struct DeltaL2NormArgs {
    k_heads: i32,
    conv_dim: i32,
    eps: f32,
}

/// Matches the Metal `delta_scan_args` struct (src/ops/delta.metal).
#[repr(C)]
#[derive(Clone, Copy)]
struct DeltaScanArgs {
    seq: i32,
    k_heads: i32,
    v_heads: i32,
    conv_dim: i32,
    n_planes: i32,
    scale: f32,
    eps: f32,
}

/// Matches the Metal `delta_scan_v2_args` struct (src/ops/delta.metal). No eps:
/// the v2 scan reads q and k already normalized, from `run_delta_l2norm`.
#[repr(C)]
#[derive(Clone, Copy)]
struct DeltaScanV2Args {
    seq: i32,
    k_heads: i32,
    v_heads: i32,
    conv_dim: i32,
    n_planes: i32,
    scale: f32,
}

/// The DeltaNet head dim the scan kernel is specialized to. Both checkpoints
/// run gated DeltaNet at 128 (27B: 16 K-heads / 48 V-heads, 35B-A3B: 16 / 32),
/// so this is the production geometry, not a restriction that ever binds; a
/// block at any other head dim keeps the reference scan.
pub(crate) const DELTA_HEAD_DIM: usize = 128;

/// Threadgroups the scan kernel launches per V-head (`DELTA_HEAD_DIM /
/// DELTA_TG_COLS` in delta.metal — the value-dim columns split four ways).
/// Kept in step with the kernel's own `#define` by
/// `scan_geometry_matches_metal` (src/ops/delta.rs): the grid below is sized
/// from this copy, so a drift would write outside a head's state slice.
pub(crate) const DELTA_COL_BLOCKS: usize = 4;

/// Threadgroups the DECODE scan kernel launches per V-head
/// (`DELTA_DEC_COL_BLOCKS` in delta.metal — the value-dim columns split four
/// ways, four columns to a thread). Kept in step with the kernel's own
/// `#define` by `scan_geometry_matches_metal` (src/ops/delta.rs): the grid is
/// sized from this copy, so a drift would write outside a head's state slice.
pub(crate) const DELTA_DEC_COL_BLOCKS: usize = 4;

/// Simdgroups per threadgroup in the v2 scan (`DELTA_V2_SGS` in delta.metal) —
/// also the number of state value-columns a threadgroup covers, one per
/// simdgroup, since a simdgroup owns a column outright.
pub(crate) const DELTA_V2_SGS: usize = 4;

/// Threadgroups the v2 scan launches per V-head (`DELTA_V2_COL_TGS` in
/// delta.metal — `DELTA_HEAD_DIM / DELTA_V2_SGS`). Kept in step with the
/// kernel's own `#define`s by `scan_geometry_matches_metal` (src/ops/delta.rs):
/// the grid below is sized from these copies, so a drift would leave part of a
/// head's state unowned.
pub(crate) const DELTA_V2_COL_TGS: usize = 32;

/// Output columns one `kernel_delta_ba_fused_*` threadgroup owns
/// (`DELTA_BA_COLS` in delta.metal). The column partition needs no
/// cross-threadgroup reduction, so this alone sets how many threadgroups the
/// launch has per token tile: `ceil(2 * v_heads / DELTA_BA_COLS)`.
///
/// FITTED, 2026-08-30 (`delta_ba_timing`, Flash-Next geometry, decode): 8 gives
/// 12 threadgroups at 96 columns and 7.4 us; 16 gives 6 and 9.9 us; 4 gives 24
/// and ties at decode but loses by half at seq 8-32. Narrower than 8 the
/// per-threadgroup weight run stops filling a cache line on its own and the
/// blocks sharing one line have to arrive together to make up for it.
pub(crate) const DELTA_BA_COLS: usize = 8;

/// Row chunks of the hidden dim one `kernel_delta_ba_fused_*` threadgroup
/// splits its columns' dot products across (`DELTA_BA_ROWS` in delta.metal).
/// `DELTA_BA_COLS * DELTA_BA_ROWS` is the threadgroup width.
pub(crate) const DELTA_BA_ROWS: usize = 128;

/// Tokens the `_t4` specialization tiles into one threadgroup
/// (`DELTA_BA_TOKS` in delta.metal), so a short chunk reads the weight once per
/// tile instead of once per token. Kept in step with the kernel's own
/// `#define`s by `ba_fused_geometry_matches_metal` (src/ops/delta.rs): the grid
/// below is sized from these copies, so a drift would leave columns or tokens
/// unwritten.
pub(crate) const DELTA_BA_TOKS: usize = 4;

/// Token ceiling for the fused beta|alpha projection. The kernel reads the
/// whole `[hidden, 2 * v_heads]` weight once per token TILE, so its advantage
/// over candle's gemm — which reads it once per chunk — decays with the token
/// count; above this the gemm wins and the two-dispatch chain is taken instead.
/// Covers decode (1) and a DFlash verify block (16).
///
/// The measured crossover is farther out than 32 — at the Flash-Next geometry
/// the fused arm is 18.8 us against the chain's 71.7 at seq 32
/// (`delta_ba_timing`, 2026-08-30) — and this ceiling is deliberately short of
/// it: prefill chunks are hundreds of tokens, where the fused kernel's
/// once-per-tile weight read has never been measured and the gemm's reuse is
/// the whole reason prefill is shaped the way it is.
pub(crate) const DELTA_BA_MAX_SEQ: usize = 32;

/// V-head ceiling for the fused beta|alpha projection: the shipped geometries
/// are 48 and 32, and the kernel's grid is `2 * v_heads` columns wide.
pub(crate) const DELTA_BA_MAX_V_HEADS: usize = 64;

/// Hidden-dim ceiling for the fused beta|alpha projection (shipped: 5120, 2560,
/// 2048). Each thread walks `hidden / DELTA_BA_ROWS` weight rows, so a far
/// wider hidden is a shape this launch was not measured at.
pub(crate) const DELTA_BA_MAX_HIDDEN: usize = 8192;

/// The simdgroup width the delta kernels are written against.
/// `kernel_delta_scan` reduces q/k through `red[2][DELTA_D / 32]` indexed by
/// simdgroup index, and `kernel_delta_gnorm` folds its sum of squares through a
/// 32-entry partial row, so a device that executed simdgroups at any other
/// width would reduce over the wrong lane set (and, in the scan, index past the
/// reduction row). Every Apple GPU is 32-wide; the assumption is checked at
/// pipeline setup so it fails at load rather than silently at dispatch.
const DELTA_SIMD_WIDTH: usize = 32;

/// Refuse a delta pipeline whose simdgroups are not `DELTA_SIMD_WIDTH` wide.
fn check_delta_simd_width(pipeline: &ComputePipeline, what: &str) -> Result<()> {
    use objc2::runtime::ProtocolObject;
    use objc2_metal::MTLComputePipelineState;

    let raw: &ProtocolObject<dyn MTLComputePipelineState> = pipeline.as_ref();
    let width = raw.threadExecutionWidth();
    if width != DELTA_SIMD_WIDTH {
        bail!(
            "{what} is written for {DELTA_SIMD_WIDTH}-wide simdgroups, this device executes \
             them {width} wide"
        );
    }
    Ok(())
}

/// Guard on an operand that must be a contiguous f32 tensor of a known shape.
fn check_f32(t: &Tensor, want: &[usize], what: &str) -> Result<()> {
    if t.dtype() != DType::F32 {
        bail!("{what} must be f32, got {:?}", t.dtype());
    }
    if t.dims() != want {
        bail!("{what} shape {:?} must be {want:?}", t.dims());
    }
    if !t.is_contiguous() {
        bail!("{what} must be contiguous");
    }
    Ok(())
}

/// Byte offset of a contiguous f32 tensor's first element inside its buffer.
fn f32_off(layout: &candle_core::Layout) -> usize {
    layout.start_offset() * DType::F32.size_in_bytes()
}

/// Fused causal depthwise conv + silu + next conv window against
/// `kernel_delta_conv` (delta.metal). `state` is the carried window
/// `[taps - 1, conv_dim]`, `qkv` this chunk's fused projection rows
/// `[seq, conv_dim]`, `w` the taps `[taps, conv_dim]` (oldest tap first); all
/// f32 contiguous. Returns the silu'd conv output `[seq, conv_dim]` and the
/// window the next call starts from `[taps - 1, conv_dim]`, replacing the
/// reference's cat + per-tap broadcast chain + silu + slice_set materialization
/// with ONE pass. Bit-identical to that chain (see delta.metal / the delta.rs
/// test).
pub(crate) fn run_delta_conv(state: &Tensor, qkv: &Tensor, w: &Tensor) -> Result<(Tensor, Tensor)> {
    let cdev = qkv.device().clone();
    let Device::Metal(mdev) = &cdev else {
        bail!("delta_conv requires qkv on a Metal device");
    };

    let (seq, conv_dim) = qkv
        .dims2()
        .map_err(|e| anyhow::anyhow!("qkv must be rank-2 [seq, conv_dim]: {e}"))?;
    let (taps, w_dim) = w
        .dims2()
        .map_err(|e| anyhow::anyhow!("w must be rank-2 [taps, conv_dim]: {e}"))?;
    if w_dim != conv_dim {
        bail!("conv taps span {w_dim} channels, expected conv_dim {conv_dim}");
    }
    if taps == 0 {
        bail!("conv needs at least one tap");
    }
    let tail = taps - 1;
    check_f32(qkv, &[seq, conv_dim], "qkv")?;
    check_f32(w, &[taps, conv_dim], "w")?;
    check_f32(state, &[tail, conv_dim], "conv state")?;
    if seq == 0 {
        bail!("delta_conv needs at least one token");
    }
    for (name, t) in [("state", state), ("w", w)] {
        if !qkv.device().same_device(t.device()) {
            bail!("{name} must live on the same Metal device as qkv");
        }
    }
    let n = checked_elems(&[seq, conv_dim], "delta_conv")?;
    glue_index_fits_i32(checked_elems(&[seq + tail, conv_dim], "delta_conv stream")?)?;

    let pipeline = pipelines::delta_pipeline(mdev.device(), "kernel_delta_conv")?;
    let dst = mdev.new_buffer(n, DType::F32, "delta_conv")?;
    let nstate_len = tail * conv_dim;
    let nstate = mdev.new_buffer(nstate_len.max(1), DType::F32, "delta_conv_state")?;

    let (state_guard, state_layout) = state.storage_and_layout();
    let Storage::Metal(state_storage) = &*state_guard else {
        bail!("conv state is not on a Metal device");
    };
    let (qkv_guard, qkv_layout) = qkv.storage_and_layout();
    let Storage::Metal(qkv_storage) = &*qkv_guard else {
        bail!("qkv is not on a Metal device");
    };
    let (w_guard, w_layout) = w.storage_and_layout();
    let Storage::Metal(w_storage) = &*w_guard else {
        bail!("w is not on a Metal device");
    };

    let args = DeltaConvArgs {
        seq: seq as i32,
        conv_dim: conv_dim as i32,
        taps: taps as i32,
        tail: tail as i32,
    };
    {
        let cmd = mdev.command_encoder()?;
        let ep = &cmd;
        let encoder = ep.encoder();
        let encoder: &ComputeCommandEncoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_bytes(0, &args);
        encoder.set_input_buffer(1, Some(state_storage.buffer()), f32_off(state_layout));
        encoder.set_input_buffer(2, Some(qkv_storage.buffer()), f32_off(qkv_layout));
        encoder.set_input_buffer(3, Some(w_storage.buffer()), f32_off(w_layout));
        encoder.set_output_buffer(4, Some(&dst), 0);
        encoder.set_output_buffer(5, Some(&nstate), 0);
        dispatch_linear(encoder, &pipeline, n);
    }
    drop(state_guard);
    drop(qkv_guard);
    drop(w_guard);

    Ok((
        output_tensor(dst, mdev, n, (seq, conv_dim)),
        output_tensor(nstate, mdev, nstate_len, (tail, conv_dim)),
    ))
}

/// Fused beta / log-decay head against `kernel_delta_ba` (delta.metal). `ba` is
/// the fused `[seq, 2 * v_heads]` beta|alpha projection output, `ssm_a` the
/// pre-baked `-exp(A_log)` `[v_heads]`, `dt_bias` the dt offset `[v_heads]`.
/// Returns `beta = sigmoid(b_raw)` and the LOG decay
/// `g = ssm_a * softplus(a_raw + dt_bias)`, both `[seq, v_heads]` — the scan
/// kernel exponentiates g, so the reference's separate exp pass is folded away.
/// Bit-identical to the candle chain given the same `ba` (see the delta.rs test).
pub(crate) fn run_delta_ba(
    ba: &Tensor,
    ssm_a: &Tensor,
    dt_bias: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let cdev = ba.device().clone();
    let Device::Metal(mdev) = &cdev else {
        bail!("delta_ba requires ba on a Metal device");
    };

    let (seq, two_vh) = ba
        .dims2()
        .map_err(|e| anyhow::anyhow!("ba must be rank-2 [seq, 2 * v_heads]: {e}"))?;
    if two_vh % 2 != 0 {
        bail!("ba has {two_vh} columns, expected an even 2 * v_heads");
    }
    let v_heads = two_vh / 2;
    check_f32(ba, &[seq, two_vh], "ba")?;
    check_f32(ssm_a, &[v_heads], "ssm_a")?;
    check_f32(dt_bias, &[v_heads], "dt_bias")?;
    if seq == 0 || v_heads == 0 {
        bail!("delta_ba needs at least one token and one V-head");
    }
    for (name, t) in [("ssm_a", ssm_a), ("dt_bias", dt_bias)] {
        if !ba.device().same_device(t.device()) {
            bail!("{name} must live on the same Metal device as ba");
        }
    }
    let n = checked_elems(&[seq, v_heads], "delta_ba")?;
    glue_index_fits_i32(checked_elems(&[seq, two_vh], "delta_ba input")?)?;

    let pipeline = pipelines::delta_pipeline(mdev.device(), "kernel_delta_ba")?;
    let beta = mdev.new_buffer(n, DType::F32, "delta_beta")?;
    let g = mdev.new_buffer(n, DType::F32, "delta_g")?;

    let (ba_guard, ba_layout) = ba.storage_and_layout();
    let Storage::Metal(ba_storage) = &*ba_guard else {
        bail!("ba is not on a Metal device");
    };
    let (a_guard, a_layout) = ssm_a.storage_and_layout();
    let Storage::Metal(a_storage) = &*a_guard else {
        bail!("ssm_a is not on a Metal device");
    };
    let (dt_guard, dt_layout) = dt_bias.storage_and_layout();
    let Storage::Metal(dt_storage) = &*dt_guard else {
        bail!("dt_bias is not on a Metal device");
    };

    let args = DeltaBaArgs {
        seq: seq as i32,
        v_heads: v_heads as i32,
    };
    {
        let cmd = mdev.command_encoder()?;
        let ep = &cmd;
        let encoder = ep.encoder();
        let encoder: &ComputeCommandEncoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_bytes(0, &args);
        encoder.set_input_buffer(1, Some(ba_storage.buffer()), f32_off(ba_layout));
        encoder.set_input_buffer(2, Some(a_storage.buffer()), f32_off(a_layout));
        encoder.set_input_buffer(3, Some(dt_storage.buffer()), f32_off(dt_layout));
        encoder.set_output_buffer(4, Some(&beta), 0);
        encoder.set_output_buffer(5, Some(&g), 0);
        dispatch_linear(encoder, &pipeline, n);
    }
    drop(ba_guard);
    drop(a_guard);
    drop(dt_guard);

    Ok((
        output_tensor(beta, mdev, n, (seq, v_heads)),
        output_tensor(g, mdev, n, (seq, v_heads)),
    ))
}

/// The kernel `run_delta_ba_fused` dispatches for a chunk of `seq` tokens, and
/// the tokens its threadgroup tiles. One token per threadgroup at decode, where
/// the tile clamp folds away. The predicate below asks the same question, so
/// both sides name the same pipeline.
fn delta_ba_fused_kernel(seq: usize) -> (&'static str, usize) {
    if seq == 1 {
        ("kernel_delta_ba_fused_t1", 1)
    } else {
        ("kernel_delta_ba_fused_t4", DELTA_BA_TOKS)
    }
}

/// Whether `run_delta_ba_fused` can serve this pair — the predicate the block
/// consults before choosing between the fused kernel and the candle gemv plus
/// `kernel_delta_ba`. Everything it checks is a geometry, layout or device
/// capability fact the kernel's grid depends on, so a `false` here is "take the
/// other path", never an error.
///
/// The capability half is the pipeline's own threadgroup limit: the grid is a
/// fixed `DELTA_BA_COLS * DELTA_BA_ROWS` threads wide, so a device that derates
/// this pipeline below that cannot run it at any geometry. It is checked HERE
/// rather than at the dispatch fork precisely because a derated device must
/// fall back rather than fail (the dispatch keeps the same check as a
/// defensive bail).
pub(crate) fn delta_ba_fused_applies(x: &Tensor, w: &Tensor) -> bool {
    let (Ok((seq, hidden)), Ok((w_rows, two_vh))) = (x.dims2(), w.dims2()) else {
        return false;
    };
    let Device::Metal(mdev) = x.device() else {
        return false;
    };
    let geometry = x.device().same_device(w.device())
        && x.dtype() == DType::F32
        && w.dtype() == DType::F32
        && x.is_contiguous()
        && w.is_contiguous()
        && w_rows == hidden
        && two_vh % 2 == 0
        && two_vh / 2 <= DELTA_BA_MAX_V_HEADS
        && two_vh >= 2
        && (1..=DELTA_BA_MAX_SEQ).contains(&seq)
        && (1..=DELTA_BA_MAX_HIDDEN).contains(&hidden);
    if !geometry {
        return false;
    }
    let (kernel, _) = delta_ba_fused_kernel(seq);
    pipelines::delta_max_threads(mdev.device(), kernel)
        .is_ok_and(|limit| limit >= DELTA_BA_COLS * DELTA_BA_ROWS)
}

/// The beta|alpha PROJECTION and the beta/decay head in ONE dispatch, against
/// `kernel_delta_ba_fused_t{1,4}` (delta.metal). `x` is the layer input
/// `[seq, hidden]` f32, `w` the concatenated `[hidden, 2 * v_heads]` beta|alpha
/// weight built at load (column block 0 beta, block 1 alpha), `ssm_a` the
/// pre-baked `-exp(A_log)` and `dt_bias` the dt offset, both `[v_heads]`.
/// Returns the same `(beta, g)` pair as [`run_delta_ba`] over the gemv output,
/// with `g` the LOG decay.
///
/// This replaces a candle f32 gemv that cost 29 us per layer for 96 dot
/// products — ~50x off its byte floor — plus the head dispatch behind it. It is
/// for SMALL token counts only ([`delta_ba_fused_applies`]): the weight is read
/// once per `DELTA_BA_TOKS`-token tile, so a prefill chunk keeps the gemm.
///
/// NOT bit-identical to the gemv + [`run_delta_ba`] chain: the dot product is
/// summed as `DELTA_BA_ROWS` partials folded in a tree where candle sums in its
/// own order. The epilogue is the same Metal helper, so that reassociation is
/// the only difference (graded at 2e-6 in delta.rs).
pub(crate) fn run_delta_ba_fused(
    x: &Tensor,
    w: &Tensor,
    ssm_a: &Tensor,
    dt_bias: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let cdev = x.device().clone();
    let Device::Metal(mdev) = &cdev else {
        bail!("delta_ba_fused requires x on a Metal device");
    };

    let (seq, hidden) = x
        .dims2()
        .map_err(|e| anyhow::anyhow!("x must be rank-2 [seq, hidden]: {e}"))?;
    let (w_rows, two_vh) = w
        .dims2()
        .map_err(|e| anyhow::anyhow!("the beta|alpha weight must be rank-2: {e}"))?;
    if two_vh % 2 != 0 {
        bail!("the beta|alpha weight has {two_vh} columns, expected an even 2 * v_heads");
    }
    let v_heads = two_vh / 2;
    check_f32(x, &[seq, hidden], "x")?;
    check_f32(w, &[w_rows, two_vh], "beta|alpha weight")?;
    check_f32(ssm_a, &[v_heads], "ssm_a")?;
    check_f32(dt_bias, &[v_heads], "dt_bias")?;
    if w_rows != hidden {
        bail!("the beta|alpha weight has {w_rows} rows, expected x's hidden dim {hidden}");
    }
    if seq == 0 || v_heads == 0 || hidden == 0 {
        bail!("delta_ba_fused needs at least one token, one V-head and one input column");
    }
    if seq > DELTA_BA_MAX_SEQ {
        bail!("delta_ba_fused is for at most {DELTA_BA_MAX_SEQ} tokens, got {seq}");
    }
    if v_heads > DELTA_BA_MAX_V_HEADS {
        bail!("delta_ba_fused is for at most {DELTA_BA_MAX_V_HEADS} V-heads, got {v_heads}");
    }
    if hidden > DELTA_BA_MAX_HIDDEN {
        bail!("delta_ba_fused is for a hidden dim of at most {DELTA_BA_MAX_HIDDEN}, got {hidden}");
    }
    for (name, t) in [
        ("the beta|alpha weight", w),
        ("ssm_a", ssm_a),
        ("dt_bias", dt_bias),
    ] {
        if !x.device().same_device(t.device()) {
            bail!("{name} must live on the same Metal device as x");
        }
    }
    let n = checked_elems(&[seq, v_heads], "delta_ba_fused")?;
    glue_index_fits_i32(checked_elems(&[hidden, two_vh], "delta_ba_fused weight")?)?;
    glue_index_fits_i32(checked_elems(&[seq, hidden], "delta_ba_fused input")?)?;

    let (kernel, toks) = delta_ba_fused_kernel(seq);
    let pipeline = pipelines::delta_pipeline(mdev.device(), kernel)?;
    let width = DELTA_BA_COLS * DELTA_BA_ROWS;
    // Defensive: `delta_ba_fused_applies` has already refused a device that
    // derates this pipeline, so a caller reaching here on one skipped the
    // predicate rather than being told to fall back.
    if pipeline.max_total_threads_per_threadgroup() < width {
        bail!(
            "{kernel} needs {width} threads per threadgroup, the pipeline allows {}",
            pipeline.max_total_threads_per_threadgroup()
        );
    }
    let beta = mdev.new_buffer(n, DType::F32, "delta_beta")?;
    let g = mdev.new_buffer(n, DType::F32, "delta_g")?;

    let (x_guard, x_layout) = x.storage_and_layout();
    let Storage::Metal(x_storage) = &*x_guard else {
        bail!("x is not on a Metal device");
    };
    let (w_guard, w_layout) = w.storage_and_layout();
    let Storage::Metal(w_storage) = &*w_guard else {
        bail!("the beta|alpha weight is not on a Metal device");
    };
    let (a_guard, a_layout) = ssm_a.storage_and_layout();
    let Storage::Metal(a_storage) = &*a_guard else {
        bail!("ssm_a is not on a Metal device");
    };
    let (dt_guard, dt_layout) = dt_bias.storage_and_layout();
    let Storage::Metal(dt_storage) = &*dt_guard else {
        bail!("dt_bias is not on a Metal device");
    };

    let args = DeltaBaFusedArgs {
        seq: seq as i32,
        hidden: hidden as i32,
        v_heads: v_heads as i32,
    };
    {
        let cmd = mdev.command_encoder()?;
        let ep = &cmd;
        let encoder = ep.encoder();
        let encoder: &ComputeCommandEncoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_bytes(0, &args);
        encoder.set_input_buffer(1, Some(x_storage.buffer()), f32_off(x_layout));
        encoder.set_input_buffer(2, Some(w_storage.buffer()), f32_off(w_layout));
        encoder.set_input_buffer(3, Some(a_storage.buffer()), f32_off(a_layout));
        encoder.set_input_buffer(4, Some(dt_storage.buffer()), f32_off(dt_layout));
        encoder.set_output_buffer(5, Some(&beta), 0);
        encoder.set_output_buffer(6, Some(&g), 0);
        encoder.dispatch_thread_groups(
            mtl_size(two_vh.div_ceil(DELTA_BA_COLS), seq.div_ceil(toks), 1),
            mtl_size(width, 1, 1),
        );
    }
    drop(x_guard);
    drop(w_guard);
    drop(a_guard);
    drop(dt_guard);

    Ok((
        output_tensor(beta, mdev, n, (seq, v_heads)),
        output_tensor(g, mdev, n, (seq, v_heads)),
    ))
}

/// Fused gated output RMSNorm against `kernel_delta_gnorm` (delta.metal):
/// `rms_norm(o, eps) * ssm_norm_weight * gate(z)` per head, in one pass. `o` is
/// `[seq, v_heads, head_dim]` f32, `z` the gate projection output in the same
/// element order, `w` the `[head_dim]` weight. The gate is applied AFTER the
/// weight and outside the norm, so it does not enter the statistic.
///
/// `gate` selects the activation, and with it the kernel: `silu(z)` on the
/// qwen35/qwen35moe graphs, `sigmoid(z)` on qwen4exp. The two kernels share one
/// templated body, so the silu arm is the same instruction stream it always was.
pub(crate) fn run_delta_gnorm(
    o: &Tensor,
    z: &Tensor,
    w: &Tensor,
    eps: f32,
    gate: ZGate,
) -> Result<Tensor> {
    let cdev = o.device().clone();
    let Device::Metal(mdev) = &cdev else {
        bail!("delta_gnorm requires o on a Metal device");
    };

    let (seq, v_heads, head_dim) = o
        .dims3()
        .map_err(|e| anyhow::anyhow!("o must be rank-3 [seq, v_heads, head_dim]: {e}"))?;
    check_f32(o, &[seq, v_heads, head_dim], "o")?;
    check_f32(z, &[seq, v_heads, head_dim], "z")?;
    check_f32(w, &[head_dim], "ssm_norm weight")?;
    // The sum of squares folds through simd_sum over whole simdgroups, so the
    // head dim must tile them exactly.
    if head_dim == 0 || head_dim % 32 != 0 {
        bail!("delta_gnorm head_dim ({head_dim}) must be a nonzero multiple of 32");
    }
    // The grid is one threadgroup per (token, head).
    if seq == 0 || v_heads == 0 {
        bail!("delta_gnorm needs at least one token and one head");
    }
    for (name, t) in [("z", z), ("ssm_norm weight", w)] {
        if !o.device().same_device(t.device()) {
            bail!("{name} must live on the same Metal device as o");
        }
    }
    let n = checked_elems(&[seq, v_heads, head_dim], "delta_gnorm")?;
    glue_index_fits_i32(n)?;

    let kernel = match gate {
        ZGate::Silu => "kernel_delta_gnorm",
        ZGate::Sigmoid => "kernel_delta_gnorm_sigmoid",
    };
    let pipeline = pipelines::delta_pipeline(mdev.device(), kernel)?;
    if pipeline.max_total_threads_per_threadgroup() < head_dim {
        bail!(
            "delta_gnorm needs {head_dim} threads per threadgroup, the pipeline allows {}",
            pipeline.max_total_threads_per_threadgroup()
        );
    }
    check_delta_simd_width(&pipeline, kernel)?;
    let dst = mdev.new_buffer(n, DType::F32, "delta_gnorm")?;

    let (o_guard, o_layout) = o.storage_and_layout();
    let Storage::Metal(o_storage) = &*o_guard else {
        bail!("o is not on a Metal device");
    };
    let (z_guard, z_layout) = z.storage_and_layout();
    let Storage::Metal(z_storage) = &*z_guard else {
        bail!("z is not on a Metal device");
    };
    let (w_guard, w_layout) = w.storage_and_layout();
    let Storage::Metal(w_storage) = &*w_guard else {
        bail!("ssm_norm weight is not on a Metal device");
    };

    let args = DeltaGnormArgs {
        v_heads: v_heads as i32,
        head_dim: head_dim as i32,
        eps,
    };
    {
        let cmd = mdev.command_encoder()?;
        let ep = &cmd;
        let encoder = ep.encoder();
        let encoder: &ComputeCommandEncoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_bytes(0, &args);
        encoder.set_input_buffer(1, Some(o_storage.buffer()), f32_off(o_layout));
        encoder.set_input_buffer(2, Some(z_storage.buffer()), f32_off(z_layout));
        encoder.set_input_buffer(3, Some(w_storage.buffer()), f32_off(w_layout));
        encoder.set_output_buffer(4, Some(&dst), 0);
        encoder.dispatch_thread_groups(mtl_size(seq * v_heads, 1, 1), mtl_size(head_dim, 1, 1));
    }
    drop(o_guard);
    drop(z_guard);
    drop(w_guard);

    Ok(output_tensor(dst, mdev, n, (seq, v_heads, head_dim)))
}

/// The q/k L2 clamp-norm against `kernel_delta_l2norm` (delta.metal). `conv` is
/// the silu'd conv output `[seq, conv_dim]` (q | k | v fused); its leading
/// `2 * k_heads * DELTA_HEAD_DIM` columns are the per-K-head q planes followed
/// by the k planes. Returns exactly those columns, normalized, in the same
/// order — the `[seq, 2 * k_heads * DELTA_HEAD_DIM]` buffer the scan reads q and
/// k from. v stays where it is, unnormalized, in `conv`.
///
/// A dispatch of its own on purpose: the scan's threadgroups outnumber the
/// K-head planes by the V-head ratio times the value-column split, so folding
/// this in recomputes every norm many times over on every timestep. Bounded, not
/// bitwise, against `linear_attn::l2_norm` — only the 128-term sum of squares
/// reassociates.
pub(crate) fn run_delta_l2norm(conv: &Tensor, k_heads: usize, eps: f32) -> Result<Tensor> {
    let cdev = conv.device().clone();
    let Device::Metal(mdev) = &cdev else {
        bail!("delta_l2norm requires conv on a Metal device");
    };

    let (seq, conv_dim) = conv
        .dims2()
        .map_err(|e| anyhow::anyhow!("conv must be rank-2 [seq, conv_dim]: {e}"))?;
    check_f32(conv, &[seq, conv_dim], "conv")?;
    if seq == 0 || k_heads == 0 {
        bail!("delta_l2norm needs at least one token and one K-head");
    }
    let qk_dim = 2 * k_heads * DELTA_HEAD_DIM;
    if conv_dim < qk_dim {
        bail!("conv width {conv_dim} is narrower than the {qk_dim} q|k columns to normalize");
    }
    let n = checked_elems(&[seq, qk_dim], "delta_l2norm")?;
    glue_index_fits_i32(checked_elems(&[seq, conv_dim], "delta_l2norm conv")?)?;
    glue_index_fits_i32(n)?;

    let pipeline = pipelines::delta_pipeline(mdev.device(), "kernel_delta_l2norm")?;
    if pipeline.max_total_threads_per_threadgroup() < DELTA_HEAD_DIM {
        bail!(
            "delta_l2norm needs {DELTA_HEAD_DIM} threads per threadgroup, the pipeline allows {}",
            pipeline.max_total_threads_per_threadgroup()
        );
    }
    check_delta_simd_width(&pipeline, "kernel_delta_l2norm")?;
    let dst = mdev.new_buffer(n, DType::F32, "delta_l2norm")?;

    let (conv_guard, conv_layout) = conv.storage_and_layout();
    let Storage::Metal(conv_storage) = &*conv_guard else {
        bail!("conv is not on a Metal device");
    };

    let args = DeltaL2NormArgs {
        k_heads: k_heads as i32,
        conv_dim: conv_dim as i32,
        eps,
    };
    {
        let cmd = mdev.command_encoder()?;
        let ep = &cmd;
        let encoder = ep.encoder();
        let encoder: &ComputeCommandEncoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_bytes(0, &args);
        encoder.set_input_buffer(1, Some(conv_storage.buffer()), f32_off(conv_layout));
        encoder.set_output_buffer(2, Some(&dst), 0);
        encoder.dispatch_thread_groups(
            mtl_size(seq * 2 * k_heads, 1, 1),
            mtl_size(DELTA_HEAD_DIM, 1, 1),
        );
    }
    drop(conv_guard);

    Ok(output_tensor(dst, mdev, n, (seq, qk_dim)))
}

/// The operand contract both scan decompositions hold their inputs to. Returns
/// `(seq, conv_dim, v_heads)`.
fn check_delta_scan_operands(
    conv: &Tensor,
    beta: &Tensor,
    g: &Tensor,
    s: &Tensor,
    k_heads: usize,
    planes: usize,
) -> Result<(usize, usize, usize)> {
    let (seq, conv_dim) = conv
        .dims2()
        .map_err(|e| anyhow::anyhow!("conv must be rank-2 [seq, conv_dim]: {e}"))?;
    let (v_heads, d_k, d_v) = s
        .dims3()
        .map_err(|e| anyhow::anyhow!("state must be rank-3 [v_heads, d_k, d_v]: {e}"))?;
    if d_k != DELTA_HEAD_DIM || d_v != DELTA_HEAD_DIM {
        bail!(
            "delta_scan is specialized to head dim {DELTA_HEAD_DIM}, got state [{v_heads}, {d_k}, {d_v}]"
        );
    }
    if k_heads == 0 || v_heads == 0 || v_heads % k_heads != 0 {
        bail!("v_heads ({v_heads}) must be a nonzero multiple of k_heads ({k_heads})");
    }
    let want_conv_dim = (2 * k_heads + v_heads) * DELTA_HEAD_DIM;
    if conv_dim != want_conv_dim {
        bail!(
            "conv width {conv_dim} does not match (2 * {k_heads} + {v_heads}) * {DELTA_HEAD_DIM} = {want_conv_dim}"
        );
    }
    if seq == 0 {
        bail!("delta_scan needs at least one token");
    }
    // Plane p of the state output is the state after token seq-1-p, so a chunk
    // of `seq` tokens has exactly `seq` states to name.
    if planes == 0 || planes > seq {
        bail!("delta_scan needs 1 <= state planes <= seq, got {planes} planes for seq {seq}");
    }
    check_f32(conv, &[seq, conv_dim], "conv")?;
    check_f32(beta, &[seq, v_heads], "beta")?;
    check_f32(g, &[seq, v_heads], "g")?;
    check_f32(s, &[v_heads, d_k, d_v], "delta state")?;
    for (name, t) in [("beta", beta), ("g", g), ("delta state", s)] {
        if !conv.device().same_device(t.device()) {
            bail!("{name} must live on the same Metal device as conv");
        }
    }
    Ok((seq, conv_dim, v_heads))
}

/// The delta-rule recurrence: ONE scan dispatch advances the whole
/// `[v_heads, head_dim, head_dim]` f32 state across all `seq` timesteps. `conv`
/// is `[seq, conv_dim]` (q | k | v fused, silu'd), `beta` and `g` are
/// `[seq, v_heads]` (g the LOG decay), `s` the incoming state. Returns the
/// per-token output `[seq, v_heads, head_dim]` and a
/// `[planes, v_heads, head_dim, head_dim]` state output, leaving `s` untouched
/// so a caller holding it for a rollback trail still sees the value it passed
/// in.
///
/// The state planes are MOST-RECENT-FIRST: plane p is the state after token
/// `seq - 1 - p`, so plane 0 is always the state after the last token and
/// `planes == 1` is the plain scan. `planes == seq` is the rollback trail a
/// speculative verify walk needs — llama.cpp's snapshot-slot ordering, kept so
/// the two are readable against each other.
///
/// Three kernels, one contract. Every length takes `kernel_delta_scan` by
/// default, which normalizes q and k in its own load stage. Two opt-in arms
/// exist and neither is faster: `XWEN_DELTA_DECODE_KERNEL` sends a ONE-TOKEN
/// chunk to `kernel_delta_scan_decode`, the same recurrence with the timestep
/// loop gone, and `XWEN_DELTA_SCAN_V2` selects `run_delta_l2norm` followed by
/// `kernel_delta_scan_v2` at every length. All three are bounded against the
/// reference scan and differ only in the order the key-dim contractions are
/// summed.
pub(crate) fn run_delta_scan(
    conv: &Tensor,
    beta: &Tensor,
    g: &Tensor,
    s: &Tensor,
    k_heads: usize,
    eps: f32,
    planes: usize,
) -> Result<(Tensor, Tensor)> {
    if crate::ops::delta_scan_v2() {
        return run_delta_scan_v2(conv, beta, g, s, k_heads, eps, planes);
    }
    // A one-token chunk goes to the decode kernel only when the opt-in switch
    // asks for it (it is a measured wash — `ops::delta_decode_kernel`).
    // `planes` is 1 there by construction (1..=seq), and the float4 state
    // pointer needs a 16-byte buffer offset — every state a cache hands over
    // starts at 0, and a head-aligned test view is 64 KiB in, but a view that
    // is neither falls back rather than reading misaligned.
    let decode = crate::ops::delta_decode_kernel()
        && matches!(conv.dims2(), Ok((1, _)))
        && f32_off(s.layout()).is_multiple_of(16);
    if decode {
        return run_delta_scan_decode(conv, beta, g, s, k_heads, eps, planes);
    }
    run_delta_scan_default(conv, beta, g, s, k_heads, eps, planes)
}

/// `XWEN_DELTA_SCAN_V2`: the q/k norm as its own dispatch, then
/// `kernel_delta_scan_v2` — one simdgroup per (V-head, state value-column). See
/// `run_delta_scan`, and `ops::delta_scan_v2` for why this is not the default.
fn run_delta_scan_v2(
    conv: &Tensor,
    beta: &Tensor,
    g: &Tensor,
    s: &Tensor,
    k_heads: usize,
    eps: f32,
    planes: usize,
) -> Result<(Tensor, Tensor)> {
    let cdev = conv.device().clone();
    let Device::Metal(mdev) = &cdev else {
        bail!("delta_scan requires conv on a Metal device");
    };
    let (seq, conv_dim, v_heads) = check_delta_scan_operands(conv, beta, g, s, k_heads, planes)?;

    let out_len = checked_elems(&[seq, v_heads, DELTA_HEAD_DIM], "delta_scan out")?;
    let state_len = checked_elems(
        &[planes, v_heads, DELTA_HEAD_DIM, DELTA_HEAD_DIM],
        "delta_scan state",
    )?;
    glue_index_fits_i32(checked_elems(&[seq, conv_dim], "delta_scan conv")?)?;
    glue_index_fits_i32(out_len)?;

    let qk = run_delta_l2norm(conv, k_heads, eps)?;

    let threads = DELTA_SIMD_WIDTH * DELTA_V2_SGS;
    let pipeline = pipelines::delta_pipeline(mdev.device(), "kernel_delta_scan_v2")?;
    if pipeline.max_total_threads_per_threadgroup() < threads {
        bail!(
            "delta_scan needs {threads} threads per threadgroup, the pipeline allows {}",
            pipeline.max_total_threads_per_threadgroup()
        );
    }
    check_delta_simd_width(&pipeline, "kernel_delta_scan_v2")?;
    let out = mdev.new_buffer(out_len, DType::F32, "delta_scan_out")?;
    let s_out = mdev.new_buffer(state_len, DType::F32, "delta_scan_state")?;

    let (qk_guard, qk_layout) = qk.storage_and_layout();
    let Storage::Metal(qk_storage) = &*qk_guard else {
        bail!("normalized q|k is not on a Metal device");
    };
    let (conv_guard, conv_layout) = conv.storage_and_layout();
    let Storage::Metal(conv_storage) = &*conv_guard else {
        bail!("conv is not on a Metal device");
    };
    let (beta_guard, beta_layout) = beta.storage_and_layout();
    let Storage::Metal(beta_storage) = &*beta_guard else {
        bail!("beta is not on a Metal device");
    };
    let (g_guard, g_layout) = g.storage_and_layout();
    let Storage::Metal(g_storage) = &*g_guard else {
        bail!("g is not on a Metal device");
    };
    let (s_guard, s_layout) = s.storage_and_layout();
    let Storage::Metal(s_storage) = &*s_guard else {
        bail!("delta state is not on a Metal device");
    };

    let args = DeltaScanV2Args {
        seq: seq as i32,
        k_heads: k_heads as i32,
        v_heads: v_heads as i32,
        conv_dim: conv_dim as i32,
        n_planes: planes as i32,
        scale: 1.0 / (DELTA_HEAD_DIM as f32).sqrt(),
    };
    {
        let cmd = mdev.command_encoder()?;
        let ep = &cmd;
        let encoder = ep.encoder();
        let encoder: &ComputeCommandEncoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_bytes(0, &args);
        encoder.set_input_buffer(1, Some(qk_storage.buffer()), f32_off(qk_layout));
        encoder.set_input_buffer(2, Some(conv_storage.buffer()), f32_off(conv_layout));
        encoder.set_input_buffer(3, Some(beta_storage.buffer()), f32_off(beta_layout));
        encoder.set_input_buffer(4, Some(g_storage.buffer()), f32_off(g_layout));
        encoder.set_input_buffer(5, Some(s_storage.buffer()), f32_off(s_layout));
        encoder.set_output_buffer(6, Some(&out), 0);
        encoder.set_output_buffer(7, Some(&s_out), 0);
        encoder.dispatch_thread_groups(
            mtl_size(DELTA_V2_COL_TGS, v_heads, 1),
            mtl_size(DELTA_SIMD_WIDTH, DELTA_V2_SGS, 1),
        );
    }
    drop(qk_guard);
    drop(conv_guard);
    drop(beta_guard);
    drop(g_guard);
    drop(s_guard);

    Ok((
        output_tensor(out, mdev, out_len, (seq, v_heads, DELTA_HEAD_DIM)),
        output_tensor(
            s_out,
            mdev,
            state_len,
            (planes, v_heads, DELTA_HEAD_DIM, DELTA_HEAD_DIM),
        ),
    ))
}

/// The DECODE arm: `kernel_delta_scan_decode` — one token, no timestep loop,
/// the state read and written as float4 with the row-slice folds done inside a
/// simdgroup. Same operands, same outputs and the same single state plane as
/// `run_delta_scan_default` at `seq == 1`. OPT-IN and a measured wash — see
/// `ops::delta_decode_kernel`, which is the only thing that selects it.
///
/// Preconditions the caller has already established: `seq == 1` (so `planes` is
/// 1, the only plane count a one-token chunk can name) and a state view whose
/// byte offset is float4-aligned.
pub(crate) fn run_delta_scan_decode(
    conv: &Tensor,
    beta: &Tensor,
    g: &Tensor,
    s: &Tensor,
    k_heads: usize,
    eps: f32,
    planes: usize,
) -> Result<(Tensor, Tensor)> {
    let cdev = conv.device().clone();
    let Device::Metal(mdev) = &cdev else {
        bail!("delta_scan requires conv on a Metal device");
    };
    let (seq, conv_dim, v_heads) = check_delta_scan_operands(conv, beta, g, s, k_heads, planes)?;
    let (d_k, d_v) = (DELTA_HEAD_DIM, DELTA_HEAD_DIM);
    if seq != 1 || planes != 1 {
        bail!("delta_scan_decode is the seq == 1 kernel, got seq {seq} with {planes} planes");
    }
    let s_off = f32_off(s.layout());
    if !s_off.is_multiple_of(16) {
        bail!(
            "delta_scan_decode reads the state as float4; its byte offset {s_off} is not 16-byte aligned"
        );
    }

    let out_len = checked_elems(&[seq, v_heads, DELTA_HEAD_DIM], "delta_scan out")?;
    let state_len = checked_elems(&[planes, v_heads, d_k, d_v], "delta_scan state")?;
    glue_index_fits_i32(checked_elems(&[seq, conv_dim], "delta_scan conv")?)?;
    glue_index_fits_i32(out_len)?;

    let pipeline = pipelines::delta_pipeline(mdev.device(), "kernel_delta_scan_decode")?;
    if pipeline.max_total_threads_per_threadgroup() < DELTA_HEAD_DIM {
        bail!(
            "delta_scan needs {DELTA_HEAD_DIM} threads per threadgroup, the pipeline allows {}",
            pipeline.max_total_threads_per_threadgroup()
        );
    }
    check_delta_simd_width(&pipeline, "kernel_delta_scan_decode")?;
    let out = mdev.new_buffer(out_len, DType::F32, "delta_scan_out")?;
    let s_out = mdev.new_buffer(state_len, DType::F32, "delta_scan_state")?;

    let (conv_guard, conv_layout) = conv.storage_and_layout();
    let Storage::Metal(conv_storage) = &*conv_guard else {
        bail!("conv is not on a Metal device");
    };
    let (beta_guard, beta_layout) = beta.storage_and_layout();
    let Storage::Metal(beta_storage) = &*beta_guard else {
        bail!("beta is not on a Metal device");
    };
    let (g_guard, g_layout) = g.storage_and_layout();
    let Storage::Metal(g_storage) = &*g_guard else {
        bail!("g is not on a Metal device");
    };
    let (s_guard, s_layout) = s.storage_and_layout();
    let Storage::Metal(s_storage) = &*s_guard else {
        bail!("delta state is not on a Metal device");
    };

    let args = DeltaScanArgs {
        seq: seq as i32,
        k_heads: k_heads as i32,
        v_heads: v_heads as i32,
        conv_dim: conv_dim as i32,
        n_planes: planes as i32,
        scale: 1.0 / (DELTA_HEAD_DIM as f32).sqrt(),
        eps,
    };
    {
        let cmd = mdev.command_encoder()?;
        let ep = &cmd;
        let encoder = ep.encoder();
        let encoder: &ComputeCommandEncoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_bytes(0, &args);
        encoder.set_input_buffer(1, Some(conv_storage.buffer()), f32_off(conv_layout));
        encoder.set_input_buffer(2, Some(beta_storage.buffer()), f32_off(beta_layout));
        encoder.set_input_buffer(3, Some(g_storage.buffer()), f32_off(g_layout));
        encoder.set_input_buffer(4, Some(s_storage.buffer()), f32_off(s_layout));
        encoder.set_output_buffer(5, Some(&out), 0);
        encoder.set_output_buffer(6, Some(&s_out), 0);
        encoder.dispatch_thread_groups(
            mtl_size(v_heads * DELTA_DEC_COL_BLOCKS, 1, 1),
            mtl_size(DELTA_HEAD_DIM, 1, 1),
        );
    }
    drop(conv_guard);
    drop(beta_guard);
    drop(g_guard);
    drop(s_guard);

    Ok((
        output_tensor(out, mdev, out_len, (seq, v_heads, DELTA_HEAD_DIM)),
        output_tensor(s_out, mdev, state_len, (planes, v_heads, d_k, d_v)),
    ))
}

/// The shipped path: `kernel_delta_scan` alone — a threadgroup per head per
/// value-column block, with the q/k L2 clamp-norm folded into its load stage, so
/// the whole recurrence is one dispatch. See `run_delta_scan`.
pub(crate) fn run_delta_scan_default(
    conv: &Tensor,
    beta: &Tensor,
    g: &Tensor,
    s: &Tensor,
    k_heads: usize,
    eps: f32,
    planes: usize,
) -> Result<(Tensor, Tensor)> {
    let cdev = conv.device().clone();
    let Device::Metal(mdev) = &cdev else {
        bail!("delta_scan requires conv on a Metal device");
    };
    let (seq, conv_dim, v_heads) = check_delta_scan_operands(conv, beta, g, s, k_heads, planes)?;
    let (d_k, d_v) = (DELTA_HEAD_DIM, DELTA_HEAD_DIM);

    let out_len = checked_elems(&[seq, v_heads, DELTA_HEAD_DIM], "delta_scan out")?;
    let state_len = checked_elems(&[planes, v_heads, d_k, d_v], "delta_scan state")?;
    glue_index_fits_i32(checked_elems(&[seq, conv_dim], "delta_scan conv")?)?;
    glue_index_fits_i32(out_len)?;

    let pipeline = pipelines::delta_pipeline(mdev.device(), "kernel_delta_scan")?;
    if pipeline.max_total_threads_per_threadgroup() < DELTA_HEAD_DIM {
        bail!(
            "delta_scan needs {DELTA_HEAD_DIM} threads per threadgroup, the pipeline allows {}",
            pipeline.max_total_threads_per_threadgroup()
        );
    }
    check_delta_simd_width(&pipeline, "kernel_delta_scan")?;
    let out = mdev.new_buffer(out_len, DType::F32, "delta_scan_out")?;
    let s_out = mdev.new_buffer(state_len, DType::F32, "delta_scan_state")?;

    let (conv_guard, conv_layout) = conv.storage_and_layout();
    let Storage::Metal(conv_storage) = &*conv_guard else {
        bail!("conv is not on a Metal device");
    };
    let (beta_guard, beta_layout) = beta.storage_and_layout();
    let Storage::Metal(beta_storage) = &*beta_guard else {
        bail!("beta is not on a Metal device");
    };
    let (g_guard, g_layout) = g.storage_and_layout();
    let Storage::Metal(g_storage) = &*g_guard else {
        bail!("g is not on a Metal device");
    };
    let (s_guard, s_layout) = s.storage_and_layout();
    let Storage::Metal(s_storage) = &*s_guard else {
        bail!("delta state is not on a Metal device");
    };

    let args = DeltaScanArgs {
        seq: seq as i32,
        k_heads: k_heads as i32,
        v_heads: v_heads as i32,
        conv_dim: conv_dim as i32,
        n_planes: planes as i32,
        scale: 1.0 / (DELTA_HEAD_DIM as f32).sqrt(),
        eps,
    };
    {
        let cmd = mdev.command_encoder()?;
        let ep = &cmd;
        let encoder = ep.encoder();
        let encoder: &ComputeCommandEncoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_bytes(0, &args);
        encoder.set_input_buffer(1, Some(conv_storage.buffer()), f32_off(conv_layout));
        encoder.set_input_buffer(2, Some(beta_storage.buffer()), f32_off(beta_layout));
        encoder.set_input_buffer(3, Some(g_storage.buffer()), f32_off(g_layout));
        encoder.set_input_buffer(4, Some(s_storage.buffer()), f32_off(s_layout));
        encoder.set_output_buffer(5, Some(&out), 0);
        encoder.set_output_buffer(6, Some(&s_out), 0);
        encoder.dispatch_thread_groups(
            mtl_size(v_heads * DELTA_COL_BLOCKS, 1, 1),
            mtl_size(DELTA_HEAD_DIM, 1, 1),
        );
    }
    drop(conv_guard);
    drop(beta_guard);
    drop(g_guard);
    drop(s_guard);

    Ok((
        output_tensor(out, mdev, out_len, (seq, v_heads, DELTA_HEAD_DIM)),
        output_tensor(s_out, mdev, state_len, (planes, v_heads, d_k, d_v)),
    ))
}

/// The flash prefill kernel's fixed tile geometry (flash.metal): BQ=32 query
/// rows per block, BK=16 key columns per block, head_dim locked at 128.
const FLASH_BQ: usize = 32;
const FLASH_BK: usize = 16;
const FLASH_BD: usize = 128;

/// Vendored flash-attention prefill against the `kernel_flash_attn_*` family
/// (flash.metal — the modified copy of candle's MLX steel attention kernel).
/// `q` is `[n_head, seq, 128]` f32 contiguous (the rope output); `k`/`v` are
/// `[n_kv, K, 128]` f16 cache views that may be HEAD-STRIDED (rows within a
/// head contiguous, head_dim stride 1; the head axis may carry the cache's
/// max_ctx gap — passed to the kernel as explicit strides, never forced
/// contiguous). Masking runs in-kernel: query row i (absolute `pos + i`) sees
/// key column j (absolute `k_off + j`) iff it is not future and within
/// `window` (None = full attention) — exactly `kv_cache::attn_mask_for`'s
/// rule. Returns `[n_head, seq, 128]` f32 contiguous. `disable_skip` defeats
/// the block-level skip bounds (test-only; the skip is exact — see flash.metal).
pub(crate) fn run_flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    pos: usize,
    k_off: usize,
    window: Option<usize>,
    scale: f32,
    disable_skip: bool,
) -> Result<Tensor> {
    let cdev = q.device().clone();
    let Device::Metal(mdev) = &cdev else {
        bail!("flash_attn requires q on a Metal device");
    };

    let (n_head, seq, head_dim) = q
        .dims3()
        .map_err(|e| anyhow::anyhow!("q must be rank-3 [n_head, seq, head_dim]: {e}"))?;
    if q.dtype() != DType::F32 {
        bail!("q must be f32, got {:?}", q.dtype());
    }
    if !q.is_contiguous() {
        bail!("q must be contiguous");
    }
    if head_dim != FLASH_BD {
        bail!("flash_attn is compiled for head_dim {FLASH_BD}, got {head_dim}");
    }
    if seq == 0 {
        bail!("flash_attn requires at least one query row");
    }

    let (n_kv, k_len, _) = check_flash_kv(k, "k", head_dim, &cdev)?;
    let (n_kv_v, v_len, _) = check_flash_kv(v, "v", head_dim, &cdev)?;
    if (n_kv_v, v_len) != (n_kv, k_len) {
        bail!("k [{n_kv}, {k_len}] and v [{n_kv_v}, {v_len}] disagree on shape");
    }
    if n_kv == 0 || !n_head.is_multiple_of(n_kv) {
        bail!("n_head ({n_head}) must be a positive multiple of n_kv ({n_kv})");
    }

    // The mask semantics require every query's own key in range: row i's own
    // key sits at column `pos + i - k_off`, which must lie in [0, K). A row
    // with NO visible key would divide 0/0 in the softmax normalizer.
    if k_off > pos {
        bail!("k_off ({k_off}) exceeds pos ({pos}): query rows before the key range");
    }
    let q_end = pos
        .checked_add(seq)
        .ok_or_else(|| anyhow::anyhow!("pos + seq ({pos} + {seq}) overflows usize"))?;
    let k_end = k_off
        .checked_add(k_len)
        .ok_or_else(|| anyhow::anyhow!("k_off + K ({k_off} + {k_len}) overflows usize"))?;
    if q_end > k_end {
        bail!(
            "query rows reach absolute position {q_end} but keys end at {k_end}: \
             each query's own key must be present"
        );
    }
    // The kernel does its position math in i32.
    for (what, val) in [("pos + seq", q_end), ("k_off + K", k_end)] {
        if i32::try_from(val).is_err() {
            bail!("flash_attn {what} ({val}) overflows the kernel's i32 position math");
        }
    }
    let window = match window {
        None => i32::MAX,
        Some(0) => bail!("flash_attn window must be >= 1"),
        Some(w) => i32::try_from(w).unwrap_or(i32::MAX),
    };

    let out_count = checked_elems(&[n_head, seq, head_dim], "flash_attn")?;
    let dst = mdev.new_buffer(out_count, DType::F32, "flash_attn")?;

    let (q_guard, q_layout) = q.storage_and_layout();
    let Storage::Metal(q_storage) = &*q_guard else {
        bail!("q is not on a Metal device");
    };
    let (k_guard, k_layout) = k.storage_and_layout();
    let Storage::Metal(k_storage) = &*k_guard else {
        bail!("k is not on a Metal device");
    };
    let (v_guard, v_layout) = v.storage_and_layout();
    let Storage::Metal(v_storage) = &*v_guard else {
        bail!("v is not on a Metal device");
    };

    let nq = seq.div_ceil(FLASH_BQ);
    let nk = k_len.div_ceil(FLASH_BK);
    let align_q = seq.is_multiple_of(FLASH_BQ);
    let align_k = k_len.is_multiple_of(FLASH_BK);
    let name = match (align_q, align_k) {
        (true, true) => "kernel_flash_attn_q1_k1",
        (true, false) => "kernel_flash_attn_q1_k0",
        (false, true) => "kernel_flash_attn_q0_k1",
        (false, false) => "kernel_flash_attn_q0_k0",
    };
    let pipeline = pipelines::flash_pipeline(mdev.device(), name)?;

    let i64_stride = |s: usize, what: &str| -> Result<i64> {
        i64::try_from(s).map_err(|_| anyhow::anyhow!("flash_attn {what} stride {s} overflows i64"))
    };
    let args = FlashAttnArgs {
        gqa_factor: (n_head / n_kv) as i32,
        scale,
        nk: nk as i32,
        nq_aligned: (seq / FLASH_BQ) as i32,
        nk_aligned: (k_len / FLASH_BK) as i32,
        ql_rem: (seq % FLASH_BQ) as i32,
        kl_rem: (k_len % FLASH_BK) as i32,
        kl: k_len as i32,
        q_off: pos as i32,
        k_off: k_off as i32,
        window,
        disable_skip: disable_skip as i32,
        q_stride_h: i64_stride(q_layout.stride()[0], "q head")?,
        q_stride_r: i64_stride(q_layout.stride()[1], "q row")?,
        k_stride_h: i64_stride(k_layout.stride()[0], "k head")?,
        k_stride_r: i64_stride(k_layout.stride()[1], "k row")?,
        v_stride_h: i64_stride(v_layout.stride()[0], "v head")?,
        v_stride_r: i64_stride(v_layout.stride()[1], "v row")?,
        o_stride_h: (seq * head_dim) as i64,
        o_stride_r: head_dim as i64,
    };

    {
        let cmd = mdev.command_encoder()?;
        let ep = &cmd;
        let encoder = ep.encoder();
        let encoder: &ComputeCommandEncoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_bytes(0, &args);
        encoder.set_input_buffer(
            1,
            Some(q_storage.buffer()),
            q_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder.set_input_buffer(
            2,
            Some(k_storage.buffer()),
            k_layout.start_offset() * DType::F16.size_in_bytes(),
        );
        encoder.set_input_buffer(
            3,
            Some(v_storage.buffer()),
            v_layout.start_offset() * DType::F16.size_in_bytes(),
        );
        encoder.set_output_buffer(4, Some(&dst), 0);
        // One threadgroup per (query block, query head); 4 simdgroups.
        let grid = mtl_size(nq, n_head, 1);
        encoder.dispatch_thread_groups(grid, mtl_size(32, 4, 1));
    }
    drop(q_guard);
    drop(k_guard);
    drop(v_guard);

    Ok(output_tensor(dst, mdev, out_count, (n_head, seq, head_dim)))
}

/// Validate one flash k/v cache view: rank-3 `[n_kv, K, head_dim]` f16 with
/// head_dim stride 1 and contiguous rows (stride `head_dim`); the head stride
/// is free (the cache's max_ctx gap). Returns (n_kv, K, head stride).
fn check_flash_kv(
    t: &Tensor,
    what: &str,
    head_dim: usize,
    q_device: &Device,
) -> Result<(usize, usize, usize)> {
    let (n_kv, len, hd) = t
        .dims3()
        .map_err(|e| anyhow::anyhow!("{what} must be rank-3 [n_kv, K, head_dim]: {e}"))?;
    if t.dtype() != DType::F16 {
        bail!("{what} must be f16, got {:?}", t.dtype());
    }
    if hd != head_dim {
        bail!("{what} head_dim {hd} != q head_dim {head_dim}");
    }
    if len == 0 {
        bail!("{what} has no keys");
    }
    let stride = t.layout().stride();
    if stride[2] != 1 || stride[1] != head_dim {
        bail!(
            "{what} must have contiguous rows (strides [_, {head_dim}, 1]), got {:?}",
            stride
        );
    }
    if stride[0] < len * head_dim {
        bail!("{what} head stride {} overlaps its {len} rows", stride[0]);
    }
    if !t.device().same_device(q_device) {
        bail!("{what} must live on the same Metal device as q");
    }
    Ok((n_kv, len, stride[0]))
}

// ---------------------------------------------------------------------------
// QSA decode row gather — see src/ops/qsa_gather.metal.

/// Matches the Metal `qsa_gather_args` struct (src/ops/qsa_gather.metal).
#[repr(C)]
#[derive(Clone, Copy)]
struct QsaGatherArgs {
    heads: i32,
    len: i32,
    head_dim: i32,
    n_sel: i32,
    src_stride_h: i64,
    src_stride_r: i64,
}

/// Pack the rows `rows` (u32 `[n_sel]`) names out of a `[heads, len,
/// head_dim]` cache view (f16 or f32; rows contiguous, head stride free)
/// into a contiguous `[heads, n_sel, head_dim]` tensor of the same dtype,
/// against `kernel_qsa_gather_{f16,f32}`. One threadgroup per (row, head).
pub(crate) fn run_qsa_gather(t: &Tensor, rows: &Tensor) -> Result<Tensor> {
    let cdev = t.device().clone();
    let Device::Metal(mdev) = &cdev else {
        bail!("qsa_gather requires its source on a Metal device");
    };
    let (heads, len, head_dim) = t.dims3().map_err(|e| {
        anyhow::anyhow!("qsa_gather source must be rank-3 [heads, len, head_dim]: {e}")
    })?;
    let name = match t.dtype() {
        DType::F16 => "kernel_qsa_gather_f16",
        DType::F32 => "kernel_qsa_gather_f32",
        other => bail!("qsa_gather has no kernel for {other:?}"),
    };
    if head_dim == 0 || !head_dim.is_multiple_of(4) {
        bail!("qsa_gather head_dim {head_dim} must be a positive multiple of 4");
    }
    let stride = t.layout().stride();
    if stride[2] != 1 || stride[1] != head_dim {
        bail!(
            "qsa_gather source must have contiguous rows (strides [_, {head_dim}, 1]), got {stride:?}"
        );
    }
    if heads > 1 && stride[0] < len * head_dim {
        bail!(
            "qsa_gather source head stride {} overlaps its {len} rows",
            stride[0]
        );
    }
    // The kernel copies through 4-element vector device pointers, which Metal
    // requires aligned to the vector (8 bytes at f16, 16 at f32): rows are
    // (head_dim % 4 == 0 above), so only a view start or a head stride that is
    // not a multiple of 4 elements could break it. The cache views are
    // (start 0, head stride max_ctx * head_dim); a hand-sliced view lands here.
    let start = t.layout().start_offset();
    if !start.is_multiple_of(4) || !stride[0].is_multiple_of(4) {
        bail!(
            "qsa_gather source must start and stride heads at multiples of 4 elements (vector \
             loads), got start {start}, head stride {}",
            stride[0]
        );
    }
    if rows.dtype() != DType::U32 {
        bail!("qsa_gather rows must be u32, got {:?}", rows.dtype());
    }
    if !rows.is_contiguous() {
        bail!("qsa_gather rows must be contiguous");
    }
    let n_sel = rows.dims1()?;
    if heads == 0 || len == 0 || n_sel == 0 {
        bail!(
            "qsa_gather over an empty source or selection ({heads} heads, {len} rows, {n_sel} selected)"
        );
    }
    if !rows.device().same_device(&cdev) {
        bail!("qsa_gather rows must live on the source's Metal device");
    }
    for (v, what) in [
        (heads, "heads"),
        (len, "len"),
        (head_dim, "head_dim"),
        (n_sel, "n_sel"),
    ] {
        if v > i32::MAX as usize {
            bail!("qsa_gather {what} {v} overflows i32");
        }
    }
    let out_count = checked_elems(&[heads, n_sel, head_dim], "qsa_gather")?;
    let i64_stride = |s: usize, what: &str| -> Result<i64> {
        i64::try_from(s).map_err(|_| anyhow::anyhow!("qsa_gather {what} stride {s} overflows i64"))
    };

    let pipeline = pipelines::qsa_gather_pipeline(mdev.device(), name)?;
    let dst = mdev.new_buffer(out_count, t.dtype(), "qsa_gather")?;

    let (t_guard, t_layout) = t.storage_and_layout();
    let Storage::Metal(t_storage) = &*t_guard else {
        bail!("qsa_gather source is not on a Metal device");
    };
    let (r_guard, r_layout) = rows.storage_and_layout();
    let Storage::Metal(r_storage) = &*r_guard else {
        bail!("qsa_gather rows are not on a Metal device");
    };

    let args = QsaGatherArgs {
        heads: heads as i32,
        len: len as i32,
        head_dim: head_dim as i32,
        n_sel: n_sel as i32,
        src_stride_h: i64_stride(stride[0], "head")?,
        src_stride_r: i64_stride(stride[1], "row")?,
    };
    {
        let cmd = mdev.command_encoder()?;
        let ep = &cmd;
        let encoder = ep.encoder();
        let encoder: &ComputeCommandEncoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_bytes(0, &args);
        encoder.set_input_buffer(
            1,
            Some(t_storage.buffer()),
            t_layout.start_offset() * t.dtype().size_in_bytes(),
        );
        encoder.set_input_buffer(
            2,
            Some(r_storage.buffer()),
            r_layout.start_offset() * DType::U32.size_in_bytes(),
        );
        encoder.set_output_buffer(3, Some(&dst), 0);
        // One threadgroup per (selected row, head); a thread per 4-wide vector
        // of the row, capped at the pipeline's width (the kernel strides).
        let width = pipeline
            .max_total_threads_per_threadgroup()
            .min(256)
            .min(head_dim / 4)
            .max(1);
        encoder.dispatch_thread_groups(mtl_size(n_sel, heads, 1), mtl_size(width, 1, 1));
    }
    drop(t_guard);
    drop(r_guard);

    let storage = MetalStorage::new(dst, mdev.clone(), out_count, t.dtype());
    Ok(Tensor::from_storage(
        Storage::Metal(storage),
        (heads, n_sel, head_dim),
        candle_core::op::BackpropOp::none(),
        false,
    ))
}

// ---------------------------------------------------------------------------
// QSA decode block selection — see src/ops/qsa_select.metal.

/// Matches the Metal `qsa_select_args` struct (src/ops/qsa_select.metal).
#[repr(C)]
#[derive(Clone, Copy)]
struct QsaSelectArgs {
    nb: i32,
    keep: i32,
    ratio: i32,
    tail: i32,
}

/// The threadgroup width `kernel_qsa_select` is written for: one threadgroup
/// per call, every thread owning a contiguous stripe of the `nb` scores. The
/// kernel sizes its per-simdgroup scratch for this many threads.
const QSA_SELECT_MAX_THREADS: usize = 1024;

/// Select the top-`keep` of the `nb` block scores `scores` (f32 `[nb]`) and
/// expand them, plus the `tail` positions above the last complete block, into
/// the ascending u32 row list `[keep * ratio + tail]` — `QsaIndexer::top_blocks`
/// + `expand_into` on device, against `kernel_qsa_select`. One threadgroup.
pub(crate) fn run_qsa_select(
    scores: &Tensor,
    keep: usize,
    ratio: usize,
    tail: usize,
) -> Result<Tensor> {
    let cdev = scores.device().clone();
    let Device::Metal(mdev) = &cdev else {
        bail!("qsa_select requires its scores on a Metal device");
    };
    if scores.dtype() != DType::F32 {
        bail!("qsa_select scores must be f32, got {:?}", scores.dtype());
    }
    if !scores.is_contiguous() {
        bail!("qsa_select scores must be contiguous");
    }
    let nb = scores.dims1()?;
    if nb == 0 {
        bail!("qsa_select over no blocks");
    }
    if keep == 0 || keep > nb {
        bail!("qsa_select keep {keep} must be in 1..={nb}");
    }
    if ratio == 0 || tail >= ratio {
        bail!("qsa_select tail {tail} must be below the block ratio {ratio}");
    }
    for (v, what) in [(nb, "nb"), (keep, "keep"), (ratio, "ratio"), (tail, "tail")] {
        if v > i32::MAX as usize {
            bail!("qsa_select {what} {v} overflows i32");
        }
    }
    // Row indices are u32 on the wire (the gather reads them as such).
    let n_sel = checked_elems(&[keep, ratio], "qsa_select")? + tail;
    if checked_elems(&[nb, ratio], "qsa_select")? + tail > u32::MAX as usize {
        bail!("qsa_select positions overflow u32");
    }

    let pipeline = pipelines::qsa_select_pipeline(mdev.device(), "kernel_qsa_select")?;
    // The scans are built from simdgroup prefix sums, so the width must be
    // whole simdgroups; a pipeline that cannot run 32 threads has no width
    // this kernel can use.
    let width = pipeline
        .max_total_threads_per_threadgroup()
        .min(QSA_SELECT_MAX_THREADS);
    if width < 32 {
        bail!("qsa_select pipeline admits only {width} threads per threadgroup");
    }
    let width = width - width % 32;
    let dst = mdev.new_buffer(n_sel, DType::U32, "qsa_select")?;

    let (s_guard, s_layout) = scores.storage_and_layout();
    let Storage::Metal(s_storage) = &*s_guard else {
        bail!("qsa_select scores are not on a Metal device");
    };
    let args = QsaSelectArgs {
        nb: nb as i32,
        keep: keep as i32,
        ratio: ratio as i32,
        tail: tail as i32,
    };
    {
        let cmd = mdev.command_encoder()?;
        let ep = &cmd;
        let encoder = ep.encoder();
        let encoder: &ComputeCommandEncoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_bytes(0, &args);
        encoder.set_input_buffer(
            1,
            Some(s_storage.buffer()),
            s_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder.set_output_buffer(2, Some(&dst), 0);
        encoder.dispatch_thread_groups(mtl_size(1, 1, 1), mtl_size(width, 1, 1));
    }
    drop(s_guard);

    let storage = MetalStorage::new(dst, mdev.clone(), n_sel, DType::U32);
    Ok(Tensor::from_storage(
        Storage::Metal(storage),
        n_sel,
        candle_core::op::BackpropOp::none(),
        false,
    ))
}

/// Matches the Metal `qsa_select_mask_args` struct (src/ops/qsa_select.metal).
#[repr(C)]
#[derive(Clone, Copy)]
struct QsaSelectMaskArgs {
    n_blocks: i32,
    n_kv: i32,
    pos: i32,
    ratio: i32,
    keep_max: i32,
}

/// The prefill overlay of one chunk on device: for each of the `n` queries of
/// `scores` (f32 `[n, n_blocks]`, query `i` at absolute position `pos + i`),
/// the top-`min(keep_max, nb_i)` of its first `nb_i = min((pos + i + 1) /
/// ratio, n_blocks)` block scores, expanded into an additive f32 mask row
/// `[pos + n]` — `-inf` everywhere, `0` at the selected blocks' positions and
/// at the query's raw tail — against `kernel_qsa_select_mask`. Returns the
/// `[n, pos + n]` plane, the same bits `QsaIndexer::top_blocks` +
/// `expand_into` + the host fill would produce. One threadgroup per query.
pub(crate) fn run_qsa_select_mask(
    scores: &Tensor,
    pos: usize,
    ratio: usize,
    keep_max: usize,
) -> Result<Tensor> {
    let cdev = scores.device().clone();
    let Device::Metal(mdev) = &cdev else {
        bail!("qsa_select_mask requires its scores on a Metal device");
    };
    if scores.dtype() != DType::F32 {
        bail!(
            "qsa_select_mask scores must be f32, got {:?}",
            scores.dtype()
        );
    }
    if !scores.is_contiguous() {
        bail!("qsa_select_mask scores must be contiguous");
    }
    let (n, n_blocks) = scores.dims2()?;
    if n == 0 || n_blocks == 0 {
        bail!("qsa_select_mask over an empty [{n}, {n_blocks}] score plane");
    }
    if ratio == 0 {
        bail!("qsa_select_mask block ratio must be nonzero");
    }
    let Some(n_kv) = pos.checked_add(n) else {
        bail!("qsa_select_mask position {pos} plus {n} queries overflows");
    };
    if checked_elems(&[n_blocks, ratio], "qsa_select_mask")? > n_kv {
        bail!("qsa_select_mask scores {n_blocks} blocks of {ratio}, beyond the {n_kv} positions");
    }
    for (v, what) in [
        (n_blocks, "n_blocks"),
        (n_kv, "n_kv"),
        (pos, "pos"),
        (ratio, "ratio"),
        (keep_max, "keep_max"),
    ] {
        if v > i32::MAX as usize {
            bail!("qsa_select_mask {what} {v} overflows i32");
        }
    }
    let n_elems = checked_elems(&[n, n_kv], "qsa_select_mask")?;

    let pipeline = pipelines::qsa_select_pipeline(mdev.device(), "kernel_qsa_select_mask")?;
    let width = pipeline
        .max_total_threads_per_threadgroup()
        .min(QSA_SELECT_MAX_THREADS);
    if width < 32 {
        bail!("qsa_select_mask pipeline admits only {width} threads per threadgroup");
    }
    let width = width - width % 32;
    // Unlike the host arm's `Tensor::from_vec` (an exact-size buffer the pool
    // never hands out again), this goes through candle's allocator, which
    // rounds to a power of two and reuses any free buffer at least this
    // large — so a prefill walk's ever-different chunk shapes share buffers.
    let dst = mdev.new_buffer(n_elems, DType::F32, "qsa_select_mask")?;

    let (s_guard, s_layout) = scores.storage_and_layout();
    let Storage::Metal(s_storage) = &*s_guard else {
        bail!("qsa_select_mask scores are not on a Metal device");
    };
    let args = QsaSelectMaskArgs {
        n_blocks: n_blocks as i32,
        n_kv: n_kv as i32,
        pos: pos as i32,
        ratio: ratio as i32,
        keep_max: keep_max as i32,
    };
    {
        let cmd = mdev.command_encoder()?;
        let ep = &cmd;
        let encoder = ep.encoder();
        let encoder: &ComputeCommandEncoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_bytes(0, &args);
        encoder.set_input_buffer(
            1,
            Some(s_storage.buffer()),
            s_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder.set_output_buffer(2, Some(&dst), 0);
        encoder.dispatch_thread_groups(mtl_size(n, 1, 1), mtl_size(width, 1, 1));
    }
    drop(s_guard);

    let storage = MetalStorage::new(dst, mdev.clone(), n_elems, DType::F32);
    Ok(Tensor::from_storage(
        Storage::Metal(storage),
        (n, n_kv),
        candle_core::op::BackpropOp::none(),
        false,
    ))
}

// ---------------------------------------------------------------------------
// Hyper-connections (qwen4exp carrier read/write) — see src/ops/hc.metal.
// ---------------------------------------------------------------------------

/// The widest carrier `hc.metal` sizes its per-thread injection accumulators for
/// (`HC_MAX_STREAMS` there). The shipped geometry is 4; the two numbers are
/// spelled out in both languages, so keep them in step.
pub(crate) const HC_MAX_STREAMS: usize = 8;

/// Threads per threadgroup in `kernel_hc_gate_down` — five simdgroups, and a
/// divisor of the 320 q8_0 blocks a production carrier row holds.
pub(crate) const HC_GATE_THREADS: usize = 160;

/// q8_0 blocks of the carrier ONE thread of `kernel_hc_gate_down` stages in
/// registers. It bounds the kernel's register footprint (32 floats per block),
/// so a carrier wider than `HC_GATE_THREADS * HC_GATE_MAX_BLK_PER_THREAD` blocks
/// is refused rather than run at whatever occupancy it would spill to.
pub(crate) const HC_GATE_MAX_BLK_PER_THREAD: usize = 2;

/// Down-projection output rows one threadgroup of `kernel_hc_gate_down`
/// computes, sharing one pass over the staged carrier. The per-row accumulators
/// are registers, which is what bounds it.
pub(crate) const HC_GATE_ROWS_PER_TG: usize = 8;

/// Threads per threadgroup in `kernel_hc_gate_up_mix`: `hc_count` adjacent lanes
/// take the streams of one carrier column, so a threadgroup covers
/// `HC_GATE_MIX_THREADS / hc_count` columns.
pub(crate) const HC_GATE_MIX_THREADS: usize = 256;

/// The widest bottleneck `kernel_hc_gate_up_mix` stages in threadgroup memory
/// (one f32 per `low_rank` column, read by every thread of the threadgroup).
pub(crate) const HC_GATE_MAX_LOW_RANK: usize = 1024;

/// Whether the fused decode gate covers this geometry and weight dtype. Every
/// bound is a kernel's, and a gate outside them keeps the seven-dispatch split
/// path rather than failing:
///
/// * both bottleneck projections q8_0 — the two gate kernels read the GGUF
///   block layout directly rather than through `QMatMul`;
/// * `hc_count` a power of two dividing [`HC_GATE_MIX_THREADS`], and at most
///   [`HC_MAX_STREAMS`], so the mix's stream fold is a simd_shuffle butterfly
///   over adjacent lanes of one simdgroup;
/// * `hidden` a whole number of q8_0 blocks, so each block of the carrier
///   belongs to exactly one stream and the per-stream statistics stay disjoint;
/// * the carrier a whole number of `HC_GATE_THREADS`-block passes, at most
///   [`HC_GATE_MAX_BLK_PER_THREAD`] of them — the register bound above;
/// * `low_rank` a whole number of blocks (the up projection's k) and no wider
///   than [`HC_GATE_MAX_LOW_RANK`] — the threadgroup-memory bound above.
pub(crate) fn hc_gate_fused_supported(
    hc_count: usize,
    hidden: usize,
    low_rank: usize,
    dtype: GgmlDType,
) -> bool {
    if dtype != GgmlDType::Q8_0 {
        return false;
    }
    let block = GgmlDType::Q8_0.block_size();
    if hc_count == 0
        || hc_count > HC_MAX_STREAMS
        || !hc_count.is_power_of_two()
        || !HC_GATE_MIX_THREADS.is_multiple_of(hc_count)
    {
        return false;
    }
    if hidden == 0 || !hidden.is_multiple_of(block) {
        return false;
    }
    if low_rank == 0 || !low_rank.is_multiple_of(block) || low_rank > HC_GATE_MAX_LOW_RANK {
        return false;
    }
    let Some(width) = hc_count.checked_mul(hidden) else {
        return false;
    };
    let nblk = width / block;
    nblk.is_multiple_of(HC_GATE_THREADS) && nblk / HC_GATE_THREADS <= HC_GATE_MAX_BLK_PER_THREAD
}

/// Threadgroup width for `kernel_hc_norm`: the largest multiple of the simd
/// width up to 256 that DIVIDES `hidden`. Dividing is what keeps each thread's
/// strided walk inside a single stream, which is what makes the `hc_count`
/// sum-of-squares reductions disjoint; the multiple-of-32 keeps `simd_sum`
/// folding over full simdgroups. `None` when no such width exists — the caller
/// falls back to the candle chain rather than failing.
fn hc_norm_threads(hidden: usize) -> Option<usize> {
    if hidden == 0 {
        // Every width divides zero; a zero-wide stream has no launch at all.
        return None;
    }
    (1..=8)
        .rev()
        .map(|m| m * DELTA_SIMD_WIDTH)
        .find(|w| hidden.is_multiple_of(*w))
}

/// Whether the fused hyper-connection read kernels cover this geometry. The
/// bounds are the kernel's, so a gate outside them keeps the candle chain
/// instead of failing (the tiny fixture geometries are inside it; a `hidden`
/// under 32, or not a multiple of it, is not).
pub(crate) fn hc_norm_supported(hc_count: usize, hidden: usize) -> bool {
    hc_count > 0 && hc_count <= HC_MAX_STREAMS && hc_norm_threads(hidden).is_some()
}

/// Matches the Metal `hc_norm_args` struct (src/ops/hc.metal).
#[repr(C)]
#[derive(Clone, Copy)]
struct HcNormArgs {
    hc_count: i32,
    hidden: i32,
    width: i32,
    eps: f32,
    inv_hc: f32,
}

/// Matches the Metal `hc_silu_args` struct (src/ops/hc.metal).
#[repr(C)]
#[derive(Clone, Copy)]
struct HcSiluArgs {
    n: i32,
    scale: f32,
}

/// Matches the Metal `hc_mix_args` struct (src/ops/hc.metal).
#[repr(C)]
#[derive(Clone, Copy)]
struct HcMixArgs {
    n: i32,
    hc_count: i32,
    hidden: i32,
    inv_hc: f32,
}

/// Matches the Metal `hc_write_args` struct (src/ops/hc.metal).
#[repr(C)]
#[derive(Clone, Copy)]
struct HcWriteArgs {
    n: i32,
    hc_count: i32,
    hidden: i32,
}

/// The carrier's grouped RMS norm — per-stream statistics, FULL-width weight —
/// plus, when `inject_w` is present, the injection head's `hc_count` full-row
/// dot products and their `2*sigmoid(./hc_count)`.
///
/// `stream` is the raw carrier `[n, hc_count * hidden]` f32, `norm_w` the
/// `[hc_count * hidden]` multiply-ready norm weight, `inject_w` the dense
/// `[hc_count, hc_count * hidden]` head (`None` on the tail mixer). Returns the
/// normed carrier `[n, hc_count * hidden]` and, with a head, the write
/// strengths `[n, hc_count]`.
///
/// TWO launch shapes over identical arithmetic, chosen by token count:
/// `kernel_hc_norm[_inject]` puts one threadgroup on a whole token and folds
/// every reduction there, which fills the machine only when the token grid is
/// wide; the `kernel_hc_norm_split` + `kernel_hc_inject` pair spreads the same
/// work over `n * hc_count` threadgroups, which is what a decode step (n = 1)
/// needs. [`run_hc_norm_with`] is the switch.
pub(crate) fn run_hc_norm(
    stream: &Tensor,
    norm_w: &Tensor,
    inject_w: Option<&Tensor>,
    hc_count: usize,
    hidden: usize,
    eps: f32,
) -> Result<(Tensor, Option<Tensor>)> {
    run_hc_norm_with(stream, norm_w, inject_w, hc_count, hidden, eps, None)
}

/// [`run_hc_norm`] with the launch shape pinned: `Some(true)` forces the split
/// pair, `Some(false)` the single-threadgroup kernel, `None` picks by token
/// count (`n < crate::ops::hc_split_max_n()` takes the split pair).
///
/// Only the tests pin it. The two arms are required to agree BIT-FOR-BIT —
/// same thread count, same per-thread strided partition, so every reduction
/// folds in the same association order — and `split_matches_single_bitwise` is
/// what holds them to that.
pub(crate) fn run_hc_norm_with(
    stream: &Tensor,
    norm_w: &Tensor,
    inject_w: Option<&Tensor>,
    hc_count: usize,
    hidden: usize,
    eps: f32,
    split: Option<bool>,
) -> Result<(Tensor, Option<Tensor>)> {
    let cdev = stream.device().clone();
    let Device::Metal(mdev) = &cdev else {
        bail!("hc_norm requires the carrier on a Metal device");
    };

    if hc_count == 0 || hc_count > HC_MAX_STREAMS {
        bail!("hc_norm hc_count ({hc_count}) must be in 1..={HC_MAX_STREAMS}");
    }
    let width = checked_elems(&[hc_count, hidden], "hc_norm carrier width")?;
    let Some(threads) = hc_norm_threads(hidden) else {
        bail!(
            "hc_norm hidden ({hidden}) must be a multiple of {DELTA_SIMD_WIDTH} so the \
             per-stream reduction tiles whole simdgroups"
        );
    };
    let (n, row) = stream
        .dims2()
        .map_err(|e| anyhow::anyhow!("the carrier must be rank-2 [n, hc_count * hidden]: {e}"))?;
    if n == 0 {
        bail!("hc_norm needs at least one token");
    }
    if row != width {
        bail!("the carrier is {row} wide, expected hc_count * hidden = {width}");
    }
    check_f32(stream, &[n, width], "carrier")?;
    check_f32(norm_w, &[width], "hc norm weight")?;
    if let Some(inject_w) = inject_w {
        check_f32(inject_w, &[hc_count, width], "hc injection head")?;
    }
    for (name, t) in [
        Some(("hc norm weight", norm_w)),
        inject_w.map(|t| ("hc injection head", t)),
    ]
    .into_iter()
    .flatten()
    {
        if !stream.device().same_device(t.device()) {
            bail!("{name} must live on the same Metal device as the carrier");
        }
    }
    let n_normed = checked_elems(&[n, width], "hc_norm")?;
    glue_index_fits_i32(n_normed)?;
    let n_inject = checked_elems(&[n, hc_count], "hc_norm injection")?;

    let split = split.unwrap_or(n < crate::ops::hc_split_max_n());
    // The split arm runs the injection head as its own launch where the single
    // arm folds it into the norm's second pass, so which kernels are needed
    // depends on both switches.
    let kernels: &[&str] = match (split, inject_w.is_some()) {
        (true, true) => &["kernel_hc_norm_split", "kernel_hc_inject"],
        (true, false) => &["kernel_hc_norm_split"],
        (false, true) => &["kernel_hc_norm_inject"],
        (false, false) => &["kernel_hc_norm"],
    };
    let mut pipes = Vec::with_capacity(kernels.len());
    for kernel in kernels {
        let pipeline = pipelines::hc_pipeline(mdev.device(), kernel)?;
        if pipeline.max_total_threads_per_threadgroup() < threads {
            bail!(
                "{kernel} needs {threads} threads per threadgroup, the pipeline allows {}",
                pipeline.max_total_threads_per_threadgroup()
            );
        }
        // Same 32-wide-simdgroup assumption the delta kernels make, checked at
        // pipeline setup so it fails at load rather than silently at dispatch.
        check_delta_simd_width(&pipeline, kernel)?;
        pipes.push(pipeline);
    }

    let normed = mdev.new_buffer(n_normed, DType::F32, "hc_norm")?;
    // Allocated whether or not there is a head: the single kernel takes both
    // output bindings, and its no-head arm never writes through this one.
    let inject = mdev.new_buffer(
        if inject_w.is_some() { n_inject } else { 1 },
        DType::F32,
        "hc_inject",
    )?;

    let (x_guard, x_layout) = stream.storage_and_layout();
    let Storage::Metal(x_storage) = &*x_guard else {
        bail!("the carrier is not on a Metal device");
    };
    let (w_guard, w_layout) = norm_w.storage_and_layout();
    let Storage::Metal(w_storage) = &*w_guard else {
        bail!("the hc norm weight is not on a Metal device");
    };
    let inj_parts = inject_w.map(|t| t.storage_and_layout());

    let args = HcNormArgs {
        hc_count: hc_count as i32,
        hidden: hidden as i32,
        width: width as i32,
        eps,
        // The classic chain scales by candle's `affine(1/hc_count, 0)`, a
        // MULTIPLY; matching it keeps the two paths' rounding identical where
        // 1/hc_count is not exact.
        inv_hc: (1.0 / hc_count as f64) as f32,
    };
    {
        let inj_bind = match &inj_parts {
            Some((guard, layout)) => {
                let Storage::Metal(storage) = &**guard else {
                    bail!("the hc injection head is not on a Metal device");
                };
                Some((storage.buffer(), f32_off(layout)))
            }
            None => None,
        };
        let cmd = mdev.command_encoder()?;
        let ep = &cmd;
        let encoder = ep.encoder();
        let encoder: &ComputeCommandEncoder = encoder.as_ref();
        let tg = mtl_size(threads, 1, 1);
        if split {
            // One threadgroup per (token, stream) for the norm, and — with a
            // head — one per (token, injection row) for the gate. The gate
            // reads `normed` back, which candle's concurrent encoder turns into
            // a barrier on its own: it tracks that buffer as the preceding
            // dispatch's output.
            encoder.set_compute_pipeline_state(&pipes[0]);
            encoder.set_bytes(0, &args);
            encoder.set_input_buffer(1, Some(x_storage.buffer()), f32_off(x_layout));
            encoder.set_input_buffer(2, Some(w_storage.buffer()), f32_off(w_layout));
            encoder.set_output_buffer(3, Some(&normed), 0);
            encoder.dispatch_thread_groups(mtl_size(n, hc_count, 1), tg);
            if let Some((inj_buf, inj_off)) = inj_bind {
                encoder.set_compute_pipeline_state(&pipes[1]);
                encoder.set_bytes(0, &args);
                encoder.set_input_buffer(1, Some(inj_buf), inj_off);
                encoder.set_input_buffer(2, Some(&normed), 0);
                encoder.set_output_buffer(3, Some(&inject), 0);
                encoder.dispatch_thread_groups(mtl_size(n, hc_count, 1), tg);
            }
        } else {
            // The unused-head arm binds the norm weight into the injection
            // slot: the kernel never dereferences it, and a bound buffer keeps
            // the argument table uniform across both specializations.
            let (inj_buf, inj_off) = inj_bind.unwrap_or((w_storage.buffer(), f32_off(w_layout)));
            encoder.set_compute_pipeline_state(&pipes[0]);
            encoder.set_bytes(0, &args);
            encoder.set_input_buffer(1, Some(x_storage.buffer()), f32_off(x_layout));
            encoder.set_input_buffer(2, Some(w_storage.buffer()), f32_off(w_layout));
            encoder.set_input_buffer(3, Some(inj_buf), inj_off);
            encoder.set_output_buffer(4, Some(&normed), 0);
            encoder.set_output_buffer(5, Some(&inject), 0);
            encoder.dispatch_thread_groups(mtl_size(n, 1, 1), tg);
        }
    }
    drop(x_guard);
    drop(w_guard);
    drop(inj_parts);

    let normed = output_tensor(normed, mdev, n_normed, (n, width));
    let inject = inject_w.map(|_| output_tensor(inject, mdev, n_inject, (n, hc_count)));
    Ok((normed, inject))
}

/// The bottleneck activation `silu(x / hc_count)` against
/// `kernel_hc_silu_quarter` (hc.metal), replacing candle's affine + silu pair.
/// `x` is the down projection's output, any shape, f32 contiguous; the result
/// keeps that shape. Bit-identical to the pair it replaces.
pub(crate) fn run_hc_silu_quarter(x: &Tensor, hc_count: usize) -> Result<Tensor> {
    let cdev = x.device().clone();
    let Device::Metal(mdev) = &cdev else {
        bail!("hc_silu_quarter requires its input on a Metal device");
    };
    if hc_count == 0 {
        bail!("hc_silu_quarter hc_count must be positive");
    }
    if x.dtype() != DType::F32 {
        bail!("hc_silu_quarter input must be f32, got {:?}", x.dtype());
    }
    if !x.is_contiguous() {
        bail!("hc_silu_quarter input must be contiguous");
    }
    let shape = x.shape().clone();
    let n = checked_elems(shape.dims(), "hc_silu_quarter")?;
    if n == 0 {
        bail!("hc_silu_quarter needs at least one element");
    }
    glue_index_fits_i32(n)?;

    let pipeline = pipelines::hc_pipeline(mdev.device(), "kernel_hc_silu_quarter")?;
    let dst = mdev.new_buffer(n, DType::F32, "hc_silu_quarter")?;

    let (x_guard, x_layout) = x.storage_and_layout();
    let Storage::Metal(x_storage) = &*x_guard else {
        bail!("hc_silu_quarter input is not on a Metal device");
    };

    let args = HcSiluArgs {
        n: n as i32,
        scale: (1.0 / hc_count as f64) as f32,
    };
    {
        let cmd = mdev.command_encoder()?;
        let ep = &cmd;
        let encoder = ep.encoder();
        let encoder: &ComputeCommandEncoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_bytes(0, &args);
        encoder.set_input_buffer(1, Some(x_storage.buffer()), f32_off(x_layout));
        encoder.set_output_buffer(2, Some(&dst), 0);
        dispatch_linear(encoder, &pipeline, n);
    }
    drop(x_guard);

    Ok(output_tensor(dst, mdev, n, shape))
}

/// The stream mix and collapse against `kernel_hc_mix` (hc.metal):
/// `mixed[j] = mean_s sigmoid(up[s * hidden + j]) * normed[s * hidden + j]`.
/// `up` is the up projection's RAW pre-sigmoid logits `[n, hc_count * hidden]`
/// and `normed` the normed carrier of the same shape; returns `[n, hidden]`.
pub(crate) fn run_hc_mix(
    up: &Tensor,
    normed: &Tensor,
    hc_count: usize,
    hidden: usize,
) -> Result<Tensor> {
    let cdev = up.device().clone();
    let Device::Metal(mdev) = &cdev else {
        bail!("hc_mix requires its input on a Metal device");
    };
    if hc_count == 0 || hidden == 0 {
        bail!("hc_mix geometry must be positive, got hc_count {hc_count}, hidden {hidden}");
    }
    let width = checked_elems(&[hc_count, hidden], "hc_mix carrier width")?;
    let (n, row) = up
        .dims2()
        .map_err(|e| anyhow::anyhow!("hc_mix up must be rank-2 [n, hc_count * hidden]: {e}"))?;
    if n == 0 {
        bail!("hc_mix needs at least one token");
    }
    if row != width {
        bail!("hc_mix up is {row} wide, expected hc_count * hidden = {width}");
    }
    check_f32(up, &[n, width], "hc_mix up")?;
    check_f32(normed, &[n, width], "hc_mix normed")?;
    if !up.device().same_device(normed.device()) {
        bail!("hc_mix operands must live on the same Metal device");
    }
    let n_out = checked_elems(&[n, hidden], "hc_mix")?;
    glue_index_fits_i32(checked_elems(&[n, width], "hc_mix input")?)?;

    let pipeline = pipelines::hc_pipeline(mdev.device(), "kernel_hc_mix")?;
    let dst = mdev.new_buffer(n_out, DType::F32, "hc_mix")?;

    let (up_guard, up_layout) = up.storage_and_layout();
    let Storage::Metal(up_storage) = &*up_guard else {
        bail!("hc_mix up is not on a Metal device");
    };
    let (nm_guard, nm_layout) = normed.storage_and_layout();
    let Storage::Metal(nm_storage) = &*nm_guard else {
        bail!("hc_mix normed is not on a Metal device");
    };

    let args = HcMixArgs {
        n: n_out as i32,
        hc_count: hc_count as i32,
        hidden: hidden as i32,
        inv_hc: (1.0 / hc_count as f64) as f32,
    };
    {
        let cmd = mdev.command_encoder()?;
        let ep = &cmd;
        let encoder = ep.encoder();
        let encoder: &ComputeCommandEncoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_bytes(0, &args);
        encoder.set_input_buffer(1, Some(up_storage.buffer()), f32_off(up_layout));
        encoder.set_input_buffer(2, Some(nm_storage.buffer()), f32_off(nm_layout));
        encoder.set_output_buffer(3, Some(&dst), 0);
        dispatch_linear(encoder, &pipeline, n_out);
    }
    drop(up_guard);
    drop(nm_guard);

    Ok(output_tensor(dst, mdev, n_out, (n, hidden)))
}

/// The write-back against `kernel_hc_write` (hc.metal):
/// `new[s * hidden + j] = stream[s * hidden + j] + block_out[j] * inject[s]`,
/// onto the RAW carrier and OUT OF PLACE. `stream` is `[n, hc_count * hidden]`,
/// `block_out` `[n, hidden]`, `inject` `[n, hc_count]`, all f32 contiguous.
/// Bit-identical to the candle broadcast-multiply-then-add chain it replaces.
pub(crate) fn run_hc_write(stream: &Tensor, block_out: &Tensor, inject: &Tensor) -> Result<Tensor> {
    let cdev = stream.device().clone();
    let Device::Metal(mdev) = &cdev else {
        bail!("hc_write requires the carrier on a Metal device");
    };

    let (n, width) = stream
        .dims2()
        .map_err(|e| anyhow::anyhow!("the carrier must be rank-2 [n, hc_count * hidden]: {e}"))?;
    let (n_out, hidden) = block_out
        .dims2()
        .map_err(|e| anyhow::anyhow!("the block output must be rank-2 [n, hidden]: {e}"))?;
    let (n_inj, hc_count) = inject
        .dims2()
        .map_err(|e| anyhow::anyhow!("the injection must be rank-2 [n, hc_count]: {e}"))?;
    if n == 0 {
        bail!("hc_write needs at least one token");
    }
    if n != n_out || n != n_inj {
        bail!("row counts differ: carrier {n}, block output {n_out}, injection {n_inj}");
    }
    if hc_count == 0 || hidden == 0 {
        bail!("hc_write geometry must be positive, got hc_count {hc_count}, hidden {hidden}");
    }
    if checked_elems(&[hc_count, hidden], "hc_write carrier width")? != width {
        bail!(
            "the carrier is {width} wide but the block output ({hidden}) times the stream \
             count ({hc_count}) is {}",
            hc_count * hidden
        );
    }
    check_f32(stream, &[n, width], "carrier")?;
    check_f32(block_out, &[n, hidden], "block output")?;
    check_f32(inject, &[n, hc_count], "injection")?;
    for (name, t) in [("block output", block_out), ("injection", inject)] {
        if !stream.device().same_device(t.device()) {
            bail!("the {name} must live on the same Metal device as the carrier");
        }
    }
    let n_elems = checked_elems(&[n, width], "hc_write")?;
    glue_index_fits_i32(n_elems)?;

    let pipeline = pipelines::hc_pipeline(mdev.device(), "kernel_hc_write")?;
    let dst = mdev.new_buffer(n_elems, DType::F32, "hc_write")?;

    let (x_guard, x_layout) = stream.storage_and_layout();
    let Storage::Metal(x_storage) = &*x_guard else {
        bail!("the carrier is not on a Metal device");
    };
    let (b_guard, b_layout) = block_out.storage_and_layout();
    let Storage::Metal(b_storage) = &*b_guard else {
        bail!("the block output is not on a Metal device");
    };
    let (i_guard, i_layout) = inject.storage_and_layout();
    let Storage::Metal(i_storage) = &*i_guard else {
        bail!("the injection is not on a Metal device");
    };

    let args = HcWriteArgs {
        n: n_elems as i32,
        hc_count: hc_count as i32,
        hidden: hidden as i32,
    };
    {
        let cmd = mdev.command_encoder()?;
        let ep = &cmd;
        let encoder = ep.encoder();
        let encoder: &ComputeCommandEncoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_bytes(0, &args);
        encoder.set_input_buffer(1, Some(x_storage.buffer()), f32_off(x_layout));
        encoder.set_input_buffer(2, Some(b_storage.buffer()), f32_off(b_layout));
        encoder.set_input_buffer(3, Some(i_storage.buffer()), f32_off(i_layout));
        encoder.set_output_buffer(4, Some(&dst), 0);
        dispatch_linear(encoder, &pipeline, n_elems);
    }
    drop(x_guard);
    drop(b_guard);
    drop(i_guard);

    Ok(output_tensor(dst, mdev, n_elems, (n, width)))
}

/// The plane's declared shape must fit the allocation it views.
///
/// Both gate kernels index the weight by row and block with no bound of their
/// own — a q8_0 plane is raw bytes, not a `Tensor`, so nothing else in the call
/// carries its length. Every f32 operand is shape-checked by `check_f32` and the
/// projections' `out_dim`/`in_dim` are checked against the geometry, but a plane
/// whose declared shape outruns its buffer would pass both of those and read off
/// the end of device memory.
pub(crate) fn check_plane_fits(plane: &QuantPlane, what: &str) -> Result<()> {
    let dt = plane.dtype;
    // `in_dim` is a whole number of blocks by the time this runs
    // (`hc_gate_fused_supported` and the shape check above).
    let Some(need) = (plane.in_dim / dt.block_size())
        .checked_mul(dt.type_size())
        .and_then(|row| row.checked_mul(plane.out_dim))
        .and_then(|body| body.checked_add(plane.base_off))
    else {
        bail!(
            "the {what} plane's [{}, {}] {dt:?} shape overflows a byte count",
            plane.out_dim,
            plane.in_dim
        );
    };
    let have = plane.buffer.length();
    if need > have {
        bail!(
            "the {what} plane declares [{}, {}] {dt:?} at offset {}, which needs {need} bytes of \
             a {have}-byte buffer",
            plane.out_dim,
            plane.in_dim,
            plane.base_off
        );
    }
    Ok(())
}

/// Matches the Metal `hc_gate_down_args` struct (src/ops/hc.metal).
#[repr(C)]
#[derive(Clone, Copy)]
struct HcGateDownArgs {
    hc_count: i32,
    hidden: i32,
    width: i32,
    low_rank: i32,
    nblk: i32,
    n_down_tg: i32,
    eps: f32,
    inv_hc: f32,
}

/// Matches the Metal `hc_gate_up_args` struct (src/ops/hc.metal).
#[repr(C)]
#[derive(Clone, Copy)]
struct HcGateUpArgs {
    hc_count: i32,
    hidden: i32,
    width: i32,
    low_rank: i32,
    nblk_low: i32,
    inv_hc: f32,
}

/// The first half of the fused decode gate against `kernel_hc_gate_down`
/// (hc.metal): the carrier's grouped RMS norm, the injection head, the q8_0 down
/// projection and the bottleneck activation, in ONE dispatch where the split
/// path spends four.
///
/// `stream` is the raw carrier `[n, hc_count * hidden]` f32, `norm_w` the
/// `[hc_count * hidden]` multiply-ready norm weight, `inject_w` the dense
/// `[hc_count, hc_count * hidden]` head (`None` on the tail mixer) and `down`
/// the `[low_rank, hc_count * hidden]` q8_0 projection's raw bytes. Returns
/// `silu((down . normed) / hc_count)` as `[n, low_rank]`, the write strengths
/// `[n, hc_count]` when there is a head, and the per-stream `[n, hc_count]`
/// scales — which are not a debugging output but [`run_hc_gate_up_mix`]'s way of
/// rebuilding `normed` without either kernel materializing it.
///
/// Geometry outside [`hc_gate_fused_supported`] is an error here: the caller
/// asks that predicate first and keeps the split path when it says no.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_hc_gate_down(
    stream: &Tensor,
    norm_w: &Tensor,
    inject_w: Option<&Tensor>,
    down: &QuantPlane,
    hc_count: usize,
    hidden: usize,
    low_rank: usize,
    eps: f32,
) -> Result<(Tensor, Option<Tensor>, Tensor)> {
    let cdev = stream.device().clone();
    let Device::Metal(mdev) = &cdev else {
        bail!("hc_gate_down requires the carrier on a Metal device");
    };
    if !hc_gate_fused_supported(hc_count, hidden, low_rank, down.dtype) {
        bail!(
            "hc_gate_down does not cover hc_count {hc_count}, hidden {hidden}, low_rank \
             {low_rank}, {:?} weights",
            down.dtype
        );
    }
    let width = checked_elems(&[hc_count, hidden], "hc_gate carrier width")?;
    let (n, row) = stream
        .dims2()
        .map_err(|e| anyhow::anyhow!("the carrier must be rank-2 [n, hc_count * hidden]: {e}"))?;
    if n == 0 {
        bail!("hc_gate_down needs at least one token");
    }
    if row != width {
        bail!("the carrier is {row} wide, expected hc_count * hidden = {width}");
    }
    if down.out_dim != low_rank || down.in_dim != width {
        bail!(
            "the down projection is [{}, {}], expected [{low_rank}, {width}]",
            down.out_dim,
            down.in_dim
        );
    }
    check_f32(stream, &[n, width], "carrier")?;
    check_f32(norm_w, &[width], "hc norm weight")?;
    if let Some(inject_w) = inject_w {
        check_f32(inject_w, &[hc_count, width], "hc injection head")?;
    }
    for (name, t) in [
        Some(("hc norm weight", norm_w)),
        inject_w.map(|t| ("hc injection head", t)),
    ]
    .into_iter()
    .flatten()
    {
        if !stream.device().same_device(t.device()) {
            bail!("{name} must live on the same Metal device as the carrier");
        }
    }
    // The kernel indexes the weight through `device const hc_block_q8_0 *`,
    // whose alignment is that of its `half` scale. Rows are whole blocks, so
    // only the bound base offset could break it.
    if !down.base_off.is_multiple_of(2) {
        bail!(
            "hc_gate_down needs a 2-byte-aligned weight view, got offset {}",
            down.base_off
        );
    }
    check_plane_fits(down, "hc down projection")?;
    glue_index_fits_i32(checked_elems(&[n, width], "hc_gate carrier")?)?;
    let n_low = checked_elems(&[n, low_rank], "hc_gate bottleneck")?;
    let n_inject = checked_elems(&[n, hc_count], "hc_gate injection")?;

    let pipeline = pipelines::hc_pipeline(mdev.device(), "kernel_hc_gate_down")?;
    if pipeline.max_total_threads_per_threadgroup() < HC_GATE_THREADS {
        bail!(
            "kernel_hc_gate_down needs {HC_GATE_THREADS} threads per threadgroup, the pipeline \
             allows {}",
            pipeline.max_total_threads_per_threadgroup()
        );
    }
    check_delta_simd_width(&pipeline, "kernel_hc_gate_down")?;

    let low = mdev.new_buffer(n_low, DType::F32, "hc_gate_low")?;
    // Allocated whether or not there is a head: the kernel takes both output
    // bindings, and a headless launch has no threadgroup that writes this one.
    let inject = mdev.new_buffer(
        if inject_w.is_some() { n_inject } else { 1 },
        DType::F32,
        "hc_gate_inject",
    )?;
    let scales = mdev.new_buffer(n_inject, DType::F32, "hc_gate_scales")?;

    let (x_guard, x_layout) = stream.storage_and_layout();
    let Storage::Metal(x_storage) = &*x_guard else {
        bail!("the carrier is not on a Metal device");
    };
    let (w_guard, w_layout) = norm_w.storage_and_layout();
    let Storage::Metal(w_storage) = &*w_guard else {
        bail!("the hc norm weight is not on a Metal device");
    };
    let inj_parts = inject_w.map(|t| t.storage_and_layout());

    let n_down_tg = low_rank.div_ceil(HC_GATE_ROWS_PER_TG);
    let args = HcGateDownArgs {
        hc_count: hc_count as i32,
        hidden: hidden as i32,
        width: width as i32,
        low_rank: low_rank as i32,
        nblk: (width / GgmlDType::Q8_0.block_size()) as i32,
        n_down_tg: n_down_tg as i32,
        eps,
        // The classic chain scales by candle's `affine(1/hc_count, 0)`, a
        // MULTIPLY; matching it keeps the two paths' rounding identical where
        // 1/hc_count is not exact.
        inv_hc: (1.0 / hc_count as f64) as f32,
    };
    {
        let inj_bind = match &inj_parts {
            Some((guard, layout)) => {
                let Storage::Metal(storage) = &**guard else {
                    bail!("the hc injection head is not on a Metal device");
                };
                Some((storage.buffer(), f32_off(layout)))
            }
            None => None,
        };
        // The headless arm binds the norm weight into the injection slot: no
        // threadgroup of that launch dereferences it, and a bound buffer keeps
        // the argument table uniform across both grids.
        let (inj_buf, inj_off) = inj_bind.unwrap_or((w_storage.buffer(), f32_off(w_layout)));
        let cmd = mdev.command_encoder()?;
        let ep = &cmd;
        let encoder = ep.encoder();
        let encoder: &ComputeCommandEncoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_bytes(0, &args);
        encoder.set_input_buffer(1, Some(x_storage.buffer()), f32_off(x_layout));
        encoder.set_input_buffer(2, Some(w_storage.buffer()), f32_off(w_layout));
        encoder.set_input_buffer(3, Some(inj_buf), inj_off);
        encoder.set_input_buffer(4, Some(&down.buffer), down.base_off);
        encoder.set_output_buffer(5, Some(&low), 0);
        encoder.set_output_buffer(6, Some(&inject), 0);
        encoder.set_output_buffer(7, Some(&scales), 0);
        // One threadgroup per (row tile of the down weight, token), plus one for
        // the injection head where there is one.
        let grid_x = n_down_tg + usize::from(inject_w.is_some());
        encoder.dispatch_thread_groups(mtl_size(grid_x, n, 1), mtl_size(HC_GATE_THREADS, 1, 1));
    }
    drop(x_guard);
    drop(w_guard);
    drop(inj_parts);

    let low = output_tensor(low, mdev, n_low, (n, low_rank));
    let scales = output_tensor(scales, mdev, n_inject, (n, hc_count));
    let inject = inject_w.map(|_| output_tensor(inject, mdev, n_inject, (n, hc_count)));
    Ok((low, inject, scales))
}

/// The second half of the fused decode gate against `kernel_hc_gate_up_mix`
/// (hc.metal): the q8_0 up projection, its sigmoid, and the mix and collapse, in
/// ONE dispatch where the split path spends two.
///
/// `low` is [`run_hc_gate_down`]'s `[n, low_rank]` bottleneck activation, `up`
/// the `[hc_count * hidden, low_rank]` q8_0 projection's raw bytes, and
/// `stream` / `norm_w` / `scales` are what the normed carrier is rebuilt from
/// per element. Returns `[n, hidden]`, the vector the block runs on:
/// `mean_s sigmoid(up[s*hidden+j] . low) * normed[s*hidden+j]`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_hc_gate_up_mix(
    low: &Tensor,
    up: &QuantPlane,
    stream: &Tensor,
    norm_w: &Tensor,
    scales: &Tensor,
    hc_count: usize,
    hidden: usize,
    low_rank: usize,
) -> Result<Tensor> {
    let cdev = stream.device().clone();
    let Device::Metal(mdev) = &cdev else {
        bail!("hc_gate_up_mix requires the carrier on a Metal device");
    };
    if !hc_gate_fused_supported(hc_count, hidden, low_rank, up.dtype) {
        bail!(
            "hc_gate_up_mix does not cover hc_count {hc_count}, hidden {hidden}, low_rank \
             {low_rank}, {:?} weights",
            up.dtype
        );
    }
    let width = checked_elems(&[hc_count, hidden], "hc_gate carrier width")?;
    let (n, row) = stream
        .dims2()
        .map_err(|e| anyhow::anyhow!("the carrier must be rank-2 [n, hc_count * hidden]: {e}"))?;
    if n == 0 {
        bail!("hc_gate_up_mix needs at least one token");
    }
    if row != width {
        bail!("the carrier is {row} wide, expected hc_count * hidden = {width}");
    }
    if up.out_dim != width || up.in_dim != low_rank {
        bail!(
            "the up projection is [{}, {}], expected [{width}, {low_rank}]",
            up.out_dim,
            up.in_dim
        );
    }
    check_f32(stream, &[n, width], "carrier")?;
    check_f32(norm_w, &[width], "hc norm weight")?;
    check_f32(low, &[n, low_rank], "hc bottleneck activation")?;
    check_f32(scales, &[n, hc_count], "hc stream scales")?;
    for (name, t) in [
        ("hc norm weight", norm_w),
        ("hc bottleneck activation", low),
        ("hc stream scales", scales),
    ] {
        if !stream.device().same_device(t.device()) {
            bail!("{name} must live on the same Metal device as the carrier");
        }
    }
    if !up.base_off.is_multiple_of(2) {
        bail!(
            "hc_gate_up_mix needs a 2-byte-aligned weight view, got offset {}",
            up.base_off
        );
    }
    check_plane_fits(up, "hc up projection")?;
    glue_index_fits_i32(checked_elems(&[n, width], "hc_gate carrier")?)?;
    let n_out = checked_elems(&[n, hidden], "hc_gate mixed")?;

    let pipeline = pipelines::hc_pipeline(mdev.device(), "kernel_hc_gate_up_mix")?;
    if pipeline.max_total_threads_per_threadgroup() < HC_GATE_MIX_THREADS {
        bail!(
            "kernel_hc_gate_up_mix needs {HC_GATE_MIX_THREADS} threads per threadgroup, the \
             pipeline allows {}",
            pipeline.max_total_threads_per_threadgroup()
        );
    }
    check_delta_simd_width(&pipeline, "kernel_hc_gate_up_mix")?;

    let dst = mdev.new_buffer(n_out, DType::F32, "hc_gate_mixed")?;

    let (l_guard, l_layout) = low.storage_and_layout();
    let Storage::Metal(l_storage) = &*l_guard else {
        bail!("the hc bottleneck activation is not on a Metal device");
    };
    let (x_guard, x_layout) = stream.storage_and_layout();
    let Storage::Metal(x_storage) = &*x_guard else {
        bail!("the carrier is not on a Metal device");
    };
    let (w_guard, w_layout) = norm_w.storage_and_layout();
    let Storage::Metal(w_storage) = &*w_guard else {
        bail!("the hc norm weight is not on a Metal device");
    };
    let (s_guard, s_layout) = scales.storage_and_layout();
    let Storage::Metal(s_storage) = &*s_guard else {
        bail!("the hc stream scales are not on a Metal device");
    };

    let args = HcGateUpArgs {
        hc_count: hc_count as i32,
        hidden: hidden as i32,
        width: width as i32,
        low_rank: low_rank as i32,
        nblk_low: (low_rank / GgmlDType::Q8_0.block_size()) as i32,
        inv_hc: (1.0 / hc_count as f64) as f32,
    };
    {
        let cmd = mdev.command_encoder()?;
        let ep = &cmd;
        let encoder = ep.encoder();
        let encoder: &ComputeCommandEncoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_bytes(0, &args);
        encoder.set_input_buffer(1, Some(l_storage.buffer()), f32_off(l_layout));
        encoder.set_input_buffer(2, Some(&up.buffer), up.base_off);
        encoder.set_input_buffer(3, Some(x_storage.buffer()), f32_off(x_layout));
        encoder.set_input_buffer(4, Some(w_storage.buffer()), f32_off(w_layout));
        encoder.set_input_buffer(5, Some(s_storage.buffer()), f32_off(s_layout));
        encoder.set_output_buffer(6, Some(&dst), 0);
        // One threadgroup per (column tile of the carrier, token); each covers
        // HC_GATE_MIX_THREADS / hc_count columns.
        encoder.dispatch_thread_groups(
            mtl_size(hidden.div_ceil(HC_GATE_MIX_THREADS / hc_count), n, 1),
            mtl_size(HC_GATE_MIX_THREADS, 1, 1),
        );
    }
    drop(l_guard);
    drop(x_guard);
    drop(w_guard);
    drop(s_guard);

    Ok(output_tensor(dst, mdev, n_out, (n, hidden)))
}

#[repr(C)]
#[derive(Clone, Copy)]
struct PleArgs {
    n: i32,
    hidden: i32,
    width: i32,
    k: i32,
    dilation: i32,
    state_len: i32,
    eps: f32,
    dot_scale: f32,
}

/// Stateless PLE tail; the caller owns the channel-major convolution history.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_ple_tail(
    key: &Tensor,
    value: &Tensor,
    stream: &Tensor,
    query_w: &Tensor,
    norm_w: &Tensor,
    conv_w: &Tensor,
    prior: &Tensor,
    hidden: usize,
    dilation: usize,
    eps: f32,
) -> Result<(Tensor, Tensor)> {
    let Device::Metal(mdev) = stream.device() else {
        bail!("ple_tail requires Metal inputs");
    };
    let (n, width) = stream.dims2()?;
    let (cw, k) = conv_w.dims2()?;
    ensure!(
        n > 0 && hidden > 0 && width > 0 && width.is_multiple_of(hidden),
        "ple_tail invalid grouped geometry"
    );
    ensure!(
        cw == width && k > 0 && dilation > 0,
        "ple_tail invalid convolution geometry"
    );
    ensure!(
        eps.is_finite() && eps > 0.0,
        "ple_tail epsilon must be finite and positive"
    );
    let state_len = checked_elems(&[k - 1, dilation], "ple_tail history")?;
    let count = checked_elems(&[n, width], "ple_tail output")?;
    let line_len = n
        .checked_add(state_len)
        .ok_or_else(|| anyhow::anyhow!("ple_tail history extent overflow"))?;
    for dims in [
        &[n, width][..],
        &[width, k],
        &[width, state_len],
        &[k, dilation],
        &[line_len],
    ] {
        glue_index_fits_i32(checked_elems(dims, "ple_tail indexing")?)?;
    }
    let operands = [key, value, stream, query_w, norm_w, conv_w, prior];
    let shapes: [&[usize]; 7] = [
        &[n, width],
        &[n, hidden],
        &[n, width],
        &[width],
        &[width],
        &[width, k],
        &[width, state_len],
    ];
    for (t, shape) in operands.iter().zip(shapes) {
        check_f32(t, shape, "ple_tail operand")?;
        ensure!(
            stream.device().same_device(t.device()),
            "ple_tail operands must share a Metal device"
        );
    }
    let threads = 256;
    let gate = pipelines::ple_pipeline(mdev.device(), "kernel_ple_gate")?;
    let conv = pipelines::ple_pipeline(mdev.device(), "kernel_ple_conv")?;
    for (name, pipe) in [("kernel_ple_gate", &gate), ("kernel_ple_conv", &conv)] {
        ensure!(
            pipe.max_total_threads_per_threadgroup() >= threads,
            "{name} cannot launch {threads} threads"
        );
        check_delta_simd_width(pipe, name)?;
    }
    // `gated` is private-pool scratch written by the gate kernel and read by the
    // conv kernel in the same encoder. It is held until both dispatches are
    // encoded (the `prepare_mm_id_map0` rule); candle orders the read after the
    // write and any later reuse of the allocation after the read, and the
    // private pool never hands it to a CPU-side upload, so it need not outlive
    // GPU completion the way a readback staging buffer must.
    let gated = mdev.new_buffer(count, DType::F32, "ple_gated")?;
    let normed = mdev.new_buffer(count, DType::F32, "ple_normed")?;
    let out = mdev.new_buffer(count, DType::F32, "ple_out")?;
    let held: Vec<_> = operands.iter().map(|t| t.storage_and_layout()).collect();
    let mut buffers = Vec::with_capacity(held.len());
    for (storage, layout) in &held {
        let Storage::Metal(storage) = &**storage else {
            bail!("ple_tail expected Metal storage")
        };
        buffers.push((storage.buffer(), f32_off(layout)));
    }
    let args = PleArgs {
        n: n as i32,
        hidden: hidden as i32,
        width: width as i32,
        k: k as i32,
        dilation: dilation as i32,
        state_len: state_len as i32,
        eps,
        dot_scale: 1.0 / (hidden as f32).sqrt(),
    };
    {
        let cmd = mdev.command_encoder()?;
        let ep = &cmd;
        let encoder = ep.encoder();
        let encoder: &ComputeCommandEncoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(&gate);
        encoder.set_bytes(0, &args);
        for (i, &(buffer, off)) in buffers[..5].iter().enumerate() {
            encoder.set_input_buffer(i + 1, Some(buffer), off);
        }
        encoder.set_output_buffer(6, Some(&gated), 0);
        encoder.set_output_buffer(7, Some(&normed), 0);
        encoder.dispatch_thread_groups(mtl_size(n, width / hidden, 1), mtl_size(threads, 1, 1));
        encoder.set_compute_pipeline_state(&conv);
        encoder.set_bytes(0, &args);
        encoder.set_input_buffer(1, Some(&gated), 0);
        encoder.set_input_buffer(2, Some(&normed), 0);
        encoder.set_input_buffer(3, Some(buffers[6].0), buffers[6].1);
        encoder.set_input_buffer(4, Some(buffers[5].0), buffers[5].1);
        encoder.set_output_buffer(5, Some(&out), 0);
        encoder.dispatch_thread_groups(
            mtl_size(count.div_ceil(threads), 1, 1),
            mtl_size(threads, 1, 1),
        );
    }
    Ok((
        output_tensor(out, mdev, count, (n, width)),
        output_tensor(normed, mdev, count, (n, width)),
    ))
}

#[cfg(test)]
mod combine_guard_tests {
    use super::{combine_index_fits_i32, combine_reduction_width};

    /// The i32 index guard: the combine kernels address `down` with i32 math
    /// (`down_base = s*top_k*n_out + c`), so the grid's flat element count must
    /// stay within i32. Tested directly — the overflowing case (seq ≈ 70k at
    /// top_k=10 / n_out=3072) is a ~8.6TB tensor that cannot be allocated.
    #[test]
    fn index_guard_rejects_i32_overflow() {
        // Production decode/prefill geometry stays well within i32.
        assert!(combine_index_fits_i32(1, 10, 3072));
        assert!(combine_index_fits_i32(4096, 10, 3072)); // 125.8M < 2.1B
        // Just under and just over i32::MAX with a top_k=10 / n_out=3072 row.
        let per_seq = 10 * 3072; // 30720 elements per seq row
        let max_ok = i32::MAX as usize / per_seq; // largest seq that still fits
        assert!(combine_index_fits_i32(max_ok, 10, 3072));
        assert!(!combine_index_fits_i32(max_ok + 1, 10, 3072));
    }

    /// The single-simdgroup width guard threshold: `next_pow2(top_k/2)` must stay
    /// <= 32. Production top_k=10 gives width 8; top_k=66 is the first that needs
    /// width 64 (66/2=33 → next_pow2 64).
    #[test]
    fn reduction_width_threshold() {
        assert_eq!(combine_reduction_width(10), 8);
        assert_eq!(combine_reduction_width(64), 32); // 64/2=32, still one simdgroup
        assert_eq!(combine_reduction_width(65), 32); // 65/2=32
        assert_eq!(combine_reduction_width(66), 64); // 66/2=33 → 64, over the limit
        assert!(combine_reduction_width(66) > 32);
        assert!(combine_reduction_width(10) <= 32);
    }
}

#[cfg(test)]
pub(crate) mod testutil {
    use anyhow::Result;
    use candle_core::quantized::{GgmlDType, QStorage, QTensor};
    use candle_core::{Device, Tensor};
    use std::sync::Arc;

    use crate::gguf::ExpertStack;

    /// Build an expert stack `[n_expert, n_out, k]` on `device` by quantizing a
    /// fixed pseudo-random f32 tensor to `dt`. Returns the stack plus the
    /// dequantized-then-reread weights the kernel effectively sees, so the oracle
    /// compares against the same rounding the kernel does.
    pub(crate) fn build_stack(
        device: &Device,
        dt: GgmlDType,
        n_expert: usize,
        n_out: usize,
        k: usize,
        seed: u64,
    ) -> Result<(ExpertStack, Vec<f32>)> {
        let w = pseudo_random(n_expert * n_out * k, seed, -1.0, 1.0);
        let w_t = Tensor::from_vec(w, (n_expert, n_out, k), device)?;
        let qt = QTensor::quantize(&w_t, dt)?;
        // What the kernel actually multiplies: the quantized weights, dequantized.
        let deq = qt
            .dequantize(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        // Mirror the production `expert_stack` load path exactly: upload the
        // quantized bytes once via `from_data`, retain the buffer handle for the
        // fused kernels, then MOVE that storage into the QTensor. The qtensor and
        // the retained `buffer` must share one allocation — if `qtensor` came from
        // a separate `quantize` instead, the shared buffer's only pool reference
        // would hit strong_count 1 and candle's `drop_unused_buffers` (triggered
        // by any readback) would evict it from the residency set, so a later fused
        // dispatch reads a non-resident buffer. (Test-only difference from
        // production: the bytes come from `qt.data()`, not the GGUF file.)
        let storage = QStorage::from_data(qt.data()?, device, dt)?;
        let buffer = match &storage {
            QStorage::Metal(qms) => Some(Arc::new(qms.buffer().clone())),
            _ => None,
        };
        let qtensor = Arc::new(QTensor::new(storage, (n_expert, n_out, k))?);
        let stack = ExpertStack {
            qtensor: Some(qtensor),
            buffer,
            base_off: 0,
            mmap: None,
            dtype: dt,
            n_expert,
            n_out,
            k,
        };
        Ok((stack, deq))
    }

    /// Deterministic reference: for each (token, slot) select expert `ids[token][slot]`,
    /// pick x row `slot` (per-slot) or `0` (shared) per `x_per_row`, and dot each of
    /// the expert's `n_out` rows with it. Layout matches the kernel output
    /// `[t, top_k, n_out]`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn oracle(
        deq_weights: &[f32],
        x: &[f32],
        ids: &[u32],
        n_out: usize,
        k: usize,
        t: usize,
        top_k: usize,
        x_per_row: usize,
    ) -> Vec<f32> {
        let mut out = vec![0f32; t * top_k * n_out];
        for token in 0..t {
            for slot in 0..top_k {
                let e = ids[token * top_k + slot] as usize;
                let x_row = if x_per_row == 1 { 0 } else { slot };
                let x_base = (token * x_per_row + x_row) * k;
                for o in 0..n_out {
                    let w_base = (e * n_out + o) * k;
                    let mut acc = 0f32;
                    for i in 0..k {
                        acc += deq_weights[w_base + i] * x[x_base + i];
                    }
                    out[(token * top_k + slot) * n_out + o] = acc;
                }
            }
        }
        out
    }

    /// Relative L2 error between two equal-length slices.
    pub(crate) fn rel_l2(got: &[f32], want: &[f32]) -> f32 {
        let mut num = 0f64;
        let mut den = 0f64;
        for (g, w) in got.iter().zip(want) {
            num += (*g as f64 - *w as f64).powi(2);
            den += (*w as f64).powi(2);
        }
        (num / den.max(1e-30)).sqrt() as f32
    }

    pub(crate) fn max_abs(got: &[f32], want: &[f32]) -> f32 {
        got.iter()
            .zip(want)
            .map(|(g, w)| (g - w).abs())
            .fold(0f32, f32::max)
    }

    /// Small xorshift so tests do not depend on rand's distributions.
    pub(crate) fn pseudo_random(n: usize, seed: u64, lo: f32, hi: f32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let u = (s >> 11) as f64 / (1u64 << 53) as f64; // [0,1)
            out.push(lo + (hi - lo) * u as f32);
        }
        out
    }

    pub(crate) fn random_ids(t: usize, top_k: usize, n_expert: usize, seed: u64) -> Vec<u32> {
        let r = pseudo_random(t * top_k, seed, 0.0, n_expert as f32);
        r.into_iter()
            .map(|v| (v as usize % n_expert) as u32)
            .collect()
    }

    /// Ids with `top_k` DISTINCT experts per token — the invariant real top-k
    /// routing always satisfies (argsort top-k never repeats an index). The
    /// two-pass mm_id kernel relies on it: map0 collapses each token's slots for
    /// an expert into one row, so a token selecting the same expert twice would
    /// lose a slot. mv_id has no such requirement, but distinct ids exercise both.
    pub(crate) fn distinct_ids(t: usize, top_k: usize, n_expert: usize, seed: u64) -> Vec<u32> {
        assert!(
            top_k <= n_expert,
            "cannot pick {top_k} distinct of {n_expert} experts"
        );
        let mut out = Vec::with_capacity(t * top_k);
        let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
        for _ in 0..t {
            let mut chosen: Vec<u32> = Vec::with_capacity(top_k);
            while chosen.len() < top_k {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                let e = (s % n_expert as u64) as u32;
                if !chosen.contains(&e) {
                    chosen.push(e);
                }
            }
            out.extend_from_slice(&chosen);
        }
        out
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct BwArgs {
    n4: u32,
    groups: u32,
}

/// Encodes one bandwidth probe over `n4` float4s starting `src_off` bytes into
/// `src`: the reduce-only read kernel (one partial sum per threadgroup into `dst`)
/// when `read`, else a copy into `dst` at `dst_off`. `groups` threadgroups of 256
/// threads, grid-strided. Bench-only; see `ops::bandwidth`.
pub(crate) fn run_bw_probe(
    mdev: &MetalDevice,
    read: bool,
    src: &Buffer,
    src_off: usize,
    dst: &Buffer,
    dst_off: usize,
    n4: usize,
    groups: usize,
) -> Result<()> {
    ensure!(
        n4 > 0 && u32::try_from(n4).is_ok() && groups > 0 && u32::try_from(groups).is_ok(),
        "bw probe geometry out of range"
    );
    let name = if read {
        "kernel_bw_read"
    } else {
        "kernel_bw_copy"
    };
    let pipe = pipelines::bandwidth_pipeline(mdev.device(), name)?;
    let args = BwArgs {
        n4: n4 as u32,
        groups: groups as u32,
    };
    let cmd = mdev.command_encoder()?;
    let ep = &cmd;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipe);
    encoder.set_bytes(0, &args);
    encoder.set_input_buffer(1, Some(src), src_off);
    encoder.set_output_buffer(2, Some(dst), dst_off);
    encoder.dispatch_thread_groups(mtl_size(groups, 1, 1), mtl_size(256, 1, 1));
    Ok(())
}
