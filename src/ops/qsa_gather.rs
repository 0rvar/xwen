//! Host side of the QSA decode row gather — the K/V rows a
//! [`crate::qwen4exp::indexer::QsaSelection::Rows`] names, packed out of the
//! cache's `[heads, len, head_dim]` view into one contiguous plane per head.
//! `src/attention.rs` is the caller, above the indexer's token budget only;
//! the shipped 3.6/3.8 checkpoints have no indexer and never reach it.
//!
//! One dispatch per plane (`kernel_qsa_gather_*`, qsa_gather.metal) in place of
//! one `index_select` per head plus a `stack`: on the shipped geometry (2 KV
//! heads) that is 2 dispatches per QSA layer where the candle chain spent 3 per
//! plane (2 index_selects + 1 stack) x K-and-V = 6. The kernel is a copy, so it
//! is bit-identical to the chain it replaces; the kill switch back to the chain
//! is `XWEN_QSA_CLASSIC` ([`crate::ops::qsa_classic`]). A source that is not on
//! a Metal device (the CPU / reference-oracle attention path) takes the chain
//! too, with no switch.

use anyhow::Result;
use candle_core::{Device, Tensor};

use crate::ops::dispatch;

/// Pack the cache rows `rows` (u32 `[n_sel]`) names out of a `[heads, len,
/// head_dim]` cache view into a contiguous `[heads, n_sel, head_dim]` plane of
/// the same dtype. The fused kernel by default on Metal, the candle chain under
/// `XWEN_QSA_CLASSIC` and on every other device.
pub fn gather_rows(t: &Tensor, rows: &Tensor) -> Result<Tensor> {
    if crate::ops::qsa_classic() || !matches!(t.device(), Device::Metal(_)) {
        gather_rows_classic(t, rows)
    } else {
        dispatch::run_qsa_gather(t, rows)
    }
}

/// The candle chain: one `index_select` per head, then a `stack`.
///
/// One `index_select` per head, deliberately, rather than one call over the
/// whole rank-3 view. A cache view is a `narrow` of a `max_ctx`-slot buffer, so
/// it is strided across the head axis, and candle's Metal `index_select`
/// MIS-HANDLES a strided source at the pinned rev: `call_index_select` passes
/// the indexed dimension's SIZE where the kernel's `get_strided_index` expects
/// the tensor's RANK (candle-metal-kernels indexing.metal), so every gathered
/// element is read from a garbage offset — silently, with the right shape. A
/// single head's slice IS contiguous by candle's own rule (a leading axis of
/// extent 1 is skipped when checking strides), which puts each of these
/// dispatches on the kernel's correct contiguous path.
pub fn gather_rows_classic(t: &Tensor, rows: &Tensor) -> Result<Tensor> {
    let (heads, len, head_dim) = t.dims3()?;
    let mut packed = Vec::with_capacity(heads);
    for h in 0..heads {
        packed.push(
            t.narrow(0, h, 1)?
                .reshape((len, head_dim))?
                .index_select(rows, 0)?,
        );
    }
    Ok(Tensor::stack(&packed, 0)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::metal_device;
    use crate::ops::dispatch::testutil::pseudo_random;
    use candle_core::{DType, Device};

    /// A `[heads, max_ctx, head_dim]` cache buffer of pseudo-random values in
    /// `dtype`, narrowed to its first `len` positions — the head-strided view
    /// the attention hands the gather.
    fn cache_view(
        dev: &Device,
        dtype: DType,
        heads: usize,
        max_ctx: usize,
        len: usize,
        head_dim: usize,
        seed: u64,
    ) -> Tensor {
        Tensor::from_vec(
            pseudo_random(heads * max_ctx * head_dim, seed, -4.0, 4.0),
            (heads, max_ctx, head_dim),
            &Device::Cpu,
        )
        .unwrap()
        .to_dtype(dtype)
        .unwrap()
        .to_device(dev)
        .unwrap()
        .narrow(1, 0, len)
        .unwrap()
    }

    /// An ascending selection of `n_sel` distinct rows below `len` — the shape
    /// the indexer produces (whole blocks plus the tail).
    fn selection(len: usize, n_sel: usize, seed: u64) -> Vec<u32> {
        let mut rows: Vec<u32> = pseudo_random(len, seed, 0.0, 1.0)
            .iter()
            .enumerate()
            .filter(|(_, r)| **r < 0.5)
            .map(|(i, _)| i as u32)
            .take(n_sel)
            .collect();
        rows.sort_unstable();
        rows
    }

    fn bits(t: &Tensor) -> Vec<u32> {
        t.to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect()
    }

    fn assert_gather_matches(
        dtype: DType,
        heads: usize,
        max_ctx: usize,
        len: usize,
        head_dim: usize,
    ) {
        let dev = metal_device().unwrap();
        let view = cache_view(&dev, dtype, heads, max_ctx, len, head_dim, 0x9A7);
        let n_sel = (len / 2).max(1);
        let rows = selection(len, n_sel, 0x9A8);
        let n_sel = rows.len();
        let rows_t = Tensor::from_vec(rows.clone(), n_sel, &dev).unwrap();

        let fused = dispatch::run_qsa_gather(&view, &rows_t).unwrap();
        let classic = gather_rows_classic(&view, &rows_t).unwrap();
        assert_eq!(fused.dtype(), dtype);
        assert_eq!(fused.dims(), &[heads, n_sel, head_dim]);
        assert_eq!(fused.dims(), classic.dims());
        assert!(fused.is_contiguous());
        assert_eq!(
            bits(&fused),
            bits(&classic),
            "{dtype:?} [{heads}, {len} of {max_ctx}, {head_dim}] x {n_sel} rows: the gather is a \
             copy and must match the index_select chain bit for bit"
        );

        // And against the source itself, so the test does not lean on the
        // chain being right about the strided view.
        let src = bits(&view.contiguous().unwrap());
        let got = bits(&fused);
        for h in 0..heads {
            for (s, &r) in rows.iter().enumerate() {
                let want = &src[(h * len + r as usize) * head_dim..][..head_dim];
                let have = &got[(h * n_sel + s) * head_dim..][..head_dim];
                assert_eq!(have, want, "head {h} selected {s} (row {r})");
            }
        }
    }

    /// The shipped decode geometry: 2 KV heads of 256 f16 in an 8192-slot cache,
    /// a 2051-row selection over 3810 positions (512 blocks of 4 plus a 3-row
    /// tail is 2051; the selection here is the same size, not the same rows).
    #[test]
    fn f16_matches_the_index_select_chain_bitwise() {
        assert_gather_matches(DType::F16, 2, 8192, 3810, 256);
    }

    /// f32 planes, a head count and lengths that leave the vector loop with a
    /// remainder and the head stride with a gap.
    #[test]
    fn f32_matches_the_index_select_chain_bitwise() {
        assert_gather_matches(DType::F32, 3, 300, 257, 128);
        assert_gather_matches(DType::F32, 1, 64, 64, 12);
    }

    /// A single-head view (contiguous by candle's rule) and a full-capacity
    /// view (no head gap at all) both take the kernel.
    #[test]
    fn contiguous_sources_gather_the_same() {
        assert_gather_matches(DType::F16, 1, 128, 100, 256);
        assert_gather_matches(DType::F16, 4, 100, 100, 64);
    }

    /// A row index at or beyond `len` reads nothing: the kernel writes zeros for
    /// that selected row and leaves the others intact.
    #[test]
    fn an_out_of_range_row_gathers_zeros() {
        let dev = metal_device().unwrap();
        let (heads, max_ctx, len, head_dim) = (2, 64, 40, 32);
        let view = cache_view(&dev, DType::F16, heads, max_ctx, len, head_dim, 0x9B1);
        let rows_t = Tensor::from_vec(vec![3u32, 40, 7], 3, &dev).unwrap();
        let got = dispatch::run_qsa_gather(&view, &rows_t).unwrap();
        let g = bits(&got);
        let src = bits(&view.contiguous().unwrap());
        for h in 0..heads {
            let sel = |s: usize| &g[(h * 3 + s) * head_dim..][..head_dim];
            assert_eq!(sel(0), &src[(h * len + 3) * head_dim..][..head_dim]);
            assert!(
                sel(1).iter().all(|&b| b == 0),
                "head {h}: row 40 is past len 40"
            );
            assert_eq!(sel(2), &src[(h * len + 7) * head_dim..][..head_dim]);
        }
    }

    /// A CPU source — the attention's non-Metal path reaches `gather_rows`
    /// above budget too — takes the candle chain without any switch set.
    #[test]
    fn a_cpu_source_takes_the_chain() {
        let (heads, len, head_dim) = (2, 10, 8);
        let view = Tensor::from_vec(
            pseudo_random(heads * 16 * head_dim, 0x9C1, -1.0, 1.0),
            (heads, 16, head_dim),
            &Device::Cpu,
        )
        .unwrap()
        .narrow(1, 0, len)
        .unwrap();
        let rows = vec![1u32, 4, 9];
        let rows_t = Tensor::from_vec(rows.clone(), 3, &Device::Cpu).unwrap();
        let got = gather_rows(&view, &rows_t).unwrap();
        assert_eq!(got.dims(), &[heads, 3, head_dim]);
        let src = bits(&view.contiguous().unwrap());
        let g = bits(&got);
        for h in 0..heads {
            for (s, &r) in rows.iter().enumerate() {
                assert_eq!(
                    &g[(h * 3 + s) * head_dim..][..head_dim],
                    &src[(h * len + r as usize) * head_dim..][..head_dim],
                    "head {h} selected {s}"
                );
            }
        }
    }

    /// The geometry the kernel does not cover is refused, not silently wrong.
    #[test]
    fn unsupported_shapes_are_refused() {
        let dev = metal_device().unwrap();
        let rows = Tensor::from_vec(vec![0u32], 1, &dev).unwrap();
        // head_dim not a multiple of 4.
        let odd = cache_view(&dev, DType::F16, 1, 8, 8, 6, 1);
        assert!(dispatch::run_qsa_gather(&odd, &rows).is_err());
        // rows of the wrong dtype.
        let view = cache_view(&dev, DType::F16, 1, 8, 8, 8, 2);
        let bad_rows = Tensor::from_vec(vec![0i64], 1, &dev).unwrap();
        assert!(dispatch::run_qsa_gather(&view, &bad_rows).is_err());
        // a source whose rows are not contiguous.
        let strided = view.transpose(1, 2).unwrap();
        assert!(dispatch::run_qsa_gather(&strided, &rows).is_err());
    }
}
