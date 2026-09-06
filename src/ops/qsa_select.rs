//! Host side of the QSA block selection — the top-k over the indexer's block
//! scores, run on device so neither a decode step nor a prefill chunk reads
//! the scores back. Two kernels in qsa_select.metal share the ranking:
//! `kernel_qsa_select` expands one query's selection into the row list the
//! attention gathers ([`select_rows`]), and `kernel_qsa_select_mask` writes a
//! whole chunk's `[n, n_kv]` additive mask, one threadgroup per query
//! ([`select_mask`]). `src/qwen4exp/indexer.rs` is the caller, above the
//! token budget only; the shipped 3.6/3.8 checkpoints have no indexer and
//! never reach either.
//!
//! One dispatch per QSA layer per step in place of a `to_vec1` readback: on
//! Flash-Next that was 12 pipeline drains per token, each stalling the CPU's
//! encoding of the next layer until the GPU had finished this one. The kernel
//! implements the host's own total order (score descending, block index
//! ascending) over the canonicalized score bits, so its rows are the host's
//! rows exactly; the kill switches back to the readback are
//! `XWEN_QSA_HOST_TOPK` at decode ([`crate::ops::qsa_host_topk`]) and
//! `XWEN_QSA_HOST_MASK` at prefill ([`crate::ops::qsa_host_mask`]), and
//! `XWEN_QSA_CLASSIC` implies both. Scores that are not on a Metal device
//! take the host path too, with no switch.

use anyhow::Result;
use candle_core::Tensor;

use crate::ops::dispatch;

/// The ascending row list for one query: the top-`keep` of the `nb` block
/// scores `scores` (f32 `[nb]`, ≥ 0), each block's `ratio` positions in order,
/// then the `tail` positions `nb * ratio ..` above the last complete block.
/// u32 `[keep * ratio + tail]`, on `scores`' device. `keep` must be in
/// `1..=nb` and `tail` below `ratio`.
pub fn select_rows(scores: &Tensor, keep: usize, ratio: usize, tail: usize) -> Result<Tensor> {
    dispatch::run_qsa_select(scores, keep, ratio, tail)
}

/// The additive f32 mask of one prefill chunk: for each query `i` of
/// `scores` (f32 `[n, n_blocks]`, at absolute position `pos + i`), `-inf`
/// over its `[pos + n]` row except `0` at the positions of its top
/// `min(keep_max, nb_i)` of the `nb_i = min((pos + i + 1) / ratio, n_blocks)`
/// blocks it sees and at its raw tail. `[n, pos + n]`, on `scores`' device;
/// the same bits as `QsaIndexer::top_blocks` + `expand_into` + the host
/// fill. `n_blocks * ratio` must fit within `pos + n`.
pub fn select_mask(scores: &Tensor, pos: usize, ratio: usize, keep_max: usize) -> Result<Tensor> {
    dispatch::run_qsa_select_mask(scores, pos, ratio, keep_max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::metal_device;
    use candle_core::{DType, Device};

    /// Every precondition `run_qsa_select` states is refused before a
    /// dispatch, not discovered as garbage rows.
    #[test]
    fn unsupported_inputs_are_refused() {
        let dev = metal_device().unwrap();
        let scores = Tensor::from_vec(vec![1.0f32, 0.5, 0.0, 2.0], 4, &dev).unwrap();
        assert!(
            select_rows(&scores, 2, 4, 1).is_ok(),
            "the well-formed case"
        );
        // scores on the CPU
        let cpu = Tensor::from_vec(vec![1.0f32, 0.5, 0.0, 2.0], 4, &Device::Cpu).unwrap();
        assert!(select_rows(&cpu, 2, 4, 1).is_err());
        // not f32
        let f16 = scores.to_dtype(DType::F16).unwrap();
        assert!(select_rows(&f16, 2, 4, 1).is_err());
        // not contiguous
        let strided = Tensor::from_vec(vec![1.0f32; 8], (4, 2), &dev)
            .unwrap()
            .t()
            .unwrap()
            .narrow(0, 0, 1)
            .unwrap()
            .squeeze(0)
            .unwrap();
        assert!(!strided.is_contiguous());
        assert!(select_rows(&strided, 2, 4, 1).is_err());
        // no blocks
        let empty = Tensor::zeros(0, DType::F32, &dev).unwrap();
        assert!(select_rows(&empty, 1, 4, 0).is_err());
        // keep out of range
        assert!(select_rows(&scores, 0, 4, 1).is_err());
        assert!(select_rows(&scores, 5, 4, 1).is_err());
        // tail not below the ratio
        assert!(select_rows(&scores, 2, 4, 4).is_err());
        assert!(select_rows(&scores, 2, 0, 0).is_err());
    }

    /// `run_qsa_select_mask`'s preconditions are refused before a dispatch.
    #[test]
    fn unsupported_mask_inputs_are_refused() {
        let dev = metal_device().unwrap();
        let plane = Tensor::from_vec(vec![1.0f32, 0.5, 0.0, 2.0, 0.0, 0.0], (2, 3), &dev).unwrap();
        assert!(
            select_mask(&plane, 10, 4, 2).is_ok(),
            "the well-formed case"
        );
        // blocks beyond the row: 3 blocks of 4 need 12 positions, pos + n is 11
        assert!(select_mask(&plane, 9, 4, 2).is_err());
        // zero ratio
        assert!(select_mask(&plane, 10, 0, 2).is_err());
        // pos + n overflows
        assert!(select_mask(&plane, usize::MAX, 4, 2).is_err());
        // scores on the CPU
        let cpu = plane.to_device(&Device::Cpu).unwrap();
        assert!(select_mask(&cpu, 10, 4, 2).is_err());
        // not f32
        assert!(select_mask(&plane.to_dtype(DType::F16).unwrap(), 10, 4, 2).is_err());
        // not rank 2
        assert!(select_mask(&plane.flatten_all().unwrap(), 10, 4, 2).is_err());
        // not contiguous
        assert!(select_mask(&plane.t().unwrap(), 10, 4, 2).is_err());
    }
}
