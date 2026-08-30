//! Host side of the QSA decode block selection — the top-k over the indexer's
//! block scores and its expansion into the row list the attention gathers,
//! run on device by `kernel_qsa_select` (qsa_select.metal) so a decode step
//! never reads the scores back. `src/qwen4exp/indexer.rs` is the caller, for
//! a single-token step above the token budget only; the shipped 3.6/3.8
//! checkpoints have no indexer and never reach it.
//!
//! One dispatch per QSA layer per step in place of a `to_vec1` readback: on
//! Flash-Next that was 12 pipeline drains per token, each stalling the CPU's
//! encoding of the next layer until the GPU had finished this one. The kernel
//! implements the host's own total order (score descending, block index
//! ascending) over the canonicalized score bits, so its rows are the host's
//! rows exactly; the kill switch back to the readback is `XWEN_QSA_HOST_TOPK`
//! ([`crate::ops::qsa_host_topk`]), and `XWEN_QSA_CLASSIC` implies it. Scores
//! that are not on a Metal device take the host path too, with no switch.

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
}
