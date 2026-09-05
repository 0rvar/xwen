use anyhow::Result;
use candle_core::Tensor;
use candle_metal_kernels::metal::Buffer;

use crate::ops::dispatch;

/// q8_0-weight x f32-activation mat-vec against the vendored ggml-geometry kernel
/// (`q8.metal`) — the attention DECODE gemv (seq <= 8) of a q8_0-quantized
/// checkpoint. `weight` is the rank-2 `[n_out, k]` q8_0 tensor's raw device
/// buffer, bound at `w_off` (the mmap alias's `base_off`, 0 for the classic
/// private copy); `x` is `[t, k]` f32; returns `[t, n_out]` f32. Semantically the
/// fork's mixed-dtype mul_mat: f32 products/accumulation, f32 output, with the
/// stored q8_0 weights as the only quantized values (no activation or output
/// rounding). Prefill (seq > 8) stays on the f16 dense plane (`matmul_f16`), so
/// this handles the gemv only. Metal only; the caller's fallback is the
/// dequant-f16 dense plane (`XWEN_ATTN_DEQUANT`) or the dequant-f32 `QMatMul`
/// path (`XWEN_ATTN_F32`), which bypass this module entirely.
pub fn matmul_q8(
    weight: &Buffer,
    w_off: usize,
    n_out: usize,
    k: usize,
    x: &Tensor,
) -> Result<Tensor> {
    dispatch::run_matmul_q8(weight, w_off, n_out, k, x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::metal_device;
    use crate::ops::dispatch::testutil::{max_abs, pseudo_random, rel_l2};
    use candle_core::quantized::{GgmlDType, QStorage, QTensor};
    use candle_core::{DType, Device, Module, Tensor};
    use std::sync::Arc;

    /// Build a rank-2 `[n_out, k]` q8_0 weight sharing one allocation between a
    /// retained Metal `Buffer` and a `QMatMul`, exactly as `qlinear_with_buffer`
    /// does at load. Returns (buffer, QMatMul oracle, CPU-dequantized weights).
    fn build_q8_weight(
        device: &Device,
        n_out: usize,
        k: usize,
        seed: u64,
    ) -> (Arc<Buffer>, candle_core::quantized::QMatMul, Vec<f32>) {
        let w = pseudo_random(n_out * k, seed, -0.5, 0.5);
        let w_t = Tensor::from_vec(w, (n_out, k), device).unwrap();
        let qt = QTensor::quantize(&w_t, GgmlDType::Q8_0).unwrap();
        let deq = qt
            .dequantize(&Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        let storage = QStorage::from_data(qt.data().unwrap(), device, GgmlDType::Q8_0).unwrap();
        let buffer = match &storage {
            QStorage::Metal(qms) => Arc::new(qms.buffer().clone()),
            _ => panic!("expected Metal storage"),
        };
        let qtensor = Arc::new(QTensor::new(storage, (n_out, k)).unwrap());
        let qmm = candle_core::quantized::QMatMul::from_arc(qtensor).unwrap();
        (buffer, qmm, deq)
    }

    /// The vendored q8_0 gemv vs a CPU f32 reference over the SAME q8_0-rounded
    /// weights (the kernel's only rounding, which the reference shares): the
    /// residual is pure f32 accumulation-order noise. Also sanity-checks candle's
    /// baked QMatMul over the shared allocation tracks the same reference.
    fn q8_case(n_out: usize, k: usize, t: usize, seed: u64) -> (f32, f32, f32) {
        let device = metal_device().unwrap();
        let (buffer, qmm, deq) = build_q8_weight(&device, n_out, k, seed);

        let x_vec = pseudo_random(t * k, seed ^ 0x5A5A, -1.0, 1.0);
        let x = Tensor::from_vec(x_vec.clone(), (t, k), &device).unwrap();

        let out = matmul_q8(&buffer, 0, n_out, k, &x).unwrap();
        assert_eq!(out.dims(), &[t, n_out]);
        assert_eq!(out.dtype(), DType::F32);
        let got = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let mut want = vec![0f32; t * n_out];
        for ti in 0..t {
            for o in 0..n_out {
                let mut acc = 0f32;
                for i in 0..k {
                    acc += deq[o * k + i] * x_vec[ti * k + i];
                }
                want[ti * n_out + o] = acc;
            }
        }

        let qmm_out = qmm
            .forward(&x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        (
            rel_l2(&got, &want),
            max_abs(&got, &want),
            rel_l2(&qmm_out, &want),
        )
    }

    /// Diagnostic, not a gate: prints the measured q8-gemv error magnitude vs
    /// the CPU f32 reference at production shapes. Distinguishes pure f32
    /// accumulation-order noise (~1e-7 rel) from precision loss at f16-rounding
    /// scale (~1e-4) that the 1e-3 gate bound cannot resolve.
    #[test]
    #[ignore]
    fn q8_error_magnitude_probe() {
        for (n_out, k) in [
            (6144usize, 3072usize),
            (1024, 3072),
            (48, 3072),
            (72, 3072),
            (3072, 6144),
            (3072, 9216),
        ] {
            let (rel, max, rel_qmm) = q8_case(n_out, k, 1, 0x51 + n_out as u64);
            eprintln!(
                "q8 gemv [{n_out}x{k}] t=1: rel_l2 {rel:.3e} max_abs {max:.3e} (QMatMul oracle rel_l2 {rel_qmm:.3e})"
            );
        }
    }

    /// The production attention projection shapes at decode seqs (t <= 8 — the
    /// q8 gemv's active range, up to and including the f16 mv/mm break-even). q8_0
    /// accumulates f32 with weight-only rounding, so the error is f32
    /// accumulation-order noise; bound at 1e-3 (well clear).
    #[test]
    fn q8_decode_production_shapes() {
        for (n_out, k) in [
            (6144, 3072),
            (9216, 3072),
            (1024, 3072),
            (48, 3072),
            (72, 3072),
            (3072, 9216),
        ] {
            let (rel, max, rel_qmm) = q8_case(n_out, k, 1, 0x51 + n_out as u64);
            assert!(
                rel < 1e-3,
                "q8 gemv [{n_out}x{k}] t=1 rel_l2 {rel} (max_abs {max})"
            );
            assert!(
                rel_qmm < 1e-3,
                "QMatMul oracle sanity [{n_out}x{k}] rel_l2 {rel_qmm}"
            );
        }
        // The top of the q8 gemv range: t=8 (the f16 mv/mm break-even — Proj runs
        // the q8 gemv here, not the f16 plane).
        let (rel, max, _) = q8_case(1024, 3072, 8, 0x61);
        assert!(rel < 1e-3, "q8 gemv t=8 rel_l2 {rel} (max_abs {max})");
        // A ragged out-dim (not a multiple of N_R0=2) exercises the helper's
        // `r0 + row < ne01` store guard.
        let (rel, max, _) = q8_case(31, 256, 2, 0x71);
        assert!(rel < 1e-3, "q8 gemv ragged rel_l2 {rel} (max_abs {max})");
    }

    /// The vendored gemv must honor a nonzero weight buffer offset (the mmap alias
    /// binds a page-floored view, so the tensor sits at a sub-page `base_off`):
    /// the same q8_0 bytes uploaded at offset 0 and behind a 32-byte-aligned pad
    /// must produce bitwise-identical results.
    #[test]
    fn q8_gemv_honors_base_off() {
        let device = metal_device().unwrap();
        let (n_out, k, t) = (64usize, 256usize, 3usize);
        let (buffer, _, _) = build_q8_weight(&device, n_out, k, 0x0FF5E7);

        let bytes: Vec<u8> = {
            let w = pseudo_random(n_out * k, 0x0FF5E7, -0.5, 0.5);
            let qt = QTensor::quantize(
                &Tensor::from_vec(w, (n_out, k), &device).unwrap(),
                GgmlDType::Q8_0,
            )
            .unwrap();
            qt.data().unwrap().into_owned()
        };
        // The same quantized bytes again, 96 bytes (32-aligned, like a GGUF
        // tensor offset inside its page) into a fresh buffer.
        let base_off = 96usize;
        let mut padded = vec![0u8; base_off];
        padded.extend_from_slice(&bytes);
        let Device::Metal(mdev) = &device else {
            unreachable!()
        };
        let offset_buf = mdev.new_buffer_with_data(&padded).unwrap();

        let x = Tensor::from_vec(pseudo_random(t * k, 0xABCD, -1.0, 1.0), (t, k), &device).unwrap();
        let want = matmul_q8(&buffer, 0, n_out, k, &x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let got = matmul_q8(&offset_buf, base_off, n_out, k, &x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(
                g.to_bits(),
                w.to_bits(),
                "q8 gemv with base_off {base_off} differs at element {i}: {g} vs {w}"
            );
        }
    }

    #[test]
    fn q8_shape_and_dtype_errors() {
        let device = metal_device().unwrap();
        let (buffer, _, _) = build_q8_weight(&device, 64, 256, 0x900);
        // k mismatch.
        let x = Tensor::from_vec(vec![0f32; 128], (1, 128), &device).unwrap();
        assert!(matmul_q8(&buffer, 0, 64, 256, &x).is_err());
        // k not a multiple of 32.
        let x = Tensor::from_vec(vec![0f32; 20], (1, 20), &device).unwrap();
        assert!(matmul_q8(&buffer, 0, 64, 20, &x).is_err());
    }

    /// A synthetic `[n_out, k]` q8_0 weight plane as a bare device `Buffer`,
    /// built from raw block bytes rather than `QTensor::quantize`. Timing does
    /// not depend on the weight VALUES, and quantizing a 26M-element tensor on
    /// the CPU once per plane would dominate the bench's wall time; the deltas
    /// are a fixed, finite half so no denormal/NaN path can perturb the timing.
    fn synthetic_q8_plane(device: &Device, n_out: usize, k: usize, seed: u64) -> Arc<Buffer> {
        match synthetic_q8_storage(device, n_out, k, seed) {
            QStorage::Metal(qms) => Arc::new(qms.buffer().clone()),
            _ => panic!("expected Metal storage"),
        }
    }

    fn synthetic_q8_storage(device: &Device, n_out: usize, k: usize, seed: u64) -> QStorage {
        let blocks = n_out * k / 32;
        let mut raw = vec![0u8; blocks * 34];
        let d = half::f16::from_f32(0.0125).to_le_bytes();
        // One page of quants, tiled: distinct planes are distinct allocations,
        // which is all the cache-residency arms depend on.
        let mut pattern = [0u8; 4096];
        let mut s = seed | 1;
        for b in pattern.iter_mut() {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *b = (s >> 33) as u8;
        }
        for (b, chunk) in raw.chunks_exact_mut(34).enumerate() {
            chunk[0] = d[0];
            chunk[1] = d[1];
            let off = (b * 32) % (pattern.len() - 32);
            chunk[2..].copy_from_slice(&pattern[off..off + 32]);
        }
        QStorage::from_data(std::borrow::Cow::Borrowed(&raw), device, GgmlDType::Q8_0).unwrap()
    }

    /// Flash-Next hc decode: identical weight allocations, rotating 334 MB per
    /// shape, outputs retained through each sync. Run alone on an idle GPU;
    /// repeat with 60 s idle between invocations to check thermal drift.
    #[test]
    #[ignore = "perf bench"]
    fn hc_q8_decode_bench() {
        use candle_core::quantized::QMatMul;
        use std::time::Instant;
        let device = metal_device().unwrap();
        const PLANES: usize = 96;
        for (n_out, k) in [(320, 10240), (10240, 320)] {
            let weights: Vec<_> = (0..PLANES)
                .map(|i| {
                    let storage = synthetic_q8_storage(&device, n_out, k, i as u64 + 71);
                    let QStorage::Metal(qms) = &storage else {
                        unreachable!()
                    };
                    let buffer = Arc::new(qms.buffer().clone());
                    let qt = Arc::new(QTensor::new(storage, (n_out, k)).unwrap());
                    (buffer, QMatMul::from_arc(qt).unwrap())
                })
                .collect();
            let x = Tensor::from_vec(pseudo_random(k, 71, -1., 1.), (1, k), &device).unwrap();
            for round in 0..5 {
                for vendored in if round % 2 == 0 {
                    [false, true]
                } else {
                    [true, false]
                } {
                    device.synchronize().unwrap();
                    let start = Instant::now();
                    let mut outputs = Vec::with_capacity(PLANES);
                    for (buffer, qmm) in &weights {
                        outputs.push(if vendored {
                            matmul_q8(buffer, 0, n_out, k, &x).unwrap()
                        } else {
                            qmm.forward(&x).unwrap()
                        });
                    }
                    device.synchronize().unwrap();
                    let us = start.elapsed().as_secs_f64() * 1e6 / PLANES as f64;
                    if round > 0 {
                        eprintln!(
                            "hc [{n_out},{k}] round={round} vendored={vendored}: {us:.3} us/dispatch"
                        );
                    }
                    std::hint::black_box(outputs);
                }
            }
        }
    }

    /// Isolation timing for the vendored q8_0 decode gemv
    /// (`kernel_mul_mv_q8_0_f32_attn`) across output widths at fixed K, and
    /// across K at fixed output width — the question being why the GDN
    /// `attn_qkv` plane ([2560 -> 10240]) reads far below the rate its
    /// same-kernel siblings `attn_gate` ([2560 -> 6144]) and `ssm_out`
    /// ([6144 -> 2560]) reach in the same decode step.
    ///
    /// Two arms per shape, because the answer turns on where the weight bytes
    /// come from:
    ///   * `reuse`  — one plane, `BATCH` back-to-back dispatches per sync. The
    ///     plane is re-read every dispatch, so anything that fits the system
    ///     cache is served from it and the reported rate is not a DRAM rate.
    ///   * `rotate` — distinct planes covering ~`WORKING_SET` of weights,
    ///     dispatched round-robin. No plane can be cache-resident when its turn
    ///     comes back, which is the situation a real decode step is in (every
    ///     layer's projection is a different weight and the whole model streams
    ///     once per token).
    /// Both are amortized (BATCH dispatches per flush, every output held alive
    /// to the readback), per this repo's benching rules.
    ///
    /// `#[ignore]`d — run on a `pgrep`-verified free GPU with:
    ///   cargo test --release -p xwen q8_gemv_shape_sweep -- --ignored --nocapture
    /// `XWEN_BENCH_WARMUP` / `XWEN_BENCH_ITERS` override the loop counts.
    #[test]
    #[ignore = "perf bench"]
    fn q8_gemv_shape_sweep() {
        use std::time::Instant;

        let device = metal_device().unwrap();
        let get = |k: &str, d: usize| {
            std::env::var(k)
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(d)
        };
        let (warm, rounds) = (get("XWEN_BENCH_WARMUP", 3), get("XWEN_BENCH_ITERS", 15));
        const BATCH: usize = 32;
        // Comfortably past any plausible last-level cache on this part, so the
        // rotating arm is a DRAM measurement.
        const WORKING_SET: usize = 512 << 20;

        // (n_out, k, label)
        let shapes: [(usize, usize, &str); 8] = [
            (2560, 2560, ""),
            (6144, 2560, "attn_gate"),
            (8192, 2560, ""),
            (10240, 2560, "attn_qkv"),
            (12288, 2560, ""),
            (2560, 6144, "ssm_out"),
            (2560, 10240, ""),
            (6144, 6144, ""),
        ];

        eprintln!(
            "q8 gemv sweep: t=1, {BATCH} dispatches/round, {rounds} rounds \
             (rotate arm holds ~{} MB of planes)",
            WORKING_SET >> 20
        );
        for (n_out, k, label) in shapes {
            let bytes = (n_out * k / 32 * 34) as f64;
            let x = Tensor::from_vec(
                pseudo_random(k, 0x2000 + k as u64, -1.0, 1.0),
                (1, k),
                &device,
            )
            .unwrap();
            for (arm, n_planes, batch) in [
                ("reuse ", 1usize, BATCH),
                ("rotate", (WORKING_SET / (bytes as usize)).max(2), BATCH),
                // The condition XWEN_GDN_PROFILE actually measures in: every
                // step is bracketed by a `device.synchronize()`, so each
                // dispatch runs alone with nothing before or after it to
                // overlap with. Included because the production profile's
                // per-shape rates disagree with the amortized arms, and a
                // one-dispatch-per-flush arm is the only way to tell a kernel
                // property from a measurement property.
                ("synced", (WORKING_SET / (bytes as usize)).max(2), 1),
            ] {
                let planes: Vec<Arc<Buffer>> = (0..n_planes)
                    .map(|i| synthetic_q8_plane(&device, n_out, k, 0x900 + i as u64))
                    .collect();
                let mut sink = 0f32;
                let mut cursor = 0usize;
                let round = |cursor: &mut usize| {
                    let outs: Vec<Tensor> = (0..batch)
                        .map(|_| {
                            let p = &planes[*cursor % planes.len()];
                            *cursor += 1;
                            matmul_q8(p, 0, n_out, k, &x).unwrap()
                        })
                        .collect();
                    // ONE element, not the whole output: the readback is the
                    // round's only host transfer, and copying `n_out` floats
                    // would make it scale with the very axis being swept —
                    // which silently taxes the wide shapes in the
                    // one-dispatch-per-flush arm, where it is not amortized.
                    outs.last()
                        .unwrap()
                        .narrow(1, 0, 1)
                        .unwrap()
                        .to_vec2::<f32>()
                        .unwrap()[0][0]
                };
                for _ in 0..warm {
                    sink += round(&mut cursor);
                }
                let mut times = Vec::with_capacity(rounds);
                for _ in 0..rounds {
                    let t = Instant::now();
                    sink += round(&mut cursor);
                    times.push(t.elapsed().as_secs_f64() / batch as f64);
                }
                times.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let med = times[times.len() / 2];
                let best = times[0];
                eprintln!(
                    "  [{k:5} -> {n_out:5}] {arm} planes={n_planes:3} batch={batch:2} \
                     med {:7.1} us {:6.1} GB/s | best {:7.1} us {:6.1} GB/s  {label} (sink {sink:.1})",
                    med * 1e6,
                    bytes / med / 1e9,
                    best * 1e6,
                    bytes / best / 1e9,
                );
                drop(planes);
            }
        }
    }
}
