//! Device PLE tail with host-owned convolution state. The normalized key and
//! projected value remain device tensors; only normalized history rows need
//! downloading after the tail. The frozen oracle is qwen4exp::ref_ple.
use anyhow::{Result, ensure};
use candle_core::{DType, Device, Storage, Tensor};

/// Returns `(addend, normalized_gated)`, both token-major `[n, width]`.
/// `prior` is channel-major `[width, (kernel-1)*dilation]`, oldest first.
/// Norm weights are `[width]`, convolution weights `[width, kernel]`, and
/// `value` is `[n, hidden]`, shared across the `width/hidden` streams.
#[allow(clippy::too_many_arguments)]
pub fn ple_tail(
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
    super::dispatch::run_ple_tail(
        key, value, stream, query_w, norm_w, conv_w, prior, hidden, dilation, eps,
    )
}

/// Download exactly the last `rows` normalized rows, including offset views.
/// Candle's ordinary readback of a narrow view copies the whole underlying
/// allocation, so this blits just the tail rows into its own staging buffer.
pub fn readback_tail(normalized: &Tensor, rows: usize) -> Result<Vec<f32>> {
    let (n, width) = normalized.dims2()?;
    ensure!(rows <= n, "PLE tail readback exceeds the token count");
    ensure!(
        normalized.dtype() == DType::F32 && normalized.is_contiguous(),
        "PLE tail readback needs contiguous f32"
    );
    let Device::Metal(mdev) = normalized.device() else {
        anyhow::bail!("PLE tail readback needs Metal")
    };
    if rows == 0 {
        return Ok(Vec::new());
    }
    let count = rows
        .checked_mul(width)
        .ok_or_else(|| anyhow::anyhow!("PLE tail readback size overflow"))?;
    let bytes = count
        .checked_mul(4)
        .ok_or_else(|| anyhow::anyhow!("PLE tail readback size overflow"))?;
    let staging = mdev
        .new_buffer_builder()
        .with_size(bytes)
        .with_label("ple_tail_readback")
        .build()?;
    let (storage, layout) = normalized.storage_and_layout();
    let Storage::Metal(storage) = &*storage else {
        anyhow::bail!("PLE tail readback needs Metal storage")
    };
    let start = (n - rows)
        .checked_mul(width)
        .and_then(|skip| layout.start_offset().checked_add(skip))
        .and_then(|v| v.checked_mul(4))
        .ok_or_else(|| anyhow::anyhow!("PLE tail readback offset overflow"))?;
    {
        let mut blit = mdev.blit_command_encoder()?;
        blit.copy_from_buffer(storage.buffer(), start, &staging, 0, bytes);
    }
    mdev.flush_and_wait_current()?;
    let ptr = staging.contents() as *const f32;
    ensure!(!ptr.is_null(), "PLE staging buffer is not CPU accessible");
    // SAFETY: the completed blit initialized exactly count f32s in CPU-accessible
    // storage. Source and staging allocations stay alive through this copy.
    Ok(unsafe { std::slice::from_raw_parts(ptr, count) }.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::metal_device;
    use crate::ops::dispatch::testutil::{pseudo_random, rel_l2};
    use crate::qwen4exp::ref_hc::grouped_rms_norm;
    use crate::qwen4exp::ref_ple::{PleLayerRef, gate_function_probe};
    const EPS: f32 = 1e-6;

    fn tensor(v: &[f32], shape: impl Into<candle_core::Shape>, dev: &Device) -> Tensor {
        Tensor::from_vec(v.to_vec(), shape, dev).unwrap()
    }
    fn flat(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1().unwrap()
    }
    fn close(got: &[f32], want: &[f32], label: &str) {
        assert_eq!(got.len(), want.len());
        let err = rel_l2(got, want);
        assert!(err <= 1e-5, "{label}: rel_l2 {err}");
        assert!(
            got.iter().all(|x| x.is_finite()),
            "{label}: nonfinite output"
        );
    }
    fn reference(hidden: usize, streams: usize, k: usize, dilation: usize) -> PleLayerRef {
        let width = hidden * streams;
        PleLayerRef {
            hidden,
            hc_count: streams,
            n_heads: 1,
            head_dim: 1,
            table: vec![0.7, -1.2, 0.01],
            key_w: pseudo_random(width, 1, -1.0, 1.0),
            key_norm_w: pseudo_random(width, 2, 0.9, 1.1),
            value_w: pseudo_random(hidden, 3, -1.0, 1.0),
            query_norm_w: pseudo_random(width, 4, 0.9, 1.1),
            conv_norm_w: pseudo_random(width, 5, 0.9, 1.1),
            conv_w: pseudo_random(width * k, 6, -0.5, 0.5),
            k,
            ngram_size: dilation,
            eps: EPS,
        }
    }
    fn inputs(r: &PleLayerRef, rows: &[u64]) -> (Vec<f32>, Vec<f32>) {
        let mut key = Vec::new();
        let mut value = Vec::new();
        for &row in rows {
            let emb = r.table[row as usize];
            let raw: Vec<_> = r.key_w.iter().map(|w| w * emb).collect();
            key.extend(grouped_rms_norm(&raw, &r.key_norm_w, r.hidden, EPS));
            value.extend(r.value_w.iter().map(|w| w * emb));
        }
        (key, value)
    }
    fn run(
        r: &PleLayerRef,
        key: &[f32],
        value: &[f32],
        stream: &[f32],
        prior: &[f32],
        dev: &Device,
    ) -> (Tensor, Tensor) {
        let width = r.width();
        let n = stream.len() / width;
        ple_tail(
            &tensor(key, (n, width), dev),
            &tensor(value, (n, r.hidden), dev),
            &tensor(stream, (n, width), dev),
            &tensor(&r.query_norm_w, width, dev),
            &tensor(&r.conv_norm_w, width, dev),
            &tensor(&r.conv_w, (width, r.k), dev),
            &tensor(prior, (width, r.conv_state_len()), dev),
            r.hidden,
            r.dilation(),
            EPS,
        )
        .unwrap()
    }
    fn update_state(prior: &[f32], rows: &[f32], width: usize, state_len: usize) -> Vec<f32> {
        let n = rows.len() / width;
        let mut out = vec![0.0; width * state_len];
        for c in 0..width {
            for j in 0..state_len {
                out[c * state_len + j] = if j + n < state_len {
                    prior[c * state_len + j + n]
                } else {
                    rows[(j + n - state_len) * width + c]
                };
            }
        }
        out
    }

    #[test]
    fn tail_matches_frozen_oracle_and_compact_history() {
        let dev = metal_device().unwrap();
        for &(hidden, streams, n) in &[
            (3, 2, 1),
            (7, 3, 2),
            (33, 2, 4),
            (13, 2, 8),
            (13, 2, 9),
            (13, 2, 10),
            (2560, 4, 1),
            (2560, 4, 13),
            (2560, 4, 2048),
        ] {
            let r = reference(hidden, streams, 4, 3);
            let width = r.width();
            let rows: Vec<_> = (0..n).map(|i| (i % 3) as u64).collect();
            let stream = pseudo_random(n * width, 31, -2.0, 2.0);
            let prior = pseudo_random(width * r.conv_state_len(), 32, -0.5, 0.5);
            let mut expected_state = prior.clone();
            let expected = r.forward(&rows, &stream, &mut expected_state);
            let (key, value) = inputs(&r, &rows);
            let (out, normed) = run(&r, &key, &value, &stream, &prior, &dev);
            close(
                &flat(&out),
                &expected.output,
                &format!("addend n={n} hidden={hidden}"),
            );
            close(&flat(&normed), &expected.gated_normed, "normalized gate");
            let tail = readback_tail(&normed, n.min(r.conv_state_len())).unwrap();
            let got_state = update_state(&prior, &tail, width, r.conv_state_len());
            close(&got_state, &expected_state, "compact next state");
        }
    }

    #[test]
    fn chunked_history_matches_frozen_oracle() {
        let dev = metal_device().unwrap();
        let r = reference(33, 3, 4, 3);
        let width = r.width();
        let mut expected_state = r.zero_conv_state();
        let mut state = expected_state.clone();
        for (step, n) in [2, 1, 3, 9, 11, 1].into_iter().enumerate() {
            let rows: Vec<_> = (0..n).map(|i| ((step + i) % 3) as u64).collect();
            let stream = pseudo_random(n * width, 110 + step as u64, -2.0, 2.0);
            let expected = r.forward(&rows, &stream, &mut expected_state);
            let (key, value) = inputs(&r, &rows);
            let (out, normed) = run(&r, &key, &value, &stream, &state, &dev);
            close(&flat(&out), &expected.output, "chunked addend");
            let tail = readback_tail(&normed, n.min(r.conv_state_len())).unwrap();
            state = update_state(&state, &tail, width, r.conv_state_len());
            close(&state, &expected_state, "chunked next state");
        }
    }

    #[test]
    fn gate_zero_near_zero_and_nan() {
        let dev = metal_device().unwrap();
        for raw in [0.0f32, -0.0, 1e-12, -1e-12, 0.25, -0.25, f32::NAN] {
            // Unit query norm and zero conv isolate the scalar gate in addend.
            let stream = [1.0];
            let query_w = [(1.0 + EPS).sqrt()];
            let (out, _) = ple_tail(
                &tensor(&[raw], (1, 1), &dev),
                &tensor(&[1.0], (1, 1), &dev),
                &tensor(&stream, (1, 1), &dev),
                &tensor(&query_w, 1, &dev),
                &tensor(&[1.0], 1, &dev),
                &tensor(&[0.0], (1, 1), &dev),
                &tensor(&[0.0], 1, &dev)
                    .narrow(0, 0, 0)
                    .unwrap()
                    .reshape((1, 0))
                    .unwrap(),
                1,
                2,
                EPS,
            )
            .unwrap();
            let got = flat(&out)[0];
            let want = gate_function_probe(raw);
            if raw.is_nan() {
                assert!(got.is_nan());
            } else {
                assert!((got - want).abs() < 2e-7, "raw={raw}: {got} vs {want}");
            }
        }
    }

    #[test]
    fn offset_operands_and_readback_match_materialized() {
        let dev = metal_device().unwrap();
        let r = reference(7, 2, 4, 3);
        let width = r.width();
        let n = 5;
        let rows = vec![0; n];
        let (key, value) = inputs(&r, &rows);
        let stream = pseudo_random(n * width, 44, -2.0, 2.0);
        let prior = pseudo_random(width * r.conv_state_len(), 45, -0.5, 0.5);
        let operands = [
            tensor(&key, (n, width), &dev),
            tensor(&value, (n, r.hidden), &dev),
            tensor(&stream, (n, width), &dev),
            tensor(&r.query_norm_w, width, &dev),
            tensor(&r.conv_norm_w, width, &dev),
            tensor(&r.conv_w, (width, r.k), &dev),
            tensor(&prior, (width, r.conv_state_len()), &dev),
        ];
        let views: Vec<_> = operands
            .iter()
            .map(|t| {
                let flat = t.flatten_all().unwrap();
                let padded = Tensor::cat(&[&flat, &flat], 0).unwrap();
                padded
                    .narrow(0, t.elem_count(), t.elem_count())
                    .unwrap()
                    .reshape(t.shape())
                    .unwrap()
            })
            .collect();
        let call = |v: &[Tensor]| {
            ple_tail(
                &v[0],
                &v[1],
                &v[2],
                &v[3],
                &v[4],
                &v[5],
                &v[6],
                r.hidden,
                r.dilation(),
                EPS,
            )
            .unwrap()
        };
        let (a, an) = call(&operands);
        let (b, bn) = call(&views);
        assert_eq!(flat(&a), flat(&b));
        assert_eq!(flat(&an), flat(&bn));
        let bigger = Tensor::cat(&[&an, &an], 0).unwrap();
        let view = bigger.narrow(0, n, n).unwrap();
        assert_eq!(
            readback_tail(&view, 2).unwrap(),
            flat(&an)[(n - 2) * width..]
        );
        assert!(readback_tail(&view, n + 1).is_err());
        assert!(
            ple_tail(
                &operands[0],
                &operands[1],
                &operands[2],
                &operands[3],
                &operands[4],
                &operands[5],
                &operands[6],
                0,
                3,
                EPS
            )
            .is_err()
        );
        assert!(
            ple_tail(
                &operands[0],
                &operands[1],
                &operands[2],
                &operands[3],
                &operands[4],
                &operands[5],
                &operands[6],
                7,
                0,
                EPS
            )
            .is_err()
        );
    }

    // Literal host tail as shipped: frozen grouped norms/scalar gate, padded
    // channel-major history, sequential dot and convolution accumulations.
    fn host_tail(
        r: &PleLayerRef,
        key: &[f32],
        value: &[f32],
        stream: &[f32],
        prior: &[f32],
    ) -> (Vec<f32>, Vec<f32>) {
        let width = r.width();
        let n = stream.len() / width;
        let state_len = r.conv_state_len();
        let mut gated = vec![0.0; n * width];
        let mut normed = vec![0.0; n * width];
        for t in 0..n {
            let row = t * width..(t + 1) * width;
            let q = grouped_rms_norm(&stream[row.clone()], &r.query_norm_w, r.hidden, EPS);
            for s in 0..r.hc_count {
                let lo = s * r.hidden;
                let dot: f32 = key[t * width + lo..t * width + lo + r.hidden]
                    .iter()
                    .zip(&q[lo..lo + r.hidden])
                    .map(|(a, b)| a * b)
                    .sum();
                let g = gate_function_probe(dot * (1.0 / (r.hidden as f32).sqrt()));
                for j in 0..r.hidden {
                    gated[t * width + lo + j] = g * value[t * r.hidden + j];
                }
            }
            normed[row.clone()].copy_from_slice(&grouped_rms_norm(
                &gated[row],
                &r.conv_norm_w,
                r.hidden,
                EPS,
            ));
        }
        let line_len = state_len + n;
        let mut padded = vec![0.0; width * line_len];
        for c in 0..width {
            let line = &mut padded[c * line_len..(c + 1) * line_len];
            line[..state_len].copy_from_slice(&prior[c * state_len..(c + 1) * state_len]);
            for t in 0..n {
                line[state_len + t] = normed[t * width + c];
            }
        }
        let mut out = vec![0.0; n * width];
        let mut state = vec![0.0; width * state_len];
        for c in 0..width {
            let line = &padded[c * line_len..(c + 1) * line_len];
            for t in 0..n {
                let acc: f32 = (0..r.k)
                    .map(|j| {
                        r.conv_w[c * r.k + j] * line[state_len + t - (r.k - 1 - j) * r.dilation()]
                    })
                    .sum();
                out[t * width + c] = gated[t * width + c] + acc * (1.0 / (1.0 + (-acc).exp()));
            }
            state[c * state_len..(c + 1) * state_len]
                .copy_from_slice(&line[line_len - state_len..]);
        }
        (out, state)
    }

    #[test]
    #[ignore = "manual amortized PLE benchmark; exclusive GPU, report pmset verbatim"]
    fn ple_tail_bench() {
        use std::{hint::black_box, time::Instant};
        let dev = metal_device().unwrap();
        let r = reference(2560, 4, 4, 3);
        let width = r.width();
        let sizes: Vec<usize> = std::env::var("XWEN_PLE_BENCH_N")
            .map(|v| vec![v.parse().expect("XWEN_PLE_BENCH_N must be an integer")])
            .unwrap_or_else(|_| vec![1, 512, 2048]);
        let rounds: usize = std::env::var("XWEN_PLE_BENCH_ROUNDS")
            .map(|v| v.parse().unwrap())
            .unwrap_or(4);
        let power = std::process::Command::new("pmset")
            .arg("-g")
            .output()
            .unwrap();
        eprintln!("{}", String::from_utf8_lossy(&power.stdout));
        for n in sizes {
            assert!(n > 0);
            let batch = if n == 1 { 64 } else { 4 };
            let rows = vec![0; n];
            let (key, value) = inputs(&r, &rows);
            let stream = pseudo_random(n * width, 91, -2.0, 2.0);
            let prior = pseudo_random(width * r.conv_state_len(), 92, -0.5, 0.5);
            let key_t = tensor(&key, (n, width), &dev);
            let value_t = tensor(&value, (n, r.hidden), &dev);
            let stream_t = tensor(&stream, (n, width), &dev);
            let qw = tensor(&r.query_norm_w, width, &dev);
            let nw = tensor(&r.conv_norm_w, width, &dev);
            let cw = tensor(&r.conv_w, (width, r.k), &dev);
            let prior_t = tensor(&prior, (width, r.conv_state_len()), &dev);
            let kernel = |p: &Tensor| {
                ple_tail(
                    &key_t,
                    &value_t,
                    &stream_t,
                    &qw,
                    &nw,
                    &cw,
                    p,
                    r.hidden,
                    r.dilation(),
                    EPS,
                )
                .unwrap()
            };
            let warm = kernel(&prior_t);
            dev.synchronize().unwrap();
            drop(warm);
            let expected = host_tail(&r, &key, &value, &stream, &prior);
            let oracle = r.forward(&rows, &stream, &mut prior.clone());
            close(&expected.0, &oracle.output, "benchmark host transcription");
            for round in 0..rounds {
                if round > 0 {
                    eprintln!("PLE tail idle 60s before round {round}");
                    std::thread::sleep(std::time::Duration::from_secs(60));
                }
                let arms = if round % 2 == 0 {
                    [0, 1, 2, 3]
                } else {
                    [3, 2, 1, 0]
                };
                for arm in arms {
                    dev.synchronize().unwrap();
                    let start = Instant::now();
                    match arm {
                        0 => {
                            for _ in 0..batch {
                                black_box(host_tail(&r, &key, &value, &stream, &prior));
                            }
                        }
                        1 => {
                            let mut held = Vec::with_capacity(batch);
                            for _ in 0..batch {
                                held.push(kernel(&prior_t));
                            }
                            dev.synchronize().unwrap();
                            black_box(held);
                        }
                        2 => {
                            for _ in 0..batch {
                                let p = tensor(&prior, (width, r.conv_state_len()), &dev);
                                let (out, norm) = kernel(&p);
                                let tail = readback_tail(&norm, n.min(r.conv_state_len())).unwrap();
                                black_box(update_state(&prior, &tail, width, r.conv_state_len()));
                                black_box(out);
                            }
                        }
                        _ => {
                            for _ in 0..batch {
                                let [kh, vh, sh] = crate::qwen4exp::ple::readback_inputs(
                                    [&key_t, &value_t, &stream_t],
                                    n != 1,
                                )
                                .unwrap();
                                let (out, state) = host_tail(&r, &kh, &vh, &sh, &prior);
                                black_box(tensor(&out, (n, width), &dev));
                                black_box(state);
                            }
                        }
                    }
                    dev.synchronize().unwrap();
                    eprintln!(
                        "PLE tail n={n} round={round} arm={} batch={batch} {:.6} ms/op",
                        [
                            "host_arithmetic",
                            "device_amortized",
                            "device_transaction",
                            "classic_transaction"
                        ][arm],
                        start.elapsed().as_secs_f64() * 1000.0 / batch as f64
                    );
                }
            }
        }
    }
}
