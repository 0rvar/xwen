//! THROWAWAY Phase-0 microbench (unstaged, not shipped, not wired into anything):
//! what does ONE Qwen3.8 MTP draft step cost, against one target decode forward?
//!
//! Run: `cargo run --release --example mtp_step_bench -- <mtp.gguf> <target.gguf>`
//!
//! It measures three things, all AMORTIZED per CLAUDE.md's benching rules —
//! BATCH dispatches per sync, every output held alive so nothing is optimized
//! away, medians over repeats, a warm-up batch discarded:
//!
//!   (a) one MTP draft step at seq=1: eh_proj over [norm(embed) ⊕ norm(hidden)],
//!       the full-attention layer (q/k/v, QK-norm, rope, sdpa over a KV cache of
//!       `--ctx` rows, o_proj), the SwiGLU FFN, and the lm_head matvec over the
//!       TARGET's Q6_K `output.weight`;
//!   (b) that lm_head matvec alone — the chain-drafter tax, the crux number;
//!   (c) one target decode forward, as ms/token.
//!
//! Correctness of the target graph is explicitly NOT the point: activations are
//! random-but-realistic and no reference is compared. What must be real is the
//! WEIGHTS (bytes moved is what is being measured), and they are, straight from
//! the shipped GGUFs through the shipped loader.
//!
//! WHERE THIS DIVERGES FROM THE SHIPPED HEAD, so nobody reads it back as a
//! fidelity benchmark once `src/mtp.rs` exists. Its attention is a hand-rolled
//! stand-in, not the trunk's `AttnBlock`: it splits q and the output gate by
//! HALVES where the real `attn_q` is per-head interleaved `[q, gate]`, it runs
//! neither the QK-RMSNorm nor the partial NEoX rope, and it never appends to a
//! KV cache — the `--ctx` rows it attends over are a fixed synthetic block. Each
//! of those changes the step's TIME by some amount this never measured, so the
//! per-step timings here are an estimate of the shipped step and not a
//! measurement of it.
//!
//! What survives unaffected is the byte budget, which is what Phase 0 actually
//! concluded from: the weights are the shipped ones at the shipped quantization,
//! so bytes moved per step, and the lm_head's share of them, are exact whatever
//! the graph around them does. The timing half was always the corroborating
//! evidence and the byte half the load-bearing one. Re-measure against the real
//! head (`MtpDrafter::step`) before quoting a per-step time again.
//!
//! (c) is measured in this same process by default, which loads target + sidecar
//! together (~21 GB); pass `--no-target-forward` to skip it when memory is tight
//! and derive ms/token from a plain decode run instead.

use std::time::Instant;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Module, Tensor};
use xwen::gguf::{self, Weights};

/// Dispatches per sync. The rate that matters is amortized: a per-dispatch
/// figure on this machine sums to more than the wall clock it claims to explain
/// (CLAUDE.md, "Benching rules").
const BATCH: usize = 32;
const REPEATS: usize = 7;

struct Args {
    mtp: String,
    target: String,
    ctx: usize,
    target_forward: bool,
}

fn parse_args() -> Result<Args> {
    let mut positional = Vec::new();
    let mut ctx = 1024;
    let mut target_forward = true;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--ctx" => ctx = it.next().context("--ctx needs a value")?.parse()?,
            "--no-target-forward" => target_forward = false,
            other => positional.push(other.to_string()),
        }
    }
    anyhow::ensure!(
        positional.len() == 2,
        "usage: mtp_step_bench <mtp.gguf> <target.gguf> [--ctx N] [--no-target-forward]"
    );
    Ok(Args {
        mtp: positional[0].clone(),
        target: positional[1].clone(),
        ctx,
        target_forward,
    })
}

/// Median of an amortized rate: `BATCH` dispatches, one sync, repeated.
fn amortized_ms(
    label: &str,
    device: &Device,
    mut step: impl FnMut() -> Result<Tensor>,
) -> Result<f64> {
    // Warm-up batch, discarded: first-dispatch pipeline compilation is not the
    // steady state this is asked about.
    {
        let mut alive = Vec::with_capacity(BATCH);
        for _ in 0..BATCH {
            alive.push(step()?);
        }
        device.synchronize()?;
        drop(alive);
    }
    let mut per_step = Vec::with_capacity(REPEATS);
    for _ in 0..REPEATS {
        // Outputs held alive across the whole batch: dropping each one inside
        // the loop would let the allocator recycle buffers the real pipeline
        // keeps, and would sync more than the batch intends.
        let mut alive = Vec::with_capacity(BATCH);
        let t0 = Instant::now();
        for _ in 0..BATCH {
            alive.push(step()?);
        }
        device.synchronize()?;
        per_step.push(t0.elapsed().as_secs_f64() * 1000.0 / BATCH as f64);
        drop(alive);
    }
    per_step.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = per_step[per_step.len() / 2];
    println!(
        "  {label:<28} {median:>8.3} ms/step   (runs {})",
        per_step
            .iter()
            .map(|v| format!("{v:.3}"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    Ok(median)
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let device = gguf::metal_device()?;

    // ---- the MTP sidecar: one 65th-layer-shaped block -----------------------
    let mtp = gguf::open(&args.mtp, &device)?;
    let w = Weights::from_gguf(mtp.clone());
    let blk = w.pp("blk.64");

    let hidden = 5120usize;
    let eh_proj = blk.qlinear("nextn.eh_proj")?;
    let enorm = blk.rms_norm("nextn.enorm", 1e-6)?;
    let hnorm = blk.rms_norm("nextn.hnorm", 1e-6)?;
    let shared_head_norm = blk.rms_norm("nextn.shared_head_norm", 1e-6)?;
    let attn_norm = blk.rms_norm("attn_norm", 1e-6)?;
    let post_norm = blk.rms_norm("post_attention_norm", 1e-6)?;
    let q_proj = blk.qlinear("attn_q")?;
    let k_proj = blk.qlinear("attn_k")?;
    let v_proj = blk.qlinear("attn_v")?;
    let o_proj = blk.qlinear("attn_output")?;
    let ffn_gate = blk.qlinear("ffn_gate")?;
    let ffn_up = blk.qlinear("ffn_up")?;
    let ffn_down = blk.qlinear("ffn_down")?;

    // ---- the target's lm_head: the chain-drafter tax ------------------------
    // The real design reuses the TARGET's quantized lm_head rather than the
    // sidecar's BF16 duplicate, so that is what is measured — and through the
    // path a real seq=1 draft step would take. `XwenModel::forward` does NOT run
    // QMatMul here: at one query position it dispatches the vendored plain
    // mat-vec over the retained lm_head buffer, because candle's baked quantized
    // mv runs far under bandwidth on this shape (model.rs, "Decode bypass").
    // Measuring QMatMul instead would inflate the crux number by that whole
    // factor, so this measures the vendored path and reports QMatMul only as a
    // contrast.
    let target = gguf::open(&args.target, &device)?;
    let tw = Weights::from_gguf(target.clone());
    let (lm_head, lm_buf, lm_dtype) = tw.qlinear_with_buffer("output")?;
    let lm_buf = lm_buf.context("the lm_head has no retained Metal buffer (not a Metal load?)")?;
    anyhow::ensure!(
        xwen::ops::mv_vendored_supported(lm_dtype),
        "the vendored mat-vec does not support the lm_head dtype {lm_dtype:?}"
    );
    println!(
        "lm_head dtype {lm_dtype:?}, [{}, {}]",
        lm_head.out_dim, lm_head.in_dim
    );

    println!(
        "\nctx {} rows | BATCH {BATCH} dispatches/sync | median of {REPEATS}",
        args.ctx
    );
    println!("lowpowermode/power state is the CALLER's to record — see the report.\n");

    // Random-but-realistic activations, allocated once: what is being measured
    // is weight traffic, not the cost of making inputs.
    let embed = Tensor::randn(0f32, 1.0, (1, hidden), &device)?;
    let hstate = Tensor::randn(0f32, 1.0, (1, hidden), &device)?;
    // A KV cache to attend over: 4 KV heads x 256 head_dim, f16, `ctx` rows.
    let kv_heads = 4usize;
    let head_dim = 256usize;
    let n_head = 24usize;
    let k_cache =
        Tensor::randn(0f32, 1.0, (kv_heads, args.ctx, head_dim), &device)?.to_dtype(DType::F16)?;
    let v_cache =
        Tensor::randn(0f32, 1.0, (kv_heads, args.ctx, head_dim), &device)?.to_dtype(DType::F16)?;

    // ---- (b) the lm_head matvec alone --------------------------------------
    let head_in = Tensor::randn(0f32, 1.0, (1, hidden), &device)?;
    let lm_ms = amortized_ms("(b) lm_head mv (vendored)", &device, || {
        Ok(xwen::ops::mul_mv(
            &lm_buf,
            lm_dtype,
            lm_head.out_dim,
            lm_head.in_dim,
            &head_in,
        )?)
    })?;
    // Contrast only, not the number the gate reads: what the same matvec costs
    // through candle's quantized QMatMul, which is the prefill path.
    let lm_qmm_ms = amortized_ms("    (contrast) via QMatMul", &device, || {
        Ok(lm_head.forward(&head_in)?)
    })?;

    // ---- (a) the whole MTP draft step --------------------------------------
    let step_ms = amortized_ms("(a) MTP draft step", &device, || {
        // eh_proj over the concatenated normed embedding and hidden state.
        let e = enorm.forward(&embed)?;
        let h = hnorm.forward(&hstate)?;
        let x = Tensor::cat(&[&e, &h], 1)?;
        let x = eh_proj.forward(&x)?;

        // One full-attention layer, in the trunk's shape.
        let normed = attn_norm.forward(&x)?;
        let qg = q_proj.forward(&normed)?; // [1, 2 * n_head * head_dim]
        // Dispatched for their cost; candle is eager, so the work happens here even
        // though this bench attends over a pre-made cache instead of appending.
        let _k = k_proj.forward(&normed)?;
        let _v = v_proj.forward(&normed)?;
        // The double-width q carries [q_h, gate_h] per head; the gate half is a
        // strided view in the real graph. Splitting it is the same traffic.
        let q = qg.narrow(1, 0, n_head * head_dim)?;
        let gate = qg.narrow(1, n_head * head_dim, n_head * head_dim)?;

        // sdpa over the cache: q [n_head, 1, head_dim] against k/v repeated to
        // n_head. Cost here is dominated by the cache read, which is the point.
        let q = q.reshape((n_head, 1, head_dim))?.to_dtype(DType::F16)?;
        let repeat = n_head / kv_heads;
        let kk = k_cache.repeat((repeat, 1, 1))?;
        let vv = v_cache.repeat((repeat, 1, 1))?;
        let scores = (q.matmul(&kk.transpose(1, 2)?)? * (1.0 / (head_dim as f64).sqrt()))?;
        let probs = candle_nn::ops::softmax_last_dim(&scores.to_dtype(DType::F32)?)?;
        let ctx_out = probs.to_dtype(DType::F16)?.matmul(&vv)?;
        let ctx_out = ctx_out
            .reshape((1, n_head * head_dim))?
            .to_dtype(DType::F32)?;
        // The sigmoid output gate the arch fuses before o_proj.
        let gated = (ctx_out * candle_nn::ops::sigmoid(&gate)?)?;
        let attn_out = o_proj.forward(&gated)?;
        let x = (x + attn_out)?;

        // SwiGLU FFN.
        let normed = post_norm.forward(&x)?;
        let g = ffn_gate.forward(&normed)?;
        let u = ffn_up.forward(&normed)?;
        let act = (candle_nn::ops::silu(&g)? * u)?;
        let x = (x + ffn_down.forward(&act)?)?;

        // The shared head norm, then the target's lm_head through the decode path.
        let normed = shared_head_norm.forward(&x)?;
        Ok(xwen::ops::mul_mv(
            &lm_buf,
            lm_dtype,
            lm_head.out_dim,
            lm_head.in_dim,
            &normed,
        )?)
    })?;

    // ---- (d) the same step, but paying a readback every step ---------------
    // A chain drafter proposes token N+1 from token N, so SOMETHING has to turn
    // logits into a token id between steps. Doing that on the CPU costs a
    // synchronize plus a device→host copy per step; keeping the chain on the GPU
    // does not. The gap between (a) and (d) IS that tax, and it decides whether
    // the draft chain has to stay on-device. Measured the honest way: one
    // synchronize and one 4-byte readback per step, no batching.
    let readback_ms = {
        let mut per_step = Vec::with_capacity(REPEATS);
        for r in 0..=REPEATS {
            let t0 = Instant::now();
            for _ in 0..BATCH {
                let logits = xwen::ops::mul_mv(
                    &lm_buf,
                    lm_dtype,
                    lm_head.out_dim,
                    lm_head.in_dim,
                    &head_in,
                )?;
                // What a CPU-side argmax costs at minimum: sync, then pull.
                let top = logits.flatten_all()?.argmax(0)?;
                let _id: u32 = top.to_scalar()?;
            }
            device.synchronize()?;
            let ms = t0.elapsed().as_secs_f64() * 1000.0 / BATCH as f64;
            if r > 0 {
                per_step.push(ms);
            }
        }
        per_step.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = per_step[per_step.len() / 2];
        println!(
            "  {:<28} {median:>8.3} ms/step   (runs {})",
            "(d) lm_head + CPU readback",
            per_step
                .iter()
                .map(|v| format!("{v:.3}"))
                .collect::<Vec<_>>()
                .join(", ")
        );
        median
    };

    // ---- (c) one target decode forward -------------------------------------
    let mut target_ms = None;
    if args.target_forward {
        let mut model = xwen::XwenModel::load(
            xwen::CheckpointSource::Gguf(target.clone()),
            xwen::ops::ExpertRunner::Fused,
            2048,
        )?;
        let token = Tensor::new(&[1u32], &device)?;
        // Not amortized the same way: a forward mutates the KV cache, so each
        // call must advance a position rather than repeat one. Timed as a run of
        // BATCH sequential decodes with one sync, which is what decoding is.
        let mut per_step = Vec::with_capacity(REPEATS);
        for r in 0..=REPEATS {
            model.reset_cache()?;
            let t0 = Instant::now();
            let mut alive = Vec::with_capacity(BATCH);
            for pos in 0..BATCH {
                alive.push(model.forward(&token, pos)?);
            }
            device.synchronize()?;
            let ms = t0.elapsed().as_secs_f64() * 1000.0 / BATCH as f64;
            if r > 0 {
                per_step.push(ms);
            }
            drop(alive);
        }
        per_step.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = per_step[per_step.len() / 2];
        println!(
            "  {:<28} {median:>8.3} ms/token (runs {})",
            "(c) target decode forward",
            per_step
                .iter()
                .map(|v| format!("{v:.3}"))
                .collect::<Vec<_>>()
                .join(", ")
        );
        target_ms = Some(median);
    }

    println!("\n=== ratios ===");
    println!(
        "  lm_head share of the MTP step : {:.1}%",
        100.0 * lm_ms / step_ms
    );
    println!(
        "  QMatMul lm_head, for contrast : {:.1}x the vendored path (not used by decode)",
        lm_qmm_ms / lm_ms
    );
    println!(
        "  per-step CPU readback tax     : {:+.3} ms/step over the same op batched ({:.1}x)",
        readback_ms - lm_ms,
        readback_ms / lm_ms
    );
    if let Some(t) = target_ms {
        println!(
            "  MTP step / target forward     : {:.1}%",
            100.0 * step_ms / t
        );
        println!(
            "  lm_head / target forward      : {:.1}%",
            100.0 * lm_ms / t
        );
    }
    Ok(())
}
