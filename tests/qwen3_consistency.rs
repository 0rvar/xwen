//! Decode-path consistency of the Qwen3 dense stack, with no oracle: the same
//! ids teacher-forced as one prefill, one token at a time, and in chunks of
//! 7 / 8 / 9 / 16 must give the same logits at every position. This is where
//! the three splits the stack straddles would show a seam — the flash kernel
//! (multi-token) against the f16 vector sdpa (one token), `matmul_bf16`'s gemv
//! (t <= 8) against its tensor gemm (t > 8), and the shipped attention arm
//! against the f32 sdpa bisect arm (`XWEN_QWEN3_ATTN=sdpa`). Bar: max-abs
//! logit difference <= 2e-2 and an identical argmax at every position
//! (docs/parity.md, the qwen3 section).
//!
//! Ignored by default (needs the 8 GB checkpoint and a Metal device):
//!
//!   cargo test --release --test qwen3_consistency -- --ignored --nocapture
//!
//! `XWEN_QWEN3_DIR` names the safetensors directory; unset, the base
//! checkpoint's cached snapshot (`Qwen/Qwen3-4B`) is used and the test skips
//! with a message when it is not in the Hugging Face cache.

use std::path::PathBuf;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use serde_json::Value;

use xwen::checkpoint::CheckpointSource;
use xwen::hub::Model;
use xwen::model::XwenModel;
use xwen::ops::ExpertRunner;
use xwen::qwen3::stack::{ATTN_ENV, AttnImpl};

const MAX_ABS: f32 = 2e-2;
const CHUNKS: [usize; 5] = [1, 7, 8, 9, 16];
const MAX_CTX: usize = 4096;

/// The three fixture prompts the brief names: short, medium and the 610-token one.
const PROMPTS: [&str; 3] = ["parity-code-short", "corpus-middle", "parity-long-mixed"];

fn checkpoint_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("XWEN_QWEN3_DIR") {
        return Some(PathBuf::from(dir));
    }
    // `cached_model` hands back the entry's first file, `config.json`.
    xwen::hub::cached_model(Model::Qwen34B)
        .map(|config| config.parent().map(|p| p.to_path_buf()).unwrap_or(config))
}

fn fixture_ids() -> Result<Vec<(String, Vec<u32>)>> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen3-prompts.json");
    let text =
        std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    let json: Value = serde_json::from_str(&text)?;
    let prompts = json["prompts"]
        .as_array()
        .context("fixture has no `prompts` array")?;
    let mut out = Vec::new();
    for want in PROMPTS {
        let p = prompts
            .iter()
            .find(|p| p["id"] == want)
            .with_context(|| format!("fixture has no prompt {want:?}"))?;
        let ids: Vec<u32> = p["ids"]
            .as_array()
            .context("prompt has no `ids`")?
            .iter()
            .map(|v| v.as_u64().map(|x| x as u32).context("id is not an integer"))
            .collect::<Result<_>>()?;
        out.push((want.to_string(), ids));
    }
    Ok(out)
}

fn load(dir: &PathBuf, device: &Device) -> Result<XwenModel> {
    // The entry supplies nothing a base checkpoint needs (no allowlist, the
    // tokenizer beside the shards), but a custom `XWEN_QWEN3_DIR` may be the
    // Instruct release, whose theta the cross-check would refuse under the
    // wrong entry — so no entry is named and the directory identifies itself.
    let source = CheckpointSource::open(dir, device, None)?;
    XwenModel::load(source, ExpertRunner::Fused, MAX_CTX)
}

fn rows(t: &Tensor) -> Result<Vec<Vec<f32>>> {
    Ok(t.to_dtype(DType::F32)?
        .to_device(&Device::Cpu)?
        .to_vec2::<f32>()?)
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap()
}

/// All-position logits over `ids` fed in `chunk`-token steps from an empty
/// cache, positions continuous.
fn logits_in_chunks(
    model: &mut XwenModel,
    ids: &[u32],
    chunk: usize,
    device: &Device,
) -> Result<Vec<Vec<f32>>> {
    model.reset_cache()?;
    let mut out = Vec::with_capacity(ids.len());
    let mut pos = 0;
    for c in ids.chunks(chunk) {
        let t = Tensor::new(c, device)?;
        out.extend(rows(&model.forward_all_logits(&t, pos)?)?);
        pos += c.len();
    }
    model.reset_cache()?;
    Ok(out)
}

/// Compare two all-position logit sets under the bar; returns the measured
/// max-abs difference and panics with the first offending position.
fn assert_same(label: &str, a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
    assert_eq!(a.len(), b.len(), "{label}: position counts differ");
    let mut worst = 0f32;
    for (p, (x, y)) in a.iter().zip(b).enumerate() {
        assert_eq!(
            x.len(),
            y.len(),
            "{label}: vocab widths differ at position {p}"
        );
        let d = x
            .iter()
            .zip(y)
            .map(|(u, v)| (u - v).abs())
            .fold(0f32, f32::max);
        worst = worst.max(d);
        assert!(
            d <= MAX_ABS,
            "{label}: position {p}: max |Δlogit| {d:.4e} exceeds {MAX_ABS:.0e}"
        );
        assert_eq!(
            argmax(x),
            argmax(y),
            "{label}: argmax differs at position {p}"
        );
    }
    worst
}

#[test]
#[ignore]
fn chunked_teacher_forcing_matches_one_prefill_at_every_position() -> Result<()> {
    let Some(dir) = checkpoint_dir() else {
        eprintln!(
            "skipping: Qwen3-4B is not in the Hugging Face cache and XWEN_QWEN3_DIR is unset"
        );
        return Ok(());
    };
    let device = xwen::gguf::metal_device()?;
    let prompts = fixture_ids()?;

    // The shipped arm first: one prefill against every chunking.
    let mut flash = load(&dir, &device)?;
    assert_eq!(
        flash.qwen3_parts().map(|p| p.attn_impl()),
        Some(AttnImpl::Fused),
        "the default load must run the fused attention arm"
    );
    let mut single_flash: Vec<Vec<Vec<f32>>> = Vec::new();
    for (name, ids) in &prompts {
        let single = logits_in_chunks(&mut flash, ids, ids.len(), &device)?;
        for &chunk in &CHUNKS {
            let chunked = logits_in_chunks(&mut flash, ids, chunk, &device)?;
            let worst = assert_same(&format!("{name} chunk {chunk}"), &single, &chunked);
            println!(
                "{name} ({} tokens): chunk {chunk:>2} vs one prefill: max |Δlogit| {worst:.3e}",
                ids.len()
            );
        }
        single_flash.push(single);
    }
    drop(flash);

    // The bisect arm, loaded in-process with the switch set before the load
    // (the arm is resolved per load, not per process), against the fused
    // arm's one-prefill logits and its own chunkings.
    // SAFETY: single-threaded test binary; nothing else reads the environment.
    unsafe { std::env::set_var(ATTN_ENV, "sdpa") };
    let mut sdpa = load(&dir, &device)?;
    assert_eq!(
        sdpa.qwen3_parts().map(|p| p.attn_impl()),
        Some(AttnImpl::Sdpa),
        "{ATTN_ENV}=sdpa must select the sdpa arm at load"
    );
    for ((name, ids), single) in prompts.iter().zip(&single_flash) {
        let sdpa_single = logits_in_chunks(&mut sdpa, ids, ids.len(), &device)?;
        let worst = assert_same(&format!("{name} flash vs sdpa"), single, &sdpa_single);
        println!("{name}: flash vs sdpa, one prefill: max |Δlogit| {worst:.3e}");
        for &chunk in &CHUNKS {
            let chunked = logits_in_chunks(&mut sdpa, ids, chunk, &device)?;
            let worst = assert_same(
                &format!("{name} sdpa chunk {chunk}"),
                &sdpa_single,
                &chunked,
            );
            println!("{name}: sdpa chunk {chunk:>2} vs one prefill: max |Δlogit| {worst:.3e}");
        }
    }
    unsafe { std::env::remove_var(ATTN_ENV) };
    Ok(())
}

/// `encode` follows transformers' `hidden_states` numbering on the real
/// checkpoint: index 0 is the embedding rows, index 36 is the normed residual
/// after layer 35 (the full forward's `l_out-35` tap through `output_norm`),
/// and index 35 is that residual raw (the `l_out-34` tap).
#[test]
#[ignore]
fn encode_indices_match_the_forward_taps() -> Result<()> {
    let Some(dir) = checkpoint_dir() else {
        eprintln!(
            "skipping: Qwen3-4B is not in the Hugging Face cache and XWEN_QWEN3_DIR is unset"
        );
        return Ok(());
    };
    let device = xwen::gguf::metal_device()?;
    let mut model = load(&dir, &device)?;
    let n_layer = model.config().n_layer;
    assert_eq!(n_layer, 36);
    let (_, ids) = fixture_ids()?.swap_remove(1); // corpus-middle, 199 tokens

    let (h0, t) = model.encode(&ids, 0)?;
    assert_eq!(
        (h0.dims(), h0.dtype(), t),
        (&[ids.len(), 2560][..], DType::BF16, ids.len())
    );
    let embed = model.embed_ids(&ids)?.to_dtype(DType::BF16)?;
    assert_eq!(
        assert_same("encode 0 vs embeddings", &rows(&h0)?, &rows(&embed)?),
        0.0
    );

    model.set_tap_capture(true);
    model.reset_cache()?;
    model.forward(&Tensor::new(ids.as_slice(), &device)?, 0)?;
    let taps = model.take_taps();
    model.set_tap_capture(false);
    model.reset_cache()?;
    let tap = |name: &str| -> Tensor {
        taps.iter()
            .find(|(n, _)| n == name)
            .map(|(_, t)| t.clone())
            .unwrap_or_else(|| panic!("no tap {name}"))
    };

    let (h35, _) = model.encode(&ids, 35)?;
    let l_out_34 = tap("l_out-34").to_dtype(DType::BF16)?;
    let d35 = assert_same("encode 35 vs l_out-34", &rows(&h35)?, &rows(&l_out_34)?);
    println!("encode 35 vs l_out-34: max |Δ| {d35:.3e}");

    let (h36, _) = model.encode(&ids, 36)?;
    let normed = model.final_norm(&tap("l_out-35"))?.to_dtype(DType::BF16)?;
    let d36 = assert_same("encode 36 vs norm(l_out-35)", &rows(&h36)?, &rows(&normed)?);
    println!("encode 36 vs norm(l_out-35): max |Δ| {d36:.3e}");

    assert!(
        model.encode(&ids, 37).is_err(),
        "index 37 must be refused on a 36-layer model"
    );
    assert_eq!(tap("kqv_out-0").dims(), &[ids.len(), 4096]);
    Ok(())
}
