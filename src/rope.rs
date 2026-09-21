use std::f64::consts::PI;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};

use crate::config::RopeKind;

/// Precomputed rotary tables for the full-attention layers: plain NEoX rope
/// with partial rotary (n_rot = 64 of head_dim 256, theta 1e7). NEOX pairing:
/// dim i pairs with i + n_rot/2, dims >= n_rot pass through untouched. The
/// YaRN variant is retained but unwired (long-context scaling, TODO P13).
pub struct Rope {
    /// cos(theta) * mscale, shape [max_ctx, n_rot/2], f32.
    cos: Tensor,
    /// sin(theta) * mscale, shape [max_ctx, n_rot/2], f32.
    sin: Tensor,
    n_rot: usize,
}

impl Rope {
    pub fn new(kind: &RopeKind, max_ctx: usize, device: &Device) -> Result<Self> {
        // Per-pair inverse frequencies (one entry per rotated dimension pair) and
        // the magnitude scaling applied uniformly to cos/sin. `mscale` is 1 for
        // plain rope; for YaRN it is the net attention factor (see below).
        let (n_rot, inv_freq, mscale) = match kind {
            RopeKind::Plain { freq_base, n_rot } => {
                let base = *freq_base as f64;
                let n = *n_rot as f64;
                let inv_freq: Vec<f64> = (0..n_rot / 2)
                    .map(|j| base.powf(-(2.0 * j as f64) / n))
                    .collect();
                (*n_rot, inv_freq, 1.0)
            }
            RopeKind::Yarn {
                freq_base,
                factor,
                original_ctx,
                beta_fast,
                beta_slow,
                attn_factor,
                n_rot,
            } => {
                let base = *freq_base as f64;
                let n = *n_rot as f64;
                let factor = *factor as f64;

                // Correction range (in dim-pair units): high-frequency pairs
                // below `low` are left extrapolated (unscaled), low-frequency
                // pairs above `high` are fully interpolated (scaled by 1/factor),
                // and a linear ramp blends the two in between.
                let corr = |num_rot: f32| {
                    n * ((*original_ctx as f64) / (num_rot as f64 * 2.0 * PI)).ln()
                        / (2.0 * base.ln())
                };
                let low = corr(*beta_fast).floor().max(0.0);
                let mut high = corr(*beta_slow).ceil().min(n - 1.0);
                if low == high {
                    high += 0.001;
                }

                let inv_freq: Vec<f64> = (0..n_rot / 2)
                    .map(|j| {
                        let extrap = base.powf(-(2.0 * j as f64) / n);
                        let interp = extrap / factor;
                        let ramp = (((j as f64) - low) / (high - low)).clamp(0.0, 1.0);
                        interp * ramp + extrap * (1.0 - ramp)
                    })
                    .collect();

                // The fork applies YaRN's magnitude scaling once: ggml's rope
                // multiplies cos/sin by attn_factor * (1 + 0.1*ln(1/freq_scale)),
                // and llama_context pre-divides attn_factor by that same term, so
                // the net factor reaching the tables is exactly the config
                // attention_factor. Replicate that net effect directly.
                (*n_rot, inv_freq, *attn_factor as f64)
            }
        };

        let half = inv_freq.len();
        let mut cos = vec![0f32; max_ctx * half];
        let mut sin = vec![0f32; max_ctx * half];
        for p in 0..max_ctx {
            for j in 0..half {
                let theta = p as f64 * inv_freq[j];
                cos[p * half + j] = (theta.cos() * mscale) as f32;
                sin[p * half + j] = (theta.sin() * mscale) as f32;
            }
        }

        let cos = Tensor::from_vec(cos, (max_ctx, half), device)?;
        let sin = Tensor::from_vec(sin, (max_ctx, half), device)?;
        Ok(Self { cos, sin, n_rot })
    }

    /// q, k: [n_head, seq, head_dim]; positions pos..pos+seq. Both outputs f32.
    pub fn apply(&self, q: &Tensor, k: &Tensor, pos: usize) -> Result<(Tensor, Tensor)> {
        self.apply_dt(q, k, pos, DType::F32, DType::F32)
    }

    /// `apply` with per-tensor OUTPUT dtypes (f32 or f16). On the fused Metal
    /// kernel an f16 request narrows only the final store (one RTNE rounding —
    /// bit-identical to f32 rope + `cast_f16`), letting `AttnBlock::forward`
    /// fold the standalone post-rope casts away: k feeds the f16 cache
    /// directly, decode q feeds the f16 sdpa directly.
    pub fn apply_dt(
        &self,
        q: &Tensor,
        k: &Tensor,
        pos: usize,
        q_dtype: DType,
        k_dtype: DType,
    ) -> Result<(Tensor, Tensor)> {
        Ok((self.rotate(q, pos, q_dtype)?, self.rotate(k, pos, k_dtype)?))
    }

    /// Rotate the first `n_rot` dims of x (f32) with NEOX pairing (dim i with
    /// i + n_rot/2); any trailing dims pass through untouched. The result is
    /// stored as `out_dtype` (f32, or f16 = the f32 result rounded once).
    ///
    /// On Metal the default is the fused single-pass kernel (ops::rope_neox),
    /// which folds the partial-rotary narrow/contiguous/cat glue into the rope
    /// itself and is bit-identical to the candle chain below (the attn_glue.rs
    /// bitwise test proves it). XWEN_ATTN_GLUE_CLASSIC (the shared
    /// attention-glue kill-switch) reverts to the chain; non-Metal devices
    /// always run it. (Fused-glue callers are the only ones that request f16;
    /// the chain still honors it — via a trailing candle cast, the same
    /// rounding — so the choice of path stays invisible.)
    pub(crate) fn rotate(&self, x: &Tensor, pos: usize, out_dtype: DType) -> Result<Tensor> {
        check_out_dtype(out_dtype)?;
        // (The fused kernel wants a contiguous input; AttnBlock always provides
        // one. A strided caller falls through to the chain, which narrows and
        // copies — the two paths are bit-identical, so the choice is invisible.)
        if matches!(x.device(), Device::Metal(_))
            && x.is_contiguous()
            && !crate::ops::attn_glue_classic()
        {
            return crate::ops::rope_neox(x, &self.cos, &self.sin, pos, self.n_rot, out_dtype);
        }
        let seq = x.dim(1)?;
        let cos = self.cos.narrow(0, pos, seq)?;
        let sin = self.sin.narrow(0, pos, seq)?;
        self.rotate_with_tables(x, &cos, &sin, out_dtype)
    }

    /// `rotate` for tokens whose positions are NOT a consecutive run from a
    /// scalar start: `positions` is a u32 `[seq]` tensor naming each row's rope
    /// position, gathered out of the same `cos`/`sin` tables.
    ///
    /// The QSA indexer is the caller that needs it. Its pooled block keys are
    /// roped at the position of each block's FIRST member token — stride
    /// `compress_ratio` through the sequence — which no scalar start describes.
    /// The fused single-pass kernel takes a scalar `pos`, so this variant always
    /// runs the candle chain; `rotate` proves the two are bit-identical, and the
    /// gathered rows are the same table rows a consecutive run would narrow to.
    pub(crate) fn rotate_at(
        &self,
        x: &Tensor,
        positions: &Tensor,
        out_dtype: DType,
    ) -> Result<Tensor> {
        check_out_dtype(out_dtype)?;
        let seq = x.dim(1)?;
        anyhow::ensure!(
            positions.dims1()? == seq,
            "rope: {} positions for {seq} rows",
            positions.dims1()?
        );
        let cos = self.cos.index_select(positions, 0)?;
        let sin = self.sin.index_select(positions, 0)?;
        self.rotate_with_tables(x, &cos, &sin, out_dtype)
    }

    /// The candle rope chain over already-selected `[seq, n_rot/2]` cos/sin
    /// rows: rotate the first `n_rot` dims of `x` `[heads, seq, head_dim]`,
    /// pass the rest through, store as `out_dtype`. Also the entry for rows that
    /// are not rows of this table at all: the three-axis ones
    /// [`mrope_interleaved_tables`] builds.
    pub(crate) fn rotate_with_tables(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        out_dtype: DType,
    ) -> Result<Tensor> {
        let (_, _, head_dim) = x.dims3()?;

        // candle's rope kernel wants a batch dim and pairs the two contiguous
        // halves of its input, so feed it exactly the rotated block.
        let x = x.unsqueeze(0)?;
        let rotated =
            candle_nn::rotary_emb::rope(&x.narrow(3, 0, self.n_rot)?.contiguous()?, cos, sin)?;
        let out = if self.n_rot < head_dim {
            let pass = x.narrow(3, self.n_rot, head_dim - self.n_rot)?;
            Tensor::cat(&[&rotated, &pass], 3)?
        } else {
            rotated
        };
        let out = out.squeeze(0)?.contiguous()?;
        // f32 requests (every classic-path caller) leave the chain untouched.
        Ok(if out_dtype == DType::F32 {
            out
        } else {
            out.to_dtype(out_dtype)?
        })
    }
}

/// One position per axis for every token of a multimodal sequence. A text
/// token carries the same id on all three; an image token carries its frame,
/// row and column.
#[derive(Debug, Clone, Copy)]
pub struct MropeIds<'a> {
    pub temporal: &'a [u32],
    pub height: &'a [u32],
    pub width: &'a [u32],
}

/// Which axis frequency slot `slot` reads under interleaved MRoPE.
///
/// transformers' `Qwen3VLTextRotaryEmbedding` starts every slot on the temporal
/// axis and then overwrites `slice(1, 3 * height, 3)` with the height axis and
/// `slice(2, 3 * width, 3)` with the width axis. So with sections
/// `[24, 20, 20]` over 64 slots, 1, 4, …, 58 read height, 2, 5, …, 59 read
/// width, and 0, 3, …, 57 together with 60..63 read temporal.
fn mrope_axis(sections: crate::qwen3::MropeSections, slot: usize) -> usize {
    match slot % 3 {
        1 if slot < 3 * sections.height => 1,
        2 if slot < 3 * sections.width => 2,
        _ => 0,
    }
}

/// cos/sin rows `[seq, head_dim / 2]` f32 for interleaved three-axis MRoPE with
/// unscaled frequencies: slot `j` is `id * theta^(-2j / head_dim)` with `id`
/// taken from the axis [`mrope_axis`] assigns that slot. The rows go to
/// [`Rope::rotate_with_tables`], whose NEoX pairing is the `cat(freqs, freqs)`
/// plus `rotate_half` transformers applies.
///
/// The arithmetic is [`Rope::new`]'s, f64 rounded once to f32, so a token whose
/// three ids are equal gets bit for bit the row the plain table holds at that
/// position. That is what makes text-only encoding on the plain path exact
/// rather than approximately right.
pub fn mrope_interleaved_tables(
    sections: crate::qwen3::MropeSections,
    theta: f64,
    ids: MropeIds<'_>,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let seq = ids.temporal.len();
    anyhow::ensure!(
        ids.height.len() == seq && ids.width.len() == seq,
        "mrope: {seq} temporal ids against {} height and {} width ids",
        ids.height.len(),
        ids.width.len()
    );
    let half = sections.temporal + sections.height + sections.width;
    // A split whose spatial sections reach past the slots that exist would
    // leave some of their frequencies unassigned without saying so.
    anyhow::ensure!(
        3 * sections.height <= half + 2 && 3 * sections.width <= half + 2,
        "mrope: sections {sections:?} do not interleave into {half} slots"
    );
    let n = (2 * half) as f64;
    let inv_freq: Vec<f64> = (0..half)
        .map(|j| theta.powf(-(2.0 * j as f64) / n))
        .collect();
    let axes = [ids.temporal, ids.height, ids.width];
    let mut cos = vec![0f32; seq * half];
    let mut sin = vec![0f32; seq * half];
    for t in 0..seq {
        for j in 0..half {
            let angle = axes[mrope_axis(sections, j)][t] as f64 * inv_freq[j];
            cos[t * half + j] = angle.cos() as f32;
            sin[t * half + j] = angle.sin() as f32;
        }
    }
    Ok((
        Tensor::from_vec(cos, (seq, half), device)?,
        Tensor::from_vec(sin, (seq, half), device)?,
    ))
}

/// F32/F16 is the contract on every rope path (the fused kernel enforces it;
/// the chain must not silently accept more just because to_dtype can).
fn check_out_dtype(out_dtype: DType) -> Result<()> {
    anyhow::ensure!(
        matches!(out_dtype, DType::F32 | DType::F16),
        "rope: out_dtype must be F32 or F16, got {out_dtype:?}"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen3::MropeSections;

    const QWEN3_VL_SECTIONS: MropeSections = MropeSections {
        temporal: 24,
        height: 20,
        width: 20,
    };

    /// The slot-to-axis map, against the slices transformers writes.
    #[test]
    fn interleaved_mrope_assigns_slots_as_transformers_does() {
        let mut want = [0usize; 64];
        for slot in (1..60).step_by(3) {
            want[slot] = 1;
        }
        for slot in (2..60).step_by(3) {
            want[slot] = 2;
        }
        let got: Vec<usize> = (0..64).map(|s| mrope_axis(QWEN3_VL_SECTIONS, s)).collect();
        assert_eq!(got, want);
        for (axis, count) in [(0, 24), (1, 20), (2, 20)] {
            assert_eq!(got.iter().filter(|a| **a == axis).count(), count);
        }
    }

    /// Text tokens carry one id on all three axes, and their rows are the plain
    /// NEoX table's rows to the bit — at every position, not just small ones.
    #[test]
    fn equal_ids_on_every_axis_reproduce_the_plain_table_bitwise() {
        let device = Device::Cpu;
        let theta = 5e6;
        let plain = Rope::new(
            &RopeKind::Plain {
                freq_base: theta as f32,
                n_rot: 128,
            },
            4096,
            &device,
        )
        .unwrap();
        let ids: Vec<u32> = vec![0, 1, 2, 3, 17, 511, 2048, 4095];
        let (cos, sin) = mrope_interleaved_tables(
            QWEN3_VL_SECTIONS,
            theta,
            MropeIds {
                temporal: &ids,
                height: &ids,
                width: &ids,
            },
            &device,
        )
        .unwrap();
        let positions = Tensor::new(ids.as_slice(), &device).unwrap();
        for (got, table) in [(&cos, &plain.cos), (&sin, &plain.sin)] {
            let want = table.index_select(&positions, 0).unwrap();
            let got: Vec<u32> = got
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .into_iter()
                .map(f32::to_bits)
                .collect();
            let want: Vec<u32> = want
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .into_iter()
                .map(f32::to_bits)
                .collect();
            assert_eq!(got, want);
        }
    }

    /// An image token's row mixes three positions: each slot is the plain row
    /// of whichever axis owns it.
    #[test]
    fn distinct_ids_take_each_slot_from_its_own_axis() {
        let device = Device::Cpu;
        let theta = 5e6;
        let plain = Rope::new(
            &RopeKind::Plain {
                freq_base: theta as f32,
                n_rot: 128,
            },
            64,
            &device,
        )
        .unwrap();
        let (cos, _) = mrope_interleaved_tables(
            QWEN3_VL_SECTIONS,
            theta,
            MropeIds {
                temporal: &[7],
                height: &[19],
                width: &[42],
            },
            &device,
        )
        .unwrap();
        let cos = cos.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let rows = [cos_row(&plain, 7), cos_row(&plain, 19), cos_row(&plain, 42)];
        for (slot, got) in cos.iter().enumerate() {
            let axis = mrope_axis(QWEN3_VL_SECTIONS, slot);
            assert_eq!(got.to_bits(), rows[axis][slot].to_bits(), "slot {slot}");
        }
        assert_ne!(cos[1].to_bits(), rows[0][1].to_bits());
    }

    #[test]
    fn mismatched_id_rows_are_refused() {
        let err = mrope_interleaved_tables(
            QWEN3_VL_SECTIONS,
            5e6,
            MropeIds {
                temporal: &[0, 1],
                height: &[0],
                width: &[0, 1],
            },
            &Device::Cpu,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("2 temporal ids"), "{err}");
    }

    fn cos_row(rope: &Rope, pos: usize) -> Vec<f32> {
        rope.cos
            .narrow(0, pos, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    }

    fn sin_row(rope: &Rope, pos: usize) -> Vec<f32> {
        rope.sin
            .narrow(0, pos, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    }

    /// Laguna's YaRN attention_factor = 1 + 0.1*ln(128), applied to cos/sin once.
    const XWEN_MSCALE: f64 = 1.4852030263919618;

    fn laguna_yarn() -> RopeKind {
        RopeKind::Yarn {
            freq_base: 500_000.0,
            factor: 128.0,
            original_ctx: 8192,
            beta_fast: 32.0,
            beta_slow: 1.0,
            attn_factor: XWEN_MSCALE as f32,
            n_rot: 64,
        }
    }

    #[test]
    fn rope_plain_hand_computed() {
        // theta=10000, n_rot=4, head_dim=4: pairs (0,2) with inv_freq 1.0 and
        // (1,3) with inv_freq 10000^(-1/2) = 0.01.
        let dev = Device::Cpu;
        let rope = Rope::new(
            &RopeKind::Plain {
                freq_base: 10_000.0,
                n_rot: 4,
            },
            8,
            &dev,
        )
        .unwrap();

        let pos = 3usize;
        let (f0, f1) = (1.0f64, 0.01f64);
        let (c0, s0) = (
            (pos as f64 * f0).cos() as f32,
            (pos as f64 * f0).sin() as f32,
        );
        let (c1, s1) = (
            (pos as f64 * f1).cos() as f32,
            (pos as f64 * f1).sin() as f32,
        );

        let cos = cos_row(&rope, pos);
        let sin = sin_row(&rope, pos);
        assert!((cos[0] - c0).abs() < 1e-6 && (sin[0] - s0).abs() < 1e-6);
        assert!((cos[1] - c1).abs() < 1e-6 && (sin[1] - s1).abs() < 1e-6);

        // A single query vector rotated at position 3.
        let x = vec![0.5f32, -1.0, 2.0, 0.25];
        let q = Tensor::from_vec(x.clone(), (1, 1, 4), &dev).unwrap();
        let (out, _) = rope.apply(&q, &q, pos).unwrap();
        let out = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let expect = vec![
            x[0] * c0 - x[2] * s0,
            x[1] * c1 - x[3] * s1,
            x[0] * s0 + x[2] * c0,
            x[1] * s1 + x[3] * c1,
        ];
        for (a, b) in out.iter().zip(expect.iter()) {
            assert!((a - b).abs() < 1e-6, "got {out:?} expected {expect:?}");
        }
    }

    #[test]
    fn yarn_pass_through_dims_untouched() {
        // Dims n_rot..head_dim (64..128) must be bit-identical after apply.
        let dev = Device::Cpu;
        let rope = Rope::new(&laguna_yarn(), 32, &dev).unwrap();

        let n_head = 3;
        let seq = 5;
        let head_dim = 128;
        let data: Vec<f32> = (0..n_head * seq * head_dim)
            .map(|i| (i as f32) * 0.001 - 0.5)
            .collect();
        let q = Tensor::from_vec(data, (n_head, seq, head_dim), &dev).unwrap();
        let k = q.clone();

        let (out_q, _) = rope.apply(&q, &k, 7).unwrap();

        let before = q
            .narrow(2, 64, 64)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let after = out_q
            .narrow(2, 64, 64)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(before, after, "pass-through dims must be bit-identical");
    }

    #[test]
    fn pos_shift_consistency() {
        // apply at pos=5 on one token equals row 5 of apply at pos=0 on six tokens.
        let dev = Device::Cpu;
        let rope = Rope::new(&laguna_yarn(), 16, &dev).unwrap();

        let head_dim = 128;
        let n_head = 2;
        let single: Vec<f32> = (0..n_head * head_dim)
            .map(|i| ((i * 7 % 13) as f32) * 0.1 - 0.6)
            .collect();

        // One token placed at position 5.
        let q1 = Tensor::from_vec(single.clone(), (n_head, 1, head_dim), &dev).unwrap();
        let (out1, _) = rope.apply(&q1, &q1, 5).unwrap();
        let out1 = out1.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        // Six tokens at positions 0..6, with the same vector in slot 5.
        let mut six = vec![0f32; n_head * 6 * head_dim];
        for h in 0..n_head {
            for d in 0..head_dim {
                six[h * 6 * head_dim + 5 * head_dim + d] = single[h * head_dim + d];
            }
        }
        let q6 = Tensor::from_vec(six, (n_head, 6, head_dim), &dev).unwrap();
        let (out6, _) = rope.apply(&q6, &q6, 0).unwrap();
        let row5 = out6
            .narrow(1, 5, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        for (a, b) in out1.iter().zip(row5.iter()) {
            assert!((a - b).abs() < 1e-6, "pos-shift mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn yarn_sanity() {
        let dev = Device::Cpu;
        let rope = Rope::new(&laguna_yarn(), 8192, &dev).unwrap();

        // Position 0: rotation is identity scaled by mscale (cos = mscale, sin = 0).
        let cos0 = cos_row(&rope, 0);
        let sin0 = sin_row(&rope, 0);
        for &c in &cos0 {
            assert!(
                (c as f64 - XWEN_MSCALE).abs() < 1e-6,
                "cos row 0 should equal mscale, got {c}"
            );
        }
        for &s in &sin0 {
            assert!(s.abs() < 1e-6, "sin row 0 should be zero, got {s}");
        }

        // The lowest-frequency pair (j = 31) is fully interpolated for the Laguna
        // config (its ramp saturates to 1), so its angle is scaled down by exactly
        // `factor` relative to unscaled rope over the same geometry. Compare angles
        // via atan2, which cancels the shared mscale, at a position small enough to
        // avoid wraparound.
        let plain = Rope::new(
            &RopeKind::Plain {
                freq_base: 500_000.0,
                n_rot: 64,
            },
            8192,
            &dev,
        )
        .unwrap();
        let p = 4096usize;
        let (cy, sy) = (cos_row(&rope, p)[31], sin_row(&rope, p)[31]);
        let (cp, sp) = (cos_row(&plain, p)[31], sin_row(&plain, p)[31]);
        let theta_yarn = (sy as f64).atan2(cy as f64);
        let theta_plain = (sp as f64).atan2(cp as f64);

        assert!(
            theta_plain > 0.0,
            "plain angle should be a small positive value"
        );
        assert!(
            (theta_yarn - theta_plain / 128.0).abs() < 1e-6,
            "yarn low-freq angle should be plain/factor: {theta_yarn} vs {}",
            theta_plain / 128.0
        );
        // And the raw rotation differs from plain (scaling actually engaged).
        assert!(
            (theta_yarn - theta_plain).abs() > 1e-3,
            "yarn scaling must change the angle"
        );
    }

    /// `rotate_at` over a consecutive run is exactly `rotate` from that start:
    /// it gathers the same table rows the scalar-start path narrows to.
    #[test]
    fn rotate_at_matches_rotate_on_a_consecutive_run() {
        let dev = Device::Cpu;
        let rope = Rope::new(&laguna_yarn(), 64, &dev).unwrap();
        let (n_head, seq, head_dim) = (3usize, 7usize, 128usize);
        let data: Vec<f32> = (0..n_head * seq * head_dim)
            .map(|i| ((i * 13 % 29) as f32) * 0.07 - 1.0)
            .collect();
        let x = Tensor::from_vec(data, (n_head, seq, head_dim), &dev).unwrap();

        let pos = 11usize;
        let want = rope.rotate(&x, pos, DType::F32).unwrap();
        let positions = Tensor::from_vec(
            (0..seq).map(|t| (pos + t) as u32).collect::<Vec<_>>(),
            seq,
            &dev,
        )
        .unwrap();
        let got = rope.rotate_at(&x, &positions, DType::F32).unwrap();

        let g: Vec<f32> = got.flatten_all().unwrap().to_vec1().unwrap();
        let w: Vec<f32> = want.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(g, w, "a consecutive gather must reproduce the scalar start");
    }

    /// The same consecutive-run identity on Metal, where the two calls take
    /// GENUINELY DIFFERENT routes: `rotate` on a contiguous Metal input runs the
    /// fused single-pass `ops::rope_neox` kernel, while `rotate_at` always runs
    /// the candle chain. So this pins the cross-path claim `rotate_at`'s doc
    /// comment makes — that the gathered table rows and the fused kernel's
    /// narrowed ones produce the same bits — at the shape the QSA indexer
    /// actually uses (4 heads, head_dim 128, f32 out). A rounding difference
    /// between the two routes would show up here as scored blocks drifting from
    /// what a dense run would have selected, which no output-level test would
    /// name.
    #[test]
    fn rotate_at_matches_rotate_on_metal_at_the_indexer_shape() {
        let Ok(dev) = crate::gguf::metal_device() else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let rope = Rope::new(&laguna_yarn(), 512, &dev).unwrap();
        let (n_head, seq, head_dim) = (4usize, 37usize, 128usize);
        let data: Vec<f32> = (0..n_head * seq * head_dim)
            .map(|i| ((i * 13 % 29) as f32) * 0.07 - 1.0)
            .collect();
        let x = Tensor::from_vec(data, (n_head, seq, head_dim), &dev).unwrap();

        let pos = 128usize;
        let want = rope.rotate(&x, pos, DType::F32).unwrap();
        let positions = Tensor::from_vec(
            (0..seq).map(|t| (pos + t) as u32).collect::<Vec<_>>(),
            seq,
            &dev,
        )
        .unwrap();
        let got = rope.rotate_at(&x, &positions, DType::F32).unwrap();

        let g: Vec<f32> = got.flatten_all().unwrap().to_vec1().unwrap();
        let w: Vec<f32> = want.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(
            g, w,
            "the gathered chain and the fused kernel must agree bit for bit"
        );
    }

    /// Each row is roped at the position `positions` names for it, and at
    /// nothing else — pinned with a scrambled, non-monotonic set, which is the
    /// shape the QSA indexer's block-first positions take.
    #[test]
    fn rotate_at_ropes_every_row_at_its_own_position() {
        let dev = Device::Cpu;
        let rope = Rope::new(&laguna_yarn(), 256, &dev).unwrap();
        let (n_head, seq, head_dim) = (2usize, 5usize, 128usize);
        let data: Vec<f32> = (0..n_head * seq * head_dim)
            .map(|i| ((i * 7 % 17) as f32) * 0.11 - 0.8)
            .collect();
        let x = Tensor::from_vec(data, (n_head, seq, head_dim), &dev).unwrap();

        let scrambled = [200usize, 4, 91, 0, 137];
        let positions = Tensor::from_vec(
            scrambled.iter().map(|&p| p as u32).collect::<Vec<_>>(),
            seq,
            &dev,
        )
        .unwrap();
        let got = rope.rotate_at(&x, &positions, DType::F32).unwrap();

        for (t, &p) in scrambled.iter().enumerate() {
            let row = x.narrow(1, t, 1).unwrap().contiguous().unwrap();
            let want = rope.rotate(&row, p, DType::F32).unwrap();
            let want: Vec<f32> = want.flatten_all().unwrap().to_vec1().unwrap();
            let one = got.narrow(1, t, 1).unwrap().contiguous().unwrap();
            let one: Vec<f32> = one.flatten_all().unwrap().to_vec1().unwrap();
            assert_eq!(one, want, "row {t} at position {p}");
        }

        // And the gather is actually consulted: the default `b * ratio`-style
        // consecutive run gives a different answer.
        let plain = rope.rotate(&x, 0, DType::F32).unwrap();
        let plain: Vec<f32> = plain.flatten_all().unwrap().to_vec1().unwrap();
        let g: Vec<f32> = got.flatten_all().unwrap().to_vec1().unwrap();
        assert!(
            g.iter().zip(&plain).any(|(a, b)| (a - b).abs() > 1e-5),
            "scrambling the positions changed nothing — the gather is ignored"
        );
    }
}
