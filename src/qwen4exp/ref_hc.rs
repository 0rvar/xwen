//! Frozen CPU f32 reference for hyper-connections — the residual carrier that
//! replaces the usual `attn_norm` / `post_attention_norm` / `output_norm`
//! chain in qwen4exp — plus the two norm flavours it and the GDN block need.
//!
//! Everything here is plain `&[f32]` row-major arithmetic with explicit dims:
//! no candle, no tensors, no cleverness. It is a correctness oracle graded
//! against `tests/fixtures/qwen4exp/`, and it is never optimized.
//!
//! # The residual stream
//!
//! Instead of one `hidden`-wide residual, qwen4exp carries `hc_count` parallel
//! streams concatenated into a single `hc_count * hidden` row (seeded by
//! repeating the token embedding `hc_count` times). Every block reads a single
//! `hidden`-wide vector out of that carrier and writes its output back into all
//! streams with per-stream strength:
//!
//! - read: [`GatedResidualRef::read`] → `(mixed, inject)`, where `mixed` is the
//!   block input and `inject` is the per-stream write strength;
//! - write: [`GatedResidualRef::write`] → `stream += block_out ⊗ inject`, onto
//!   the RAW un-normed stream (the norm from the read is not carried forward).
//!
//! The model tail uses the same read path with no injection head
//! ([`HcMixerRef`]) to collapse the carrier down to `hidden` for the lm_head.
//!
//! # Weight layout
//!
//! Every weight matrix here is row-major `[out][in]`: element `(o, i)` lives at
//! `o * n_in + i`, so a matrix-vector product is `n_out` contiguous dot
//! products of length `n_in`. This is the layout the fixtures store and the
//! layout GGUF yields when a `[in, out]` ggml tensor (ne-reversed convention)
//! is read as flat rows.
//!
//! Norm weights are the multiply-ready GGUF form: multiply by them directly,
//! never add 1 first. (HF stores zero-centered `(1 + w)` norms and the
//! converter bakes the +1 in; `gated_rms_norm`'s weight was never zero-centered
//! upstream, so both end up as a plain multiply.)

/// The activation applied to the `z` stream of [`gated_rms_norm`].
///
/// qwen4exp's GDN output norm constructs the sigmoid arm; the silu arm is what
/// the Qwen3.6 gated DeltaNet norm uses, and both are exercised by the fixture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZGateRef {
    Silu,
    Sigmoid,
}

impl ZGateRef {
    fn apply(self, z: f32) -> f32 {
        match self {
            ZGateRef::Silu => silu(z),
            ZGateRef::Sigmoid => sigmoid(z),
        }
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

/// `y = m @ v` where `m` is row-major `[n_out][n_in]` and `v` is `[n_in]`.
fn matvec(m: &[f32], v: &[f32], n_out: usize, n_in: usize) -> Vec<f32> {
    assert_eq!(m.len(), n_out * n_in, "matrix is not [n_out][n_in]");
    assert_eq!(
        v.len(),
        n_in,
        "vector width does not match matrix input width"
    );
    let mut out = Vec::with_capacity(n_out);
    for o in 0..n_out {
        let row = &m[o * n_in..(o + 1) * n_in];
        let mut acc = 0.0f32;
        for (w, x) in row.iter().zip(v.iter()) {
            acc += w * x;
        }
        out.push(acc);
    }
    out
}

/// RMS norm whose statistics are taken per contiguous group of `group`
/// elements, with the weight applied elementwise over the FULL width.
///
/// `x` is one row of `weight.len()` values, viewed as `weight.len() / group`
/// consecutive groups; each group is scaled by its own
/// `1 / sqrt(mean(x²) + eps)`. Grouping is load-bearing: the hyper-connection
/// carrier's streams can sit decades apart in scale, and a single set of
/// statistics over the whole row would flatten that (the fixture pins the
/// divergence).
///
/// The weight is `[weight.len()]` wide — one value per element of the row, not
/// one per group — and is the multiply-ready GGUF form.
///
/// The sum of squares accumulates in f64, matching ggml's CPU RMS norm
/// (`sum` is a `ggml_float`, the mean is rounded back to f32 before the
/// reciprocal square root). A 2560-wide group holding one large outlier loses
/// several digits of the small terms in an f32 accumulator; the double keeps
/// them.
pub fn grouped_rms_norm(x: &[f32], weight: &[f32], group: usize, eps: f32) -> Vec<f32> {
    assert_eq!(
        x.len(),
        weight.len(),
        "row width does not match weight width"
    );
    assert!(
        group > 0 && x.len().is_multiple_of(group),
        "width is not a whole number of groups"
    );
    let mut out = vec![0.0f32; x.len()];
    for g in 0..x.len() / group {
        let lo = g * group;
        let hi = lo + group;
        let mut sum_sq = 0.0f64;
        for v in &x[lo..hi] {
            sum_sq += (v * v) as f64;
        }
        let mean = (sum_sq / group as f64) as f32;
        let scale = 1.0f32 / (mean + eps).sqrt();
        for i in lo..hi {
            out[i] = x[i] * scale * weight[i];
        }
    }
    out
}

/// [`grouped_rms_norm`] over `[n_tok × weight.len()]` rows.
pub fn grouped_rms_norm_batch(x: &[f32], weight: &[f32], group: usize, eps: f32) -> Vec<f32> {
    let width = weight.len();
    assert!(
        width > 0 && x.len().is_multiple_of(width),
        "input is not a whole number of rows"
    );
    let mut out = Vec::with_capacity(x.len());
    for row in x.chunks_exact(width) {
        out.extend(grouped_rms_norm(row, weight, group, eps));
    }
    out
}

/// RMS norm over the full row, multiplied by `weight`, then gated by
/// `act(z)` — the gated DeltaNet output norm.
///
/// The norm is taken over `x` alone; `z` only enters through the gate. `weight`
/// is a plain multiply (it was never zero-centered upstream).
///
/// The sum of squares accumulates in f64 for the same reason
/// [`grouped_rms_norm`] does.
pub fn gated_rms_norm(x: &[f32], weight: &[f32], z: &[f32], gate: ZGateRef, eps: f32) -> Vec<f32> {
    assert_eq!(
        x.len(),
        weight.len(),
        "row width does not match weight width"
    );
    assert_eq!(x.len(), z.len(), "gate width does not match row width");
    let mut sum_sq = 0.0f64;
    for v in x {
        sum_sq += (v * v) as f64;
    }
    let mean = (sum_sq / x.len() as f64) as f32;
    let scale = 1.0f32 / (mean + eps).sqrt();
    (0..x.len())
        .map(|i| x[i] * scale * weight[i] * gate.apply(z[i]))
        .collect()
}

/// [`gated_rms_norm`] over `[n_tok × weight.len()]` rows of `x` and `z`.
pub fn gated_rms_norm_batch(
    x: &[f32],
    weight: &[f32],
    z: &[f32],
    gate: ZGateRef,
    eps: f32,
) -> Vec<f32> {
    let width = weight.len();
    assert!(
        width > 0 && x.len().is_multiple_of(width),
        "input is not a whole number of rows"
    );
    assert_eq!(
        x.len(),
        z.len(),
        "gate has a different number of rows than the input"
    );
    let mut out = Vec::with_capacity(x.len());
    for (row, zrow) in x.chunks_exact(width).zip(z.chunks_exact(width)) {
        out.extend(gated_rms_norm(row, weight, zrow, gate, eps));
    }
    out
}

/// The shared hyper-connection read path: grouped-norm the carrier, build the
/// per-element mix weights through the low-rank bottleneck, and take the
/// mean over streams.
///
/// Returns `(mixed [hidden], normed [hc_count * hidden])`. The normed carrier
/// is returned because the injection head reads it too — and only it: the write
/// side goes back onto the raw stream.
fn hc_read(r: HcReadWeights<'_>, stream: &[f32], eps: f32) -> (Vec<f32>, Vec<f32>) {
    let HcReadWeights {
        hc_count,
        hidden,
        low_rank,
        norm_w,
        down_w,
        up_w,
    } = r;
    let width = hc_count * hidden;
    assert_eq!(
        stream.len(),
        width,
        "stream width does not match hc_count * hidden"
    );

    // Statistics per stream, so the streams keep their own scales.
    let normed = grouped_rms_norm(stream, norm_w, hidden, eps);

    // Low-rank bottleneck: down to `low_rank`, scaled by 1/hc_count (the
    // carrier is a sum over hc_count streams), silu, back up to full width.
    let mut down = matvec(down_w, &normed, low_rank, width);
    for v in down.iter_mut() {
        *v = silu(*v / hc_count as f32);
    }
    let mix = matvec(up_w, &down, width, low_rank)
        .into_iter()
        .map(sigmoid)
        .collect::<Vec<f32>>();

    // Mean — not sum — of the gated streams.
    let mut mixed = vec![0.0f32; hidden];
    for s in 0..hc_count {
        for j in 0..hidden {
            mixed[j] += mix[s * hidden + j] * normed[s * hidden + j];
        }
    }
    for v in mixed.iter_mut() {
        *v /= hc_count as f32;
    }
    (mixed, normed)
}

/// The slice of a hyper-connection gate that [`hc_read`] needs — everything
/// except the injection head, which only [`GatedResidualRef`] has.
struct HcReadWeights<'a> {
    hc_count: usize,
    hidden: usize,
    low_rank: usize,
    norm_w: &'a [f32],
    down_w: &'a [f32],
    up_w: &'a [f32],
}

/// One block's hyper-connection gate: the read that produces the block input
/// and the per-stream injection weights, and the write-back of the block
/// output.
///
/// Every attention and every MLP block owns one of these (GGUF
/// `hc_{attn,ffn}_{norm,down,up,inject}`).
#[derive(Debug, Clone)]
pub struct GatedResidualRef {
    /// Number of parallel residual streams.
    pub hc_count: usize,
    /// Width of one stream — the model's hidden size.
    pub hidden: usize,
    /// Width of the mix-weight bottleneck.
    pub low_rank: usize,
    /// Grouped-norm weight, `[hc_count * hidden]`, multiply-ready.
    pub norm_w: Vec<f32>,
    /// Bottleneck down projection, `[low_rank][hc_count * hidden]`.
    pub down_w: Vec<f32>,
    /// Bottleneck up projection, `[hc_count * hidden][low_rank]`.
    pub up_w: Vec<f32>,
    /// Injection head, `[hc_count][hc_count * hidden]`.
    pub inject_w: Vec<f32>,
}

impl GatedResidualRef {
    /// Width of the residual carrier, `hc_count * hidden`.
    pub fn width(&self) -> usize {
        self.hc_count * self.hidden
    }

    /// The read-path weights, borrowed for [`hc_read`].
    fn read_weights(&self) -> HcReadWeights<'_> {
        HcReadWeights {
            hc_count: self.hc_count,
            hidden: self.hidden,
            low_rank: self.low_rank,
            norm_w: &self.norm_w,
            down_w: &self.down_w,
            up_w: &self.up_w,
        }
    }

    /// Reads one token's carrier row.
    ///
    /// Returns `(mixed [hidden], inject [hc_count])`: the vector the block runs
    /// on, and the per-stream strengths [`write`](Self::write) will scale the
    /// block output by. `inject` is `2·sigmoid(·)`, so it spans `(0, 2)` and is
    /// centered on 1 — an untrained gate leaves the stream's scale alone.
    pub fn read(&self, stream: &[f32], eps: f32) -> (Vec<f32>, Vec<f32>) {
        let (mixed, normed) = hc_read(self.read_weights(), stream, eps);
        let inject = matvec(&self.inject_w, &normed, self.hc_count, self.width())
            .into_iter()
            .map(|v| 2.0 * sigmoid(v / self.hc_count as f32))
            .collect();
        (mixed, inject)
    }

    /// [`read`](Self::read) over `[n_tok × width]`, returning
    /// `([n_tok × hidden], [n_tok × hc_count])`.
    pub fn read_batch(&self, streams: &[f32], eps: f32) -> (Vec<f32>, Vec<f32>) {
        let width = self.width();
        assert!(
            streams.len().is_multiple_of(width),
            "input is not a whole number of rows"
        );
        let n_tok = streams.len() / width;
        let mut mixed = Vec::with_capacity(n_tok * self.hidden);
        let mut inject = Vec::with_capacity(n_tok * self.hc_count);
        for row in streams.chunks_exact(width) {
            let (m, i) = self.read(row, eps);
            mixed.extend(m);
            inject.extend(i);
        }
        (mixed, inject)
    }

    /// Writes one block output back into the RAW carrier row:
    /// `stream[s] += block_out * inject[s]` for every stream `s`.
    ///
    /// The normed carrier from [`read`](Self::read) is deliberately not used
    /// here — normalization feeds the block, it does not replace the residual.
    pub fn write(&self, stream: &mut [f32], block_out: &[f32], inject: &[f32]) {
        assert_eq!(
            stream.len(),
            self.width(),
            "stream width does not match hc_count * hidden"
        );
        assert_eq!(
            block_out.len(),
            self.hidden,
            "block output is not hidden-wide"
        );
        assert_eq!(
            inject.len(),
            self.hc_count,
            "injection vector is not hc_count-wide"
        );
        for s in 0..self.hc_count {
            let g = inject[s];
            for j in 0..self.hidden {
                stream[s * self.hidden + j] += block_out[j] * g;
            }
        }
    }

    /// [`write`](Self::write) over `[n_tok × width]` carrier rows,
    /// `[n_tok × hidden]` block outputs and `[n_tok × hc_count]` injections.
    pub fn write_batch(&self, streams: &mut [f32], block_out: &[f32], inject: &[f32]) {
        let width = self.width();
        assert!(
            streams.len().is_multiple_of(width),
            "input is not a whole number of rows"
        );
        let n_tok = streams.len() / width;
        assert_eq!(
            block_out.len(),
            n_tok * self.hidden,
            "block output row count differs"
        );
        assert_eq!(
            inject.len(),
            n_tok * self.hc_count,
            "injection row count differs"
        );
        for (t, row) in streams.chunks_exact_mut(width).enumerate() {
            self.write(
                row,
                &block_out[t * self.hidden..(t + 1) * self.hidden],
                &inject[t * self.hc_count..(t + 1) * self.hc_count],
            );
        }
    }
}

/// The model tail's `hyper_connection_mixer`: the same read path as
/// [`GatedResidualRef`] with no injection head, collapsing the carrier to one
/// `hidden`-wide vector for the lm_head (GGUF `output_hc_{norm,down,up}`).
#[derive(Debug, Clone)]
pub struct HcMixerRef {
    /// Number of parallel residual streams.
    pub hc_count: usize,
    /// Width of one stream — the model's hidden size.
    pub hidden: usize,
    /// Width of the mix-weight bottleneck.
    pub low_rank: usize,
    /// Grouped-norm weight, `[hc_count * hidden]`, multiply-ready.
    pub norm_w: Vec<f32>,
    /// Bottleneck down projection, `[low_rank][hc_count * hidden]`.
    pub down_w: Vec<f32>,
    /// Bottleneck up projection, `[hc_count * hidden][low_rank]`.
    pub up_w: Vec<f32>,
}

impl HcMixerRef {
    /// Width of the residual carrier, `hc_count * hidden`.
    pub fn width(&self) -> usize {
        self.hc_count * self.hidden
    }

    /// The read-path weights, borrowed for [`hc_read`].
    fn read_weights(&self) -> HcReadWeights<'_> {
        HcReadWeights {
            hc_count: self.hc_count,
            hidden: self.hidden,
            low_rank: self.low_rank,
            norm_w: &self.norm_w,
            down_w: &self.down_w,
            up_w: &self.up_w,
        }
    }

    /// Collapses one carrier row to `[hidden]`.
    pub fn mix(&self, stream: &[f32], eps: f32) -> Vec<f32> {
        let (mixed, _) = hc_read(self.read_weights(), stream, eps);
        mixed
    }

    /// [`mix`](Self::mix) over `[n_tok × width]`, returning `[n_tok × hidden]`.
    pub fn mix_batch(&self, streams: &[f32], eps: f32) -> Vec<f32> {
        let width = self.width();
        assert!(
            streams.len().is_multiple_of(width),
            "input is not a whole number of rows"
        );
        let mut out = Vec::with_capacity(streams.len() / width * self.hidden);
        for row in streams.chunks_exact(width) {
            out.extend(self.mix(row, eps));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const FIXTURE_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/qwen4exp/");

    fn fixture(name: &str) -> Value {
        let path = format!("{FIXTURE_DIR}{name}");
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("{path}: {e}"))
    }

    /// Fixture floats are the shortest f64 repr of an exact f32, so parsing as
    /// f64 and casting recovers the original bits.
    fn vec1(v: &Value) -> Vec<f32> {
        v.as_array()
            .expect("expected an array")
            .iter()
            .map(|x| x.as_f64().expect("expected a number") as f32)
            .collect()
    }

    /// A `[rows][cols]` fixture array flattened row-major.
    fn vec2(v: &Value) -> Vec<f32> {
        v.as_array()
            .expect("expected an array of rows")
            .iter()
            .flat_map(vec1)
            .collect()
    }

    fn rows(v: &Value) -> usize {
        v.as_array().expect("expected an array of rows").len()
    }

    fn f32_field(v: &Value, key: &str) -> f32 {
        v[key]
            .as_f64()
            .unwrap_or_else(|| panic!("missing float field {key}")) as f32
    }

    fn assert_close(got: &[f32], want: &[f32], tol: f32, what: &str) {
        assert_eq!(
            got.len(),
            want.len(),
            "{what}: length {} != {}",
            got.len(),
            want.len()
        );
        let mut worst = 0.0f32;
        let mut worst_at = 0usize;
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            let d = (g - w).abs();
            if d > worst {
                worst = d;
                worst_at = i;
            }
        }
        assert!(
            worst <= tol,
            "{what}: max abs deviation {worst} > {tol} at index {worst_at} \
             (got {}, want {})",
            got[worst_at],
            want[worst_at]
        );
    }

    /// The elementwise floor the fixture reports, widened as the README
    /// suggests: an f32 reimplementation should land well inside it.
    fn suggested_tol(floor: f32) -> f32 {
        (10.0 * floor).max(1e-6)
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    /// The GGUF converter's baked +1: `weight_mult` is exactly `1 + weight_hf`
    /// in f32, which is why the GGUF path multiplies and never adds.
    fn assert_mult_is_hf_plus_one(hf: &Value, mult: &Value, what: &str) {
        let hf = vec1(hf);
        let mult = vec1(mult);
        assert_eq!(hf.len(), mult.len(), "{what}: weight lengths differ");
        for (i, (h, m)) in hf.iter().zip(mult.iter()).enumerate() {
            assert_eq!(
                (1.0f32 + h).to_bits(),
                m.to_bits(),
                "{what}: 1 + weight_hf[{i}] ({}) != weight_mult[{i}] ({m})",
                1.0f32 + h
            );
        }
    }

    fn gated_residual_ref(weights: &Value, config: &Value) -> GatedResidualRef {
        GatedResidualRef {
            hc_count: config["hc_count"].as_u64().unwrap() as usize,
            hidden: config["hidden_size"].as_u64().unwrap() as usize,
            low_rank: config["hc_lowrank"].as_u64().unwrap() as usize,
            norm_w: vec1(&weights["hc_norm_weight_mult"]),
            down_w: vec2(&weights["input_mix_weight_down"]),
            up_w: vec2(&weights["input_mix_weight_up"]),
            inject_w: vec2(&weights["block_inject_weight"]),
        }
    }

    #[test]
    fn grouped_rms_norm_matches_fixture() {
        let f = fixture("grouped_rmsnorm.json");
        let group = f["config"]["group_size"].as_u64().unwrap() as usize;
        let eps = f32_field(&f["config"], "eps");
        let weight = vec1(&f["weight_mult"]);
        let input = vec2(&f["input"]);
        let want = vec2(&f["output"]);
        let tol = suggested_tol(f32_field(&f, "f64_delta_output"));

        let got = grouped_rms_norm_batch(&input, &weight, group, eps);
        assert_close(&got, &want, tol, "grouped_rmsnorm output");

        // The batch variant is exactly the per-row one.
        let width = weight.len();
        for (t, row) in input.chunks_exact(width).enumerate() {
            let one = grouped_rms_norm(row, &weight, group, eps);
            assert_eq!(
                one,
                got[t * width..(t + 1) * width],
                "row {t} differs from the batch"
            );
        }
    }

    /// Grouping is not cosmetic: with per-group scales decades apart, one set of
    /// statistics over the whole row gives a materially different answer. The
    /// ungrouped path is exactly `group = dim`, and its output must match the
    /// fixture's contrast tensor while diverging from the grouped one.
    #[test]
    fn grouped_rms_norm_differs_from_ungrouped() {
        let f = fixture("grouped_rmsnorm.json");
        let dim = f["config"]["dim"].as_u64().unwrap() as usize;
        let group = f["config"]["group_size"].as_u64().unwrap() as usize;
        let eps = f32_field(&f["config"], "eps");
        let weight = vec1(&f["weight_mult"]);
        let input = vec2(&f["input"]);
        let grouped = vec2(&f["output"]);
        let tol = suggested_tol(f32_field(&f, "f64_delta_output"));

        let ungrouped = grouped_rms_norm_batch(&input, &weight, dim, eps);
        assert_close(
            &ungrouped,
            &vec2(&f["output_ungrouped_for_contrast"]),
            tol,
            "ungrouped contrast output",
        );
        assert!(
            max_abs_diff(&ungrouped, &grouped) > tol,
            "ungrouped output is within tolerance of the grouped one — the \
             fixture no longer pins that grouping matters"
        );
        assert_ne!(
            group, dim,
            "the contrast is only meaningful for a real grouping"
        );
    }

    #[test]
    fn grouped_rms_norm_weight_mult_is_hf_plus_one() {
        let f = fixture("grouped_rmsnorm.json");
        assert_mult_is_hf_plus_one(&f["weight_hf"], &f["weight_mult"], "grouped_rmsnorm");
    }

    #[test]
    fn gated_rms_norm_matches_fixture() {
        let f = fixture("gated_norm.json");
        let eps = f32_field(&f["config"], "eps");
        for (arm, gate) in [("sigmoid", ZGateRef::Sigmoid), ("silu", ZGateRef::Silu)] {
            let a = &f[arm];
            let weight = vec1(&a["norm_weight"]);
            let x = vec2(&a["o"]);
            let z = vec2(&a["z"]);
            let want = vec2(&a["output"]);
            let tol = suggested_tol(f32_field(a, "f64_delta"));

            let got = gated_rms_norm_batch(&x, &weight, &z, gate, eps);
            assert_close(&got, &want, tol, &format!("gated_norm {arm}"));

            let width = weight.len();
            for (t, (xr, zr)) in x.chunks_exact(width).zip(z.chunks_exact(width)).enumerate() {
                let one = gated_rms_norm(xr, &weight, zr, gate, eps);
                assert_eq!(
                    one,
                    got[t * width..(t + 1) * width],
                    "{arm} row {t} differs from the batch"
                );
            }
        }
    }

    /// The two gate arms are genuinely different functions; a fixture that
    /// stopped distinguishing them would let a swapped enum arm pass.
    #[test]
    fn gated_rms_norm_arms_disagree() {
        let f = fixture("gated_norm.json");
        let eps = f32_field(&f["config"], "eps");
        let a = &f["sigmoid"];
        let weight = vec1(&a["norm_weight"]);
        let x = vec2(&a["o"]);
        let z = vec2(&a["z"]);
        let with_sigmoid = gated_rms_norm_batch(&x, &weight, &z, ZGateRef::Sigmoid, eps);
        let with_silu = gated_rms_norm_batch(&x, &weight, &z, ZGateRef::Silu, eps);
        assert!(
            max_abs_diff(&with_sigmoid, &with_silu) > 1e-3,
            "the ZGate arms coincide"
        );
    }

    #[test]
    fn gated_residual_read_matches_fixture() {
        let f = fixture("gated_residual.json");
        let eps = f32_field(&f["config"], "rms_norm_eps");
        let gr = gated_residual_ref(&f["weights"], &f["config"]);
        let streams = vec2(&f["input_stream"]);
        let want_mixed = vec2(&f["mixed_output"]);
        let want_inject = vec2(&f["injection_weights"]);
        let tol_mixed = suggested_tol(f32_field(&f, "f64_delta_mixed"));
        let tol_inject = suggested_tol(f32_field(&f, "f64_delta_injection"));

        let (mixed, inject) = gr.read_batch(&streams, eps);
        assert_close(
            &mixed,
            &want_mixed,
            tol_mixed,
            "gated_residual mixed_output",
        );
        assert_close(
            &inject,
            &want_inject,
            tol_inject,
            "gated_residual injection_weights",
        );

        let width = gr.width();
        for (t, row) in streams.chunks_exact(width).enumerate() {
            let (m, i) = gr.read(row, eps);
            assert_eq!(
                m,
                mixed[t * gr.hidden..(t + 1) * gr.hidden],
                "mixed row {t}"
            );
            assert_eq!(
                i,
                inject[t * gr.hc_count..(t + 1) * gr.hc_count],
                "inject row {t}"
            );
        }
    }

    /// The write-back lands on the RAW stream, not the normed one. The fixture
    /// runs an identity block, so `block_out` is the mixed vector itself.
    #[test]
    fn gated_residual_write_matches_fixture() {
        let f = fixture("gated_residual.json");
        let eps = f32_field(&f["config"], "rms_norm_eps");
        let gr = gated_residual_ref(&f["weights"], &f["config"]);
        let streams = vec2(&f["input_stream"]);
        let want = vec2(&f["stream_out_identity_block"]);
        let tol = suggested_tol(f32_field(&f, "f64_delta_stream_out"));

        let (mixed, inject) = gr.read_batch(&streams, eps);
        let mut out = streams.clone();
        gr.write_batch(&mut out, &mixed, &inject);
        assert_close(&out, &want, tol, "gated_residual stream_out_identity_block");

        let width = gr.width();
        for (t, row) in streams.chunks_exact(width).enumerate() {
            let mut one = row.to_vec();
            gr.write(
                &mut one,
                &mixed[t * gr.hidden..(t + 1) * gr.hidden],
                &inject[t * gr.hc_count..(t + 1) * gr.hc_count],
            );
            assert_eq!(one, out[t * width..(t + 1) * width], "stream row {t}");
        }
    }

    /// A zero block output must leave the carrier untouched no matter what the
    /// injection weights are — the write is purely additive.
    #[test]
    fn gated_residual_write_of_zero_is_identity() {
        let f = fixture("gated_residual.json");
        let eps = f32_field(&f["config"], "rms_norm_eps");
        let gr = gated_residual_ref(&f["weights"], &f["config"]);
        let streams = vec2(&f["input_stream"]);
        let (_, inject) = gr.read_batch(&streams, eps);
        let mut out = streams.clone();
        let zeros = vec![0.0f32; streams.len() / gr.width() * gr.hidden];
        gr.write_batch(&mut out, &zeros, &inject);
        assert_eq!(out, streams);
    }

    /// Each stream is scaled by ITS OWN injection weight, and by nothing else.
    /// With a distinct strength per stream — including a negative one and an
    /// exact zero — stream `s` must move by exactly `block_out * inject[s]`,
    /// which pins both the per-stream indexing and the broadcast of the single
    /// `hidden`-wide block output across all streams.
    #[test]
    fn gated_residual_write_scales_each_stream_by_its_own_injection() {
        let f = fixture("gated_residual.json");
        let gr = gated_residual_ref(&f["weights"], &f["config"]);
        let streams = vec2(&f["input_stream"]);
        let width = gr.width();
        let n_tok = streams.len() / width;

        // Non-uniform per stream and per token, so no stream can borrow
        // another's weight and no token can borrow another's row.
        let inject: Vec<f32> = (0..n_tok * gr.hc_count)
            .map(|i| i as f32 * 0.25 - 1.25)
            .collect();
        let block_out: Vec<f32> = (0..n_tok * gr.hidden)
            .map(|i| (i as f32 * 0.037).sin())
            .collect();

        let mut out = streams.clone();
        gr.write_batch(&mut out, &block_out, &inject);

        for t in 0..n_tok {
            for s in 0..gr.hc_count {
                let g = inject[t * gr.hc_count + s];
                for j in 0..gr.hidden {
                    let before = streams[t * width + s * gr.hidden + j];
                    let after = out[t * width + s * gr.hidden + j];
                    let want = before + block_out[t * gr.hidden + j] * g;
                    assert_eq!(
                        after.to_bits(),
                        want.to_bits(),
                        "token {t} stream {s} element {j}"
                    );
                }
            }
        }
    }

    #[test]
    fn hc_mixer_matches_fixture() {
        let f = fixture("gated_residual.json");
        let tm = &f["tail_mixer"];
        let eps = f32_field(&f["config"], "rms_norm_eps");
        let mixer = HcMixerRef {
            hc_count: f["config"]["hc_count"].as_u64().unwrap() as usize,
            hidden: f["config"]["hidden_size"].as_u64().unwrap() as usize,
            low_rank: f["config"]["hc_lowrank"].as_u64().unwrap() as usize,
            norm_w: vec1(&tm["weights"]["hc_norm_weight_mult"]),
            down_w: vec2(&tm["weights"]["input_mix_weight_down"]),
            up_w: vec2(&tm["weights"]["input_mix_weight_up"]),
        };
        let streams = vec2(&tm["input_stream"]);
        let want = vec2(&tm["mixed_output"]);
        let tol = suggested_tol(f32_field(&f, "f64_delta_tail_mixed"));

        assert_eq!(rows(&tm["input_stream"]), rows(&tm["mixed_output"]));
        let got = mixer.mix_batch(&streams, eps);
        assert_close(&got, &want, tol, "tail_mixer mixed_output");

        let width = mixer.width();
        for (t, row) in streams.chunks_exact(width).enumerate() {
            let one = mixer.mix(row, eps);
            assert_eq!(
                one,
                got[t * mixer.hidden..(t + 1) * mixer.hidden],
                "tail row {t}"
            );
        }
    }

    #[test]
    fn gated_residual_weight_mult_is_hf_plus_one() {
        let f = fixture("gated_residual.json");
        assert_mult_is_hf_plus_one(
            &f["weights"]["hc_norm_weight_hf"],
            &f["weights"]["hc_norm_weight_mult"],
            "gated_residual hc_norm",
        );
        assert_mult_is_hf_plus_one(
            &f["tail_mixer"]["weights"]["hc_norm_weight_hf"],
            &f["tail_mixer"]["weights"]["hc_norm_weight_mult"],
            "tail_mixer hc_norm",
        );
    }

    /// The fixture's stored weight shapes are the reference's `[out][in]`
    /// layout. A transposed read would silently produce garbage of the right
    /// length for `up_w` only if `low_rank == width`, so pin the shapes.
    #[test]
    fn fixture_weight_shapes_are_out_by_in() {
        let f = fixture("gated_residual.json");
        let hc_count = f["config"]["hc_count"].as_u64().unwrap() as usize;
        let hidden = f["config"]["hidden_size"].as_u64().unwrap() as usize;
        let low_rank = f["config"]["hc_lowrank"].as_u64().unwrap() as usize;
        let width = hc_count * hidden;
        let w = &f["weights"];
        assert_eq!(rows(&w["input_mix_weight_down"]), low_rank);
        assert_eq!(rows(&w["input_mix_weight_down"][0]), width);
        assert_eq!(rows(&w["input_mix_weight_up"]), width);
        assert_eq!(rows(&w["input_mix_weight_up"][0]), low_rank);
        assert_eq!(rows(&w["block_inject_weight"]), hc_count);
        assert_eq!(rows(&w["block_inject_weight"][0]), width);
        assert_eq!(vec1(&w["hc_norm_weight_mult"]).len(), width);
    }
}
