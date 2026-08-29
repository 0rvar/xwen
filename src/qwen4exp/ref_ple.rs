//! PLE (per-layer embeddings): the n-gram hash table and its injection layer.
//!
//! Frozen CPU f32 correctness oracle. Never optimized; the Metal path is graded
//! against it. Ground truth is llama.cpp
//! `reference/llama.cpp/src/models/qwen4exp.cpp` (`set_input` for the hash,
//! `build_ple` for the layer) with the numbers pinned by
//! `tests/fixtures/qwen4exp/ple.json` (transformers-generated).
//!
//! Two halves, deliberately separate because they live in different places at
//! runtime: `PleHashRef` is host-side integer work over raw token ids that
//! produces table row indices, and `PleLayerRef` is the float forward that
//! gathers those rows and turns them into the addend for the hyper-connection
//! stream.

/// The grouped RMS norm is shared with the hyper-connection reference rather
/// than reimplemented: one implementation of the math means one set of shape
/// asserts, and PLE's three norm weights are all `[hc_count * hidden]` wide
/// with `hidden`-wide groups — a `[hidden]`-wide weight must panic, not
/// silently leave streams 1.. at zero.
use super::ref_hc::grouped_rms_norm;

/// The n-gram hash. Turns raw token ids into one table row index per head.
///
/// Heads are grouped by n-gram order: with `heads_per_ngram = p`, heads
/// `0..p` are the 2-gram heads, `p..2p` the 3-gram heads, and so on, so
/// `n_heads == (ngram_size - 1) * heads_per_ngram`. There are no unigram heads.
///
/// The multipliers are 64-bit and shipped as GGUF metadata
/// (`ple.layer_multipliers` / `ple.head_vocab_sizes` / `ple.head_offsets`);
/// they are read, never recomputed. All arithmetic is wrapping u64 over RAW
/// token ids — no NFKC, no lowercasing.
pub struct PleHashRef {
    /// Longest n-gram order. `multipliers` has this many entries.
    pub ngram_size: usize,
    /// Heads per n-gram order.
    pub heads_per_ngram: usize,
    /// One multiplier per context slot, `ngram_size` entries. Slot 0 is the
    /// token itself, slot s the predecessor s positions back.
    pub multipliers: Vec<u64>,
    /// Per-head modulus, `n_heads` entries.
    pub head_vocab_sizes: Vec<u64>,
    /// Per-head base row, `n_heads` entries.
    pub head_offsets: Vec<u64>,
    /// The segment separator. For the shipped checkpoint this is the scalar
    /// `<|endoftext|>` (248044), NOT the chat `<|im_end|>`.
    pub eos: u32,
}

impl PleHashRef {
    /// There are no unigram heads, so an n-gram size below 2 describes no
    /// table at all. llama.cpp raises on it at load
    /// (`reference/llama.cpp/src/models/qwen4exp.cpp:66-68`); here it is an
    /// assert so a mis-parsed metadata key cannot reach the hash loop.
    fn check_geometry(&self) {
        assert!(
            self.ngram_size >= 2,
            "ngram_size must be at least 2 (no unigram heads exist)"
        );
    }

    pub fn n_heads(&self) -> usize {
        self.check_geometry();
        (self.ngram_size - 1) * self.heads_per_ngram
    }

    /// The shifted-token context, one level per predecessor slot:
    /// `levels[s - 1][i]` is the context slot `s` of `tokens[i]` (so
    /// `levels[0]` is the shift-by-1 stream, `levels[1]` the shift-by-2 one).
    /// Each level has `tokens.len()` entries.
    ///
    /// `history` is the rolling raw-token state, oldest first, holding at most
    /// `ngram_size - 1` tokens; a shorter history means the sequence starts
    /// there and the missing predecessors read as eos.
    ///
    /// The shift never crosses a segment boundary: once a predecessor is eos
    /// (or missing), that slot and every deeper one read as eos. A token's own
    /// eos does not cut its context — only the tokens before it do.
    pub fn shifted(&self, history: &[u32], tokens: &[u32]) -> Vec<Vec<u32>> {
        self.check_geometry();
        let n_prev = self.ngram_size - 1;
        let hist = &history[history.len().saturating_sub(n_prev)..];
        let mut levels: Vec<Vec<u32>> = vec![Vec::with_capacity(tokens.len()); n_prev];
        for i in 0..tokens.len() {
            let mut cut = false;
            for (slot, level) in levels.iter_mut().enumerate() {
                let s = slot + 1;
                // the predecessor s positions back; off the front of the sequence
                // reads the same as an eos, and so does anything past a cut
                let back = hist.len() + i;
                let t = if cut || back < s {
                    None
                } else {
                    let gi = back - s;
                    Some(if gi < hist.len() {
                        hist[gi]
                    } else {
                        tokens[gi - hist.len()]
                    })
                };
                cut = t.is_none_or(|v| v == self.eos);
                level.push(if cut { self.eos } else { t.unwrap() });
            }
        }
        levels
    }

    /// Table row indices for `tokens`, flat with an `n_heads()` stride:
    /// `rows[i * n_heads + h]` is the row for token `i`, head `h`.
    ///
    /// Per order n, `mixed = ctx[0]*m[0] ^ ctx[1]*m[1] ^ … ^ ctx[n-1]*m[n-1]`
    /// (wrapping), and every head of that order reads
    /// `mixed % head_vocab_size + head_offset`.
    pub fn rows(&self, history: &[u32], tokens: &[u32]) -> Vec<u64> {
        self.check_geometry();
        let n_heads = self.n_heads();
        let levels = self.shifted(history, tokens);
        let mut out = vec![0u64; tokens.len() * n_heads];
        for (i, &tok) in tokens.iter().enumerate() {
            let mut ctx = Vec::with_capacity(self.ngram_size);
            ctx.push(u64::from(tok));
            ctx.extend(levels.iter().map(|l| u64::from(l[i])));
            for n in 2..=self.ngram_size {
                let mut mixed = ctx[0].wrapping_mul(self.multipliers[0]);
                for (c, m) in ctx[1..n].iter().zip(&self.multipliers[1..n]) {
                    mixed ^= c.wrapping_mul(*m);
                }
                let base = (n - 2) * self.heads_per_ngram;
                for g in 0..self.heads_per_ngram {
                    let h = base + g;
                    out[i * n_heads + h] = mixed % self.head_vocab_sizes[h] + self.head_offsets[h];
                }
            }
        }
        out
    }

    /// The rolling token history after consuming `tokens`: the last
    /// `ngram_size - 1` raw ids of `history ++ tokens`, oldest first, left-padded
    /// with eos when fewer than that many tokens have been seen. Feeding this
    /// back into `rows` makes a chunked run identical to a single-shot one.
    pub fn next_history(&self, history: &[u32], tokens: &[u32]) -> Vec<u32> {
        self.check_geometry();
        let n_prev = self.ngram_size - 1;
        let mut all: Vec<u32> = history
            .iter()
            .copied()
            .chain(tokens.iter().copied())
            .collect();
        while all.len() < n_prev {
            all.insert(0, self.eos);
        }
        all.split_off(all.len() - n_prev)
    }
}

/// `sigmoid(sign(s) · √max(|s|, 1e-6))` — the PLE gate's scalar map.
///
/// The clamp is what HF applies (modular line 770) and llama.cpp mirrors with
/// `ggml_clamp(abs(s), 1e-6, INF)`. `sign(0) == 0`, so `s == 0` gates to exactly
/// 0.5; Rust's `f32::signum` returns 1.0 there and must not be used.
pub fn gate_function_probe(s: f32) -> f32 {
    sigmoid(signed_sqrt(s))
}

/// `sign(s) · √max(|s|, 1e-6)` — the gate before its sigmoid.
///
/// NaN propagates explicitly. Without the guard it would not: `f32::max`
/// returns the non-NaN operand, so `NaN.abs().max(1e-6)` is `1e-6` and the
/// zero sign would multiply it away to a perfectly innocent `0.0` — a NaN
/// upstream would gate to exactly 0.5 and vanish.
fn signed_sqrt(s: f32) -> f32 {
    if s.is_nan() {
        return f32::NAN;
    }
    let sign = if s > 0.0 {
        1.0
    } else if s < 0.0 {
        -1.0
    } else {
        0.0
    };
    sign * s.abs().max(1e-6).sqrt()
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

/// `y = W x` with `w` row-major `[out, inp]`.
///
/// The shape assert is load-bearing: a too-tall weight would otherwise be
/// silently truncated to its first `out_dim` rows and yield a plausible wrong
/// answer (the PLE value projection is square at every shipped geometry, so
/// nothing downstream would notice).
fn matvec(w: &[f32], x: &[f32], out_dim: usize) -> Vec<f32> {
    assert_eq!(
        w.len(),
        out_dim * x.len(),
        "matrix is not [out_dim][x.len()]"
    );
    w.chunks_exact(x.len())
        .map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum())
        .collect()
}

/// Every intermediate of one `PleLayerRef::forward`, so each can be pinned
/// against the fixture independently. All token-major `[n_tok, …]`.
pub struct PleIntermediates {
    /// Gathered table rows concatenated per token, `[n_tok, n_heads * head_dim]`.
    pub embeddings: Vec<f32>,
    /// Per-stream `⟨key, query⟩ / √hidden` before the signed sqrt, `[n_tok, hc_count]`.
    pub gate_raw: Vec<f32>,
    /// `sign(s) · √max(|s|, 1e-6)`, still before the sigmoid, `[n_tok, hc_count]`.
    pub gate_signed_sqrt: Vec<f32>,
    /// `sigmoid(gate) · value`, broadcast over the streams, `[n_tok, hc_count * hidden]`.
    pub gated: Vec<f32>,
    /// The conv's input: `gated` through its own grouped norm.
    pub gated_normed: Vec<f32>,
    /// `silu(dilated depthwise conv(gated_normed))`.
    pub conv_out_silu: Vec<f32>,
    /// `gated + conv_out_silu` — the addend the caller adds to the
    /// hyper-connection stream BEFORE the attention hyper-connection read.
    pub output: Vec<f32>,
}

/// The PLE injection layer: one per checkpoint, living on a DeltaNet layer.
///
/// Weight layouts are row-major `[out, inp]` for the projections, flat
/// `[rows, head_dim]` for the table, and `[width, kernel]` for the depthwise
/// conv (one contiguous kernel per channel), where `width = hc_count * hidden`.
pub struct PleLayerRef {
    /// Model hidden size; also the grouped-norm group and the value width.
    pub hidden: usize,
    /// Hyper-connection stream count; the streams the gate and norms group over.
    pub hc_count: usize,
    /// n-gram head count, matching `PleHashRef::n_heads`.
    pub n_heads: usize,
    /// Per-head table row width. The embedding concat is `n_heads * head_dim`.
    pub head_dim: usize,
    /// `[rows, head_dim]`, row count padded well past the reachable range.
    pub table: Vec<f32>,
    /// `[hc_count * hidden, n_heads * head_dim]`.
    pub key_w: Vec<f32>,
    /// `[hc_count * hidden]`, grouped over `hidden`.
    pub key_norm_w: Vec<f32>,
    /// `[hidden, n_heads * head_dim]` — one value shared by every stream.
    pub value_w: Vec<f32>,
    /// `[hc_count * hidden]`, grouped over `hidden`; norms the incoming stream.
    pub query_norm_w: Vec<f32>,
    /// `[hc_count * hidden]`, grouped over `hidden`; norms the conv input.
    pub conv_norm_w: Vec<f32>,
    /// `[hc_count * hidden, k]`, depthwise: one kernel per channel.
    pub conv_w: Vec<f32>,
    /// Conv kernel size (4 in every shipped configuration).
    pub k: usize,
    /// Longest n-gram order, matching `PleHashRef::ngram_size`. The conv
    /// dilation is derived from it — see [`dilation`](Self::dilation).
    pub ngram_size: usize,
    /// RMS norm epsilon.
    pub eps: f32,
}

impl PleLayerRef {
    /// `hc_count * hidden` — the stream width the layer reads and writes.
    pub fn width(&self) -> usize {
        self.hc_count * self.hidden
    }

    /// The conv dilation, which is the n-gram size (3 in every shipped
    /// configuration), giving a receptive field of 10.
    ///
    /// This is DERIVED, never loaded: HF computes `conv_dilation =
    /// config.ngram_size` (modular:723) and llama.cpp reads
    /// `hparams.ple_ngram_size` for it, and there is no GGUF dilation key at
    /// all. A loader that invents one and defaults it to 1 would allocate a
    /// conv state it never fills and quietly shrink the receptive field.
    pub fn dilation(&self) -> usize {
        assert!(
            self.ngram_size >= 2,
            "ngram_size must be at least 2 (no unigram heads exist)"
        );
        self.ngram_size
    }

    /// Conv state columns per channel, `(k - 1) * dilation`.
    pub fn conv_state_len(&self) -> usize {
        (self.k - 1) * self.dilation()
    }

    /// A zeroed conv state, the correct start for a fresh sequence.
    /// Layout: channel-major, `[width, conv_state_len]`, oldest column first, so
    /// channel `c`'s most recent past input is the last element of its slice.
    pub fn zero_conv_state(&self) -> Vec<f32> {
        vec![0.0; self.width() * self.conv_state_len()]
    }

    /// One forward over `n_tok` tokens.
    ///
    /// `rows` is `PleHashRef::rows` output (flat, `n_heads` stride); `stream` is
    /// the incoming hyper-connection stream, token-major `[n_tok, width]`;
    /// `conv_state` is read for the pre-chunk history and overwritten with the
    /// new tail, so consecutive chunks of one sequence reproduce a single-shot
    /// run exactly.
    pub fn forward(
        &self,
        rows: &[u64],
        stream: &[f32],
        conv_state: &mut [f32],
    ) -> PleIntermediates {
        let width = self.width();
        let n_tok = rows.len() / self.n_heads;
        let emb_dim = self.n_heads * self.head_dim;
        let state_len = self.conv_state_len();

        let mut embeddings = Vec::with_capacity(n_tok * emb_dim);
        let mut gate_raw = Vec::with_capacity(n_tok * self.hc_count);
        let mut gate_signed_sqrt = Vec::with_capacity(n_tok * self.hc_count);
        let mut gated = Vec::with_capacity(n_tok * width);
        let mut gated_normed = Vec::with_capacity(n_tok * width);

        for (t, row_set) in rows.chunks_exact(self.n_heads).enumerate() {
            let emb_start = embeddings.len();
            for &row in row_set {
                let off = row as usize * self.head_dim;
                embeddings.extend_from_slice(&self.table[off..off + self.head_dim]);
            }
            let emb = &embeddings[emb_start..emb_start + emb_dim];

            let key = grouped_rms_norm(
                &matvec(&self.key_w, emb, width),
                &self.key_norm_w,
                self.hidden,
                self.eps,
            );
            let value = matvec(&self.value_w, emb, self.hidden);
            let query = grouped_rms_norm(
                &stream[t * width..(t + 1) * width],
                &self.query_norm_w,
                self.hidden,
                self.eps,
            );

            let scale = 1.0 / (self.hidden as f32).sqrt();
            for s in 0..self.hc_count {
                let span = s * self.hidden..(s + 1) * self.hidden;
                let dot: f32 = key[span.clone()]
                    .iter()
                    .zip(&query[span])
                    .map(|(a, b)| a * b)
                    .sum();
                let raw = dot * scale;
                gate_raw.push(raw);
                let signed = signed_sqrt(raw);
                gate_signed_sqrt.push(signed);
                let g = sigmoid(signed);
                gated.extend(value.iter().map(|v| g * v));
            }

            gated_normed.extend(grouped_rms_norm(
                &gated[t * width..(t + 1) * width],
                &self.conv_norm_w,
                self.hidden,
                self.eps,
            ));
        }

        // Depthwise causal conv, channel-major over the state ++ chunk line:
        //   out[c, t] = Σ_j w[c, j] · x[c, t - (k - 1 - j) · dilation]
        // so tap j = k-1 is the current position and tap 0 reaches the oldest
        // state column. Prepending the state is exactly what makes a chunked
        // prefill agree with a single-shot one.
        let mut padded = vec![0.0f32; width * (state_len + n_tok)];
        for c in 0..width {
            let dst = &mut padded[c * (state_len + n_tok)..(c + 1) * (state_len + n_tok)];
            dst[..state_len].copy_from_slice(&conv_state[c * state_len..(c + 1) * state_len]);
            for (t, d) in dst[state_len..].iter_mut().enumerate() {
                *d = gated_normed[t * width + c];
            }
        }

        let mut conv_out_silu = vec![0.0f32; n_tok * width];
        for c in 0..width {
            let line = &padded[c * (state_len + n_tok)..(c + 1) * (state_len + n_tok)];
            let kern = &self.conv_w[c * self.k..(c + 1) * self.k];
            for t in 0..n_tok {
                let acc: f32 = kern
                    .iter()
                    .enumerate()
                    .map(|(j, w)| w * line[state_len + t - (self.k - 1 - j) * self.dilation()])
                    .sum();
                conv_out_silu[t * width + c] = silu(acc);
            }
        }

        for c in 0..width {
            let line = &padded[c * (state_len + n_tok)..(c + 1) * (state_len + n_tok)];
            conv_state[c * state_len..(c + 1) * state_len]
                .copy_from_slice(&line[line.len() - state_len..]);
        }

        let output = gated
            .iter()
            .zip(&conv_out_silu)
            .map(|(g, c)| g + c)
            .collect();

        PleIntermediates {
            embeddings,
            gate_raw,
            gate_signed_sqrt,
            gated,
            gated_normed,
            conv_out_silu,
            output,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const FIXTURE: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/qwen4exp/ple.json"
    );

    fn fixture() -> Value {
        serde_json::from_str(&std::fs::read_to_string(FIXTURE).unwrap()).unwrap()
    }

    /// JSON floats are the shortest f64 repr of an exact f32, so the cast is bit-exact.
    fn vec_f32(v: &Value) -> Vec<f32> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect()
    }

    fn flat_f32(v: &Value) -> Vec<f32> {
        v.as_array().unwrap().iter().flat_map(vec_f32).collect()
    }

    fn vec_u32(v: &Value) -> Vec<u32> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect()
    }

    fn flat_u64(v: &Value) -> Vec<u64> {
        v.as_array()
            .unwrap()
            .iter()
            .flat_map(|r| {
                r.as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_u64().unwrap())
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    fn hasher(j: &Value) -> PleHashRef {
        let c = &j["config"];
        PleHashRef {
            ngram_size: c["ngram_size"].as_u64().unwrap() as usize,
            heads_per_ngram: c["heads_per_ngram"].as_u64().unwrap() as usize,
            multipliers: c["layer_multipliers_i64_str"]
                .as_array()
                .unwrap()
                .iter()
                .map(|s| s.as_str().unwrap().parse::<u64>().unwrap())
                .collect(),
            head_vocab_sizes: c["head_vocab_sizes"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap())
                .collect(),
            head_offsets: c["head_offsets"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap())
                .collect(),
            eos: c["eos_token_id"].as_u64().unwrap() as u32,
        }
    }

    fn layer(j: &Value) -> PleLayerRef {
        let c = &j["config"];
        let w = &j["weights"];
        PleLayerRef {
            hidden: c["hidden_size"].as_u64().unwrap() as usize,
            hc_count: c["hc_count"].as_u64().unwrap() as usize,
            n_heads: c["ngram_heads"].as_u64().unwrap() as usize,
            head_dim: c["head_dim_per_ngram"].as_u64().unwrap() as usize,
            table: flat_f32(&w["ngram_embedding_table"]),
            key_w: flat_f32(&w["key_proj"]),
            key_norm_w: vec_f32(&w["norm_key_weight_mult"]),
            value_w: flat_f32(&w["value_proj"]),
            query_norm_w: vec_f32(&w["norm_query_weight_mult"]),
            conv_norm_w: vec_f32(&w["norm_conv_weight_mult"]),
            conv_w: flat_f32(&w["conv1d_weight"]),
            k: c["ple_conv_kernel_size"].as_u64().unwrap() as usize,
            ngram_size: c["ngram_size"].as_u64().unwrap() as usize,
            eps: c["rms_norm_eps"].as_f64().unwrap() as f32,
        }
    }

    /// `f32::max` returns the non-NaN operand, so a NaN anywhere in either
    /// side would fold away to whatever the largest finite deviation is and
    /// the comparison would pass. Reject NaN before it can reach the fold.
    fn max_abs(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(x, y)| {
                assert!(!x.is_nan() && !y.is_nan(), "NaN in a compared tensor");
                (x - y).abs()
            })
            .fold(0.0f32, f32::max)
    }

    fn assert_close(what: &str, got: &[f32], want: &[f32], tol: f32) {
        let d = max_abs(got, want);
        assert!(d <= tol, "{what}: max abs {d:e} exceeds {tol:e}");
    }

    /// The fixture's shift arrays run over `[eos, eos] ++ input_ids`, which is
    /// what a no-cache sequence sees: the padding positions shift in eos too.
    #[test]
    fn shifts_never_cross_eos() {
        let j = fixture();
        let h = hasher(&j);
        let history = vec![h.eos; h.ngram_size - 1];
        let toks = vec_u32(&j["hash_case"]["token_history"]);
        let levels = h.shifted(&history, &toks);
        assert_eq!(levels[0], vec_u32(&j["hash_case"]["shift1_of_history"]));
        assert_eq!(levels[1], vec_u32(&j["hash_case"]["shift2_of_history"]));
    }

    #[test]
    fn hash_rows_match_fixture() {
        let j = fixture();
        let h = hasher(&j);
        let toks = vec_u32(&j["hash_case"]["input_ids"]);
        assert_eq!(h.rows(&[], &toks), flat_u64(&j["hash_case"]["row_indices"]));
        // A fresh sequence and an all-eos history are the same thing.
        let padded = vec![h.eos; h.ngram_size - 1];
        assert_eq!(h.rows(&padded, &toks), h.rows(&[], &toks));
    }

    #[test]
    fn hash_rows_survive_chunking() {
        let j = fixture();
        let h = hasher(&j);
        let toks = vec_u32(&j["hash_case"]["input_ids"]);
        let want = h.rows(&[], &toks);
        for split in 1..toks.len() {
            let (a, b) = toks.split_at(split);
            let mut got = h.rows(&[], a);
            let hist = h.next_history(&[], a);
            got.extend(h.rows(&hist, b));
            assert_eq!(got, want, "split at {split}");
        }
    }

    #[test]
    fn gate_probe_matches_fixture() {
        let j = fixture();
        let s = vec_f32(&j["gate_function_probe"]["s"]);
        let want = vec_f32(&j["gate_function_probe"]["sigmoid_gate"]);
        let got: Vec<f32> = s.iter().map(|&x| gate_function_probe(x)).collect();
        assert_close("gate probe", &got, &want, 1e-6);
        // s == 0 sits exactly on the fence: sign(0) == 0, so the gate is 0.5.
        assert_eq!(gate_function_probe(0.0), 0.5);
        // A NaN score must stay a NaN: the clamp would otherwise absorb it.
        assert!(gate_function_probe(f32::NAN).is_nan());
    }

    /// The dilation is the n-gram size, derived and never loaded. The fixture
    /// carries both numbers independently, which is what makes this an
    /// assertion rather than a tautology.
    #[test]
    fn conv_dilation_is_the_ngram_size() {
        let j = fixture();
        let l = layer(&j);
        assert_eq!(
            l.dilation(),
            j["config"]["conv_dilation"].as_u64().unwrap() as usize
        );
        assert_eq!(l.dilation(), l.ngram_size);
        assert_eq!(
            l.conv_state_len(),
            j["config"]["conv_state_len"].as_u64().unwrap() as usize
        );
    }

    /// `value_proj` is square at the tiny geometry (32×32) and square at the
    /// real one (2560×2560, port-doc trap #13), so no shape assertion can
    /// catch a transposed load — only the numbers can. Transposing it must
    /// break the fixture match; if this test ever passes trivially, the
    /// orientation has stopped being pinned.
    #[test]
    fn transposed_value_proj_does_not_match_the_fixture() {
        let j = fixture();
        let h = hasher(&j);
        let l = layer(&j);
        let c = &j["layer_case"];
        let toks = vec_u32(&c["input_ids"]);
        let rows = h.rows(&[], &toks);
        let stream = flat_f32(&c["hidden_stream_in"]);
        let want = flat_f32(&c["output"]);

        let n = l.hidden;
        assert_eq!(l.value_w.len(), n * n, "the hazard needs a square matrix");
        let mut transposed = layer(&j);
        transposed.value_w = (0..n)
            .flat_map(|i| (0..n).map(move |jj| (i, jj)))
            .map(|(i, jj)| l.value_w[jj * n + i])
            .collect();
        assert_ne!(
            transposed.value_w, l.value_w,
            "value_proj is symmetric, so a transpose is a no-op and this guard \
             cannot detect anything"
        );

        let mut state = transposed.zero_conv_state();
        let got = transposed.forward(&rows, &stream, &mut state).output;
        let d = max_abs(&got, &want);
        assert!(
            d > 1e-3,
            "a transposed value_proj still reproduced the fixture (max abs {d:e})"
        );
    }

    #[test]
    fn layer_forward_matches_fixture() {
        let j = fixture();
        let h = hasher(&j);
        let l = layer(&j);
        let c = &j["layer_case"];
        let toks = vec_u32(&c["input_ids"]);

        let rows = h.rows(&[], &toks);
        assert_eq!(rows, flat_u64(&c["hash_row_indices"]));

        let stream = flat_f32(&c["hidden_stream_in"]);
        let mut state = l.zero_conv_state();
        let out = l.forward(&rows, &stream, &mut state);

        // Table gathers are exact f32 copies.
        assert_eq!(out.embeddings, flat_f32(&c["ngram_embeddings"]));
        assert_close(
            "gate_raw_dot",
            &out.gate_raw,
            &flat_f32(&c["gate_raw_dot"]),
            4e-6,
        );
        assert_close(
            "gate_signed_sqrt",
            &out.gate_signed_sqrt,
            &flat_f32(&c["gate_signed_sqrt"]),
            1.5e-5,
        );
        assert_close(
            "gated_value",
            &out.gated,
            &flat_f32(&c["gated_value"]),
            4e-6,
        );
        assert_close(
            "gated_value_normed",
            &out.gated_normed,
            &flat_f32(&c["gated_value_normed"]),
            7e-6,
        );
        assert_close(
            "conv_out_silu",
            &out.conv_out_silu,
            &flat_f32(&c["conv_out_silu"]),
            3e-6,
        );
        assert_close("output", &out.output, &flat_f32(&c["output"]), 4.8e-6);

        // The state left behind is the conv input's last 9 columns.
        let n_tok = toks.len();
        let width = l.width();
        let state_len = l.conv_state_len();
        for c_i in 0..width {
            for s in 0..state_len {
                let t = n_tok - state_len + s;
                assert_eq!(
                    state[c_i * state_len + s],
                    out.gated_normed[t * width + c_i]
                );
            }
        }
    }

    /// Split-sequence equivalence: carrying the conv state and the token history
    /// across a chunk boundary must reproduce the single-shot run bit for bit.
    /// This is what P2's recurrent-state plumbing has to preserve.
    #[test]
    fn chunked_forward_matches_single_shot() {
        let j = fixture();
        let h = hasher(&j);
        let l = layer(&j);
        let c = &j["layer_case"];
        let toks = vec_u32(&c["input_ids"]);
        let stream = flat_f32(&c["hidden_stream_in"]);
        let width = l.width();

        let mut state = l.zero_conv_state();
        let want = l.forward(&h.rows(&[], &toks), &stream, &mut state).output;

        for split in 1..toks.len() {
            let (ta, tb) = toks.split_at(split);
            let mut chunk_state = l.zero_conv_state();
            let mut got = l
                .forward(&h.rows(&[], ta), &stream[..split * width], &mut chunk_state)
                .output;
            let hist = h.next_history(&[], ta);
            got.extend(
                l.forward(
                    &h.rows(&hist, tb),
                    &stream[split * width..],
                    &mut chunk_state,
                )
                .output,
            );
            assert_eq!(got, want, "split at {split}");
        }
    }

    /// Three chunks, not two. A two-chunk split only ever reads the state the
    /// first chunk wrote, so an implementation that writes the state correctly
    /// once and then corrupts it (writing the chunk's own input instead of the
    /// running tail, say, or forgetting that a chunk shorter than the state
    /// must carry old columns forward) still passes. The third chunk reads a
    /// state written by a forward that itself started from a non-zero state.
    #[test]
    fn three_chunk_forward_matches_single_shot() {
        let j = fixture();
        let h = hasher(&j);
        let l = layer(&j);
        let c = &j["layer_case"];
        let toks = vec_u32(&c["input_ids"]);
        let stream = flat_f32(&c["hidden_stream_in"]);
        let width = l.width();
        let n_tok = toks.len();

        let mut state = l.zero_conv_state();
        let want = l.forward(&h.rows(&[], &toks), &stream, &mut state).output;

        for a in 1..n_tok - 1 {
            for b in a + 1..n_tok {
                let bounds = [0, a, b, n_tok];
                let mut chunk_state = l.zero_conv_state();
                let mut hist: Vec<u32> = Vec::new();
                let mut got: Vec<f32> = Vec::new();
                for w in bounds.windows(2) {
                    let (lo, hi) = (w[0], w[1]);
                    got.extend(
                        l.forward(
                            &h.rows(&hist, &toks[lo..hi]),
                            &stream[lo * width..hi * width],
                            &mut chunk_state,
                        )
                        .output,
                    );
                    hist = h.next_history(&hist, &toks[lo..hi]);
                }
                assert_eq!(got, want, "splits at {a} and {b}");
            }
        }
    }
}
