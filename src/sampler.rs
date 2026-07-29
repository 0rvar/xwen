use anyhow::{Result, anyhow, bail, ensure};
use candle_core::{DType, Device, Tensor};
use rand::SeedableRng;
use rand::distr::Distribution;
use rand::distr::weighted::WeightedIndex;
use rand::rngs::StdRng;

/// Sampling per generation_config.json defaults: temp 1.0, top_k 20, top_p 0.95.
/// Top-k then top-p, plus dual-EOG stop detection.
///
/// The draw is deliberately structured as one pass over the vocabulary and then
/// `O(top_k)` work: the vocabulary is ~248k wide and the whole tail sits inside
/// a ~17 ms decode budget, so every extra full-width pass is a measurable slice
/// of tok/s. The vocabulary-wide softmax runs on whatever device holds the
/// logits (a single Metal kernel) rather than as a CPU exp loop, and the
/// candidate set comes from a streaming top-k over the probabilities instead of
/// a `select_nth` over every index.
///
/// A NaN anywhere in the logit row is an error on every path — see [`argmax`]
/// and [`top_k_desc`].
pub struct Sampler {
    strategy: Strategy,
    rng: StdRng,
    eog: Vec<u32>,
    /// The number of leading ids that carry text. The output layer is wider
    /// than the tokenizer: the rows past this bound are padding, decode to
    /// nothing, and are cut off the logit row before anything else looks at it.
    encodable: usize,
}

enum Strategy {
    ArgMax,
    TopKThenTopP { k: usize, p: f32, temperature: f64 },
}

pub struct SamplerOptions {
    pub temperature: f64,
    pub top_k: usize,
    pub top_p: f64,
    pub seed: u64,
}

impl Default for SamplerOptions {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 20,
            top_p: 0.95,
            seed: 42,
        }
    }
}

/// Post-logit adjustments applied to a single draw, all on the CPU copy of the
/// logits and in this order: `allowed` -> `banned` -> `bias` -> `pull` ->
/// `force`.
///
/// The pieces compose the way their names suggest. The allow-mask and the bans
/// are both absolute (-inf), so no bias or pull can revive an excluded id;
/// `force` outranks everything, including a ban, because it exists to inject a
/// token sequence the caller has already decided on.
///
/// The mask runs before the softmax and the top-k/top-p filtering, so a masked
/// draw is true mask-first constrained sampling: the truncation set is chosen
/// among legal tokens only, and a seeded run reproduces exactly.
#[derive(Default)]
pub struct SampleControl<'a> {
    /// Allow-bitmask over the vocabulary: bit `t` (word `t / 32`, bit
    /// `t % 32`) set means token `t` may be drawn; every clear bit goes to
    /// -inf. `None` allows everything. The mask must cover every encodable id
    /// — a short mask is an error, not an implicit ban. Covering more than
    /// that is fine: a mask sized to the padded logit width spans the ids the
    /// sampler draws from and then some.
    pub allowed: Option<&'a [u32]>,
    /// Ids excluded from the draw (logit set to -inf). Ids past the vocab are
    /// skipped rather than erroring.
    pub banned: &'a [u32],
    /// Additive logit deltas, applied after the bans.
    pub bias: &'a [(u32, f32)],
    /// `(id, α)`: lift `id` toward the current maximum by
    /// `α · max(0, max_logit − logit[id])`. Self-limiting — at `α = 1` the id
    /// lands exactly on the max and never above it — so a pull can make a token
    /// competitive without overriding a decisively different prediction. Skipped
    /// when the target is banned (its logit is no longer finite).
    pub pull: Option<(u32, f32)>,
    /// Collapse the draw onto this id: every other logit goes to -inf. The draw
    /// still runs through the sampler, consuming the RNG exactly once, so a
    /// forced token costs the same RNG advance a sampled one does.
    pub force: Option<u32>,
}

impl SampleControl<'_> {
    /// True when nothing would change, so the draw can softmax on the device
    /// and skip the CPU adjustment pass entirely (the default generation path).
    fn is_noop(&self) -> bool {
        self.allowed.is_none()
            && self.banned.is_empty()
            && self.bias.is_empty()
            && self.pull.is_none()
            && self.force.is_none()
    }
}

impl Sampler {
    /// `encodable` is the number of ids that have text — the tokenizer's
    /// vocabulary size (`LagunaTokenizer::vocab_size`), not the model's logit
    /// width. The two differ: the checkpoint pads its output layer past the
    /// last real token, and a padded row that wins a slot would put an id the
    /// tokenizer cannot decode into the generated text. Everything at or above
    /// the bound is dropped before the draw.
    pub fn new(opts: SamplerOptions, eog_tokens: Vec<u32>, encodable: usize) -> Self {
        // A zero (or negative) temperature, or a top-k that keeps at most one
        // candidate, collapses to greedy decoding; otherwise apply top-k then
        // top-p filtering at the configured temperature.
        let strategy = if opts.temperature <= 0.0 || opts.top_k <= 1 {
            Strategy::ArgMax
        } else {
            Strategy::TopKThenTopP {
                k: opts.top_k,
                p: opts.top_p as f32,
                temperature: opts.temperature,
            }
        };
        Self {
            strategy,
            rng: StdRng::seed_from_u64(opts.seed),
            eog: eog_tokens,
            encodable,
        }
    }

    /// The encodable bound this sampler draws within (see [`Sampler::new`]).
    pub fn encodable(&self) -> usize {
        self.encodable
    }

    /// logits: [vocab] f32 on any device; reads back and samples on CPU.
    pub fn sample(&mut self, logits: &Tensor) -> Result<u32> {
        self.sample_masked(logits, &[])
    }

    /// Like `sample`, but the `banned` token ids are excluded from the draw
    /// (their logits are forced to -inf before filtering). Used to hold the
    /// model inside a `<think>` block by banning `</think>` and the EOG ids.
    pub fn sample_masked(&mut self, logits: &Tensor, banned: &[u32]) -> Result<u32> {
        self.sample_controlled(
            logits,
            &SampleControl {
                banned,
                ..SampleControl::default()
            },
        )
    }

    /// Draw one token with the adjustments in `ctl` applied to the CPU copy of
    /// the logits. Every path ends in exactly one weighted draw — the invariant
    /// the speculative decode loop depends on, since its output only matches
    /// plain decode's while the RNG advances once per committed token. (Greedy
    /// decoding never touches the RNG, in either loop.)
    ///
    /// The logits cross the bus exactly once per call: an unadjusted stochastic
    /// draw softmaxes on the device and reads back probabilities, an adjusted
    /// one reads back raw logits and softmaxes on the CPU after applying the
    /// controls. Neither reads twice.
    ///
    /// The row is cut to the encodable ids first, so the padding rows of the
    /// output layer take part in nothing: not the softmax denominator, not the
    /// argmax, not the candidate set. Every id this returns has text. The two
    /// softmax backends therefore see the same values and differ only in
    /// floating-point association order.
    pub fn sample_controlled(&mut self, logits: &Tensor, ctl: &SampleControl) -> Result<u32> {
        let logits = logits.flatten_all()?;
        let width = logits.dim(0)?;
        // A row narrower than the bound means the sampler and the model
        // disagree about the vocabulary. Cutting to the shorter of the two
        // would hide that, so say it instead.
        ensure!(
            width >= self.encodable,
            "logit row is {width} wide but the sampler draws over {} encodable ids",
            self.encodable
        );
        let logits = logits.narrow(0, 0, self.encodable)?.to_dtype(DType::F32)?;
        let (k, top_p, temperature) = match self.strategy {
            Strategy::ArgMax => {
                let mut values = read_back(&logits)?;
                if !ctl.is_noop() {
                    apply_control(&mut values, ctl)?;
                }
                return argmax(&values);
            }
            Strategy::TopKThenTopP { k, p, temperature } => (k, p, temperature),
        };

        // Probabilities over the whole encodable vocabulary. The top-p cut
        // renormalizes over the top-k survivors, so only their ratios reach the
        // outcome and the shared denominator cancels — see `truncate_top_p`.
        let probs = if ctl.is_noop() {
            // Multiplying by exactly 1.0 is the identity in f32, so the default
            // temperature skips the scaling pass outright.
            let scaled = if temperature == 1.0 {
                logits
            } else {
                (logits / temperature)?
            };
            read_back(&candle_nn::ops::softmax_last_dim(&scaled)?)?
        } else {
            let mut values = read_back(&logits)?;
            apply_control(&mut values, ctl)?;
            softmax_in_place(&mut values, temperature);
            values
        };
        self.draw(&probs, k, top_p)
    }

    /// Draw from the candidate set, consuming exactly one RNG step.
    fn draw(&mut self, probs: &[f32], k: usize, top_p: f32) -> Result<u32> {
        let candidates = candidate_set(probs, k, top_p)?;
        let weights: Vec<f32> = candidates.iter().map(|c| c.0).collect();
        let distr = WeightedIndex::new(&weights).map_err(|e| {
            anyhow!("no candidate carries any probability mass after top-k/top-p filtering: {e}")
        })?;
        let picked = distr.sample(&mut self.rng);
        Ok(candidates[picked].1)
    }

    pub fn is_eog(&self, token: u32) -> bool {
        self.eog.contains(&token)
    }

    /// The configured end-of-generation token ids.
    pub fn eog_ids(&self) -> &[u32] {
        &self.eog
    }
}

/// The single GPU->CPU crossing a draw gets.
fn read_back(t: &Tensor) -> Result<Vec<f32>> {
    Ok(t.to_device(&Device::Cpu)?.to_vec1::<f32>()?)
}

/// The highest-scoring id, ties going to the lowest id.
///
/// A NaN in the row is an error, not a skipped entry. NaN loses every ordered
/// comparison, so a silent skip would hand back the best of whatever survived
/// and report a corrupt forward as an ordinary token — and greedy decoding is
/// what the parity gate runs, so it is exactly where a corrupt forward has to
/// be loud. A -inf is a different thing entirely: that is how the controls
/// exclude an id, and it stays skippable.
fn argmax(values: &[f32]) -> Result<u32> {
    let mut best = f32::NEG_INFINITY;
    let mut arg = None;
    let mut nans = 0usize;
    for (id, &v) in values.iter().enumerate() {
        if v.is_nan() {
            nans += 1;
        } else if v > best {
            best = v;
            arg = Some(id as u32);
        }
    }
    ensure!(
        nans == 0,
        "{nans} of {} logits are NaN: the forward pass produced a corrupt row",
        values.len()
    );
    arg.ok_or_else(|| anyhow!("no logit is greater than -inf: there is no token to draw"))
}

/// `softmax(values / temperature)`, in place.
fn softmax_in_place(values: &mut [f32], temperature: f64) {
    if temperature != 1.0 {
        let inv = (1.0 / temperature) as f32;
        for v in values.iter_mut() {
            *v *= inv;
        }
    }
    // Shifting by the maximum keeps the exponentials in range; it cancels out
    // of the ratio, so it does not change the distribution.
    let max = values
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |a, b| if b > a { b } else { a });
    let mut sum = 0f32;
    for v in values.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in values.iter_mut() {
        *v /= sum;
    }
}

/// The tokens a draw may return, as `(weight, id)` sorted by descending
/// weight. `probs` is the softmax over the whole encodable vocabulary; a zeroed
/// weight is a candidate the top-p cut removed.
///
/// Order of operations: vocabulary-wide softmax, then top-k, then top-p over
/// the survivors renormalized to sum to one — `top_p` is a threshold on mass
/// relative to the top-k set, which is llama.cpp's convention and HF's. The
/// surviving weights come back renormalized for the same reason llama.cpp's
/// `cur_p->p` is: they are what the cut was measured against. The draw
/// normalizes anyway, so the scale reaches nothing but the threshold.
fn candidate_set(probs: &[f32], k: usize, top_p: f32) -> Result<Vec<(f32, u32)>> {
    let mut candidates = if k >= probs.len() {
        // Nothing for top-k to remove: the whole vocabulary is the candidate
        // set the top-p cut then applies to.
        ensure!(
            !probs.iter().any(|p| p.is_nan()),
            "the probability row holds a NaN: the forward pass produced a corrupt row"
        );
        let mut all: Vec<(f32, u32)> = probs.iter().copied().zip(0u32..).collect();
        all.sort_by(|a, b| b.0.total_cmp(&a.0));
        all
    } else {
        top_k_desc(probs, k)?
    };
    truncate_top_p(&mut candidates, top_p);
    Ok(candidates)
}

/// The `k` largest entries of `probs` as `(probability, id)`, sorted
/// descending, ties going to the lower id.
///
/// One pass, one comparison per entry against the current k-th best; the
/// insertion only runs when an entry beats it, which for a vocabulary this wide
/// happens on the order of `k · ln(n / k)` times. `k < probs.len()` is the
/// caller's business — the whole-vocabulary case is handled separately, because
/// its truncation rule differs.
///
/// The strict `>` against the floor is what gives the tie-break its direction:
/// an entry that merely equals the k-th best does not displace it, so among
/// equal probabilities the ids reached first — the lower ones — are the ones
/// kept. A NaN is an error here for the same reason it is in [`argmax`], and it
/// costs nothing to catch: NaN fails every ordered comparison, so testing the
/// negation of `<=` routes it into the same rarely-taken branch as a genuine
/// improvement, leaving one comparison per entry.
fn top_k_desc(probs: &[f32], k: usize) -> Result<Vec<(f32, u32)>> {
    debug_assert!(k > 0 && k < probs.len());
    // Checked up front so the floor below is a real number; a NaN floor would
    // send every later entry down the insertion branch.
    ensure!(
        !probs[..k].iter().any(|p| p.is_nan()),
        "the probability row holds a NaN: the forward pass produced a corrupt row"
    );
    let mut best: Vec<(f32, u32)> = probs[..k].iter().copied().zip(0u32..).collect();
    best.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut floor = best[k - 1].0;
    for (offset, &p) in probs[k..].iter().enumerate() {
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        if !(p <= floor) {
            ensure!(
                !p.is_nan(),
                "the probability row holds a NaN: the forward pass produced a corrupt row"
            );
            let mut j = k - 1;
            while j > 0 && best[j - 1].0 < p {
                best[j] = best[j - 1];
                j -= 1;
            }
            best[j] = (p, (offset + k) as u32);
            floor = best[k - 1].0;
        }
    }
    Ok(best)
}

/// Renormalize `candidates` to sum to one, then zero everything past the
/// shortest prefix whose cumulative mass reaches `top_p`. `candidates` must be
/// sorted descending.
///
/// This is `llama_sampler_top_p_apply`: `p >= 1.0` is a no-op sampler,
/// otherwise the probabilities are recomputed over the surviving set (llama.cpp
/// re-softmaxes the candidates' logits; renormalizing their probabilities is
/// the same quotient) and the walk keeps the token that *crosses* the
/// threshold, comparing `cum_sum >= top_p` — so the kept mass is at least
/// `top_p`, never just short of it. At least one candidate always survives:
/// the first iterate can only set the cut at index 1 or later.
///
/// llama.cpp's `min_keep` floor is not carried: its default is 0 (disabled),
/// and the loop's at-least-one guarantee is the only floor that default
/// produces.
fn truncate_top_p(candidates: &mut [(f32, u32)], top_p: f32) {
    // Matching llama.cpp's early return matters at exactly 1.0: renormalized
    // weights sum to one only to within rounding, so a cumulative walk could
    // otherwise drop the last candidate over an ulp.
    if top_p >= 1.0 {
        return;
    }
    // A softmax always leaves its maximum positive, so this only guards the
    // degenerate rows that would fail the draw regardless.
    let sum: f32 = candidates.iter().map(|c| c.0).sum();
    if sum <= 0.0 {
        return;
    }
    for c in candidates.iter_mut() {
        c.0 /= sum;
    }
    let mut cumsum = 0f32;
    let mut keep = candidates.len();
    for (i, c) in candidates.iter().enumerate() {
        cumsum += c.0;
        if cumsum >= top_p {
            keep = i + 1;
            break;
        }
    }
    for c in &mut candidates[keep..] {
        c.0 = 0.0;
    }
}

/// Apply `ctl`'s adjustments to the CPU copy of the logits, in the documented
/// order: `allowed` -> `banned` -> `bias` -> `pull` -> `force`.
fn apply_control(values: &mut [f32], ctl: &SampleControl) -> Result<()> {
    if let Some(words) = ctl.allowed {
        // A mask narrower than the vocabulary would silently ban the tail —
        // ids the caller never considered — so the widths must agree.
        ensure!(
            words.len() * 32 >= values.len(),
            "allow-mask covers {} ids but the vocabulary holds {}",
            words.len() * 32,
            values.len()
        );
        for (id, v) in values.iter_mut().enumerate() {
            if words[id / 32] & (1 << (id % 32)) == 0 {
                *v = f32::NEG_INFINITY;
            }
        }
    }
    for &id in ctl.banned {
        if let Some(v) = values.get_mut(id as usize) {
            *v = f32::NEG_INFINITY;
        }
    }
    for &(id, delta) in ctl.bias {
        if let Some(v) = values.get_mut(id as usize) {
            *v += delta;
        }
    }
    if let Some((id, alpha)) = ctl.pull {
        // A banned (or otherwise non-finite) target is left alone: pulling it
        // would evaluate -inf + inf = NaN and poison the whole distribution.
        if values.get(id as usize).is_some_and(|v| v.is_finite()) {
            let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let v = &mut values[id as usize];
            *v += alpha * (max - *v).max(0.0);
        }
    }
    if let Some(id) = ctl.force {
        // An out-of-range force is a caller bug, and a silent one: the draw
        // would return some other token while the caller counts it as the
        // forced one. Say so instead.
        let target = *values.get(id as usize).ok_or_else(|| {
            anyhow!(
                "forced token id {id} is outside the {}-entry vocabulary",
                values.len()
            )
        })?;
        values.fill(f32::NEG_INFINITY);
        // Force beats ban: if the caller banned the id it is asking for, the
        // -inf just written would leave nothing to draw from, so the target
        // gets a finite logit back.
        values[id as usize] = if target.is_finite() { target } else { 0.0 };
    } else if !values.iter().any(|v| v.is_finite()) {
        // Nothing survived the controls. Left alone this is silent: greedy
        // argmax over all -inf returns index 0, and the stochastic path
        // softmaxes to NaN. Neither is a token anyone chose.
        bail!("every logit was excluded: the allow-mask and ban list leave no token to draw");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn logits(values: &[f32]) -> Tensor {
        Tensor::new(values, &Device::Cpu).unwrap()
    }

    // A fixed seed and identical logits must produce the same token every time,
    // so generation is reproducible run-to-run.
    #[test]
    fn seeded_sampling_is_deterministic() {
        let opts = || SamplerOptions {
            temperature: 1.0,
            top_k: 20,
            top_p: 1.0,
            seed: 1234,
        };
        let l = logits(&[0.5, 1.5, 0.2, 2.5, 1.0]);
        let a = Sampler::new(opts(), vec![], 5).sample(&l).unwrap();
        let b = Sampler::new(opts(), vec![], 5).sample(&l).unwrap();
        assert_eq!(a, b);
    }

    // Temperature 0 is greedy: the highest-logit id is always chosen regardless
    // of seed.
    #[test]
    fn temperature_zero_is_argmax() {
        let l = logits(&[0.1, 0.2, 5.0, 0.3]);
        for seed in [0u64, 1, 99] {
            let mut s = Sampler::new(
                SamplerOptions {
                    temperature: 0.0,
                    top_k: 20,
                    top_p: 1.0,
                    seed,
                },
                vec![],
                4,
            );
            assert_eq!(s.sample(&l).unwrap(), 2);
        }
    }

    // top_k of 0 or 1 also collapses to greedy.
    #[test]
    fn top_k_one_is_argmax() {
        let l = logits(&[0.1, 4.0, 0.2, 0.3]);
        let mut s = Sampler::new(
            SamplerOptions {
                temperature: 1.0,
                top_k: 1,
                top_p: 1.0,
                seed: 7,
            },
            vec![],
            4,
        );
        assert_eq!(s.sample(&l).unwrap(), 1);
    }

    // With top_k = 2, only the two highest-logit ids may ever be drawn, no matter
    // how the mass is split across the rest of the vocabulary.
    #[test]
    fn top_k_restricts_to_top_two() {
        // Highest two logits are at indices 0 (3.0) and 2 (2.0).
        let l = logits(&[3.0, 0.1, 2.0, 0.2]);
        let mut s = Sampler::new(
            SamplerOptions {
                temperature: 1.0,
                top_k: 2,
                top_p: 1.0,
                seed: 2024,
            },
            vec![],
            4,
        );
        for _ in 0..64 {
            let id = s.sample(&l).unwrap();
            assert!(
                id == 0 || id == 2,
                "drew id {id}, outside the top-2 set {{0, 2}}"
            );
        }
    }

    // A banned id is never drawn, even when it holds by far the highest logit;
    // the draw falls to the best unbanned candidate under greedy settings.
    #[test]
    fn masked_sampling_excludes_banned_ids() {
        let l = logits(&[0.1, 9.0, 3.0, 0.2]);
        let mut s = Sampler::new(
            SamplerOptions {
                temperature: 0.0,
                top_k: 20,
                top_p: 1.0,
                seed: 7,
            },
            vec![],
            4,
        );
        assert_eq!(s.sample_masked(&l, &[1]).unwrap(), 2);
        // Out-of-range ids are ignored rather than erroring.
        assert_eq!(s.sample_masked(&l, &[1, 999]).unwrap(), 2);
    }

    /// A stochastic sampler (top-k 20, temp 1.0) at a fixed seed, drawing over
    /// `n` encodable ids.
    fn stochastic(seed: u64, n: usize) -> Sampler {
        Sampler::new(
            SamplerOptions {
                temperature: 1.0,
                top_k: 20,
                top_p: 1.0,
                seed,
            },
            vec![],
            n,
        )
    }

    fn greedy(seed: u64, n: usize) -> Sampler {
        Sampler::new(
            SamplerOptions {
                temperature: 0.0,
                top_k: 20,
                top_p: 1.0,
                seed,
            },
            vec![],
            n,
        )
    }

    // A forced draw returns its target whatever the logits say, under both the
    // greedy and the stochastic sampling modes.
    #[test]
    fn force_yields_the_target() {
        let l = logits(&[0.1, 9.0, 3.0, 0.2]);
        let ctl = SampleControl {
            force: Some(3),
            ..SampleControl::default()
        };
        assert_eq!(greedy(7, 4).sample_controlled(&l, &ctl).unwrap(), 3);
        assert_eq!(stochastic(7, 4).sample_controlled(&l, &ctl).unwrap(), 3);
    }

    // Force overrides a ban on the same id: the caller injecting a token
    // sequence has already decided, and a ban that emptied the distribution
    // would leave nothing to draw.
    #[test]
    fn force_beats_a_ban_on_the_same_id() {
        let l = logits(&[0.1, 9.0, 3.0, 0.2]);
        let ctl = SampleControl {
            banned: &[3],
            force: Some(3),
            ..SampleControl::default()
        };
        assert_eq!(stochastic(7, 4).sample_controlled(&l, &ctl).unwrap(), 3);
    }

    // A forced draw consumes exactly one RNG step, like any other draw: after
    // one forced draw the sampler's subsequent draws match a control sampler
    // that took an ordinary draw first. This is what keeps injected tokens from
    // desynchronizing speculative decode from plain decode.
    #[test]
    fn force_consumes_exactly_one_rng_draw() {
        let l = logits(&[0.5, 1.5, 0.2, 2.5, 1.0]);
        let mut forced = stochastic(1234, 5);
        let mut control = stochastic(1234, 5);
        forced
            .sample_controlled(
                &l,
                &SampleControl {
                    force: Some(2),
                    ..SampleControl::default()
                },
            )
            .unwrap();
        control.sample(&l).unwrap();
        for step in 0..16 {
            assert_eq!(
                forced.sample(&l).unwrap(),
                control.sample(&l).unwrap(),
                "RNG streams diverged {step} draws after the forced draw",
            );
        }
    }

    // A force pointing past the end of the vocabulary is a caller bug that would
    // otherwise return some other token while the caller books it as forced.
    #[test]
    fn out_of_range_force_errors() {
        let l = logits(&[0.1, 9.0, 3.0]);
        let ctl = SampleControl {
            force: Some(99),
            ..SampleControl::default()
        };
        assert!(greedy(7, 3).sample_controlled(&l, &ctl).is_err());
    }

    // Banning every candidate leaves nothing to draw. Unreported it is silent
    // and wrong in both modes: greedy argmax over all -inf returns index 0, and
    // the stochastic path softmaxes to NaN.
    #[test]
    fn fully_banned_control_errors() {
        let l = logits(&[0.1, 9.0, 3.0]);
        let ctl = SampleControl {
            banned: &[0, 1, 2],
            ..SampleControl::default()
        };
        assert!(greedy(7, 3).sample_controlled(&l, &ctl).is_err());
        assert!(stochastic(7, 3).sample_controlled(&l, &ctl).is_err());
        // A force still wins: it is the caller's explicit choice, and it is what
        // lets an injected token through a ban that would otherwise cover it.
        let ctl = SampleControl {
            banned: &[0, 1, 2],
            force: Some(1),
            ..SampleControl::default()
        };
        assert_eq!(greedy(7, 3).sample_controlled(&l, &ctl).unwrap(), 1);
    }

    // Bias is additive and applied after the bans; under greedy settings a large
    // enough negative bias moves the winner to the next candidate.
    #[test]
    fn bias_shifts_the_greedy_winner() {
        let l = logits(&[0.0, 5.0, 4.0, 0.0]);
        let ctl = SampleControl {
            bias: &[(1, -2.0)],
            ..SampleControl::default()
        };
        assert_eq!(greedy(7, 4).sample_controlled(&l, &ctl).unwrap(), 2);
        // Not quite enough to overtake: 5.0 - 0.5 still leads 4.0.
        let ctl = SampleControl {
            bias: &[(1, -0.5)],
            ..SampleControl::default()
        };
        assert_eq!(greedy(7, 4).sample_controlled(&l, &ctl).unwrap(), 1);
    }

    // The pull lifts its target a fraction of the way to the current maximum and
    // never past it: at α = 1 the target ties the max (and wins under greedy's
    // first-maximal argmax only if it comes first), at α = 0.5 it lands halfway.
    #[test]
    fn pull_lifts_toward_the_maximum() {
        let l = logits(&[10.0, 0.0]);
        let ctl = SampleControl {
            pull: Some((1, 0.5)),
            ..SampleControl::default()
        };
        // 0 + 0.5 * (10 - 0) = 5, still below 10.
        assert_eq!(greedy(7, 2).sample_controlled(&l, &ctl).unwrap(), 0);
        // 0 + 1.0 * (10 - 0) = 10, a tie that argmax resolves to the first index.
        let ctl = SampleControl {
            pull: Some((1, 1.0)),
            ..SampleControl::default()
        };
        assert_eq!(greedy(7, 2).sample_controlled(&l, &ctl).unwrap(), 0);
        // Bias runs before the pull, so it lowers the bar the pull measures
        // against: max becomes 4, and the target is lifted onto it.
        let ctl = SampleControl {
            bias: &[(0, -6.0)],
            pull: Some((1, 1.0)),
            ..SampleControl::default()
        };
        assert_eq!(greedy(7, 2).sample_controlled(&l, &ctl).unwrap(), 0);
    }

    // Pulling the token that already holds the maximum is a no-op, whatever α
    // is: the lift is `max(0, max - logit)`, which is zero there.
    #[test]
    fn pull_is_zero_when_the_target_is_already_the_maximum() {
        let l = logits(&[1.0, 7.0, 2.0]);
        let mut pulled = greedy(7, 3);
        let mut plain = greedy(7, 3);
        let ctl = SampleControl {
            pull: Some((1, 1.0)),
            ..SampleControl::default()
        };
        assert_eq!(
            pulled.sample_controlled(&l, &ctl).unwrap(),
            plain.sample(&l).unwrap()
        );
    }

    // A banned target cannot be pulled: -inf plus an infinite lift would be NaN
    // and would poison the entire distribution. The ban stands.
    #[test]
    fn pull_leaves_a_banned_target_banned() {
        let l = logits(&[0.1, 9.0, 3.0]);
        let ctl = SampleControl {
            banned: &[1],
            pull: Some((1, 1.0)),
            ..SampleControl::default()
        };
        assert_eq!(greedy(7, 3).sample_controlled(&l, &ctl).unwrap(), 2);
    }

    // The allow-mask excludes every clear bit: under greedy settings the draw
    // falls to the best allowed candidate, not the global maximum.
    #[test]
    fn allow_mask_restricts_the_draw() {
        let l = logits(&[0.1, 9.0, 3.0, 0.2]);
        // Allow ids 0 and 2 only.
        let ctl = SampleControl {
            allowed: Some(&[0b0101]),
            ..SampleControl::default()
        };
        assert_eq!(greedy(7, 4).sample_controlled(&l, &ctl).unwrap(), 2);
        // The stochastic path can only ever draw an allowed id.
        let mut s = stochastic(2024, 4);
        for _ in 0..64 {
            let id = s.sample_controlled(&l, &ctl).unwrap();
            assert!(id == 0 || id == 2, "drew masked-out id {id}");
        }
    }

    // Mask and bans intersect: a ban on the only allowed ids empties the
    // distribution, which is reported rather than silently drawing id 0.
    #[test]
    fn allow_mask_intersecting_bans_to_nothing_errors() {
        let l = logits(&[0.1, 9.0, 3.0, 0.2]);
        let ctl = SampleControl {
            allowed: Some(&[0b0101]),
            banned: &[0, 2],
            ..SampleControl::default()
        };
        assert!(greedy(7, 4).sample_controlled(&l, &ctl).is_err());
    }

    // A mask narrower than the vocabulary is a caller bug: it would silently
    // ban the uncovered tail.
    #[test]
    fn short_allow_mask_errors() {
        let values: Vec<f32> = (0..40).map(|i| i as f32).collect();
        let l = logits(&values);
        let ctl = SampleControl {
            allowed: Some(&[u32::MAX]), // 32 bits for a 40-entry vocab
            ..SampleControl::default()
        };
        assert!(greedy(7, 40).sample_controlled(&l, &ctl).is_err());
    }

    #[test]
    fn is_eog_matches_configured_tokens() {
        let s = Sampler::new(SamplerOptions::default(), vec![2, 24], 32);
        assert!(s.is_eog(2));
        assert!(s.is_eog(24));
        assert!(!s.is_eog(23));
    }

    // --- the truncation conventions and what each oracle is good for ---------
    //
    // Top-p follows llama.cpp: renormalize over the top-k survivors, then keep
    // the shortest prefix whose cumulative mass reaches `top_p`. The oracle for
    // that is `llamacpp_filtered` below.
    //
    // candle's `LogitsProcessor` — which this sampler replaced, and which is
    // still linked — measures the same cut against UNrenormalized mass, so the
    // two conventions genuinely disagree wherever the cut can bite. candle
    // therefore stays an oracle only for the parts that never depended on the
    // convention: top-k selection, and the whole pipeline whenever `top_p` is
    // 1.0 and no cut applies. What that residual agreement claims, precisely:
    //
    // * For inputs with no exact tie at the top-k boundary, the two select the
    //   same candidate set and draw from the same distribution over it.
    // * At an exact boundary tie — more entries share the k-th largest
    //   probability than there are slots left — this sampler keeps the LOWEST
    //   ids, deterministically. candle's `select_nth_unstable_by` leaves which
    //   of the tied entries survives unspecified, so there is nothing to agree
    //   with; determinism is the stronger contract and is pinned on its own in
    //   `top_k_boundary_ties_keep_the_lowest_ids`.
    // * The agreement is up to floating-point rounding of whichever backend ran
    //   the softmax, not bit-for-bit. The production fast path softmaxes on the
    //   device holding the logits; candle softmaxes on the CPU. The denominator
    //   is a scale factor shared by every candidate, and renormalizing over the
    //   survivors divides it back out, so it cancels from the cut as well as
    //   from the draw: the two backends truncate identically.
    // * Candidate ORDER differs and always did: candle's list comes out of the
    //   `select_nth_unstable_by` in unspecified order, this one is sorted by
    //   descending probability, and a weighted draw maps its single uniform
    //   through the cumulative weights. Seeded draws therefore differ
    //   token-for-token from the pre-replacement build.

    /// Deterministic pseudo-random logits, spread over `scale`.
    fn det_logits(n: usize, seed: u64, scale: f32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u = ((s >> 40) as f32) / ((1u64 << 24) as f32);
                (u - 0.5) * 2.0 * scale
            })
            .collect()
    }

    /// A literal transcription of candle `LogitsProcessor::sample_topk_topp`'s
    /// filtering, stopping short of the draw: full softmax at `temperature`,
    /// `select_nth` down to `k`, then the top-p pass unless the top-k mass is
    /// already at or below `top_p`. Returns `(id, weight)` sorted by id.
    fn candle_filtered(logits: &[f32], k: usize, top_p: f32, temperature: f64) -> Vec<(u32, f32)> {
        let scaled = Tensor::new(logits, &Device::Cpu)
            .unwrap()
            .affine(1.0 / temperature, 0.0)
            .unwrap();
        let mut prs: Vec<f32> = candle_nn::ops::softmax_last_dim(&scaled)
            .unwrap()
            .to_vec1()
            .unwrap();

        let topp = |prs: &mut Vec<f32>, top_p: f32| {
            let mut order = (0..prs.len()).collect::<Vec<_>>();
            order.sort_by(|&i, &j| prs[j].total_cmp(&prs[i]));
            let mut cumsum = 0.;
            for i in &order {
                if cumsum >= top_p {
                    prs[*i] = 0.0;
                } else {
                    cumsum += prs[*i];
                }
            }
        };

        let mut out: Vec<(u32, f32)> = if k >= prs.len() {
            topp(&mut prs, top_p);
            prs.iter()
                .copied()
                .zip(0u32..)
                .map(|(p, i)| (i, p))
                .collect()
        } else {
            let mut idx = (0..prs.len()).collect::<Vec<_>>();
            let (top, _, _) = idx.select_nth_unstable_by(k, |&i, &j| prs[j].total_cmp(&prs[i]));
            let ids: Vec<u32> = top.iter().map(|&i| i as u32).collect();
            let mut kept: Vec<f32> = ids.iter().map(|&i| prs[i as usize]).collect();
            let sum_p: f32 = kept.iter().sum();
            if top_p > 0.0 && top_p < sum_p {
                topp(&mut kept, top_p);
            }
            ids.into_iter().zip(kept).collect()
        };
        out.sort_by_key(|&(id, _)| id);
        out
    }

    /// A literal transcription of llama.cpp's `llama_sampler_top_k_impl`
    /// followed by `llama_sampler_top_p_apply`: full softmax at `temperature`,
    /// descending sort, cut to `k`, then — unless `top_p` is 1.0 or more, where
    /// llama.cpp's sampler is a no-op — renormalize the survivors and keep the
    /// shortest prefix whose cumulative mass reaches `top_p`, the crossing
    /// token included. Returns the surviving `(id, weight)` sorted by id.
    ///
    /// `min_keep` is llama.cpp's other knob and is left at its default of 0,
    /// which disables it; the prefix always holds at least one token anyway.
    fn llamacpp_filtered(
        logits: &[f32],
        k: usize,
        top_p: f32,
        temperature: f64,
    ) -> Vec<(u32, f32)> {
        let mut probs = logits.to_vec();
        softmax_in_place(&mut probs, temperature);
        let mut ranked: Vec<(u32, f32)> = (0u32..).zip(probs.iter().copied()).collect();
        // A stable sort on the probability alone leaves equal entries in id
        // order, which is the low-id tie-break both sides resolve toward.
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
        ranked.truncate(k.min(ranked.len()));

        if top_p < 1.0 {
            let sum: f32 = ranked.iter().map(|c| c.1).sum();
            for c in ranked.iter_mut() {
                c.1 /= sum;
            }
            let mut cumsum = 0f32;
            let mut keep = ranked.len();
            for (i, c) in ranked.iter().enumerate() {
                cumsum += c.1;
                if cumsum >= top_p {
                    keep = i + 1;
                    break;
                }
            }
            ranked.truncate(keep);
        }
        ranked.sort_by_key(|c| c.0);
        ranked
    }

    /// This sampler's candidate set in the same `(id, weight)` form, dropping
    /// the candidates top-p zeroed so the two sides describe the same support.
    fn ours_filtered(logits: &[f32], k: usize, top_p: f32, temperature: f64) -> Vec<(u32, f32)> {
        let mut probs = logits.to_vec();
        softmax_in_place(&mut probs, temperature);
        let mut out: Vec<(u32, f32)> = candidate_set(&probs, k, top_p)
            .unwrap()
            .into_iter()
            .map(|(w, id)| (id, w))
            .collect();
        out.sort_by_key(|&(id, _)| id);
        out
    }

    // The candidate set matches llama.cpp's — the same ids survive top-k and
    // top-p, carrying the same renormalized probabilities — across flat,
    // moderately peaked and sharply peaked logits, at cuts that bite hard, cuts
    // that barely bite, and `top_p` 1.0 where llama.cpp's sampler is a no-op.
    //
    // Weights agree to a relative tolerance rather than bit-for-bit: the two
    // softmax denominators are the same sum in a different association order
    // (candle's f32 reduction is a NEON lane tree, this one is sequential, and
    // the Metal kernel the production path uses is a third order).
    #[test]
    fn candidate_set_matches_llamacpp_filtering() {
        for (n, scale) in [(64usize, 1.0f32), (64, 4.0), (2048, 3.0), (2048, 12.0)] {
            for seed in [1u64, 77, 4242] {
                for (k, p, temp) in [
                    (20usize, 0.95f32, 1.0f64),
                    (8, 0.5, 0.7),
                    (40, 1.0, 1.3),
                    (20, 0.2, 1.0),
                ] {
                    let l = det_logits(n, seed, scale);
                    let case =
                        format!("n {n} scale {scale} seed {seed} k {k} top_p {p} temp {temp}");
                    let want = llamacpp_filtered(&l, k, p, temp);
                    let got: Vec<(u32, f32)> = ours_filtered(&l, k, p, temp)
                        .into_iter()
                        .filter(|&(_, w)| w > 0.0)
                        .collect();
                    let ids = |v: &[(u32, f32)]| v.iter().map(|c| c.0).collect::<Vec<_>>();
                    assert_eq!(ids(&want), ids(&got), "{case}");
                    for (&(id, a), &(_, b)) in want.iter().zip(&got) {
                        assert!(
                            (a - b).abs() <= 1e-5 * a.abs(),
                            "{case}: token {id} weighted {b} against llama.cpp's {a}"
                        );
                    }
                }
            }
        }
    }

    // Where no top-p cut applies — `top_p` 1.0, so both samplers are doing pure
    // top-k — the candidate set still matches candle's exactly, ids and
    // weights. This is the part of the pipeline the convention switch left
    // alone, and candle remains a real oracle for it.
    #[test]
    fn candidate_set_matches_candle_when_no_top_p_cut_applies() {
        for (n, scale) in [(64usize, 1.0f32), (64, 4.0), (2048, 3.0), (2048, 12.0)] {
            for seed in [1u64, 77, 4242] {
                for (k, temp) in [(20usize, 1.0f64), (8, 0.7), (40, 1.3)] {
                    let l = det_logits(n, seed, scale);
                    let case = format!("n {n} scale {scale} seed {seed} k {k} temp {temp}");
                    let want: Vec<(u32, f32)> = candle_filtered(&l, k, 1.0, temp)
                        .into_iter()
                        .filter(|&(_, w)| w > 0.0)
                        .collect();
                    let got: Vec<(u32, f32)> = ours_filtered(&l, k, 1.0, temp)
                        .into_iter()
                        .filter(|&(_, w)| w > 0.0)
                        .collect();
                    let ids = |v: &[(u32, f32)]| v.iter().map(|c| c.0).collect::<Vec<_>>();
                    assert_eq!(ids(&want), ids(&got), "{case}");
                    for (&(id, a), &(_, b)) in want.iter().zip(&got) {
                        assert!(
                            (a - b).abs() <= 1e-5 * a.abs(),
                            "{case}: token {id} weighted {b} against candle's {a}"
                        );
                    }
                }
            }
        }
    }

    // The top-p cut measures mass RENORMALIZED over the top-k survivors, which
    // is llama.cpp's convention and HF's: the survivors are rescaled to sum to
    // one and the shortest prefix reaching `top_p` is kept, the token that
    // crosses the threshold included.
    //
    // Both cases below are hand-computed on exact probabilities, and both are
    // ones where renormalizing and measuring absolute mass give different
    // answers — which is the point of pinning the convention at all.
    #[test]
    fn top_p_measures_mass_renormalized_over_the_candidate_set() {
        /// Logits whose softmax is exactly `weights` normalized.
        fn from_weights(weights: &[f32]) -> Vec<f32> {
            weights.iter().map(|w| w.ln()).collect()
        }
        let surviving = |probs: &[f32], k: usize, p: f32| -> Vec<u32> {
            let mut ids: Vec<u32> = candidate_set(probs, k, p)
                .unwrap()
                .into_iter()
                .filter(|c| c.0 > 0.0)
                .map(|c| c.1)
                .collect();
            ids.sort_unstable();
            ids
        };

        // Probabilities .40 .30 .20 .05 .03 .02; top-3 holds .90 of the total.
        // Renormalized the three are .4444 / .3333 / .2222, so the cumulative
        // sum passes 0.75 at the second one and the third is cut. Measuring
        // absolute mass instead, the running total reaches only .70 before the
        // third candidate and it survives.
        let mut probs = from_weights(&[40.0, 30.0, 20.0, 5.0, 3.0, 2.0]);
        softmax_in_place(&mut probs, 1.0);
        assert_eq!(surviving(&probs, 3, 0.75), vec![0, 1]);

        // The same distinction where the absolute convention cannot cut at all:
        // the top-3 of this row hold .76 of the total, short of top_p .90, so
        // an absolute-mass cut is skipped outright and all three survive.
        // Renormalized they are .5263 / .3947 / .0789, the running sum passes
        // .90 at the second, and the third is cut.
        let mut weights = vec![3.0f32; 11];
        weights[0] = 40.0;
        weights[1] = 30.0;
        weights[2] = 6.0;
        let mut probs = from_weights(&weights);
        softmax_in_place(&mut probs, 1.0);
        let top3: f32 = probs[..3].iter().sum();
        assert!((top3 - 0.76).abs() < 1e-5, "top-3 mass {top3}");
        assert_eq!(surviving(&probs, 3, 0.90), vec![0, 1]);

        // `top_p` 1.0 keeps everything the top-k selected: llama.cpp's top-p
        // sampler is a no-op at or above 1.0, and a cumulative walk over
        // weights that sum to one only to within rounding could otherwise drop
        // the last candidate.
        assert_eq!(surviving(&probs, 3, 1.0), vec![0, 1, 2]);
    }

    // `top_p` 0.0 is a live value — the serve layers pass a client's `top_p`
    // through verbatim — and the cumulative walk crosses a zero threshold at
    // the first candidate, so exactly the top-probability token survives, as
    // it does in llama.cpp.
    #[test]
    fn top_p_of_zero_keeps_only_the_top_candidate() {
        let logits = det_logits(64, 90210, 3.0);
        let want = llamacpp_filtered(&logits, 20, 0.0, 1.0);
        assert_eq!(want.len(), 1);

        let mut probs = logits.clone();
        softmax_in_place(&mut probs, 1.0);
        let got: Vec<(u32, f32)> = candidate_set(&probs, 20, 0.0)
            .unwrap()
            .into_iter()
            .filter(|c| c.0 > 0.0)
            .map(|c| (c.1, c.0))
            .collect();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].0, want[0].0);
        assert!((got[0].1 - want[0].1).abs() < 1e-6);
    }

    // The drawn tokens follow llama.cpp's distribution end to end: every token
    // the renormalized cut kept comes up at its own weight, and every token it
    // cut never comes up at all.
    #[test]
    fn draw_frequencies_follow_llamacpp_semantics() {
        const N: usize = 64;
        const DRAWS: usize = 40_000;
        const SEED: u64 = 5150;
        const K: usize = 20;
        const P: f64 = 0.95;
        const TEMP: f64 = 1.0;
        let l = det_logits(N, 31337, 3.0);
        let tensor = Tensor::new(l.as_slice(), &Device::Cpu).unwrap();

        let want = llamacpp_filtered(&l, K, P as f32, TEMP);
        let mass: f32 = want.iter().map(|c| c.1).sum();
        assert!(
            want.len() < K,
            "test needs a cut that bites, {} of {K} survived",
            want.len()
        );

        let mut ours = Sampler::new(
            SamplerOptions {
                temperature: TEMP,
                top_k: K,
                top_p: P,
                seed: SEED,
            },
            vec![],
            N,
        );
        let mut hist = vec![0usize; N];
        for _ in 0..DRAWS {
            hist[ours.sample(&tensor).unwrap() as usize] += 1;
        }

        for id in 0..N as u32 {
            let drawn = hist[id as usize] as f64 / DRAWS as f64;
            match want.iter().find(|c| c.0 == id) {
                Some(&(_, w)) => {
                    let expect = (w / mass) as f64;
                    assert!(
                        (drawn - expect).abs() < 0.01,
                        "token {id} drawn at {drawn}, llama.cpp weights it {expect}"
                    );
                }
                None => assert_eq!(drawn, 0.0, "token {id} was cut but was drawn at {drawn}"),
            }
        }
    }

    // With no top-p cut in play the drawn tokens still follow candle's
    // distribution: over many draws the two samplers agree on which tokens are
    // reachable and on how often each comes up. Frequencies, not sequences —
    // the candidate ordering differs, so the seeded token streams legitimately
    // do not match.
    #[test]
    fn draw_frequencies_match_candle_when_no_top_p_cut_applies() {
        use candle_transformers::generation::{LogitsProcessor, Sampling};

        const N: usize = 64;
        const DRAWS: usize = 40_000;
        const SEED: u64 = 5150;
        const K: usize = 20;
        const P: f64 = 1.0;
        const TEMP: f64 = 1.0;
        let l = det_logits(N, 31337, 3.0);
        let tensor = Tensor::new(l.as_slice(), &Device::Cpu).unwrap();

        let mut ours = Sampler::new(
            SamplerOptions {
                temperature: TEMP,
                top_k: K,
                top_p: P,
                seed: SEED,
            },
            vec![],
            N,
        );
        let mut theirs = LogitsProcessor::from_sampling(
            SEED,
            Sampling::TopKThenTopP {
                k: K,
                p: P,
                temperature: TEMP,
            },
        );

        let mut ours_hist = vec![0usize; N];
        let mut theirs_hist = vec![0usize; N];
        for _ in 0..DRAWS {
            ours_hist[ours.sample(&tensor).unwrap() as usize] += 1;
            theirs_hist[theirs.sample(&tensor).unwrap() as usize] += 1;
        }

        for id in 0..N {
            let a = ours_hist[id] as f64 / DRAWS as f64;
            let b = theirs_hist[id] as f64 / DRAWS as f64;
            assert_eq!(
                a > 0.0,
                b > 0.0,
                "token {id} is reachable in one sampler and not the other \
                 (ours {a}, candle {b})"
            );
            assert!(
                (a - b).abs() < 0.01,
                "token {id} drawn at {a} vs candle's {b} over {DRAWS} draws"
            );
        }
    }

    /// The logit shapes the equivalence matrix sweeps.
    #[derive(Clone, Copy, Debug)]
    enum Shape {
        /// Near-uniform. The top-k set holds a small share of the total mass,
        /// which is where the renormalizing cut and an absolute-mass one differ
        /// most: this one still trims, an absolute-mass one could not.
        Flat,
        /// A handful of ids hold nearly all the mass, so the top-p cut bites
        /// early and most of the top-k set is zeroed.
        Peaked,
        /// More ids tie for the top than fit in `k`. This is the case candle's
        /// `select_nth_unstable_by` leaves unspecified, so the two samplers are
        /// only required to agree on which ids are ELIGIBLE.
        Ties,
    }

    fn shaped(shape: Shape, n: usize, k: usize, seed: u64) -> Vec<f32> {
        let mut l = det_logits(n, seed, 0.5);
        match shape {
            Shape::Flat => {}
            Shape::Peaked => {
                for i in 0..8.min(n) {
                    l[(i * 977) % n] = 18.0 - 2.0 * i as f32;
                }
            }
            Shape::Ties => {
                let step = n / (k + 2);
                if step == 0 {
                    // Fewer ids than the tie needs: tie the whole row.
                    l.fill(6.0);
                } else {
                    for i in 0..k + 2 {
                        l[i * step] = 6.0;
                    }
                }
            }
        }
        l
    }

    /// The ids eligible for the top-k set, ties included: everything whose
    /// probability reaches the k-th largest. Any sampler's top-k survivors are
    /// a subset of this, however it resolves ties.
    fn tie_inclusive_top_k(probs: &[f32], k: usize) -> Vec<u32> {
        if k >= probs.len() {
            return (0..probs.len() as u32).collect();
        }
        let mut sorted = probs.to_vec();
        sorted.sort_by(|a, b| b.total_cmp(a));
        let kth = sorted[k - 1];
        probs
            .iter()
            .enumerate()
            .filter(|&(_, &p)| p >= kth)
            .map(|(i, _)| i as u32)
            .collect()
    }

    // The same sweep run against the llama.cpp truncation oracle and against
    // the REAL `LogitsProcessor` rather than a transcription of it, so a
    // misreading of either source cannot pass: vocabulary widths from a toy row
    // up to the checkpoint's own logit width, every k from "collapses to
    // greedy" through "wider than the row", and top_p from a cut that bites
    // hard to one that cannot bite at all.
    //
    // What each oracle is allowed to claim differs, and the difference is the
    // point. llama.cpp pins the candidate set exactly, at every shape and every
    // p. candle pins the same thing only at top_p 1.0, where no cut applies —
    // below that it measures the cut against unrenormalized mass and keeps
    // candidates this sampler drops. Tied logits pin only eligibility against
    // candle, because which of the tied entries its `select_nth_unstable_by`
    // keeps is unspecified; the deterministic answer this sampler gives is
    // pinned separately in `top_k_boundary_ties_keep_the_lowest_ids`.
    #[test]
    fn draws_match_the_real_logits_processor_across_the_matrix() {
        use crate::tokenizer::LagunaTokenizer;
        use candle_transformers::generation::{LogitsProcessor, Sampling};

        const TEMP: f64 = 1.0;
        const SEED: u64 = 90210;

        // Draws per case, and the frequency tolerance that many draws support.
        // The widest row is sampled only enough to catch an id escaping the
        // candidate set; a unit test cannot afford to estimate 248k-wide
        // frequencies and does not need to.
        let widths = [
            (64usize, 10_000usize),
            (2048, 4_000),
            (LagunaTokenizer::PADDED_VOCAB, 100),
        ];

        for (n, draws) in widths {
            for shape in [Shape::Flat, Shape::Peaked, Shape::Ties] {
                for k in [1usize, 20, 64] {
                    for p in [0.5f64, 0.95, 1.0] {
                        let l = shaped(shape, n, k, SEED);
                        let tensor = Tensor::new(l.as_slice(), &Device::Cpu).unwrap();
                        let case = format!("n {n} {shape:?} k {k} top_p {p}");

                        let mut probs = l.clone();
                        softmax_in_place(&mut probs, TEMP);
                        let ours_set: Vec<u32> = candidate_set(&probs, k, p as f32)
                            .unwrap()
                            .into_iter()
                            .filter(|c| c.0 > 0.0)
                            .map(|c| c.1)
                            .collect();
                        let eligible = tie_inclusive_top_k(&probs, k);

                        // The candidate set is llama.cpp's, exactly, whatever
                        // the shape and whatever the cut.
                        let mut ours_by_id = ours_set.clone();
                        ours_by_id.sort_unstable();
                        let llamacpp_set: Vec<u32> = llamacpp_filtered(&l, k, p as f32, TEMP)
                            .into_iter()
                            .map(|c| c.0)
                            .collect();
                        assert_eq!(ours_by_id, llamacpp_set, "{case}");

                        let mut ours = Sampler::new(
                            SamplerOptions {
                                temperature: TEMP,
                                top_k: k,
                                top_p: p,
                                seed: SEED,
                            },
                            vec![],
                            n,
                        );
                        let mut theirs = LogitsProcessor::from_sampling(
                            SEED,
                            Sampling::TopKThenTopP {
                                k,
                                p,
                                temperature: TEMP,
                            },
                        );

                        let mut ours_hist = std::collections::HashMap::new();
                        let mut theirs_hist = std::collections::HashMap::new();
                        for _ in 0..draws {
                            let a = ours.sample(&tensor).unwrap();
                            let b = theirs.sample(&tensor).unwrap();
                            *ours_hist.entry(a).or_insert(0usize) += 1;
                            *theirs_hist.entry(b).or_insert(0usize) += 1;
                        }

                        for &id in ours_hist.keys() {
                            assert!(
                                ours_set.contains(&id),
                                "{case}: drew {id}, outside its own candidate set"
                            );
                        }
                        match shape {
                            Shape::Ties => {
                                // The top-p cut only ever removes candidates,
                                // so whatever candle's unspecified tie
                                // resolution kept, it still reaches the k-th
                                // largest probability — at every p.
                                for &id in theirs_hist.keys() {
                                    assert!(
                                        eligible.contains(&id),
                                        "{case}: candle drew {id}, which does not reach the \
                                         k-th largest probability"
                                    );
                                }
                            }
                            // Below top_p 1.0 the two conventions part company
                            // and candle keeps candidates this sampler cuts,
                            // so candle is an oracle only where no cut applies.
                            Shape::Flat | Shape::Peaked if p >= 1.0 => {
                                for &id in theirs_hist.keys() {
                                    assert!(
                                        ours_set.contains(&id),
                                        "{case}: candle drew {id}, which this sampler excluded"
                                    );
                                }
                                // Frequencies, where there are enough draws to
                                // mean anything. Three standard errors of a
                                // Bernoulli at this sample size, floored so a
                                // rare candidate is not held to noise.
                                if draws >= 2000 {
                                    let tol = (3.0 / (draws as f64).sqrt()).max(0.02);
                                    for &id in &ours_set {
                                        let a =
                                            *ours_hist.get(&id).unwrap_or(&0) as f64 / draws as f64;
                                        let b = *theirs_hist.get(&id).unwrap_or(&0) as f64
                                            / draws as f64;
                                        assert!(
                                            (a - b).abs() < tol,
                                            "{case}: token {id} drawn at {a} against candle's {b}"
                                        );
                                    }
                                }
                            }
                            Shape::Flat | Shape::Peaked => {}
                        }
                    }
                }
            }
        }
    }

    // The streaming top-k keeps exactly the k largest, and resolves ties toward
    // the lower id, so the candidate list is a pure function of the
    // probabilities rather than of the traversal that built it.
    #[test]
    fn top_k_desc_keeps_the_largest_and_breaks_ties_low() {
        let v = [0.1f32, 0.9, 0.4, 0.9, 0.2, 0.7];
        assert_eq!(
            top_k_desc(&v, 3).unwrap(),
            vec![(0.9, 1), (0.9, 3), (0.7, 5)]
        );
        // Ascending input is the worst case for a streaming selection: every
        // entry displaces the current floor.
        let asc: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        assert_eq!(
            top_k_desc(&asc, 3).unwrap(),
            vec![(999.0, 999), (998.0, 998), (997.0, 997)]
        );
    }

    // More entries tie for the top than there are slots: the k lowest ids are
    // the ones kept, and which ones they are does not depend on how far into
    // the row the tie runs. The selection has to pick some k of the tied
    // entries; picking the lowest ids makes the choice a property of the input
    // rather than of the traversal, which is what lets a seeded run reproduce.
    #[test]
    fn top_k_boundary_ties_keep_the_lowest_ids() {
        for k in [1usize, 3, 8] {
            // k + 1 entries share the maximum, so exactly one has to lose.
            let mut v = vec![0.01f32; 64];
            for slot in v.iter_mut().take(k + 1) {
                *slot = 0.5;
            }
            let got = top_k_desc(&v, k).unwrap();
            let ids: Vec<u32> = got.iter().map(|c| c.1).collect();
            assert_eq!(ids, (0..k as u32).collect::<Vec<_>>(), "k {k}");

            // The same tie, reached late: entries 40..40+k+1 tie for the max
            // while everything before them is smaller.
            let mut v = vec![0.01f32; 64];
            for slot in v.iter_mut().skip(40).take(k + 1) {
                *slot = 0.5;
            }
            let got = top_k_desc(&v, k).unwrap();
            let ids: Vec<u32> = got.iter().map(|c| c.1).collect();
            assert_eq!(ids, (40..40 + k as u32).collect::<Vec<_>>(), "k {k} late");
        }
    }

    // Ids past the encodable bound are output-layer padding with no text behind
    // them. They never win a slot, however large their logit, on either path —
    // the alternative is a token the tokenizer cannot decode landing in the
    // generated string.
    #[test]
    fn padded_ids_are_never_drawn() {
        const ENCODABLE: usize = 5;
        // The padding rows hold the three largest logits in the row.
        let l = logits(&[0.1, 4.0, 0.2, 3.0, 0.3, 9.0, 8.0, 7.0]);
        assert_eq!(greedy(7, ENCODABLE).sample(&l).unwrap(), 1);

        let mut s = stochastic(2024, ENCODABLE);
        for _ in 0..256 {
            let id = s.sample(&l).unwrap();
            assert!((id as usize) < ENCODABLE, "drew padded id {id}");
        }

        // Banning the best encodable id moves the draw to the next one, not
        // out into the padding.
        assert_eq!(
            greedy(7, ENCODABLE).sample_masked(&l, &[1]).unwrap(),
            3,
            "the ban must fall through to the next encodable id"
        );

        // The padding is excluded before the softmax, so it does not dilute the
        // probabilities either: the encodable weights sum to one on their own.
        let mut probs = vec![0.1f32, 4.0, 0.2, 3.0, 0.3];
        softmax_in_place(&mut probs, 1.0);
        let drawn: f32 = candidate_set(&probs, 20, 1.0)
            .unwrap()
            .iter()
            .map(|c| c.0)
            .sum();
        assert!((drawn - 1.0).abs() < 1e-6, "encodable mass {drawn}");
    }

    // A logit row narrower than the encodable bound means the model and the
    // sampler disagree about the vocabulary. Drawing from the shorter of the
    // two would bury that.
    #[test]
    fn logits_narrower_than_the_bound_error() {
        let l = logits(&[0.1, 9.0, 3.0]);
        assert!(greedy(7, 5).sample(&l).is_err());
        assert!(stochastic(7, 5).sample(&l).is_err());
    }

    // A NaN in the row is a corrupt forward pass, and it fails the draw instead
    // of being skipped over: a skipped NaN would return the best of whatever
    // was left and read as an ordinary token. Both greedy — which is what the
    // parity gate runs — and the stochastic path report it.
    #[test]
    fn nan_logits_are_an_error_but_neg_inf_is_not() {
        let l = logits(&[0.1, f32::NAN, 3.0, 0.2]);
        assert!(greedy(7, 4).sample(&l).is_err());
        assert!(stochastic(7, 4).sample(&l).is_err());

        // A NaN deep in a row wide enough for the streaming selection to run is
        // caught just the same, wherever it sits.
        let mut wide = vec![0.5f32; 256];
        wide[0] = 9.0;
        wide[200] = f32::NAN;
        let l = logits(&wide);
        assert!(greedy(7, 256).sample(&l).is_err());
        assert!(stochastic(7, 256).sample(&l).is_err());

        // Including when it is below the running k-th best, where the selection
        // takes no interest in it.
        let mut probs = vec![0.001f32; 256];
        probs[..20].fill(0.04);
        probs[200] = f32::NAN;
        assert!(top_k_desc(&probs, 20).is_err());

        // -inf is not corruption — it is how the controls exclude an id — and
        // stays skippable on both paths.
        let l = logits(&[0.1, f32::NEG_INFINITY, 3.0, 0.2]);
        assert_eq!(greedy(7, 4).sample(&l).unwrap(), 2);
        assert!(stochastic(7, 4).sample(&l).unwrap() != 1);
    }
}
