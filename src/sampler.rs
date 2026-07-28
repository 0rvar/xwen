use anyhow::{Result, anyhow, bail, ensure};
use candle_core::{DType, Device, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};

/// Sampling per generation_config.json defaults: temp 1.0, top_k 20, top_p 0.95.
/// Wraps candle's LogitsProcessor (TopKThenTopP) + dual-EOG stop detection.
pub struct Sampler {
    processor: LogitsProcessor,
    eog: Vec<u32>,
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
/// The mask runs before candle's top-k/top-p filtering (it edits the logits
/// the processor sees), so a masked draw is true mask-first constrained
/// sampling: the truncation set is chosen among legal tokens only, and a
/// seeded run reproduces exactly.
#[derive(Default)]
pub struct SampleControl<'a> {
    /// Allow-bitmask over the vocabulary: bit `t` (word `t / 32`, bit
    /// `t % 32`) set means token `t` may be drawn; every clear bit goes to
    /// -inf. `None` allows everything. The mask must cover the whole
    /// vocabulary — a short mask is an error, not an implicit ban.
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
    /// True when nothing would change, so the draw can skip the CPU copy and go
    /// straight through the processor (the default generation path).
    fn is_noop(&self) -> bool {
        self.allowed.is_none()
            && self.banned.is_empty()
            && self.bias.is_empty()
            && self.pull.is_none()
            && self.force.is_none()
    }
}

impl Sampler {
    pub fn new(opts: SamplerOptions, eog_tokens: Vec<u32>) -> Self {
        // A zero (or negative) temperature, or a top-k that keeps at most one
        // candidate, collapses to greedy decoding; otherwise apply top-k then
        // top-p filtering at the configured temperature.
        let sampling = if opts.temperature <= 0.0 || opts.top_k <= 1 {
            Sampling::ArgMax
        } else {
            Sampling::TopKThenTopP {
                k: opts.top_k,
                p: opts.top_p,
                temperature: opts.temperature,
            }
        };
        Self {
            processor: LogitsProcessor::from_sampling(opts.seed, sampling),
            eog: eog_tokens,
        }
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
    /// the logits. Every path ends in exactly one `processor.sample` call — the
    /// invariant the speculative decode loop depends on, since its output only
    /// matches plain decode's while the RNG advances once per committed token.
    pub fn sample_controlled(&mut self, logits: &Tensor, ctl: &SampleControl) -> Result<u32> {
        let logits = logits
            .flatten_all()?
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?;
        if ctl.is_noop() {
            return Ok(self.processor.sample(&logits)?);
        }
        let mut values = logits.to_vec1::<f32>()?;
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
        let masked = Tensor::new(values, &Device::Cpu)?;
        Ok(self.processor.sample(&masked)?)
    }

    pub fn is_eog(&self, token: u32) -> bool {
        self.eog.contains(&token)
    }

    /// The configured end-of-generation token ids.
    pub fn eog_ids(&self) -> &[u32] {
        &self.eog
    }
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
        let a = Sampler::new(opts(), vec![]).sample(&l).unwrap();
        let b = Sampler::new(opts(), vec![]).sample(&l).unwrap();
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
        );
        assert_eq!(s.sample_masked(&l, &[1]).unwrap(), 2);
        // Out-of-range ids are ignored rather than erroring.
        assert_eq!(s.sample_masked(&l, &[1, 999]).unwrap(), 2);
    }

    /// A stochastic sampler (top-k 20, temp 1.0) at a fixed seed.
    fn stochastic(seed: u64) -> Sampler {
        Sampler::new(
            SamplerOptions {
                temperature: 1.0,
                top_k: 20,
                top_p: 1.0,
                seed,
            },
            vec![],
        )
    }

    fn greedy(seed: u64) -> Sampler {
        Sampler::new(
            SamplerOptions {
                temperature: 0.0,
                top_k: 20,
                top_p: 1.0,
                seed,
            },
            vec![],
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
        assert_eq!(greedy(7).sample_controlled(&l, &ctl).unwrap(), 3);
        assert_eq!(stochastic(7).sample_controlled(&l, &ctl).unwrap(), 3);
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
        assert_eq!(stochastic(7).sample_controlled(&l, &ctl).unwrap(), 3);
    }

    // A forced draw consumes exactly one RNG step, like any other draw: after
    // one forced draw the sampler's subsequent draws match a control sampler
    // that took an ordinary draw first. This is what keeps injected tokens from
    // desynchronizing speculative decode from plain decode.
    #[test]
    fn force_consumes_exactly_one_rng_draw() {
        let l = logits(&[0.5, 1.5, 0.2, 2.5, 1.0]);
        let mut forced = stochastic(1234);
        let mut control = stochastic(1234);
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
        assert!(greedy(7).sample_controlled(&l, &ctl).is_err());
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
        assert!(greedy(7).sample_controlled(&l, &ctl).is_err());
        assert!(stochastic(7).sample_controlled(&l, &ctl).is_err());
        // A force still wins: it is the caller's explicit choice, and it is what
        // lets an injected token through a ban that would otherwise cover it.
        let ctl = SampleControl {
            banned: &[0, 1, 2],
            force: Some(1),
            ..SampleControl::default()
        };
        assert_eq!(greedy(7).sample_controlled(&l, &ctl).unwrap(), 1);
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
        assert_eq!(greedy(7).sample_controlled(&l, &ctl).unwrap(), 2);
        // Not quite enough to overtake: 5.0 - 0.5 still leads 4.0.
        let ctl = SampleControl {
            bias: &[(1, -0.5)],
            ..SampleControl::default()
        };
        assert_eq!(greedy(7).sample_controlled(&l, &ctl).unwrap(), 1);
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
        assert_eq!(greedy(7).sample_controlled(&l, &ctl).unwrap(), 0);
        // 0 + 1.0 * (10 - 0) = 10, a tie that argmax resolves to the first index.
        let ctl = SampleControl {
            pull: Some((1, 1.0)),
            ..SampleControl::default()
        };
        assert_eq!(greedy(7).sample_controlled(&l, &ctl).unwrap(), 0);
        // Bias runs before the pull, so it lowers the bar the pull measures
        // against: max becomes 4, and the target is lifted onto it.
        let ctl = SampleControl {
            bias: &[(0, -6.0)],
            pull: Some((1, 1.0)),
            ..SampleControl::default()
        };
        assert_eq!(greedy(7).sample_controlled(&l, &ctl).unwrap(), 0);
    }

    // Pulling the token that already holds the maximum is a no-op, whatever α
    // is: the lift is `max(0, max - logit)`, which is zero there.
    #[test]
    fn pull_is_zero_when_the_target_is_already_the_maximum() {
        let l = logits(&[1.0, 7.0, 2.0]);
        let mut pulled = greedy(7);
        let mut plain = greedy(7);
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
        assert_eq!(greedy(7).sample_controlled(&l, &ctl).unwrap(), 2);
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
        assert_eq!(greedy(7).sample_controlled(&l, &ctl).unwrap(), 2);
        // The stochastic path can only ever draw an allowed id.
        let mut s = stochastic(2024);
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
        assert!(greedy(7).sample_controlled(&l, &ctl).is_err());
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
        assert!(greedy(7).sample_controlled(&l, &ctl).is_err());
    }

    #[test]
    fn is_eog_matches_configured_tokens() {
        let s = Sampler::new(SamplerOptions::default(), vec![2, 24]);
        assert!(s.is_eog(2));
        assert!(s.is_eog(24));
        assert!(!s.is_eog(23));
    }
}
