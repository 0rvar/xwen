// Vendored from candle rev 21cca0b (candle-transformers/src/models/z_image/scheduler.rs,
// PR #3261, SpenserCai), MIT / Apache-2.0. The sigma grid is rewritten to follow
// diffusers; upstream followed the official repo, which is a different grid. Which one
// is which, and why this one, is below.
//! The flow-matching Euler scheduler as Z-Image runs it.
//!
//! **The grid is diffusers', and the official repo's differs.** The two
//! references really do disagree, by a little, and this file follows the one
//! the parity oracle will come from:
//!
//! - diffusers' `ZImagePipeline` hands the scheduler an EXPLICIT sigma grid,
//!   `get_default_z_image_sigmas` = `linspace(1.0, 1/n, n)` — the same as
//!   `1 - k/n` for `k = 0..n-1` — and
//!   `FlowMatchEulerDiscreteScheduler.set_timesteps` applies the STATIC shift
//!   `s * σ / (1 + (s - 1) * σ)` with `s = 3.0` from `scheduler_config.json`
//!   ONCE, then appends a terminal 0. That is what [`Self::set_timesteps`]
//!   below computes.
//! - `Tongyi-MAI/Z-Image`'s own scheduler passes `sigmas=None`, shifts the
//!   full 1..1000 training grid in its constructor, interpolates `n + 1` points
//!   between those ALREADY-SHIFTED extremes and shifts a second time. At 8
//!   steps the two grids agree to a maximum |Δσ| of 5.0e-3, and the largest
//!   single Euler step of the run has `dt` −0.305009 there against −0.300000
//!   here: visibly the same image, not the same numbers.
//!
//! diffusers is the reference here because it is the oracle Stage 3 and
//! Stage 4 will be graded against — it is what `scripts/zimage-ref-dump.py`
//! drives, and it is the pipeline HuggingFace publishes for these weights
//! (decisions.md "The sigma grid follows diffusers, not the official repo").
//! Moving to the official grid means moving the oracle with it.
//!
//! `use_dynamic_shifting` is false in every shipped Z-Image config, so the
//! resolution-dependent `mu` both pipelines still compute is dead code there
//! and is refused rather than implemented here.
//!
//! Two sign conventions travel with this: the timestep the transformer sees is
//! `1 - σ` (0 at pure noise, rising towards 1), and the transformer's output is
//! NEGATED before the Euler step `x += (σ_next - σ) * v`.

use candle_core::{Result, Tensor};

/// `scheduler/scheduler_config.json`.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct SchedulerConfig {
    #[serde(default = "default_num_train_timesteps")]
    pub num_train_timesteps: usize,
    #[serde(default = "default_shift")]
    pub shift: f64,
    #[serde(default)]
    pub use_dynamic_shifting: bool,
}

fn default_num_train_timesteps() -> usize {
    1000
}
fn default_shift() -> f64 {
    3.0
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self::z_image_turbo()
    }
}

impl SchedulerConfig {
    /// The shipped Z-Image-Turbo configuration.
    pub fn z_image_turbo() -> Self {
        Self {
            num_train_timesteps: 1000,
            shift: 3.0,
            use_dynamic_shifting: false,
        }
    }
}

/// Flow-matching Euler scheduler over the Z-Image sigma grid.
#[derive(Debug, Clone)]
pub struct FlowMatchEulerDiscreteScheduler {
    pub config: SchedulerConfig,
    /// One entry per inference step: the shifted sigma times
    /// `num_train_timesteps`.
    pub timesteps: Vec<f64>,
    /// One entry per inference step PLUS the terminal 0.
    pub sigmas: Vec<f64>,
    step_index: usize,
}

impl FlowMatchEulerDiscreteScheduler {
    /// A scheduler with no steps set yet; call [`Self::set_timesteps`].
    pub fn new(config: SchedulerConfig) -> Result<Self> {
        if config.use_dynamic_shifting {
            candle_core::bail!(
                "use_dynamic_shifting is not implemented: every shipped Z-Image scheduler \
                 config has it off"
            );
        }
        Ok(Self {
            config,
            timesteps: Vec::new(),
            sigmas: Vec::new(),
            step_index: 0,
        })
    }

    /// The static shift Z-Image applies to every sigma.
    pub fn shift_sigma(shift: f64, sigma: f64) -> f64 {
        shift * sigma / (1.0 + (shift - 1.0) * sigma)
    }

    /// Lay out `num_inference_steps` steps: raw sigmas `1 - k/n`, shifted,
    /// with a terminal 0 appended, and reset to step 0.
    pub fn set_timesteps(&mut self, num_inference_steps: usize) -> Result<()> {
        if num_inference_steps == 0 {
            candle_core::bail!("num_inference_steps must be at least 1");
        }
        let n = num_inference_steps as f64;
        let mut sigmas: Vec<f64> = (0..num_inference_steps)
            .map(|k| 1.0 - k as f64 / n)
            .map(|sigma| Self::shift_sigma(self.config.shift, sigma))
            .collect();
        self.timesteps = sigmas
            .iter()
            .map(|&s| s * self.config.num_train_timesteps as f64)
            .collect();
        sigmas.push(0.0);
        self.sigmas = sigmas;
        self.step_index = 0;
        Ok(())
    }

    /// The current step's sigma.
    pub fn current_sigma(&self) -> f64 {
        self.sigmas[self.step_index]
    }

    /// The timestep the transformer is fed at the current step: `1 - σ`.
    pub fn current_timestep_normalized(&self) -> f64 {
        let t = self.timesteps.get(self.step_index).copied().unwrap_or(0.0);
        (self.config.num_train_timesteps as f64 - t) / self.config.num_train_timesteps as f64
    }

    /// One Euler step: `sample + (σ_next - σ) * model_output`, where
    /// `model_output` is the transformer's prediction already negated by the
    /// caller. Advances the step index.
    pub fn step(&mut self, model_output: &Tensor, sample: &Tensor) -> Result<Tensor> {
        if self.is_complete() {
            candle_core::bail!("the scheduler has already run every step");
        }
        let sigma = self.sigmas[self.step_index];
        let sigma_next = self.sigmas[self.step_index + 1];
        let dt = sigma_next - sigma;
        let prev_sample = (sample + (model_output * dt)?)?;
        self.step_index += 1;
        Ok(prev_sample)
    }

    pub fn reset(&mut self) {
        self.step_index = 0;
    }

    pub fn num_inference_steps(&self) -> usize {
        self.timesteps.len()
    }

    pub fn step_index(&self) -> usize {
        self.step_index
    }

    pub fn is_complete(&self) -> bool {
        self.step_index >= self.timesteps.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shipped defaults: 8 steps at shift 3.0, pinned to all nine of
    /// DIFFUSERS' sigmas — the shifted grid `3σ / (1 + 2σ)` over
    /// `σ = 1 - k/8`, plus the terminal 0.
    ///
    /// The tolerance is deliberately tighter than the gap to the official
    /// repo's double-shifted grid, so this test is the thing that would go red
    /// if anyone moved the schedule back to it, and the last comparison below
    /// says so in the one place it matters most: the final step, where the two
    /// references are 5.0e-3 apart.
    #[test]
    fn eight_steps_at_shift_three_match_the_diffusers_schedule() {
        let mut s = FlowMatchEulerDiscreteScheduler::new(SchedulerConfig::z_image_turbo()).unwrap();
        s.set_timesteps(8).unwrap();
        let expected_sigmas = [1.0, 0.954545, 0.9, 0.833333, 0.75, 0.642857, 0.5, 0.3, 0.0];
        assert_eq!(s.sigmas.len(), 9);
        for (got, want) in s.sigmas.iter().zip(expected_sigmas) {
            assert!((got - want).abs() < 1e-6, "sigma {got} != {want}");
        }
        // The official repo's grid at the same step is 0.3050089, and that is
        // the value this must NOT be.
        assert!(
            (s.sigmas[7] - 0.3050089).abs() > 1e-4,
            "sigma {} is the official repo's grid, not diffusers'",
            s.sigmas[7]
        );
        assert_eq!(s.timesteps.len(), 8);
        assert!((s.timesteps[1] - 954.545454).abs() < 1e-3);
        // The model's timestep is 1 - sigma.
        assert!((s.current_timestep_normalized() - 0.0).abs() < 1e-9);
        let x = Tensor::zeros(4, candle_core::DType::F32, &candle_core::Device::Cpu).unwrap();
        let v = Tensor::ones(4, candle_core::DType::F32, &candle_core::Device::Cpu).unwrap();
        // dt for the first step is 0.954545 - 1.0.
        let y = s.step(&v, &x).unwrap().to_vec1::<f32>().unwrap();
        assert!((y[0] + 0.045454).abs() < 1e-5, "{y:?}");
        assert!((s.current_timestep_normalized() - 0.045454).abs() < 1e-5);
        for _ in 1..8 {
            let x = s.step(&v, &x).unwrap();
            drop(x);
        }
        assert!(s.is_complete());
        assert!(s.step(&v, &x).is_err());
    }

    #[test]
    fn dynamic_shifting_is_refused() {
        let cfg = SchedulerConfig {
            use_dynamic_shifting: true,
            ..SchedulerConfig::z_image_turbo()
        };
        assert!(FlowMatchEulerDiscreteScheduler::new(cfg).is_err());
    }
}
