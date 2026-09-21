//! The flow-match Euler scheduler as Qwen-Image 2.1 configures it: a
//! resolution-dependent exponential time shift, stretched to a terminal sigma.
//!
//! Ported from diffusers `FlowMatchEulerDiscreteScheduler` at 6256aa7 and the
//! `calculate_shift` call in `pipeline_qwenimage21.py`. It is its own file
//! rather than a second arm of `zimage::scheduler`, which is a vendored module
//! for a static-shift model and refuses dynamic shifting on purpose.
//!
//! Float widths follow the reference, because the grid is compared entry for
//! entry against a dump: `mu` is a Python float, so f64; the sigma grid is a
//! numpy f32 array from `linspace(...).astype(float32)` onwards, and numpy's
//! weak-scalar promotion keeps the shift and the stretch in f32 with `exp(mu)`
//! and `1 - shift_terminal` rounded to f32 first. The step is
//! `x + (sigma_next - sigma) * v` with the difference taken in f32.

use candle_core::{Result, Tensor};
use serde::Deserialize;

fn default_num_train_timesteps() -> usize {
    1000
}
fn default_base_shift() -> f64 {
    0.5
}
fn default_max_shift() -> f64 {
    1.15
}
fn default_base_image_seq_len() -> usize {
    256
}
fn default_max_image_seq_len() -> usize {
    4096
}
fn default_time_shift_type() -> String {
    "exponential".to_string()
}

/// `scheduler/scheduler_config.json`. The defaults are the reference class's,
/// which are NOT what the checkpoint ships (`max_shift` 0.9 over 8192 tokens),
/// so a config is always read from the file.
#[derive(Debug, Clone, Deserialize)]
pub struct SchedulerConfig {
    #[serde(default = "default_num_train_timesteps")]
    pub num_train_timesteps: usize,
    #[serde(default)]
    pub use_dynamic_shifting: bool,
    #[serde(default = "default_base_shift")]
    pub base_shift: f64,
    #[serde(default = "default_max_shift")]
    pub max_shift: f64,
    #[serde(default = "default_base_image_seq_len")]
    pub base_image_seq_len: usize,
    #[serde(default = "default_max_image_seq_len")]
    pub max_image_seq_len: usize,
    #[serde(default)]
    pub shift_terminal: Option<f64>,
    #[serde(default = "default_time_shift_type")]
    pub time_shift_type: String,
    #[serde(default)]
    pub invert_sigmas: bool,
    #[serde(default)]
    pub stochastic_sampling: bool,
    #[serde(default)]
    pub use_karras_sigmas: bool,
    #[serde(default)]
    pub use_exponential_sigmas: bool,
    #[serde(default)]
    pub use_beta_sigmas: bool,
}

impl SchedulerConfig {
    /// What the checkpoint ships, for tests and for callers with no file.
    pub fn qwen_image_21() -> Self {
        Self {
            num_train_timesteps: 1000,
            use_dynamic_shifting: true,
            base_shift: 0.5,
            max_shift: 0.9,
            base_image_seq_len: 256,
            max_image_seq_len: 8192,
            shift_terminal: Some(0.02),
            time_shift_type: "exponential".to_string(),
            invert_sigmas: false,
            stochastic_sampling: false,
            use_karras_sigmas: false,
            use_exponential_sigmas: false,
            use_beta_sigmas: false,
        }
    }

    /// Refuse every configuration this scheduler does not implement, so an
    /// unsupported checkpoint fails before anything large is resident.
    fn validate(&self) -> Result<()> {
        if !self.use_dynamic_shifting {
            candle_core::bail!(
                "qwen-image scheduler: use_dynamic_shifting is false; this scheduler implements \
                 the resolution-dependent shift only"
            );
        }
        if self.time_shift_type != "exponential" {
            candle_core::bail!(
                "qwen-image scheduler: time_shift_type {:?} is not implemented, only \
                 `exponential`",
                self.time_shift_type
            );
        }
        for (set, name) in [
            (self.invert_sigmas, "invert_sigmas"),
            (self.stochastic_sampling, "stochastic_sampling"),
            (self.use_karras_sigmas, "use_karras_sigmas"),
            (self.use_exponential_sigmas, "use_exponential_sigmas"),
            (self.use_beta_sigmas, "use_beta_sigmas"),
        ] {
            if set {
                candle_core::bail!("qwen-image scheduler: {name} is not implemented");
            }
        }
        if self.num_train_timesteps == 0 {
            candle_core::bail!("qwen-image scheduler: num_train_timesteps is 0");
        }
        if self.max_image_seq_len <= self.base_image_seq_len {
            candle_core::bail!(
                "qwen-image scheduler: max_image_seq_len {} is not above base_image_seq_len {}",
                self.max_image_seq_len,
                self.base_image_seq_len
            );
        }
        if let Some(terminal) = self.shift_terminal
            && !(0.0..1.0).contains(&terminal)
        {
            candle_core::bail!("qwen-image scheduler: shift_terminal {terminal} is outside [0, 1)");
        }
        Ok(())
    }
}

/// The scheduler. Built once per pipeline, [`Self::set_timesteps`] once per
/// image.
#[derive(Debug, Clone)]
pub struct DynamicShiftScheduler {
    config: SchedulerConfig,
    /// `steps + 1` entries, the last one 0.
    sigmas: Vec<f32>,
    step_index: usize,
}

impl DynamicShiftScheduler {
    pub fn new(config: SchedulerConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config,
            sigmas: vec![0.0],
            step_index: 0,
        })
    }

    pub fn config(&self) -> &SchedulerConfig {
        &self.config
    }

    /// The shift exponent for an image of `image_tokens` TARGET tokens: the
    /// line through `(base_image_seq_len, base_shift)` and
    /// `(max_image_seq_len, max_shift)`. It is not clamped, so a token count
    /// past `max_image_seq_len` extrapolates, as the reference does at the
    /// native 2048x2048.
    pub fn mu(&self, image_tokens: usize) -> f64 {
        let c = &self.config;
        let m = (c.max_shift - c.base_shift)
            / (c.max_image_seq_len as f64 - c.base_image_seq_len as f64);
        let b = c.base_shift - m * c.base_image_seq_len as f64;
        image_tokens as f64 * m + b
    }

    /// Lay out `steps` sigmas for an image of `image_tokens` target tokens:
    /// `linspace(1, 1/steps, steps)`, the exponential shift
    /// `e^mu / (e^mu + (1/s - 1))`, the terminal stretch, then a trailing 0.
    pub fn set_timesteps(&mut self, steps: usize, image_tokens: usize) -> Result<()> {
        if steps == 0 {
            candle_core::bail!("qwen-image scheduler: zero inference steps");
        }
        if image_tokens == 0 {
            candle_core::bail!("qwen-image scheduler: zero image tokens");
        }
        let exp_mu = self.mu(image_tokens).exp() as f32;
        let last = 1.0 / steps as f64;
        let mut sigmas: Vec<f32> = (0..steps)
            .map(|i| {
                // numpy's linspace: `start + i * step` in f64, the endpoint
                // assigned rather than accumulated, then rounded to f32.
                let s = if i + 1 == steps {
                    last
                } else {
                    1.0 + i as f64 * ((last - 1.0) / (steps as f64 - 1.0))
                };
                let s = s as f32;
                exp_mu / (exp_mu + (1.0 / s - 1.0))
            })
            .collect();
        if let Some(terminal) = self.config.shift_terminal
            && terminal != 0.0
        {
            let one_minus_last = 1.0 - sigmas[steps - 1];
            let scale = one_minus_last / (1.0 - terminal) as f32;
            for s in &mut sigmas {
                *s = 1.0 - (1.0 - *s) / scale;
            }
        }
        sigmas.push(0.0);
        self.sigmas = sigmas;
        self.step_index = 0;
        Ok(())
    }

    /// Every sigma, the trailing 0 included.
    pub fn sigmas(&self) -> &[f32] {
        &self.sigmas
    }

    pub fn num_inference_steps(&self) -> usize {
        self.sigmas.len() - 1
    }

    pub fn step_index(&self) -> usize {
        self.step_index
    }

    pub fn is_complete(&self) -> bool {
        self.step_index >= self.num_inference_steps()
    }

    /// The sigma of the step about to run, which is also the `t` the
    /// transformer takes: the model multiplies by 1000 itself.
    pub fn current_sigma(&self) -> f32 {
        self.sigmas[self.step_index]
    }

    /// One Euler step on the velocity as the model returns it, in f32:
    /// `sample + (sigma_next - sigma) * velocity`.
    pub fn step(&mut self, velocity: &Tensor, sample: &Tensor) -> Result<Tensor> {
        if self.is_complete() {
            candle_core::bail!("qwen-image scheduler: stepped past the last sigma");
        }
        let dt = self.sigmas[self.step_index + 1] - self.sigmas[self.step_index];
        let sample = sample.to_dtype(candle_core::DType::F32)?;
        let velocity = velocity.to_dtype(candle_core::DType::F32)?;
        let next = (sample + (velocity * f64::from(dt))?)?;
        self.step_index += 1;
        Ok(next)
    }

    /// Back to the first sigma of the current grid.
    pub fn reset(&mut self) {
        self.step_index = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn shipped() -> DynamicShiftScheduler {
        DynamicShiftScheduler::new(SchedulerConfig::qwen_image_21()).unwrap()
    }

    #[test]
    fn mu_is_the_line_through_the_two_configured_points() {
        let s = shipped();
        assert!((s.mu(256) - 0.5).abs() < 1e-12);
        assert!((s.mu(8192) - 0.9).abs() < 1e-12);
        // 1024x1024 and the native 2048x2048, the latter past
        // `max_image_seq_len` and therefore extrapolated.
        assert_eq!(format!("{:.6}", s.mu(4096)), "0.693548");
        assert_eq!(format!("{:.6}", s.mu(16384)), "1.312903");
    }

    #[test]
    fn the_forty_step_grid_runs_from_one_to_the_terminal_sigma() {
        let mut s = shipped();
        s.set_timesteps(40, 4096).unwrap();
        let sigmas = s.sigmas();
        assert_eq!(sigmas.len(), 41);
        assert_eq!(sigmas[0], 1.0);
        assert!((sigmas[39] - 0.02).abs() < 1e-6, "{}", sigmas[39]);
        assert_eq!(sigmas[40], 0.0);
        assert!(sigmas.windows(2).all(|w| w[0] > w[1]), "{sigmas:?}");
    }

    #[test]
    fn the_shift_matches_the_closed_form_in_f64() {
        let mut s = shipped();
        s.set_timesteps(40, 4096).unwrap();
        let exp_mu = s.mu(4096).exp();
        let shift = |t: f64| exp_mu / (exp_mu + (1.0 / t - 1.0));
        let last = shift(1.0 / 40.0);
        let scale = (1.0 - last) / (1.0 - 0.02);
        for (i, &got) in s.sigmas()[..40].iter().enumerate() {
            let t = 1.0 + i as f64 * ((1.0 / 40.0 - 1.0) / 39.0);
            let want = 1.0 - (1.0 - shift(t)) / scale;
            assert!(
                (f64::from(got) - want).abs() < 2e-6,
                "sigma {i}: {got} vs {want}"
            );
        }
    }

    #[test]
    fn a_step_adds_the_velocity_times_the_sigma_difference() {
        let mut s = shipped();
        s.set_timesteps(4, 1024).unwrap();
        let dev = Device::Cpu;
        let x = Tensor::new(&[1.0f32, -2.0], &dev).unwrap();
        let v = Tensor::new(&[0.5f32, 4.0], &dev).unwrap();
        let dt = s.sigmas()[1] - s.sigmas()[0];
        assert_eq!(s.current_sigma(), 1.0);
        let y = s.step(&v, &x).unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(y, vec![1.0 + dt * 0.5, -2.0 + dt * 4.0]);
        assert_eq!(s.step_index(), 1);
        for _ in 0..3 {
            s.step(&v, &x).unwrap();
        }
        assert!(s.is_complete());
        assert!(s.step(&v, &x).is_err());
    }

    #[test]
    fn the_shipped_config_parses_and_everything_unimplemented_is_refused() {
        let shipped: SchedulerConfig = serde_json::from_str(
            r#"{"_class_name": "FlowMatchEulerDiscreteScheduler", "base_image_seq_len": 256,
                "base_shift": 0.5, "invert_sigmas": false, "max_image_seq_len": 8192,
                "max_shift": 0.9, "num_train_timesteps": 1000, "shift": 1.0,
                "shift_terminal": 0.02, "stochastic_sampling": false,
                "time_shift_type": "exponential", "use_beta_sigmas": false,
                "use_dynamic_shifting": true, "use_exponential_sigmas": false,
                "use_karras_sigmas": false}"#,
        )
        .unwrap();
        DynamicShiftScheduler::new(shipped).unwrap();

        let refused = |edit: fn(&mut SchedulerConfig), needle: &str| {
            let mut cfg = SchedulerConfig::qwen_image_21();
            edit(&mut cfg);
            let err = DynamicShiftScheduler::new(cfg).unwrap_err().to_string();
            assert!(err.contains(needle), "{err}");
        };
        refused(|c| c.use_dynamic_shifting = false, "use_dynamic_shifting");
        refused(|c| c.time_shift_type = "linear".into(), "time_shift_type");
        refused(|c| c.invert_sigmas = true, "invert_sigmas");
        refused(|c| c.stochastic_sampling = true, "stochastic_sampling");
        refused(|c| c.use_karras_sigmas = true, "use_karras_sigmas");
        refused(
            |c| c.use_exponential_sigmas = true,
            "use_exponential_sigmas",
        );
        refused(|c| c.use_beta_sigmas = true, "use_beta_sigmas");
    }
}
