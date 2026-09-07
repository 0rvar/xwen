//! The linear layer of the Z-Image transformer: a bf16 weight plane applied
//! to f32 activations through xwen's Metal-4 cooperative-tensor gemm
//! (`ops::matmul_bf16`), which is the kernel the language models' prefill
//! runs on. The weights stay bf16 on the device, as stored after the load-time
//! cast; everything that flows between layers is f32.
//!
//! The kernel stages each weight tile to f16 on its way into the tensor unit,
//! so a weight past f16's finite range would be garbage with no error.
//! [`ensure_weights_fit_f16`] refuses such a checkpoint at load, the way the
//! DFlash loader does for its own bf16 planes. Values below f16's normal floor
//! (6.1e-5) round or flush there; the shipped checkpoint has 0.035% of its
//! weights in that band (2,136,677 of 6,153,863,168, measured once with a
//! device-side count that cost 1.2 s of load and was dropped), and the parity
//! gate's bars hold with them.
//!
//! `XWEN_ZIMAGE_LINEAR=candle` is the bisect arm: candle's own bf16 gemm on
//! bf16-rounded activations, the path every projection ran before this kernel
//! landed. It shares no matmul code with the shipped arm.

use candle_core::{D, DType, Device, Module, Result, Tensor};
use candle_nn::VarBuilder;

/// The environment switch that picks the linear-layer kernel, read when a
/// [`super::transformer::Config`] is built. `xwen` (or unset) is the shipped
/// Metal-4 tensor gemm; `candle` is the bisect arm.
pub const LINEAR_ENV: &str = "XWEN_ZIMAGE_LINEAR";

/// f16's largest finite value; a weight above it cannot pass through the
/// tensor gemm's half staging tile.
pub const F16_MAX: f32 = 65504.0;

/// Which matmul a [`Projection`] runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinearImpl {
    /// The shipped path: `ops::matmul_bf16`, bf16 weight times f32 activation
    /// with f32 accumulation, on a Metal device. Off Metal it falls through to
    /// the candle chain, the kernel being Metal-only.
    Xwen,
    /// candle's own gemm over the bf16 weight, with the activation rounded to
    /// bf16 first and the product widened back — the pre-kernel path.
    Candle,
}

impl LinearImpl {
    /// Resolve from [`LINEAR_ENV`]: unset means the shipped path, anything
    /// else must name an arm.
    pub fn from_env() -> Result<Self> {
        match std::env::var(LINEAR_ENV) {
            Err(std::env::VarError::NotPresent) => Ok(Self::Xwen),
            Err(std::env::VarError::NotUnicode(_)) => {
                candle_core::bail!("{LINEAR_ENV} is not valid UTF-8")
            }
            Ok(value) => Self::parse(&value),
        }
    }

    /// [`Self::from_env`] with a bad value read as the shipped path, for a
    /// `serde` default, which cannot fail. `ZImagePipeline::load` calls
    /// [`Self::from_env`] with a `?` first, so a typo is a load error there.
    pub fn from_env_or_default() -> Self {
        Self::from_env().unwrap_or(Self::Xwen)
    }

    /// `xwen` / `tensor` (or empty) select the shipped path, `candle` the
    /// bisect arm; anything else is refused rather than defaulted.
    pub fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "" | "xwen" | "tensor" => Ok(Self::Xwen),
            "candle" => Ok(Self::Candle),
            other => candle_core::bail!(
                "{LINEAR_ENV}={other:?}: expected `xwen` (the default) or `candle`"
            ),
        }
    }

    /// Whether this arm is the xwen kernel, which is how `Config` stores it.
    pub fn is_xwen(self) -> bool {
        matches!(self, Self::Xwen)
    }

    /// The name a dump or a log records for provenance.
    pub fn label(self) -> &'static str {
        match self {
            Self::Xwen => "xwen",
            Self::Candle => "candle",
        }
    }
}

/// A linear layer: bf16 `[out, in]` weight, optional f32 bias, f32 in and out.
#[derive(Debug, Clone)]
pub struct Projection {
    /// `[out, in]` bf16, contiguous, as stored.
    weight: Tensor,
    /// `[out]` f32.
    bias: Option<Tensor>,
    /// The tensor's name in the checkpoint, for the load-time range report.
    name: String,
    use_xwen: bool,
}

impl Projection {
    /// Fetch `<prefix>.weight` (and `.bias` when `bias` is set) through `vb`,
    /// the weight as bf16 and the bias in `vb`'s own dtype.
    ///
    /// The kernel's contract is asked here, once, rather than on every
    /// forward: `in % 32 == 0` and `out % 4 == 0`. Every projection of the
    /// shipped checkpoint satisfies both. Off Metal the kernel never runs, so
    /// a CPU build of a small test model is not held to it.
    pub fn new(
        in_dim: usize,
        out_dim: usize,
        bias: bool,
        vb: VarBuilder,
        arm: LinearImpl,
    ) -> Result<Self> {
        let name = format!("{}.weight", vb.prefix());
        let weight = vb
            .clone()
            .set_dtype(DType::BF16)
            .get((out_dim, in_dim), "weight")?
            .contiguous()?;
        if arm.is_xwen()
            && vb.device().is_metal()
            && !(in_dim.is_multiple_of(32) && out_dim.is_multiple_of(4))
        {
            candle_core::bail!(
                "{name}: [{out_dim}, {in_dim}] needs in % 32 == 0 and out % 4 == 0 for \
                 matmul_bf16"
            );
        }
        let bias = if bias {
            Some(vb.get(out_dim, "bias")?.to_dtype(DType::F32)?)
        } else {
            None
        };
        Ok(Self {
            weight,
            bias,
            name,
            use_xwen: arm.is_xwen(),
        })
    }

    /// Build from tensors already in hand: `weight` any float dtype `[out, in]`
    /// (cast to bf16 here), `bias` `[out]`.
    pub fn from_weights(
        weight: Tensor,
        bias: Option<Tensor>,
        name: impl Into<String>,
        arm: LinearImpl,
    ) -> Result<Self> {
        let weight = weight.to_dtype(DType::BF16)?.contiguous()?;
        let bias = bias.map(|b| b.to_dtype(DType::F32)).transpose()?;
        Ok(Self {
            weight,
            bias,
            name: name.into(),
            use_xwen: arm.is_xwen(),
        })
    }

    /// The bf16 `[out, in]` weight.
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    /// The checkpoint name of the weight.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// `x @ W^T + b` for an f32 `x` of rank 2 or 3 whose last dim is `in`.
    fn project(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.to_dtype(DType::F32)?;
        let (lead, k) = match x.dims() {
            [t, k] => (vec![*t], *k),
            [b, t, k] => (vec![*b, *t], *k),
            dims => candle_core::bail!("projection input must be rank 2 or 3, got {dims:?}"),
        };
        let t: usize = lead.iter().product();
        let x2 = x.reshape((t, k))?.contiguous()?;
        let out = self.dims_out();
        let y = if self.use_xwen && x2.device().is_metal() {
            crate::ops::matmul_bf16(&self.weight, &x2)
                .map_err(|e| candle_core::Error::Msg(format!("{}: {e:#}", self.name)))?
        } else if x2.device().is_metal() {
            // The pre-kernel path: candle's bf16 gemm over bf16-rounded activations.
            x2.to_dtype(DType::BF16)?
                .matmul(&self.weight.t()?)?
                .to_dtype(DType::F32)?
        } else {
            // candle's CPU backend has no bf16 gemm: the same rounding of the
            // activation, then the product in f32 over the widened weight.
            let x_rounded = x2.to_dtype(DType::BF16)?.to_dtype(DType::F32)?;
            x_rounded.matmul(&self.weight.to_dtype(DType::F32)?.t()?)?
        };
        let y = match &self.bias {
            Some(b) => y.broadcast_add(b)?,
            None => y,
        };
        let mut shape = lead;
        shape.push(out);
        y.reshape(shape)
    }

    fn dims_out(&self) -> usize {
        self.weight.dim(0).expect("a projection weight is rank 2")
    }
}

impl Module for Projection {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.project(x)
    }
}

/// What [`ensure_weights_fit_f16`] found: the largest magnitude over every
/// projection, its tensor, and how many values it looked at.
#[derive(Debug, Clone, PartialEq)]
pub struct WeightRange {
    pub max_abs: f32,
    pub max_abs_tensor: String,
    pub total: u64,
}

impl WeightRange {
    /// One line for the load log.
    pub fn summary(&self) -> String {
        format!(
            "z-image projections: max |w| {:.4} in {} over {} values, inside f16's range",
            self.max_abs, self.max_abs_tensor, self.total
        )
    }
}

/// Refuse a set of projections whose bf16 weights would not survive the
/// tensor gemm's f16 staging: any `|w| > 65504` is an error naming the tensor.
/// The reduction runs in bf16 on the weights' own device and is read back per
/// tensor, so the answer is the one that tensor's bytes give and nothing is
/// copied out to f32. About 0.2 s over the shipped checkpoint's 300 planes.
pub fn ensure_weights_fit_f16<'a>(
    projections: impl IntoIterator<Item = &'a Projection>,
    _device: &Device,
) -> Result<WeightRange> {
    let mut range = WeightRange {
        max_abs: 0.0,
        max_abs_tensor: String::new(),
        total: 0,
    };
    let mut first = true;
    for p in projections {
        let a = p.weight.abs()?;
        // Two-stage reductions: one row per threadgroup first, then the
        // handful of row results. A whole-tensor reduction in one call runs
        // as a single threadgroup on Metal and costs tens of milliseconds per
        // plane, which over 300 planes was seconds of load time.
        let max = a
            .max_keepdim(D::Minus1)?
            .max_all()?
            .to_dtype(DType::F32)?
            .to_scalar::<f32>()?;
        if !max.is_finite() {
            candle_core::bail!(
                "{} holds a non-finite weight; the checkpoint cannot run through the bf16 \
                 kernels",
                p.name
            );
        }
        if max > F16_MAX {
            candle_core::bail!(
                "{} holds a weight of magnitude {max}, past f16's finite range ({F16_MAX}); the \
                 bf16 kernels stage weights to f16, so this checkpoint cannot run through them",
                p.name
            );
        }
        range.total += a.elem_count() as u64;
        if first || max > range.max_abs {
            range.max_abs = max;
            range.max_abs_tensor = p.name.clone();
            first = false;
        }
    }
    Ok(range)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu() -> Device {
        Device::Cpu
    }

    #[test]
    fn the_linear_env_switch_names_its_arms_and_refuses_a_typo() {
        assert_eq!(LinearImpl::parse("").unwrap(), LinearImpl::Xwen);
        assert_eq!(LinearImpl::parse("xwen").unwrap(), LinearImpl::Xwen);
        assert_eq!(LinearImpl::parse(" Tensor ").unwrap(), LinearImpl::Xwen);
        assert_eq!(LinearImpl::parse("candle").unwrap(), LinearImpl::Candle);
        let err = LinearImpl::parse("fast").unwrap_err().to_string();
        assert!(err.contains(LINEAR_ENV) && err.contains("candle"), "{err}");
    }

    /// The candle arm on the CPU: `x @ W^T + b` against a hand computation,
    /// with the activation rounded to bf16 the way that arm does.
    #[test]
    fn the_projection_applies_the_stored_orientation_and_the_bias() {
        let dev = cpu();
        let w = Tensor::from_vec(
            vec![1f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], // [2, 4] = [out, in]
            (2, 4),
            &dev,
        )
        .unwrap();
        let b = Tensor::from_vec(vec![0.5f32, -0.5], 2, &dev).unwrap();
        let p = Projection::from_weights(w, Some(b), "probe", LinearImpl::Candle).unwrap();
        let x = Tensor::from_vec(
            vec![1f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            (1, 2, 4),
            &dev,
        )
        .unwrap();
        let y = p.forward(&x).unwrap();
        assert_eq!(y.dims(), &[1, 2, 2]);
        assert_eq!(y.dtype(), DType::F32);
        let y = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(y, vec![1.5, 4.5, 2.5, 5.5]);
        assert_eq!(p.weight().dtype(), DType::BF16);
    }

    /// The range guard: a weight past f16's range is refused by name, a
    /// normal set passes with its maximum reported.
    #[test]
    fn the_range_guard_refuses_an_f16_overflow() {
        let dev = cpu();
        let fine = Projection::from_weights(
            Tensor::from_vec(vec![14f32, -3.0, 1e-5, 0.0], (2, 2), &dev).unwrap(),
            None,
            "fine.weight",
            LinearImpl::Xwen,
        )
        .unwrap();
        let range = ensure_weights_fit_f16([&fine], &dev).unwrap();
        assert_eq!(range.max_abs_tensor, "fine.weight");
        assert!((range.max_abs - 14.0).abs() < 1e-6, "{range:?}");
        assert_eq!(range.total, 4);
        assert!(range.summary().contains("fine.weight"));

        let huge = Projection::from_weights(
            Tensor::from_vec(vec![1f32, 70000.0, 0.0, 0.0], (2, 2), &dev).unwrap(),
            None,
            "huge.weight",
            LinearImpl::Xwen,
        )
        .unwrap();
        let err = ensure_weights_fit_f16([&fine, &huge], &dev)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("huge.weight") && err.contains("65504"),
            "{err}"
        );
    }

    /// The same guard on Metal, where the shipped load runs it: the bf16
    /// reduction there agrees with the CPU on the max and the element count
    /// over a plane large enough to take the strided kernels.
    #[test]
    fn the_range_guard_agrees_between_metal_and_the_cpu() {
        let Ok(dev) = crate::gguf::metal_device() else {
            eprintln!("skipping the_range_guard_agrees_between_metal_and_the_cpu: no Metal");
            return;
        };
        let (n, k) = (1024usize, 3840usize);
        let mut values: Vec<f32> = (0..n * k)
            .map(|i| ((i * 2_654_435_761usize) % 1_000_003) as f32 / 1_000_003.0 - 0.5)
            .collect();
        values[12_345] = 14.0;
        values[777] = 0.0;
        let w = Tensor::from_vec(values, (n, k), &cpu()).unwrap();
        let on_cpu = Projection::from_weights(w.clone(), None, "plane", LinearImpl::Xwen).unwrap();
        let on_metal =
            Projection::from_weights(w.to_device(&dev).unwrap(), None, "plane", LinearImpl::Xwen)
                .unwrap();
        let a = ensure_weights_fit_f16([&on_cpu], &cpu()).unwrap();
        let b = ensure_weights_fit_f16([&on_metal], &dev).unwrap();
        assert_eq!(a, b, "cpu {a:?} vs metal {b:?}");
        assert_eq!(a.max_abs, 14.0);
        assert_eq!(a.total, (n * k) as u64);
    }

    /// The xwen kernel against the candle arm on the same weights, at a shape
    /// the tensor gemm takes (t > 8, in % 32 == 0, out % 4 == 0). Bounded,
    /// not bitwise: the two arms round the activation differently on purpose.
    #[test]
    fn the_xwen_arm_agrees_with_the_candle_arm_on_metal() {
        let Ok(dev) = crate::gguf::metal_device() else {
            eprintln!("skipping the_xwen_arm_agrees_with_the_candle_arm_on_metal: no Metal");
            return;
        };
        let (t, k, n) = (64usize, 256usize, 128usize);
        let w = Tensor::randn(0f32, 0.05, (n, k), &dev).unwrap();
        let b = Tensor::randn(0f32, 0.1, n, &dev).unwrap();
        let x = Tensor::randn(0f32, 1.0, (1, t, k), &dev).unwrap();
        let xwen =
            Projection::from_weights(w.clone(), Some(b.clone()), "p", LinearImpl::Xwen).unwrap();
        let candle = Projection::from_weights(w, Some(b), "p", LinearImpl::Candle).unwrap();
        let a = xwen.forward(&x).unwrap();
        let c = candle.forward(&x).unwrap();
        assert_eq!(a.dims(), &[1, t, n]);
        let a = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let c = c.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let (mut num, mut den, mut worst) = (0f64, 0f64, 0f32);
        for (p, q) in a.iter().zip(&c) {
            assert!(p.is_finite() && q.is_finite());
            num += ((p - q) * (p - q)) as f64;
            den += (q * q) as f64;
            worst = worst.max((p - q).abs());
        }
        let rel_l2 = (num / den).sqrt();
        eprintln!(
            "z-image projection, xwen vs candle: rel_l2 {rel_l2:.3e}, max |delta| {worst:.3e}"
        );
        // bf16 activation rounding on the candle side is ~4e-3 relative per
        // element and averages down over k = 256; the tensor gemm's own class
        // is ~2e-4.
        assert!(rel_l2 < 5e-3, "rel_l2 {rel_l2}");
        assert!(
            worst > 0.0,
            "bit-identical output means the arms ran the same kernel"
        );
    }
}
