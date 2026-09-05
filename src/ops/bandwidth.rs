//! Achievable GPU memory bandwidth on this machine, measured rather than quoted.
//!
//! Bytes-vs-time arguments in this repo need a real streaming rate to divide by;
//! the spec-sheet figure is an upper bound no kernel reaches. `bandwidth_sweep`
//! streams a 2 GB buffer (far past any cache) with a reduce-only read kernel and a
//! copy kernel, amortized (many dispatches per sync), and also at the plane sizes
//! a decode step actually dispatches (a 28 MB `attn_qkv`, a 3.5 MB hc plane), so
//! the small-plane rows show how much of the streaming rate a weight-sized
//! dispatch keeps once launch overhead is in the picture. Nothing here is on a
//! model path.
use anyhow::{Result, bail};
use candle_core::{MetalDevice, Storage, Tensor};
use candle_metal_kernels::metal::Buffer;

use crate::ops::dispatch::run_bw_probe;

/// Raw Metal buffer behind a contiguous f32 tensor, with its byte offset.
pub fn buffer_of(t: &Tensor) -> Result<(Buffer, usize)> {
    let (storage, layout) = t.storage_and_layout();
    let Storage::Metal(ms) = &*storage else {
        bail!("bandwidth probe expects Metal storage")
    };
    Ok((ms.buffer().clone(), layout.start_offset() * 4))
}

/// One reduce-only read of `bytes` bytes at `src_off` into `src`; the per-group
/// partial sums land in `out` (which must hold `groups` floats).
pub fn read_probe(
    mdev: &MetalDevice,
    src: &Buffer,
    src_off: usize,
    out: &Buffer,
    bytes: usize,
    groups: usize,
) -> Result<()> {
    run_bw_probe(mdev, true, src, src_off, out, 0, bytes / 16, groups)
}

/// One copy of `bytes` bytes from `src` at `src_off` to `dst` at `dst_off`.
pub fn copy_probe(
    mdev: &MetalDevice,
    src: &Buffer,
    src_off: usize,
    dst: &Buffer,
    dst_off: usize,
    bytes: usize,
    groups: usize,
) -> Result<()> {
    run_bw_probe(mdev, false, src, src_off, dst, dst_off, bytes / 16, groups)
}

#[cfg(test)]
mod tests {
    use std::hint::black_box;
    use std::time::Instant;

    use candle_core::{DType, Device, Tensor};

    use super::*;
    use crate::gguf::metal_device;

    fn env_or(k: &str, d: usize) -> usize {
        std::env::var(k)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(d)
    }

    /// Correctness of the read kernel: the partial sums add up to the buffer's sum.
    #[test]
    fn read_probe_sums_the_buffer() {
        let device = metal_device().unwrap();
        let Device::Metal(mdev) = &device else {
            unreachable!("metal_device returned a non-Metal device")
        };
        let n = 1 << 20;
        let host: Vec<f32> = (0..n).map(|i| ((i % 97) as f32) - 48.0).collect();
        let expected: f64 = host.iter().map(|&v| v as f64).sum();
        let t = Tensor::from_vec(host, n, &device).unwrap();
        let (src, off) = buffer_of(&t).unwrap();
        for groups in [1usize, 7, 64, 1000] {
            let out_t = Tensor::zeros(groups, DType::F32, &device).unwrap();
            let (out, out_off) = buffer_of(&out_t).unwrap();
            assert_eq!(out_off, 0);
            read_probe(mdev, &src, off, &out, n * 4, groups).unwrap();
            let got: f64 = out_t
                .to_vec1::<f32>()
                .unwrap()
                .iter()
                .map(|&v| v as f64)
                .sum();
            assert!(
                (got - expected).abs() <= 1e-3 * expected.abs().max(1.0),
                "groups {groups}: {got} vs {expected}"
            );
        }
    }

    /// Correctness of the copy kernel, including a non-zero source offset.
    #[test]
    fn copy_probe_copies_the_range() {
        let device = metal_device().unwrap();
        let Device::Metal(mdev) = &device else {
            unreachable!("metal_device returned a non-Metal device")
        };
        let n = 1 << 18;
        let host: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let t = Tensor::from_vec(host.clone(), n, &device).unwrap();
        let dst_t = Tensor::zeros(n, DType::F32, &device).unwrap();
        let (src, off) = buffer_of(&t).unwrap();
        let (dst, dst_off) = buffer_of(&dst_t).unwrap();
        let skip = 1 << 16;
        copy_probe(
            mdev,
            &src,
            off + skip * 4,
            &dst,
            dst_off,
            (n - skip) * 4,
            33,
        )
        .unwrap();
        let got = dst_t.to_vec1::<f32>().unwrap();
        assert_eq!(&got[..n - skip], &host[skip..]);
        assert!(got[n - skip..].iter().all(|&v| v == 0.0));
    }

    /// Achievable-bandwidth sweep. `#[ignore]`d — run on a free GPU with:
    ///   cargo test --release -p xwen bandwidth_sweep -- --ignored --nocapture
    /// Arms are interleaved within a round (order reversed on odd rounds), one
    /// sync per arm after `batch` dispatches, `XWEN_BENCH_IDLE` seconds (60)
    /// idle between rounds, `XWEN_BENCH_ITERS` rounds (5) after one warm-up.
    /// Reports median and best per arm as µs/dispatch and GB/s of bytes touched
    /// (read: bytes read; copy: bytes read + bytes written).
    #[test]
    #[ignore = "perf bench"]
    fn bandwidth_sweep() {
        let device = metal_device().unwrap();
        let Device::Metal(mdev) = &device else {
            unreachable!("metal_device returned a non-Metal device")
        };
        let rounds = env_or("XWEN_BENCH_ITERS", 5);
        let idle = env_or("XWEN_BENCH_IDLE", 60);
        let power = std::process::Command::new("pmset")
            .arg("-g")
            .output()
            .unwrap();
        eprintln!("{}", String::from_utf8_lossy(&power.stdout));

        const MB: usize = 1 << 20;
        let big_bytes = 2048 * MB;
        let dst_bytes = 1024 * MB;
        // Filled on the device (no 2 GB host staging copy); random so the read
        // kernel sums real data rather than a constant.
        let big_t = Tensor::rand(0f32, 1f32, big_bytes / 4, &device).unwrap();
        let dst_t = Tensor::zeros(dst_bytes / 4, DType::F32, &device).unwrap();
        let max_groups = 16384;
        let out_t = Tensor::zeros(max_groups, DType::F32, &device).unwrap();
        let (big, big_off) = buffer_of(&big_t).unwrap();
        let (dst, dst_off) = buffer_of(&dst_t).unwrap();
        let (out, _) = buffer_of(&out_t).unwrap();
        device.synchronize().unwrap();

        // (label, read?, plane bytes, batch, groups). Planes rotate through the 2 GB
        // source (and the 1 GB destination for copies), so no plane is cache-resident
        // when its turn comes back.
        struct Arm {
            label: &'static str,
            read: bool,
            plane: usize,
            batch: usize,
            groups: usize,
        }
        let arms = [
            Arm {
                label: "read  2 GB   g1024 ",
                read: true,
                plane: 2048 * MB,
                batch: 4,
                groups: 1024,
            },
            Arm {
                label: "read  2 GB   g4096 ",
                read: true,
                plane: 2048 * MB,
                batch: 4,
                groups: 4096,
            },
            Arm {
                label: "read  2 GB   g16384",
                read: true,
                plane: 2048 * MB,
                batch: 4,
                groups: 16384,
            },
            Arm {
                label: "read  256 MB g4096 ",
                read: true,
                plane: 256 * MB,
                batch: 32,
                groups: 4096,
            },
            Arm {
                label: "read  32 MB  g4096 ",
                read: true,
                plane: 32 * MB,
                batch: 64,
                groups: 4096,
            },
            Arm {
                label: "read  32 MB  g1024 ",
                read: true,
                plane: 32 * MB,
                batch: 64,
                groups: 1024,
            },
            Arm {
                label: "read  4 MB   g1024 ",
                read: true,
                plane: 4 * MB,
                batch: 256,
                groups: 1024,
            },
            Arm {
                label: "read  1 MB   g256  ",
                read: true,
                plane: MB,
                batch: 512,
                groups: 256,
            },
            Arm {
                label: "read  256 KB g64   ",
                read: true,
                plane: 256 << 10,
                batch: 512,
                groups: 64,
            },
            Arm {
                label: "read  64 KB  g16   ",
                read: true,
                plane: 64 << 10,
                batch: 512,
                groups: 16,
            },
            Arm {
                label: "read  4 KB   g1    ",
                read: true,
                plane: 4 << 10,
                batch: 512,
                groups: 1,
            },
            Arm {
                label: "copy  1 GB   g4096 ",
                read: false,
                plane: 1024 * MB,
                batch: 4,
                groups: 4096,
            },
            Arm {
                label: "copy  32 MB  g4096 ",
                read: false,
                plane: 32 * MB,
                batch: 64,
                groups: 4096,
            },
            Arm {
                label: "copy  4 MB   g1024 ",
                read: false,
                plane: 4 * MB,
                batch: 256,
                groups: 1024,
            },
        ];
        let run_arm = |arm: &Arm| -> f64 {
            let src_planes = big_bytes / arm.plane;
            let dst_planes = dst_bytes / arm.plane;
            device.synchronize().unwrap();
            let t = Instant::now();
            for i in 0..arm.batch {
                let s_off = big_off + (i % src_planes) * arm.plane;
                if arm.read {
                    read_probe(mdev, &big, s_off, &out, arm.plane, arm.groups).unwrap();
                } else {
                    let d_off = dst_off + (i % dst_planes) * arm.plane;
                    copy_probe(mdev, &big, s_off, &dst, d_off, arm.plane, arm.groups).unwrap();
                }
            }
            device.synchronize().unwrap();
            t.elapsed().as_secs_f64() / arm.batch as f64
        };

        let mut times: Vec<Vec<f64>> = vec![Vec::new(); arms.len()];
        for round in 0..=rounds {
            if round > 0 {
                eprintln!("bandwidth sweep: idle {idle}s before round {round}/{rounds}");
                std::thread::sleep(std::time::Duration::from_secs(idle as u64));
            }
            let order: Vec<usize> = if round % 2 == 0 {
                (0..arms.len()).collect()
            } else {
                (0..arms.len()).rev().collect()
            };
            for i in order {
                let per = run_arm(&arms[i]);
                if round > 0 {
                    times[i].push(per);
                }
            }
        }
        black_box(out_t.to_vec1::<f32>().unwrap());
        eprintln!("arm | med us/dispatch | med GB/s | best GB/s | worst GB/s");
        for (arm, ts) in arms.iter().zip(times.iter_mut()) {
            ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let bytes = if arm.read { arm.plane } else { 2 * arm.plane } as f64;
            let med = ts[ts.len() / 2];
            eprintln!(
                "{} batch={:3} | {:9.1} | {:6.1} | {:6.1} | {:6.1}",
                arm.label,
                arm.batch,
                med * 1e6,
                bytes / med / 1e9,
                bytes / ts[0] / 1e9,
                bytes / ts[ts.len() - 1] / 1e9,
            );
        }
    }
}
