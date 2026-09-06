//! Tile-batched sparse attention for a QSA prefill chunk (`qsa_tiles.metal`),
//! the route that makes the sparse layers' attention actually sparse at
//! prefill. Before it, a chunk's `[n, n_kv]` mask went to candle's sdpa over
//! the WHOLE cache, so every masked-out column was computed and discarded:
//! at 131072 tokens the sdpa on the 12 QSA layers was 52% of the prefill wall
//! (the 2026-09-06 probe), while each query actually sees 2048 positions of
//! it.
//!
//! The route: queries in tiles of T; per tile the union of the blocks its T
//! queries selected (an exact superset of every query's set), that union's K
//! and V rows gathered into `[n_tiles, kv_heads, S, head_dim]`, the tile's
//! `[T, S]` mask read off the full mask at the union's columns, and ONE
//! batched sdpa over the tiles. Padding columns (to the host-chosen S) copy
//! row 0 and carry `-inf`; padding query rows past the chunk copy the last
//! real query and are dropped. Every column a query needs is in its tile's
//! union and every column it must not see is `-inf`, so the result is dense
//! masked attention's up to the online softmax's summation order — NOT bit
//! identical, which is why `XWEN_QSA_ATTN_CLASSIC` (the dense-over-the-mask
//! route) is a parity row and not a bitwise switch.
//!
//! One readback per chunk per layer: the tile counts (`n_tiles` u32), which
//! size S. That sync is the price of a fixed-shape batched sdpa; it waits on
//! this layer's selection only, not on a host-side fill.

use anyhow::{Result, ensure};
use candle_core::{DType, Tensor};

use crate::ops::dispatch;

/// What a QSA prefill chunk hands the attention when this route is on: the
/// full additive mask (the dense fallback, and the source of the tile masks)
/// plus the per-query block lists the mask kernel wrote beside it.
pub struct SparsePrefill {
    /// `[n, n_kv]` f32 additive.
    pub mask: Tensor,
    /// `[n, keep_max + 1]` u32, ascending per row within `nsel`.
    pub blocks: Tensor,
    /// `[n]` u32.
    pub nsel: Tensor,
    pub ratio: usize,
    /// The scored (complete) blocks; a list may also hold the id one past
    /// this, the tail's incomplete block.
    pub n_blocks: usize,
}

/// Padded column counts are rounded up to this, a whole number of the sdpa's
/// key tiles either way.
const S_ALIGN: usize = 64;

/// Attention of `q` (`[n_head, n, head_dim]` f32) over the cache views `k_all`
/// / `v_all` (`[n_kv_head, n_kv, head_dim]` f16) under `sp`'s selection, in
/// tiles of `t` queries. `[n_head, n, head_dim]` f32.
pub fn attend(
    q: &Tensor,
    k_all: &Tensor,
    v_all: &Tensor,
    sp: &SparsePrefill,
    scale: f32,
    t: usize,
) -> Result<Tensor> {
    let (n_head, n, head_dim) = q.dims3()?;
    ensure!(
        n > 1,
        "qsa_tiles::attend is the prefill route; a single query takes Rows"
    );
    ensure!(
        t >= 1 && t % 32 == 0,
        "qsa tile size {t} must be a positive multiple of 32"
    );
    // The lists name the tail's INCOMPLETE block too (id `n_kv / ratio` when
    // `n_kv % ratio != 0`), one past the scored blocks, so the union's universe
    // is the ceiling, not `sp.n_blocks`.
    let n_kv = sp.mask.dim(1)?;
    let universe = n_kv.div_ceil(sp.ratio);
    let (union, count) = dispatch::run_qsa_tile_union(&sp.blocks, &sp.nsel, t, universe)?;
    let n_tiles = union.dim(0)?;
    let counts = count.to_vec1::<u32>()?;
    let u_max = counts.iter().copied().max().unwrap_or(0) as usize;
    let s = (u_max * sp.ratio).max(t).div_ceil(S_ALIGN) * S_ALIGN;
    let (k_sel, v_sel) =
        dispatch::run_qsa_tile_gather_kv(&union, &count, k_all, v_all, n, t, s, sp.ratio)?;
    let mask = dispatch::run_qsa_tile_mask(&union, &count, &sp.mask, t, s, sp.ratio)?
        .broadcast_as((n_tiles, n_head, t, s))?;
    let qt = dispatch::run_qsa_tile_q(q, t)?;
    let out = candle_nn::ops::sdpa(&qt, &k_sel, &v_sel, Some(&mask), false, scale, 1.0)?;
    // [n_tiles, n_head, t, hd] f16 -> [n_head, n, hd] f32.
    Ok(out
        .permute((1, 0, 2, 3))?
        .contiguous()?
        .reshape((n_head, n_tiles * t, head_dim))?
        .narrow(1, 0, n)?
        .to_dtype(DType::F32)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::metal_device;
    use crate::ops::dispatch::testutil::pseudo_random;
    use candle_core::Device;

    /// Random per-query block lists of the mask kernel's shape: ascending,
    /// distinct, `nsel[i]` long within a `cap`-wide row.
    fn lists(n: usize, cap: usize, n_blocks: usize, seed: u64) -> (Vec<u32>, Vec<u32>) {
        let r = pseudo_random(n * cap * 2, seed, 0.0, 1.0);
        let mut blocks = vec![u32::MAX; n * cap];
        let mut nsel = Vec::with_capacity(n);
        for i in 0..n {
            let m = ((r[i * 2] * cap as f32) as usize).min(cap);
            let mut set: Vec<u32> = (0..m)
                .map(|s| {
                    ((r[n * cap + i * cap + s] * n_blocks as f32) as u32).min(n_blocks as u32 - 1)
                })
                .collect();
            set.sort_unstable();
            set.dedup();
            for (s, b) in set.iter().enumerate() {
                blocks[i * cap + s] = *b;
            }
            nsel.push(set.len() as u32);
        }
        (blocks, nsel)
    }

    fn host_union(blocks: &[u32], nsel: &[u32], n: usize, cap: usize, t: usize) -> Vec<Vec<u32>> {
        (0..n.div_ceil(t))
            .map(|tile| {
                let mut u: Vec<u32> = (tile * t..((tile + 1) * t).min(n))
                    .flat_map(|i| blocks[i * cap..i * cap + nsel[i] as usize].to_vec())
                    .collect();
                u.sort_unstable();
                u.dedup();
                u
            })
            .collect()
    }

    /// `kernel_qsa_tile_union` names exactly the distinct blocks of each tile,
    /// ascending, with the count beside them — over a small shape, the
    /// shipped shape (2048 queries, 513-wide lists, 32768 blocks) and the
    /// bitmap's full width.
    #[test]
    fn tile_union_matches_the_host_union() {
        let dev = metal_device().unwrap();
        for &(n, cap, n_blocks, t, seed) in &[
            (100usize, 17usize, 5000usize, 32usize, 1u64),
            (33, 5, 40, 32, 2),
            (2048, 513, 32768, 64, 3),
            (70, 9, 65536, 32, 4),
            // The universe is exactly 32 words and the last id sits in the
            // last bit: the tail block's case when `n_kv / ratio` is a
            // multiple of 32 and the ceiling is one past it.
            (40, 7, 1024 + 1, 32, 5),
            (40, 7, 32, 32, 6),
        ] {
            let (mut blocks, mut nsel) = lists(n, cap, n_blocks, seed);
            // Every row also names the last id, as a tail block would.
            for i in 0..n {
                let m = nsel[i] as usize;
                let last = n_blocks as u32 - 1;
                if m < cap && (m == 0 || blocks[i * cap + m - 1] != last) {
                    blocks[i * cap + m] = last;
                    nsel[i] += 1;
                }
            }
            let want = host_union(&blocks, &nsel, n, cap, t);
            let bt = Tensor::from_vec(blocks.clone(), (n, cap), &dev).unwrap();
            let nt = Tensor::from_vec(nsel.clone(), n, &dev).unwrap();
            let (u, c) = dispatch::run_qsa_tile_union(&bt, &nt, t, n_blocks).unwrap();
            let (n_tiles, cap_out) = u.dims2().unwrap();
            assert_eq!(n_tiles, n.div_ceil(t));
            let u = u.to_vec2::<u32>().unwrap();
            let c = c.to_vec1::<u32>().unwrap();
            for tile in 0..n_tiles {
                assert_eq!(
                    c[tile] as usize,
                    want[tile].len(),
                    "n {n} tile {tile}: count"
                );
                assert_eq!(
                    &u[tile][..want[tile].len()],
                    &want[tile][..],
                    "n {n} tile {tile}: union"
                );
                assert!(want[tile].len() <= cap_out);
            }
        }
    }

    /// The K/V gather, the tile mask and the query tiling each reproduce a
    /// host gather element for element, padding included.
    #[test]
    fn tile_gathers_match_host_gathers() {
        let dev = metal_device().unwrap();
        let (n, cap, n_blocks, t, ratio) = (50usize, 9usize, 60usize, 32usize, 4usize);
        let (heads, hd) = (2usize, 8usize);
        let n_kv = n_blocks * ratio + 3; // a tail: block 60 is incomplete
        let universe = n_kv.div_ceil(ratio);
        let (mut blocks, mut nsel) = lists(n, cap, universe, 7);
        // The last row names the tail block, whose last column is past n_kv.
        let m = nsel[n - 1] as usize;
        if m < cap && (m == 0 || blocks[(n - 1) * cap + m - 1] != n_blocks as u32) {
            blocks[(n - 1) * cap + m] = n_blocks as u32;
            nsel[n - 1] += 1;
        }
        let want_union = host_union(&blocks, &nsel, n, cap, t);
        let u_max = want_union.iter().map(Vec::len).max().unwrap();
        let s = (u_max * ratio).max(t).div_ceil(S_ALIGN) * S_ALIGN;
        let bt = Tensor::from_vec(blocks, (n, cap), &dev).unwrap();
        let nt = Tensor::from_vec(nsel, n, &dev).unwrap();
        let (u, c) = dispatch::run_qsa_tile_union(&bt, &nt, t, universe).unwrap();
        let n_tiles = n.div_ceil(t);

        // K/V: a head-strided view of a larger buffer, like the cache's.
        let big = |seed: u64| {
            Tensor::from_vec(
                pseudo_random(heads * (n_kv + 20) * hd, seed, -1.0, 1.0),
                (heads, n_kv + 20, hd),
                &dev,
            )
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap()
        };
        let (kb, vb) = (big(8), big(18));
        let k_all = kb.narrow(1, 0, n_kv).unwrap();
        let v_all = vb.narrow(1, 0, n_kv).unwrap();
        let (ks, vs) =
            dispatch::run_qsa_tile_gather_kv(&u, &c, &k_all, &v_all, n, t, s, ratio).unwrap();
        assert_eq!(ks.dims(), &[n_tiles, heads, s, hd]);
        let kh = k_all
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec3::<f32>()
            .unwrap();
        let vh = v_all
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec3::<f32>()
            .unwrap();
        let k4 = ks
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let v4 = vs
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for tile in 0..n_tiles {
            for h in 0..heads {
                for j in 0..s {
                    let uu = j / ratio;
                    let row = if uu < want_union[tile].len() {
                        let r = want_union[tile][uu] as usize * ratio + j % ratio;
                        if r < n_kv { r } else { 0 }
                    } else {
                        0
                    };
                    let off = ((tile * heads + h) * s + j) * hd;
                    assert_eq!(
                        &k4[off..off + hd],
                        &kh[h][row][..],
                        "k tile {tile} h {h} j {j}"
                    );
                    assert_eq!(
                        &v4[off..off + hd],
                        &vh[h][row][..],
                        "v tile {tile} h {h} j {j}"
                    );
                }
            }
        }

        // Mask: -inf/0 pattern with the tile's columns; padding rows copy row n-1.
        let mr = pseudo_random(n * n_kv, 9, 0.0, 1.0);
        let mh: Vec<f32> = mr
            .iter()
            .map(|v| if *v < 0.5 { 0.0 } else { f32::NEG_INFINITY })
            .collect();
        let mask = Tensor::from_vec(mh.clone(), (n, n_kv), &dev).unwrap();
        let tm = dispatch::run_qsa_tile_mask(&u, &c, &mask, t, s, ratio).unwrap();
        assert_eq!(tm.dims(), &[n_tiles, 1, t, s]);
        let tm = tm
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for tile in 0..n_tiles {
            for tq in 0..t {
                let row = (tile * t + tq).min(n - 1);
                for j in 0..s {
                    let uu = j / ratio;
                    let col = if uu < want_union[tile].len() {
                        Some(want_union[tile][uu] as usize * ratio + j % ratio)
                    } else {
                        None
                    };
                    let want = match col {
                        Some(c) if c < n_kv => mh[row * n_kv + c],
                        _ => f32::NEG_INFINITY,
                    };
                    assert_eq!(
                        tm[(tile * t + tq) * s + j],
                        want,
                        "mask tile {tile} q {tq} j {j}"
                    );
                }
            }
        }

        // Queries: [heads, n, hd] f32 -> [tiles, heads, t, hd] f16.
        let q = Tensor::from_vec(
            pseudo_random(heads * n * hd, 10, -3.0, 3.0),
            (heads, n, hd),
            &dev,
        )
        .unwrap();
        let qt = dispatch::run_qsa_tile_q(&q, t).unwrap();
        assert_eq!(qt.dims(), &[n_tiles, heads, t, hd]);
        let qh = q
            .to_dtype(DType::F16)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec3::<f32>()
            .unwrap();
        let qt = qt
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for tile in 0..n_tiles {
            for h in 0..heads {
                for tq in 0..t {
                    let row = (tile * t + tq).min(n - 1);
                    let off = ((tile * heads + h) * t + tq) * hd;
                    assert_eq!(
                        &qt[off..off + hd],
                        &qh[h][row][..],
                        "q tile {tile} h {h} q {tq}"
                    );
                }
            }
        }
    }

    /// The dispatch preconditions are refused, not run.
    #[test]
    fn unsupported_tile_inputs_are_refused() {
        let dev = metal_device().unwrap();
        let (blocks, nsel) = lists(10, 3, 100, 11);
        let bt = Tensor::from_vec(blocks, (10, 3), &dev).unwrap();
        let nt = Tensor::from_vec(nsel, 10, &dev).unwrap();
        assert!(dispatch::run_qsa_tile_union(&bt, &nt, 32, 100).is_ok());
        assert!(dispatch::run_qsa_tile_union(&bt, &nt, 0, 100).is_err());
        assert!(
            dispatch::run_qsa_tile_union(&bt, &nt, 32, dispatch::QSA_TILES_MAX_BLOCKS + 1).is_err()
        );
        let cpu = bt.to_device(&Device::Cpu).unwrap();
        assert!(dispatch::run_qsa_tile_union(&cpu, &nt, 32, 100).is_err());
    }
}
