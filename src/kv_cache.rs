use std::borrow::Cow;
use std::io::{Read, Write};

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use half::f16;

use crate::config::{LayerKind, XwenConfig};

/// Per-layer state. Full-attention layers preallocate K/V to `max_ctx` and grow
/// via slice_set (cache dtype f16, because sdpa runs in f16); gated-DeltaNet
/// layers keep a fixed-size recurrent state instead, in f32.
///
/// Positions are absolute. The caller drives the cache sequentially, so the
/// number of tokens stored before an `append` equals the `pos` handed to the
/// matching `attn_mask` call — this invariant is what lets `attn_mask` describe
/// the exact key ordering `append` returns without sharing extra state.
///
/// K/V are laid out `[n_kv_head, slot, head_dim]`, matching what `attention.rs`
/// feeds into sdpa after adding a leading batch dim.
pub enum LayerCache {
    /// Full attention: every past key is kept, keys returned in position order.
    Full { k: Tensor, v: Tensor, len: usize },
    /// Sliding-window attention: a ring of `window` slots keyed by `pos % window`.
    ///
    /// No Qwen 3.6 layer is of this kind — nothing in the model path constructs
    /// one. The variant and its ring machinery are retained because the SWA
    /// rollback/snapshot semantics are the non-obvious half of this module and
    /// deleting them would take their tests with them.
    Swa {
        k: Tensor,
        v: Tensor,
        len: usize,
        window: usize,
    },
    /// Gated DeltaNet: a rolling conv window plus the delta state, both f32.
    /// Unlike the two attention kinds this state is DESTROYED by each step, so a
    /// rollback cannot be a truncation — see `trail`.
    Linear {
        /// The last `conv_kernel - 1` columns of the pre-conv fused qkv stream,
        /// `[conv_kernel - 1, conv_dim]` f32.
        conv: Tensor,
        /// The delta state, `[v_heads, head_dim, head_dim]` f32: the first matrix
        /// axis contracts with k and q, the second carries the value dimension.
        delta: Tensor,
        len: usize,
        /// Per-token states recorded since `checkpoint` armed this layer, one
        /// entry per token stepped, in token order. Empty (and unrecorded) when
        /// no checkpoint is live. A rollback to a partial commit reads the entry
        /// for the last accepted token; nothing shorter reconstructs the state,
        /// because each step overwrites the last.
        trail: Vec<(Tensor, Tensor)>,
        /// Tokens the live checkpoint reserved room for, or `None` when unarmed.
        armed: Option<usize>,
    },
}

/// The per-layer state the attention-mask math reads — the attention kind, plus
/// the window for SWA. Nothing else in a `LayerCache` affects the mask, so a
/// mask built from a `MaskKind` is identical across every layer of that kind
/// and can be hoisted out of the per-layer loop.
#[derive(Clone, Copy)]
pub enum MaskKind {
    Full,
    Swa { window: usize },
}

/// The host-side attention mask before it reaches the device: the additive
/// `[rows, cols]` values in row-major order.
pub struct MaskHost {
    pub data: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

/// Attention mask for `seq_len` new queries at absolute position `pos`, or None
/// when a single decode token needs no mask.
///
/// The mask is additive (0.0 to attend, -inf to block) with shape
/// `[seq_len, k_seq]`, where column `c` corresponds to the same key `append`
/// places at position `c` in the returned view. This is a pure function of
/// (kind, seq_len, pos) — it reads no cache contents — so a caller can build it
/// once per forward and share it across every layer of the same kind.
pub fn attn_mask_for(
    kind: MaskKind,
    seq_len: usize,
    pos: usize,
    device: &Device,
) -> Result<Option<Tensor>> {
    let Some(host) = attn_mask_data(kind, seq_len, pos) else {
        return Ok(None);
    };
    Ok(Some(mask_tensor(host, device)?))
}

/// Upload a host mask to `device`, unchanged.
pub fn mask_tensor(host: MaskHost, device: &Device) -> Result<Tensor> {
    let MaskHost { data, rows, cols } = host;
    Ok(Tensor::from_vec(data, (rows, cols), device)?)
}

/// The CPU half of `attn_mask_for`: the mask values, with no device involved.
/// Split out so a caller can time the fill and the upload separately; the values
/// are exactly what `attn_mask_for` uploads.
pub fn attn_mask_data(kind: MaskKind, seq_len: usize, pos: usize) -> Option<MaskHost> {
    if seq_len == 1 {
        return None;
    }
    let mask = match kind {
        MaskKind::Full => {
            // Keys 0..pos+seq_len in position order; block strictly-future keys.
            let k_seq = pos + seq_len;
            let mut data = vec![0f32; seq_len * k_seq];
            for qi in 0..seq_len {
                let q_abs = pos + qi;
                for kj in 0..k_seq {
                    if kj > q_abs {
                        data[qi * k_seq + kj] = f32::NEG_INFINITY;
                    }
                }
            }
            MaskHost {
                data,
                rows: seq_len,
                cols: k_seq,
            }
        }
        MaskKind::Swa { window: w } => {
            // Columns are the m surviving past keys (abs pos-m..pos-1) then the
            // seq new keys (abs pos..pos+seq-1). Block future keys and any key
            // older than the window (matches llama.cpp is_masked_swa STANDARD:
            // q_abs - k_abs >= n_swa is masked).
            let m = pos.min(w);
            let k_seq = m + seq_len;
            let mut data = vec![0f32; seq_len * k_seq];
            for qi in 0..seq_len {
                let q_abs = pos + qi;
                for c in 0..k_seq {
                    let k_abs = pos - m + c;
                    if k_abs > q_abs || q_abs - k_abs >= w {
                        data[qi * k_seq + c] = f32::NEG_INFINITY;
                    }
                }
            }
            MaskHost {
                data,
                rows: seq_len,
                cols: k_seq,
            }
        }
    };
    Some(mask)
}

impl LayerCache {
    /// `slots` is the INITIAL allocation of a full-attention layer, not its
    /// ceiling: `ensure_full_capacity` grows the buffers on demand, so a caller
    /// with a large context budget starts small and pays for positions only as
    /// a sequence actually reaches them. A DeltaNet layer's state is fixed-size
    /// whatever the context, so `slots` does not apply to it.
    pub fn new(cfg: &XwenConfig, il: usize, slots: usize, device: &Device) -> Result<Self> {
        match cfg.layer_kind(il) {
            LayerKind::Full => {
                let alloc = |slots: usize| {
                    Tensor::zeros((cfg.n_kv_head, slots, cfg.head_dim), DType::F16, device)
                };
                Ok(LayerCache::Full {
                    k: alloc(slots)?,
                    v: alloc(slots)?,
                    len: 0,
                })
            }
            LayerKind::Linear => Ok(LayerCache::Linear {
                conv: Tensor::zeros((cfg.conv_kernel - 1, cfg.conv_dim()), DType::F32, device)?,
                delta: Tensor::zeros(
                    (cfg.linear_v_heads, cfg.linear_head_dim, cfg.linear_head_dim),
                    DType::F32,
                    device,
                )?,
                len: 0,
                trail: Vec::new(),
                armed: None,
            }),
        }
    }

    /// The recurrent state a DeltaNet layer starts its next step from: the conv
    /// window and the delta state, as cheap handles (candle tensors are
    /// immutable, so the layer can build new states from these without copying).
    pub fn linear_state(&self) -> Result<(Tensor, Tensor)> {
        match self {
            LayerCache::Linear { conv, delta, .. } => Ok((conv.clone(), delta.clone())),
            _ => anyhow::bail!("linear_state: this layer is not a gated-DeltaNet layer"),
        }
    }

    /// Whether a live checkpoint needs this layer's per-token state trail. The
    /// DeltaNet layer asks before its scan so an unarmed prefill keeps only the
    /// final state and lets every intermediate one drop.
    pub fn linear_trail_armed(&self) -> bool {
        matches!(self, LayerCache::Linear { armed: Some(_), .. })
    }

    /// Advance a DeltaNet layer by `n_tokens`. `states` holds the state AFTER
    /// each stepped token in token order when the layer is armed, and just the
    /// final state otherwise; the last entry becomes the layer's state either
    /// way.
    pub fn advance_linear(&mut self, n_tokens: usize, states: Vec<(Tensor, Tensor)>) -> Result<()> {
        let LayerCache::Linear {
            conv,
            delta,
            len,
            trail,
            armed,
        } = self
        else {
            anyhow::bail!("advance_linear: this layer is not a gated-DeltaNet layer")
        };
        let (last_conv, last_delta) = states
            .last()
            .context("advance_linear: no state to advance to")?;
        if let Some(span) = *armed {
            anyhow::ensure!(
                states.len() == n_tokens,
                "advance_linear: an armed layer needs one state per token, got {} for {n_tokens}",
                states.len()
            );
            anyhow::ensure!(
                trail.len() + n_tokens <= span,
                "advance_linear: {n_tokens} more tokens overrun the {span}-token checkpoint span \
                 ({} already recorded)",
                trail.len()
            );
        }
        *conv = last_conv.clone();
        *delta = last_delta.clone();
        *len += n_tokens;
        if armed.is_some() {
            trail.extend(states);
        }
        Ok(())
    }

    /// Append k/v ([n_kv_head, seq, head_dim] f16) and return the effective
    /// cached (k, v) views to attend over. For SWA prefill (seq > 1) the view is
    /// ordered oldest→newest so it lines up with `attn_mask`; for single-token
    /// decode the ring is returned as-is (softmax is order-independent when no
    /// mask is applied).
    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let seq = k.dim(1)?;
        match self {
            LayerCache::Linear { .. } => {
                anyhow::bail!("kv_append: a gated-DeltaNet layer has no K/V, it advances a state")
            }
            LayerCache::Full { k: kb, v: vb, len } => {
                kb.slice_set(k, 1, *len)?;
                vb.slice_set(v, 1, *len)?;
                *len += seq;
                Ok((kb.narrow(1, 0, *len)?, vb.narrow(1, 0, *len)?))
            }
            LayerCache::Swa {
                k: kr,
                v: vr,
                len,
                window,
            } => {
                let w = *window;
                if seq == 1 {
                    let idx = *len % w;
                    kr.slice_set(k, 1, idx)?;
                    vr.slice_set(v, 1, idx)?;
                    *len += 1;
                    let view = (*len).min(w);
                    Ok((kr.narrow(1, 0, view)?, vr.narrow(1, 0, view)?))
                } else {
                    // Effective view = existing valid keys (oldest→newest) ++ new keys.
                    let eff_k = prepend_existing(kr, k, *len, w)?;
                    let eff_v = prepend_existing(vr, v, *len, w)?;
                    write_ring_multi(kr, k, *len, w)?;
                    write_ring_multi(vr, v, *len, w)?;
                    *len += seq;
                    Ok((eff_k, eff_v))
                }
            }
        }
    }

    /// Grow a full-attention layer to hold at least `needed` positions, up to
    /// the hard ceiling `cap` (the model's max_ctx). Doubles from the current
    /// allocation so a long prefill costs O(log) reallocations.
    ///
    /// The WHOLE old buffer is copied, not just the committed rows `[0, len)`:
    /// the grown buffer is then the old one plus a zeroed extension,
    /// byte-identical row for row, so every property that held of the
    /// allocation before the growth — a data-free [`LayerSnapshot::Full`]
    /// restoring by truncation, rows above a rewound `len` still holding what
    /// they held — keeps holding, with nothing to argue about which flows
    /// depend on rows past `len`. The copy is bounded by the old capacity —
    /// per layer that peaks at ~268 MB on the 27B's top doubling step
    /// (65536 → 131072) — and happens O(log) times per model lifetime; the
    /// model-level grower syncs the device after a growth pass so the old
    /// buffers actually leave the pool. No-op for the other layer kinds and
    /// for a `needed` already within the allocation.
    pub fn ensure_full_capacity(&mut self, needed: usize, cap: usize) -> Result<usize> {
        let LayerCache::Full { k, v, .. } = self else {
            return Ok(0);
        };
        let slots = k.dim(1)?;
        if needed <= slots {
            return Ok(slots);
        }
        anyhow::ensure!(
            needed <= cap,
            "kv_grow: {needed} positions exceed the {cap}-token context ceiling"
        );
        let mut grown = slots.max(1);
        while grown < needed {
            grown = grown.saturating_mul(2);
        }
        let grown = grown.min(cap);
        let (n_kv, _, head_dim) = k.dims3()?;
        let device = k.device().clone();
        let nk = Tensor::zeros((n_kv, grown, head_dim), DType::F16, &device)?;
        let nv = Tensor::zeros((n_kv, grown, head_dim), DType::F16, &device)?;
        // The whole old tensor is contiguous, so this is a real blit into the
        // fresh allocation (not the shallow-clone trap `materialize` guards
        // against — that bites `contiguous()` on views, and this is no view).
        nk.slice_set(k, 1, 0)?;
        nv.slice_set(v, 1, 0)?;
        *k = nk;
        *v = nv;
        Ok(grown)
    }

    /// Convenience wrapper over `attn_mask_for` for a single cache. Production
    /// prefill hoists the build out of the per-layer loop and calls the free
    /// function once per kind; single-cache callers (tests, benches) use this.
    /// See `attn_mask_for` for the mask semantics.
    pub fn attn_mask(&self, seq_len: usize, pos: usize) -> Result<Option<Tensor>> {
        attn_mask_for(self.mask_kind(), seq_len, pos, &self.device())
    }

    /// The mask kind for this cache — the only per-layer state the mask reads.
    /// A DeltaNet layer answers `Full` because it has no mask at all and no
    /// window: the model dispatches on layer kind long before an attention mask
    /// is wanted, so nothing consumes this on that path.
    pub fn mask_kind(&self) -> MaskKind {
        match self {
            LayerCache::Full { .. } | LayerCache::Linear { .. } => MaskKind::Full,
            LayerCache::Swa { window, .. } => MaskKind::Swa { window: *window },
        }
    }

    /// The device the cache tensors live on.
    pub fn device(&self) -> Device {
        match self {
            LayerCache::Full { k, .. } | LayerCache::Swa { k, .. } => k.device().clone(),
            LayerCache::Linear { conv, .. } => conv.device().clone(),
        }
    }

    /// Return the layer to its empty state. An attention layer only has to
    /// forget its length — its slots are reclaimed by the next append — but a
    /// DeltaNet layer's state IS the history, so it is zeroed for real.
    pub fn reset(&mut self) -> Result<()> {
        match self {
            LayerCache::Full { len, .. } => *len = 0,
            LayerCache::Swa { len, .. } => *len = 0,
            LayerCache::Linear {
                conv,
                delta,
                len,
                trail,
                armed,
            } => {
                *conv = conv.zeros_like()?;
                *delta = delta.zeros_like()?;
                *len = 0;
                trail.clear();
                *armed = None;
            }
        }
        Ok(())
    }

    /// Number of positions currently stored (absolute count of appends, the same
    /// `len` both variants advance). All layers are driven in lockstep, so this
    /// is identical across every cache after a shared forward.
    pub fn len(&self) -> usize {
        match self {
            LayerCache::Full { len, .. }
            | LayerCache::Swa { len, .. }
            | LayerCache::Linear { len, .. } => *len,
        }
    }

    /// Whether this is a full-attention layer, i.e. the kind whose data has to
    /// travel in a `HostFullKv` (SWA rings travel in a `HostSnapshot` instead).
    pub fn is_full(&self) -> bool {
        matches!(self, LayerCache::Full { .. })
    }

    /// Host image of a full-attention layer's committed rows `[0, len)`: raw f16
    /// little-endian bytes for K and V, row-major `(n_kv_head, len, head_dim)`.
    ///
    /// SWA layers reject this: their state is the whole ring in ring order, which
    /// only `snapshot` captures.
    pub fn export_full_rows(&self) -> Result<(Vec<u8>, Vec<u8>)> {
        match self {
            LayerCache::Full { k, v, len } => {
                if *len == 0 {
                    return Ok((Vec::new(), Vec::new()));
                }
                Ok((f16_rows(k, *len)?, f16_rows(v, *len)?))
            }
            LayerCache::Swa { .. } | LayerCache::Linear { .. } => {
                anyhow::bail!(
                    "kv_export: only a full-attention layer has a row image; the others \
                               travel in a snapshot"
                )
            }
        }
    }

    /// Upload `pos` rows of K/V — raw f16 little-endian bytes, row-major
    /// `(n_kv_head, pos, head_dim)`, as produced by `export_full_rows` — into
    /// slots `[0, pos)` and set the length to `pos`. Slots past `pos` keep
    /// whatever they held; nothing reads them, and the next append reclaims
    /// them (same regime as `restore`'s truncation).
    pub fn import_full_rows(&mut self, k_bytes: &[u8], v_bytes: &[u8], pos: usize) -> Result<()> {
        match self {
            LayerCache::Full { k, v, len } => {
                let (n_kv, slots, head_dim) = k.dims3()?;
                anyhow::ensure!(
                    pos <= slots,
                    "kv_import: pos {pos} exceeds the {slots} allocated slots"
                );
                let want = f16_bytes_for(&[n_kv, pos, head_dim])?;
                anyhow::ensure!(
                    k_bytes.len() == want && v_bytes.len() == want,
                    "kv_import: expected {want} K and V bytes for {pos} positions, got {} and {}",
                    k_bytes.len(),
                    v_bytes.len()
                );
                if pos > 0 {
                    let device = k.device().clone();
                    let shape = [n_kv, pos, head_dim];
                    let ki = f16_tensor(k_bytes, &shape, &device)?;
                    let vi = f16_tensor(v_bytes, &shape, &device)?;
                    k.slice_set(&ki, 1, 0)?;
                    v.slice_set(&vi, 1, 0)?;
                }
                *len = pos;
                Ok(())
            }
            LayerCache::Swa { .. } | LayerCache::Linear { .. } => {
                anyhow::bail!(
                    "kv_import: only a full-attention layer takes a row image; the \
                               others are restored from a snapshot"
                )
            }
        }
    }

    /// Snapshot the state a `span`-token verify forward is about to clobber, so
    /// `rollback` can restore it on a partial accept. Taken BEFORE the verify
    /// forward, at the current length `len0`.
    ///
    /// Full-attention layers write to fresh absolute slots [len0, len0+span) and
    /// never overwrite live data, so their rollback is a pure length truncation —
    /// no data snapshot (`LayerCheckpoint::Full`). SWA-ring layers overwrite the
    /// slots positions [len0, len0+span) map to, evicting whatever those slots
    /// held; the snapshot deep-copies that pre-verify K/V so rejected slots can be
    /// put back byte-for-byte.
    /// Gated-DeltaNet layers take neither route: their state is overwritten by
    /// every step and no image of a single moment reconstructs an intermediate
    /// one, so the checkpoint ARMS the layer and the verify forward records the
    /// state after each token as it goes (llama.cpp keeps the equivalent as K
    /// most-recent-first snapshot slots). That trail is what a partial-accept
    /// rollback reads, which is why this takes `&mut self`.
    pub fn checkpoint(&mut self, span: usize) -> Result<LayerCheckpoint> {
        match self {
            LayerCache::Full { .. } => Ok(LayerCheckpoint::Full),
            LayerCache::Swa { k, v, len, window } => {
                let (len0, w) = (*len, *window);
                Ok(LayerCheckpoint::Swa {
                    k: snapshot_ring(k, len0, span, w)?,
                    v: snapshot_ring(v, len0, span, w)?,
                })
            }
            LayerCache::Linear {
                conv,
                delta,
                trail,
                armed,
                ..
            } => {
                let base = (materialize(conv)?, materialize(delta)?);
                trail.clear();
                *armed = Some(span);
                Ok(LayerCheckpoint::Linear {
                    conv: base.0,
                    delta: base.1,
                })
            }
        }
    }

    /// Deep-copy whatever a later `restore` needs to put this layer back to its
    /// current length. Full-attention layers carry no data: they write each
    /// position to its own absolute slot, so every position below a restore
    /// point still holds its original key. SWA rings are copied whole — the
    /// slots older positions occupy are overwritten as generation runs past
    /// them, so nothing short of the full ring reconstructs the state.
    pub fn snapshot(&self) -> Result<LayerSnapshot> {
        match self {
            LayerCache::Full { .. } => Ok(LayerSnapshot::Full),
            LayerCache::Swa { k, v, window, .. } => Ok(LayerSnapshot::Swa {
                k: materialize(k)?,
                v: materialize(v)?,
                window: *window,
            }),
            LayerCache::Linear { conv, delta, .. } => Ok(LayerSnapshot::Linear {
                conv: materialize(conv)?,
                delta: materialize(delta)?,
            }),
        }
    }

    /// Put this layer back to the state `snap` captured, at `pos` committed
    /// tokens. Full layers only shorten (their slots past `pos` are stale and
    /// reclaimed by the next append); SWA rings take back their captured
    /// contents byte-for-byte, which covers every ring regime — pre-wraparound,
    /// wrapped, and `pos` below the window — because the whole ring is stored.
    pub fn restore(&mut self, snap: &LayerSnapshot, pos: usize) -> Result<()> {
        // Asked before anything is written, and asked again by whoever wants to know
        // whether a restore WOULD work — a caller that has to give the cache away before
        // it can try needs the answer in advance, and two copies of these rules would
        // drift.
        self.check_restorable(snap, pos)?;
        match (self, snap) {
            (LayerCache::Full { len, .. }, LayerSnapshot::Full) => {
                *len = pos;
                Ok(())
            }
            (
                LayerCache::Swa {
                    k: kr, v: vr, len, ..
                },
                LayerSnapshot::Swa { k, v, .. },
            ) => {
                kr.slice_set(k, 1, 0)?;
                vr.slice_set(v, 1, 0)?;
                *len = pos;
                Ok(())
            }
            (
                LayerCache::Linear {
                    conv: c,
                    delta: d,
                    len,
                    trail,
                    armed,
                },
                LayerSnapshot::Linear { conv, delta },
            ) => {
                *c = materialize(conv)?;
                *d = materialize(delta)?;
                *len = pos;
                // A restore replaces the history the trail was recorded against.
                trail.clear();
                *armed = None;
                Ok(())
            }
            _ => anyhow::bail!("kv_restore: snapshot kind does not match cache kind"),
        }
    }

    /// Everything [`LayerCache::restore`] requires of `snap`, decided without touching
    /// this layer.
    ///
    /// The restore writes layer by layer, so a stack that disagrees at layer five has
    /// already had four layers overwritten by the time it says so. That is why this is
    /// separable: a caller can ask the whole stack first and decline the restore rather
    /// than discover it halfway.
    pub fn check_restorable(&self, snap: &LayerSnapshot, pos: usize) -> Result<()> {
        match (self, snap) {
            (LayerCache::Full { k, .. }, LayerSnapshot::Full) => {
                let slots = k.dim(1)?;
                anyhow::ensure!(
                    pos <= slots,
                    "kv_restore: snapshot pos {pos} exceeds the {slots} allocated slots"
                );
                Ok(())
            }
            (
                LayerCache::Swa { window, .. },
                LayerSnapshot::Swa {
                    k,
                    v,
                    window: snap_window,
                },
            ) => {
                anyhow::ensure!(
                    window == snap_window,
                    "kv_restore: snapshot window {snap_window} does not match cache window {window}"
                );
                // The snapshot has to cover the whole ring. A shorter one would
                // slice_set only its leading slots and leave the rest holding
                // whatever the cache last wrote there — silently mixing another
                // conversation's keys into this one.
                anyhow::ensure!(
                    k.dim(1)? == *window && v.dim(1)? == *window,
                    "kv_restore: snapshot covers {} K and {} V slots, not the whole {window}-slot \
                     ring",
                    k.dim(1)?,
                    v.dim(1)?
                );
                Ok(())
            }
            (
                LayerCache::Linear {
                    conv: c, delta: d, ..
                },
                LayerSnapshot::Linear { conv, delta },
            ) => {
                anyhow::ensure!(
                    c.dims() == conv.dims() && d.dims() == delta.dims(),
                    "kv_restore: snapshot state is {:?}/{:?}, the cache holds {:?}/{:?}",
                    conv.dims(),
                    delta.dims(),
                    c.dims(),
                    d.dims()
                );
                Ok(())
            }
            _ => anyhow::bail!("kv_restore: snapshot kind does not match cache kind"),
        }
    }

    /// Roll a verify forward back to `len0 + commit` (0 <= commit <= span), where
    /// `len0`/`span` are the checkpoint's. Full layers truncate the length,
    /// discarding the rejected tail (its stale slots sit past `len` and are
    /// overwritten by the next append). SWA layers restore the ring slots the
    /// rejected positions [len0+commit, len0+span) overwrote to their pre-verify
    /// contents, then lower the length. After this, the cache is byte-identical to
    /// one that only ever appended the committed prefix.
    pub fn rollback(
        &mut self,
        ckpt: &LayerCheckpoint,
        len0: usize,
        span: usize,
        commit: usize,
    ) -> Result<()> {
        match (self, ckpt) {
            (LayerCache::Full { len, .. }, LayerCheckpoint::Full) => {
                *len = len0 + commit;
                Ok(())
            }
            (
                LayerCache::Swa {
                    k: kr,
                    v: vr,
                    len,
                    window,
                },
                LayerCheckpoint::Swa { k: sk, v: sv },
            ) => {
                let w = *window;
                let rejected = span - commit;
                if rejected > 0 {
                    // The snapshot is stored in position order [len0, len0+span);
                    // the rejected tail is its suffix [commit, span). Write it back
                    // to the ring at the rejected positions' slots (wrap-aware).
                    let start = (len0 + commit) % w;
                    write_wrapped(kr, &sk.narrow(1, commit, rejected)?, start, w)?;
                    write_wrapped(vr, &sv.narrow(1, commit, rejected)?, start, w)?;
                }
                *len = len0 + commit;
                Ok(())
            }
            (
                LayerCache::Linear {
                    conv: c,
                    delta: d,
                    len,
                    trail,
                    armed,
                },
                LayerCheckpoint::Linear { conv, delta },
            ) => {
                anyhow::ensure!(
                    trail.len() == span,
                    "kv_rollback: the verify forward recorded {} of {span} states; a DeltaNet \
                     layer cannot be rolled back to a token it never stepped through",
                    trail.len()
                );
                // Commit 0 is the checkpoint's own state; commit n is the state
                // after the nth accepted token, which the trail holds at n-1.
                let (nc, nd) = match commit {
                    0 => (conv, delta),
                    n => {
                        let (c, d) = &trail[n - 1];
                        (c, d)
                    }
                };
                *c = materialize(nc)?;
                *d = materialize(nd)?;
                *len = len0 + commit;
                trail.clear();
                *armed = None;
                Ok(())
            }
            _ => anyhow::bail!("kv_rollback: checkpoint kind does not match cache kind"),
        }
    }
}

/// Per-layer rollback state produced by `LayerCache::checkpoint`. Full layers
/// carry no data (truncation restores them); SWA layers carry the pre-verify K/V
/// of the ring slots the verify span overwrites, deep-copied into independent
/// storage and laid out in position order `[n_kv_head, span, head_dim]`.
pub enum LayerCheckpoint {
    Full,
    Swa {
        k: Tensor,
        v: Tensor,
    },
    /// A DeltaNet layer's state at the checkpoint, which is only ever the
    /// commit-0 answer; every other commit comes from the trail the armed layer
    /// recorded during the verify forward.
    Linear {
        conv: Tensor,
        delta: Tensor,
    },
}

/// Per-layer restore state produced by `LayerCache::snapshot`. Full layers carry
/// no data (shortening restores them); SWA layers carry a deep copy of their
/// whole ring, in ring order, alongside the window it was taken at.
pub enum LayerSnapshot {
    Full,
    Swa {
        k: Tensor,
        v: Tensor,
        window: usize,
    },
    /// A DeltaNet layer's whole recurrent state, deep-copied. Fixed size — it
    /// does not grow with the conversation.
    Linear {
        conv: Tensor,
        delta: Tensor,
    },
}

/// A restore point for the whole cache, taken between forwards: the committed
/// token count plus one `LayerSnapshot` per layer, in layer order. Restoring it
/// returns the cache to exactly the state it had when the snapshot was taken, so
/// a conversation can be rewound to a turn boundary and continued differently.
///
/// Cost is the SWA rings alone — 36 layers x 2 x [8, 512, 128] f16 is ~72 MiB on
/// the shipped checkpoint, independent of how long the conversation is.
pub struct CacheSnapshot {
    pos: usize,
    layers: Vec<LayerSnapshot>,
}

impl CacheSnapshot {
    pub(crate) fn new(pos: usize, layers: Vec<LayerSnapshot>) -> Self {
        Self { pos, layers }
    }

    /// Committed token count this snapshot restores to.
    pub fn pos(&self) -> usize {
        self.pos
    }

    pub(crate) fn layers(&self) -> &[LayerSnapshot] {
        &self.layers
    }

    /// Copy this snapshot down to host RAM. The device tensors are read out into
    /// owned byte buffers, so the result outlives the GPU cache it came from and
    /// several of them can be held at once (one per parked conversation) without
    /// consuming device memory.
    pub fn to_host(&self) -> Result<HostSnapshot> {
        let layers = self
            .layers
            .iter()
            .map(|layer| match layer {
                LayerSnapshot::Full => Ok(HostLayerSnapshot::Full),
                LayerSnapshot::Swa { k, v, window } => Ok(HostLayerSnapshot::Swa {
                    k: f16_bytes(k)?,
                    v: f16_bytes(v)?,
                    shape: k.dims3()?,
                    window: *window,
                }),
                LayerSnapshot::Linear { conv, delta } => Ok(HostLayerSnapshot::Linear {
                    conv: f32_bytes(conv)?,
                    delta: f32_bytes(delta)?,
                    conv_shape: conv.dims2()?,
                    delta_shape: delta.dims3()?,
                }),
            })
            .collect::<Result<_>>()?;
        HostSnapshot::new(self.pos, layers)
    }
}

/// Little-endian write of a `u32` — the width the persisted records use for tags
/// and versions. Every count, position and byte length is a `u64` instead, so a
/// record's framing never depends on the writing host's word size.
pub(crate) fn write_u32(w: &mut impl Write, v: u32) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

pub(crate) fn write_u64(w: &mut impl Write, v: u64) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

/// A count, position or byte length, widened to the `u64` the format stores.
pub(crate) fn write_count(w: &mut impl Write, v: usize) -> std::io::Result<()> {
    write_u64(w, v as u64)
}

/// A byte plane: its `u64` length, then the bytes.
pub(crate) fn write_plane(w: &mut impl Write, bytes: &[u8]) -> std::io::Result<()> {
    write_count(w, bytes.len())?;
    w.write_all(bytes)
}

/// Bytes a `write_plane` of a `len`-byte plane occupies, for the record length a
/// container directory has to declare before the bodies are written.
pub(crate) const fn plane_bytes(len: usize) -> usize {
    size_of::<u64>() + len
}

/// What a persisted record may claim before it is refused outright, in the units
/// the records store: token counts, layer counts, and the attention geometry a
/// plane's shape is built from.
///
/// These exist because a file's length is not a bound on what reading it costs. A
/// sparse `.lkv` — one `truncate` away for anything that can write the directory —
/// can declare gigabytes the filesystem never stored, and a reader that sizes a
/// buffer from the declaration allocates for real. An allocation that fails aborts
/// the process, which no `catch_unwind` can turn back into a cache miss, so every
/// count is checked against a number no model this code serves could produce
/// BEFORE it reaches a `Vec`. They are deliberately loose: ten times the shipped
/// checkpoint's geometry is still four orders of magnitude below anything
/// dangerous.
pub(crate) const MAX_STORED_TOKENS: usize = 1 << 20;
pub(crate) const MAX_STORED_LAYERS: usize = 64;
pub(crate) const MAX_STORED_SNAPSHOTS: usize = 64;
pub(crate) const MAX_STORED_KV_HEADS: usize = 64;
pub(crate) const MAX_STORED_HEAD_DIM: usize = 512;
pub(crate) const MAX_STORED_WINDOW: usize = 4096;
pub(crate) const MAX_STORED_POS: usize = 1 << 20;

/// What one K or V plane may occupy. The per-dimension caps above bound each field
/// on its own and their PRODUCT not at all: 64 heads x 2^20 positions x 512 dims of
/// f16 clears every one of them and asks for 64 GiB, which `vec![0u8; want]` would
/// try to satisfy and the allocator would abort the process over.
///
/// The largest plane this checkpoint can legitimately produce is a full-attention
/// layer at the trained context: 8 KV heads x 262144 positions x 128 dims of f16, or
/// 512 MiB. This is twice that.
pub(crate) const MAX_STORED_PLANE_BYTES: usize = 1 << 30;

/// Refuse a stored count that no checkpoint could have produced, naming both the
/// value and the bound so a rejected file can be told from a misconfigured one.
pub(crate) fn plausible(what: &str, value: usize, cap: usize) -> Result<usize> {
    anyhow::ensure!(
        value <= cap,
        "record: stored {what} is {value}, past the {cap} this build will read"
    );
    Ok(value)
}

/// Reader for ONE framed record body, carrying the byte budget the container's
/// directory declared for it. Every read is charged against that budget before it
/// allocates, so a length field corrupted to a huge value costs an error rather
/// than a multi-gigabyte allocation, and `finish` catches a body that framed
/// fewer bytes than it declared. Nothing here decides a record is COHERENT —
/// that stays the records' own validating constructors' job.
pub(crate) struct RecordReader<R> {
    inner: R,
    remaining: u64,
}

impl<R: Read> RecordReader<R> {
    pub(crate) fn new(inner: R, len: u64) -> Self {
        Self {
            inner,
            remaining: len,
        }
    }

    /// Charge `n` bytes against the record's budget, refusing a read that would
    /// run past the end the directory declared.
    fn charge(&mut self, n: u64) -> Result<()> {
        self.remaining = self.remaining.checked_sub(n).with_context(|| {
            format!(
                "record: a {n}-byte read runs past the record's framed end ({} bytes left)",
                self.remaining
            )
        })?;
        Ok(())
    }

    pub(crate) fn u32(&mut self) -> Result<u32> {
        self.charge(size_of::<u32>() as u64)?;
        let mut buf = [0u8; size_of::<u32>()];
        self.inner.read_exact(&mut buf)?;
        Ok(u32::from_le_bytes(buf))
    }

    pub(crate) fn u64(&mut self) -> Result<u64> {
        self.charge(size_of::<u64>() as u64)?;
        let mut buf = [0u8; size_of::<u64>()];
        self.inner.read_exact(&mut buf)?;
        Ok(u64::from_le_bytes(buf))
    }

    /// A stored count or position, read back as a `usize` and refused when it does
    /// not fit one.
    pub(crate) fn count(&mut self) -> Result<usize> {
        let v = self.u64()?;
        usize::try_from(v)
            .with_context(|| format!("record: stored count {v} does not fit a usize on this host"))
    }

    /// A byte plane as `write_plane` wrote it, of exactly the length the shape
    /// already read implies.
    ///
    /// The length is compared BEFORE the buffer is sized, which is the difference
    /// between allocating what a plausible shape needs and allocating what the file
    /// claims: a declared length is untrusted input, while `want` was derived from
    /// a shape whose every field passed [`plausible`]. The records' constructors
    /// check the same equality afterwards, and keep doing so — this is the check
    /// that has to happen before the allocation, not instead of it.
    pub(crate) fn plane_of(&mut self, want: usize) -> Result<Vec<u8>> {
        self.plane_of_up_to(want, MAX_STORED_PLANE_BYTES)
    }

    /// `plane_of` against a caller-chosen size cap, for planes whose legitimate
    /// extent is a different order of magnitude from a K/V plane's — a DeltaNet
    /// recurrent state is fixed-size and megabytes, not context-scaled.
    pub(crate) fn plane_of_up_to(&mut self, want: usize, cap: usize) -> Result<Vec<u8>> {
        // The shape's own fields each passed a cap; their product did not, and it is
        // the product that gets allocated.
        plausible("plane byte count", want, cap)?;
        let len = self.u64()?;
        anyhow::ensure!(
            len == want as u64,
            "record: plane declares {len} bytes where its shape needs {want}"
        );
        self.charge(len)?;
        let mut buf = vec![0u8; want];
        self.inner.read_exact(&mut buf)?;
        Ok(buf)
    }

    /// Every byte the directory declared for this record has been consumed. A
    /// record that read short means its own framing disagrees with the
    /// directory's, so nothing it produced can be trusted.
    pub(crate) fn finish(self) -> Result<()> {
        anyhow::ensure!(
            self.remaining == 0,
            "record: {} framed bytes went unread",
            self.remaining
        );
        Ok(())
    }
}

/// Layer-kind tags inside a persisted `HostSnapshot`. Full-attention layers carry
/// no ring, but they are still written so the record's layer count matches the
/// stack it came from.
const LAYER_FULL: u32 = 0;
const LAYER_SWA: u32 = 1;
const LAYER_LINEAR: u32 = 2;

/// What one DeltaNet state plane may claim. The largest a shipped checkpoint
/// produces is the 27B's delta state: 48 V-heads x 128 x 128 f32, 3 MiB. This is
/// well clear of that and still nowhere near an allocation worth worrying about.
pub(crate) const MAX_STORED_STATE_BYTES: usize = 1 << 26;
/// Caps on the DeltaNet geometry a record may declare, each an order of
/// magnitude past the shipped checkpoints (48 V-heads, head dim 128, conv width
/// 10240 over 3 carried columns).
pub(crate) const MAX_STORED_V_HEADS: usize = 512;
pub(crate) const MAX_STORED_CONV_DIM: usize = 1 << 17;
pub(crate) const MAX_STORED_CONV_TAIL: usize = 16;

/// Host-RAM mirror of a `CacheSnapshot`: the SWA rings as raw bytes plus the
/// shape and window they were captured at. No candle types appear in the stored
/// data, so a snapshot can be parked in host memory while another conversation
/// owns the GPU cache stack.
///
/// This is also the record a persistent (on-disk) cache slot is written from, so
/// the layout is part of the contract, not an implementation detail: per SWA
/// layer, K then V, little-endian f16, row-major `(n_kv_head, window, head_dim)`
/// in RING order (slot index = absolute position % window, which is what
/// `LayerCache::restore` writes back verbatim). Full-attention layers store no
/// data — they restore by truncation, and their rows travel separately in a
/// `HostFullKv`. `write_to`/`read_from` frame exactly that.
pub struct HostSnapshot {
    /// Committed token count this snapshot restores to.
    pub pos: usize,
    layers: Vec<HostLayerSnapshot>,
}

/// One layer's host image inside a `HostSnapshot`; see that type for the byte
/// layout. Crate-visible so a `HostSnapshot` can be assembled from parts, but the
/// only way to turn parts into a snapshot is `HostSnapshot::new`, which validates
/// them.
pub(crate) enum HostLayerSnapshot {
    Full,
    Swa {
        k: Vec<u8>,
        v: Vec<u8>,
        /// `(n_kv_head, window, head_dim)` of the ring the bytes came from.
        shape: (usize, usize, usize),
        window: usize,
    },
    /// A DeltaNet layer's recurrent state: little-endian f32, row-major, the conv
    /// window `(conv_kernel - 1, conv_dim)` then the delta state
    /// `(v_heads, head_dim, head_dim)`.
    Linear {
        conv: Vec<u8>,
        delta: Vec<u8>,
        conv_shape: (usize, usize),
        delta_shape: (usize, usize, usize),
    },
}

/// The layout checks a DeltaNet layer must pass before anything is built from
/// it: the planes hold exactly the bytes their shapes describe, and the delta
/// state is square in the two matrix axes (k and v head dims are equal on every
/// Qwen 3.6 conversion, and the recurrence contracts one against the other).
fn check_linear_layer(
    idx: usize,
    conv: &[u8],
    delta: &[u8],
    conv_shape: (usize, usize),
    delta_shape: (usize, usize, usize),
) -> Result<()> {
    let (tail, conv_dim) = conv_shape;
    let (v_heads, d_k, d_v) = delta_shape;
    anyhow::ensure!(
        tail > 0 && conv_dim > 0 && v_heads > 0 && d_k > 0 && d_v > 0,
        "kv_host_snapshot: layer {idx} declares an empty recurrent state \
         ({conv_shape:?} / {delta_shape:?})"
    );
    let want_conv = f32_bytes_for(&[tail, conv_dim])?;
    let want_delta = f32_bytes_for(&[v_heads, d_k, d_v])?;
    anyhow::ensure!(
        conv.len() == want_conv && delta.len() == want_delta,
        "kv_host_snapshot: layer {idx} expected {want_conv} conv and {want_delta} delta bytes \
         for {conv_shape:?} / {delta_shape:?}, got {} and {}",
        conv.len(),
        delta.len()
    );
    Ok(())
}

/// The layout checks a ring layer must pass before anything is built from it: the
/// stored slot count IS the stored window, and the planes hold exactly the bytes
/// that shape describes. A record whose slot count falls short of its window would
/// rebuild a snapshot covering only the leading slots, leaving the rest holding
/// whatever was in the ring before — the previous conversation's keys.
fn check_ring_layer(
    idx: usize,
    k: &[u8],
    v: &[u8],
    shape: (usize, usize, usize),
    window: usize,
) -> Result<()> {
    let (n_kv, slots, head_dim) = shape;
    anyhow::ensure!(
        slots == window,
        "kv_host_snapshot: layer {idx} shape {shape:?} holds {slots} slots but the window is \
         {window}"
    );
    let want = f16_bytes_for(&[n_kv, slots, head_dim])?;
    anyhow::ensure!(
        k.len() == want && v.len() == want,
        "kv_host_snapshot: layer {idx} expected {want} K and V bytes for shape {shape:?}, got {} \
         and {}",
        k.len(),
        v.len()
    );
    Ok(())
}

impl HostSnapshot {
    /// Assemble a snapshot image, checking every ring layer against the shape and
    /// window it claims. Both producers go through here — a device readback and a
    /// persisted record — so a record read off disk is held to exactly the
    /// invariants an in-process one is.
    pub(crate) fn new(pos: usize, layers: Vec<HostLayerSnapshot>) -> Result<Self> {
        for (idx, layer) in layers.iter().enumerate() {
            if let HostLayerSnapshot::Swa {
                k,
                v,
                shape,
                window,
            } = layer
            {
                check_ring_layer(idx, k, v, *shape, *window)?;
            }
            if let HostLayerSnapshot::Linear {
                conv,
                delta,
                conv_shape,
                delta_shape,
            } = layer
            {
                check_linear_layer(idx, conv, delta, *conv_shape, *delta_shape)?;
            }
        }
        Ok(Self { pos, layers })
    }

    /// Frame this image into `w`, in the layout the type's doc comment fixes: the
    /// position, the layer count, then per layer a kind tag and — for a ring — the
    /// shape, the window and the K/V planes.
    pub(crate) fn write_to(&self, w: &mut impl Write) -> std::io::Result<()> {
        write_count(w, self.pos)?;
        write_count(w, self.layers.len())?;
        for layer in &self.layers {
            match layer {
                HostLayerSnapshot::Full => write_u32(w, LAYER_FULL)?,
                HostLayerSnapshot::Swa {
                    k,
                    v,
                    shape,
                    window,
                } => {
                    write_u32(w, LAYER_SWA)?;
                    let (n_kv, slots, head_dim) = *shape;
                    write_count(w, n_kv)?;
                    write_count(w, slots)?;
                    write_count(w, head_dim)?;
                    write_count(w, *window)?;
                    write_plane(w, k)?;
                    write_plane(w, v)?;
                }
                HostLayerSnapshot::Linear {
                    conv,
                    delta,
                    conv_shape,
                    delta_shape,
                } => {
                    write_u32(w, LAYER_LINEAR)?;
                    write_count(w, conv_shape.0)?;
                    write_count(w, conv_shape.1)?;
                    write_count(w, delta_shape.0)?;
                    write_count(w, delta_shape.1)?;
                    write_count(w, delta_shape.2)?;
                    write_plane(w, conv)?;
                    write_plane(w, delta)?;
                }
            }
        }
        Ok(())
    }

    /// Bytes `write_to` produces, for the container directory that has to declare
    /// this record's length before the planes are streamed out.
    pub(crate) fn serialized_len(&self) -> usize {
        let mut n = 2 * size_of::<u64>();
        for layer in &self.layers {
            n += size_of::<u32>();
            match layer {
                HostLayerSnapshot::Full => {}
                HostLayerSnapshot::Swa { k, v, .. } => {
                    n += 4 * size_of::<u64>() + plane_bytes(k.len()) + plane_bytes(v.len());
                }
                HostLayerSnapshot::Linear { conv, delta, .. } => {
                    n += 5 * size_of::<u64>() + plane_bytes(conv.len()) + plane_bytes(delta.len());
                }
            }
        }
        n
    }

    /// Read back a `write_to` body of `len` bytes. Every field is untrusted input:
    /// the planes are bounded by the record's framed length as they are read, and
    /// the result is assembled through `new`, so a shape that does not describe the
    /// bytes fails here rather than in a later upload.
    pub(crate) fn read_from(src: impl Read, len: u64) -> Result<Self> {
        let mut r = RecordReader::new(src, len);
        let pos = plausible("snapshot position", r.count()?, MAX_STORED_POS)?;
        let layer_count = plausible("layer count", r.count()?, MAX_STORED_LAYERS)?;
        // Grown rather than reserved: the count is capped above, but a reader that
        // sizes anything from a number a file chose is a reader one bad number away
        // from an allocation nothing can catch.
        let mut layers = Vec::new();
        for idx in 0..layer_count {
            match r.u32()? {
                LAYER_FULL => layers.push(HostLayerSnapshot::Full),
                LAYER_SWA => {
                    let n_kv = plausible("KV head count", r.count()?, MAX_STORED_KV_HEADS)?;
                    let slots = plausible("ring slot count", r.count()?, MAX_STORED_WINDOW)?;
                    let head_dim = plausible("head dim", r.count()?, MAX_STORED_HEAD_DIM)?;
                    let window = plausible("window", r.count()?, MAX_STORED_WINDOW)?;
                    let shape = (n_kv, slots, head_dim);
                    let plane = f16_bytes_for(&[n_kv, slots, head_dim])?;
                    let k = r.plane_of(plane)?;
                    let v = r.plane_of(plane)?;
                    layers.push(HostLayerSnapshot::Swa {
                        k,
                        v,
                        shape,
                        window,
                    });
                }
                LAYER_LINEAR => {
                    let tail = plausible("conv tail", r.count()?, MAX_STORED_CONV_TAIL)?;
                    let conv_dim = plausible("conv width", r.count()?, MAX_STORED_CONV_DIM)?;
                    let v_heads = plausible("V-head count", r.count()?, MAX_STORED_V_HEADS)?;
                    let d_k = plausible("delta key dim", r.count()?, MAX_STORED_HEAD_DIM)?;
                    let d_v = plausible("delta value dim", r.count()?, MAX_STORED_HEAD_DIM)?;
                    let conv_shape = (tail, conv_dim);
                    let delta_shape = (v_heads, d_k, d_v);
                    let conv = r.plane_of_up_to(
                        f32_bytes_for(&[tail, conv_dim])?,
                        MAX_STORED_STATE_BYTES,
                    )?;
                    let delta = r.plane_of_up_to(
                        f32_bytes_for(&[v_heads, d_k, d_v])?,
                        MAX_STORED_STATE_BYTES,
                    )?;
                    layers.push(HostLayerSnapshot::Linear {
                        conv,
                        delta,
                        conv_shape,
                        delta_shape,
                    });
                }
                tag => anyhow::bail!("kv_host_snapshot: layer {idx} has unknown kind tag {tag}"),
            }
        }
        r.finish()?;
        Self::new(pos, layers)
    }

    /// Rebuild a device `CacheSnapshot` by uploading the stored bytes. Every
    /// layer gets freshly allocated device storage, so the result shares nothing
    /// with this image or with any cache: a subsequent `restore` blits the bytes
    /// in and later appends cannot reach back into them.
    pub fn to_snapshot(&self, device: &Device) -> Result<CacheSnapshot> {
        let layers = self
            .layers
            .iter()
            .enumerate()
            .map(|(idx, layer)| match layer {
                HostLayerSnapshot::Full => Ok(LayerSnapshot::Full),
                HostLayerSnapshot::Swa {
                    k,
                    v,
                    shape,
                    window,
                } => {
                    // Re-checked at the upload, not only where the image was
                    // assembled: this is the last point before the bytes reach a
                    // ring another conversation may still be holding.
                    check_ring_layer(idx, k, v, *shape, *window)?;
                    let (n_kv, slots, head_dim) = *shape;
                    let dims = [n_kv, slots, head_dim];
                    Ok(LayerSnapshot::Swa {
                        k: f16_tensor(k, &dims, device)?,
                        v: f16_tensor(v, &dims, device)?,
                        window: *window,
                    })
                }
                HostLayerSnapshot::Linear {
                    conv,
                    delta,
                    conv_shape,
                    delta_shape,
                } => {
                    check_linear_layer(idx, conv, delta, *conv_shape, *delta_shape)?;
                    let (tail, conv_dim) = *conv_shape;
                    let (v_heads, d_k, d_v) = *delta_shape;
                    Ok(LayerSnapshot::Linear {
                        conv: f32_tensor(conv, &[tail, conv_dim], device)?,
                        delta: f32_tensor(delta, &[v_heads, d_k, d_v], device)?,
                    })
                }
            })
            .collect::<Result<_>>()?;
        Ok(CacheSnapshot::new(self.pos, layers))
    }

    /// Everything restoring these rings into `caches` will require, decided on the host
    /// without uploading anything.
    ///
    /// The device restore needs `to_snapshot` first, which allocates a tensor per ring —
    /// so a caller asking "would this work?" would otherwise have to build the very thing
    /// it is trying to avoid committing to. The rules are the per-layer ones, applied to
    /// the stored shapes instead of the uploaded tensors.
    pub fn check_restorable(&self, caches: &[LayerCache], _pos: usize) -> Result<()> {
        anyhow::ensure!(
            self.layers.len() == caches.len(),
            "kv_host_snapshot: snapshot covers {} layers, the cache stack has {}",
            self.layers.len(),
            caches.len()
        );
        for (idx, (layer, cache)) in self.layers.iter().zip(caches).enumerate() {
            match (layer, cache) {
                // No position bound here: a full layer's allocation grows on
                // demand (`LayerCache::ensure_full_capacity`), and the row
                // import that precedes this restore is what grows it. The
                // model-level check holds `pos` to max_ctx, the real ceiling.
                (HostLayerSnapshot::Full, LayerCache::Full { .. }) => {}
                (
                    HostLayerSnapshot::Swa { shape, window, .. },
                    LayerCache::Swa {
                        window: cache_window,
                        ..
                    },
                ) => {
                    anyhow::ensure!(
                        window == cache_window,
                        "kv_host_snapshot: layer {idx} has window {window}, the cache has \
                         {cache_window}"
                    );
                    anyhow::ensure!(
                        shape.1 == *cache_window,
                        "kv_host_snapshot: layer {idx} covers {} slots, not the whole \
                         {cache_window}-slot ring",
                        shape.1
                    );
                }
                (
                    HostLayerSnapshot::Linear {
                        conv_shape,
                        delta_shape,
                        ..
                    },
                    LayerCache::Linear { conv, delta, .. },
                ) => {
                    let (tail, conv_dim) = *conv_shape;
                    let (v_heads, d_k, d_v) = *delta_shape;
                    anyhow::ensure!(
                        conv.dims() == [tail, conv_dim] && delta.dims() == [v_heads, d_k, d_v],
                        "kv_host_snapshot: layer {idx} holds a {conv_shape:?} / {delta_shape:?} \
                         state, the cache has {:?} / {:?}",
                        conv.dims(),
                        delta.dims()
                    );
                }
                _ => {
                    anyhow::bail!("kv_host_snapshot: layer {idx}'s kind does not match the cache's")
                }
            }
        }
        Ok(())
    }

    /// Host bytes this image occupies — the number to budget when deciding how
    /// many snapshots a parked conversation may keep.
    pub fn byte_len(&self) -> usize {
        self.layers
            .iter()
            .map(|layer| match layer {
                HostLayerSnapshot::Full => 0,
                HostLayerSnapshot::Swa { k, v, .. } => k.len() + v.len(),
                HostLayerSnapshot::Linear { conv, delta, .. } => conv.len() + delta.len(),
            })
            .sum()
    }
}

/// One layer's K/V byte images as `HostFullKv::prefix` hands them out: borrowed
/// from the stored image when the whole thing is wanted, owned when a shorter
/// prefix had to be gathered.
type KvBytes<'a> = (Cow<'a, [u8]>, Cow<'a, [u8]>);

/// Host-RAM image of the full-attention layers' K/V for positions `[0, pos)` —
/// the half of a conversation's cache state a `HostSnapshot` deliberately does
/// not carry. Full layers restore by truncation only while their slots still
/// hold the same tokens, so once another conversation writes over those slots
/// their rows have to have been copied out.
///
/// Like `HostSnapshot` this is an on-disk record, and its layout is part of the
/// contract: one `(K, V)` pair per full-attention layer in layer order,
/// little-endian f16, row-major `(n_kv_head, pos, head_dim)`. `write_to`/`read_from`
/// frame exactly that.
pub struct HostFullKv {
    /// Positions the image covers: rows `[0, pos)` of every full-attention layer.
    pub pos: usize,
    /// KV heads and head dim the plane bytes are laid out over. Kept in the image
    /// so it describes itself — an on-disk record cannot lean on a live config.
    n_kv_head: usize,
    head_dim: usize,
    planes: Vec<(Vec<u8>, Vec<u8>)>,
}

impl HostFullKv {
    /// Assemble an image, checking every plane against the shape it claims. The
    /// record is only self-describing if those agree, and every later read
    /// (`prefix`, `byte_len`, a future on-disk write) trusts them.
    pub(crate) fn new(
        pos: usize,
        n_kv_head: usize,
        head_dim: usize,
        planes: Vec<(Vec<u8>, Vec<u8>)>,
    ) -> Result<Self> {
        let want = f16_bytes_for(&[n_kv_head, pos, head_dim])?;
        for (idx, (k, v)) in planes.iter().enumerate() {
            anyhow::ensure!(
                k.len() == want && v.len() == want,
                "kv_host_full: layer {idx} holds {} K and {} V bytes, expected {want} for \
                 ({n_kv_head}, {pos}, {head_dim}) f16",
                k.len(),
                v.len()
            );
        }
        Ok(Self {
            pos,
            n_kv_head,
            head_dim,
            planes,
        })
    }

    /// Frame this image into `w`: the position, the shape its planes are laid out
    /// over, the plane count, then K and V per layer in layer order.
    pub(crate) fn write_to(&self, w: &mut impl Write) -> std::io::Result<()> {
        write_count(w, self.pos)?;
        write_count(w, self.n_kv_head)?;
        write_count(w, self.head_dim)?;
        write_count(w, self.planes.len())?;
        for (k, v) in &self.planes {
            write_plane(w, k)?;
            write_plane(w, v)?;
        }
        Ok(())
    }

    /// Bytes `write_to` produces, for the container directory that has to declare
    /// this record's length before the planes are streamed out.
    pub(crate) fn serialized_len(&self) -> usize {
        4 * size_of::<u64>()
            + self
                .planes
                .iter()
                .map(|(k, v)| plane_bytes(k.len()) + plane_bytes(v.len()))
                .sum::<usize>()
    }

    /// Read back a `write_to` body of `len` bytes. The shape and plane lengths are
    /// untrusted input, bounded by the record's framed length while reading and
    /// then reconciled by `new`, which is where a shape that does not describe the
    /// bytes is rejected.
    pub(crate) fn read_from(src: impl Read, len: u64) -> Result<Self> {
        let mut r = RecordReader::new(src, len);
        let pos = plausible("position count", r.count()?, MAX_STORED_POS)?;
        let n_kv_head = plausible("KV head count", r.count()?, MAX_STORED_KV_HEADS)?;
        let head_dim = plausible("head dim", r.count()?, MAX_STORED_HEAD_DIM)?;
        let plane_count = plausible("layer count", r.count()?, MAX_STORED_LAYERS)?;
        let plane = f16_bytes_for(&[n_kv_head, pos, head_dim])?;
        // Grown rather than reserved, for the reason `plausible` documents.
        let mut planes = Vec::new();
        for _ in 0..plane_count {
            let k = r.plane_of(plane)?;
            let v = r.plane_of(plane)?;
            planes.push((k, v));
        }
        r.finish()?;
        Self::new(pos, n_kv_head, head_dim, planes)
    }

    /// Check that `cache` is laid out the way this image says it is, before any
    /// of its bytes are written there. Byte counts alone cannot catch this: an
    /// image whose heads and head dim are swapped has exactly the right total
    /// size and would import as silently transposed keys.
    pub(crate) fn ensure_matches(&self, cache: &LayerCache) -> Result<()> {
        let LayerCache::Full { k, .. } = cache else {
            anyhow::bail!("kv_host_full: a sliding-window layer is restored from a snapshot")
        };
        let (n_kv_head, _, head_dim) = k.dims3()?;
        anyhow::ensure!(
            n_kv_head == self.n_kv_head && head_dim == self.head_dim,
            "kv_host_full: image is ({}, _, {}) but the cache is ({n_kv_head}, _, {head_dim})",
            self.n_kv_head,
            self.head_dim
        );
        Ok(())
    }

    /// Host bytes this image occupies — 48 KiB per token on the shipped
    /// checkpoint (12 full layers x K/V x 8 heads x 128 dims x f16).
    pub fn byte_len(&self) -> usize {
        self.planes.iter().map(|(k, v)| k.len() + v.len()).sum()
    }

    /// Number of full-attention layers covered.
    pub(crate) fn layers(&self) -> usize {
        self.planes.len()
    }

    /// K/V bytes for rows `[0, pos)` of full-attention layer `idx`, ready to hand
    /// to `LayerCache::import_full_rows`. The whole image is borrowed when `pos`
    /// is its full length; a shorter prefix is gathered because the stored layout
    /// is `(n_kv_head, self.pos, head_dim)` — each head's rows are one contiguous
    /// run, so a prefix of the positions is `n_kv_head` runs, not a prefix of the
    /// buffer. The stored image is left intact either way.
    pub(crate) fn prefix(&self, idx: usize, pos: usize) -> Result<KvBytes<'_>> {
        anyhow::ensure!(
            pos <= self.pos,
            "kv_host_full: requested pos {pos} exceeds the image's {} positions",
            self.pos
        );
        let (k, v) = self.planes.get(idx).ok_or_else(|| {
            anyhow::anyhow!(
                "kv_host_full: layer {idx} is past the image's {} layers",
                self.planes.len()
            )
        })?;
        let stride = f16_bytes_for(&[self.pos, self.head_dim])?;
        let total = f16_bytes_for(&[self.n_kv_head, self.pos, self.head_dim])?;
        anyhow::ensure!(
            k.len() == total && v.len() == total,
            "kv_host_full: layer {idx} holds {} K and {} V bytes, expected {total} for {} positions",
            k.len(),
            v.len(),
            self.pos
        );
        if pos == self.pos {
            return Ok((Cow::Borrowed(k), Cow::Borrowed(v)));
        }
        let take = f16_bytes_for(&[pos, self.head_dim])?;
        let gathered = f16_bytes_for(&[self.n_kv_head, pos, self.head_dim])?;
        let gather = |src: &[u8]| {
            let mut out = Vec::with_capacity(gathered);
            for head in 0..self.n_kv_head {
                out.extend_from_slice(&src[head * stride..head * stride + take]);
            }
            out
        };
        Ok((Cow::Owned(gather(k)), Cow::Owned(gather(v))))
    }

    /// The rows for positions `[start, end)` as an image of their own, covering
    /// `end - start` positions.
    ///
    /// A token range is never one slice of the buffer: the layout is
    /// `(n_kv_head, self.pos, head_dim)`, so a range of positions is one run per head
    /// at a stride the SOURCE's length sets. Which is why this is a gather and why
    /// the result's stride differs from this image's — the same re-striding `prefix`
    /// does, over a window that does not have to start at zero. This is what a
    /// segment's own rows are cut out of a whole conversation's with, and what a
    /// re-partition of a stored segment moves.
    pub(crate) fn range(&self, start: usize, end: usize) -> Result<Self> {
        anyhow::ensure!(
            start <= end && end <= self.pos,
            "kv_host_full: range [{start}, {end}) is not inside the image's {} positions",
            self.pos
        );
        let stride = f16_bytes_for(&[self.pos, self.head_dim])?;
        let row = f16_bytes_for(&[self.head_dim])?;
        let want = f16_bytes_for(&[self.n_kv_head, end - start, self.head_dim])?;
        let planes = self
            .planes
            .iter()
            .map(|(k, v)| {
                let gather = |src: &[u8]| {
                    let mut out = Vec::with_capacity(want);
                    for head in 0..self.n_kv_head {
                        let from = head * stride + start * row;
                        out.extend_from_slice(&src[from..from + (end - start) * row]);
                    }
                    out
                };
                (gather(k), gather(v))
            })
            .collect();
        Self::new(end - start, self.n_kv_head, self.head_dim, planes)
    }

    /// One image covering the concatenation of `spans`, in the order given.
    ///
    /// The inverse of [`HostFullKv::range`], and a gather for the same reason: the
    /// stride each span's rows are laid out at is its own length, and the result's is
    /// the total, so nothing can be appended byte-wise. This is how a chain of stored
    /// segments becomes the one image a cache slot installs.
    /// The spans are CONSUMED, and each one's buffers are released as soon as its
    /// rows have been scattered into the result. The composed image is the size of the
    /// whole chain, so holding the spans alongside it would peak at twice that — and
    /// this runs on the engine thread with the model's weights resident, where the
    /// difference is a re-prefill against an allocation failure that ends the process.
    /// Scattering (walking the spans on the outside, the heads within) is what makes
    /// releasing them early possible: the gather order would have to revisit every span
    /// once per head.
    pub(crate) fn concat(spans: Vec<Self>) -> Result<Self> {
        let first = spans
            .first()
            .ok_or_else(|| anyhow::anyhow!("kv_host_full: nothing to concatenate"))?;
        let (n_kv_head, head_dim, layers) = (first.n_kv_head, first.head_dim, first.planes.len());
        let mut pos = 0usize;
        for (idx, span) in spans.iter().enumerate() {
            anyhow::ensure!(
                span.n_kv_head == n_kv_head
                    && span.head_dim == head_dim
                    && span.planes.len() == layers,
                "kv_host_full: span {idx} is ({}, _, {}) over {} layers, the first is \
                 ({n_kv_head}, _, {head_dim}) over {layers}",
                span.n_kv_head,
                span.head_dim,
                span.planes.len(),
            );
            pos = pos
                .checked_add(span.pos)
                .ok_or_else(|| anyhow::anyhow!("kv_host_full: the spans' lengths overflow"))?;
        }
        let row = f16_bytes_for(&[head_dim])?;
        let want = f16_bytes_for(&[n_kv_head, pos, head_dim])?;
        let mut planes: Vec<(Vec<u8>, Vec<u8>)> = (0..layers)
            .map(|_| (vec![0u8; want], vec![0u8; want]))
            .collect();
        let mut at = 0usize;
        for span in spans {
            let span_pos = span.pos;
            for (idx, (sk, sv)) in span.planes.into_iter().enumerate() {
                let (k, v) = &mut planes[idx];
                for head in 0..n_kv_head {
                    let to = head * pos * row + at * row;
                    let from = head * span_pos * row;
                    let take = span_pos * row;
                    k[to..to + take].copy_from_slice(&sk[from..from + take]);
                    v[to..to + take].copy_from_slice(&sv[from..from + take]);
                }
                // `sk` and `sv` go here, one layer at a time, rather than at the end of
                // the composition.
            }
            at += span_pos;
        }
        Self::new(pos, n_kv_head, head_dim, planes)
    }
}

/// Copy the full-attention layers' committed rows out of a cache stack into one
/// host image, in layer order; SWA layers are skipped, since only a
/// `CacheSnapshot::to_host` can carry a ring. The image's shape metadata is read
/// off the cache tensors themselves, so the record describes what it was taken
/// from rather than a config that could have drifted from it. A stack with no
/// full-attention layers yields an empty image.
///
/// Split out of `XwenModel::export_full_kv` so the layer pairing is testable
/// against a hand-built stack, with no checkpoint loaded.
pub(crate) fn export_full_kv_from(caches: &[LayerCache]) -> Result<HostFullKv> {
    let mut shape: Option<(usize, usize, usize)> = None;
    let mut planes = Vec::new();
    for cache in caches {
        let LayerCache::Full { k, len, .. } = cache else {
            continue;
        };
        let (n_kv_head, _, head_dim) = k.dims3()?;
        shape.get_or_insert((*len, n_kv_head, head_dim));
        planes.push(cache.export_full_rows()?);
    }
    let (pos, n_kv_head, head_dim) = shape.unwrap_or((0, 0, 0));
    HostFullKv::new(pos, n_kv_head, head_dim, planes)
}

/// Upload `image`'s rows `[0, pos)` into a cache stack's full-attention layers,
/// pairing the image's planes with those layers in order, and set each of their
/// lengths to `pos`. SWA rings are untouched — the caller restores those from the
/// `HostSnapshot` taken at the same position.
///
/// Split out of `XwenModel::import_full_kv` for the same reason as
/// `export_full_kv_from`.
pub(crate) fn import_full_kv_into(
    caches: &mut [LayerCache],
    image: &HostFullKv,
    pos: usize,
) -> Result<()> {
    check_full_kv_importable(caches, image, pos)?;
    for (idx, cache) in caches.iter_mut().filter(|c| c.is_full()).enumerate() {
        let (k, v) = image.prefix(idx, pos)?;
        cache.import_full_rows(&k, &v, pos)?;
    }
    Ok(())
}

/// Everything [`import_full_kv_into`] requires of `image`, decided without writing a
/// byte into `caches`.
///
/// Separable for the same reason the per-layer restore check is: a caller that must hand
/// the cache away before it can attempt the import needs to know in advance whether the
/// import would take, and the answer must come from the same rules the import applies
/// rather than a second set that can drift from them.
pub(crate) fn check_full_kv_importable(
    caches: &[LayerCache],
    image: &HostFullKv,
    pos: usize,
) -> Result<()> {
    anyhow::ensure!(
        pos <= image.pos,
        "kv_host_full: requested pos {pos} exceeds the image's {} positions",
        image.pos
    );
    let full = caches.iter().filter(|c| c.is_full()).count();
    anyhow::ensure!(
        image.layers() == full,
        "kv_host_full: image covers {} full-attention layers, the cache stack has {full}",
        image.layers()
    );
    for cache in caches.iter().filter(|c| c.is_full()) {
        image.ensure_matches(cache)?;
    }
    Ok(())
}

/// A whole-model KV rollback point: the length at checkpoint time, the verify
/// span, and the per-layer snapshots (one entry per cache, in layer order).
pub struct KvCheckpoint {
    pub(crate) len0: usize,
    pub(crate) span: usize,
    pub(crate) layers: Vec<LayerCheckpoint>,
}

impl KvCheckpoint {
    pub(crate) fn new(len0: usize, span: usize, layers: Vec<LayerCheckpoint>) -> Self {
        Self { len0, span, layers }
    }

    /// The verify span this checkpoint covers (commit on rollback must be <= it).
    pub fn span(&self) -> usize {
        self.span
    }
}

/// Deep-copy the ring slots that positions [len0, len0+span) occupy, returned in
/// POSITION order as `[n_kv_head, span, head_dim]` (wrap-aware). Independent of
/// the ring's storage: on Metal at our pinned candle rev `Tensor::copy()` /
/// `contiguous()` on an already-contiguous view is a shallow Arc clone that
/// copies no data (CLAUDE.md trap), so `materialize` forces a real allocation.
fn snapshot_ring(ring: &Tensor, len0: usize, span: usize, w: usize) -> Result<Tensor> {
    let start = len0 % w;
    let seg1 = span.min(w - start);
    let block = if seg1 == span {
        ring.narrow(1, start, span)?
    } else {
        // Span straddles the ring boundary: [start, w) then [0, span - seg1).
        Tensor::cat(
            &[
                &ring.narrow(1, start, seg1)?,
                &ring.narrow(1, 0, span - seg1)?,
            ],
            1,
        )?
    };
    materialize(&block)
}

/// Force `t` into freshly allocated, independent device storage, byte-for-byte.
/// `contiguous()` first (needed for `slice_set`, and a no-op-or-shallow view is
/// fine here — we immediately blit its bytes elsewhere); then `slice_set` into a
/// fresh zero tensor, which copies the raw bits (unlike `affine`, which would
/// canonicalize -0.0).
pub(crate) fn materialize(t: &Tensor) -> Result<Tensor> {
    let src = t.contiguous()?;
    let dst = src.zeros_like()?;
    dst.slice_set(&src, 1, 0)?;
    Ok(dst)
}

/// Byte count of an f32 block with the given dimensions — `f16_bytes_for` for
/// the recurrent-state planes, and untrusted-input-safe for the same reason.
fn f32_bytes_for(dims: &[usize]) -> Result<usize> {
    dims.iter().try_fold(size_of::<f32>(), |acc, &d| {
        acc.checked_mul(d)
            .ok_or_else(|| anyhow::anyhow!("kv_host: f32 byte count for {dims:?} overflows usize"))
    })
}

/// Raw little-endian f32 bytes of `t` in row-major order — how a DeltaNet
/// recurrent state travels in a host image. `t` must own its storage; the
/// readback copies the whole allocation before applying the layout.
fn f32_bytes(t: &Tensor) -> Result<Vec<u8>> {
    anyhow::ensure!(
        t.dtype() == DType::F32,
        "kv_export: recurrent state images are f32, got {:?}",
        t.dtype()
    );
    let values = t.flatten_all()?.to_vec1::<f32>()?;
    // SAFETY: as in `f16_bytes` — `f32` has no padding or niche, and on the
    // little-endian targets this engine builds for its in-memory bytes are the
    // little-endian encoding the image format specifies.
    let bytes = unsafe {
        std::slice::from_raw_parts(
            values.as_ptr().cast::<u8>(),
            values.len() * size_of::<f32>(),
        )
    };
    Ok(bytes.to_vec())
}

/// The inverse of `f32_bytes`: a fresh `shape` tensor on `device` owning those
/// bytes.
fn f32_tensor(bytes: &[u8], shape: &[usize], device: &Device) -> Result<Tensor> {
    Ok(Tensor::from_raw_buffer(bytes, DType::F32, shape, device)?)
}

/// Byte count of an f16 block with the given dimensions, erroring rather than
/// wrapping on overflow.
///
/// The host cache images compute these products from their OWN declared shapes.
/// In process those come from real allocations and cannot overflow, but the same
/// records are the on-disk format, where the shape fields are untrusted input —
/// a wrapped product would size a buffer that a later index walks straight past.
fn f16_bytes_for(dims: &[usize]) -> Result<usize> {
    dims.iter().try_fold(size_of::<f16>(), |acc, &d| {
        acc.checked_mul(d)
            .ok_or_else(|| anyhow::anyhow!("kv_host: f16 byte count for {dims:?} overflows usize"))
    })
}

/// Raw little-endian f16 bytes of rows `[0, len)` of a `(n_kv_head, slots,
/// head_dim)` cache plane.
///
/// A short prefix is compacted on the device first, and that is not optional:
/// reading a tensor out to host copies its WHOLE storage and only then applies
/// the layout, so a `narrow` view of a max_ctx-slot cache would drag the entire
/// allocation across to fetch a few thousand rows. Rows that fill the plane have
/// nothing to trim, so exactly one device-side copy happens either way, never
/// two.
fn f16_rows(plane: &Tensor, len: usize) -> Result<Vec<u8>> {
    let slots = plane.dim(1)?;
    let view = plane.narrow(1, 0, len)?;
    if len == slots {
        f16_bytes(&view)
    } else {
        // A real strided copy into fresh storage, unlike `contiguous()`, which is
        // a shallow Arc clone on Metal when the input already passes the
        // contiguity check (CLAUDE.md trap).
        f16_bytes(&view.force_contiguous()?)
    }
}

/// Raw little-endian f16 bytes of `t` in row-major order — the host cache
/// images' storage format.
///
/// `t` must own its storage rather than view part of a larger allocation: the
/// readback copies the whole storage and only then applies the layout. Callers
/// holding a slice of a cache plane go through `f16_rows`.
fn f16_bytes(t: &Tensor) -> Result<Vec<u8>> {
    anyhow::ensure!(
        t.dtype() == DType::F16,
        "kv_export: cache images are f16, got {:?}",
        t.dtype()
    );
    let values = t.flatten_all()?.to_vec1::<f16>()?;
    // SAFETY: `f16` is `repr(transparent)` over `u16`, so a slice of them is a
    // valid slice of 2x the bytes with no padding or niche. On a little-endian
    // target — the only kind this engine builds for — those bytes ARE the
    // little-endian encoding the image format calls for, so this is byte-for-byte
    // what a per-element `to_le_bytes` loop produces and an order of magnitude
    // cheaper (one memcpy instead of half a billion two-byte appends).
    let bytes = unsafe {
        std::slice::from_raw_parts(
            values.as_ptr().cast::<u8>(),
            values.len() * size_of::<f16>(),
        )
    };
    Ok(bytes.to_vec())
}

/// The inverse of `f16_bytes`: a fresh `shape` tensor on `device` holding those
/// bytes. Storage is newly allocated and shares nothing with `bytes`, so the
/// caller may drop or reuse the image afterwards.
///
/// Both directions reinterpret f16 in host byte order, which is the
/// little-endian layout the image format specifies on every target this engine
/// builds for. A reader on a big-endian host would have to byte-swap first.
fn f16_tensor(bytes: &[u8], shape: &[usize], device: &Device) -> Result<Tensor> {
    Ok(Tensor::from_raw_buffer(bytes, DType::F16, shape, device)?)
}

/// The `min(len, window)` most recent ring entries, reordered oldest→newest.
fn ordered_existing(ring: &Tensor, len: usize, w: usize) -> Result<Option<Tensor>> {
    if len == 0 {
        Ok(None)
    } else if len <= w {
        Ok(Some(ring.narrow(1, 0, len)?))
    } else {
        // Oldest surviving key sits at ring index len % w; rotate so it leads.
        let s = len % w;
        Ok(Some(Tensor::cat(
            &[&ring.narrow(1, s, w - s)?, &ring.narrow(1, 0, s)?],
            1,
        )?))
    }
}

/// Concatenate the ordered surviving ring entries in front of the new keys.
fn prepend_existing(ring: &Tensor, new: &Tensor, len: usize, w: usize) -> Result<Tensor> {
    match ordered_existing(ring, len, w)? {
        None => Ok(new.clone()),
        Some(existing) => Ok(Tensor::cat(&[&existing, new], 1)?),
    }
}

/// Write new keys (abs positions len..len+seq) into the ring, wrap-aware,
/// leaving the ring holding the most recent `min(len+seq, w)` keys.
fn write_ring_multi(ring: &Tensor, new: &Tensor, len: usize, w: usize) -> Result<()> {
    let seq = new.dim(1)?;
    if seq >= w {
        // Only the last w new keys survive; place them at their ring indices.
        let last = new.narrow(1, seq - w, w)?;
        write_wrapped(ring, &last, (len + seq - w) % w, w)
    } else {
        write_wrapped(ring, new, len % w, w)
    }
}

/// slice_set `src` (length <= w) into `ring` starting at `start`, splitting at
/// the ring boundary when it wraps.
fn write_wrapped(ring: &Tensor, src: &Tensor, start: usize, w: usize) -> Result<()> {
    let s = src.dim(1)?;
    if start + s <= w {
        ring.slice_set(&src.contiguous()?, 1, start)?;
    } else {
        let first = w - start;
        ring.slice_set(&src.narrow(1, 0, first)?.contiguous()?, 1, start)?;
        ring.slice_set(&src.narrow(1, first, s - first)?.contiguous()?, 1, 0)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Prefer the real target device (Metal) so the shallow-copy trap the
    /// snapshot guards against is actually exercised; fall back to CPU where
    /// Metal is unavailable (the rollback math is device-agnostic).
    fn dev() -> Device {
        Device::new_metal(0).unwrap_or(Device::Cpu)
    }

    /// Deterministic pseudo-random f32s (LCG, no deps), in roughly [-0.5, 0.5].
    fn seeded(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) as f32 / u32::MAX as f32) - 0.5
            })
            .collect()
    }

    /// K/V for absolute positions `start..start+len` as `[n_kv, len, hd]` f16,
    /// each position's row keyed by that position so identical positions produce
    /// identical bits across independent cache builds.
    fn kv_block(
        start: usize,
        len: usize,
        n_kv: usize,
        hd: usize,
        dev: &Device,
    ) -> (Tensor, Tensor) {
        let build = |base: u64| {
            let rows: Vec<Tensor> = (start..start + len)
                .map(|p| {
                    Tensor::from_vec(seeded(n_kv * hd, base + p as u64), (n_kv, 1, hd), dev)
                        .unwrap()
                })
                .collect();
            Tensor::cat(&rows.iter().collect::<Vec<_>>(), 1)
                .unwrap()
                .to_dtype(DType::F16)
                .unwrap()
        };
        (build(1000), build(9000))
    }

    fn fresh_swa(window: usize, n_kv: usize, hd: usize, dev: &Device) -> LayerCache {
        let z = || Tensor::zeros((n_kv, window, hd), DType::F16, dev).unwrap();
        LayerCache::Swa {
            k: z(),
            v: z(),
            len: 0,
            window,
        }
    }

    /// An empty DeltaNet layer at a small test geometry.
    fn fresh_linear(
        tail: usize,
        conv_dim: usize,
        v_heads: usize,
        hd: usize,
        dev: &Device,
    ) -> LayerCache {
        LayerCache::Linear {
            conv: Tensor::zeros((tail, conv_dim), DType::F32, dev).unwrap(),
            delta: Tensor::zeros((v_heads, hd, hd), DType::F32, dev).unwrap(),
            len: 0,
            trail: Vec::new(),
            armed: None,
        }
    }

    /// The state a token `p` would leave behind, keyed by `p` so a state
    /// restored from the wrong step is visible as wrong values.
    fn linear_state_for(
        p: usize,
        tail: usize,
        conv_dim: usize,
        v_heads: usize,
        hd: usize,
        dev: &Device,
    ) -> (Tensor, Tensor) {
        (
            Tensor::from_vec(
                seeded(tail * conv_dim, 100 + p as u64),
                (tail, conv_dim),
                dev,
            )
            .unwrap(),
            Tensor::from_vec(
                seeded(v_heads * hd * hd, 900 + p as u64),
                (v_heads, hd, hd),
                dev,
            )
            .unwrap(),
        )
    }

    fn linear_values(c: &LayerCache) -> (Vec<f32>, Vec<f32>) {
        let (conv, delta) = c.linear_state().unwrap();
        (
            conv.flatten_all().unwrap().to_vec1().unwrap(),
            delta.flatten_all().unwrap().to_vec1().unwrap(),
        )
    }

    fn fresh_full(max_ctx: usize, n_kv: usize, hd: usize, dev: &Device) -> LayerCache {
        let z = || Tensor::zeros((n_kv, max_ctx, hd), DType::F16, dev).unwrap();
        LayerCache::Full {
            k: z(),
            v: z(),
            len: 0,
        }
    }

    /// A cache stack shaped like the model's: `kinds[il] == true` is a
    /// full-attention layer, false an SWA ring. Each layer is filled with `pos`
    /// positions keyed to its own index, so an image paired with the wrong layer
    /// shows up as wrong bits instead of passing silently.
    fn mixed_stack(
        kinds: &[bool],
        pos: usize,
        n_kv: usize,
        hd: usize,
        max_ctx: usize,
        window: usize,
        dev: &Device,
    ) -> Vec<LayerCache> {
        kinds
            .iter()
            .enumerate()
            .map(|(il, &full)| {
                let mut cache = if full {
                    fresh_full(max_ctx, n_kv, hd, dev)
                } else {
                    fresh_swa(window, n_kv, hd, dev)
                };
                append_range(&mut cache, il * 1000, pos, n_kv, hd, dev);
                cache
            })
            .collect()
    }

    fn append_range(
        c: &mut LayerCache,
        start: usize,
        len: usize,
        n_kv: usize,
        hd: usize,
        dev: &Device,
    ) {
        if len == 0 {
            return;
        }
        let (k, v) = kv_block(start, len, n_kv, hd, dev);
        c.append(&k, &v).unwrap();
    }

    /// The stored f16 bit pattern of every element, read without widening, so the
    /// comparison catches everything a float `==` would miss: the sign of zero,
    /// and each NaN's exact payload (widening to f32 is where a payload could be
    /// canonicalized away, hiding a difference the storage really has).
    fn bits(t: &Tensor) -> Vec<u16> {
        t.flatten_all()
            .unwrap()
            .to_vec1::<f16>()
            .unwrap()
            .iter()
            .map(|x| x.to_bits())
            .collect()
    }

    /// f16 bit patterns that a merely value-preserving path can still fail to
    /// reproduce: a quiet NaN with a nonzero payload, a SIGNALING NaN (0xFD55 —
    /// any arithmetic quiets it to 0xFF55), both infinities, negative zero, and
    /// the smallest subnormal. A raw byte copy carries all six unchanged, which is
    /// what these tests assert. The ones with teeth: adding +0.0 turns -0.0 into
    /// +0.0, which is why `materialize` blits instead of going through `affine`;
    /// a subnormal-flushing kernel would drop 0x0001; a clamping one would drop
    /// the infinities.
    const SPECIAL_F16_BITS: [u16; 6] = [0x7E01, 0xFD55, 0x7C00, 0xFC00, 0x8000, 0x0001];

    /// Overwrite ring/row SLOT `slot` of `c` (a raw slot index, not a position)
    /// with `SPECIAL_F16_BITS`, cycling through them across the row. Applied
    /// identically to a cache and to whatever it is compared against, so a path
    /// that mangles any of those patterns shows up as a bit difference instead of
    /// passing on the ordinary values around it.
    fn write_specials(c: &LayerCache, slot: usize, n_kv: usize, hd: usize, dev: &Device) {
        let row: Vec<f16> = (0..n_kv * hd)
            .map(|i| f16::from_bits(SPECIAL_F16_BITS[i % SPECIAL_F16_BITS.len()]))
            .collect();
        let t = Tensor::from_vec(row, (n_kv, 1, hd), dev).unwrap();
        let (k, v) = ring_kv(c);
        k.slice_set(&t, 1, slot).unwrap();
        v.slice_set(&t, 1, slot).unwrap();
    }

    fn ring_kv(c: &LayerCache) -> (Tensor, Tensor) {
        match c {
            LayerCache::Full { k, v, .. } | LayerCache::Swa { k, v, .. } => (k.clone(), v.clone()),
            LayerCache::Linear { .. } => unreachable!("no ring test builds a DeltaNet layer"),
        }
    }

    /// Test 1: a full-attention layer truncated mid-stream and re-extended equals
    /// a fresh cache built from the surviving prefix plus the same continuation —
    /// over the valid `[0, len)` view (stale slots past `len` are ignored by
    /// every reader and reclaimed by the next append).
    #[test]
    fn full_layer_truncate_matches_fresh() {
        let dev = dev();
        let (n_kv, hd, max_ctx) = (2, 4, 64);
        for commit in [0usize, 3, 8] {
            let span = 8;
            let len0 = 12;
            let n_new = 4;

            let mut a = fresh_full(max_ctx, n_kv, hd, &dev);
            append_range(&mut a, 0, len0, n_kv, hd, &dev);
            let lc = a.checkpoint(span).unwrap();
            append_range(&mut a, len0, span, n_kv, hd, &dev); // verify span
            a.rollback(&lc, len0, span, commit).unwrap();
            append_range(&mut a, len0 + commit, n_new, n_kv, hd, &dev);

            let mut b = fresh_full(max_ctx, n_kv, hd, &dev);
            append_range(&mut b, 0, len0 + commit, n_kv, hd, &dev);
            append_range(&mut b, len0 + commit, n_new, n_kv, hd, &dev);

            let end = len0 + commit + n_new;
            assert_eq!(a.len(), end, "commit {commit}");
            assert_eq!(b.len(), end, "commit {commit}");
            let (ka, va) = ring_kv(&a);
            let (kb, vb) = ring_kv(&b);
            let view = |t: &Tensor| t.narrow(1, 0, end).unwrap();
            assert_eq!(bits(&view(&ka)), bits(&view(&kb)), "commit {commit} K");
            assert_eq!(bits(&view(&va)), bits(&view(&vb)), "commit {commit} V");
        }
    }

    /// Growth is invisible to the data: a cache that started small and grew on
    /// demand ends bit-identical (over the committed rows) to one allocated at
    /// the final size from the start, and a snapshot taken before the growth
    /// still restores after it — LayerSnapshot::Full is a pure length, so this
    /// is exactly the property the copy in `ensure_full_capacity` must uphold.
    #[test]
    fn full_layer_growth_preserves_rows_and_snapshots() {
        let dev = dev();
        let (n_kv, hd) = (2, 4);

        let mut grown = fresh_full(8, n_kv, hd, &dev);
        append_range(&mut grown, 0, 6, n_kv, hd, &dev);
        let snap = grown.snapshot().unwrap();
        assert_eq!(grown.ensure_full_capacity(20, 64).unwrap(), 32);
        append_range(&mut grown, 6, 14, n_kv, hd, &dev);

        let mut flat = fresh_full(32, n_kv, hd, &dev);
        append_range(&mut flat, 0, 6, n_kv, hd, &dev);
        append_range(&mut flat, 6, 14, n_kv, hd, &dev);

        assert_eq!(grown.len(), 20);
        let (kg, vg) = ring_kv(&grown);
        let (kf, vf) = ring_kv(&flat);
        let view = |t: &Tensor| t.narrow(1, 0, 20).unwrap();
        assert_eq!(bits(&view(&kg)), bits(&view(&kf)), "K");
        assert_eq!(bits(&view(&vg)), bits(&view(&vf)), "V");

        // The pre-growth snapshot still rewinds the grown cache, and the
        // committed prefix it lands on is the one the copy carried over.
        grown.restore(&snap, 6).unwrap();
        assert_eq!(grown.len(), 6);
        let (kr, _) = ring_kv(&grown);
        assert_eq!(
            bits(&kr.narrow(1, 0, 6).unwrap()),
            bits(&kf.narrow(1, 0, 6).unwrap()),
            "restored prefix"
        );
    }

    /// Growth copies the WHOLE old buffer, so even rows above a rewound `len`
    /// survive it: a snapshot taken at a deeper position than the cache holds
    /// at growth time still restores afterwards, exactly as it would have with
    /// the buffers never moving. (Whether any caller may restore forward is the
    /// documented rewind discipline's business; growth must not narrow it.)
    #[test]
    fn full_layer_growth_survives_a_forward_restore() {
        let dev = dev();
        let (n_kv, hd) = (2, 4);
        let mut c = fresh_full(8, n_kv, hd, &dev);
        append_range(&mut c, 0, 8, n_kv, hd, &dev);
        let deep = c.snapshot().unwrap(); // pos 8, the whole allocation
        let shallow = c.snapshot().unwrap();
        c.restore(&shallow, 5).unwrap(); // rewind: rows [5, 8) stay in storage
        c.ensure_full_capacity(20, 64).unwrap();
        c.restore(&deep, 8).unwrap(); // forward again, across the growth

        let mut want = fresh_full(32, n_kv, hd, &dev);
        append_range(&mut want, 0, 8, n_kv, hd, &dev);
        let (kc, vc) = ring_kv(&c);
        let (kw, vw) = ring_kv(&want);
        let view = |t: &Tensor| t.narrow(1, 0, 8).unwrap();
        assert_eq!(bits(&view(&kc)), bits(&view(&kw)), "K");
        assert_eq!(bits(&view(&vc)), bits(&view(&vw)), "V");
    }

    /// The ceiling is hard: growth doubles only up to `cap`, and a `needed`
    /// past it is an error rather than a silent partial allocation.
    #[test]
    fn full_layer_growth_respects_the_ceiling() {
        let dev = dev();
        let mut c = fresh_full(8, 2, 4, &dev);
        // Doubling from 8 toward 50 overshoots to 64, which the cap trims.
        assert_eq!(c.ensure_full_capacity(50, 60).unwrap(), 60);
        assert!(c.ensure_full_capacity(61, 60).is_err());
        // Within the allocation, a no-op reports the standing capacity.
        assert_eq!(c.ensure_full_capacity(10, 60).unwrap(), 60);
        // Non-full layers have nothing to grow and say so with a 0.
        let mut lin = fresh_linear(3, 6, 2, 4, &dev);
        assert_eq!(lin.ensure_full_capacity(1000, 60).unwrap(), 0);
    }

    /// Test 2: SWA ring checkpoint/rollback is byte-exact across the interesting
    /// regimes — (a) len0 < window (slots empty pre-wrap), (b) len0 > window with
    /// the span fully inside the current turn of the ring, (c) span straddling the
    /// ring boundary — for every commit in 0..=span and with the rejected tail
    /// both re-covered and not re-covered by the subsequent appends (n_new). The
    /// property: rollback(commit) then any continuation == never appending the
    /// rejected positions. The FULL ring is compared, unwritten slots included.
    #[test]
    fn swa_ring_rollback_is_bit_exact() {
        let dev = dev();
        let (n_kv, hd) = (2, 4);
        // (window, prefix_len == len0, span)
        for &(window, len0, span) in &[(16usize, 5usize, 4usize), (8, 12, 3), (8, 14, 4)] {
            for commit in 0..=span {
                for &n_new in &[0usize, 2] {
                    let mut a = fresh_swa(window, n_kv, hd, &dev);
                    append_range(&mut a, 0, len0, n_kv, hd, &dev);
                    let lc = a.checkpoint(span).unwrap();
                    append_range(&mut a, len0, span, n_kv, hd, &dev); // verify span
                    a.rollback(&lc, len0, span, commit).unwrap();
                    append_range(&mut a, len0 + commit, n_new, n_kv, hd, &dev);

                    let mut b = fresh_swa(window, n_kv, hd, &dev);
                    append_range(&mut b, 0, len0, n_kv, hd, &dev);
                    append_range(&mut b, len0, commit, n_kv, hd, &dev); // committed prefix
                    append_range(&mut b, len0 + commit, n_new, n_kv, hd, &dev);

                    let label = format!(
                        "window {window} len0 {len0} span {span} commit {commit} n_new {n_new}"
                    );
                    assert_eq!(a.len(), b.len(), "{label}: len");
                    let (ka, va) = ring_kv(&a);
                    let (kb, vb) = ring_kv(&b);
                    assert_eq!(bits(&ka), bits(&kb), "{label}: K");
                    assert_eq!(bits(&va), bits(&vb), "{label}: V");
                }
            }
        }
    }

    /// Test 3: snapshot/restore returns an SWA ring to its captured state after
    /// generation ran past it and overwrote the ring slots — across the regimes
    /// that make the ring's layout differ: `pos` below the window (nothing has
    /// wrapped yet), `pos` past the window with the continuation staying inside
    /// one turn of the ring, and a continuation long enough to wrap. The
    /// restored cache must equal one that only ever saw the snapshot's prefix,
    /// which is exactly what a shallow copy of the ring cannot deliver: the
    /// continuation would have mutated the snapshot's own storage.
    #[test]
    fn swa_snapshot_restore_survives_overwrite() {
        let dev = dev();
        let (n_kv, hd) = (2, 4);
        // (window, tokens before the snapshot, tokens appended after it)
        for &(window, pos, extra) in &[
            (16usize, 5usize, 4usize), // pre-wraparound, pos < window
            (8, 6, 3),                 // pre-wraparound, continuation wraps
            (8, 19, 5),                // wrapped before the snapshot
            (8, 12, 20),               // continuation laps the ring
        ] {
            let mut c = fresh_swa(window, n_kv, hd, &dev);
            append_range(&mut c, 0, pos, n_kv, hd, &dev);
            let snap = c.snapshot().unwrap();
            append_range(&mut c, pos, extra, n_kv, hd, &dev);
            c.restore(&snap, pos).unwrap();

            let mut want = fresh_swa(window, n_kv, hd, &dev);
            append_range(&mut want, 0, pos, n_kv, hd, &dev);

            let label = format!("window {window} pos {pos} extra {extra}");
            assert_eq!(c.len(), pos, "{label}: len");
            let (kc, vc) = ring_kv(&c);
            let (kw, vw) = ring_kv(&want);
            assert_eq!(bits(&kc), bits(&kw), "{label}: K");
            assert_eq!(bits(&vc), bits(&vw), "{label}: V");
        }
    }

    /// Test 4: a restored cache keeps generating the same keys a never-rewound
    /// one would — the property the prefix cache relies on when a conversation
    /// is rewound to a turn boundary and continued with different tokens.
    #[test]
    fn swa_restore_then_append_matches_fresh() {
        let dev = dev();
        let (n_kv, hd, window, pos) = (2, 4, 8, 13);
        let mut c = fresh_swa(window, n_kv, hd, &dev);
        append_range(&mut c, 0, pos, n_kv, hd, &dev);
        let snap = c.snapshot().unwrap();
        // An abandoned branch: appended, then rewound and replaced by a
        // continuation of a different length.
        append_range(&mut c, pos, 9, n_kv, hd, &dev);
        c.restore(&snap, pos).unwrap();
        append_range(&mut c, pos, 4, n_kv, hd, &dev);

        let mut want = fresh_swa(window, n_kv, hd, &dev);
        append_range(&mut want, 0, pos + 4, n_kv, hd, &dev);

        assert_eq!(c.len(), want.len());
        let (kc, vc) = ring_kv(&c);
        let (kw, vw) = ring_kv(&want);
        assert_eq!(bits(&kc), bits(&kw), "K");
        assert_eq!(bits(&vc), bits(&vw), "V");
    }

    /// Test 5: a full-attention layer restores by shortening alone — its valid
    /// `[0, pos)` view is untouched by the appends that ran past the snapshot,
    /// and the stale tail is reclaimed by the next append.
    #[test]
    fn full_layer_restore_is_a_truncation() {
        let dev = dev();
        let (n_kv, hd, max_ctx, pos) = (2, 4, 64, 11);
        let mut c = fresh_full(max_ctx, n_kv, hd, &dev);
        append_range(&mut c, 0, pos, n_kv, hd, &dev);
        let snap = c.snapshot().unwrap();
        append_range(&mut c, pos, 7, n_kv, hd, &dev);
        c.restore(&snap, pos).unwrap();
        append_range(&mut c, pos, 3, n_kv, hd, &dev);

        let mut want = fresh_full(max_ctx, n_kv, hd, &dev);
        append_range(&mut want, 0, pos + 3, n_kv, hd, &dev);

        assert_eq!(c.len(), pos + 3);
        let (kc, vc) = ring_kv(&c);
        let (kw, vw) = ring_kv(&want);
        let view = |t: &Tensor| t.narrow(1, 0, pos + 3).unwrap();
        assert_eq!(bits(&view(&kc)), bits(&view(&kw)), "K");
        assert_eq!(bits(&view(&vc)), bits(&view(&vw)), "V");
    }

    /// Test 6: the snapshot owns its storage. Capture a ring, overwrite every
    /// slot by appending a full window of new positions, and the snapshot's bits
    /// must be unchanged — a Metal `Tensor::copy()`/`try_clone` would alias the
    /// ring and fail here.
    #[test]
    fn swa_snapshot_owns_its_storage() {
        let dev = dev();
        let (n_kv, hd, window, pos) = (2, 4, 8, 10);
        let mut c = fresh_swa(window, n_kv, hd, &dev);
        append_range(&mut c, 0, pos, n_kv, hd, &dev);
        let snap = c.snapshot().unwrap();
        let before = match &snap {
            LayerSnapshot::Swa { k, v, .. } => (bits(k), bits(v)),
            _ => unreachable!("an SWA cache yields an SWA snapshot"),
        };
        append_range(&mut c, pos, window, n_kv, hd, &dev);
        let after = match &snap {
            LayerSnapshot::Swa { k, v, .. } => (bits(k), bits(v)),
            _ => unreachable!(),
        };
        assert_eq!(
            before.0, after.0,
            "snapshot K aliased the ring (shallow copy)"
        );
        assert_eq!(
            before.1, after.1,
            "snapshot V aliased the ring (shallow copy)"
        );
    }

    /// Test 7: the snapshot kind is tied to the cache kind; crossing them is a
    /// caller bug that must not silently leave a cache half-restored.
    #[test]
    fn restore_rejects_a_mismatched_snapshot_kind() {
        let dev = dev();
        let (n_kv, hd) = (2, 4);
        let swa = fresh_swa(8, n_kv, hd, &dev);
        let full = fresh_full(32, n_kv, hd, &dev);
        assert!(
            fresh_full(32, n_kv, hd, &dev)
                .restore(&swa.snapshot().unwrap(), 0)
                .is_err()
        );
        assert!(
            fresh_swa(8, n_kv, hd, &dev)
                .restore(&full.snapshot().unwrap(), 0)
                .is_err()
        );
    }

    /// Test 8: the snapshot is a real copy, not a Metal shallow Arc clone. Take a
    /// checkpoint, then overwrite the very slots it captured (by running the verify
    /// span) and assert the snapshot's bytes are unchanged. If `materialize`
    /// regressed to a shallow view aliasing the ring, the append would mutate the
    /// snapshot and this fails.
    #[test]
    fn swa_snapshot_is_a_real_copy() {
        let dev = dev();
        let (n_kv, hd, window, len0, span) = (2, 4, 8, 12, 4);
        let mut c = fresh_swa(window, n_kv, hd, &dev);
        append_range(&mut c, 0, len0, n_kv, hd, &dev);
        let lc = c.checkpoint(span).unwrap();
        let (before_k, before_v) = match &lc {
            LayerCheckpoint::Swa { k, v } => (bits(k), bits(v)),
            _ => unreachable!("an SWA cache yields an SWA checkpoint"),
        };
        // Clobber the captured slots.
        append_range(&mut c, len0, span, n_kv, hd, &dev);
        let (after_k, after_v) = match &lc {
            LayerCheckpoint::Swa { k, v } => (bits(k), bits(v)),
            _ => unreachable!(),
        };
        assert_eq!(
            before_k, after_k,
            "snapshot K aliased the ring (shallow copy)"
        );
        assert_eq!(
            before_v, after_v,
            "snapshot V aliased the ring (shallow copy)"
        );
    }

    /// Test 9: a snapshot survives a trip through host RAM. Take one over a mixed
    /// full/SWA stack, copy it to host, run the live caches on until every ring
    /// slot has been overwritten, then rebuild the snapshot from the host bytes
    /// and restore — the caches must equal ones that only ever saw the snapshot's
    /// prefix. Per-layer kinds and their order have to survive too, so the full
    /// layer is restored by the same rebuilt snapshot. One slot per layer carries
    /// the special f16 patterns, so the trip has to move bytes rather than values.
    #[test]
    fn host_snapshot_round_trip_restores_the_captured_state() {
        let dev = dev();
        let (n_kv, hd, window, max_ctx, pos, extra) = (2, 4, 8, 64, 12, 9);
        let special_slot = (pos - 1) % window;
        let mut full = fresh_full(max_ctx, n_kv, hd, &dev);
        let mut swa = fresh_swa(window, n_kv, hd, &dev);
        append_range(&mut full, 0, pos, n_kv, hd, &dev);
        append_range(&mut swa, 0, pos, n_kv, hd, &dev);
        write_specials(&full, pos - 1, n_kv, hd, &dev);
        write_specials(&swa, special_slot, n_kv, hd, &dev);

        let host = CacheSnapshot::new(pos, vec![full.snapshot().unwrap(), swa.snapshot().unwrap()])
            .to_host()
            .unwrap();
        assert_eq!(host.pos, pos);
        // Only the SWA ring carries data; the full layer restores by truncation.
        assert_eq!(host.byte_len(), 2 * n_kv * window * hd * size_of::<f16>());

        // Generation runs past the snapshot and laps the ring.
        append_range(&mut full, pos, extra, n_kv, hd, &dev);
        append_range(&mut swa, pos, extra, n_kv, hd, &dev);

        let rebuilt = host.to_snapshot(&dev).unwrap();
        assert_eq!(rebuilt.pos(), pos);
        full.restore(&rebuilt.layers()[0], rebuilt.pos()).unwrap();
        swa.restore(&rebuilt.layers()[1], rebuilt.pos()).unwrap();

        let mut want_full = fresh_full(max_ctx, n_kv, hd, &dev);
        let mut want_swa = fresh_swa(window, n_kv, hd, &dev);
        append_range(&mut want_full, 0, pos, n_kv, hd, &dev);
        append_range(&mut want_swa, 0, pos, n_kv, hd, &dev);
        write_specials(&want_full, pos - 1, n_kv, hd, &dev);
        write_specials(&want_swa, special_slot, n_kv, hd, &dev);

        // The specials really are in the data the assertions below compare, so
        // those assertions exercise them and not only the values around them.
        let specials: Vec<u16> = (0..n_kv * hd)
            .map(|i| SPECIAL_F16_BITS[i % SPECIAL_F16_BITS.len()])
            .collect();
        let ring_bits = bits(&ring_kv(&want_swa).0);
        for head in 0..n_kv {
            let at = (head * window + special_slot) * hd;
            assert_eq!(
                &ring_bits[at..at + hd],
                &specials[head * hd..(head + 1) * hd],
                "head {head}: the special row was not written"
            );
        }

        assert_eq!(swa.len(), pos, "SWA len");
        assert_eq!(full.len(), pos, "full len");
        let (ks, vs) = ring_kv(&swa);
        let (kw, vw) = ring_kv(&want_swa);
        assert_eq!(bits(&ks), bits(&kw), "SWA K");
        assert_eq!(bits(&vs), bits(&vw), "SWA V");
        let (kf, vf) = ring_kv(&full);
        let (kfw, vfw) = ring_kv(&want_full);
        let view = |t: &Tensor| t.narrow(1, 0, pos).unwrap();
        assert_eq!(bits(&view(&kf)), bits(&view(&kfw)), "full K");
        assert_eq!(bits(&view(&vf)), bits(&view(&vfw)), "full V");
    }

    /// Test 10: a full-attention layer's committed rows survive a host round trip
    /// bit-for-bit. Export, let a different conversation write over every slot,
    /// import, and the cache holds exactly what it did before, at the same length
    /// — this is what makes a parked conversation resumable without re-prefilling.
    /// The truncated import (`pos` below the image's length, i.e. resuming at an
    /// earlier turn boundary) restores just that prefix and reports it as the
    /// length, leaving the stored image intact for a later full import. Row 0
    /// carries the special f16 patterns, so every case — including the truncated
    /// ones, whose prefix is gathered per head — moves bytes rather than values.
    #[test]
    fn full_kv_host_round_trip_is_bit_exact() {
        let dev = dev();
        let (n_kv, hd, max_ctx, pos) = (2, 4, 64, 13);
        for want_pos in [pos, pos - 5, 0] {
            let mut c = fresh_full(max_ctx, n_kv, hd, &dev);
            append_range(&mut c, 0, pos, n_kv, hd, &dev);
            write_specials(&c, 0, n_kv, hd, &dev);
            let (k_bytes, v_bytes) = c.export_full_rows().unwrap();
            let image = HostFullKv::new(pos, n_kv, hd, vec![(k_bytes, v_bytes)]).unwrap();
            assert_eq!(image.byte_len(), 2 * n_kv * pos * hd * size_of::<f16>());

            // Another conversation takes the slots over: different positions, so
            // different bits in every row the import has to overwrite.
            c.reset().unwrap();
            append_range(&mut c, 500, pos + 3, n_kv, hd, &dev);

            let (k, v) = image.prefix(0, want_pos).unwrap();
            c.import_full_rows(&k, &v, want_pos).unwrap();
            assert_eq!(image.pos, pos, "the image is left intact");

            let mut want = fresh_full(max_ctx, n_kv, hd, &dev);
            append_range(&mut want, 0, want_pos, n_kv, hd, &dev);

            let label = format!("pos {want_pos}");
            assert_eq!(c.len(), want_pos, "{label}: len");
            if want_pos > 0 {
                write_specials(&want, 0, n_kv, hd, &dev);
                let (kc, vc) = ring_kv(&c);
                let (kw, vw) = ring_kv(&want);
                let view = |t: &Tensor| t.narrow(1, 0, want_pos).unwrap();
                let want_k = bits(&view(&kw));
                // The specials survived into what is being compared, so the
                // equality below is exercising them and not only the ordinary
                // values around them.
                for head in 0..n_kv {
                    let at = head * want_pos * hd;
                    let expect: Vec<u16> = (0..hd)
                        .map(|i| SPECIAL_F16_BITS[(head * hd + i) % SPECIAL_F16_BITS.len()])
                        .collect();
                    assert_eq!(want_k[at..at + hd], expect, "{label}: head {head} specials");
                }
                assert_eq!(bits(&view(&kc)), want_k, "{label}: K");
                assert_eq!(bits(&view(&vc)), bits(&view(&vw)), "{label}: V");
            }
        }
    }

    /// Test 11: the host images police their own bounds. Asking for more positions
    /// than an image holds, or for a layer it does not cover, is a caller bug that
    /// must not silently import a short or wrong prefix; SWA rings reject the row
    /// interface outright (a ring is only recoverable from a snapshot).
    #[test]
    fn host_full_kv_rejects_out_of_range_requests() {
        let dev = dev();
        let (n_kv, hd, max_ctx, pos) = (2, 4, 32, 6);
        let mut c = fresh_full(max_ctx, n_kv, hd, &dev);
        append_range(&mut c, 0, pos, n_kv, hd, &dev);
        let planes = vec![c.export_full_rows().unwrap()];
        let image = HostFullKv::new(pos, n_kv, hd, planes).unwrap();

        assert!(image.prefix(0, pos + 1).is_err(), "pos past the image");
        assert!(image.prefix(1, pos).is_err(), "layer past the image");
        assert!(
            fresh_swa(8, n_kv, hd, &dev).export_full_rows().is_err(),
            "an SWA ring has no row image"
        );
        let (k, v) = image.prefix(0, pos).unwrap();
        assert!(
            fresh_swa(8, n_kv, hd, &dev)
                .import_full_rows(&k, &v, pos)
                .is_err(),
            "an SWA ring is not importable from rows"
        );
        // A byte count that does not match the requested positions means the
        // image and the shape metadata disagree.
        assert!(
            fresh_full(max_ctx, n_kv, hd, &dev)
                .import_full_rows(&k, &v, pos - 1)
                .is_err(),
            "byte count must match the requested positions"
        );
        // The declared shape is checked against the planes at construction, so a
        // record can never be read with a layout its bytes do not support.
        assert!(
            HostFullKv::new(pos, n_kv, hd, vec![(vec![0u8; 4], vec![0u8; 4])]).is_err(),
            "plane size must match the declared shape"
        );
    }

    /// A token range is one run per head, not one slice of the buffer, and the stride
    /// those runs sit at is the SOURCE image's length. So cutting a range out and
    /// putting ranges back together are both gathers, and a byte-wise append would
    /// interleave one span's heads with another's. Cutting a whole conversation at
    /// every position and reassembling it reproduces the original bit for bit, which
    /// is the property a stored chain's hydration rests on.
    #[test]
    fn full_kv_ranges_cut_and_compose_per_head() {
        let (n_kv, hd, pos, layers) = (3, 4, 9, 2);
        let row = hd * size_of::<f16>();
        // Every byte says which layer, plane, head and position it came from, so a
        // run gathered from the wrong offset fails on content rather than length.
        let plane = |layer: usize, which: u8| {
            let mut bytes = Vec::new();
            for head in 0..n_kv {
                for p in 0..pos {
                    for i in 0..row {
                        bytes.push(
                            (layer as u8) << 6
                                | which << 5
                                | (head as u8) << 3
                                | (p as u8 ^ i as u8) & 0x07,
                        );
                    }
                }
            }
            bytes
        };
        let planes = (0..layers).map(|l| (plane(l, 0), plane(l, 1))).collect();
        let whole = HostFullKv::new(pos, n_kv, hd, planes).unwrap();

        // A range's rows are the source's rows for those positions, per head.
        for (start, end) in [(0, pos), (0, 4), (4, 9), (3, 3), (2, 7)] {
            let cut = whole.range(start, end).unwrap();
            assert_eq!(cut.pos, end - start);
            for idx in 0..layers {
                let (k, v) = cut.prefix(idx, end - start).unwrap();
                let (wk, wv) = whole.prefix(idx, pos).unwrap();
                for head in 0..n_kv {
                    let want = &wk[head * pos * row + start * row..][..(end - start) * row];
                    assert_eq!(
                        &k[head * (end - start) * row..][..(end - start) * row],
                        want,
                        "layer {idx} head {head} K over [{start}, {end})"
                    );
                    let want = &wv[head * pos * row + start * row..][..(end - start) * row];
                    assert_eq!(
                        &v[head * (end - start) * row..][..(end - start) * row],
                        want,
                        "layer {idx} head {head} V over [{start}, {end})"
                    );
                }
            }
        }

        // Every partition of the conversation composes back into it.
        let bytes_of = |image: &HostFullKv| {
            let mut out = Vec::new();
            image.write_to(&mut out).unwrap();
            out
        };
        for cut in 0..=pos {
            let spans = [whole.range(0, cut).unwrap(), whole.range(cut, pos).unwrap()];
            let back = HostFullKv::concat(spans.into()).unwrap();
            assert_eq!(back.pos, pos);
            assert_eq!(bytes_of(&back), bytes_of(&whole), "recomposed at {cut}");
        }
        // Three spans, which is what a split leaves behind: base, tail, new tail.
        let spans = [
            whole.range(0, 2).unwrap(),
            whole.range(2, 6).unwrap(),
            whole.range(6, pos).unwrap(),
        ];
        assert_eq!(
            bytes_of(&HostFullKv::concat(spans.into()).unwrap()),
            bytes_of(&whole)
        );

        // Bounds and shape agreement are the caller's to get right, and getting them
        // wrong must not compose a silently transposed image.
        assert!(whole.range(0, pos + 1).is_err());
        assert!(whole.range(5, 4).is_err());
        assert!(HostFullKv::concat(Vec::new()).is_err());
        let other = HostFullKv::new(pos, hd, n_kv, vec![]).unwrap();
        assert!(HostFullKv::concat(vec![whole.range(0, 2).unwrap(), other]).is_err());
    }

    /// Test 12: over a stack shaped like the model's, each full-attention layer's
    /// image goes back to the layer it came from. Every layer holds values keyed
    /// to its own index, so a mispaired plane fails on bits rather than passing —
    /// which is what makes the layer-order contract testable at all. The SWA rings
    /// are left exactly as the intruding conversation wrote them: only a
    /// `HostSnapshot` restore can put those back.
    #[test]
    fn full_kv_stack_round_trip_pairs_each_layer_with_its_own_image() {
        let dev = dev();
        let (n_kv, hd, max_ctx, window, pos) = (2, 4, 64, 8, 11);
        // The shipped model's interleave: full attention iff il % 4 == 0.
        let kinds = [true, false, false, false, true, false];
        let mut caches = mixed_stack(&kinds, pos, n_kv, hd, max_ctx, window, &dev);

        let image = export_full_kv_from(&caches).unwrap();
        assert_eq!(image.pos, pos);
        assert_eq!(image.layers(), 2, "one plane per full-attention layer");
        assert_eq!(image.byte_len(), 2 * 2 * n_kv * pos * hd * size_of::<f16>());

        // A different conversation takes the whole stack over.
        for (il, cache) in caches.iter_mut().enumerate() {
            cache.reset().unwrap();
            append_range(cache, 500 + il * 1000, pos + 3, n_kv, hd, &dev);
        }

        import_full_kv_into(&mut caches, &image, pos).unwrap();

        let want = mixed_stack(&kinds, pos, n_kv, hd, max_ctx, window, &dev);
        for (il, (got, want)) in caches.iter().zip(&want).enumerate() {
            let (kg, vg) = ring_kv(got);
            if got.is_full() {
                assert_eq!(got.len(), pos, "layer {il}: len");
                let (kw, vw) = ring_kv(want);
                let view = |t: &Tensor| t.narrow(1, 0, pos).unwrap();
                assert_eq!(bits(&view(&kg)), bits(&view(&kw)), "layer {il}: K");
                assert_eq!(bits(&view(&vg)), bits(&view(&vw)), "layer {il}: V");
            } else {
                assert_eq!(
                    got.len(),
                    pos + 3,
                    "layer {il}: the import must not touch SWA rings"
                );
            }
        }
    }

    /// Test 13: an image that does not describe this stack is refused before any
    /// of its bytes land. Each rejection covers a way a record can be wrong that
    /// the next one cannot catch: too many positions, the wrong number of
    /// full-attention layers, and a transposed layout whose byte count is exactly
    /// right.
    #[test]
    fn import_full_kv_rejects_images_that_do_not_describe_the_stack() {
        let dev = dev();
        let (n_kv, hd, max_ctx, window, pos) = (2, 4, 64, 8, 9);
        let kinds = [true, false, true, false];
        let mut caches = mixed_stack(&kinds, pos, n_kv, hd, max_ctx, window, &dev);
        let image = export_full_kv_from(&caches).unwrap();

        assert!(
            import_full_kv_into(&mut caches, &image, pos + 1).is_err(),
            "more positions than the image holds"
        );

        let fewer = mixed_stack(&[true, false], pos, n_kv, hd, max_ctx, window, &dev);
        let fewer_image = export_full_kv_from(&fewer).unwrap();
        assert!(
            import_full_kv_into(&mut caches, &fewer_image, pos).is_err(),
            "an image covering a different number of full-attention layers"
        );

        // Heads and head dim swapped. The totals are identical, so only the
        // record's own shape metadata can tell this from a valid image.
        let swapped = mixed_stack(&kinds, pos, hd, n_kv, max_ctx, window, &dev);
        let swapped_image = export_full_kv_from(&swapped).unwrap();
        assert_eq!(swapped_image.byte_len(), image.byte_len());
        assert!(
            import_full_kv_into(&mut caches, &swapped_image, pos).is_err(),
            "a transposed image has the right size and the wrong layout"
        );
    }

    /// Test 14: a ring record has to cover its whole ring. A snapshot short of the
    /// window would be written into the leading slots only, leaving the rest
    /// holding whatever was there before — the previous conversation's keys — so
    /// both the host record and the device restore reject it.
    #[test]
    fn ring_snapshots_must_cover_the_whole_window() {
        let dev = dev();
        let (n_kv, hd, window) = (2, 4, 8);
        let short_slots = window - 1;
        let short_bytes = n_kv * short_slots * hd * size_of::<f16>();
        let inconsistent = HostSnapshot {
            pos: window,
            layers: vec![HostLayerSnapshot::Swa {
                k: vec![0u8; short_bytes],
                v: vec![0u8; short_bytes],
                shape: (n_kv, short_slots, hd),
                window,
            }],
        };
        assert!(
            inconsistent.to_snapshot(&dev).is_err(),
            "the stored slot count must match the stored window"
        );

        let z = || Tensor::zeros((n_kv, short_slots, hd), DType::F16, &dev).unwrap();
        let short = LayerSnapshot::Swa {
            k: z(),
            v: z(),
            window,
        };
        assert!(
            fresh_swa(window, n_kv, hd, &dev)
                .restore(&short, window)
                .is_err(),
            "a partial ring must not be restored into a full one"
        );
    }

    /// Bytes keyed to their own offset, so a plane that ends up framed at the wrong
    /// place — or swapped with its neighbour — fails on content rather than length.
    fn plane_pattern(len: usize, seed: u8) -> Vec<u8> {
        (0..len).map(|i| (i as u8) ^ seed ^ 0xa5).collect()
    }

    /// Test 15: a full-attention record survives the framed round trip. The shape
    /// is deliberately asymmetric — heads, positions and head dim all differ — so a
    /// transposition or a swapped field cannot pass on byte counts alone, and the
    /// declared length has to be exactly what the writer emits, since the container
    /// directory commits to it before the planes are written.
    #[test]
    fn full_kv_record_survives_a_framed_round_trip() {
        let (pos, n_kv, hd, layers) = (7usize, 3usize, 5usize, 4usize);
        let plane = f16_bytes_for(&[n_kv, pos, hd]).unwrap();
        let planes: Vec<(Vec<u8>, Vec<u8>)> = (0..layers)
            .map(|il| {
                (
                    plane_pattern(plane, il as u8),
                    plane_pattern(plane, 0x40 + il as u8),
                )
            })
            .collect();
        let image = HostFullKv::new(pos, n_kv, hd, planes.clone()).unwrap();

        let mut bytes = Vec::new();
        image.write_to(&mut bytes).unwrap();
        assert_eq!(
            bytes.len(),
            image.serialized_len(),
            "the declared record length must be what the writer emits"
        );

        let back = HostFullKv::read_from(&bytes[..], bytes.len() as u64).unwrap();
        assert_eq!(back.pos, pos);
        assert_eq!(back.n_kv_head, n_kv);
        assert_eq!(back.head_dim, hd);
        assert_eq!(back.layers(), layers);
        for (il, (k, v)) in planes.iter().enumerate() {
            let (bk, bv) = back.prefix(il, pos).unwrap();
            assert_eq!(&*bk, &k[..], "layer {il}: K");
            assert_eq!(&*bv, &v[..], "layer {il}: V");
        }
    }

    /// Test 16: a snapshot record survives the framed round trip, including the
    /// full-attention layers that carry no ring — their positions in the layer
    /// order are what pairs the rings with the right layers on restore, so an
    /// image that dropped them would restore every ring one layer too early.
    #[test]
    fn snapshot_record_survives_a_framed_round_trip() {
        let (pos, n_kv, hd, window) = (23usize, 3usize, 5usize, 7usize);
        let ring = f16_bytes_for(&[n_kv, window, hd]).unwrap();
        // The shipped model's interleave: full attention iff il % 4 == 0.
        let kinds = [true, false, false, false, true, false];
        let layers: Vec<HostLayerSnapshot> = kinds
            .iter()
            .enumerate()
            .map(|(il, full)| {
                if *full {
                    HostLayerSnapshot::Full
                } else {
                    HostLayerSnapshot::Swa {
                        k: plane_pattern(ring, il as u8),
                        v: plane_pattern(ring, 0x40 + il as u8),
                        shape: (n_kv, window, hd),
                        window,
                    }
                }
            })
            .collect();
        let image = HostSnapshot::new(pos, layers).unwrap();
        let rings = kinds.iter().filter(|f| !**f).count();
        assert_eq!(image.byte_len(), 2 * rings * ring);

        let mut bytes = Vec::new();
        image.write_to(&mut bytes).unwrap();
        assert_eq!(bytes.len(), image.serialized_len());

        let back = HostSnapshot::read_from(&bytes[..], bytes.len() as u64).unwrap();
        assert_eq!(back.pos, pos);
        assert_eq!(back.layers.len(), kinds.len());
        for (il, (got, want)) in back.layers.iter().zip(&image.layers).enumerate() {
            match (got, want) {
                (HostLayerSnapshot::Full, HostLayerSnapshot::Full) => {}
                (
                    HostLayerSnapshot::Swa {
                        k,
                        v,
                        shape,
                        window,
                    },
                    HostLayerSnapshot::Swa {
                        k: wk,
                        v: wv,
                        shape: ws,
                        window: ww,
                    },
                ) => {
                    assert_eq!(k, wk, "layer {il}: K");
                    assert_eq!(v, wv, "layer {il}: V");
                    assert_eq!(shape, ws, "layer {il}: shape");
                    assert_eq!(window, ww, "layer {il}: window");
                }
                _ => panic!("layer {il} changed kind across the round trip"),
            }
        }
    }

    /// Test 17: the framing refuses records whose own numbers do not add up. A
    /// shape that describes more bytes than the planes hold is caught by the
    /// validating constructor — the same one the in-process path uses — and a
    /// length that disagrees with the container's directory is caught by the
    /// budget, in both directions: a body cannot read past its frame, and a body
    /// that stops short of it means the two disagree about what was written.
    #[test]
    fn framed_records_reject_shape_and_length_lies() {
        let (pos, n_kv, hd) = (7usize, 3usize, 5usize);
        let plane = f16_bytes_for(&[n_kv, pos, hd]).unwrap();
        let image = HostFullKv::new(
            pos,
            n_kv,
            hd,
            vec![(plane_pattern(plane, 1), plane_pattern(plane, 2))],
        )
        .unwrap();
        let mut bytes = Vec::new();
        image.write_to(&mut bytes).unwrap();

        assert!(
            HostFullKv::read_from(&bytes[..], bytes.len() as u64 - 8).is_err(),
            "a record framed shorter than its body must not read past the frame"
        );
        assert!(
            HostFullKv::read_from(&bytes[..], bytes.len() as u64 + 8).is_err(),
            "a record that leaves framed bytes unread disagrees with the directory"
        );
        assert!(
            HostFullKv::read_from(&bytes[..bytes.len() - 4], bytes.len() as u64).is_err(),
            "a truncated body must fail rather than yield a short plane"
        );

        // A self-consistent body whose declared shape wants more bytes than its
        // planes carry: the framing is fine, and it is caught where the plane's
        // declared length is compared against the shape already read — before a
        // buffer is sized from either of them.
        let mut lie = Vec::new();
        write_count(&mut lie, pos).unwrap();
        write_count(&mut lie, n_kv).unwrap();
        write_count(&mut lie, hd).unwrap();
        write_count(&mut lie, 1usize).unwrap();
        write_plane(&mut lie, &plane_pattern(plane - 2, 1)).unwrap();
        write_plane(&mut lie, &plane_pattern(plane - 2, 2)).unwrap();
        let Err(err) = HostFullKv::read_from(&lie[..], lie.len() as u64) else {
            panic!("a shape lie must not parse");
        };
        assert!(
            format!("{err:#}").contains("its shape needs"),
            "a shape lie must be refused before it sizes anything: {err:#}"
        );

        // The same for a ring: slots short of the window, framed consistently.
        let ring = f16_bytes_for(&[n_kv, 6, hd]).unwrap();
        let mut short_ring = Vec::new();
        write_count(&mut short_ring, pos).unwrap();
        write_count(&mut short_ring, 1usize).unwrap();
        write_u32(&mut short_ring, LAYER_SWA).unwrap();
        write_count(&mut short_ring, n_kv).unwrap();
        write_count(&mut short_ring, 6usize).unwrap();
        write_count(&mut short_ring, hd).unwrap();
        write_count(&mut short_ring, 7usize).unwrap();
        write_plane(&mut short_ring, &plane_pattern(ring, 1)).unwrap();
        write_plane(&mut short_ring, &plane_pattern(ring, 2)).unwrap();
        assert!(
            HostSnapshot::read_from(&short_ring[..], short_ring.len() as u64).is_err(),
            "a ring covering fewer slots than its window is not restorable"
        );

        // An unknown layer kind is a record this build did not write; guessing at
        // it would pair the rings that follow with the wrong layers.
        let mut bad_kind = Vec::new();
        write_count(&mut bad_kind, pos).unwrap();
        write_count(&mut bad_kind, 1usize).unwrap();
        write_u32(&mut bad_kind, 99).unwrap();
        assert!(
            HostSnapshot::read_from(&bad_kind[..], bad_kind.len() as u64).is_err(),
            "unknown layer kind tag"
        );
    }

    /// Test 18: a plane length corrupted to a huge value costs an error, not an
    /// allocation. The length is compared against the shape already read, and every
    /// read is charged against the record's framed length, both BEFORE a buffer is
    /// sized — which is what keeps a single flipped byte in a cache file from being
    /// an out-of-memory abort at startup, a failure no `catch_unwind` can turn back
    /// into a cache miss.
    #[test]
    fn framed_records_refuse_an_implausible_plane_length() {
        let mut body = Vec::new();
        write_count(&mut body, 1usize).unwrap();
        write_count(&mut body, 1usize).unwrap();
        write_count(&mut body, 1usize).unwrap();
        write_count(&mut body, 1usize).unwrap();
        write_u64(&mut body, u64::MAX).unwrap();
        let Err(err) = HostFullKv::read_from(&body[..], body.len() as u64) else {
            panic!("an implausible plane length must not parse");
        };
        assert!(
            format!("{err:#}").contains("its shape needs"),
            "an implausible plane length must be refused before it sizes a buffer: {err:#}"
        );
    }

    /// Test 19: a count no checkpoint could have produced is refused before it
    /// reaches a `Vec`, whatever the file's length says.
    ///
    /// A file's length is not a bound on what reading it costs: a sparse `.lkv` can
    /// declare gigabytes the filesystem never stored, so the framed-length budget
    /// alone would let a declaration of 250 million layers build 250 million enum
    /// values before the first missing byte was noticed.
    #[test]
    fn framed_records_refuse_counts_no_checkpoint_could_produce() {
        // 250 million layers, framed as though the bytes were there.
        let flood = 250_000_000u64;
        let mut body = Vec::new();
        write_count(&mut body, 8).unwrap();
        write_u64(&mut body, flood).unwrap();
        let Err(err) = HostSnapshot::read_from(&body[..], 8 + 8 + flood * 4) else {
            panic!("an implausible layer count must not parse");
        };
        assert!(
            format!("{err:#}").contains("layer count"),
            "a layer flood must be refused by the plausibility cap: {err:#}"
        );

        // The same for the geometry a plane's size is computed from.
        let geometry = |n_kv: usize, head_dim: usize, pos: usize| {
            let mut body = Vec::new();
            write_count(&mut body, pos).unwrap();
            write_count(&mut body, n_kv).unwrap();
            write_count(&mut body, head_dim).unwrap();
            write_count(&mut body, 1usize).unwrap();
            HostFullKv::read_from(&body[..], 1 << 40)
        };
        assert!(geometry(MAX_STORED_KV_HEADS + 1, 128, 16).is_err());
        assert!(geometry(8, MAX_STORED_HEAD_DIM + 1, 16).is_err());
        assert!(geometry(8, 128, MAX_STORED_POS + 1).is_err());
        // A shape whose every field is plausible and whose PRODUCT is not: 64 heads x
        // 2^20 positions x 512 dims of f16 is 64 GiB, and the per-field caps see
        // nothing wrong with it. The plane ceiling is what stands between that number
        // and an allocator abort.
        let Err(err) = geometry(MAX_STORED_KV_HEADS, MAX_STORED_HEAD_DIM, MAX_STORED_POS) else {
            panic!("a 64 GiB plane must not parse");
        };
        assert!(
            format!("{err:#}").contains("plane byte count"),
            "the product needs its own bound: {err:#}"
        );

        // And the shipped checkpoint's own geometry is nowhere near the caps: a
        // 262144-position image fails on the plane bytes it does not have here, not
        // on a bound.
        let Err(err) = geometry(8, 128, 262_144) else {
            panic!("a body with no planes at all must not parse");
        };
        let err = format!("{err:#}");
        assert!(
            !err.contains("past the"),
            "a real geometry is readable: {err}"
        );
    }

    /// A DeltaNet layer's state is destroyed by every step, so its rollback
    /// cannot be a truncation: the checkpoint arms the layer, the verify forward
    /// records the state after each token, and the rollback picks the entry for
    /// the last accepted one. Commit 0 goes back to the checkpoint's own state.
    #[test]
    fn linear_rollback_lands_on_the_accepted_token_state() {
        let dev = dev();
        let (tail, conv_dim, v_heads, hd) = (3usize, 6usize, 2usize, 4usize);
        let span = 4usize;

        for commit in 0..=span {
            let mut c = fresh_linear(tail, conv_dim, v_heads, hd, &dev);
            // Two committed tokens before the verify span.
            let pre: Vec<(Tensor, Tensor)> = (0..2)
                .map(|p| linear_state_for(p, tail, conv_dim, v_heads, hd, &dev))
                .collect();
            c.advance_linear(2, pre).unwrap();
            let at_checkpoint = linear_values(&c);

            let ckpt = c.checkpoint(span).unwrap();
            assert!(c.linear_trail_armed(), "the checkpoint arms the layer");

            // The verify forward steps the whole span, one state per token.
            let stepped: Vec<(Tensor, Tensor)> = (2..2 + span)
                .map(|p| linear_state_for(p, tail, conv_dim, v_heads, hd, &dev))
                .collect();
            c.advance_linear(span, stepped).unwrap();
            assert_eq!(c.len(), 2 + span);

            c.rollback(&ckpt, 2, span, commit).unwrap();
            assert_eq!(c.len(), 2 + commit);
            assert!(
                !c.linear_trail_armed(),
                "the rollback closes the checkpoint"
            );

            let want = if commit == 0 {
                at_checkpoint.clone()
            } else {
                let (conv, delta) = linear_state_for(1 + commit, tail, conv_dim, v_heads, hd, &dev);
                (
                    conv.flatten_all().unwrap().to_vec1().unwrap(),
                    delta.flatten_all().unwrap().to_vec1().unwrap(),
                )
            };
            assert_eq!(
                linear_values(&c),
                want,
                "commit {commit} restored the wrong step's state"
            );
        }
    }

    /// An armed layer that is stepped without recording every token cannot be
    /// rolled back, and says so rather than restoring a state from the wrong
    /// position.
    #[test]
    fn linear_rollback_refuses_an_incomplete_trail() {
        let dev = dev();
        let (tail, conv_dim, v_heads, hd) = (3usize, 6usize, 2usize, 4usize);
        let mut c = fresh_linear(tail, conv_dim, v_heads, hd, &dev);
        let ckpt = c.checkpoint(4).unwrap();
        let two: Vec<(Tensor, Tensor)> = (0..2)
            .map(|p| linear_state_for(p, tail, conv_dim, v_heads, hd, &dev))
            .collect();
        c.advance_linear(2, two).unwrap();
        assert!(c.rollback(&ckpt, 0, 4, 3).is_err());
    }

    /// Stepping past the span a checkpoint reserved is refused: the trail would
    /// hold states for tokens no rollback can name.
    #[test]
    fn linear_advance_refuses_to_overrun_the_span() {
        let dev = dev();
        let (tail, conv_dim, v_heads, hd) = (3usize, 6usize, 2usize, 4usize);
        let mut c = fresh_linear(tail, conv_dim, v_heads, hd, &dev);
        let _ = c.checkpoint(2).unwrap();
        let three: Vec<(Tensor, Tensor)> = (0..3)
            .map(|p| linear_state_for(p, tail, conv_dim, v_heads, hd, &dev))
            .collect();
        assert!(c.advance_linear(3, three).is_err());
    }

    /// A conversation's state is (KV rows) + (recurrent state) as one unit, so
    /// the snapshot record has to carry the DeltaNet layers through a framed
    /// round trip and back onto the device unchanged.
    #[test]
    fn linear_state_survives_a_host_snapshot_round_trip() {
        let dev = dev();
        let (tail, conv_dim, v_heads, hd) = (3usize, 6usize, 2usize, 4usize);
        let mut linear = fresh_linear(tail, conv_dim, v_heads, hd, &dev);
        let state: Vec<(Tensor, Tensor)> = (0..5)
            .map(|p| linear_state_for(p, tail, conv_dim, v_heads, hd, &dev))
            .collect();
        linear.advance_linear(5, state).unwrap();
        let want = linear_values(&linear);

        let mut caches = vec![fresh_full(16, 2, 8, &dev), linear];
        let snapshot = CacheSnapshot::new(
            5,
            caches
                .iter()
                .map(LayerCache::snapshot)
                .collect::<Result<_>>()
                .unwrap(),
        );
        let host = snapshot.to_host().unwrap();

        let mut framed = Vec::new();
        host.write_to(&mut framed).unwrap();
        assert_eq!(framed.len(), host.serialized_len());
        let read = HostSnapshot::read_from(&framed[..], framed.len() as u64).unwrap();
        assert_eq!(read.pos, 5);

        read.check_restorable(&caches, 5).unwrap();
        let rebuilt = read.to_snapshot(&dev).unwrap();
        // Clobber the live state so the restore has something to put back.
        caches[1].reset().unwrap();
        assert_ne!(linear_values(&caches[1]), want, "reset really cleared it");
        for (cache, layer) in caches.iter_mut().zip(rebuilt.layers()) {
            cache.restore(layer, 5).unwrap();
        }
        assert_eq!(linear_values(&caches[1]), want);
        assert_eq!(caches[1].len(), 5);
    }

    /// A record whose declared state shape does not describe its bytes is
    /// refused where it is read, not where it would later be uploaded.
    #[test]
    fn linear_records_reject_shape_and_length_lies() {
        let short = HostLayerSnapshot::Linear {
            conv: vec![0u8; 4 * 3 * 6],
            delta: vec![0u8; 4 * 2 * 4 * 4 - 4],
            conv_shape: (3, 6),
            delta_shape: (2, 4, 4),
        };
        assert!(HostSnapshot::new(0, vec![short]).is_err());

        let empty = HostLayerSnapshot::Linear {
            conv: Vec::new(),
            delta: Vec::new(),
            conv_shape: (0, 6),
            delta_shape: (2, 4, 4),
        };
        assert!(HostSnapshot::new(0, vec![empty]).is_err());
    }
}
