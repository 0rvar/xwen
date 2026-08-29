# P2 territory map — assembling a second trunk (`Qwen4ExpModel`)

Companion to the "P2 plan" section of docs/qwen4exp-port.md: the file:line map
its four decisions rest on, read against HEAD on 2026-08-29.
Point-in-time — every file:line ref below drifts; re-verify before acting on one.

Verified against current HEAD on 2026-08-29. The 2026-08-26 seams-map is still
directionally right, but **three of its predicted edits have already landed**
(split-GGUF loader, the Option-shaped config fields, the `ZGate`/`moe_sum_floor`
enums), and one of its numbers is badly off (call-site count).

## 1. `src/model.rs` — the stack

- `XwenModel::load(gguf: Arc<GgufFile>, runner: ExpertRunner, max_ctx: usize) -> Result<Self>`
  at `src/model.rs:166`. Order: `XwenConfig::from_gguf` -> `check_arch` ->
  `Weights::from_gguf(gguf.clone())` -> rope -> embed dequant -> per-layer loop
  (`:224-251`) -> `output_norm` + `qlinear_with_buffer("output")` (`:271-272`) ->
  `register_views` per shard (`:277`) -> `warn_if_over_budget` (`:280`).
- Blocks are constructed from a **prefixed loader handle**, not pre-fetched
  tensors: `let lw = w.pp(format!("blk.{il}"))` (`:225`), then
  - `AttnBlock::new(&lw, &cfg, il, rope.clone(), attn_weights)` (`:228`; def
    `src/attention.rs:218`) — needs `il` only for `cfg.n_head(il)`; loads
    `attn_q/attn_k/attn_v/attn_output` via `Proj::load` and
    `attn_q_norm/attn_k_norm` via `w.dense_f32`.
  - `LinearAttnBlock::new(&lw, &cfg, attn_weights)` (`:230`; def
    `src/linear_attn.rs:119`) — no layer index; loads `attn_qkv`, `attn_gate`,
    `ssm_out` (Proj), `ssm_conv1d`, `ssm_beta`, `ssm_alpha`, `ssm_a`, `ssm_dt`,
    `ssm_norm`.
  - `MoeBlock::new(&lw, &cfg, runner)` (`:234`; def `src/moe.rs:53`), which
    internally builds `SharedExpert::new(w)` (`src/moe.rs:422`) off the same handle.
  - `DenseMlp::new(&lw)` (`:233`).
  - `LayerCache::new(&cfg, il, kv_slots, &device)` (`:250`).
- `fn run_stack(&mut self, tokens: &Tensor, pos: usize) -> Result<(Tensor,
  Vec<(String, Tensor)>, Vec<(usize, Tensor)>)>` at `src/model.rs:356` (the audit
  said 337 — same function, shifted). Returns post-`output_norm` hidden + parity
  taps + spec taps. Per-layer body `:424-467`: `attn_norm` -> mixer -> `x + attn`
  -> `ffn_norm` -> ffn -> `ffn_inp + ffn_out`, taps via the `tap!` macro (`:396`),
  `l_out` tap at `:457`, spec capture at `:462`. Prefill mask hoisted once at
  `:412-422` via `build_prefill_mask` (`:489`), passed as `Option<&PrefillMask>`
  to every full-attn layer.
- `forward(&mut self, tokens: &Tensor, pos: usize) -> Result<Tensor>` (`:515`,
  last-position `[vocab]`); `forward_all_logits(&mut self, tokens, pos) ->
  Result<Tensor>` (`:556`, `[seq, vocab]`, clears parity taps deliberately,
  publishes spec taps). `take_post_norm_hidden(&mut self) -> Option<Tensor>`
  (`:666`), armed by `set_keep_post_norm` (`:657`), filled at `:480`.
- Caches are a flat `Vec<LayerCache>` (`:78`), threaded per layer by index
  (`:426`), and driven in lockstep — `cache_len()` reads layer 0 (`:804`).
  `kv_checkpoint(&mut self, span)` (`:757`) maps `LayerCache::checkpoint` over all
  layers; `kv_rollback(&ckpt, commit)` (`:784`) zips; `take_cache_snapshot`
  (`:812`) / `restore_cache_snapshot` (`:827`) / `export_full_kv` (`:854`) /
  `import_full_kv` (`:895`) / `reset_cache` (`:910`) / `grow_kv_capacity` (`:930`)
  all follow the same map-over-`self.caches` shape. `warn_if_over_budget` is
  `src/model.rs:995`, and its recurrent-state term (`:1008-1017`) still hardcodes
  conv+delta only — a PLE conv plane (10240x9 f32 ~ 368 KB/layer) and the id
  history need adding there.
- `_weights_mmap: Vec<Arc<MmapSource>>` (`:146`) — **already a Vec**, split-file
  ready.

## 2. Plumbing seam (enum vs trait)

The audit's "~25 call sites" is wrong by 3.5x. `src/generate.rs` holds a concrete
`model: XwenModel` (`:34`), constructed in `Generator::new` (`:1378`) and
`Generator::load` (`:1418`). **87 `model.*` call sites**, across 26 distinct
methods:

`forward`(10), `set_phase`(8), `device`(8), `take_spec_taps`(7), `reset_cache`(7),
`max_ctx`(7), `cache_len`(5), `embed_ids`(4), `dump_stack_profile`(4),
`reset_stack_profile`(3), `kv_rollback`(3), `kv_checkpoint`(3),
`forward_all_logits`(3), `take_post_norm_hidden`(2), `set_spec_taps`(2),
`set_keep_post_norm`(2), `lm_head`(2), `config`(2), and one each of
`take_cache_snapshot`, `restore_cache_snapshot`, `lm_head_row`, `import_full_kv`,
`export_full_kv`, `embed_rows`, `checkpoint_id`, `check_importable`. Plus free
functions taking the type directly: `take_post_norm(model: &mut XwenModel)`
(`:3744`), and `&XwenModel` at `:3765` and `:3810`.

Good news for scoping: **`src/serve/engine.rs` never names `XwenModel`** — it holds
`generator: Generator` (`:374`, built at `:495`) and reaches the model only through
`Generator` methods. `src/batch.rs` mentions it only in doc comments (`:13`,
`:1455`). So the trunk abstraction has exactly three owners: `Generator`,
`src/bin/logits-dump.rs` (`:46, 443, 494, 620, 675, 688, 751` — uses `forward`,
`forward_all_logits`, `max_ctx`, `set_tap_capture`,
`attn_mm/attn_dtype/attn_decode`), and `src/bin/spec-verify-bench.rs` (`:48, 121,
267` — `forward`, `forward_all_logits`, `kv_checkpoint/rollback`, snapshot pair,
profile).

Recommendation: an **enum** (`Trunk::{Xwen, Qwen4Exp}`) with a delegating impl.
26 methods is a large trait, all of them concrete-shaped, and the two bins want
`attn_*` provenance accessors that qwen4exp will answer differently.

## 3. Recurrent state / caches

`src/kv_cache.rs`: `LayerCache` (`:21`) = `Full{k,v,len}` /
`Swa{k,v,len,window}` / `Linear{conv, delta, len, trail: Vec<(Tensor,Tensor)>,
armed: Option<usize>}` (`:39-55`). Conv plane is `[conv_kernel-1, conv_dim]` f32,
delta `[v_heads, head_dim, head_dim]` f32, both allocated in
`LayerCache::new(cfg, il, slots, device)` (`:161`), which matches on
`cfg.layer_kind(il)` — the natural PLE attach point (extra `Option` planes on the
`Linear` arm, per the audit's preference). Accessors that would need PLE siblings:
`linear_state()` (`:190`), `linear_trail_armed()` (`:200`),
`advance_linear(n_tokens, states)` (`:208`).

Enums/tags that grow:
- `LayerCheckpoint` (`:723`) — `Full` / `Swa{k,v}` / `Linear{conv,delta}`
- `LayerSnapshot` (`:741`) — `Full` / `Swa{k,v,window}` / `Linear{conv,delta}`
- `CacheSnapshot{pos, layers}` (`:763`, `pos()` at `:774`) — the sequence-level
  2-id token history belongs beside `pos`, not per layer
- `HostLayerSnapshot` (`:1015`) — `Full` / `Swa{k,v,shape,window}` /
  `Linear{conv,delta,conv_shape,delta_shape}` (byte planes + shapes)
- disk tags `LAYER_FULL/LAYER_SWA/LAYER_LINEAR = 0/1/2` (`:979-981`); a new
  `LAYER_*` tag correctly rejects on old readers
- validator `check_linear_layer` (`:1040`), `check_ring_layer` (`:1071`)
- record caps `MAX_STORED_STATE_BYTES` (1<<26), `MAX_STORED_V_HEADS` (512),
  `MAX_STORED_CONV_DIM` (1<<17 — covers 10240), `MAX_STORED_CONV_TAIL` (16 —
  covers 9) at `:984-993`

Plane readers are f32-typed (`plane_of_up_to`, `:948`; `f32_bytes`/`f32_tensor`
`:1784/:1805`), so the u32 id history needs its own reader + validator, as
predicted.

## 4. `LinearAttnBlock` / the D9 ZGate seam

`forward(&self, x_normed: &Tensor, cache: &mut LayerCache) -> Result<Tensor>`
(`src/linear_attn.rs:190`) picks fused vs classic by
`head_dim == ops::DELTA_HEAD_DIM && !delta_classic() && device.is_metal()`
(`:191-193`).

**There are TWO gate sites, not one.**
- Classic: `let o = (o * silu(&z)?...)` at `src/linear_attn.rs:360`.
- Fused: the silu is baked into the Metal kernel — `src/ops/delta.metal:229`,
  `dst[idx] = ((x/den) * w[d]) * (zv / (1 + exp(-zv)));` reached through
  `ops::delta_gnorm(o, z, w, eps)` (`src/ops/delta.rs:46`, dispatch
  `src/ops/dispatch.rs:3263`, pipeline name at `:3292`).

D9 therefore needs a gate selector threaded into `delta_gnorm`'s signature (a
function constant or a second pipeline name) **and** the classic site.
`Arch::z_gate()` exists (`src/config.rs:99`, returns `ZGate::Sigmoid` for
Qwen4Exp) but **has zero consumers outside `config.rs`** — `LinearAttnBlock::new`
does not read it yet. Same for `Arch::moe_sum_floor()` (`:110`). Only
`src/qwen4exp/ref_hc.rs` has its own `ZGateRef` (`:43`).

## 5. `MoeBlock` / D10

`MoeBlock::new(w, cfg, runner)` (`src/moe.rs:53`): `n_expert` is read from the
router tensor's shape (`:55`), `n_expert_used` from `cfg` (`:66`). The floor
constant `WEIGHTS_SUM_FLOOR: f64 = 6.103515625e-5` is `src/moe.rs:14`, consumed at
**two** sites: the fused kernel arg (`:88`) and the candle `clamp` (`:142`, inside
the free fn `route_from_logits(logits, n_expert_used)` at `:131`). D10 = a
`sum_floor: f32` field on `MoeBlock` plus a parameter on `route_from_logits`; 0.0
is a legal clamp lower bound, so no branch is needed.

Caps confirmed for 512 experts / top-10: `MOE_ROUTER_MAX_EXPERTS = 512`
(`src/ops/dispatch.rs:2337`), `MOE_ROUTER_MAX_TOP_K = 32` (`:2339`), gate
`moe_router_supported` (`src/ops/moe_glue.rs:29-35`) tests
`n_expert.next_power_of_two() <= 512` — 512 is itself a power of two, so it passes
exactly (a >512-expert file falls back gracefully rather than failing).
`FusedExperts` has no expert-count ceiling (`src/moe.rs:219`); 640-wide experts are
unconstrained. `SharedExpert::new` already normalizes the `[1, hidden]` vs
`[hidden]` gate to `[hidden,1]` (`src/moe.rs:425-427`), so qwen4exp's `[1,2560]`
gate weight is handled. `gate_logits` returns the RAW pre-sigmoid `[seq,1]`
(`src/moe.rs:468`); the sigmoid lives with whoever applies the gate
(`forward` `:441`, or the fused epilogue).

## 6. `AttnBlock` / D11

`forward(&self, x_normed: &Tensor, cache: &mut LayerCache, pos: usize,
mask: Option<&PrefillMask>) -> Result<Tensor>` (`src/attention.rs:305`) — yes, it
takes `Option<&PrefillMask>`.

Decode is chosen by `seq == 1` at four independent sites (`:358` layout reshape,
`:388` rope out-dtype, `:438` gate permute, `:450` output regroup) — there is no
single "decode path" function. `PrefillMask{sdpa, raw}` (`:30`, built `:35`,
hoisted by `model.rs:412-422`).

The QSA overlay slot: `cache.append(&k16, &v16)` returns `(k_all, v_all)`
(~`:404`), then `sdpa_attention(&q, &k_all, &v_all, mask.map(|m| &m.sdpa), scale)`
(`:424`), non-Metal falling to `manual_attention` (`:426`). A per-query block mask
is cheapest as an extra additive plane composed into `PrefillMask` before that
call; a K/V row gather would slot between `append` and `sdpa_attention`. Note sdpa
is candle's — head dim 256, and the vendored flash kernel is compiled at 128 only,
so it is not a route here (`:415-423` comment).

`Rope` (`src/rope.rs:12`): public `apply(&self, q, k, pos) -> Result<(Tensor,
Tensor)>` (`:96`) and `apply_dt(&self, q, k, pos, q_dtype, k_dtype)` (`:105`); the
workhorse `fn rotate(&self, x: &Tensor, pos: usize, out_dtype: DType) ->
Result<Tensor>` (`:127`) is **private** and assumes consecutive positions —
`cos.narrow(0, pos, seq)` (`:145`) on the candle chain,
`ops::rope_neox(x, &self.cos, &self.sin, pos, self.n_rot, out_dtype)` (`:141`) on
the fused path, scalar `pos` on both. QSA's block-first-position rope needs a new
`rotate_at(&self, x, positions: &Tensor)` variant; the tables themselves
(`cos`/`sin`, `[max_ctx, half]`, `src/rope.rs:90-92`) are directly gatherable.

## 7. `src/config.rs` — already done

`Arch::Qwen4Exp` (`:52`), `key()`->`"qwen4exp"` (`:77`), `model()`->`None` (`:93`),
`z_gate()`->`ZGate::Sigmoid` (`:99`), `moe_sum_floor()`->`0.0` (`:113`).
`XwenConfig.qwen4exp: Option<Qwen4ExpConfig>` (`:167`).

`Qwen4ExpConfig` (`:172`): `hc_count`, `hc_low_rank`, `indexer_heads`,
`indexer_head_dim`, `indexer_top_k`, `indexer_compress_ratio`,
`ple: Option<PleConfig>`. Parsed at `:402-419` from
`<arch>.hyper_connection.{count,low_rank}` and
`<arch>.attention.indexer.{head_count,key_length,top_k}` plus a reduced
`compress_ratios` array.

`PleConfig` (`:199`): `layers: Vec<usize>` (0-based, converter-shifted),
`ngram_size`, `heads_per_ngram`, `conv_kernel`, `eos_token_id` (248044),
`image_token_id: Option<u32>`, `row_dim`, `layer_multipliers: Vec<u64>`,
`head_vocab_sizes: Vec<u64>`, `head_offsets: Vec<u64>`. `from_gguf` (`:424`)
returns `None` when `<arch>.ple.layers` is absent/empty and validates head counts,
nonzero moduli and contiguous offsets.

`Meta` gained `usize_array` (`:674`) and `u64_array` (`:688`) — the audit's "no
i64/u64-array accessor" gap is closed. Shared geometry already routes qwen4exp
through the MoE arm for expert keys (`:329`) and reads `ssm.*` under their real
meanings (`:305-324`). **Nothing obvious is missing for P2**; the only things a
qwen4exp trunk needs that are not config-derived are per-tensor (hc widths are
`hc_count * hidden`).

## 8. Loading primitives

`Weights` (`src/gguf.rs:891`) still holds `Arc<GgufFile>` — the split façade landed
*inside* `GgufFile` instead (`shards: Vec<Shard>`, `shard_of: HashMap<String,
usize>`, `shard_for()` `:379`, `mmap_sources()` `:371`, `mmap_source()` `:357`
which asserts single-file). So the audit's "highest blast radius edit" never hit
`Weights`.

Accessors (all on the `pp`-prefixed handle, `name()` appends `.weight`):
`qtensor` (`:932`), `qlinear` (`:943`), `qlinear_with_buffer` (`:971`, returns
`(QLinear, Option<Arc<Buffer>>, GgmlDType)`), `qlinear_with_plane` (`:1044`),
`rms_norm` (`:1057`), `dense_f32` (`:1064`), `dense_f16` (`:1083`, mmap-aliases an
F16-stored tensor via `f16_alias_tensor`), `attn_proj` (`:1132`, returns
`(Tensor, Option<AttnQ8>)`), `dense_f32_any` (`:1354`, for non-`.weight` names
like `ssm_a`/`ssm_dt`), `expert_stack` (`:1221`), `expert_qtensors` (`:1383`),
`has` (`:920`).

A quantized tensor becomes a device `QMatMul` through `Weights::qtensor` ->
`QLinear::from_qtensor` (`:761`) / `qlinear` (`:943`); `QLinear::forward` (`:723`).

BF16 (indexer) path: `dense_alias_tensor(src, device, abs_off, out_dim, in_dim,
dtype)` is `pub(crate)` at `src/gguf.rs:1451` and covers both 2-byte dtypes
(assert at `:1464`). The BF16 wrapper `bf16_alias_tensor(src, content, name,
device)` is **private to `src/dflash.rs:1516`** (with `ensure_bf16_fits_f16` at
`:1538`, calling `src.bytes(abs_off, len)`) — P2 should lift it into `gguf.rs`
rather than copy it. Kernel: `ops::matmul_bf16` (`src/ops/bf16.rs`), dispatched by
weight dtype (`src/dflash.rs:1453-1455`). The dflash load loop pattern (alias-or-
copy closure, then ONE batched `register_views`) is `src/dflash.rs:510-537`. Note
`DflashDrafter::load` uses `gguf.path` (`:513`), which **panics on a split GGUF**
(`SingleFilePath`, `src/gguf.rs:284-300`) — fine for single-file sidecars, but a
qwen4exp loader must not copy that idiom.

PLE raw table reads: `MmapSource::bytes(&self, abs_off: usize, len: usize) ->
Result<&[u8]>` is `pub(crate)` at `src/gguf.rs:172` — usable as-is, never
`QTensor::dequantize`.

`CheckpointId::compute` (`src/gguf.rs:220`) hashes the metadata section only; on a
split open that is shard 0's KV block, so split identity is stable as the audit
assumed.

## 9. The Qwen4Exp refusals to replace

Exactly two, plus one absence:
- `XwenModel::check_arch` (`src/model.rs:159-164`):
  `Arch::Qwen4Exp => anyhow::bail!("the qwen4exp graph is not built by XwenModel")`,
  called at `:168` before the rope table and the ~1.2 GB embedding dequant; pinned
  by the test `qwen4exp_arch_is_refused_before_any_tensor_work` (`:1045`). The
  `unreachable!` FFN arm at `:238-240` only keeps the match exhaustive.
- `Arch::model()` returns `None` for Qwen4Exp (`src/config.rs:93`), because
  `hub::Model` (`src/hub.rs:29-40`) has only three variants and `MODELS: [Model;
  3]` (`:44`) — the deferred D12 registry entry.
- No other module refuses or branches on qwen4exp: the only occurrence outside
  `src/config.rs`, `src/model.rs` and `src/qwen4exp/` is a test-fixture field
  `qwen4exp: None` at `src/serve/engine.rs:6079`.

## Ops inventory for P2 (exists vs. new)

`src/ops/` has: delta (`delta_conv(state, qkv, w)` `delta.rs:27` — **no dilation
parameter, silu folded in**; `delta_ba` `:38`; `delta_scan` / `delta_scan_with_
trail`; `delta_gnorm` `:46`; l2norm), rope (`rope_neox`, scalar pos), attn_glue
(`permute_01`, `permute_01_f16`, `cast_f16`), moe_glue (`moe_router`,
`moe_epilogue`), `silu_mul`, the f16/bf16/q8/mm_id/mv_id/dense_mm matmuls, and
flash (128-only). **No grouped RMSNorm, no dilated conv, no partial top-k, no
gathered rope** — matching the audit. The PLE conv is genuinely new code
(host-side acceptable for P2 at one layer).

P1's `ref_*` oracles are live: `src/qwen4exp/ref_hc.rs`
(`grouped_rms_norm(_batch)` `:103`/`:131`, `gated_rms_norm(_batch)` `:152`/`:171`,
`GatedResidualRef` `:263`, `HcMixerRef` `:394`, `ZGateRef` `:43`),
`src/qwen4exp/ref_ple.rs` (`PleHashRef` `:32`, `PleIntermediates` `:212`,
`PleLayerRef` `:235`, `gate_function_probe` `:161`), `src/qwen4exp/ref_qsa.rs`
(`QsaIndexerRef` `:70`, `select` `:257`). Module doc: `src/qwen4exp/mod.rs:1-11`.

## Parity taps

`tap!` (`src/model.rs:396`) keys on `attn_norm`/`attn_o_proj`/`ffn_inp`/`ffn_norm`/
`ffn_out`/`l_out` + `h_nextn` (`:471`) + `result_norm`/`result_output`
(`:537-538`); `spec_taps` capture the same `l_out` (`:462`, ordered by
`order_spec_taps` `:966`); `post_norm_hidden` hangs off `output_norm`'s output
(`:480`). None of `l_out`/`output_norm` exists on qwen4exp, so the new module
defines its own convention — the pre-`output_hc` 10240-wide carrier is the
analogue — and `take_taps` (`:615`) / `take_spec_taps` (`:648`) /
`take_post_norm_hidden` (`:666`) are the three drain methods a `Trunk` enum must
expose for `logits-dump` and the drafter glue to keep working.
