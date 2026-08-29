# Qwen3.8-Flash-Next (`qwen4exp`) port — working doc

Arc opened 2026-08-26, the day the checkpoint dropped. This file is the canonical
externalized context for the port: confirmed spec, decisions with their why, traps,
phase plan, and a running progress log. TODO.md's "Qwen3.8-Flash-Next port" section
is the ledger pointer; this doc is the substance. When the arc ships, decisions
migrate to docs/decisions.md and the story to docs/log.md per the doc system.

Hard constraint carried from the top: the three existing checkpoints
(Qwen3.6-27B, Qwen3.6-35B-A3B, Qwen3.8-27B) keep working at their current
throughput. No shared hot path may grow a per-token branch for qwen4exp's sake —
divergence is resolved at construction time.

## Status

- Phase: **P0 COMPLETE (2026-08-26); arc RESUMED 2026-08-29.** Current step:
  re-vendor the llama.cpp material at the merged `6fe749801` and download the
  first runnable file (UD-Q4_K_XL, D12); then P1 = Rust reference
  implementations of hyper-connections, PLE, and the QSA indexer against
  tests/fixtures/qwen4exp/ (D5), plus the deferred registry entry (a Q4-class
  file to name/size it now exists — D12).
- Runnable weights: `UD-Q4_K_XL` (111.33 GB, 4 shards) downloading into the HF
  cache since 2026-08-29. Full published ladder surveyed below. Metadata-only
  shard 1 (10.9 MB) stays usable for loader dev.
- llama.cpp support: PR #27742 **MERGED** into master 2026-08-27 as squash
  `6c84c7d5d`, plus follow-up `6fe749801` on 08-28. D4's re-pin gate is met;
  `reference/llama.cpp` (e9fa0781) stays frozen for 3.6/3.8 and a second clone
  at `6fe749801` becomes the qwen4exp oracle (see D4 update).

## The model in one paragraph

`qwen4_exp` (GGUF arch `qwen4exp`, HF `Qwen4ExpForConditionalGeneration`,
transformers main has first-class support — no trust_remote_code). 48 layers,
hidden 2560, ctx 262144 (YaRN to 1M). 125B MoE trunk + 51.2B PLE n-gram table +
4B MTP head = 180B on disk (360 GB BF16, 131 shards); ~6B active/token. Layer
cadence is the familiar `(i+1) % 4 == 0`: 36 gated-DeltaNet layers, 12 sparse-
attention (QSA) layers, every layer MoE (512 experts top-10 + shared expert).
Residual skeleton is hyper-connections (4 streams). Tokenizer is exactly
Qwen3.8-27B's; chat template is a near-clone of the 3.8 dialect plus vision
items. Vision is an inline ViT, cleanly droppable for text-only.

## Decisions (numbered, dated; migrate to decisions.md when the arc ships)

- **D1 (2026-08-26) Third arch, composition over forking.** `Arch::Qwen4Exp` gets
  its own graph module. Shared blocks (DeltaNet, attention internals, MoE glue,
  rope) are reused by composition and parameterized only where the math actually
  differs (e.g. gated-norm z-activation silu vs sigmoid, an enum resolved at
  construction). The qwen35/qwen35moe forward paths are not edited. Why: retains
  the existing checkpoints' perf by construction — their code is untouched — and
  avoids a divergent copy of DeltaNet that would rot.
- **D2 (2026-08-26) PLE table lives on the CPU side.** The `[320001536, 160]`
  table is mmap'd file-backed; hashing, the 16-row gather, and row dequant happen
  host-side per token; the result (2560 floats/token) feeds the GPU graph where
  key_proj/value_proj run. Why: the table is pure `get_rows` — no matmul ever
  touches it; row addresses depend only on token ids (prefetchable); page cache
  handles hot/cold. GPU residency stays reserved for the ~70 GB trunk. Engram
  paper measured 2.8% overhead for a host-resident 100B table; ours is smaller.
- **D3 (2026-08-26) Weights path: Unsloth first, self-converted blessed file for
  parity.** Dev and first testing against Unsloth's Q4-class UD file when it
  lands (requires IQ4_NL dequant — their files carry the table as IQ4_NL even in
  Q4 mixes). The eventual parity target is a self-converted file with a mix we
  control (candidate: Q4_K/Q6_K/Q8_0 trunk + Q4_K or IQ4_NL table), because
  parity floors are calibrated per quant mix and Unsloth's UD mixes are
  per-layer heterogeneous (IQ1..IQ4 spread in the IQ1_S file).
- **D4 (2026-08-26) Oracle policy.** `reference/llama.cpp` (pinned e9fa0781)
  stays frozen — it gates the 3.6/3.8 parity cycles. PR #27742's qwen4exp files
  are vendored read-only under `reference/qwen4exp/` with provenance, as reading
  material. A separate buildable clone of the PR branch comes only when we need
  executable comparison, and re-pinning the main oracle waits for the PR to merge
  plus Qwen's promised independent numeric check. An unreviewed AI-drafted branch
  is not a frozen correctness oracle.
  - **Update 2026-08-29 — the merge half of the gate is met.** PR #27742 merged
    into ggml-org/llama.cpp master 2026-08-27T19:32Z as squash `6c84c7d5d` (PR
    head `eaf9376557`, 65 commits); follow-up `6fe749801` "model: qwen4exp:
    reduce number of graph splits (#27880)" landed 08-28; master was
    `17252c769` at survey time. Our vendored `bea3b12d` is being re-vendored at
    `6fe749801` (separate unit, in progress). The other half — Qwen's
    independent numeric check — was NEVER posted; what we have is the PR body's
    own numerics: wikitext-2 ppl 4.0068±0.0227 vs 4.0126 reference, top-1
    agreement 98.0%, QSA bit-identical to dense below the 2048 budget on
    BF16/F32 (max logit delta 0.0 over 2051 rows) and diverging at ~3% of
    positions at 8192, indexer selection 0.975 mean Jaccard against a 0.991
    precision floor. Author caveats to carry: quantized models are NOT
    bit-identical across the QSA boundary (UD-IQ1_S max logit delta 2.84e-3),
    and `test-llama-archs -a qwen4exp` is weak — its synthetic model has no PLE
    tensors, and it is blind to the GDN fused-QKV segmentation convention
    (three plausible segmentations, one correct). Fixes that landed in-thread
    before merge: Hadamard rotation for quantized KV in QSA (q8_0 KV then
    matches f16), a raised `graph_max_nodes` budget, a multi-slot
    indexer/attention desync via the server prompt cache, and "bias the QSA
    selection per block, not per cell". PLE is implemented Gemma-3n-style as
    one host-side get_rows gather table, CPU-resident on CUDA automatically,
    mmap-backed — independent confirmation of D2. Reported perf, DGX Spark GB10,
    UD-Q4_K_XL: 24-25 tok/s decode, 70-99 tok/s prefill, 27.5 GB CPU + 78 GB
    CUDA buffer. Oracle layout after the re-pin: `reference/llama.cpp`
    (e9fa0781) STAYS frozen for the 3.6/3.8 parity cycles, and a SECOND clone
    at `6fe749801` is the qwen4exp oracle — where it lives and whether
    `scripts/build-llamacpp.sh` grows a target argument are Open questions.
- **D5 (2026-08-26) Reference-first for every new component.** Hyper-connections,
  the QSA indexer, and PLE each get a frozen CPU f32 reference implementation
  with fixture tests before any Metal work, mirroring the ReferenceExperts
  pattern. Fixtures come from the transformers modeling code (the one executable
  ground truth that exists today).
- **D6 (2026-08-26) Scope: text-only, MTP deferred, serve after CLI.** Vision is
  dropped (masked_scatter injection, empty deepstack — clean cut; mrope collapses
  to NEoX-64 for text exactly as on 3.6/3.8). The MTP head has no transformers
  implementation and its forward semantics are unconfirmed (separate
  `fc_embedding`/`fc_hidden` projections, NOT 3.8's concat `eh_proj`) — deferred
  to a drafting arc once vLLM/SGLang or the tech report settles it. Serve
  integration follows CLI bring-up.
- **D8 (2026-08-26) GGUF parsing ownership + IQ4_NL in three classes.** candle's
  `GgmlDType` cannot even PARSE a file containing an IQ tensor (unknown dtype →
  `Content::read` fails before any kernel question), so the split-GGUF loader
  (already xwen-owned code) also owns the tensor-table/dtype parsing; the pinned
  candle stays unpatched. IQ4_NL work splits: (1) metadata visibility — in the
  loader; (2) CPU row dequant — needed only for the PLE table per D2, ~small;
  (3) Metal matmul kernels (mv_id/mm_id) — needed only if a matmul weight is
  IQ4_NL; DEFERRED, and D3's self-converted blessed file can choose Q4_K for
  the table + down_exps to avoid class 3 entirely.
- **D9 (2026-08-26) DeltaNet z-gate as a construction-time enum.** `ZGate
  {Silu, Sigmoid}` on `LinearAttnBlock`: reference path branches at the one
  silu(z) line; fused path gets a `kernel_delta_gnorm_sigmoid` sibling selected
  by name at dispatch. Existing checkpoints construct Silu — their kernel and
  code path unchanged.
- **D10 (2026-08-26) MoE renorm clamp becomes a field.** `sum_floor` on
  `MoeBlock` set at construction (existing checkpoints keep 6.103515625e-5,
  qwen4exp passes 0.0). The fused router already takes it as a runtime param;
  only the candle-chain fallback hardcodes the const.
- **D11 (2026-08-26, provisional) QSA decode via K/V row gather.** candle's
  sdpa VECTOR kernel (the seq==1 route) is compiled without mask support and
  SILENTLY IGNORES a mask tensor — so masked decode cannot ride the stock sdpa
  path. P2 correctness: gather the ≤2051 selected K/V rows into a packed
  contiguous view and run maskless sdpa over it. P3 may replace with a vendored
  masked-decode kernel if the gather (~25 MB/token across 12 layers) shows up
  in profiles. Prefill overlays the QSA mask via the existing
  `Option<&PrefillMask>` argument — needs only a device-side mask constructor
  (today's masks are host-built).
- **D12 (2026-08-29) First target file: `UD-Q4_K_XL`.** The only Q4-class trunk
  whose quant types are ones xwen already has kernels for — Q4_K / Q8_0 / F32,
  with IQ4_NL confined to the PLE table (D8 class 2: CPU row dequant only, no
  matmul). `UD-IQ4_XS` would be roomier on 128 GiB (64.88 GB trunk vs 82.53) but
  needs IQ4_XS matmul kernels we don't have — D8 class 3, deferred. 82.53 GB of
  wired trunk plus a demand-paged PLE table plus KV is tight, but it is the same
  file llama.cpp reported 24-25 tok/s decode on for a 128 GB DGX Spark. Fallback
  if it doesn't fit comfortably: D3's self-converted mix. Download started
  2026-08-29 into the HF cache (repo `unsloth/Qwen3.8-Flash-Next-GGUF`, path
  `UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-0000N-of-00004.gguf`).
- **D7 (2026-08-26) Phase plan.** P0 scaffold (split-GGUF loader, config parse,
  registry) → P1 CPU references + fixtures for the three new components → P2
  graph assembly, load a real file, greedy smoke, ppl sanity vs PR #27742's
  claimed 4.0068 → P3 Metal/perf (fusion, IQ4_NL path, prefetch) → P4 serve,
  sampling defaults (presence penalty!), docs, parity harness extension.

## Confirmed spec (from config.json + transformers `modular_qwen4_exp.py` @ main
+ Unsloth GGUF metadata, all read 2026-08-26)

### Carries over from what we ship

- **Gated DeltaNet**: byte-identical geometry to our 27B block — 16 K-heads /
  48 V-heads / head dim 128 / inner 6144 / fused conv width 10240, depthwise
  conv k=4 no-bias causal, silu over the whole fused stream, L2 norm clamp form,
  `beta = sigmoid(b)`, `g = ssm_a * softplus(a + dt_bias)` with pre-baked
  `ssm_a = -exp(A_log)` on the GGUF path, tiled V-head order after conversion.
  **One delta: the gated RMSNorm's z-gate is `sigmoid(z)`, NOT `silu(z)`**
  (`output_gate_type: "sigmoid"`). Silent-garbage if missed.
- **Attention internals**: `attn_q` double-width interleaved `[q,gate]` per head
  (24 heads × 256 × 2 = 12288), k/v 2 heads × 256, QK-RMSNorm(256) before rope,
  scale 1/√256, `out *= sigmoid(gate)` before o_proj. Rope theta 1e7, partial
  0.25 → n_rot 64, mrope sections [11,11,10] interleaved ≡ plain NEoX-64 for
  text-only (same argument as 3.6/3.8; revalidate the sections key at load).
- **MoE machinery**: router softmax over all 512 THEN top-10 THEN renormalize.
  **No sum clamp** (the 6.1e-5 clamp is 3.6-35B-only). Shared expert 640-wide
  scaled by `sigmoid(shared_expert_gate @ x)` — gate weight shape `[1, 2560]`.
  Fused router kernel caps (top_k ≤ 32, n_expert ≤ 512) hold: 512/10 fits.
- **Tokenizer**: base vocab + merges hash-identical to our vendored 3.6 file,
  added tokens through 248076 (= the 3.8 set incl. audio specials). Vocab pad
  248320. No BOS. Stops [248046, 248044] (generation_config; GGUF single-eos as
  usual). Template: 3.8 dialect semantics (reasoning_effort xhigh/medium/low,
  preserve_thinking true, open-`<think>` seeding, `<function=...>` tool XML,
  string-args-raw) + vision items (vision in system message raises).

### New subsystem 1: hyper-connections (every layer; biggest structural change)

- Residual carrier is `[hc_count=4 × 2560] = 10240` wide, seeded by repeating
  the token embedding 4×. There are NO `attn_norm` / `post_attention_norm` /
  final `output_norm` tensors anywhere.
- Per block (attn and mlp each have one), `GatedResidual`:
  `n = hc_norm(stream)` — grouped RMSNorm, groups of 2560;
  `w = sigmoid(up(silu(down(n) / 4)))` (down: 10240→320, up: 320→10240);
  `mixed = mean over streams of (w ⊙ n)` → 2560 (the block input);
  `inject = 2·sigmoid(block_inject_weight(n) / 4)` → [4] per token;
  block output written back as `stream += out ⊗ inject` — **onto the raw
  UN-NORMED stream**.
- Model tail: `hyper_connection_mixer` (same read path, no write) → 2560 →
  lm_head. GGUF tensors: `hc_{attn,ffn}_{norm,down,up,inject}` per layer +
  `output_hc_{norm,down,up}`.
- Norm weights arrive multiply-ready on the GGUF path (converter bakes +1, same
  as every other norm — HF zero-centers them).

### New subsystem 2: QSA (the 12 attention layers)

- Config: `indexer_n_heads 4, indexer_kv_heads 1 (MQA, hard requirement),
  indexer_head_dim 128, indexer_budget 2048, indexer_compress_ratio 4` →
  top-k = 512 four-token blocks.
- `index_qk_proj: 2560 → (4+1)×128 = 640` (GGUF splits it: `indexer.q_proj
  [2560,512]`, `indexer.k_proj [2560,128]`), q/k RMSNorm(128), partial rope over
  the first 64 dims of 128.
- Keys are cached RAW (un-normed, un-roped; BF16, 256 B/token). Per query:
  visible keys → consecutive runs of 4 → mean in fp32 → k_layernorm → rope at
  the position of the block's FIRST token. Queries: q_layernorm then rope at own
  position.
- Score `relu(q @ k_block)ᵀ summed over the 4 q-heads / √128`; top-512 blocks;
  selected blocks' token indices become an additive/boolean mask overlaid on the
  causal mask; then the attention itself is the ordinary blessed path (sdpa —
  upstream explicitly disables flash/flex because of the overlay).
- **The incomplete tail block (< 4 tokens) is ALWAYS visible**, unmasked. Max
  visible tokens/query = 2051. No sliding window. Below-budget contexts (≤ 2048
  visible tokens) are exactly dense attention — PR #27742 reports bit-identical
  there, which is also our cheap dev-time equivalence check.
- config.json's `layer_types` literally says `"full_attention"`; the HF config
  class REWRITES those entries to `"qwen_sparse_attention"`. Trusting the file
  string builds dense attention that runs and quietly degrades quality.

### New subsystem 3: PLE — the 51B n-gram table (one layer)

- `ple_layer_ids: [2]` is ONE-indexed → decoder layer index 1 (a DeltaNet
  layer). One PLE layer in this checkpoint; treat the list as data.
- Hash: orders {2,3} (no unigram), 8 heads each = 16 heads; per-head prime
  vocab ≈ 20,000,003..20,000,171 (16 consecutive primes > 2e7); one flat padded
  table `[320001536, 160]` — GGUF `per_layer_token_embd.weight` (IQ4_NL,
  28.8 GB in Unsloth's file); HF ships it as 128 `shard_N.weight` tensors.
  Multipliers/vocab-sizes/offsets are SHIPPED as I64 buffers (GGUF metadata
  `ple.layer_multipliers` / `ple.head_vocab_sizes` / `ple.head_offsets`) — read
  them, never recompute. Hash: `mixed = t₀·m₀ ⊕ t₁·m₁ (⊕ t₂·m₂)`, row =
  `mixed mod head_vocab + head_offset`, over **RAW token ids** — no NFKC, no
  lowercasing (deviation from the Engram paper).
- Segmenting: the shift-right that forms n-grams never crosses an eos boundary,
  and eos here is the **SCALAR 248044** (`<|endoftext|>`), NOT 248046 — GGUF key
  `ple.eos_token_id`. Wrong id silently corrupts lookups at every turn boundary.
  Requires a 2-token rolling raw-token history as recurrent state.
- Injection (NOT Engram's scalar gate): `emb` = 16 rows concat → 2560;
  `key = norm_key(key_proj(emb))` (2560→10240, viewed [4,2560]);
  `value = value_proj(emb)` (2560→2560);
  gate per stream = `⟨key_s, norm_query(stream)_s⟩ / √2560`, then **signed
  sqrt** (`sign·√|·|`), then sigmoid; `gated = sigmoid(gate_s) · value` per
  stream → 10240; output = `gated + silu(dilated_conv(norm_conv(gated)))` where
  the conv is depthwise k=4 **dilation 3** (state 9 columns); result adds to the
  10240-wide hyper-connection stream BEFORE the attn hyper-connection read.
- A PLE layer therefore carries THREE recurrent states: GDN conv (10240×3), PLE
  conv (10240×9), and the 2-id token history. Snapshot/rollback machinery must
  learn all three.

### Registry / naming / sampling facts

- `general.name` in Unsloth's GGUF: **"Qwen3.8 Flash Next"** (spaces). Official
  full name: `Qwen3.8-Flash-Next`. `identify()` must match both spellings;
  check containment against existing full names both directions (rule from
  hub.rs — a name matching two checkpoints identifies as neither).
- Card sampling: thinking 1.0/0.95/20 (unchanged); non-thinking 0.7/0.80/20
  **plus presence_penalty 1.5** — first checkpoint whose card demands a penalty.
  Our serve layer currently accepts-and-drops penalties (TODO.md 2026-08-19);
  for this checkpoint that's a correctness gap to close in P4. The GGUF also
  bakes `general.sampling.{temp,top_p,top_k}` keys (note: `temp`, not
  `temperature`) — generic converter heuristics off generation_config.json, worth
  a cross-check but the card is authority. Converted GGUFs carry NO
  presence-penalty key at all (the converter only knows repetition_penalty), so
  the 1.5 must be hardcoded per checkpoint like the second stop id is.
- `text_config.eos_token_id` (scalar) = 248044 and bos = 248044 with
  add_bos false; generation stop list unchanged [248046, 248044].

### Split-GGUF layout (Unsloth files; gguf-split standard)

- `<base>-000NN-of-000MM.gguf`, 1-indexed names. Every shard: `split.no`
  (0-indexed!), `split.count`, `split.tensors.count` (TOTAL across shards).
- Shard 0: full KV block (67 keys incl. tokenizer + chat template), **zero
  tensors**. Shards 1..: only the three split.* keys + their own tensor table;
  data offsets restart at 0 per shard.

### Quant facts (Unsloth UD-IQ1_S, first uploaded file)

- ftype 24 (IQ1_S). 1224 tensors. Per-layer heterogeneous: experts IQ1_S..
  IQ2_XXS, `ffn_down_exps` IQ4_NL, attn/ssm Q5_K/Q6_K, hc + shexp Q8_0, indexer
  BF16, norms/routers F32, table IQ4_NL. Expect the Q4-class file to be mostly
  Q4_K/Q5_K/Q6_K with IQ4_NL persisting on the table and possibly down_exps —
  IQ4_NL dequant is unavoidable for the Unsloth path.

### Quant landscape (2026-08-29 — every published GGUF, read from headers)

Qwen published no GGUF. `unsloth/Qwen3.8-Flash-Next-GGUF` is the only full
ladder; all UD variants are imatrix quants (926 entries;
`imatrix_unsloth.gguf_file` sits in the repo). Naming is
`<folder>/Qwen3.8-Flash-Next-<VARIANT>-000NN-of-000MM.gguf`, shards ~50 GB. The
PLE tensor `per_layer_token_embd.weight` is `[160, 320001536]` =
51,200,245,760 elements, so its size is exact per type and the trunk column is
just total minus PLE. Sizes decimal GB.

| variant | shards | total GB | PLE type / GB | trunk GB | note |
| --- | --- | --- | --- | --- | --- |
| UD-IQ1_S | 3 | 72.55 | IQ4_NL / 28.80 | 43.75 | token_embd Q4_K |
| UD-IQ1_M | 3 | 74.54 | IQ4_NL / 28.80 | 45.74 | |
| UD-Q2_K_XL | 3 | 78.87 | IQ4_NL / 28.80 | 50.07 | token_embd Q5_K |
| UD-IQ3_XXS | 3 | 81.96 | IQ4_NL / 28.80 | 53.16 | token_embd Q6_K |
| UD-Q3_K_XL | 3 | 89.99 | IQ4_NL / 28.80 | 61.19 | |
| UD-IQ4_XS | 3 | 93.68 | IQ4_NL / 28.80 | 64.88 | needs IQ4_XS matmul (D8 class 3) |
| **UD-Q4_K_XL** | 4 | 111.33 | IQ4_NL / 28.80 | **82.53** | file_type 15; de-facto default; D12 target |
| UD-Q5_K_XL | 6 | 158.29 | Q8_0 / 54.40 | 103.89 | |
| UD-Q6_K_XL | 6 | 169.17 | Q8_0 / 54.40 | 114.77 | |
| Q8_0 | 6 | 188.23 | Q8_0 / 54.40 | 133.83 | |
| BF16 | 8 | 354.03 | BF16 / 102.40 | 251.63 | shard 3 is the PLE tensor alone |

The repo also carries `mmproj-BF16` / `mmproj-F16` (the vision tower — never
load) and the imatrix file.

- **UD-Q4_K_XL precision policy** (measured, not inferred): `output.weight`,
  `token_embd`, every `attn_output`, `ple_key`, `ple_value` and the
  hyper-connection up/down projections are Q8_0; all norms and `ple_conv1d` are
  F32; the routed experts carry the low bits. Unsloth's docs say the PLE table
  is held at "4-bit minimum" because of its random-access pattern — which is why
  IQ4_NL persists on the table even in the Q4 and Q3 mixes (D8 class 2 is
  unavoidable on the Unsloth path, exactly as D3 predicted).
- **Selected KV from UD-Q4_K_XL shard 1** (cross-check of the spec above, all
  confirming): `general.architecture` qwen4exp, `size_label` "512x56B", 48
  blocks, hidden 2560, 24 Q / 2 KV heads, key/value length 256, 512 experts
  top-10, expert FFN 640, `full_attention_interval` 4, rope theta 1e7,
  `dimension_sections` [11,11,10,0], `rope.dimension_count` 64,
  `hyper_connection.count` 4 / `low_rank` 320, `attention.indexer.{head_count 4,
  key_length 128, top_k 2048}`, `ple.{layers [1], ngram_size 3, heads_per_ngram
  8, conv_kernel 4}` with 16 PLE head vocab slices ~20,000,0xx each,
  `embedding_length_per_layer_input` 160, context 262144, tokenizer pre
  `qwen35`, eos 248046, no BOS, `general.sampling.{temp 1.0, top_p 0.95, top_k
  20}`.
- **Other publishers**: ggml-org has Q8_0 only (2 shards, 162.62 GB, + mmproj);
  lmstudio-community Q4_K_M 119.15 / Q6_K 167.63 / Q8_0 188.21; bartowski
  IQ1_S..IQ3_M (70-93 GB, still uploading as of 08-28); mradermacher a static
  and an i1 ladder. Nothing there beats UD-Q4_K_XL on the kernels-we-have axis.

## Conversion-baked deltas (audit of the vendored PR converter, 2026-08-26)

reference/qwen4exp/ vendors PR #27742 at the sha in PROVENANCE.md: `qwen4exp.cpp`
(graph), `qwen4exp.py` (converter), `conversion-qwen-base.py` (the inherited
Qwen3Next rules the converter subclasses — the +1/-exp/V-reorder logic lives
there, NOT in the qwen4exp file), plus the gguf-py and core-C++ diffs.

- **Norms**: every norm on the GGUF path is multiply-ready (converter bakes +1
  into all `*norm.weight` incl. hc_norm/output_hc_norm/QK-norms/indexer
  layernorms, and explicitly into the three `ple.norm_*`), with the usual
  exemption of `ssm_norm` which was never zero-centered. Same end state as our
  three checkpoints: multiply directly, never add 1.
- **`ssm_a`** pre-negated to `-exp(A_log)`; **V-head order tiled** by the exact
  inherited rule set we already implement (qkv V-rows, attn_gate, alpha/beta/
  a/dt elements, conv1d V-channels, ssm_out COLUMNS). Plain repeat broadcast
  stays correct.
- **PLE table**: the 128 HF shards are streamed into one `[160, 320001536]`
  `per_layer_token_embd.weight`; hash constants are read from the checkpoint's
  I64 buffers (never recomputed — a lazy-cast bypass exists precisely because
  base.py's f32 cast would round the 45-bit multipliers) and written as UINT64
  metadata arrays. `ple.layers` in the GGUF is already 0-BASED (converter does
  the 1-based→0-based shift) — the one-indexed trap applies to config.json only.
- **Indexer** `index_qk_proj` split into `indexer.q_proj`/`k_proj`; kept
  unquantized by the quantize-side skip list (hence BF16 in Unsloth's file).
- **`ple_conv1d` quant is UNPINNED upstream**: not on the skip list, ne0=4 can't
  take 32-block quants, lands F16 via a new fallback branch (the graph casts it
  back to F32). If we self-convert, pin it F32 with `--tensor-type`.
- **MTP skipped entirely** by the converter (no nextn keys); **vision emitted as
  a separate mmproj**, never in the text file. Both match D6.
- `attention.compress_ratios` is synthesized per layer from the raw
  `layer_types` strings; `general.sampling.*` comes from generic
  generation_config heuristics.

### Known llama.cpp-impl divergences (PR quirks, not ground truth — expect them
in oracle diffs, do not copy blindly)

1. **QSA top-k width — CONFIRMED PR-vs-HF DIVERGENCE (settled 2026-08-26 by
   fixture)**: HF selects WHOLE top-k blocks plus the raw tail — a short tail
   admits nothing from the 513th-ranked block; the `budget + ratio - 1` width
   in HF is buffer capacity only, unused slots dropped. The PR's unconditional
   `top_k + ratio - 1` token fill diverges whenever `visible mod ratio ≠
   ratio−1` above budget. xwen follows HF (pinned by
   tests/fixtures/qwen4exp/qsa_indexer.json); expect small oracle diffs vs the
   PR at long contexts. Worth reporting upstream.
2. **Partially-filled non-tail blocks are hard-masked** (-inf) rather than
   pooled over what exists — invisible on a contiguous cache, bites after
   rewind/defrag. Our rollback machinery must not inherit this silently.
3. ~~PLE gate clamp~~ RETRACTED 2026-08-26: HF DOES clamp `|s| ≥ 1e-6` inside
   the signed sqrt (modular line 770, pinned by the fixture gate probe). No
   divergence; implement the clamp.
4. **MoE renorm clamp**: llama.cpp's shared `build_moe_ffn` applies the
   6.103515625e-5 sum clamp unconditionally, including for qwen4exp — the HF
   math has no clamp. Practically a no-op (top-10 softmax sums are far larger);
   recorded so a future parity diff isn't a mystery.

## Traps checklist (all silent-failure class — each becomes a pinned test)

1. GDN z-gate `sigmoid`, not `silu`.
2. PLE eos = 248044 (not 248046) for segmenting; shift never crosses it.
3. PLE hash over raw ids; multipliers/primes/offsets read from metadata.
4. `ple_layer_ids` one-indexed.
5. `layer_types` "full_attention" means QSA (config-class rewrite).
6. Hyper-connection write-back onto the un-normed stream; `/4` inside both
   sigmoid args; `2·sigmoid` on inject; mean (not sum) over streams on read.
7. Signed sqrt in the PLE gate; dilation 3 in the PLE conv.
8. MoE: no renorm clamp; `shared_expert_gate` is `[1,2560]` not `[2560]`.
9. QSA: keys cached raw (pool→norm→rope at query time, block-first position);
   fp32 block mean; the incomplete tail (when one exists) always visible; rope
   on indexer q at own position, 64 of 128 dims. Precision (fixture-pinned):
   when `visible mod ratio == 0` there IS no tail — the query's own complete
   block competes in top-k and can lose, masking the query's own token. Whole
   blocks + tail, never a fixed token count (see divergence #1).
10. Residual stream seeded by repeat×4 of the embedding; final mixer before
    lm_head (no output_norm tensor).
11. `general.name` has spaces; don't let "Qwen3.8" substring-collide with
    "Qwen3.8-27B" in identify(). (Arch-first filtering in identify() makes the
    cross-arch collision impossible by construction; the spaced spelling still
    needs to be an accepted alias.)
12. candle's sdpa vector kernel (q_seq==1) is compiled WITHOUT mask support and
    silently ignores a passed mask — a masked QSA decode through stock sdpa
    runs dense attention with no error. See D11.

## Reuse-seams map (audited 2026-08-26; file:line refs from that audit)

- `AttnBlock`/`LinearAttnBlock`/`MoeBlock`/`SharedExpert` take pre-normed
  `[seq, 2560]` and return pre-residual output — norms and residual adds live
  in `XwenModel::run_stack` (model.rs:337), NOT in the blocks. A new
  `src/qwen4exp.rs` outer loop owning the 4-stream carrier calls them
  unchanged. The `[1,2560]`-vs-`[2560]` shexp gate shape is already handled.
- Parameterize (small): ZGate (D9); MoE sum_floor (D10); `Rope::rotate`
  assumes consecutive positions from a scalar start — QSA ropes pooled block
  keys at block-first positions (stride 4), needs a positions-gather variant;
  `XwenConfig` gains Option-shaped fields (hc/indexer/ple incl. the three
  UINT64 metadata arrays — `Meta` today has no i64/u64-array accessor);
  `register_views`/`_weights_mmap` become Vec for split files;
  `warn_if_over_budget` hardcodes conv+delta as the only recurrent state.
- Plumbing seam: `Generator` holds a concrete `XwenModel` (~25 call sites) —
  needs an enum/trait for a second trunk type; also logits-dump and
  spec-verify-bench binaries.
- Recurrent-state plumbing is wide but mechanical: `LayerCache` + the
  checkpoint/snapshot/host-snapshot/disk record enums (~15 match sites) grow a
  PLE variant (conv 10240×9 + the 2-token id history — the history is
  sequence-level, store beside `CacheSnapshot::pos`, not per layer); new disk
  LAYER_* tag correctly rejects on old readers.
- MoE caps confirmed for 512/top-10: fused router fits EXACTLY at both limits
  (MAX_EXPERTS 512, reduction width 256) — a >512-expert file falls back
  gracefully. FusedExperts/mm_id/mv_id have no expert-count ceiling.
- Ops for P2 (composed, correctness-first): HC read/write and grouped RMSNorm
  from candle primitives (~15 dispatches/layer-pair — fused `hc_mix` is the top
  P3 kernel candidate); QSA top-k via arg_sort (partial top-k kernel is P3);
  block mean-pool + ragged tail is new composed code; indexer q/k norms and
  partial rope reuse existing pieces; BF16 indexer weights ride the
  dflash-established `dense_alias_tensor` + `matmul_bf16` pattern. PLE conv
  needs dilation — `delta_conv` has none and hardcodes silu — new kernel
  (host-side conv acceptable for P2 given one layer).
- PLE table reads ride `MmapSource::bytes` (raw range reads, already
  crate-visible) — never `QTensor::dequantize`.
- Parity taps: `tap!`/`spec_taps`/`post_norm_hidden` hang off `l_out` and
  `output_norm`, neither of which exists on qwen4exp — the new module defines
  its own tap convention (the pre-`output_hc` carrier is the natural analogue).
- Split-file identity: `CheckpointId::compute` hashes ONE file's metadata
  section; for a split file the identity is shard 0's KV block (it carries the
  full metadata and zero tensors). Prefix-cache/disk-tier keying depends on
  this being stable.
- PLE state shape preference: extra optional planes on the `Linear` cache
  variant (a PLE layer IS a DeltaNet layer with extra state — keeps
  `advance_linear`'s lockstep), not a fourth variant. The 2-id token history is
  u32 in an all-f32 plane world; it needs its own plane type + validator.
  `Weights` holds `Arc<GgufFile>` — the split façade changes that field's type,
  the highest-blast-radius edit of the loader work.

## Notes captured at the P0 pause (context that would otherwise live only in
the session that ran P0)

- **Quant-sensitivity hint for D3's self-converted mix**: Qwen's own FP8 repo
  keeps a long `modules_to_not_convert` list in BF16 — lm_head, embed_tokens,
  ALL hyper-connection projections (input_mix_weight_down/up,
  block_inject_weight), ALL GDN projections and conv1d, the indexer, norms,
  routers. That is Qwen telling us which planes they consider
  quantization-sensitive; a self-converted file should keep hc/indexer/GDN
  planes at Q8_0-or-better (Unsloth's IQ1_S mix independently made the same
  call: hc Q8_0, indexer BF16, attn/ssm Q5_K/Q6_K).
- **PLE runtime design (P3)**: eviction is free — the table mmap is read-only
  file-backed, clean pages are the kernel's first reclaim target, and Zipf
  reuse means the resident slice converges on the hot few GB (judge by
  `footprint`, not RSS, as always). The effort goes into PREFETCH: at prefill,
  all row addresses are computable from token ids before layer 0 runs — hash
  everything, dedupe, batch-fault on a background thread; at decode, the
  moment token t is sampled, positions t+1's ~16 rows are known — touch them
  while the trunk runs. Never gate the fetch on the PLE gate value (it's
  computed mid-forward; acting on it serializes the lookup and kills the
  prefetch; unconditional retrieval is cheap). Wired-GPU cost of the table:
  zero — hash+gather+dequant host-side, ship 2560 floats/token to the graph.
- **Perf expectation (scaling guess, NOT a measurement)**: 6B active at
  Q4-class on this machine, scaled from the 35B-A3B's measured 104-107 tok/s
  at ~3B active (~1.7 GB/token → ~3.4 GB/token), lands around ~50 tok/s
  decode. Memory: ~70-75 GB wired trunk + ~29 GB file-backed table on 128 GB.
  One-process-at-a-time becomes absolute.
- **State-allocation note**: llama.cpp's PR adds `ple_conv_state()` into
  `n_embd_r()` for EVERY recurrent layer — 36 layers × 9 × 10240 floats of
  dead state on this checkpoint (only layer 1 has PLE). We allocate PLE state
  per-layer, only where a PLE layer exists.
- **Card sampling details beyond the doc's main bullet**: min_p 0.0 and
  repetition_penalty 1.0 in both modes; recommended output budgets 262144
  reasoning / 131072 final. (Mode-keyed temp/top_p/top_k match 3.6/3.8
  exactly; the one genuine novelty is non-thinking presence_penalty 1.5.)
- **Design lineage citations** (for whoever wonders why the math looks the way
  it does): PLE is DeepSeek Engram (arXiv 2601.07372, code
  github.com/deepseek-ai/Engram) with deviations noted in the spec section
  (raw ids, one layer, dot-product gate); allocation background: SCONE (arXiv
  2502.01637), LongCat "Scaling Embeddings Outperforms Scaling Experts"
  (arXiv 2601.21204), Meta "Memory Layers at Scale" (arXiv 2412.09764). QSA's
  nearest published relative is DeepSeek DSA (three implementations in our
  pinned llama.cpp clone; glm-dsa.cpp is the cleanest read). Hyper-connections
  trace to arXiv 2512.24880 (ggml `ggml_dsv4_hc_*`).

## MTP head (deferred to a drafting arc; facts frozen here so they aren't lost)

34 `mtp.*` tensors ship in the BF16 repo; the PR's converter SKIPS them (no
GGUF has them today — a drafting arc must convert its own sidecar or extend
the converter). Structure from tensor names + config: `pre_fc_norm_embedding`
+ `fc_embedding` over the token embedding, `pre_fc_norm_hidden` + `fc_hidden`
over the trunk's final hidden (`mtp_use_hidden_state_from_layer: null` ⇒
post-`hyper_connection_mixer`, consistent with the 3.8 head's choice) — TWO
separate projections, NOT 3.8's concat `eh_proj`; presumed summed, but the
composition is UNCONFIRMED (transformers ships no MTP class — confirm against
vLLM/SGLang or the tech report before implementing). One full trunk-flavour
layer: QSA attention with its own indexer + full MoE (experts, router, shexp)
+ both hyper-connection blocks + its own `hyper_connection_mixer`.
`mtp_use_dedicated_embeddings: false` ⇒ reuses target embd/lm_head, like 3.8.
Config: `mtp.hybrid: true`, 1 layer, rope_theta 1e7. Card: "trained with
multi-steps".

## Open questions (blocked, with what unblocks them)

- MTP head forward semantics (fc_embedding/fc_hidden composition) — vLLM/SGLang
  source or the tech report (github.com/QwenLM/Qwen3.8-Flash-Next tech_report.pdf).
- ~~Exact Q4-class quant mix~~ ANSWERED 2026-08-29: the whole ladder is
  surveyed above; UD-Q4_K_XL is the first target (D12).
- ~~Whether PR #27742 merges as-is~~ ANSWERED 2026-08-29: merged 08-27 as
  `6c84c7d5d` (D4 update). The numeric check Qwen promised was never posted, so
  `conversion/qwen4exp.py`'s stability for a self-converted blessed file rests
  on the PR body's own ppl-vs-reference.
- Where the qwen4exp oracle clone lives. `reference/llama.cpp` (e9fa0781) stays
  frozen for the 3.6/3.8 cycles, so the qwen4exp oracle is a SECOND clone at
  `6fe749801`. Candidate path `reference/llama.cpp-qwen4exp`; open whether
  `scripts/build-llamacpp.sh` grows a target argument or gets a sibling.

## Progress log

- **2026-08-26**: Arc opened. Spec confirmed from primary sources (config,
  transformers modular file, GGUF metadata — see TODO.md entry for the research
  trail). Doc created; P0 begun: split-GGUF loader, reuse-seams map, PR-file
  vendoring dispatched.
- **2026-08-26**: reference/qwen4exp/ vendored (PR #27742 @ pinned sha, see
  PROVENANCE.md); converter audited — conversion-baked deltas and four PR
  quirks recorded above. Doc corrected: sampling key is `general.sampling.temp`,
  no presence-penalty key exists in converted GGUFs, PR is open-not-draft.
- **2026-08-26**: Reuse-seams audit complete — blocks compose unchanged,
  decisions D8-D11 taken (loader owns GGUF parsing; ZGate enum; sum_floor
  field; QSA decode gather), trap #12 added (sdpa vector kernel silently
  ignores masks). Next wave dispatched: config/registry scaffold, HF fixture
  generation for the new components.
- **2026-08-26**: P0 units landed (unstaged, review in flight): split-GGUF
  loading in gguf.rs (any shard opens the set; single-file path unchanged;
  CheckpointId folds all shards; first reviewer clean); qwen4exp config
  parsing (Arch::Qwen4Exp, Qwen4ExpConfig/PleConfig sub-structs, u64-array
  Meta accessors, per-arch z_gate()/moe_sum_floor(); Arch::model() now
  Option — no registry entry until a blessed file exists; eog list now
  guarantees both stop ids whatever single eos the GGUF advertises). P1
  golden fixtures generated from transformers main @ 598d8ba8 into
  tests/fixtures/qwen4exp/ (5 files, ~556 KB, deterministic; generator +
  venv recipe in scripts/qwen4exp-fixtures/). Fixture findings: QSA
  whole-blocks-plus-tail confirmed (PR #27742 diverges); PLE gate clamp
  retraction; tail-0 case can mask the query's own token. Real shard-0
  metadata confirms tokenizer eos 248046 / ple.eos 248044 as separate keys.
- **2026-08-26 (P0 close)**: Split-GGUF loading committed (e99ffee) after
  dual review — Claude clean, Codex 6 hardening findings all fixed with tests
  (checked offset rebase, loud whole-file accessors via SingleFilePath,
  actual-table-driven allocation, filename shard-number bounds, partial-split-
  metadata errors, both-present-must-match cross-shard keys). qwen4exp config
  parsing committed (2914d7c) after dual review — Claude clean, Codex 6
  findings: 4 fixed with tests (u32_checked token ids, PLE geometry incl.
  running-sum offsets and multipliers.len()==ngram_size confirmed against the
  C++ reference hash loop, Bool-rejecting/I8-I16-accepting u64 accessor,
  load-time arch refusal before tensor work), 1 accepted-as-intended
  (both-stops eog guarantee, now commented), 1 rejected (cadence validation
  against the file's declaration is by design). Full lib suite 872/872 at
  close. **P0 done; arc paused before P1.**
- **2026-08-29**: Arc resumed. Surveyed every published qwen4exp GGUF from its
  headers (section "Quant landscape" above) and took D12: UD-Q4_K_XL is the
  first target file, download started into the HF cache. Upstream landed too —
  PR #27742 merged 08-27 as `6c84c7d5d` with a graph-splits follow-up 08-28
  (`6fe749801`), so D4's merge gate is met and the vendored material is being
  re-vendored at `6fe749801` as a second, buildable oracle clone while
  `reference/llama.cpp` stays frozen at e9fa0781 for 3.6/3.8. Qwen's promised
  independent numeric check never appeared; the PR body's own numbers stand in.
  Next: finish re-vendor + download, then P1.
