# Qwen3.8-Flash-Next UD-Q4_K_XL — tensor table (read from the file, 2026-08-29)

Source: `unsloth/Qwen3.8-Flash-Next-GGUF`, snapshot `c8b5954a`, folder `UD-Q4_K_XL`,
four shards `Qwen3.8-Flash-Next-UD-Q4_K_XL-0000N-of-00004.gguf`. Headers only — no
tensor data was read. Dims are printed in GGUF order (`ne[0]` first = the fastest /
input dimension), so a projection reads `[in, out]`.

Sizes are computed from the type block sizes (F32 4 B/elem, BF16 2, Q8_0 34 B/32,
Q4_K 144 B/256, Q5_K 176/256, Q5_1 24/32, IQ4_NL 18/32) and total **111.324 GB**,
which matches the published 111.33 GB — an independent confirmation of the per-plane
types below (notably Q5_1: at IQ4_NL the file would be ~7.8 GB smaller).

## 1. Counts and shard layout

1224 tensors total (`split.tensors.count`), 48 blocks, layers 0-47.

| shard | tensors | bytes | blk layers | non-`blk.` tensors |
| --- | --- | --- | --- | --- |
| 1 | 0 | 0 | — | — (67 KV keys only; the full metadata block) |
| 2 | 297 | 49.86 GB | 0-11 | `token_embd.weight`, `output.weight`, `per_layer_token_embd.weight`, `output_hc_{norm,down,up}.weight` |
| 3 | 752 | 49.38 GB | 11-41 | — |
| 4 | 175 | 12.09 GB | 41-47 | — |

Shard 0 (file `-00001-`) carrying zero tensors matches the documented gguf-split
layout. Layers straddle shard boundaries (11 spans shards 2/3, 41 spans 3/4), so a
loader must not assume a layer is contiguous within one file.

**All three of `per_layer_token_embd.weight`, `token_embd.weight` and
`output.weight` live in shard 2** (the `-00002-` file), together with the whole
model tail (`output_hc_*`).

Layer kinds, as predicted: GDN layers are the 36 with `(N+1) % 4 != 0`
(0-2, 4-6, …, 44-46); QSA attention layers are the 12 with `(N+1) % 4 == 0`
(3, 7, 11, …, 47); the PLE layer is `blk.1` only (`ple.layers` is already 0-based).

## 2. Distinct tensor planes

`blk.N` collapsed. "layers" is the set of block indices carrying the plane; the
plane count is planes-per-model, and GB is the summed size of all of them.

### Embeddings / head

| name | dims | dtype | n | GB |
| --- | --- | --- | --- | --- |
| `token_embd.weight` | 2560x248320 | Q8_0 | 1 | 0.675 |
| `output.weight` | 2560x248320 | Q8_0 | 1 | 0.675 |

Untied (both present), both Q8_0, vocab 248320 as documented.

### Hyper-connections (every layer + the tail)

| name | dims | dtype | n | layers | GB |
| --- | --- | --- | --- | --- | --- |
| `blk.N.hc_attn_norm.weight` | 10240 | F32 | 48 | 0-47 | 0.002 |
| `blk.N.hc_attn_down.weight` | 10240x320 | Q8_0 | 48 | 0-47 | 0.167 |
| `blk.N.hc_attn_up.weight` | 320x10240 | Q8_0 | 48 | 0-47 | 0.167 |
| `blk.N.hc_attn_inject.weight` | 10240x4 | F32 | 48 | 0-47 | 0.008 |
| `blk.N.hc_ffn_norm.weight` | 10240 | F32 | 48 | 0-47 | 0.002 |
| `blk.N.hc_ffn_down.weight` | 10240x320 | Q8_0 | 48 | 0-47 | 0.167 |
| `blk.N.hc_ffn_up.weight` | 320x10240 | Q8_0 | 48 | 0-47 | 0.167 |
| `blk.N.hc_ffn_inject.weight` | 10240x4 | F32 | 48 | 0-47 | 0.008 |
| `output_hc_norm.weight` | 10240 | F32 | 1 | — | 0.000 |
| `output_hc_down.weight` | 10240x320 | Q8_0 | 1 | — | 0.003 |
| `output_hc_up.weight` | 320x10240 | Q8_0 | 1 | — | 0.003 |

387 tensors, exactly the documented set. The tail has norm/down/up and **no**
`output_hc_inject` — consistent with "same read path, no write". There is no
`attn_norm`, no `post_attention_norm`, no `ffn_norm` and no `output_norm` anywhere
in the file, confirming the hyper-connection carrier replaces all of them.

### Attention (QSA layers 3,7,…,47)

| name | dims | dtype | n | GB |
| --- | --- | --- | --- | --- |
| `blk.N.attn_q.weight` | 2560x12288 | Q8_0 | 12 | 0.401 |
| `blk.N.attn_k.weight` | 2560x512 | Q8_0 | 12 | 0.017 |
| `blk.N.attn_v.weight` | 2560x512 | Q8_0 | 12 | 0.017 |
| `blk.N.attn_output.weight` | 6144x2560 | Q8_0 | 12 | 0.201 |
| `blk.N.attn_q_norm.weight` | 256 | F32 | 12 | 0.000 |
| `blk.N.attn_k_norm.weight` | 256 | F32 | 12 | 0.000 |
| `blk.N.indexer.q_proj.weight` | 2560x512 | BF16 | 12 | 0.031 |
| `blk.N.indexer.k_proj.weight` | 2560x128 | BF16 | 12 | 0.008 |
| `blk.N.indexer.q_norm.weight` | 128 | F32 | 12 | 0.000 |
| `blk.N.indexer.k_norm.weight` | 128 | F32 | 12 | 0.000 |

`attn_q` is the double-width interleaved `[q,gate]` plane (24 × 256 × 2 = 12288);
k/v are 2 heads × 256 = 512; `attn_output` reads 24 × 256 = 6144. The indexer is
split into q/k projections and kept unquantized (BF16), as the converter audit said.

### Gated DeltaNet (36 layers, `(N+1)%4 != 0`)

| name | dims | dtype | n | GB |
| --- | --- | --- | --- | --- |
| `blk.N.attn_qkv.weight` | 2560x10240 | Q8_0 | 36 | 1.003 |
| `blk.N.attn_gate.weight` | 2560x6144 | Q8_0 | 36 | 0.602 |
| `blk.N.ssm_conv1d.weight` | 4x10240 | F32 | 36 | 0.006 |
| `blk.N.ssm_alpha.weight` | 2560x48 | F32 | 36 | 0.018 |
| `blk.N.ssm_beta.weight` | 2560x48 | F32 | 36 | 0.018 |
| `blk.N.ssm_a` | 48 | F32 | 36 | 0.000 |
| `blk.N.ssm_dt.bias` | 48 | F32 | 36 | 0.000 |
| `blk.N.ssm_norm.weight` | 128 | F32 | 36 | 0.000 |
| `blk.N.ssm_out.weight` | 6144x2560 | Q8_0 | 36 | 0.602 |

Geometry is byte-identical to our 27B block: fused qkv width 10240
(16·128 q + 16·128 k + 48·128 v), z-gate 6144, conv kernel 4 over the full fused
width, 48 V-heads driving the `[48]`-shaped `ssm_a`/`ssm_dt.bias` and the `x48`
alpha/beta, `ssm_norm` per head dim 128, inner 6144. No `ssm_in` (the projections
ship under `attn_qkv`/`attn_gate`), and `ssm_dt.bias` is the only bias in the file.

### MoE (every layer)

| name | dims | dtype | n | layers | GB |
| --- | --- | --- | --- | --- | --- |
| `blk.N.ffn_gate_inp.weight` | 2560x512 | F32 | 48 | 0-47 | 0.252 |
| `blk.N.ffn_gate_exps.weight` | 2560x640x512 | Q4_K (L0-1,3-47) + Q5_K (L2) | 48 | 0-47 | 22.754 |
| `blk.N.ffn_up_exps.weight` | 2560x640x512 | Q4_K (L0-1,3-47) + Q5_K (L2) | 48 | 0-47 | 22.754 |
| `blk.N.ffn_down_exps.weight` | 640x2560x512 | **Q5_1** (L0-1,3,5-29,31-45) + Q8_0 (L2,4,30,46-47) | 48 | 0-47 | 31.510 |
| `blk.N.ffn_gate_inp_shexp.weight` | 2560 | F32 | 48 | 0-47 | 0.000 |
| `blk.N.ffn_gate_shexp.weight` | 2560x640 | Q8_0 | 48 | 0-47 | 0.084 |
| `blk.N.ffn_up_shexp.weight` | 2560x640 | Q8_0 | 48 | 0-47 | 0.084 |
| `blk.N.ffn_down_shexp.weight` | 640x2560 | Q8_0 | 48 | 0-47 | 0.084 |

512 experts, 640-wide, all three expert planes 3-D `[.., .., 512]`. The shared
expert is a single 640-wide FFN per layer plus the `[2560]` sigmoid gate vector.
Router `ffn_gate_inp` is F32. No dense FFN and no `feed_forward_length`-style
plane anywhere: every layer is MoE.

### PLE (layer 1 only, plus the global table)

| name | dims | dtype | n | GB |
| --- | --- | --- | --- | --- |
| `per_layer_token_embd.weight` | 160x320001536 | IQ4_NL | 1 | 28.800 |
| `blk.1.ple_key.weight` | 2560x10240 | Q8_0 | 1 | 0.028 |
| `blk.1.ple_value.weight` | 2560x2560 | Q8_0 | 1 | 0.007 |
| `blk.1.ple_norm_query.weight` | 10240 | F32 | 1 | 0.000 |
| `blk.1.ple_norm_key.weight` | 10240 | F32 | 1 | 0.000 |
| `blk.1.ple_norm_conv.weight` | 10240 | F32 | 1 | 0.000 |
| `blk.1.ple_conv1d.weight` | 4x10240 | F32 | 1 | 0.000 |

The flat table is `[160, 320001536]` = 51,200,245,760 elements at IQ4_NL — exactly
the 28.80 GB the quant-landscape table predicts, and 25.9% of the whole file. The
three PLE norms and the dilated conv are all on the 10240-wide hyper-connection
stream. `ple_key` is 2560 → 10240 (viewed `[4,2560]`), `ple_value` 2560 → 2560.

## 3. Deviations from the port doc

Checked against "Confirmed spec" and "Conversion-baked deltas" in
`docs/qwen4exp-port.md`.

1. **`ffn_down_exps` is Q5_1 on 43 of 48 layers** (raw ggml type id 7; Q8_0 on
   layers 2, 4, 30, 46, 47). The doc predicted "experts Q4_K" and floated IQ4_NL as
   the alternative for `down_exps`. Q5_1 is a legacy non-K quant that appears
   nowhere else in our three shipped checkpoints, so **a Q5_1 dequant/matmul path is
   a new kernel requirement** for this file — the single largest plane in the trunk
   (31.5 GB, 28% of the file, more than gate+up combined).
2. **`ffn_gate_exps` / `ffn_up_exps` are Q5_K on layer 2**, Q4_K on the other 47.
   The doc predicted a uniform Q4_K for these. A per-layer type lookup is required;
   an "experts are Q4_K" assumption breaks on exactly one layer, silently, only in
   the dequant path.
3. **`ple_conv1d.weight` is F32, not F16.** The conversion audit predicted that this
   tensor is off the quantize skip list, cannot take a 32-element block quant
   (ne0=4) and therefore "lands F16 via a new fallback branch". The shipped Unsloth
   file has it F32. The doc's own measured "precision policy" bullet already says
   F32, so the two halves of the doc disagree and the file settles it: F32. The
   F16-fallback prediction still stands for a self-conversion, so the
   `--tensor-type` pin advice is unaffected.
4. **`ffn_gate_inp_shexp.weight` is 1-D `[2560]`**, where the spec section writes
   the shared-expert gate weight shape as `[1, 2560]`. Same element count; the GGUF
   simply drops the degenerate dimension. Cosmetic, but a shape assert written from
   the doc would fail.
5. **No `output_hc_inject.weight`.** The doc lists the tail as
   `output_hc_{norm,down,up}` and describes it as read-only, so this is a
   confirmation rather than a surprise — noted because the per-layer `hc_*` set has
   four members and the tail has three.

Everything else matched, including several things worth calling confirmed: no
`attn_norm`/`post_attention_norm`/`output_norm` tensors at all; `attn_q` at the
double 12288 width; the fused 10240 `attn_qkv` with `attn_gate` at 6144;
`[48]`-shaped `ssm_a`/`ssm_dt.bias` (V-head count) and `[128]` `ssm_norm`;
indexer q/k split and BF16; `token_embd`, `output`, every `attn_output`, `ple_key`,
`ple_value` and both hc up/down planes at Q8_0; all norms and routers F32; the PLE
table at IQ4_NL; PLE present on `blk.1` only; 1224 tensors; shard 0 empty.

## 4. Byte totals by subsystem

| subsystem | tensors | GB | share |
| --- | --- | --- | --- |
| MoE (router + experts + shared expert) | 384 | 77.521 | 69.64% |
| PLE (table + layer-1 planes) | 7 | 28.835 | 25.90% |
| attention incl. indexer | 156 | 1.677 | 1.51% |
| embeddings / head | 2 | 1.351 | 1.21% |
| GDN | 288 | 1.245 | 1.12% |
| hyper-connections | 387 | 0.695 | 0.62% |
| **total** | **1224** | **111.324** | 100% |

103.68 GiB. Trunk (everything but the PLE table) is 82.52 GB, matching the
quant-landscape table's 82.53.

Two decode-relevant consequences. First, 95.5% of the bytes sit in MoE + PLE, both
of which are *sparsely* read per token (10 of 512 experts; 16 table rows), so the
resident-weight figure and the per-token bytes-moved figure diverge far more here
than on any checkpoint we ship. Second, the dense-per-token planes — GDN 1.245, attention
1.677, hyper-connections 0.695, shared expert 0.252, router 0.252 and `output`
0.675 — total 4.80 GB (5.47 with `token_embd`), and the subsystem hardest to get
right (hyper-connections, 387 tensors across 11 planes) is the smallest of them.
