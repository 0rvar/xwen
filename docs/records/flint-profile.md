# Flint profile: the first figures and where the time goes

The record of the session that took Flint's first pinned figures and profiled its decode
and prefill, 2026-10-26. No code changed. The binary is a release build of a detached
worktree at [91ef40a](https://github.com/0rvar/xwen/commit/91ef40a), copied to
`/tmp/flint-prof/xwen-clean`, sha256
`935771c259ec0250df1970ccccf1c5597515a4b1dd2f85b634a8d10e88fafab5`. The figures are in
[perf-state.md](../perf-state.md) "Flint, a design target with no figure yet", the
previous arc is [flint-kernels.md](flint-kernels.md), and the decisions are in
[decisions/flint.md](../decisions/flint.md).

## Conditions

`pmset -g` read ` lowpowermode         2` at the start and the end of every sweep. The
owner switched the machine out of power saver just before the session, and nothing beyond
that line is claimed. The machine was shared with other agents: load average 8.2 at the
start of the bench, 1.3 at the end, and 1.3-9 during the profile runs. One model process
ran at a time, under the GPU lock `/tmp/xwen-gpu.lock`.

The bench ran three interleaved rounds, alternating which model went first, with 15 s idle
between runs and a Flint short anchor at the start and the end. Every run was a fresh
process with `XWEN_BENCH=1`:

```
XWEN_BENCH=1 /tmp/flint-prof/xwen-clean generate --model <flint|PATH> --no-draft --raw \
  --stats --seed 1 -n <256|128|64> --prompt "$(cat <prompt>)"
```

The prompts were the lighthouse-keeper sentence (16 Flint tokens, 17 Qwen),
`tests/fixtures/bench-prompts/prefill-4k.txt` (3654 Flint tokens, 3851 Qwen) and the first
140,000 bytes of `cat docs/records/*.md` at 91ef40a (38,024 Flint tokens). The driver was
`/tmp/flint-prof/scripts/ab.ts`, and the raw outputs sit in `/tmp/flint-prof/bench/`.

## The figures

| Figure | Flint, range over reps | 35b-uncensored, same rounds |
| --- | --- | --- |
| decode, short prompt, 256 tokens | 122.7-134.1 tok/s (131.4, 131.8, 122.7; anchor 134.1) | 87.7-95.6 |
| decode at 3654 / 3851 tokens, 128 tokens | 121.0-127.0 (121.0, 121.9, 127.0) | 89.3-92.2 |
| decode at 38,024 tokens, 64 tokens | 77.6-94.4 (77.6, 91.2, 94.4) | not run |
| prefill @3654 / @3851, chunk 2048 | 2817-2891 (2878, 2817, 2891) | 2967-3045 |
| prefill @38,024 | 1067-1415 (1067, 1336, 1415) | not run |

The anchor read 131.4 tok/s at the start and 134.1 at the end, a 2.1% drift, under the 3%
flag. The first 38k rep is the low outlier on both prefill and decode; it ran while the
load average was still coming down from ~8, and the other two agree within 6%.

The official `ggml-org/Qwen3.6-35B-A3B-GGUF` was not in the HF cache and was not
downloaded. The comparator is the cached `35b-uncensored` (HauhauCS Qwen3.5-35B-A3B
Q4_K_M), the same `qwen35moe` graph with a different quant mix: attention and the shared
expert are Q4_K/Q5_K/Q6_K rather than Q8_0, so its shared expert runs the classic chain,
and `down_exps` is Q6_K. It is a class reference and its rates are not the 35B-A3B's. In
the same rounds Flint decodes 1.32-1.40x faster than it and prefills at 3.6k at
0.94-0.96x.

Decode no longer collapses with context. It is -5% at 3654 tokens against the short
prompt and about -30% at 38k, which is the KV read: 14 full layers at 2 KiB per position
come to ~1.09 GB per token at 38k, beside 2.26 GB of weights. These rows supersede the
`lowpowermode 1` observations in [flint-kernels.md](flint-kernels.md) (59.4 short, 50.7
at 3654, prefill 1149) and the session's own earlier low-power runs (48-63 short, 51-56 at
3654, 945-1318 prefill at 3654).

## Method

There is no dispatch counter in the repo, so the profile patched a /tmp copy of
candle-metal-kernels at 21cca0b into a scratch worktree with three instruments, all
removed afterwards:

1. A dispatch counter, which maps each pipeline to its `MTLFunction.name` at creation and
   records per dispatch the threadgroup count and threads per threadgroup, plus commits,
   waits and blits. It dumps after prefill and at the end of generation. On Flint's
   decode this doubles as the threadgroup-count audit the ledger carries.
2. Stage tags from `run_stack`'s `stage!` macro.
3. Per-kernel GPU time: `CANDLE_METAL_COMPUTE_PER_BUFFER=1` puts each dispatch in its own
   command buffer, which is waited on, and `GPUEndTime - GPUStartTime` is summed.

The third instrument serializes everything. At prefill its in-stage sum (1.31 s at 3654)
matches the unprofiled wall (1.28 s), so prefill shares are prices. At decode it inflates
2.2x (16.9 ms against a 7.4 ms token), because every dispatch starts on an idle GPU, so
decode rows rank and do not price.

Decode is priced by a removal A/B instead. `XWEN_MOE_GLUE_CLASSIC=1` adds exactly 517
small dependent dispatches per token (1690 against 1173), and three interleaved rounds
each read:

- short: 132.2 / 135.1 / 135.0 against 107.4 / 111.3 / 116.4 tok/s, 1.57 ms per token at
  the medians, **3.0 µs per dispatch**;
- at 3654: 116.2 / 127.8 / 126.7 against 109.2 / 110.2 / 111.1, 1.18 ms, **2.3 µs per
  dispatch**.

Every decode estimate below is a dispatch count times that measured 2.3-3.0 µs. The counts
and the rate are measured; the gains are estimates built from them.

## Decode anatomy

A token is **1161 dispatches on a short prompt and 1210 at 3654 tokens** (the split-K
combine turns on in every layer past 64 keys), 23.7-24.7 per layer over 49 layers. There is
one host wait per token, the sampled-id readback, and about 24 command-buffer commits from
candle's 50-per-buffer batching, a direction already refuted. The 35B-class file is 1026
per token, 25.6 per layer over 40 layers, so Flint is not heavier per layer; it has more
layers, and all of them run attention.

Per layer:

| Dispatches | Kernel | Kind |
| --- | --- | --- |
| 4 | `kernel_mul_mv_q8_0_f32_attn` for q, k, v and o | weights, 20 MB |
| 2 | `rmsnorm_f32` q_norm (32 threadgroups), k_norm (4) | glue |
| 2 or 1 | rope q (f32 store) and rope k (f16 store), or the k cast on the 14 NoPE layers | glue |
| 1 | `cast_f32_f16` of v, 1 threadgroup | glue |
| 2 | `copy2d_f16` K and V cache slot writes, 1 threadgroup each | glue |
| 1 + 1 | `kernel_attn_decode_g8` and `attn_decode_combine` | attention |
| 2 | `badd_f32` residual adds, 1 threadgroup each | glue |
| 2 | `rmsnorm_f32` attn_norm and ffn_norm, 1 threadgroup each | glue |
| 8 (MoE) | router f32 gemv (128 tg), `kernel_moe_router` (1 tg), mv_id gate / up / down (1536 tg), `kernel_moe_silu_mul`, the shexp gate_up pair, the shexp epilogue | at Qwen parity |

That is 11-12 glue dispatches per layer, ~560 per token, ~1.5 ms of a 7.4 ms token at the
measured rate.

The tensor table puts **2.264 GB of active weights per token**: attn_q 437 MB, attn_output
437, lm_head 327, routed experts 499, shexp 314, router 99, attn_k plus attn_v 109, dense
FFN 43. The lm_head gemv ran at 438-517 GB/s, so the bytes-only floor is ~4.4-5.2 ms,
**~190-230 tok/s**. The measured 7.4 ms sits ~3 ms above it, which is 1161 dispatches at
~2.6 µs, the rate the removal A/B gave. KV reads are ~0.15 GB per token at 3.7k and ~1.09
GB at 38k, and `attn_decode` is the second-largest decode row at 38k (74 µs serialized,
against 18 µs short).

The serialized ranking on the short prompt, which inflates small kernels: q8 attention
gemv 25%, mv_id 20%, shexp gate_up 7%, `kernel_moe_router` 5%, `attn_decode` 5%, the four
rmsnorms 10%, lm_head 4%, epilogue 4%, router gemv 4%, and badd, copy2d, cast, rope and
silu_mul together 13%.

## Prefill anatomy

Serialized prices at 3654 tokens, two chunks of 2048, 1.31 s in-stage:

| Kernel | ms | Share | Note |
| --- | --- | --- | --- |
| `kernel_mul_mm_id_q4_K_f32_t64` expert gemms | 265 | 20.2% | known dequant-bound |
| `kernel_moe_combine_rescale` | 247 | 18.8% | 2.6 ms per call, 3,741,696 threadgroups of 4 threads |
| `copy_f32_strided` in Flint attention | 227 | 17.4% | the q, k, v and output `transpose(0,1).contiguous()` |
| flash with sinks (`q0_k0` and `q1_k1`) | 218 | 16.6% | |
| `kernel_mul_mm_f16_f32_t` q/k/v/o projections | 153 | 11.7% | |
| shexp q8 gemm, classic chain | 54 | 4.1% | |
| router gemm, map0, silu_mul_l2, router | 53 | 4.0% | |
| q/k rmsnorm, rope, casts, residuals, norms | ~60 | ~5% | |

At 38,024 tokens, 25.5 s in-stage, flash is 56% (`q1_k1` on the full NoPE layers alone
49%), expert gemm 10.6%, combine_rescale 10.0%, the transposes 9.2% and the projections
6.0%.

Every prefill chunk takes the f16 overflow guard: `kernel_moe_silu_mul_l2` and
`kernel_moe_combine_rescale` run, the fused epilogue is declined (`src/moe.rs`, the
`needs_rescale` branch) and the shexp runs its classic chain. The guard costs under 1%
(`silu_mul_l2` is 12 ms). The cost is the combine kernel's geometry: to mirror candle's
`sum(1)` bitwise it launches one threadgroup per output element with
`next_pow2(top_k/2)` = 4 threads (`src/ops/combine.metal`). It moves ~117 MB, ~0.25 ms
of traffic, and takes 2.6 ms. The 35B-class file shows the same thing, 220 ms and 16% of
its 4k prefill, level with its expert gemm. That branch is already off the strict tier,
so a re-geometried combine can be graded on tolerance.

## What other engines do

From the research pass of the same day (sources in the bullets).

- Kapitsa ([nace-ai/kapitsa#286](https://github.com/nace-ai/kapitsa/pull/286)) runs a
  decode layer in about 6 dispatches: the QKV gemv with the input norm folded in; one
  attention dispatch for q/k norms, rope, cache write, split-K and a last-threadgroup
  merge with sinks; the o-proj with the residual add; the post-norm, router and shared
  gate/up together; routed gate+up+SiLU; one down over all seven slots into the residual.
  xwen runs 23.7-24.7 per layer, which the counts above confirm.
- Recursive's own `rec` ([recursive-oss/rec](https://github.com/recursive-oss/rec)) runs
  the whole token in one persistent Metal kernel with grid barriers, and samples up to 8
  tokens per launch on the GPU.
- On an M5 Pro in High Power, Kapitsa decodes 145.7 tok/s and rec 144.9 (greedy 151.9),
  llama.cpp 92.4
  ([Kapitsa M5 Pro results](https://github.com/nace-ai/kapitsa/blob/main/bench/results/2026-10-22-flint-m5pro)).
  Those engines read Recursive's 4-bit exports, so their files move less than the 2.26 GB
  this one does.

At ~2.6 µs per dispatch, Kapitsa's layout would take ~880 dispatches out of a token,
~2.3-2.6 ms, putting Flint at ~4.8-5.1 ms, about 195-210 tok/s on this file against its
~190-230 bytes floor. That is an estimate. The research pass guessed "~1.5 GB of weights
per token" and a 350-400 tok/s ceiling, which holds for a 4-bit file and not for the
ggml-org Q4_K_M, which keeps attention, the shexp and lm_head in Q8_0 (1.62 GB of the
2.26). Recursive's Q4_1 file is the lever if that ceiling ever matters, and it is already
an item. The research also said the sigmoid router runs a 6-10-dispatch candle chain; it
has not since 3063e0e. What remains there is that the router kernel is one threadgroup.

## Ranked fruit

Decode gains are dispatch counts times the measured 2.3-3.0 µs on a 7.4 ms token (7.9 at
3.7k). Prefill gains are shares of the prefill wall. The first five are ledger items in
[TODO.md](../../TODO.md); 1 and 2 are one Front item.

1. **Re-geometry `kernel_moe_combine_rescale`**, many outputs per threadgroup (say 64
   outputs by 4 lanes, quad reduce), tolerance-graded. Prefill -17% wall at 3.7k (~+21%
   tok/s), ~-9% at 38k, and about 15% on every MoE checkpoint. Small:
   `src/ops/combine.metal` and `run_combine` in `src/ops/dispatch.rs`.
2. **Qwen's `ops::permute_01` and `ops::permute_01_f16`** for q, k, v and the output in
   `FlintAttn::forward` at seq > 1, in place of the strided copies (the v one also folds
   the cast). Prefill -15% wall at 3.7k, ~-8% at 38k. Small, reused as-is; they are
   bitwise per the attention glue tests. 392 calls at ~580 µs.
3. **One decode kernel for q_norm, k_norm, rope (or the NoPE k cast), the v cast and the
   K/V cache-slot write**, 7 dispatches per layer to 1 (6 on NoPE): Kapitsa's
   `flint_attn_d2` without the attention. Decode -0.7 to -0.9 ms, +10-13%; without the
   cache write -0.45 to -0.6 ms, +6-8%. Medium: extend `rope.metal`'s f16-store rope with
   a per-head RMS pre-norm, and take the slot offset of the capacity or ring view.
4. **A fused QKV q8 gemv with the input RMSNorm folded in**, generalizing
   `kernel_moe_shexp_gate_up_ungated`, the existing multi-plane q8 gemv. Decode -0.35 to
   -0.45 ms, +5-6%; 147 dispatches per token.
5. **The residual add folded into the o-proj and MoE epilogue stores, and the ffn_norm
   into the router gemv's input**, or at minimum a fused add-plus-RMSNorm kernel. Decode
   -0.2 to -0.45 ms, +3-6%; 98 `badd` and 98 outer norms per token at 1 threadgroup each.
   The plain add-plus-norm fusion applies to the Qwen graphs as well.

Together 1 and 2 take 3.7k prefill from ~1.28 s to ~0.85 s, about 1.5x. Items 3-5, with the
combine fold, the router and the routed gate-up below, remove ~650 dispatches per token, ~1.6-2 ms, which would take decode from ~132 to ~170-190
tok/s.

## Not taken now

Candidates from the same ranking that did not enter the ledger, each with what reopens it.

- **Fold `attn_decode_combine` into `attn_decode`** with a last-threadgroup merge. Kapitsa
  measured 9 against 19 µs on a sliding layer and 15 against 29 on a full one (M4 Pro).
  About +2-3% past 64 keys, 49 dispatches per token. Reopen when item 3 lands, since it
  rewrites the same chain, or when decode at 3.7k+ is the figure being chased.
- **A faster `kernel_moe_router`**: 256 threads in one threadgroup, top-6 on packed
  (logit, index) keys by simd max, sigmoid and renorm on the winners only. About +2-3%,
  47 dispatches per token, and it would help the 35B and Flash-Next too. It must stay
  bitwise with `route_sigmoid_from_logits` and its tie tests. Reopen after 3-5, when it
  is a larger share of what is left.
- **Routed gate and up (and SiLU) in one dispatch**, grid z per plane so per-thread work
  stays the same, which is not the refuted dual-accumulator shape. About +2-3%, also on
  the 35B. Reopen with the router item.
- **Tensor-op flash for prefill**: a causal tile skip, the window and the sink seed added
  to `flash_t.metal`, which already does head_dim 128 at 45 against the steel copy's 13.1
  TFLOP/s. ~-11% at 3.7k and ~-35-40% at 38k, flash being 16.6% and 56% of those walls.
  Medium-large. Reopen when long-context prefill is the figure being chased, or when items
  1 and 2 have landed and flash is the top prefill row at 3.7k.
- **A prefill chunk sweep** (`XWEN_PREFILL_CHUNK` 1024 / 2048 / 4096). 2048 was taken
  without a measurement (`src/config.rs`). Unpriced. Reopen in the next Flint prefill
  session; it is an hour.
- **q8 gemv tuning.** It is 25% of the serialized decode, mostly its own intercept. Reopen
  with a microbench after items 3-5 change the chain around it.
- **The simdgroup-MMA decode attention**, ~0.8 ms at 32k per
  [flint-kernels.md](flint-kernels.md), behind everything above below 32k. Reopen when
  long-context decode is the figure being chased.
- **The dense layers' unfused SwiGLU**, 4 dispatches per token. Reopen never on its own;
  fold it in if a dense-layer kernel is written for another reason.
- **QKV fusion at prefill.** The projections are compute-bound gemms there. Reopen if a
  prefill profile shows them launch-bound.
- **Command-buffer batching** is refuted (decisions.md) and stays out.

## Uncertain

- The per-dispatch rate comes from one A/B over MoE glue dispatches. Attention glue
  dispatches may cost somewhat differently.
- Absolute rates on a shared machine: the 38k first rep is the outlier.
- The comparator is not the shipped 35B-A3B.
