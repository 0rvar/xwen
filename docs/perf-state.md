# Perf state

The current figures, one place. Every number here is the latest measurement of that
thing; the story of how it got there lives in [the log](log.md) and its records, and the
reasoning behind the choice that produced it lives in [decisions](decisions.md).

## How to read this table

- **Within-session ratios are the claims.** A drafted figure is a gain over the plain arm
  of its OWN sweep, and an A/B figure is a gain over the classic arm measured in the same
  rounds. Differencing two numbers from different sessions is not a result.
- **Levels drift between sessions.** The 27B's plain level moves session to session (a
  31.7 tok/s code figure at p_min 0.3 read 36.5-37.6 in the next day's own 0.3 arm), and
  Flash-Next's does too (a classic arm read 50.5 in one session where the morning's read
  51.2). Compare only inside a session.
- **Never claim high-power mode.** The `pmset -g` line is reported verbatim as of the
  session and nothing more is inferred from it. Where the source did not record a line,
  the column says so rather than guessing. See [benching.md](benching.md) for the key
  confusion behind those two spellings.
- **Take headline numbers from unprofiled runs.** The per-step profilers rank steps and
  do not price them ([benching.md](benching.md)).
- Ranges span the medians the shipped configuration was measured at, over interleaved
  rounds.

## Current figures

| Checkpoint | Figure | Value | Measured | Power line as recorded |
| --- | --- | --- | --- | --- |
| Qwen3.6-27B | plain decode | 24.8-25.3 tok/s | 2026-08-08, commit not recorded | `lowpowermode 0` |
| Qwen3.6-27B | drafted decode, code | 37.5-38.2 tok/s (+46-52% over its own plain arm) | fitted 2026-08-08 | `lowpowermode 0` |
| Qwen3.6-27B | drafted decode, chat | 36.8-37.4 tok/s (+46-52%) | fitted 2026-08-08 | `lowpowermode 0` |
| Qwen3.6-27B | prefill @880 | 702 tok/s (chunk 512) | 2026-07-29, not re-measured; recorded as `@925` at the time, which is the fixture's NAME | not recorded |
| Qwen3.6-27B | prefill @3851 | 445 tok/s (chunk 512) | 2026-07-29, not re-measured; recorded as `@4k` at the time | not recorded |
| Qwen3.6-35B-A3B | plain decode | 127.0 tok/s | 2026-09-06, 24c4069; the `XWEN_ROUTER_MV_CLASSIC` arm read 115.1 in the same session, so +10.3%, ahead in every round | not recorded |
| Qwen3.6-35B-A3B | drafted decode, code | 133.6-134.8 tok/s (+26-28%) | fitted 2026-08-08, against the pre-fold plain level; superseded 2026-09-06: reads below plain; off by default | `lowpowermode 0` |
| Qwen3.6-35B-A3B | drafted decode, chat | 122.3-123.7 tok/s (+15-17%) | fitted 2026-08-08, against the pre-fold plain level; superseded 2026-09-06: reads below plain; off by default | `lowpowermode 0` |
| Qwen3.6-35B-A3B | presence penalty A/B, code, 256 tokens | plain 126.5 (p 0) / 126.9 (p 1.5); drafted 121.1 at 63.0% acceptance (p 0) / 119.6 at 59.4% (p 1.5) | 2026-09-06, pinned build of the penalty tree, 3 interleaved reps, medians | `lowpowermode 0` |
| Qwen3.6-35B-A3B | prefill @3851 | 2634 tok/s at chunk 2048, 2429 at 512 | 2026-08-30 | `powermode 0` |
| Qwen3.6-35B-A3B | prefill @3803 | 3081-3090 tok/s after the FFN-glue levers; the same sweep's all-classic arm read 2746-2755 | 2026-08-30 | `powermode 0` |
| Qwen3.8-27B | plain decode | 23.7-24.8 tok/s | 2026-08-15 | `lowpowermode 0` |
| Qwen3.8-27B | drafted decode, code | 34.4-35.7 tok/s (+44-45%), acceptance 80.0% | fitted 2026-08-15 at p_min 0.7, depth 4 | `lowpowermode 0` |
| Qwen3.8-27B | drafted decode, chat | 33.1-34.0 tok/s (+37-38%), acceptance 77.8% | fitted 2026-08-15 at p_min 0.7, depth 4 | `lowpowermode 0` |
| Qwen3.8-27B | prefill | no figure of its own; it runs the dense 27B graph | | |
| Qwen3.8-Flash-Next | plain decode @596 | 52.9 tok/s | 2026-09-06, 24c4069; its `XWEN_ROUTER_MV_CLASSIC` arm read 50.5 in the same rounds, so +4.8%, ahead in all three | not recorded |
| Qwen3.8-Flash-Next | plain decode by context | 46 tok/s below the 2048 indexer budget, 44-45 at 3.8k-32k | 2026-08-30, after the QSA block-key cache, fused gather and device-side selection | `powermode 0` |
| Qwen3.8-Flash-Next | plain decode, 2..8-token forwards | 149.7 tok/s at chunk 8, 108.9 at 4, 68.1 at 2; the `XWEN_HC_GATE_CLASSIC` arm read 93.2 / 69.5 / 38.6, so +57-76% | 2026-09-06, every forward forced to n tokens with `XWEN_PREFILL_CHUNK` | not recorded |
| Qwen3.8-Flash-Next | drafted decode | none; the checkpoint ships no drafter and decodes plain, saying so | | |
| Qwen3.8-Flash-Next | serve decode | 42-47 tok/s through 32k, at parity with `generate` | 2026-08-30 | `powermode 0` |
| Qwen3.8-Flash-Next | prefill @131424 | 428-456 tok/s (288-307 s, four repetitions), peak footprint 31 GB, after the sparse-tile attention (282-296 and 28 GB on the device mask alone earlier the same day, 231 and 59 GB that morning on the host selection) | 2026-09-06, [record](records/qsa-sparse-prefill.md) | `lowpowermode 2` (owner-set high performance; the 282-296 row was on automatic) |
| Qwen3.8-Flash-Next | prefill @3851 | 1140 tok/s (1010 before the device-side PLE gate and conv, +12.8%) | 2026-09-05 | `lowpowermode 0` |
| Qwen3.8-Flash-Next | prefill @880 | 1262 tok/s (1118 before it, +12.9%) | 2026-09-05 | `lowpowermode 0` |
| Qwen3.8-Flash-Next | prefill @530 | ~796 tok/s | 2026-08-29, after the P3 kernel pass | `powermode 0` |
| Qwen3.8-Flash-Next | prefill @7606 | ~860 tok/s after the FFN-glue levers, against 766 in the all-classic arm | 2026-08-30 | `powermode 0` |
| all Q4_K_M checkpoints | load | 2.8-3.0 s; a cold first run adds ~9 s of Metal pipeline compilation | recorded alongside the 2026-08-30 prefill figures, checkpoint not named | not recorded |
| all Q4_K_M checkpoints | memory | 19.2 GB resident at max_ctx 8192 | recorded alongside the 2026-08-30 prefill figures, checkpoint not named | not recorded |

Notes on individual rows:

- **The 35B fold has not been re-swept with drafting**, so both drafted 35B figures are
  against the pre-fold plain level. **The 2026-09-06 penalty A/B read drafting BELOW
  plain on the 35B at penalty 0 (121.1 vs 126.5, code prompt, 256 tokens)**: after the
  router gemv lifted plain by 10%, the drafted arm on that prompt no longer clears it.
  One prompt, one length; ledgered as a measured item in "Drafting".
- **The dense 27B keeps chunk 512.** It reads 5-6% slower at 2048 (650/599 vs 608/571,
  2026-08-30); the MoE checkpoints use 2048. `XWEN_PREFILL_CHUNK` overrides, and the rule
  is `Arch::prefill_chunk_default` (decisions.md "The prefill chunk is per architecture").
- **The router-gemv session of 2026-09-06 reported Flash-Next prefill unchanged at 1171**
  without restating the prompt length; the length-tagged prefill rows above are the ones
  to quote.
- **The bench fixtures are named after laguna's tokenizer, not this one** (2026-09-06).
  `tests/fixtures/bench-prompts/prefill-925.txt` is 880 tokens under the Qwen tokenizer,
  `prefill-4k.txt` is 3851 and `decode-630.txt` is 596; the files are wikitext-2 English
  prose and nothing about them is checkpoint-specific. The names are kept because
  `scripts/bench.ts`, the log and the records all reach them by name. So `@925`, `@4k`
  and `@630` anywhere in this repo are FILE NAMES, and the counts to quote are 880, 3851
  and 596 — which is what every row measured since 2026-08-29 already does.
- **llama.cpp on the same Flash-Next file, in the same hour as the 2026-08-29 arm, ran
  789 prefill / 41.4 decode** (`pmset -g` said `powermode 0` that session).
- **Flash-Next decode is bimodal round over round** (~42 vs ~44 at the pre-fold level) and
  unexplained.
- **Cross-drafter comparison, 2026-08-15**, the only honest way to compare the two drafter
  kinds, same machine and same hour: the 3.6-27B's DFlash head runs 1.50x/1.47x over its
  own plain arm where the 3.8-27B's MTP head runs 1.45x/1.38x over its own. Same trunk
  geometry, so the block drafter is still the stronger drafter; the MTP head closes most
  of the gap and is worth roughly ten times less KV, 4 KiB/token against 40.

## Long context

Measured 2026-09-06 by `scripts/longctx.ts` against a pinned worktree build of 4a66616,
`lowpowermode         0` with no high-power claim. Medians of two interleaved
repetitions per length, lengths run A B A B rather than all reps of one length in a row.
The prompt is repo prose cut to the token target against the checkpoint's own GGUF vocab
and fed through the chat template with a 160-token thinking floor, which is what keeps
decode a rate at every length: a raw continuation of a cut-off document emits an
end-of-generation token almost immediately, and a 32768-token raw run decoded 28 tokens
in 0.67 s before this harness stopped asking it to.

Every figure below is at one large model process with another agent's builds on the same
CPU; treat the absolutes as this session's and the shape as the finding.

**Qwen3.6-35B-A3B, plain.**

| Prompt tokens | Prefill tok/s | Prefill wall | Decode tok/s | Peak footprint |
| --- | --- | --- | --- | --- |
| 8201 | 2326 | 3.6 s | 96.4 | 12.0 GB |
| 32879 | 1586 | 20.8 s | 78.9 | 14.0 GB |
| 65554 | 1145 | 57.3 s | 60.3 | 25.0 GB |
| 131382 | 668 | 196.6 s | 36.8 | 50.5 GB |

**Qwen3.8-Flash-Next, plain.**

| Prompt tokens | Prefill tok/s | Prefill wall | Decode tok/s | Peak footprint |
| --- | --- | --- | --- | --- |
| 8243 | 925 | 8.9 s | 47.1 | 20.0 GB |
| 32921 | 584 | 56.3 s | 46.0 | 20.0 GB |
| 65596 | 403 | 162.7 s | 46.9 | 27.0 GB |
| 131424 | 428-456 | 288.1-306.7 s | 44.4-48.6 | 31.0 GB |

The 131424 row was retaken twice on 2026-09-06: by [the QSA device-mask
arc](records/qsa-device-mask.md) (282-296 tok/s, 28 GB) and then by [the sparse-tile
attention](records/qsa-sparse-prefill.md), which is the row shown, four repetitions in
owner-set high performance mode; on the host-selection path the envelope had measured
it at 231 / 569.3 s / 41.9 / 59.0 GB. The other three rows predate that
arc and their prefill figures are conservative: the same arc's greedy check read the 8243
row at 947.8 on the host arm against 1096.5 on the device arm, +15.7%, one repetition
each.

Three things to read off these:

- **Flash-Next decode is flat where the 35B's is not.** 47.1 to 46.9 tok/s from 8k to
  64k, then 41.9 at 128k, against the 35B's 96.4 falling to 36.8 — a 62% loss on the
  dense-attention checkpoint against 11% on the sparse one, and none of that 11% before
  64k. QSA is doing exactly what it is for, and the 2026-08-30 "44-45 at 3.8k-32k" row
  extends to 64k unchanged. On the 35B, long-context decode is a different regime from
  the headline 127.0 and should never be quoted from it.
- **Prefill falls on both, and it is the wall an operator actually hits.** A maximal
  prefill is 197 s on the 35B and was 569 s on Flash-Next, 445-463 s after the QSA
  device mask and 288-307 s after the sparse-tile attention the same day; Flash-Next is
  still slower per token at every length, and the probe now says why on both: attention
  is 77-81% of the 35B's 128k prefill and was 52% of Flash-Next's. The
  231 tok/s floor is what `queue_timeout` is derived from (decisions/serving.md), and it
  stays at 231 on purpose: a floor that only rises keeps the timeout conservative.
- **Peak footprint quadrupled on the 35B**, 12.0 to 50.5 GB, on weights of 20.4 GB and a
  KV cache of 2.6 GB at 131072 — so ~28 GB of the peak was neither. It was the prefill
  mask, and building it on the device fixed it: the same 131072 run now peaks at a flat
  **17 GB against 42-69 GB** on the host path, measured in the same binary with
  `XWEN_HOST_MASK=1` as the control arm. Flash-Next did not move (59 GB either way)
  because its QSA indexer built its own host mask per sparse layer per chunk; that mask
  moved to the device later the same day and the peak is 28 GB since
  ([the QSA device-mask record](records/qsa-device-mask.md)). See
  [the long-context envelope record](records/long-context-envelope.md).

The prefill-mask A/B itself, one repetition per arm on the working-tree build, both arms
in the same binary:

| Checkpoint, 131072 tokens | Prefill tok/s | Decode tok/s | Peak footprint |
| --- | --- | --- | --- |
| 35B-A3B, host fill | 667.8 | 37.0 | 42 GB |
| 35B-A3B, device build | 659.2 | 36.6 | 17 GB |
| Flash-Next, host fill | 230.8 | 41.9 | 59 GB |
| Flash-Next, device build | 230.9 | 42.1 | 59 GB |

A dead heat on time on both checkpoints. Do not quote the device mask as a throughput
win; it is a memory win on the dense-attention path and nothing else.

## Ceilings

Measured 2026-09-05 for Flash-Next (log.md "Ceiling diagnosis"; decisions.md "Ceilings"),
refined 2026-09-06. These are what rank the remaining levers.

**Decode.** A token reads 6.33 GB of weights plus ~0.3 GB of state and KV, which is
11.7-12.3 ms of its 21.3 at the measured bandwidth, so the bytes-only ceiling is
**81-86 tok/s**. The other ~9 ms is ~1740 dispatches in a mostly dependent chain
(hc 672, MoE 576, GDN 252) at ~4 µs average, a residual between the measured 2.5 µs floor
and the 8.4 µs gemv intercept, plus 3 syncs and the serial scan. Decode is not CPU-bound
(3.7 ms CPU per token) and not command-buffer-bound.

**Prefill.** At 3851 tokens it runs 13.7 TFLOP/s end to end on 12.07 GFLOP/token. The
dispatch floor is under 1%, weight re-reads 9% (inside the gemm time, a lower bound), and
the expert gemms 14-43% (an amortized bench says 43%, two in-situ A/Bs bracket it lower;
~12 TFLOP/s isolated, dequant-bound by the 2026-08-30 code reading).

**The levers are dispatch COUNT for decode and the expert gemm plus hc glue for prefill,
never per-kernel bandwidth.** Three refinements to that budget, all measured:

1. The fused hc gate (2026-09-05) removed 384 launches and measured +9% against the
   budget's +7.8%, confirming the attribution in situ. The hc population is now 288 and
   ~1356 launches remain per token below the indexer budget.
2. The ~4 µs average is an average over launches of very different byte weight. It
   predicts only a fusion whose launches carry under ~2 MB, less than ~4 µs of traffic at
   rate, AND sit on the dependent chain (2026-09-06, log.md "Fused MoE shared expert").
   The shared expert's five launches per layer were byte-bound at ~535 GB/s, so removing
   192 of them recovered the gaps only, +0.6% against +3.5-4% predicted. Byte-bound or
   overlapped launches yield their gaps, never the budget figure. The MoE population is
   384 after that fusion, 12 dispatches per layer having become 8.
3. **Occupancy is a third class of decode cost beside bytes and launch gaps** (2026-09-06,
   log.md "Router projection on a 256-threadgroup gemv"). The router projection read zero
   on the probe and 4% of a token's bytes on the budget, yet moving it off candle's
   8-threadgroup mlx gemv onto a 256-threadgroup vendored one was worth +10.3% on the 35B
   and +4.8% on Flash-Next. A kernel that leaves the GPU mostly idle is invisible to both
   instruments. The audit that would find the rest, threadgroup count against bytes for
   every decode dispatch, is ledgered and unbuilt.

## Dense Qwen3-4B, which is not a design target

Kept apart from the table above on purpose. The design target names the Qwen 3.6 and 3.8
GGUF checkpoints; Qwen3-4B is in the repo as a correctness target and as the
conditioning encoder (decisions.md "Dense Qwen3-4B is a full checkpoint AND the
conditioning encoder"), so these figures exist to be known, not to be improved. Nothing
here ranks a lever.

Measured 2026-09-07 on the release binary of c05d631, plain decode with `--no-draft`
(there is no drafter for this architecture), batch 1. `pmset -g` read `lowpowermode 2`;
no high-power claim. **One caveat that would disqualify these as A/B numbers**: CPU-only
debug builds were running in the background, which the benching rules forbid beside a
bench. They are single-configuration levels with nothing to compare against inside their
own session, so the contention costs accuracy and cannot flip a comparison.

| Figure | Value |
| --- | --- |
| plain decode, 7-token prompt, 256 tokens | 63.1 tok/s |
| plain decode at a 3890-token context, 128 tokens | 55.9 tok/s |
| prefill @3890 | 3416, 3432, 3433 tok/s over three runs |

**Both are close to their ceilings, and that is the finding.** Decode reads all 4.022 G
BF16 parameters per token, 8.04 GB (7.27 GB of layers plus the 0.78 GB tied head, read in
full because the logits need every row), plus 147 KB of KV per context token, 0.57 GB at
3890. At the ~535 GB/s this repo measures for byte-bound kernels that is a bytes-only
ceiling of 66.5 tok/s short-context and 62 at 3890, so the measured figures are **95% and
90% of them**. The residual is the 36-layer dependent dispatch chain, which is a far
smaller population than the MoE checkpoints carry. Prefill at 3890 costs ~8.4
GFLOP/token (7.27 of layers, ~1.15 of causal attention), so 3433 tok/s is ~29 TFLOP/s end
to end, inside the 28-36 TFLOP/s the Metal-4 tensor gemm measures in isolation: **prefill
runs at the gemm's own rate**, with a ceiling of 3300-4300 depending where in that range
this hardware's peak sits. There is no cheap lever on either, which is the answer this
section exists to give rather than a problem it poses.

## Z-Image-Turbo, a time per image and not a tok/s target

Kept apart from everything above for the same reason the 4B is: image transformers are in
scope as a correctness target, and this figure exists so that it can be known
(decisions.md "Diffusion image transformers are in scope, held to correctness bars
first"). Nothing here ranks a lever, and nothing here justifies a change to the language
models' hot path.

Measured 2026-09-08 on a dev-tree release build of master e5d9775, which is **not a pinned
binary**, batch 1, 8 steps, no CFG, warm, nothing else on the GPU. `pmset -g` read
`lowpowermode 0`; no high-power claim
([records/zimage-perf.md](records/zimage-perf.md) "Results on master e5d9775, clean").

| Figure | 1024x1024 | 512x512 |
| --- | --- | --- |
| transformer step, first step of a warm run | **1.35 s** (was 1.78 that morning, 3.06 the day before) | 0.35 s |
| transformer step, by step 8 | **2.0-2.1 s** (was 2.15) | 0.39-0.40 s |
| eight steps together | 14.1-16.1 s (was 17.2 at the old plateau) | 3.1 s |
| VAE decode | **1.36-1.44 s** (was 5.19; one run of three read 1.78) | **0.26 s** (was 1.06-1.10) |
| render, 8 steps plus the decode | **15.5-17.5 s** (was ~21.5) | ~3.4 s |
| total wall, one image, warm | 22-25 s (was ~25, and 50.2 on 2026-09-07) | **8.1 s** (was 8.6, and 14.7) |

**A step is a range with its ramp now, not a steady number.** Three warm runs read, step by
step, 1.62 1.81 2.33 2.14 2.09 2.12 1.96 2.02; 1.47 1.56 1.65 1.71 1.80 1.93 2.04 3.27; and,
after a cold page cache, 1.35 1.36 1.61 1.77 1.94 1.91 2.06 2.12. The first step fell 1.78 to
1.35 s (24%) and the isolated attention kernel is 3.5x faster, yet the step still reaches
1.9-2.1 s by the eighth, so the plateau moved 0.1-0.2 s. The ramp is 55% over eight steps
where it was 20% that morning and 17% the day before. That is the signature of a power or
thermal cap, faster kernels reaching the throttle sooner and the step past it governed by
the envelope rather than the kernel. **It is an observation and not a confirmed cause**; the
confirmation is a `powermetrics` trace during a run, which needs sudo from a user shell, and
it is the first item on the Front because it decides whether any further kernel win converts
to wall time at the plateau (decisions.md "A Z-Image step is quoted at steady state", as
amended). Quote "1.35 s first step rising to 2.0-2.1 s by step 8".

| Figure, resolution-independent | Value |
| --- | --- |
| encode, 20-39 tokens | 6-11 ms warm, 166 ms cold |
| transformer + VAE load | 3.5-5.0 s warm, 44.7 s after a cold page cache (fp32 to bf16 cast) |
| text encoder load | 0.8-1.2 s warm, 3.3 s cold |

A 1024x1024 step is about 62 TFLOP, so **the 1.35 s first step is roughly 46 TFLOP/s end to
end** and the 2.0-2.1 s eighth step about 30, against 29 at the old 2.15 s plateau, 17 at
3.6 s and 11.6-12.4 before any of this work. The four gemms are 46.7 TFLOP; at the 37 TFLOP/s
the kernel measured in isolation on 2026-09-07 they alone would be 1.26 s, more than the
first step leaves after attention, so the cool chip runs the gemm above any isolated figure
taken on a warm one (the FFN arc's isolated gemm spanned 37 to 45 TFLOP/s across a session).
**Attention runs on the tensor units now**: `ops::flash_attn_tensor` reads 45.1 TFLOP/s
isolated at 30 x 4128 x 128 against the steel copy's 13.1, and profiled `attn.sdpa` fell 698
to 231 ms, the fifth row of the table where it was the second. **The VAE decodes on a direct
conv at 10-11 TFLOP/s** against candle's 1.2-4.4, and its profiled table now reads 1395 ms
against 1.36-1.44 s real, so the decode's marks cost almost nothing with the im2col buffers
gone. Where a step goes, on master e5d9775, PROFILED milliseconds for ranking only
(`/tmp/arc3-1024-prof.log`, 3121 ms of table against a 1.35-2.1 s step):

| row | profiled ms per step |
| --- | --- |
| `ffn.w1w3` | 863 |
| `attn.qkv` | 392 |
| `ffn.w2` | 272 |
| `ffn.silu_mul` | 254 |
| `attn.sdpa` | 231 |
| `attn.norm+scale` | 184 |
| `attn.qknorm` | 138 |
| `ffn.norm+scale` | 124 |
| `attn.transpose` | 121 |
| `attn.rope` | 119 |
| `attn.out` | 85 |
| `attn.untranspose` | 58 |
| the two gated residuals | 26 each |

**No deflator is quoted for this table**, and the reason is a rule rather than an omission:
the previous deflators were fitted against a 2.15 s plateau that no longer exists, and the
2026-09-08 refutation of the SwiGLU store showed a gemm row can carry the whole cost of its
intermediate's re-allocation (`ffn.w1w3` at 24.5 TFLOP/s profiled against 37-45 isolated).
Take a rate from `tests/zimage_microbench.rs` and never from a row (decisions.md "A profiled
row that shows a fusion win is not a result until the fusion is confirmed unprofiled"). The
ranking with what is left is [records/zimage-perf.md](records/zimage-perf.md) "Lever
ledger": gemm launch fusion unpriced, the elementwise tail bounded by the residual at
~0.1-0.3 s a step, the VAE mid-block attention ~0.1 s, and the power envelope as the
instrument that prices all of them.

Footprint was **not measured**: `footprint`, `ps` and `vmmap` were all refused from the
agent sandbox. The load figures
predict about 20 GB resident (7.6 GB encoder, 12.3 GB bf16 transformer, 0.34 GB f32 VAE)
plus activations, and measuring it from a user shell is a ledger item. One residency fact
came out of reading the VAE path rather than measuring it: candle's im2col conv rounds
its 9x-inflated buffer up to a power of two and pools it without ever freeing, so a
process that has decoded once at 1024x1024 holds a 16 GB Metal private buffer for its
lifetime. That was the candle arm; the direct conv path of 2026-09-08 allocates no im2col
buffer, so on the default `XWEN_ZIMAGE_VAE=xwen` arm that bucket should be gone, at the cost
of about 200 MB of permuted f32 weight planes beside the candle copies. Still unmeasured.

Earlier readings, kept as history. The first ones, 2026-09-07 on 493ae2e before any
performance work, were 5.0-5.5 s per 1024x1024 step, VAE 4.86-5.11 s, total wall
51.7-52.5 s warm and 82.3 s cold. After the tensor-gemm arc that evening (e54801f,
ab1cde2) the step read 3.6 s steady and 3.06 s first at 1024x1024, 0.63-0.68 s at 512x512,
and a warm image 37.7 s and 10.3 s. After the rope-and-norm and flash arcs of 2026-09-08
(bee11da, 66e7202, merged as e630ebb) the step read 2.15 s steady and 1.78 s first, the VAE
5.19 s untouched, a warm image ~25 s and 8.6 s, 29 TFLOP/s end to end; the two arcs composed
almost perfectly, 2.62-2.74 and 3.42-3.46 alone off a 3.55-3.67 base. At that state a merged
profiler pass fitted two deflators, 1.19x on the gemm and sdpa rows and about 3x on the
elementwise ones, putting a step at 1.26 s of gemms, 0.53 s of attention and 0.34 s of
everything else; the gemm half of that fit was later shown to carry the pool eviction too
([records/zimage-perf.md](records/zimage-perf.md) "The merged profile, and the deflator
refitted").

**Under the serve route, 2026-09-07.** The same pipeline behind `POST
/v1/images/generations` (Arc C), on a dev-tree release build, the language model never
loaded, power mode NOT read; ordered against the CLI figures above, not calibrated.

| Figure, images route | Value |
| --- | --- |
| 1024x1024, 8 steps, cold: load plus render | 80.4 s (load 33.6: encoder 2.3, transformer and VAE 31.4; render 46.7) |
| 1024x1024, 8 steps, warm, the stock ComfyUI payload on the proxy path | 48.6 s |
| 512x512, 8 steps, `n: 2`, warm | 23.3 s, 11.6 s per image |

The warm 1024x1024 time per image under the route is the figure to quote for "how long
does ComfyUI wait"; it is the CLI wall less the process start, and it moves only when the
step time does ([records/zimage-pipeline.md](records/zimage-pipeline.md) "Arc C"). These
three rows were taken before the tensor-gemm arc and the route has not been re-timed
since, and four performance arcs have landed on top of that, the CLI render going from
about 47 s to 15.5-17.5 s, so subtract roughly 30 s from each on the strength of the CLI
figures above rather than quoting them as current.

**The first cross-implementation datum, 2026-09-07, 512x512.** The Stage 3 dump ran the
same weights through diffusers 0.40 on torch 2.14 mps, so the two sides were timed on one
machine within minutes of each other, on the same 512x512 case. Power mode was NOT read
in that session, so these are ordered, not calibrated; dev-tree build, not a pinned
binary.

| 512x512, per transformer step | xwen, as measured then | torch mps bf16 | torch mps fp32 |
| --- | --- | --- | --- |
| transformer step | 1.23 s | 0.35 s | 1.10 s |
| VAE decode, f32 | 1.06 s | 0.7 s | |

torch's bf16 arm was about 3.5x faster per step than xwen at this size, and torch's fp32
arm ran level with xwen. That was the ceiling evidence the step-time ledger item lacked,
and the tensor-gemm arc of the same evening spent most of it, with the 2026-09-08 arcs
spending the rest: xwen's 512x512 step is 0.37-0.41 s against torch's 0.35, so the two are
now level where torch was 3.5x ahead, and the third arc of 2026-09-08 took it to
0.35-0.40. torch was not re-timed. On the VAE, torch's 0.7 s at 512x512 extrapolates to
roughly 2.8 s at 1024x1024, since decoder work scales with pixel count; xwen's decode was
5.0 s and about 1.75x off until the direct conv path of 2026-09-08, and reads 0.26 s at
512x512 and 1.36-1.44 s at 1024x1024 now, so xwen is ahead of that torch reading by 2-3x
([records/zimage-perf.md](records/zimage-perf.md)).

**The ceiling to read a step against is compute, not bandwidth, and that inverts every
intuition the rest of this file has built up.** A 1024x1024 step is about 62 TFLOP (roughly
57 of linear layers at ~4200 tokens and ~10 of attention) against 12.3 GB of weight
traffic, an arithmetic intensity near 5100 FLOP/byte where this machine's ridge point is
24 to 114. Reading the weights is about 20 ms of a multi-second step. So achieved GEMM
rate sets the time and nothing else does, and the numbers to rank a lever against are
these three. The hardware peak is **about 70 TFLOPS fp16 and bf16 at medium confidence**,
an extrapolation from a 5-core A19 microbenchmark of 1024 fp16 FLOP per core per cycle
scaled to 40 cores at an assumed 1.75 GHz, not a measurement; the published measured
figures for this chip all sit far below it and are a floor rather than a ceiling, because
this machine has beaten them. **38 TFLOPS is demonstrated by another implementation**,
torch MPS bf16 on this model at 512x512. **36-38 TFLOPS is what xwen's own tensor gemm
measures** at the model's shapes in isolation on a warm chip, 37-45 across a session, and
the first step of a warm run is about 46 TFLOP/s end to end after the third arc of
2026-09-08, so the first step runs AT the rate of its dominant kernel and every large plane
is on the tensor units: the gemms, attention at 45 TFLOP/s isolated, and the VAE's convs at
10-11 on simdgroup MMA in f32, which the 60 dB bar keeps off the tensor path. What is not at
that rate is the eighth step, 2.0-2.1 s and about 30 TFLOP/s, and whether that is the
envelope or the kernels is the open question above. At 70 TFLOPS a step would be 0.85 s.
Quantization is not on that
list: it is a footprint lever here and cannot move a step (decisions.md "The transformer
runs bf16 end to end"). The lever ranking itself lives in
[records/zimage-perf.md](records/zimage-perf.md).

## History

Narrative, protocol and the tables that produced these figures live in the log and its
records, not here:

- [records/zimage-perf.md](records/zimage-perf.md), the 2026-09-07 tensor-gemm arc and the
  three 2026-09-08 arcs (rope and norm, flash attention, then the VAE conv path and the
  tensor-op attention kernel): the microbench that priced the step, the kernel-class A/B,
  the roofline, the lever ledger, the three refutations and the ramp observation.
- [records/zimage-pipeline.md](records/zimage-pipeline.md), the 2026-09-07 first image and
  the conditions its timings were taken under.
- [records/router-gemv.md](records/router-gemv.md), the 2026-09-06 occupancy lever.
- [records/fused-moe-shared-expert.md](records/fused-moe-shared-expert.md) and
  [records/hc-gate-ragged-and-probe-decode.md](records/hc-gate-ragged-and-probe-decode.md),
  the rest of that day.
- [records/fused-hc-gate.md](records/fused-hc-gate.md),
  [records/ceiling-diagnosis.md](records/ceiling-diagnosis.md),
  [records/ple-device-tail.md](records/ple-device-tail.md) and
  [records/ple-readbacks.md](records/ple-readbacks.md), 2026-09-05.
- [records/mm-id-tiles.md](records/mm-id-tiles.md), the expert-gemm tile work whose
  end-to-end reading was later corrected by the duplicate-dispatch probe.
- The 27B prefill gap, **closed 2026-07-29 (P8c)**: it was never the DeltaNet scan, which
  is 3% of prefill, but the dense SwiGLU FFN running candle's `kernel_mul_mm_q4_K_f32` at
  ~12-13 TFLOP/s where the Metal-4 cooperative-tensor gemm does 28-36. `src/ops/dense_mm.metal`
  made 27B prefill 2.2-2.7x faster, 270 to 702 @925 and 236 to 445 @4k, against
  llama.cpp's 486 / 502. A +350-560 µs/token residual outside all measured stages still
  degrades with length (TODO.md), and it is most of why 4k fell short of the profile's 496
  upper bound while 925 met it. See
  [records/dense-ffn-prefill-gemm.md](records/dense-ffn-prefill-gemm.md) and
  [records/27b-prefill-residual.md](records/27b-prefill-residual.md).
