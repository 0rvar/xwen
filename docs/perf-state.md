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
| Qwen3.6-27B | prefill @925 | 702 tok/s (chunk 512) | 2026-07-29, not re-measured | not recorded |
| Qwen3.6-27B | prefill @4k | 445 tok/s (chunk 512) | 2026-07-29, not re-measured | not recorded |
| Qwen3.6-35B-A3B | plain decode | 127.0 tok/s | 2026-09-06, 24c4069; the `XWEN_ROUTER_MV_CLASSIC` arm read 115.1 in the same session, so +10.3%, ahead in every round | not recorded |
| Qwen3.6-35B-A3B | drafted decode, code | 133.6-134.8 tok/s (+26-28%) | fitted 2026-08-08, against the pre-fold plain level | `lowpowermode 0` |
| Qwen3.6-35B-A3B | drafted decode, chat | 122.3-123.7 tok/s (+15-17%) | fitted 2026-08-08, against the pre-fold plain level | `lowpowermode 0` |
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
| Qwen3.8-Flash-Next | prefill @3851 | 1140 tok/s (1010 before the device-side PLE gate and conv, +12.8%) | 2026-09-05 | `lowpowermode 0` |
| Qwen3.8-Flash-Next | prefill @880 | 1262 tok/s (1118 before it, +12.9%) | 2026-09-05 | `lowpowermode 0` |
| Qwen3.8-Flash-Next | prefill @530 | ~796 tok/s | 2026-08-29, after the P3 kernel pass | `powermode 0` |
| Qwen3.8-Flash-Next | prefill @7606 | ~860 tok/s after the FFN-glue levers, against 766 in the all-classic arm | 2026-08-30 | `powermode 0` |
| all Q4_K_M checkpoints | load | 2.8-3.0 s; a cold first run adds ~9 s of Metal pipeline compilation | recorded alongside the 2026-08-30 prefill figures, checkpoint not named | not recorded |
| all Q4_K_M checkpoints | memory | 19.2 GB resident at max_ctx 8192 | recorded alongside the 2026-08-30 prefill figures, checkpoint not named | not recorded |

Notes on individual rows:

- **The 35B fold has not been re-swept with drafting**, so both drafted 35B figures are
  against the pre-fold plain level.
- **The dense 27B keeps chunk 512.** It reads 5-6% slower at 2048 (650/599 vs 608/571,
  2026-08-30); the MoE checkpoints use 2048. `XWEN_PREFILL_CHUNK` overrides, and the rule
  is `Arch::prefill_chunk_default` (decisions.md "The prefill chunk is per architecture").
- **The router-gemv session of 2026-09-06 reported Flash-Next prefill unchanged at 1171**
  without restating the prompt length; the length-tagged prefill rows above are the ones
  to quote.
- **llama.cpp on the same Flash-Next file, in the same hour as the 2026-08-29 arm, ran
  789 prefill / 41.4 decode** (`pmset -g` said `powermode 0` that session).
- **Flash-Next decode is bimodal round over round** (~42 vs ~44 at the pre-fold level) and
  unexplained.
- **Cross-drafter comparison, 2026-08-15**, the only honest way to compare the two drafter
  kinds, same machine and same hour: the 3.6-27B's DFlash head runs 1.50x/1.47x over its
  own plain arm where the 3.8-27B's MTP head runs 1.45x/1.38x over its own. Same trunk
  geometry, so the block drafter is still the stronger drafter; the MTP head closes most
  of the gap and is worth roughly ten times less KV, 4 KiB/token against 40.

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

## History

Narrative, protocol and the tables that produced these figures live in the log and its
records, not here:

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
