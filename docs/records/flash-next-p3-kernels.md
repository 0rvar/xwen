# 2026-08-29 (P3, later the same day) — Flash-Next prefill 239 → 796 tok/s and decode 37.8 → 45: a Q5_1 `mm_id` arm, four fused hyper-connection kernels, and a norm split across streams below 32 tokens

Moved verbatim from [docs/log.md](../log.md) on 2026-09-06; the log keeps this entry's
opening paragraph and links here.


**Context.** P2 closed hours earlier with the graph correct and prefill 3.3-3.5x behind
llama.cpp on the identical file. P3 opened on that gap. Everything below is the
`UD-Q4_K_XL` file at a 530-token prompt, arms interleaved, `pmset -g` reporting
`powermode 0` this session — no `lowpowermode` key appeared, which is the reverse of
what this machine usually prints, and is recorded verbatim rather than normalised.
High-power mode is still not positively confirmable here, so it is not claimed.

**Measure first, and the profiler had holes.** `XWEN_STACK_PROFILE` let three
Flash-Next-only costs — the PLE layer, QSA selection and the token readback — fall into
`inter_stage_host`, so they were invisible by construction. b54046b gave each its own
bracket (the qwen35 path emits none of them) and the attribution came out unambiguous:
at 530 tokens prefill is **ffn 51.0%, hyper-connection glue 34.3%** (`hc_ffn.read` 12.1,
`hc_attn.read` 11.6, the two `hc_write`s 10.6), `mixer_delta` 8.4, `mixer_full_attn` 3.7,
`ple` 1.9, `qsa_select` 0.3, unaccounted 0. At 880 tokens the same shape with ffn at 54%.
Two things were ruled out before any kernel was written: `XWEN_CHUNK_SYNC` moved prefill
by under 0.5%, so nothing accumulates across chunks, and the first profiled run of a
fresh cache was cold — 36 s load, 136 tok/s — because `XWEN_BENCH`'s warmup does not
cover shard page-in. Discard the first run after a fresh cache; that one is a trap, not a
measurement.

**The Q5_1 `mm_id` arm (8112733) took prefill 239 → 443.** `ffn_down_exps` is Q5_1 on 43
of 48 layers by the 640-column rule, `FusedExperts::use_mm` is all-or-nothing per layer,
so one unsupported down plane sent all three expert planes of those layers through
per-token `mul_mv_id` at prefill. `block_q5_1` and `dequantize_q5_1` are copied verbatim
from the pinned llama.cpp and the tile loader was already 32-block generic through Q8_0,
so the arm is small; it is instantiated for the classic, `_hp` and `_t` families. The ffn
stage went 2887 → 1031 µs/token and the gap to llama.cpp 3.30x → 1.78x. Decode did not
move (37.7), which is expected — decode still takes candle's baked
`kernel_mul_mv_id_q5_1_f32`. `XWEN_NO_MM_ID=1` is **not** the baseline for this arm: it
forces mv on all three planes and reads 225 tok/s, below the real before-arm.

**Then the hyper-connection glue (8aeed73), and the gap closed.** The read was 20 candle
dispatches, 17 of them glue around two Q8_0 gemms, and the write made three full-carrier
passes to do one FMA — twice per layer over 48 layers. Four kernels replace the glue and
the two Q8_0 projections stay `QMatMul`, because `low_rank` is 320, not 8: those are real
gemms moving ~0.7 GB of weights per token and they belong in the library. 5+1 dispatches
per layer-pair instead of 20+2; roughly 2128 → 600 hc dispatches per forward. Measured
A/B at fd46c7a: **prefill 764.9/780.9 fused against 446.0/437.1 classic, 1.75x**, with
llama.cpp at 788-790 in the same hour — the 3.5x prefill gap from P2 is gone. Stage
detail: `attn_norm` 726 → 209 µs/token, `ffn_norm` 726 → 227, the residual writes 325 →
105, prefill wall 2105 → 1279 ms.

**But fusion cost 6% of decode, and the fix was to stop being one threadgroup.**
Fused decode read 35.8/35.8 against classic's 37.5/38.0. Opus's review named the
mechanism before the profile did: at `n == 1` the fused `hc_norm` runs ONE threadgroup
per token, so each of the 97 launches per forward walked a 10240-wide row and the 160 KiB
injection head on a single threadgroup — the exact shape a candle chain of small
dispatches beats. Below `HC_SPLIT_MAX_N` (32) the norm now runs one threadgroup per
(token, stream) and the injection dot as a second kernel over the same grid (2c8d3b3).
**Decode 37.8 → 43.1 tok/s**, and it is bit-identical: same thread count and partition
for the statistics, stream-major walk for the dot, verified as byte-identical output
against both `XWEN_HC_SPLIT_MAX_N=0` and the pre-split fused build over 128 tokens.

**The PLE cost was page faults, not driver overhead, and the 6.4 ms figure was an
artifact.** The decode profile had PLE at ~6.4 ms of a 26.5 ms step — one layer taking a
quarter of decode — with a shape (~6 ms fixed per forward, ~44 µs/token) that reads like
a mid-forward device→host sync draining the pipeline. It was not. `XWEN_PLE_PROFILE`
(fd46c7a) splits the layer into hash, gather, uploads, projections, the three readbacks,
host gate, host conv and trail: the gather is **flat per token, median ~1.1 ms with 6.5 ms
spikes and only 4.7% page-cache hits** — 16 IQ4_NL rows demand-paged out of a 28.8 GB
table, per token, essentially never reused. The fixed floor is ~0.85 ms (projections
0.33, the three readbacks 0.50). The honest total is **~2.1 ms of a ~28 ms step, not
6.4**: the profiler's own syncs inflate every decode stage it brackets, so profiled
decode numbers rank stages and must never be quoted as timings. Prefill is the opposite
regime — a warm chunk gathers 8192 rows in 2 ms, a cold first chunk takes 439 ms.

**Prefetch, on the strength of that finding (ac40526).** The row addresses depend only on
token ids, so a background thread touches one byte per distinct page for the position
about to be forwarded — hinted at sample time for decode, before layer 0 for a prefill
chunk. Advisory only and never gated on the PLE gate value, which is computed mid-forward
and would serialize the lookup. The table's byte range gets `MADV_RANDOM` so a 90-byte row
does not drag a readahead window, while the whole-file `WillNeed` stays for the weights;
the row math is single-sourced through `PleTable::row_offset` so the prefetcher and the
gather cannot drift. `XWEN_PLE_NO_PREFETCH` and `XWEN_PLE_NO_RANDOM` exist for the A/B.
Measured effect: `measured 2026-08-29 with one cold prompt per arm (the same-prompt design is invalid — greedy decode hashes every arm to the same rows, so arm k warms arm k+1): median decode gather 0.002 ms with prefetch vs 0.97-1.02 ms without, PLE total 1.05 vs 2.03 ms per token, decode 45.0 vs 43.2 tok/s, pf_dropped 0; MADV_RANDOM is neutral either way (0.002 vs 0.002 with prefetch, 0.97 vs 1.02 without) and stays on only because it is harmless and switchable`.

**Correctness, twice over.** Forced replay against llama.cpp with both the Q5_1 arm and
fused hc active: **186/192 agreeing, 0 hard mismatches, 0 non-finite**, the six
divergences all rank-2 near-ties at margins 0.009-0.30 logit. U7's pre-P3 number was
189/192 on the same instrument, and the difference is six near-ties changing sides, not a
math change. Free-run greedy text forks at token 2 on a 0.0817-logit near-tie and stays
forked, which is why free-run text comparison is not a grade on this checkpoint and
forced replay is. The shipped checkpoints were re-gated after the Metal source changes:
**35B-A3B ALL PASS** (6 graded; mm cos 0.999631, ppl Δnll 0.000791) and **27B ALL PASS**
(5 graded), both at fd46c7a. 2c8d3b3 and ac40526 move no shipped-checkpoint math — the
split path is qwen4exp-only and the prefetch is non-numeric — so they were not re-gated.

**Reviews.** Fable found no defects in d5757f1..fd46c7a; Opus found no correctness
defects and eight lower items, of which the parity-gate `baseEnv` leak was real and fixed
in 6eaf980 (the gate was inheriting `XWEN_HC_CLASSIC` and `XWEN_PLE_PROFILE` from the
caller's environment into the run env, which would have graded the wrong path silently)
and the decode single-threadgroup finding became 2c8d3b3. Second round on
6eaf980..ac40526: Fable-2 no defects, having gone and verified that candle's encoder
inserts the barrier the split kernel pair depends on; Opus-2 no correctness defects and
six polish items; Qwen no high findings. A polish commit (188ba73) took the rest of both
rounds — among them `fused()` now checking the weights it binds rather than only the
carrier, so the documented fall-back to the candle chain actually holds, and a bitwise
test that had silently drifted onto the split pair being pinned back to the single
kernel. Four low items are knowingly not fixed and are ledgered rather than dropped:
`n == 0` bails inside the fused hc path, the `-0.0` caveat on the bitwise tests (they
compare raw bit patterns, so a sign-of-zero difference reads as a mismatch),
`scripts/hf.ts`'s flash-next entry widening what `--model-size` the parity gate accepts
without any fixtures behind it, and that entry's dead `shards` key.

**The 15 GB of private memory is the design, not a leak.** P2 recorded that xwen dirties
~15 GB where llama-server dirties 751 MB on the same file, and it read like a bug. A
code-reading audit accounts for **~11.4 GB of it deliberately**: attention and GDN
projections dequantized to f16 planes for the prefill gemm (~5.35 GB raw, ~6.14 after
candle's pow2 buffer rounding), `token_embd` dequantized whole to f16 (1.27 → 2.15 GB,
plus a ~2.5 GB f32 transient), non-aliased Q8_0 copies for lm_head, the hc down/up
projections, the shared expert and the PLE k/v (1.65 GB together), the transposed router
weights at 8 MiB granularity across 48 layers (0.40), indexer raw-key planes sized at
`max_ctx` (0.81 at the 131072 default), and delta state (0.15). Every one of those is a
pattern the three shipped checkpoints already run; what is aliased is aliased correctly —
the 77.5 GB expert stacks, the 28.8 GB PLE table, the BF16 indexer projections. So the
"15 GB leak" reading is refuted. Three ways to shrink it are ledgered: alias the Q8_0
planes that only ever feed `QMatMul`, grow the indexer planes on demand, and gather
`token_embd` rows from the quantized tensor instead of materializing the whole table.

**End state, against llama.cpp on the same file in the same hour.** Prefill **795.7 vs
789**, decode **43.1 vs 41.4** before the PLE prefetch and 44.5-45.8 after — 1.01x and 1.04-1.10x, from 0.30x and 0.91x this morning.
Four rounds of interleaved medians at a 530-token prompt and 128 decoded tokens;
`powermode 0`, no high-power claim. Decode is **bimodal round over round** (42 vs 44
tok/s for the split path, 34 vs 36 for the unsplit one) with no explanation yet, which is
ledgered. The P0-pause notes' ~50 tok/s decode figure was always a scaling guess from the
35B-A3B; it is retired in favour of 43.1 measured.
