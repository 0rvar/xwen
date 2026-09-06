# 2026-08-08 — `mul_mv_ext` ships: the verify forward at span 2 goes 153 → 61 ms, drafted decode gains +11.6-13.2% on the 27B, and the kernel is 20-400x MORE accurate than the mm it replaces

Moved verbatim from [docs/log.md](../log.md) on 2026-09-06; the log keeps this entry's
opening paragraph and links here.


**Context.** The entry below decomposed the verify round's ~149 ms fixed cost and put
87.6% of it in the dense FFN's matmuls at small M — candle's `mul_mm` grid collapsing to
`ne01/64` threadgroups at M ≤ 32, ~73 GB/s against ~280 on the seq==1 mat-vec path. It
named the fix and priced it by byte arithmetic. This arc built and measured it.

**What shipped.** `src/ops/mv_ext.metal` + `mv_ext.rs`: llama.cpp's
`kernel_mul_mv_ext` multi-row quantized mat-vec, which dequantizes a weight block once
and reuses it across 2-5 output rows. Q4_K / Q6_K / Q8_0 × r1ptg 2-5. It is routed at
seq 2..=8 from a **single decision point, `QLinear::forward`**, which is what makes the
blast radius small and legible: that one site covers the 27B dense FFN, the 35B shared
expert, and `forward_all_logits`' lm_head. `XWEN_MV_EXT_CLASSIC` is the kill switch,
`XWEN_MV_EXT_MAX_SEQ` a probe knob, and a `mv_ext` provenance field at schema v8
(grandfather `classic`) records which path a dump ran.

**The gate fixtures never enter the window, so the oracle tests are the correctness
net.** Prefill chunks are 512 tokens and decode is 1, so no parity fixture produces a
2..8 forward at all — the tiers cannot see this kernel. The `mv_ext.rs` oracle tests
against `QTensor::dequantize` at production reduction lengths carry the correctness
claim instead, and `XWEN_MV_EXT_CLASSIC=1` is pinned on both sides of the strict tier
anyway (docs/parity.md "Provenance pins" explains why a kernel the gate cannot exercise
still gets a pin).

**Two-model review, four findings fixed.** Claude byte-diffed the vendored kernel
against ggml and Codex re-derived the dequant arithmetic from scratch; both cleared it.
The fixes were around the edges: `MAX_SEQ` not threaded into the plan, missing Q8_0
ragged coverage, a vacuous offset test, and a wrong geometry claim in the file header.

**The accuracy result runs the other way from `dense_mm`, and it is worth stating
because the reflex is now to expect a precision cost.** This kernel is **20-400x MORE
accurate than the `QMatMul` mm it replaces**: rel_l2 4e-7..8e-6 against ~1.8e-4. The
reason is structural, not tuning — it is f32 end to end where candle's tiled mm stages
weight tiles as half. The oracle tests assert the DIRECTION (`rel <= rel_classic` at
1.0x) rather than an absolute band, so the property is pinned without freezing a number
that a kernel change could legitimately move.

**Both parity gates ALL PASS post-change, reporting pre-change tier numbers** — expected,
since no gate fixture reaches the window, and the confirmation that the schema bump and
the routing change did not disturb anything the gate does see.

**Measured** (round 6; `lowpowermode 0`, both binaries at mtime 20:37, interleaved arms.
Verify bench: 27B, `n_past` 512, 2 reps per arm, means. End-to-end: the P9a protocol —
`spec-equivalence.ts`'s code and chat prompts verbatim, greedy, `-n 128`, `XWEN_BENCH=1`,
drafting default-on, 3 reps, medians).

Verify forward, default against `XWEN_MV_EXT_CLASSIC=1`:

| span | default | classic | |
|---|---|---|---|
| 2 | **61.45 ms** | 153.44 | 0.40x (−92.0) |
| 4 | **85.87** | 176.91 | 0.49x |
| 6 | **125.89** | 197.97 | 0.64x |
| 8 | **161.16** | 220.11 | 0.73x (−59.0) |
| 12-32 | — | — | arms match within 1.2-4.2% (ext inactive) |

One caveat that belongs next to the span-2 number: the default arm's per-rep spread was
large and one-directional — rep 1 faster by 15-30%, the known pattern, the biggest
instance of it seen so far. Bounding the win by the per-rep extremes gives **−87.5 to
−96.4 ms at span 2**, so the sign and the magnitude both survive the spread; the point
estimate is what is uncertain.

**End-to-end drafted decode, the headline:**

| | default | `XWEN_MV_EXT_CLASSIC=1` | |
|---|---|---|---|
| 27B code | **31.7 tok/s** | 28.4 | **+11.6%** |
| 27B chat | **30.9** | 27.3 | **+13.2%** |
| 35B code | **131.1** | 125.8 | +4.2% |
| 35B chat | 119.5 | 119.5 | +0.0% |

The 35B chat cell is a real dead heat, not a measurement failure: it is pause-dominated,
25-26 of 44 rounds, so most of its tokens never enter a verify forward. And the 35B's
small win is what the design predicts — only its shared expert and lm_head route through
`QLinear`, and its verify forward gained just 3.2-4.3 ms at spans 2-8 with nothing
beyond.

**The controller's economics changed, which re-opens the retune.** The 27B default arm
pauses far less than the classic arm — 16 vs 28 rounds on code, 14 vs 32 on chat — and
drafts more. Cheaper verifies make speculation worth attempting more often, exactly as
cheap verifies did after P9a. That is the **third** cost curve `p_min` 0.3 /
`pause_margin` 1.0 have now been wrong about, and the ledger item that was blocked on
this arc is unblocked by it.

**Refuted: extending the window past 8.** `XWEN_MV_EXT_MAX_SEQ=32` makes spans 16 / 24 /
32 **worse than classic by 1.11x / 1.42x / 1.69x**, with span 12 a wash at 0.98x. So
ggml's `ne11_mm_min` 8 envelope is the right ceiling and the inherited default was not
merely untested inheritance — it is now measured. Recorded under decisions.md "Refuted
perf directions"; the ledger item that flagged the upper edge as unmeasured is closed by
it.

**A cross-round caveat, so a future reader does not diff the wrong pair.** The classic
arm on this binary reads ~9-15% slower at mid-spans than round 3's binary did — fixed
intercept 172.9 vs 161.0 ms. Different binaries and machine-state variance; only the
WITHIN-round ratios above are trustworthy, never a cross-round absolute.

**What the fixed cost still holds.** Fitting the spans-2-8 arm leaves ~89 ms of
intercept. Two known non-coverages: the attention projections never touch `QLinear` on
the default path (they are f16 or q8_0 planes), and the single-row lm_head goes through
`forward` rather than `forward_all_logits`. Both are ledgered.
