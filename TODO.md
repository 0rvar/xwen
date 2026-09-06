# Deferred work ledger

Two things live here and they are not the same. **Front** is the backlog: at most ten
items, ranked, each with its expected gain against a ceiling in
[docs/perf-state.md](docs/perf-state.md), a user who is waiting for it, or, for an
instrument, what it would price. Promoting an
item means demoting one. The **area sections** below it are the ledger: every deferred
scope, grouped by the part of the system it touches, each with a one-word state and a
`From:` line naming the arc that deferred it. An item in an area is not planned until it
is promoted, and nobody owes it.

States. An open item's first line carries one tag: `[measured]` a number exists for the
gain or the cost; `[unpriced]` a candidate without a number; `[blocked]` waits on
something external; `[small]` a chore that needs no measurement. An item, or a lettered
sub-item, that ships moves verbatim to [docs/ledger-archive.md](docs/ledger-archive.md)
under the heading its `From:` line names. An item not worth planning is **retired**: a
dated line with the reason and a reopen condition, then the same move to the archive,
under a `Retired: <area>` heading. Retired means not planned, not forbidden: when its
reopen condition holds, or for a reason the retirement did not foresee, pick it up and
say why. The word **refuted** belongs to `docs/decisions/` only, with evidence; an item
here may cite a refutation recorded there, never make one.

Triage. At the end of an arc, and for any item whose latest date is more than 30 days
old: promote it, keep it with a fresh dated line saying why it is still worth it, or
retire it. Annotations stay at three lines plus a link. `bun scripts/docs-check.ts`
enforces the tag, the `From:` line, the front cap and the item length, and prints the
age histogram.

Intake. An item enters this ledger only when it carries a number — a measured gain or
cost against a ceiling in [docs/perf-state.md](docs/perf-state.md) — or a user who is
waiting for it; a chore enters as `[small]` only when the next arc is expected to do it.
Everything else belongs in the arc's record as "not taken now", with the reason and a
reopen condition. The rule is in [AGENTS.md](AGENTS.md).

## Front: planned next

[Position 2026-09-06, after the fused shared expert and the router gemv (log.md entries of
that date): **Flash-Next decode 52.9 tok/s at 596 tokens = 62-65% of the 81-86 bytes-only
ceiling**; prefill unchanged at 1140 @3851 = ~45% of the ~2500 tok/s that the same 12.07
GFLOP/token would take at the ~30 TFLOP/s the dense cooperative-tensor gemm reaches
(a derived compute ceiling, not a measured one). **35B-A3B decode 127.0** against a
bytes-only ceiling that is NOT in the docs: a back-of-envelope ~2.7 GB of active weights
per token (8 Q4_K experts + Q8_0 shexp/attention/GDN planes + the Q6_K lm_head) at
~550 GB/s is ~4.9 ms, ~200 tok/s, so ~62% as well; measuring that budget the way the
Flash-Next one was measured (bytes per token summed from the tensor table, the streaming
rate, then the residual) is an open item. The ~7 ms non-byte residual on both MoE
checkpoints is no longer "launches x 4 us": that price holds only for launches under
~2 MB on the dependent chain; the rest is occupancy losses (the router class, found by
reading after the probe priced its projection at zero; the audit in the Front's first
entry is what would find the rest) and the syncs plus the serial scan.]

1. **QSA prefill selection round-trips through the host per sparse layer per chunk, and it is the Flash-Next long-context tax** (Prefill performance, measured): the default checkpoint's prefill is 4x slower at 128k than at 8k and three times the 35B's growth, a maximal prefill takes 569 s, and 42 GB of peak memory rides the same path; pricing it is an hour and the fix is a device-side selection with no readback
2. **Drafting reads below plain on the 35B-A3B after the router gemv** (Drafting, measured): the default path of the 35B loses 8% at 1k tokens deepening to 37% at 16k, and 4% on a 256-token code prompt, in two independent measurements; the retune sweep either refits `p_min`/depth or flips the default off, and either way is worth more than any entry below
3. **Threadgroup-count-against-bytes audit of every decode dispatch** (Decode performance, unpriced): the instrument that would have found the router gemv (+10.3% on the 35B, +4.8% on Flash-Next); occupancy is the third decode cost class and nothing else names the next lever
4. **Hyper-connection carrier: 672 dispatches/token (35% of all launches), the largest population** (Decode performance, measured): (e) the 8-token decode tail after a ragged prefill read 47.9-52.1 tok/s fused against 55.4-57.6 split, all nine pairs, no valid recheck: a possible ~10% regression on the default path; (a) is a further -96 dispatches, +2%
5. **Expert gemm efficiency: 14-43% of wall, bracketed by two in-situ A/Bs** (Prefill performance, measured): prefill runs at ~45% of its ~2500 tok/s gemm-only ceiling and 38% of its wall is unpriced; pricing it is an hour and decides the second prefill lever
6. **Hyper-connection activation traffic: ~8% of wall estimated** (Prefill performance, measured): 0.39 s of 3.4 s prefill wall (11.3%) by the probe, and the whole-gate fusion is the kernel work the decode gate already shipped
7. **Above the 2048 indexer budget: +165 dispatches** (Decode performance, measured): 0.66 ms of dispatch cost plus 0.24 ms of QSA score tail, at every context past 2048 tokens; that 0.90 ms works out to ~+4% of a 18.9 ms token, a figure derived here and not stated in the item
8. **Reduce candle's CPU-side locking per dispatch** (Research candidates, measured): 1740 dispatches x 2.4 us is ~4.2 ms of a 19-21 ms token and it attacks the floor every fusion here buys against; the first step is a cheap CPU-vs-wall read
9. **The token-id readback sync (`stack.rs:511`): the host uploaded those ids one line earlier** (Decode performance, measured): +0.3 ms/token, +1.4%, no math change and no new kernel
10. **GDN: 252 dispatches (13%), the three fusion candidates in the P3 ledger** (Decode performance, measured): the three-projection merge is -72 dispatches, about +1.4%

## Decode performance

- [ ] [measured] **Hyper-connection carrier: 672 dispatches/token (35% of all launches), the largest
   population.** 7 per gate (norm, inject head — separate on the decode split arm —,
   down gemv, silu-quarter, up gemv, mix, write) × 96 gates. A fused norm+head+down
   is −192 (≈ −0.8 ms, +4%); folding the write into the mix or the next norm is −96
   (+2%); a single-kernel gate would approach −480 (≈ −1.9 ms, +10%). Bytes are 0.69
   GB (1.2 ms) and already near rate. UNPRICED in situ; the P3 ledger's own hc
   follow-ups are folded in as (f).
   **DONE, same day (dd50397): −384, not −192 — `kernel_hc_gate_down` and
   `kernel_hc_gate_up_mix` take 7 dispatches per gate to 3, 672 → 288 per token; decode
   47.0 → 51.2 median (+9%) against a +7.8% prediction, replay check PASS.**
   [Record](docs/records/fused-hc-gate.md). Still open here — (a) to (d) inline, then
   (e) and (f) below.
   (a) The write folded into the next gate's norm, −96 (the carrier must still be
   materialized for the next write);
   (b) `HC_GATE_ROWS_PER_TG` = 8 and kernel A's register shape are UNSWEPT — 4 and 16
   are one-constant A/Bs, and a silent spill would show only as tok/s; (c) the tail
   mixer's two launches; (d) provenance: schema v10 records `hc_gate` but no reader
   (`tests/parity.rs`, `parity-gate.ts`) enforces it, because no graded checkpoint has a
   hyper-connection — pin it when a qwen4exp tier exists (Codex review).
   (e) **NEW 2026-09-06, unexplained, from that same run and still open:** the 8-token
   decode tail after each ragged prefill read LOWER on every fused arm (47.9-52.1 tok/s)
   than on every split arm (55.4-57.6), all nine pairs, with no valid recheck yet. Open
   question: whether a fused-gate prefill leaves the first decode steps slower.
   [Record](docs/records/hc-gate-ragged-and-probe-decode.md).
   (f) An in-place `hc_write` FMA would drop a full-carrier write per layer-pair; and the
   two Q8_0 decode gemms still take `QMatMul`, which has no `mv_ext` plane at the `hc.rs`
   qlinear site (gguf.rs:1631-1648) — SCREENED 2026-09-05, no route changed.
  From: Flash-Next perf ledger, re-ranked from the measured budgets (2026-09-05, step 4).

- [ ] [measured] **MoE FFN: 576 dispatches/token (30%).** 12 per layer; the glue is already fused and
   the dual gate|up kernel is refuted (decisions.md `XWEN_MOE_DUAL`), so what is left
   is shape-level: the router is already one projection plus one fused kernel, so
   folding the projection in is −1/layer; shexp's four dispatches as one is −3/layer →
   −192 ≈ −0.8 ms, +3.6%. Item (15) of the archived P3 ledger.
   **The budget refinement it produced re-ranks every fusion candidate in this ledger**
   (decisions.md "Ceilings"): price a fusion candidate at (launches removed × ~4 µs) ONLY when its
   launches carry under ~2 MB AND sit on the dependent chain — the shared expert's cost
   was its BYTES, and the hc gate qualified where this one did not.
   Still open on this item: (a) `MOE_SHEXP_ROWS_PER_TG` (4) and
   the 128-thread shape are UNSWEPT one-constant A/Bs, and a bad shape would show only as
   tok/s; (b) the `moe_shexp` provenance label is written from the env predicate rather
   than from observed execution, so it records intent; the one-time "moe: shared expert
   fused|classic at N token(s)" host line is what a bench must read instead.
   (c) On (b), the fix worth
   making (Qwen review, 2026-09-06): record the label from OBSERVED execution, an
   AtomicBool on `MoeBlock` set when the fused tail actually dispatches (same for
   `router_mv` and `hc_gate`), read by logits-dump after its forwards; today a dump
   is stamped "fused" and satisfies the gate's pin even when `project()` returned
   None and the classic chain ran. Numerics are unaffected (classic is correct); the gap
   is measurement validity. Unstarted.
   (d) The remaining MoE decode lever is the glue kernels themselves (router kernel and
   epilogue: tiny bytes, on the chain), which the refined rule keeps in the launch-count
   class. The router gemv was the other candidate here — an occupancy item, not a
   launch-count one — and it landed the same day.
  From: Flash-Next perf ledger, re-ranked from the measured budgets (2026-09-05, step 4).

- [ ] [unpriced] **Threadgroup-count-against-bytes audit of every decode dispatch.**
  The router projection was invisible to both instruments
  this project has (the probe reads zero on a stage that overlaps itself, and the byte
  budget put it at 4% of a token's bytes) and it was worth +10%. Occupancy is a third
  class of decode cost beside bytes and launch gaps (decisions.md "Ceilings"), and the
  mechanical way to find the rest of it is to list each decode dispatch's threadgroup
  count next to the bytes it moves, then flag any plane over ~1 MB running under ~32
  threadgroups. Nothing is named or priced until that list exists.
  From: Flash-Next perf ledger, re-ranked from the measured budgets (2026-09-05, step 4).
  Promoted from the item «MoE FFN: 576 dispatches/token (30%)» on 2026-09-06; dated 2026-09-06 there.

- [ ] [measured] **The token-id readback sync (`stack.rs:511`): the host uploaded those ids one line
   earlier.** One drain per token for data that never left the CPU; pass the ids down
   instead. ≈ +0.3 ms, +1.4%, no math change; run the Flash-Next replay check anyway.
   NEW item, unstarted.
  From: Flash-Next perf ledger, re-ranked from the measured budgets (2026-09-05, step 4).

- [ ] [measured] **GDN: 252 dispatches (13%), the three fusion candidates in the P3 ledger.** At 4 µs
   each −36 is ≈ 0.15 ms (+0.7%), not the +1-2% the 8.41 µs arithmetic gave; the
   three-projection merge (−72) ≈ +1.4% and needs its A/B. Re-priced DOWN.
   The three, none started: conv+silu+state into the scan (−36); gnorm+zgate into
   `out_proj`'s prologue (−36); the three Q8_0 projections (`attn_qkv`, `attn_gate`,
   `ssm_out`) as one launch (−72), bandwidth-saturating, so A/B it (`XWEN_MOE_DUAL`).
  From: Flash-Next perf ledger, re-ranked from the measured budgets (2026-09-05, step 4).

- [ ] [measured] **Above the 2048 indexer budget: +165 dispatches** ≈ 0.66 ms at 4 µs, most of the
   remaining ~1 ms/step of the closed cliff; **the QSA scores tail** (5 elementwise
   dispatches × 12 layers feeding `qsa_select`, absorbable) is another 60 ≈ 0.24 ms.
   Folded in 2026-09-06, two candidates for the QSA tail that were their own items: a
   device partial-top-k for the selection (host `arg_sort` today, D16's stated P3 kernel),
   and a cooperative 256-thread threshold walk in `kernel_qsa_select` (serial on thread 0
   today; at most ~1 ms/step in the whole above-budget path, so bench the kernel's share
   first).
  From: Flash-Next perf ledger, re-ranked from the measured budgets (2026-09-05, step 4).

- [ ] [small] **PLE decode tail on device** (P3 (5)(b)): 0.13 ms of host work against one more
   readback — unpriced, small.
  From: Flash-Next perf ledger, re-ranked from the measured budgets (2026-09-05, step 4).

- [ ] [measured] **Top-k selection still crosses the bus at full vocabulary width.** The draw
  now costs 0.406 ms/token, of which 0.199 ms is the GPU→CPU copy of the 993 KB
  probability row and ~0.11 ms is the CPU streaming top-k. A Metal top-k (or a
  block-wise partial reduction that ships candidates, not the whole row) would
  leave only ~20 values to read back, and most of the 0.199 ms is command-buffer
  sync rather than copy, so the win is a fraction of it — measure before
  building. Pairs with P8; the sampler now has a bench that would show it
  (`cargo test --release sampler_decode_bench -- --ignored --nocapture`).
  Folded in 2026-09-06: the fast path's full-vocabulary device softmax collapses to the
  candidates along with the readback; it was its own item.
  From: Deferred from the sampler-tail pass (2026-07-28).

- [ ] [measured] **The `SampleControl` path still softmaxes on the CPU.** Adjusted draws
  (bans, bias, pull, force, grammar masks) read back raw logits, apply the
  controls, and pay the ~0.35 ms full-vocabulary `expf` pass the unadjusted path
  now avoids. It is the rare path — the default decode loop's control is a no-op
  whenever there is no blacklist, no grammar and no thinking floor — but forced
  reasoning (`--min-think`) and constrained decoding sit on it for entire
  generations. Fixable by applying the controls in probability space with a
  sparse renormalization (`p *= exp(delta/T)`, `p_pulled = p^(1-α)·p_max^α`,
  banned → 0, adjusting the total by the delta rather than resumming), which is
  exact for everything except `force` on a token whose probability underflowed —
  and `force` can short-circuit. Unmeasured against real control-heavy runs.
  From: Deferred from the sampler-tail pass (2026-07-28).

- [ ] [measured] **Flash-Next decode is bimodal round over round.**
  **NEW 2026-08-29, and nobody knows
  why.** Across four interleaved rounds at fixed settings the shipped arm reads
  44.0 / 42.1 / 44.1 / 42.3 tok/s and the `XWEN_HC_SPLIT_MAX_N=0` arm reads the
  same two-level pattern one step down (34 vs 36). It is not thermal drift (it
  alternates rather than decays), not the split path (both arms do it), and not
  contention as far as the runs could tell — one classic-arm outlier (34.1 in
  round 4) WAS concurrent unit tests, which is a different and identifiable
  signature. ~4% is enough to swamp a small A/B, so it matters for how the next
  perf change gets graded: until it is understood, quote medians of four or more
  interleaved rounds and never a two-round difference. First places to look: a
  two-state allocator or command-buffer reuse pattern, and per-round residency
  set churn.
  From: Qwen3.8-Flash-Next port (decided 2026-08-25, blocked on release + upstream).
  Promoted from the item «Port Qwen3.8-Flash-Next» on 2026-09-06; dated 2026-08-29 there.

- [ ] [small] **Re-profile the `ple` +3.2 ms above-budget term now that the QSA syncs are gone.**
  The earlier stack profile put `ple` +3.2 ms above budget and called it possible
  bleed from the adjacent syncs; those syncs are gone, so re-profile (rank only) to
  see whether the term went with them.
  From: Deferred from the prefill-chunk pass (2026-08-30).
  Promoted from the item «Decode on Flash-Next steps down ~11 ms/token the moment the context crosses the 2048-token QSA budget, then slopes gently» on 2026-09-06; dated 2026-08-30 there.

## Prefill performance

- [ ] [measured] **Expert gemm efficiency: 14-43% of wall, bracketed by two in-situ A/Bs** (amortized
   `mm_id_launch_shape_throughput` at t=2048: gate/up 3.9 ms + down 8.2 ms per layer
   per chunk = 1.44 s at 3851 tokens = 43%; the classic-tile A/B implies ~35%, the
   full-grid A/B ~14%; log "Ceiling diagnosis"). The down plane (Q5_1, about half the
   per-layer time) is the half to attack: a dequant that is not re-done per token tile,
   or an f16-cached expert tile — both unpriced; "dequant-bound" is the 2026-08-30 code
   reading, not a measurement. This item was ranked BELOW the glue on 2026-08-30 on the
   strength of the inflated profiler. Neither the stage profiler (2.2x inflation) nor an
   isolated bench (transfer 0.32-0.82) can price a prefill stage; the duplicate-dispatch
   probe that settled the bracket is built and run, and its design notes are archived.
   The item that remains is the gemm efficiency work itself; the down plane is 44% of the
   expert time, not half.
   **What is next (2026-09-06): price the unpriced 38%.** Nothing names it, and it is the
   size of the expert gemms. Add probe stages for the GDN projections (`attn_qkv`,
   `attn_gate`, `ssm_out`), the full-attention projections and sdpa, the QSA indexer,
   PLE, and lm_head (same `ops::dup` idiom, pure launchers only, never a cache advance),
   run the 3851-token session once, and re-rank this list on the result. Cheap (an hour),
   and it decides whether the second prefill lever is in the gemms or somewhere no one has
   looked. Unstarted.
   Folded in 2026-09-06: the one named gemm-side candidate is BaseRT-M5's shape (arXiv
   2607.00501), gate/up + SiLU kept on the cooperative-tensor accumulators in `mm_id`; the
   gate/up pair was priced at 28-32% of prefill wall by the probe on 2026-09-05.
  From: Flash-Next perf ledger, re-ranked from the measured budgets (2026-09-05, step 4).

- [ ] [measured] **Hyper-connection activation traffic: ~8% of wall estimated** — MEASURED 0.39 s
   (11.3%) by the probe, of which the two bottleneck gemms 0.14 and the glue kernels plus
   the write 0.25 s (7.3%) (the 84 MB carrier
   read/written ~8-10 times per gate). Whole-gate fusion at prefill is the same kernel
   work as «Hyper-connection carrier: 672 dispatches/token (35% of all launches), the
   largest population» in "Decode performance", paid twice.
  From: Flash-Next perf ledger, re-ranked from the measured budgets (2026-09-05, step 4).

- [ ] [measured] **GDN prefill (`mixer_delta`, ranked 2nd at 20% by the profiler, unpriced).** No
   amortized bench exists for the chunked scan at 2048 rows; build one before
   touching it. PRICED 2026-09-05 by the probe: the GDN kernels are 0.23 s (6.7%), the
   scan alone 0.16 s (4.6%); the projections are outside the probe and make up the rest
   of the profiler's 20%. **MoE glue (router, activation, epilogue) is 0.40 s (11.5%)
   by the same probe and ranks above this item; the shared expert is 0.3% and drops off.**
  From: Flash-Next perf ledger, re-ranked from the measured budgets (2026-09-05, step 4).

- [ ] [unpriced] **Prefill runs candle sdpa with a materialized mask, not the vendored
  flash kernel.** `flash.metal` is compiled at `BD == 128` and Qwen 3.6 is head
  dim 256, so the in-kernel mask path is unreachable and `model.rs` materializes
  the `[1, n_head, seq, k_seq]` f16 mask on every prefill again — the allocation
  laguna's flash path was written to avoid (1.5-2.3 GB at 4k on laguna's head
  count; ~1/3 of that here with 16 heads). Either instantiate flash at BD 256 or
  accept the mask. Pairs with P8.
  Folded in 2026-09-06: a fused sigmoid-gate kernel at the attention gate site (~2-3
  dispatches) rides along with the BD 256 flash instantiation; neither is sized against a
  measured bounty ([record](docs/records/27b-prefill-residual.md)).
  From: Deferred from the P2-P4 model-core retarget (2026-07-28).

- [ ] [measured] **+350 to +560 µs/token of prefill cost lives OUTSIDE every measured stage, and it
  grows with prompt length.** Per-token wall goes 3023 → 3599 µs between the 880- and
  3851-token fixtures (+576, i.e. the 330.7 → 277.9 tok/s drop), and the four profiled
  stages account for only **+13 µs/token** of that: the dense FFN and DeltaNet non-scan
  per-token rates are flat, and the DeltaNet scan gets *faster* per token with length,
  so it is anti-correlated. Mask + sdpa quadratic growth is real but only ~+69 µs/token
  combined.
  **State it as a range, not +576.** The FFN row of that budget is derived from an
  isolated rate ~7-8% pessimistic against a real forward (the @880 budget sums to 106.8%
  of wall, a physically impossible negative residual), so part of the swing is an
  artifact of that bias. +350 to +560 µs/token is the defensible claim.
  Ruled out by direct measurement at T=512/880/3851: dense FFN, DeltaNet non-scan,
  DeltaNet scan, and the mask+sdpa quadratic terms. NOT ruled out and unattributable
  without new instrumentation: the per-layer RMSNorms (2 × 64 layers = 128 eager
  dispatches per chunk over `[512, 5120]` f32), residual adds, KV-cache appends and
  page-touching as the 537 MB cache fills, embedding + lm_head, Metal buffer-pool
  behaviour across 8 chunks × 64 layers, and command-buffer gaps. Next step is per-layer
  timing inside `model.rs` `run_stack` — in situ, not a synthetic bench. Now the largest
  single unknown in 27B prefill: with the FFN gemm fixed this is a much bigger share of
  what remains, and it is most of why the 4k result (445 tok/s) fell short of the
  profile's 496 upper-bound counterfactual while the 925 result met it.
  ANNOTATION 2026-08-08: **the diagnosis ran; the residual is real (+410-438 µs/token
  reproduced), it is NOT inside any stage (per-stage syncs find only +102.8), and both
  cross-chunk accumulation and command-buffer batching are refuted as its mechanism.**
  [Record](docs/records/27b-prefill-residual.md), decisions.md "Refuted perf directions".
  - **Next step: an instrument that can see INSIDE a chunk.** A barrier/fence counter
    needs either a candle patch or a Metal capture; `XWEN_STACK_PROFILE` cannot separate
    the two surviving hypotheses (intra-chunk barrier storms from buffer-pointer
    recycling, fence-wait pileup) because syncing is what makes the cost disappear.
  - **Second thread, tracked here rather than split out:** the **ffn stage's +42.2
    µs/token of in-stage growth is unexplained**. The dense SwiGLU is length-INDEPENDENT
    per token by construction, so a stage that grows with prompt length under
    serialization is an anomaly on its own terms; the signature is allocator pressure,
    not arithmetic.
  Folded in 2026-09-06: the instrument this item asks for is a candle patch for
  per-resource barrier scoping and dependency-filtered fence waits (`auto_barrier` emits a
  whole-scope barrier, `encoder.rs:104-149`); it was a research candidate of its own.
  From: Deferred from the dense-FFN prefill gemm pass (2026-07-29, P8c).

- [ ] [unpriced] **Q5_1 expert kernels: vendored mv_id decode arm and per-stack use_mm.**
  **2026-08-29 (P3, from D18).** UD-Q4_K_XL carries
  `Q5_1` on `ffn_down_exps` for 43 of 48 layers (the 640-column block-size
  fallback; see docs/qwen4exp-port.md "The 640-column rule"). It RUNS today
  with no code change — decode reaches candle's baked
  `kernel_mul_mv_id_q5_1_f32`, and prefill falls back correctly — so this is
  perf, not correctness, and it is not P2's problem. Three items:
  (a) add a Q5_1 arm to the vendored `mv_id` fast path (`mv_vendored_supported`
  is Q4_K/Q5_K/Q6_K/Q8_0 today, so every Q5_1 decode takes the slower baked
  kernel); (b) add Q5_1 to the vendored two-pass `mm_id` — or give those
  layers a second encode path to candle's baked `kernel_mul_mm_id_q5_1_f32` —
  so the 43 affected layers regain grouped prefill; (c) decide whether
  `FusedExperts::use_mm` should be per-stack instead of all-or-nothing: today
  one unsupported down plane drops that layer's gate and up to per-token
  matvec too, which is the bulk of the cost. Measure before and after.
  **Quantified 2026-08-29 by U7**: prefill is 3.5x behind llama.cpp on this
  file (203.5 vs 713.4 tok/s at 530 tokens), and this is suspect number one.
  Grouped with the rest of the deferred perf work from the P3 ledger, now spread across
  the area sections of this file.
  **(b) SHIPPED 2026-08-29 in 8112733 (D20): a Q5_1 arm in the vendored two-pass
  `mm_id` took prefill 239 → 443 tok/s (1.85x) at a 530-token prompt, the `ffn`
  stage 2887 → 1031 µs/token, and the gap to llama.cpp 3.30x → 1.78x.**
  [Record](docs/records/flash-next-p3-kernels.md).
  **(a) and (c) STAY OPEN**, and (a) did not follow from (b): `mm` is now the
  prefill path for those 43 layers, but **decode still takes candle's baked
  `kernel_mul_mv_id_q5_1_f32`** and did not move at all in the A/B (37.7
  before and after), so a Q5_1 arm in the vendored `mv_id` fast path is still
  unmeasured upside. (c) per-stack `use_mm` matters less now that the mm path
  covers the plane that was forcing the fallback, but the all-or-nothing rule
  is unchanged. Note for whoever benchmarks this: `XWEN_NO_MM_ID=1` is NOT the
  before-arm — it forces mv on all three planes and reads 225 tok/s, below the
  real baseline.
  From: Qwen3.8-Flash-Next port (decided 2026-08-25, blocked on release + upstream).
  Promoted from the item «Port Qwen3.8-Flash-Next» on 2026-09-06; dated 2026-08-29 there.

- [ ] [measured] **QSA prefill selection round-trips through the host per sparse layer per chunk, and it is the Flash-Next long-context tax.** Flash-Next prefill falls 925 → 584 → 403 → 231 tok/s from 8k to
  128k (2026-09-06, [record](docs/records/long-context-envelope.md)), i.e. 1.08 → 4.33
  ms/token, three times the growth the 35B-A3B shows over the same range (0.43 → 1.50),
  while Flash-Next decode stays flat (47 → 42). The only work Flash-Next does that the 35B
  does not is the sparse selection: per QSA layer per 2048-token chunk, `select_with`
  (`qwen4exp/indexer.rs`) reads the score plane back to the host, fills an `n x n_kv`
  f32 mask there, uploads it and converts to f16 — roughly a gigabyte each way per layer
  per chunk at 128k, twelve layers, sixty-four chunks — and unlike the causal mask it is
  on the critical path, because the GPU waits for the readback before the mask exists.
  That is why the causal-mask A/B priced at 0% of wall (candle builds that one ahead of
  the GPU) and why that result does not transfer here. The same path is the 42 GB of
  Flash-Next's 59 GB peak at 128k that the device causal mask did not move (no two
  `Tensor::from_vec` allocations ask the pool for the same size, so none is recycled).
  Unpriced as a share of wall: the 128k profile was skipped. Step one is an hour's timer
  around the readback, fill and upload at 128k; if it confirms, build the selection and
  the -inf plane with scattered zeros on the device, same values, behind a kill switch,
  no readback. The decode-side cousins (device partial top-k, cooperative threshold
  walk) are folded into «Above the 2048 indexer budget: +165 dispatches» in "Decode
  performance"; the retired per-chunk readback item's reopen condition is this number.
  From: Deferred from the long-context envelope arc (2026-09-06).

## Drafting

- [ ] [measured] **The DFlash drafter's per-token cache sync, its unre-derived draft-ctx
   horizon, and the deferred ring-buffer cache.** Adaptation itself finished 2026-07-29 —
   both sidecars load, draft and verify correctly at 85-95% acceptance: 27B +4.8 to
   +6.8% on a code prompt and +1.5 to +7.4% on a chat prompt, 35B-A3B -11.5% and -12.7%;
   `draft_p_min` retuned 0.5 → 0.3, `pause_margin` stays 1.0. [Record](docs/records/dflash-drafting.md), decisions.md "Speculative
   decoding". Those figures predate P9a's K-snapshot fused verify, which made drafting a
   both-checkpoint win and flipped it to opt-out the same day (AGENTS.md "Drafting").
   Original scope: repoint drafter arch check
   (arch `dflash`, decoder arch qwen35/qwen35moe), tap indices from `target_layers`
   metadata, mask_token_id, sliding-window pattern; verify the fc.weight geometry
   (5×hidden / 8×hidden concat). Needs P4's recurrent-state rollback. Re-tune
   auto-pause and draft-ctx horizon for this drafter's cost curve.
   - **(b) The drafter's per-token cache sync costs ~1.2 ms and is what sinks the
     35B.** An arm that can never draft (`--draft-p-min 1.1`, 119/127 rounds paused)
     still decodes at 92.6 tok/s against 105.1 plain — the whole 35B loss, incurred
     before any drafting. Per committed token: `encode` (8-tap concat through a
     [2048, 16384] `fc`) plus six layers of `wk`/`wv` + QK-norm + rope + two
     `slice_set`s, ~14 small Metal dispatches. Dispatch-bound, not FLOP-bound — the
     same disease `kernel_moe_router`/`kernel_moe_epilogue` cured for the MoE block.
     Two independent levers: fuse the inject (one dispatch for all layers' K/V, or at
     least batch the projections), and/or teach the pause controller to DETACH rather
     than pause once it has enough evidence, since a paused drafter still pays this.
     Fixing either would cut what a paused drafter still pays; the flip to opt-out has
     since happened on its own, in P9a.
   (e) retired 2026-09-06 (docs/ledger-archive.md "Retired: Drafting"); (c) shipped
   2026-09-06 with the long-context envelope arc and moved verbatim to the archive under
   this item's own heading — `DEFAULT_DRAFT_CTX` stays 8192, and the re-derivation found
   drafting reading BELOW plain at every length rather than a better horizon
   ([record](docs/records/long-context-envelope.md)). (b) is what stays open here.
  From: Priority order (decided 2026-07-28; P1-P9 shipped by 2026-07-29).

- [ ] [measured] **The auto-pause controller costs 3-6% on a checkpoint it never pauses, and the
  cost is its instrumentation rather than its decisions.** Stage C's margin sweep on the
  3.8-27B made `margin 0` the winner: 35.9 tok/s mean-of-medians against 34.8 at the
  shipped 1.0 (code 37.7 vs 35.7, chat 34.1 vs 33.9; reps tight enough to be
  non-overlapping). Pausing does not explain it — BOTH arms recorded ZERO paused rounds.
  The mechanism is `PauseController`'s forced-plain cadence: with `margin > 0` it spends a
  round decoding plain every `FORCE_PLAIN_EVERY` (32, and every `WARMUP_FORCE_PLAIN_EVERY`
  = 4 until the plain warm-up is met) purely to keep `ema_plain_ms` from going stale, and
  a forced-plain round commits one token where a drafting round commits about four. In a
  128-token run of ~40 rounds that is roughly three rounds' worth of speedup spent on
  measurement, which is the size of the observed gap.
  NOT fixed by setting the margin to 0: that is one shared value at three sites, only the
  3.8's stage 2 was run, decisions.md records the controller earning its keep on the 3.6
  pair, and the depth-8 probe arm (34-80 rounds paused, drafting reduced to +2%) shows the
  safety net still catching real cases. Two real fixes, either of which keeps the
  controller and stops paying a whole round for its baseline: derive `ema_plain_ms` from
  the verify forward, which already decodes a known number of positions and could yield a
  per-token cost without a dedicated round; or make the cadence adaptive — back the forced
  plain round off geometrically while the speculative margin is wide, the way the paused
  state already backs off its probes. Wants a 3.6 stage-2 re-run alongside, so the shared
  constant moves on evidence from every checkpoint it governs rather than one.
  From: Deferred from the MTP drafting arc (2026-08-15, stages B and C).

- [ ] [unpriced] **The MTP head cannot follow a rewind, so a serve conversation that rewinds stops
  speculating until it prefills from zero.** The head's row at position `p` is built from
  the target's post-final-norm hidden at `p - 1`, and the head keeps exactly one such
  hidden — the carry, for the position it currently ends at. `sync_drafter_to(pos)` on a
  rewind therefore has no hidden to build row `pos` from, so `MtpDrafter::truncate` drops
  the head to zero rather than resume on another position's hidden. The DFlash drafter
  keeps its rows across the same rewind, because each of ITS rows is a function of that
  position's taps alone. Cost: every intra-slot rewind (engine.rs `sync_drafter_to` call
  sites, batch.rs's two) costs speculation for the rest of that conversation — unmeasured,
  and it does not arise at all on the one-shot CLI path. Three ways out, cheapest first:
  keep the last N hiddens and accept rewinds that land inside that window; keep the whole
  hidden history on device (`draft_ctx x hidden x 4` = 168 MB at the default 8192 on the
  27B, which is 40x the head's own 4 KiB/token KV and is why it was not done now); or
  recover the hidden by re-running the target's last committed token, which costs one
  decode step per rewind but no memory. Decide with a measurement of how often serve
  actually rewinds a drafting conversation.
  Folded in 2026-09-06: the disk-tier face of the same root, a stored MTP image resumes
  only at the exact position it ends at (`MtpDrafter::import_cache` refuses a shorter
  `pos`), and takes whichever fix this item gets.
  From: Deferred from the MTP drafting arc (2026-08-15, stages B and C).

- [ ] [unpriced] **The MTP head builds its own prefill mask, doubling the prefill mask cost.**
  `MtpDrafter::step` calls `AttnBlock::prefill_mask`, which materializes a
  `[1, n_head, seq, pos+seq]` f16 tensor — 24 x 512 x 4096 x 2 = 100 MB per chunk at a 4k
  prompt. The trunk builds exactly one such mask per chunk and hoists it across all
  sixteen of its full-attention layers (model.rs, `full_mask`), so the head's one extra
  layer adds a SECOND full-size mask build and upload per chunk rather than a
  sixteenth of one — which is the shape of a cost that is invisible at 1k and grows with
  the prompt, matching the measured regression's shape. The mask is reusable as-is: at
  sync time the head's committed length equals the trunk's cache length and its head
  count equals the trunk's, so the parameters `(n_head, seq, pos)` are identical. Plumb
  the trunk's hoisted mask into the sync instead of rebuilding it. Unquantified — the
  regression was measured end to end, not attributed — so measure before and after
  rather than assuming this is all of it.
  From: Deferred from the MTP drafting arc (2026-08-15, stages B and C).

- [ ] [blocked] **Flash-Next still ships no drafter, and `supports_drafting()` stays false.** The
  blocker is D6, not the sidecar: the MTP head in this checkpoint's config has no
  transformers implementation and separate `fc_embedding`/`fc_hidden` projections rather
  than 3.8's concat `eh_proj`, so its forward semantics are unconfirmed and were not
  guessed at. The verify machinery downstream of a proposal is kind-agnostic and would
  take a third kind cheaply — what is missing is the speculative tap contract on the
  qwen4exp stack (spec taps are not defined for this graph) plus a confirmed head. Until
  then `--draft` is refused rather than ignored and `DraftMode` resolution logs "no
  drafter available" for this checkpoint alone.
  Folded in 2026-09-06: the pointer item for Flash-Next's two closed gates; `auto_fetch()`
  false is a decision (AGENTS.md "Checkpoint location"), and this item is the drafting
  gate.
  From: Deferred from the qwen4exp cache-image arc (2026-08-30, P4).

- [ ] [measured] **Drafting reads below plain on the 35B-A3B after the router gemv.** The
  2026-09-06 presence-penalty A/B (code prompt, 256 tokens, 3 interleaved reps, pinned
  build) read drafted 121.1 tok/s against plain 126.5 at penalty 0, and 119.6 against
  126.9 at the shipped 1.5, acceptance 63.0% / 59.4%. The drafted 35B figures in
  docs/perf-state.md were fitted against the pre-fold plain level and never re-swept;
  plain gained +10.3% on 2026-09-06 and the drafted arm on this prompt no longer clears
  it. **2026-09-06, second workload: the loss deepens with context.** The long-context
  arc measured the same direction on long-document prose with a forced thinking decode —
  drafted against plain, medians of 2: 111.9 vs 121.9 at 1046 tokens (80.6% acceptance),
  85.3 vs 116.3 at 4117 (70.9%), 73.1 vs 104.2 at 8201 (58.5%), 62.8 vs 99.1 at 16409
  (57.4%). So it is -8% even at short context and inside the acceptance band the fits
  were made at, and -37% by 16k. Two independent workloads now agree, which removes the
  "one prompt" caveat; what is added is that a retune looking only at short prompts will
  not see half of it, and that `DEFAULT_DRAFT_CTX` = 8192 is currently LIMITING this loss
  rather than costing anything (docs/records/long-context-envelope.md). First step is
  still the standing retune (`bun scripts/retune-draft.ts`) on the 35B, now at more than
  one context length, which either refits `p_min`/depth or shows drafting should default
  off on this checkpoint.
  **2026-09-06: the default flipped off** on this checkpoint alone
  (`Model::draft_default_on`); `p_min` and depth are untouched and the retune still
  decides whether it comes back (docs/log.md, docs/decisions/speculative-decoding.md).
  [Record](docs/records/presence-penalty.md).
  From: Deferred from the presence-penalty arc (2026-09-06).

## Serve, batch and CLI

- [ ] [measured] **serve at 32k decodes 4-7% under `generate` and the cause is unconfirmed — run a
  `--ctx 8192` / `--ctx 65536` serve arm to test the state-allocation hypothesis.** At
  2k and 7.6k the two paths are at parity; only 32k separates them, and the difference
  is under the 10% bar so the profiler was never run. The one visible asymmetry: serve
  on its `--ctx 262144` default logs `state 2.0GB` where `generate` at `max_ctx 8192`
  logs 0.2 GB, so serve walks a 10x larger recurrent-state allocation every step even
  though the live context is the same length. If that is the cause, serving the same 32k
  prompt under `--ctx 65536` and `--ctx 8192` should close the gap monotonically; if the
  gap holds at every ctx, it is the queue or the per-step serve overhead instead and the
  hypothesis is refuted cheaply. Cost is three arms of one prompt, so run this before
  anything more elaborate. Note the 32k rows were taken while the anchor had drifted
  −5.8% thermally, so re-anchor between arms.
  From: Deferred from the first Flash-Next serve benchmark (2026-08-30).

- [ ] [small] **An end-to-end smoke was never run: confirm a real `generate` lands a real line.**
  The shipping session verified the whole arc with unit tests (26 in `src/metrics.rs`,
  serve at 442 against 428 before the arc, 11 in the binary, 82 in batch) and a
  hand-written fixture file, because another model process held the GPU and the standing
  rule is one large model process at a time. So every piece is tested and the
  composition — a real run, the real default path, the real file — is not. Run one
  `xwen generate`, check the line appended to
  `$HOME/.local/state/xwen/metrics.jsonl`, and check `xwen stats` reports it. Five
  minutes with the GPU free; do it before trusting any number the history reports.
  From: Deferred from the metrics arc (2026-09-05).

- [ ] [small] **Whether the header's session id equals the transcript id `claude --resume` shows
  is unconfirmed.** `x-claude-code-session-id` is documented as a per-session identifier
  and is what `--by session` keys on, which works regardless. What is not established is
  that the uuid in the header is the same uuid that names the transcript on disk — if it
  is, a row of the table maps to a resumable conversation and that is worth documenting;
  if it is not, nobody should assume it. Settle it by capturing one request's header
  next to the session id that `claude --resume` lists for the same conversation. Until
  then no doc claims the two are the same (README says so explicitly).
  From: Deferred from the metrics arc (2026-09-05).

## Cache images, memory and context

- [ ] [measured] **The router gemv holds `ffn_gate_inp` twice, ~251 MB resident f32 on Flash-Next.**
  **2026-09-06.** `MoeBlock` keeps the
  plane in both orientations because the two arms want different ones and both are on
  the DEFAULT path: prefill runs above `ROUTER_MV_MAX_N` and takes `x.matmul(router_t)`
  over the `[hidden, n_expert]` transpose, decode runs at or below it and takes the gemv
  over `[n_expert, hidden]` as loaded. Cost: 5.24 MB per layer, ~251 MB across
  Flash-Next's 48 (512 x 2560 x 4 B) and ~8 MB across the 35B-A3B's 40 (256 x 2048 x
  4 B) — resident f32, never quantized. Two ways to reclaim it, neither started: give the
  gemv a prefill form (a tiled f32 gemm over the same orientation, so `router_t` can go),
  or find a transpose-free candle route over the `[n_expert, hidden]` plane for the
  fallback arm (so `router` alone serves both). Not urgent at 251 MB of 111 GB, but it is
  a real number and it grows with `n_expert`.
  From: Flash-Next perf ledger, re-ranked from the measured budgets (2026-09-05, step 4).
  Promoted from the item «MoE FFN: 576 dispatches/token (30%)» on 2026-09-06; dated 2026-09-06 there.

- [ ] [measured] **Three shrinks for Flash-Next's ~15 GB of private memory.**
  xwen dirties ~15 GB of private memory where llama-server dirties 751 MB on the same
  file (64 GB vs 76 GB clean mapped), i.e. ~15 GB of weights are materialized rather than aliased
  from the mapping. Worth understanding under the one-large-process rule.
  **AUDITED 2026-08-29 — this is design, not a bug (D24): a code-reading audit accounts
  for ~11.4 GB of the 15 as patterns the three shipped checkpoints already run, and
  everything that should be aliased is. The "15 GB leak" reading is refuted, and what
  remains is three shrinks rather than an investigation**
  ([record](docs/records/flash-next-p3-kernels.md), decisions.md "Refuted: the ~15 GB of
  private memory is a leak"): **(i)** alias the Q8_0 planes that only
  ever feed `QMatMul` (hc, lm_head, shexp) through the q8 alias path — ~1.6 GB;
  **(ii)** grow the indexer planes on demand instead of allocating at `max_ctx`
  (the `IndexerCache` growth-path item in this section); **(iii)** gather `token_embd`
  rows from the quantized tensor instead of materializing the whole table in f16.
  From: Qwen3.8-Flash-Next port (decided 2026-08-25, blocked on release + upstream).
  Promoted from the item «Port Qwen3.8-Flash-Next» on 2026-09-06; dated 2026-08-29 there.

- [ ] [measured] **2026-08-30 — Flash-Next cache images have a ~113 MiB floor per snapshot.**
  Seen in the first real serve smoke (`cache slot 0 paged out at 20 tokens:
  113 MiB in 1229 ms`): `snapshot_bytes()` is 118 MB regardless of length,
  and nearly all of it is the DeltaNet delta state (36 layers × [128,128,48]
  f32 = 113 MB; the PLE image is 0.4 MB) — the same class of cost the 27B
  carries (48 × 3 MB). Every slot swap moves it and every anchor snapshot
  holds it in host RAM, so `--cache-slots` sizing on this checkpoint should
  count ~120 MB per retained snapshot, and `SNAPSHOT_MIN_GAIN = 1024`
  (engine.rs:57) is doing real work: a system block under 1024 tokens gets no
  anchor snapshot, which reads like a broken prefix cache and is not (a
  485-token system prompt gave C `cache_read 0`; 1157 tokens gave 1157).
  Possible reductions: f16 delta state on the image only (2x), or snapshot
  the delta state lazily (skip it for anchors that will only ever be
  rewound-to, since a rewind re-prefills from the anchor anyway — needs the
  rewind path to tolerate a missing Linear arm).
  From: Deferred from the qwen4exp cache-image arc (2026-08-30, P4).

- [ ] [measured] **`IndexerCache` still allocates at `max_ctx` up front and has no growth path,
  and page-in is now a second reason to care.** The trunk's KV grows lazily — the cache
  starts at 8k positions and extends as a conversation lengthens — while every QSA
  layer's raw-key plane is allocated whole at load, 4 MB per layer at 8k and ~1.6 GB
  across the 12 QSA layers at the checkpoint's 262144 ctx, paid whether or not the
  conversation ever gets there. That was already ledgered as a memory item under the archived P3
  ledger and as one of the three shrinks in "Refuted: the ~15 GB of private
  memory is a leak"; what the cache images add is a correctness-shaped consequence
  rather than a wasteful one: **a conversation paged back in longer than the live
  allocation is REFUSED rather than grown**, because `import_full_kv_into` sets
  `IndexerCache::len` to the imported row count and cannot set it past the plane it was
  given. On the trunk's planes the same import grows. Fixing the allocation fixes both
  halves at once, and the growth rule should be the trunk's (extend on demand, drop back
  on idle unload) rather than a second policy.
  From: Deferred from the qwen4exp cache-image arc (2026-08-30, P4).

- [ ] [small] **The disk tier's stored-image path for qwen4exp is pinned by unit tests only; no
  real serve smoke has been run against the 111 GB file.** A qwen4exp segment
  round-trips with its indexer planes and PLE state, and a v3 container is rejected, but
  both run over constructed payloads. There is no serve-engine harness that runs a real
  model — `page_out_live`/`page_in` are private free functions over a private
  `EngineState` and the engine tests use stand-in payloads — so the equivalence is
  pinned one level down, at `export_full_kv` + `take_cache_snapshot().to_host()` →
  `check_importable` → `import_full_kv` → `restore_cache_snapshot`, which is exactly the
  sequence those two functions perform. What is untested is the real file through a real
  server: load, converse, page out to disk, evict, page back in, continue. Cheap to run
  once (one conversation, `idle_unload` short, the disk tier on) and the thing most
  likely to find a shape mismatch the unit fixtures do not reach.
  From: Deferred from the qwen4exp cache-image arc (2026-08-30, P4).

## Parity, provenance and tooling

- [ ] [small] **The 27B's interactive generate/chat smoke run was never done, and its
    numbers carry a ±10% spread.** Bring-up itself finished 2026-07-28 via P7: the parity
    gate ran the 27B end to end, first forward correct, all gated tiers passing (strict is
    near-vacuous on the dense model — see P7c).
    - ANNOTATED 2026-07-28 (P8a): **the 27B perf gap is now filled** — decode 19.0 tok/s
      at 596 tokens and 17.9 at 1929, prefill 290.4 and 209.3, ~4.7x slower per decoded
      token than the 35B-A3B. Its budget is dominated by the dense SwiGLU, not dispatch
      count, so the next 27B lever is the FFN, not more glue fusion.
      [Record](docs/records/fused-deltanet-kernels.md). Still open: the interactive
      smoke run.
    - CAVEAT on those 27B numbers: its per-rep spread is materially wider than the
      35B's (the 596-token fused decode walked 21.7/19.0/17.9 across three reps as
      the machine heated, against a 35B classic arm that repeated to within 0.8%).
      Treat the 27B figures as ±10% and re-measure off an idle machine before using
      them as a baseline for anything. See decisions.md "Measurement discipline".
  From: Priority order (decided 2026-07-28; P1-P9 shipped by 2026-07-29).

- [ ] [small] **Snapshot-replay-vs-scratch has no Track-B parity case.** The equivalence was
  exercised at ship time by hand (`XWEN_BATCH_NO_CACHE=1` as the A/B arm, same request
  both ways) and the finding is recorded — values identical except one genuine near-tie,
  scores differing in the third to fourth decimal, both explained by the `mv_id`/`mm_id`
  partition split. That is a measurement, not a gate. The decode tier is the right home
  for it (greedy replay with the near-tie rule, which is exactly the rule this divergence
  class needs); it wants a fixture batch with a long shared prefix and enough items to
  make one near-tie likely. Until then a regression in the restore path would be caught
  only by someone running the demo.
  From: Deferred from the batch + scored-classification arc (2026-08-09).

- [ ] [small] **No parity-gate or retune arm for Qwen3.8-27B.** `scripts/parity-gate.ts` accepts
  `--model-size 3.8-27b` and would run it (nothing about the gate is 3.6-specific), but
  it has never been run against 3.8 and the floors in docs/parity.md were fitted on the
  3.6 files — so a first run's numbers are unvalidated, not a gate. `retune-draft.ts`
  deliberately excludes 3.8 (`draftingSizes()`, and `SHIPPED_P_MIN` has no arm for it):
  there is no drafted arm to sweep without a drafter, and it dies early saying so rather
  than sweeping a plain-vs-plain comparison. Both open up together if the MTP drafter
  items in "Drafting" are taken. Note also that `SHIPPED_P_MIN` is typed `Record<ModelSize,
  number>` and no longer covers every `ModelSize` — harmless (nothing typechecks the
  scripts, and every read is behind the drafter check) but it is a real type gap to fix
  when that file is next touched.
  ANNOTATION 2026-08-15 (Stage C, C3): **the retune half is DONE — a 3.8 arm swept, both
  shipped tables carrying the fitted 0.7 / 4, the `Record<ModelSize, number>` type gap
  closed. The parity-gate half is still open and is all that keeps this item alive:**
  `scripts/parity-gate.ts` has never been run against 3.8 and the docs/parity.md floors
  are still the ones fitted on the 3.6 files. [Record](docs/records/mtp-stage-c.md).
  Folded in 2026-09-06: the 3.8 has no
  `tests/fixtures/reference-ppl-Qwen3.8-27B-Q4_K_M.json` either, so a full-tier run bails
  on "ppl reference fixture missing"; `--regen-ppl-ref` against the 3.8 hub file is part
  of the same first run.
  From: Deferred from the Qwen3.8-27B + API-naming arc (2026-08-14).

- [ ] [unpriced] **`spec-equivalence.ts`'s sampled mode grades itself with a heuristic that
  mis-grades, and exits nonzero as though it were a gate.** Two separate problems, both
  found by running the 3.8 arm and its 27B control (Stage C, C3). First, the "a fork at
  line 1 under a fixed seed points at the sampler stream, not a near tie" rule is wrong as
  stated: 3.8 seed 7 forks at line 1 on both fixtures, and a sampler-stream bug is ruled
  out for that build by other seeds of the SAME build coming back byte-identical, which a
  structural off-by-one in draw count could not produce. Position is a weak proxy for
  cause; the strong one is seed-dependence, and the script never varies the seed. Second,
  the script exits nonzero on a sampled divergence, which reads as a regression gate —
  but the shipped 27B fails it on the chat fixture at every seed tried, so it has never
  been a gate on any checkpoint and treating it as one trains the reflex to ignore it.
  Fix: sweep two or three seeds per comparison and grade on "diverged at EVERY seed"
  (stream) versus "diverged at some" (near tie), and either make sampled advisory or hold
  it to a criterion it can actually pass. Cost of not doing it: the next person to run
  this reads a red result on a healthy build, or misses a real stream bug behind a
  heuristic that cried wolf.
  From: Deferred from the MTP drafting arc (2026-08-15, stages B and C).

- [ ] [unpriced] **The parity harness cannot run on qwen4exp.**
  **2026-08-29 (P4, from U7).**
  All four tiers of `scripts/parity-gate.ts` die on this checkpoint, because
  every tier's reference side is `--moe-impl reference` and
  `ReferenceExperts::forward` panics at `src/moe.rs:198` with "index out of
  bounds: the len is 512 but the index is 1073971200". `1073971200` is
  `0x40038000`, the f32 bit pattern of 2.0547 — so an f32 buffer of routing
  data is reaching a `to_vec1::<u32>()` read as expert ids, on the 512-expert
  / top-10 geometry. It reproduces identically through the fused router kernel
  AND the candle `route_from_logits` chain, so it is downstream of the router
  branch, not in either kernel. The FUSED runner is unaffected (U7's whole
  measurement set ran on it). **One fix unblocks all four tiers**; nothing else
  in the harness objected to this file. Alongside it: (a)
  `observed_delta_path()` in `src/bin/logits-dump.rs` hard-bails when no gated
  DeltaNet layer ran — latent, did not bite here, but it gates any layer-kind
  change; (b) no reference-ppl fixture exists for Flash-Next (same gap as the
  3.8-27B above); (c) split GGUFs work fine, but the gate's temp dir basename
  carries the `-00001-of-00004` shard suffix — cosmetic; (d) the gate's floors
  are global constants calibrated on the ggml-org Q4_K_M mix and this file is
  unsloth UD-Q4_K_XL, so they need re-deriving for this checkpoint even once
  the panic is fixed; (e) `tests/fixtures/ppl-corpus.txt` looks contaminated
  for this checkpoint — 0.37 nats is PPL 1.45 on WikiText-2 test where the 3.6
  pair scores 1.69 nats, and llama.cpp independently agrees, so it is the model
  and not a bug, but it makes the frozen corpus a weak discriminator here. Pick
  a fresh held-out corpus for flash-next and re-derive `PPL_NLL_DELTA_MAX`
  against it. (Part of what "experimental" means for this checkpoint — see the
  archived P4 ledger for the full set.)
  Folded in 2026-09-06: `parity-gate.ts --model-size flash-next` is spellable through
  `scripts/hf.ts` and fails deep rather than at argument validation, and that entry's
  `shards` key is dead; gate the accepted set on fixture existence when the harness is
  made to run here.
  From: Qwen3.8-Flash-Next port (decided 2026-08-25, blocked on release + upstream).
  Promoted from the item «Port Qwen3.8-Flash-Next» on 2026-09-06; dated 2026-08-29 there.

- [ ] [small] **Three upstream reports owed, none filed.**
  **2026-08-29.** **(1) candle
  Metal `index_select` is silently wrong on strided sources** — no error, just
  wrong rows; found in U3 and worked around by gathering per head. This is a
  correctness bug in a dependency and is the most valuable of the three to
  file. **(2) llama.cpp's QSA top-k width diverges from HF**: the PR fills
  `top_k + ratio - 1` tokens unconditionally where HF selects whole top-k
  blocks plus the raw tail, so they differ whenever `visible mod ratio ≠
  ratio−1` above budget. We follow HF (fixture-pinned); worth reporting, with
  the caveat below. **(3) the converter lost its `image_token_id` config.json
  fallback**, so a self-converted text-only file will likely carry no
  `ple.image_token_id` and silently fall back to EOS — harmless for us, looks
  like a regression. WATCH ITEM alongside these: the unmerged `origin/tmp-q4`
  branch (`f91123d2d`) reworks QSA to pack visible tokens into whole blocks in
  token order with the budget in whole blocks and pooled keys roped at the
  first member's real position — i.e. it converges on the HF semantics our
  fixtures already pin, which would retire report (2). If it merges: re-vendor,
  re-read every QSA entry in the port doc, and re-check the divergence list
  before filing anything.
  From: Qwen3.8-Flash-Next port (decided 2026-08-25, blocked on release + upstream).
  Promoted from the item «Port Qwen3.8-Flash-Next» on 2026-09-06; dated 2026-08-29 there.

- [ ] [small] **No decode-step top-2 margin tooling.** Twice on 2026-08-30 (the QSA-cache
  greedy fork, then the FFN-glue fork bisect) a temp-0 fork needed the top-2 logit
  margin at the forked step; `logits-dump` dumps prompt logits only, so both
  bisects ended at "near-tie by determinism argument" instead of a number. A
  greedy-margins mode (per-step top-2 ids + logit gap) would settle these in one
  run.
  From: Deferred from the prefill-chunk pass (2026-08-30).

- [ ] [small] **A long-prompt mm tier.** The strict/mm/decode tiers grade a 58-token prompt, so
  the 64-wide `_t64` mm_id kernel (selected when `t*top_k/n_expert ≥ 24`) is exercised
  only by the ppl tier's 4218-token corpus at the 2048 chunk. Its identity with the
  32-wide kernel is test-pinned but not structural (different `matmul2d`
  instantiations), so a toolchain change could separate them with only ppl watching. Add
  an mm-tier fixture long enough to run at least one full 2048-token chunk (2026-08-30).
  From: Deferred from the prefill-chunk pass (2026-08-30).

## Tokenizer, chat and sampling

## Research candidates

- [ ] [unpriced] **llama.cpp's 64-node lookahead reorder, to widen the concurrent dispatch sets**
  (`ggml-metal-common.cpp:300-370`). candle already encodes concurrently with
  dependency-derived barriers (decisions.md, "Refuted AS ALREADY PRESENT"), so the
  remaining question is how much independent work there is to overlap — and on the
  decode path the graph is mostly a serial chain, one layer feeding the next. **Measure
  the available parallelism before implementing anything**: how many of a decode step's
  ~77 dispatches have no dependency on their predecessor. If that number is small, the
  reorder is dead on the decode side regardless of what it does for llama.cpp, and only
  prefill (where the expert gemms are genuinely independent) is left. Estimate unknown.
  From: Deferred from the technique survey (2026-08-30).

- [ ] [measured] **Reduce candle's CPU-side locking per dispatch** (candle patch): an `EntryState`
  mutex with 4-6 lock acquisitions and `HashSet` inserts on every bind. This is the one
  survey item that **attacks the fitted 8.41 µs dispatch floor directly** rather than
  working around it, and the floor is what every dispatch-count fusion on the ledger is
  ultimately buying against. Estimate unknown — nobody has profiled the CPU side of a
  dispatch here, and it should be profiled before it is patched. [2026-09-05: the floor
  measured on byte-free dispatches is 2.4-2.7 µs (the 8.41 is the gemv's own ramp), and
  1740 × 2.4 µs ≈ 4.2 ms is close to the 3.7 ms of process CPU per decode token, so the
  CPU side of a dispatch may BE that floor. Profile it first, as this item says; the
  cheap test is process CPU against wall on `bandwidth_sweep`'s tiny arms.]
  From: Deferred from the technique survey (2026-08-30).
