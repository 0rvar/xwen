# Deferred work ledger

Two things live here and they are not the same. **Front** is the backlog: at most ten
items, ranked, each with its expected gain against a ceiling in
[docs/perf-state.md](docs/perf-state.md) or a user who is waiting for it. Promoting an
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
say why. The word **refuted** belongs to `docs/decisions/` only, with evidence; nothing
in this file or the archive uses it.

Triage. At the end of an arc, and for any item whose latest date is more than 30 days
old: promote it, keep it with a fresh dated line saying why it is still worth it, or
retire it. Annotations stay at three lines plus a link. `bun scripts/docs-check.ts`
enforces the tag, the `From:` line, the front cap and the item length, and prints the
age histogram.

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
~2 MB on the dependent chain; the rest is occupancy losses (the router class, found by the
audit in decode item 2(e)) and the syncs plus the serial scan.]

1. **Threadgroup-count-against-bytes audit of every decode dispatch** (Decode performance, unpriced): the instrument that would have found the router gemv (+10.3% on the 35B, +4.8% on Flash-Next); occupancy is the third decode cost class and nothing else names the next lever
2. **Hyper-connection carrier: 672 dispatches/token (35% of all launches), the largest population** (Decode performance, measured): (f) the 8-token decode tail after a ragged prefill read 47.9-52.1 tok/s fused against 55.4-57.6 split, all nine pairs, no valid recheck: a possible ~10% regression on the default path; (a) is a further -96 dispatches, +2%
3. **Expert gemm efficiency: 14-43% of wall, bracketed by two in-situ A/Bs** (Prefill performance, measured): prefill runs at ~45% of its ~2500 tok/s gemm-only ceiling and 38% of its wall is unpriced; pricing it is an hour and decides the second prefill lever
4. **Hyper-connection activation traffic: ~8% of wall estimated** (Prefill performance, measured): 0.39 s of 3.4 s prefill wall (11.3%) by the probe, and the whole-gate fusion is the kernel work the decode gate already shipped
5. **Above the 2048 indexer budget: +165 dispatches** (Decode performance, measured): 0.66 ms plus 0.24 ms of QSA score tail, ~+4% at every context past 2048 tokens, which is every real coding context
6. **Reduce candle's CPU-side locking per dispatch** (Research candidates, measured): 1740 dispatches x 2.4 us is ~4.2 ms of a 19-21 ms token and it attacks the floor every fusion here buys against; the first step is a cheap CPU-vs-wall read
7. **The token-id readback sync (`stack.rs:511`): the host uploaded those ids one line earlier** (Decode performance, measured): +0.3 ms/token, +1.4%, no math change and no new kernel
8. **GDN: 252 dispatches (13%), the three fusion candidates in the P3 ledger** (Decode performance, measured): the three-projection merge is -72 dispatches, about +1.4%
9. **serve at 32k decodes 4-7% under `generate` and the cause is unconfirmed** (Serve, batch and CLI, measured): 4-7% of serve decode on the default checkpoint, settled or refuted by three arms of one prompt
10. **GDN prefill (`mixer_delta`, ranked 2nd at 20% by the profiler, unpriced)** (Prefill performance, measured): the MoE glue this item ranks under is 0.40 s (11.5%) of prefill wall; the GDN kernels themselves 0.23 s (6.7%)

## Decode performance

- [ ] [measured] **Hyper-connection carrier: 672 dispatches/token (35% of all launches), the largest
   population.** 7 per gate (norm, inject head — separate on the decode split arm —,
   down gemv, silu-quarter, up gemv, mix, write) × 96 gates. A fused norm+head+down
   is −192 (≈ −0.8 ms, +4%); folding the write into the mix or the next norm is −96
   (+2%); a single-kernel gate would approach −480 (≈ −1.9 ms, +10%). Bytes are 0.69
   GB (1.2 ms) and already near rate. UNPRICED in situ; the P3 ledger's own hc
   follow-ups are folded in as (g).
   **DONE, same day (dd50397): −384, not −192 — `kernel_hc_gate_down` and
   `kernel_hc_gate_up_mix` take 7 dispatches per gate to 3, 672 → 288 per token; decode
   47.0 → 51.2 median (+9%) against a +7.8% prediction, replay check PASS.**
   [Record](docs/records/fused-hc-gate.md). Still open here — (a) to (d), and (f) below.
   (a) The write folded into the next gate's norm, −96 (the carrier must still be
   materialized for the next write);
   (b) `HC_GATE_ROWS_PER_TG` = 8 and kernel A's register shape are UNSWEPT — 4 and 16
   are one-constant A/Bs, and a silent spill would show only as tok/s; (c) the tail
   mixer's two launches; (d) provenance: schema v10 records `hc_gate` but no reader
   (`tests/parity.rs`, `parity-gate.ts`) enforces it, because no graded checkpoint has a
   hyper-connection — pin it when a qwen4exp tier exists (Codex review).
   (f) **NEW 2026-09-06, unexplained, from that same run and still open:** the 8-token
   decode tail after each ragged prefill read LOWER on every fused arm (47.9-52.1 tok/s)
   than on every split arm (55.4-57.6), all nine pairs, with no valid recheck yet. Open
   question: whether a fused-gate prefill leaves the first decode steps slower.
   [Record](docs/records/hc-gate-ragged-and-probe-decode.md).
   (g) An in-place `hc_write` FMA would drop a full-carrier write per layer-pair; and the
   two Q8_0 decode gemms still take `QMatMul`, which has no `mv_ext` plane at the `hc.rs`
   qlinear site (gguf.rs:1631-1648) — SCREENED 2026-09-05, no route changed.
  From: Flash-Next perf ledger, re-ranked from the measured budgets (2026-09-05, step 4).

- [ ] [unpriced] **MoE FFN: 576 dispatches/token (30%).** 12 per layer; the glue is already fused and
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
   (d) On (b), the fix worth
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
  **NEXT INSTRUMENT, an unpriced candidate: a threadgroup-count-against-bytes audit
  of every decode dispatch.** The router projection was invisible to both instruments
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
  From: Flash-Next perf ledger, re-ranked from the measured budgets (2026-09-05, step 4).

- [ ] [small] **PLE decode tail on device** (P3 (5)(b)): 0.13 ms of host work against one more
   readback — unpriced, small.
  From: Flash-Next perf ledger, re-ranked from the measured budgets (2026-09-05, step 4).

- [ ] [measured] **MoE block glue fusion — SHIPPED 2026-07-29.** An MoE layer went from 24
    dispatches per decoded token to 14 (960 → 560 across the 40 layers) and 35B-A3B
    decode from 92.6 to 102.8 tok/s (+11.0%), on three fusions all bit-identical to the
    candle chains they replace, behind `XWEN_MOE_GLUE_CLASSIC=1`, with both parity gates
    passing at pre-change numbers.
    [Record](docs/records/fused-moe-glue.md), decisions.md "Kernel policy".
    - **The residual add was NOT fused in, on purpose.** The briefed design folded
      `model.rs`'s `x + ffn_out` into the epilogue for a twelfth dispatch. That would
      delete the `ffn_out` tap, which docs/parity.md lists with published per-layer
      floors on both checkpoints — or force the gate onto the classic path, where it
      would never exercise the fused epilogue at all. Worth ~40 dispatches per token
      (one per layer) if someone later teaches the epilogue to write both `ffn_out` and
      `l_out` when taps are on; not worth a provenance hole.
    - **Prefill is untouched and was not attacked.** Above `MM_ID_MIN_SEQ` the epilogue
      declines (the f16-tile projection carries an L2 rescale it has no term for) and
      only the router kernel and the shared-expert activation fuse; measured +0.6% at
      925 tokens and +0.2% at 4k, i.e. nothing. Fusing the prefill combine would mean
      an epilogue variant carrying the rescale — cheap to write, but prefill is
      compute-bound at ~2100-2500 tok/s and there is no evidence it would show.
    - **`mul_mv_id_dual`'s wrapper trusts its ids buffer.** It validates rank, dtype,
      contiguity and dims but not that each id is < n_expert (values live on the GPU;
      checking means a readback) nor that ids/gate/up share x's device. Fine for the
      router-produced ids of its only caller, loose for a `pub` API. Harden if the
      dual path ever ships on by default.
  From: Priority order (decided 2026-07-28; P1-P9 shipped by 2026-07-29).

- [ ] [measured] **Top-k selection still crosses the bus at full vocabulary width.** The draw
  now costs 0.406 ms/token, of which 0.199 ms is the GPU→CPU copy of the 993 KB
  probability row and ~0.11 ms is the CPU streaming top-k. A Metal top-k (or a
  block-wise partial reduction that ships candidates, not the whole row) would
  leave only ~20 values to read back, and most of the 0.199 ms is command-buffer
  sync rather than copy, so the win is a fraction of it — measure before
  building. Pairs with P8; the sampler now has a bench that would show it
  (`cargo test --release sampler_decode_bench -- --ignored --nocapture`).
  From: Deferred from the sampler-tail pass (2026-07-28).

- [ ] [measured] **The fast path still softmaxes the full vocabulary it no longer needs.**
  Split out of the top-p convention item when that resolved (2026-07-29). The cut
  now renormalizes over the k survivors, which is arithmetically a k-wide softmax,
  so nothing downstream of the selection depends on the full-vocabulary
  denominator any more — the ~0.1 ms it was worth is unclaimed only because the
  selection itself still runs CPU-side over the whole row. Pairs with «Top-k selection
  still crosses the bus at full vocabulary width» in this section: land that and the
  device softmax collapses to the candidates along with the readback.
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

- [ ] [measured] **Option: floor the `Proj::DenseF16Q8` window at t >= 3, leaving span 2 on the
  gemv.** Added 2026-08-08 from that pass's attention-projection coverage A/B. At t=2 the
  ext kernel saves only ONE gemv weight pass, and its geometry is fixed (nsg=2, nxpsg=8)
  rather than tuned for a two-row batch, so there is a plausible mechanism for it not to pay there. The
  evidence does NOT currently show a loss: span 2 measured +1.4 ms (+2.3%), but the same
  interleave reads pairwise medians of +1.6 to +2.3% in the same direction at spans
  12/16/24 where the kernel cannot run, because the coverage arm was second in every
  pair. So this is worth at
  most ~1.4 ms at span 2 only and it might be worth nothing. **Do not ship it off the
  existing data** — it needs its own A/B, and one designed to separate the effect from
  the arm-ordering bias (alternate which arm goes first between reps, or run the two
  window floors as the two arms so both are the same binary generation). Nothing else in
  the window is affected: the `QLinear` sites keep 2..=8, which is where the −92 ms
  span-2 win lives. Cheap to try (`mv_ext_window`'s caller at the Proj site is one
  condition) and cheap to leave alone.
  From: Deferred from the small-batch mat-vec pass (2026-08-08).

- [ ] [measured] **The lm_head roughly doubles at span 48 (7.0 → 13.1 ms) and nobody knows why.**
  Added 2026-08-08 from the verify-round sweeps; first recorded inside the annotation on
  «Every verify arm goes superlinear at span 48» in "Drafting", and promoted here because
  it is a distinct phenomenon from that item's trail-memory finding and belongs with the
  matmul work. It is **arming-independent** — the doubling appears in both the armed and
  the unarmed profiled runs — which rules out the K-snapshot trail that explains the
  span-48 superlinearity itself. Outside the production regime (drafter `block_size` 16
  caps real verify spans near 17) and therefore not urgent, but it is a single dispatch
  on a well-understood shape doubling across a span step, which usually means a
  threshold or a fallback nobody has named. Note the lm_head IS one of the three sites
  the `mul_mv_ext` window covers (via `forward_all_logits`), so anyone touching that
  routing should check this at the same time. Recorded so a future contradiction has a
  trail.
  ANNOTATION 2026-08-08 (later the same day): **still open, and the attention-projection
  coverage A/B is a clean negative on it** — span 48 unchanged between the arms, medians
  526.39 (HEAD) against 521.97 (coverage), inside the run-to-run spread.
  [Record](docs/records/small-batch-window-projections.md).
  From: Deferred from the small-batch mat-vec pass (2026-08-08).

- [ ] [unpriced] **Q5_1 expert kernels: vendored mv_id decode arm and per-stack use_mm.**
  **2026-08-29 — P3: Q5_1 expert kernels (from D18).** UD-Q4_K_XL carries
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

- [ ] [unpriced] **QSA top-k runs on the host via arg_sort.** A device partial-top-k kernel
  is the intended replacement: D16 says selection is computed with candle ops in P2, and
  says explicitly that the top-k kernel is P3.
  From: Qwen3.8-Flash-Next port (decided 2026-08-25, blocked on release + upstream).
  Promoted from the item «Port Qwen3.8-Flash-Next» on 2026-09-06; dated 2026-08-29 there.

- [ ] [unpriced] **Parallelize the 16 page faults inside one PLE gather.** Follow-up on the
  shipped PLE prefetcher, if its A/B shows the overlap window is too short: the 16 faults
  inside a single gather are taken serially on one thread — parallelize them across the
  window rather than deepening the lookahead.
  From: Qwen3.8-Flash-Next port (decided 2026-08-25, blocked on release + upstream).
  Promoted from the item «Port Qwen3.8-Flash-Next» on 2026-09-06; dated 2026-08-29 there.

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

- [ ] [small] **The PLE prefetcher spawns one thread per PleTable.** That is harmless on
  every published qwen4exp file, because upstream hard-asserts `n_ple == 1`, but the code
  does not depend on that assert: a checkpoint with several PLE layers would get a
  prefetch thread each, all faulting the same table. If a multi-PLE file ever appears,
  share one prefetcher across tables rather than one per layer.
  From: Qwen3.8-Flash-Next port (decided 2026-08-25, blocked on release + upstream).
  Promoted from the item «Port Qwen3.8-Flash-Next» on 2026-09-06; dated 2026-08-29 there.

- [ ] [measured] **The decode scan still double-buffers its state, and making it in-place is not
  the kernel's call.** `run_delta_scan_decode` allocates a fresh
  `[v_heads, 128, 128]` f32 buffer per layer per token and leaves the incoming state
  untouched, which is what lets a rollback trail hold every state it recorded. Writing
  the same buffer would move the SAME bytes (3.1 MB read + 3.1 MB written either way —
  the floor arm of `delta_scan_decode_timing` prices exactly that and the scan is
  already within 1.4x of it), so the only prizes are the pool allocation and whatever
  the write-allocate costs, and the price is an aliasing promise no op-level function
  can make on its own: the armed verify trail holds device-side clones of every state
  (`kv_cache.rs` `advance_linear`), and any future holder — a prefix-cache image that
  stops materializing, a serve snapshot that keeps a handle — would be corrupted
  silently rather than loudly. If it is ever worth doing, the shape is a caller-supplied
  "this state is unaliased" flag plumbed from `LinearAttnBlock::forward_fused` (which
  knows `cache.linear_trail_armed()`), not an inference inside `dispatch`.
  From: Deferred from the DeltaNet decode-scan pass (2026-08-30).

- [ ] [unpriced] **`kernel_qsa_select`'s threshold walk is serial on thread 0.**
  The walk covers 256 bins × 4 passes on that one thread; a cooperative
  256-thread walk (per-bin prefix via scan) is the obvious next shape. Measure the
  kernel's share of the step with an amortized bench before acting — at 44-45 vs 46.7
  there is at most ~1 ms/step left in the whole above-budget path.
  From: Deferred from the prefill-chunk pass (2026-08-30).
  Promoted from the item «Decode on Flash-Next steps down ~11 ms/token the moment the context crosses the 2048-token QSA budget, then slopes gently» on 2026-09-06; dated 2026-08-30 there.

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
   strength of the inflated profiler. **FIRST, build the instrument that settles the
   bracket: an in-situ duplicate-dispatch probe** — a presence switch that encodes a
   stage's kernels twice (the expert gemms first, then the hc gates, the GDN chunked
   scan) so the wall delta IS that stage's in-situ time; no math change when unset,
   Flash-Next replay check anyway. Duplicate only the kernel dispatches, never the
   surrounding block — re-running the router, the gathers or an allocation would put
   their cost into the delta too (Qwen review, 2026-09-05). Neither the stage profiler (2.2x inflation) nor an
   isolated bench (transfer 0.32-0.82) can price a prefill stage.
   The item that remains is the gemm efficiency work itself; the down plane is 44% of the
   expert time, not half.
   **Open sub-item (2026-09-06): price the unpriced 38%.** Nothing names it, and it is the
   size of the expert gemms. Add probe stages for the GDN projections (`attn_qkv`,
   `attn_gate`, `ssm_out`), the full-attention projections and sdpa, the QSA indexer,
   PLE, and lm_head (same `ops::dup` idiom, pure launchers only, never a cache advance),
   run the 3851-token session once, and re-rank this list on the result. Cheap (an hour),
   and it decides whether the second prefill lever is in the gemms or somewhere no one has
   looked. Unstarted.
  From: Flash-Next perf ledger, re-ranked from the measured budgets (2026-09-05, step 4).

- [ ] [measured] **Hyper-connection activation traffic: ~8% of wall estimated** — MEASURED 0.39 s
   (11.3%) by the probe, of which the two bottleneck gemms 0.14 and the glue kernels plus
   the write 0.25 s (7.3%) (the 84 MB carrier
   read/written ~8-10 times per gate). Whole-gate fusion at prefill is the same kernel
   work as decode item 1, paid twice.
  From: Flash-Next perf ledger, re-ranked from the measured budgets (2026-09-05, step 4).

- [ ] [measured] **GDN prefill (`mixer_delta`, ranked 2nd at 20% by the profiler, unpriced).** No
   amortized bench exists for the chunked scan at 2048 rows; build one before
   touching it. PRICED 2026-09-05 by the probe: the GDN kernels are 0.23 s (6.7%), the
   scan alone 0.16 s (4.6%); the projections are outside the probe and make up the rest
   of the profiler's 20%. **MoE glue (router, activation, epilogue) is 0.40 s (11.5%)
   by the same probe and ranks above this item; the shared expert is 0.3% and drops off.**
  From: Flash-Next perf ledger, re-ranked from the measured budgets (2026-09-05, step 4).

- [ ] [measured] **DeltaNet Metal kernels — (a) DONE 2026-07-28, (b) still open.** Original
   scope for (b): chunked prefill scan, chunk 64, llama.cpp's chunked form as the spec
   (cumsum → tri decay mask → solve_tri → per-chunk state update) — needs tri-solve which
   candle lacks; vendored kernel. Kill-switch XWEN_DELTA_CHUNK_CLASSIC falling back to the
   P3 reference. Gate: bitwise-or-bounded vs reference per parity.md tiering.
   - **(b) The chunked scan (chunk 64, tri-solve) remains open**, and its case is now
     weaker than it looked: the single-dispatch sequential scan already put prefill at
     ~2000 tok/s, so the chunked form is competing against that rather than against
     the 300 tok/s reference. Its real remaining argument is the rollback trail (see
     the P2-P4 deferred item): a chunked scan that can replay a prefix cheaply would
     let the per-token trail be dropped entirely. Measure before building.
     ANNOTATION 2026-07-29: measured, and the picture splits by model — the weak-case
     reading holds on the 35B, while on the 27B the sequential scan looked like the cause
     of a 1.8-2.1x prefill loss to llama.cpp, making the chunked form's bounty ~2x there.
     [Log](docs/log.md#2026-07-29--first-llamacpp-head-to-head-xwen-wins-decode-on-both-models-loses-27b-prefill-2x-to-the-sequential-deltanet-scan).
     ANNOTATION 2026-07-29 (later the same day): **the ~2x bounty is WITHDRAWN — that
     reading was wrong.** The fused scan is 3% of 27B prefill, so making it FREE moves
     prefill ~297 → ~307 tok/s against llama.cpp's 486; the gap is in the dense
     projections and needs its own item.
     [Log](docs/log.md#2026-07-29--the-deltanet-scan-is-3-of-27b-prefill-llamacpps-decomposition-measured-slower-and-the-premise-behind-p8b-refuted).
     ANNOTATION 2026-07-29 (P8c): **that item was opened, root-caused and CLOSED the same
     day — the gap was the dense FFN's gemm (66-85% of 27B prefill wall through candle's
     `kernel_mul_mm_q4_K_f32`), and `src/ops/dense_mm.metal` fixed it.**
     [Record](docs/records/dense-ffn-prefill-gemm.md), decisions.md "The dense-FFN
     prefill gemm dequantizes in-kernel".
     Separately, llama.cpp
     on Metal never runs its chunked form at all (its fused `ggml_gated_delta_net` op
     pre-empts the chunked graph, delta-net-base.cpp:437-446), and its sequential
     Metal decomposition was transplanted here and measured SLOWER at both geometries
     and both lengths. So **(b) stays ledgered but is refuted as a prefill lever**;
     its remaining live rationale is chunk-boundary replay for the rollback trail, and
     even that is superseded by the K-snapshot plan under P9. Do not reopen (b) for
     prefill without a per-stage profile that contradicts the 3% figure — re-run
     `delta_scan_timing` (src/ops/delta.rs, `#[ignore]`d) to price it. See log.md
     2026-07-29 "The DeltaNet scan is 3% of 27B prefill" and decisions.md "The
     DeltaNet scan decomposition".
   - `XWEN_DELTA_CHUNK_CLASSIC` was never created — there is no chunked path to
     switch off yet. It belongs with (b).
  From: Priority order (decided 2026-07-28; P1-P9 shipped by 2026-07-29).

- [ ] [unpriced] **Prefill runs candle sdpa with a materialized mask, not the vendored
  flash kernel.** `flash.metal` is compiled at `BD == 128` and Qwen 3.6 is head
  dim 256, so the in-kernel mask path is unreachable and `model.rs` materializes
  the `[1, n_head, seq, k_seq]` f16 mask on every prefill again — the allocation
  laguna's flash path was written to avoid (1.5-2.3 GB at 4k on laguna's head
  count; ~1/3 of that here with 16 heads). Either instantiate flash at BD 256 or
  accept the mask. Pairs with P8.
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
  From: Deferred from the dense-FFN prefill gemm pass (2026-07-29, P8c).

- [ ] [unpriced] **A fused sigmoid-gate kernel at the attention gate site, and the
  head-dim-256 flash instantiation.** The gate kernel is worth ~2-3 dispatches; the flash
  instantiation is what would remove the mask (the item «Prefill runs candle sdpa with a
  materialized mask, not the vendored flash kernel» in this section). Neither is sized
  against a measured bounty. These two are all that survived this item's original premise
  — ~42 ms/layer of unfused attention glue — which was DOWNGRADED 2026-08-08 as inverted;
  that narrative is in the archive. [Record](docs/records/27b-prefill-residual.md).
  From: Deferred from the dense-FFN prefill gemm pass (2026-07-29, P8c).

- [ ] [small] **`DENSE_MM_MIN_SEQ` is unfitted for the shexp/hc shapes.** The 32-token
  floor was fitted on the 27B FFN (k 5120/17408); the shexp (k 2560/640, and
  2048/512 on the 35B) and hc (k 10240/320) routes inherit it unmeasured
  (2026-08-30). Only matters for short chunks and ragged tails; sweep
  `XWEN_DENSE_MM_MIN_SEQ` over those shapes if they ever show up hot.
  From: Deferred from the prefill-chunk pass (2026-08-30).

- [ ] [unpriced] **QSA prefill still reads the scores back once per chunk per layer to build the mask on the host.**
  Prefill is `n > 1`, and it assembles the `[n_q, n_kv]` mask on the host rather than on
  the device. A device-side mask build (top-k per query row,
  then a fill kernel) would remove it; low value — one sync per chunk, not per token.
  From: Deferred from the prefill-chunk pass (2026-08-30).
  Promoted from the item «Decode on Flash-Next steps down ~11 ms/token the moment the context crosses the 2048-token QSA budget, then slopes gently» on 2026-09-06; dated 2026-08-30 there.

- [ ] [measured] **Fuse gate/up + SiLU onto the cooperative-tensor accumulators in `mm_id`**
  (BaseRT-M5's shape, arXiv 2607.00501). Prefill only; the `_t` family already
  dequantizes the expert tile once per token tile, and this would keep both projections'
  results in the accumulators through the activation instead of round-tripping them.
  **Bounded by an already-measured fact**: the expert gemms are a MINORITY of the
  prefill `ffn` stage — the 2026-08-30 mm_id pass moved them +17-23% in isolation and
  the stage fell 3-5%. So a few percent of `ffn` at best, and the non-gemm parts of that
  stage (router, combine, SwiGLU glue) rank above it. Estimate unknown but capped low.
  [CONTESTED 2026-09-05: that "minority" reading came off the 2.2x-inflated stage
  profiler, two in-situ A/Bs bracket the expert gemms at 14-43% of prefill WALL, and the
  cap is lifted to "unpriced" — [record](docs/records/ceiling-diagnosis.md).]
  [PRICED 2026-09-05: 28-32% of prefill wall by the duplicate-dispatch probe.]
  From: Deferred from the technique survey (2026-08-30).

## Drafting

- [ ] [measured] **DFlash adaptation to the Qwen sidecars — ADAPTED 2026-07-29, but speculation is
   a 27B-only win and stays opt-in.** Both sidecars load, draft and verify correctly at
   85-95% acceptance: 27B +4.8 to +6.8% on a code prompt and +1.5 to +7.4% on a chat
   prompt, 35B-A3B -11.5% and -12.7%; `draft_p_min` retuned 0.5 → 0.3, `pause_margin`
   stays 1.0. [Record](docs/records/dflash-drafting.md), decisions.md "Speculative
   decoding". Original scope: repoint drafter arch check
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
     Fixing either could flip `--draft` to opt-out; see (d).
   - **(c) `DEFAULT_DRAFT_CTX` (8192) was NOT re-derived and its inherited rationale
     is now half wrong.** Laguna's argument was O(depth) drafter forwards plus
     collapsing proposal quality with depth. The O(depth) half no longer holds: every
     sidecar layer but the last is windowed (2048 on the 27B, 4096 on the 35B) and
     `attention` narrows the cache to the window, so only one layer of five or six
     grows with the context. The memory argument stands (40 KiB/token on the 27B,
     48 on the 35B, imaged per cache slot). Re-derive by measuring drafter cost and
     acceptance at 4k/8k/16k/32k on the Qwen sidecars before changing it.
   - **(e) A ring-buffer drafter cache is deferred.** The per-layer cache stays a flat
     `[n_kv, max_ctx, hd]` array; windowing lives in `attention`'s narrow-plus-mask.
     A ring would cap the allocation at the window rather than at `draft_ctx`, but it
     would also stop `DrafterImage` being a straight prefix copy of the committed
     rows, which is what makes export/import and the disk tier simple. Only worth it
     if `draft_ctx` grows a lot under (c).
  From: Priority order (decided 2026-07-28; P1-P9 shipped by 2026-07-29).

- [ ] [unpriced] **Serve slots persisted without drafter planes silently decode plain forever
  under default-on drafting** (Codex review of the flip, corroborated against the
  code). `hydrate`'s `None => reset_drafter()` branch (serve/engine.rs, see the
  comment there) was written for the flag-change edge; with drafting default-on it
  is the COMMON path against slots written by `--no-draft` runs or pre-drafting
  builds. A reset drafter at a nonzero restore point can never resync — its cache
  is fed by target-layer taps during target forwards, `drafter_span_rows` returns
  0 whenever `pos != committed` (unit test at generate.rs:3139 pins this), and
  re-seeding would require re-running the target prefill the snapshot exists to
  avoid. Output stays correct; speculation is lost and the server still reports
  draft ON. Options when this is picked up (fits P10 serve adaptation): (a)
  per-conversation draft status so the degradation is at least visible, (b) drop
  the snapshot when drafting is enabled and planes are absent (trade prefill reuse
  for speculation — wrong for long contexts), (c) accept and document. No option
  is obviously right without measuring how often real serve traffic hydrates
  plane-less slots.
  From: Deferred from the K-snapshot verify pass (2026-07-29, P9a).

- [ ] [blocked] **The draft-by-default flip makes a mismatched custom GGUF fail at startup.**
  With drafting opt-out, `xwen serve --model <custom.gguf>` whose geometry fails the
  drafter preflight (`DflashConfig::check_against_target`) now hard-errors where it
  previously ran plain; `--no-draft` is the workaround. Recommended shape when it
  bites someone: an IMPLICITLY-defaulted drafter that fails preflight should degrade
  to plain decoding with a warning, while an EXPLICIT `--draft` keeps the hard error.
  Not built now: the design target is the two blessed checkpoints, whose sidecars
  always match, so the edge is custom-GGUF-only.
  From: Deferred from the K-snapshot verify pass (2026-07-29, P9a).

- [ ] [measured] **Every verify arm goes superlinear at span 48.** Fused 264 → 548 ms between
  spans 32 and 48; classic 472 → 846; reproduced across reps and arms. NOT the
  dense-mm threshold (`XWEN_DENSE_MM_CLASSIC=1` arm shows the same jump). Outside
  the production regime — drafter `block_size` 16 caps real verify spans near 17 —
  so unchased, but unexplained. Also from the same pass, one anomalous
  `XWEN_DENSE_MM_CLASSIC=1` sweep read ~17% high at every span including span 2
  where that kernel cannot run, against four mutually-consistent fused sweeps and
  an immediate re-run that matched them; single unreplicated outlier, recorded
  here so a future contradiction has a trail.
  ANNOTATION 2026-08-08: **new evidence — the overshoot is ARMING-dependent (armed runs
  overshoot 1.54-1.65x, unarmed come in at 0.80-0.91x), so it is trail memory pressure
  rather than a kernel threshold. Still outside the production regime and unchased.**
  [Log](docs/log.md#2026-08-08--verify-round-diagnosis-the-149-ms-fixed-cost-is-the-dense-ffns-matmuls-at-small-m-and-none-of-the-armed-machinery-it-was-blamed-on).
  From: Deferred from the K-snapshot verify pass (2026-07-29, P9a).

- [ ] [small] **DFlash draft-slot handling across snapshot/restore was never checked against
  SGLang's pattern.** Batch replay syncs the drafter by truncation (`sync_drafter_to`),
  which is correct here because every item shares every token below the snapshot
  position — see decisions.md "Batch". The serving-SOTA research surfaced SGLang's
  snapshot/promote handling of speculative draft slots across cache reuse, which solves a
  strictly harder problem (concurrent sequences, divergent branches) and may name a case
  the truncation argument does not cover. Read it against `sync_drafter_to` and
  `DrafterImage` before the multi-level prefix tree or the serve endpoint lands, since
  both break the shared-prefix premise truncation rests on.
  From: Deferred from the batch + scored-classification arc (2026-08-09).

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
  From: Deferred from the MTP drafting arc (2026-08-15, stages B and C).

- [ ] [unpriced] **A stored MTP cache image resumes only at the position it ends at.** Same root as
  the item above, seen from the disk tier: `DrafterImage` carries one carry hidden, so
  `MtpDrafter::import_cache` refuses a `pos` short of `image.pos` rather than restoring a
  head that cannot take another token. A page-in that resumes at an earlier snapshot
  therefore loses the drafter planes and runs that conversation plain — which is the
  regime `Engine::rejects_image` already documents as acceptable (a drafter refusal costs
  speculation, not the conversation), but it is more common for this kind than for
  DFlash, whose images take any prefix. Fixed by whichever fix the item above gets.
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

- [ ] [blocked] [x] **DONE 2026-08-14 (review round): drafting is resolved per checkpoint, not per
  process.** Filed as deferred in the first pass and fixed in the review round, because a
  sidecar-less DEFAULT checkpoint silently disabled drafting for every OTHER checkpoint
  that server could load (-46 to -52% on the 27B, invisible). `ServeSettings.draft` is
  now a `DraftMode` (`Off` / `Official` / `Custom(path)`) resolved when each checkpoint
  loads. [Record](docs/records/serve-target-review.md), and the rules it settled are in
  [decisions/serving.md](docs/decisions/serving.md).
  What remains: nothing about the shape, but the fallback floor it exposes is worth a
  measurement. A custom drafter attached to a checkpoint with no fitted floor of its own
  falls back to `SpecParams::default().draft_p_min`, which is the 35B-A3B's fitted 0.3
  wearing a neutral name — an arbitrary value for that pair. If anyone actually runs a
  custom drafter on Qwen3.8-27B, fit a floor for it (`scripts/retune-draft.ts` cannot:
  it sweeps official sidecars only).
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
  From: Deferred from the qwen4exp cache-image arc (2026-08-30, P4).

## Serve, batch and CLI

- [ ] [small] **serve adaptation.** Tool-call parsing for the `<function=...>` XML-ish format in
    both API dialects (string args raw, non-string JSON), thinking-mode flags
    (enable_thinking / preserve_thinking) surfaced per dialect, prefix-cache + disk
    tier snapshots extended with recurrent state (48–96 KiB conv + 2–6 MiB delta per
    snapshot depending on model). Estimated-prefill scheduling unchanged.
  From: Priority order (decided 2026-07-28; P1-P9 shipped by 2026-07-29).

- [ ] [blocked] **Prefix grouping is single-level, and there is no cross-batch pinned snapshot.**
  One batch computes one LCP over all items; items that share more with each other than
  with the batch as a whole get no credit for it, and a system prompt shared across
  successive batch requests is re-prefilled every time. The literature says the first
  costs little: BatchLLM measured single-level collapse at roughly 1% of achievable reuse
  against a full prefix tree, and a tree brings eviction, invalidation and per-node
  snapshot accounting with it. The pinned cross-batch snapshot is the cheaper of the two
  and is the one to build first if it is built. Revisit only with a measured workload
  where the single level demonstrably loses, not on principle.
  From: Deferred from the batch + scored-classification arc (2026-08-09).

- [ ] [small] **Results are not streamed.** `xwen batch` prints one JSON document when the last
  item finishes; progress goes to stderr as unstructured lines. A long batch therefore
  gives a caller nothing machine-readable until the end. NDJSON on stdout (one
  `ItemResponse` per line, `BatchStats` last) is the obvious shape and would not change
  the core, which already completes items in request order. Wants a flag rather than a
  format change — the current single-document output is what makes `jq` over a batch
  trivial.
  From: Deferred from the batch + scored-classification arc (2026-08-09).

- [ ] [small] **Per-token logprobs are not exposed in any dialect.** `include_score` reports
  confidence over a field's ALLOWED OPTIONS, which is a different quantity from
  OpenAI's `logprobs`/`top_logprobs` (raw log-softmax over the vocabulary at each emitted
  position, top-k of it). The machinery for both now exists — `Generator::last_logprobs_for`
  is the log-softmax over an encodable slice — but the two must not be conflated in the
  surface: a client asking for `logprobs` wants token evidence, not label evidence.
  Independent of the scored path; belongs with the serve adaptation.
  From: Deferred from the batch + scored-classification arc (2026-08-09).

- [ ] [unpriced] **Scored-field probabilities are conditional on the compact skeleton, and the
  formatting channels disagree on near-ties.** The 2026-08-11 row dump behind the
  escape fix shows the first boolean slot at ` true` 54.9% / ` false` 44.9%
  (space-led spellings) while the bare-token channel the teacher-forced skeleton
  actually scores through reads true 0.444 / false 0.556 — the two channels pick
  OPPOSITE winners on this near-tie. Away from ties they agree (spam field: 0.998
  false both ways), and the scores' renormalization argument ("formatting divides
  out") holds only when format preference is independent of the value, which this
  measurement shows it is not exactly. Candidate refinement: also score each option
  through its space-led single-token spelling and sum the channels; interacts with
  check_seams and the terminator rule. Do not treat a scored near-tie (|p−0.5| small
  on a boolean) as a confident answer; the escape fix does not change this.
  From: Deferred from the batch + scored-classification arc (2026-08-09).

- [ ] [unpriced] **`escape` conflates value disagreement with format drift, materially so at
  first fields; a split is a candidate refinement.** The 2026-08-12 confirmation dump
  (log.md) shows 25-46% of the no-think field-0 outside mass is ` True`/` False` — the
  chosen answer in a spelling that would invalidate the JSON — and bare `True`/`False`
  are 28-87% of later fields' outside mass, so the mixture is everywhere, not a
  first-field artifact. Classifying these OUTSIDE is correct (the assembler would
  never emit them), but escape therefore overstates value-level disagreement wherever
  it is read, and first fields are where its absolute magnitude (1e-2 vs 1e-5) makes
  that material. Candidate: report escape's top outside
  components, or split it into value-escape (bytes that prefix no option under ANY
  casing/spelling equivalence) vs format-drift (an option's bytes under a
  non-canonical spelling). The equivalence class is the hard part — it interacts with
  the channel-summing refinement one item up, and both should be derived together if
  either ships. Until then the README definition stands and consumers should compare
  escape across categories with and without first fields.
  From: Deferred from the batch + scored-classification arc (2026-08-09).

- [ ] [unpriced] **v1's scored-schema limits are refusals, and each has a known lift.** The shape
  guard accepts a flat all-required object of enum/boolean fields and refuses everything
  else by name. Four separable extensions, in rough order of value: (1) values that merge
  with their delimiter under BPE — the seam check refuses them today; scoring the merged
  token as the option's last token is the principled fix, and it interacts with the
  terminator-token rule, so derive them together; (2) JSON-escaped values, refused
  because the escape sequence rather than the label would be what gets scored; (3)
  free-form fields alongside scored ones, which means interleaving assembly with
  grammar-masked decode inside one document; (4) free `thinking: true` combined with
  `prefill`, currently not composable on a scored item. Each is a scope decision, not a
  bug.
  From: Deferred from the batch + scored-classification arc (2026-08-09).

- [ ] [small] **Batch-over-HTTP gives no progress until the last item.** The CLI shows stderr
  progress lines; the HTTP client gets one JSON document at the end and nothing before
  it (a proxy that times out idle responses will cut a long batch off). The engine-side
  hooks already emit per-item progress into the server log, so an SSE or NDJSON variant
  of the route is wiring, not design. Related to the existing "Results are not
  streamed" item in the 2026-08-09 section — solve both with one shape when picked up.
  From: Deferred from the serve batch + multi-checkpoint arc (2026-08-11).

- [ ] [small] **Neither `/health` nor the TUI says which checkpoint is loaded.** `/health`
  reports `model_loaded` as a bare bool and the TUI vitals were built around one model
  id for the process lifetime. Post-swap, both are truthful but incomplete: nothing
  outside the log line says the resident model changed. Cheap: a `model` field on
  `/health` from a shared `AtomicU8`-style cell the engine stamps at load, and a vitals
  line. Do it with the first operational confusion, or sooner if the TUI gets touched.
  From: Deferred from the serve batch + multi-checkpoint arc (2026-08-11).

- [ ] [blocked] **A cache-miss checkpoint downloads inside the request that named it.** ~20 GB
  on a miss, inside one HTTP request, racing the watchdog deadline if one is configured
  (the download resumes in place, so a retry eventually completes — hf-hub semantics).
  Both checkpoints are cached on this machine, so this is theoretical here; if it ever
  bites, the fix is a 503-with-progress answer while a background fetch runs, not a
  longer deadline. Also: hf-hub's own byte-level progress bar writes to raw stderr
  (`ApiBuilder::with_progress(true)`), bypassing `ServeLogger` — under `--tui` it
  draws over the dashboard, the same hazard class the batch runner's `eprintln!`s
  were converted to hooks for. Route or suppress it when this item is picked up.
  From: Deferred from the serve batch + multi-checkpoint arc (2026-08-11).

- [ ] [blocked] **`/xwen/v1/generate` carries no model field.** Deliberate — the native generate
  surface documents itself as modelless and the batch route is the native surface that
  selects — but it is now the only route that cannot reach the non-default checkpoint.
  Add the field if a native-API consumer ever wants it; it is a two-line change in
  `prepare` plus tests.
  From: Deferred from the serve batch + multi-checkpoint arc (2026-08-11).

- [ ] [blocked] **Mid-batch cancellation does not reach the scored path's forced spans, nor any
  prefill.** The cancel poll runs between items and per decoded token inside an item's
  free decode; a scored item's teacher-forced assembly checks only at item boundaries,
  and neither the shared-prefix prefill nor an item's own tail prefill polls at all
  (`prefill_tokens` chunks internally but takes no callback). Items are short (≤192
  tokens in the demo), so the exposure is bounded by one item's latency plus one
  prefill — thread the poll through `assemble_scored` and the prefill chunk loop only
  if a real workload makes either span long enough to care.
  From: Deferred from the serve batch + multi-checkpoint arc (2026-08-11).

- [ ] [blocked] **The batch scheduling estimate is bytes-based and can read zero.** A batch of
  items with empty message content (schema-only probes) estimates zero prompt tokens
  and schedules as free; the real cost floor is the rendered template per item. Fold a
  per-item constant into `size_estimates` (or estimate from the rendered skeleton)
  when scheduling fairness under real mixed traffic matters; on a single-user box the
  age limit already bounds the damage.
  From: Deferred from the serve batch + multi-checkpoint arc (2026-08-11).

- [ ] [blocked] **The scheduler does not group queued jobs by checkpoint.** `shortest-prefill`
  scores by prefill cost alone, so a queue holding jobs for both checkpoints can pick
  them interleaved and pay a ~3 s swap per pickup where checkpoint-grouped ordering
  would pay two. The cost model could add the swap (a job for the non-resident
  checkpoint costs its prefill plus a load-equivalent), which also naturally batches
  same-checkpoint work without starving the other (the age limit already guards
  starvation). Do it when a real workload actually interleaves checkpoints; a
  single-user machine mostly will not.
  From: Deferred from the serve batch + multi-checkpoint arc (2026-08-11).

- [ ] [blocked] **100 MB bodies are buffered with no concurrency bound.** The batch handler
  buffers and serde-parses the whole body (typically 2-5x the text in tree form)
  BEFORE `submit_batch` can answer 429, and nothing caps concurrent connections — N
  clients can each hold ~100 MB + parse tree against 19-37 GB of resident weights.
  Accepted for now: the default bind is loopback on a single-user machine, and the
  compat dialects never need large bodies. If the server ever fronts a LAN under
  `api_key`, add a concurrency-limit layer (or move the cap per-route: 100 MB for
  `/xwen/v1/batch`, default for the dialects) before raising anything else.
  From: Deferred from the client-feedback arc (2026-08-11).

- [ ] [blocked] **The Anthropic dialect has no per-request template-effort knob.** Its API shape
  has no natural field: `thinking.budget_tokens` is a budget, not a level, and inventing
  a nonstandard field on a compat dialect defeats the point of speaking the dialect.
  Requests get the server-wide `[thinking] effort` default (which `count_tokens` also
  renders, so counts match generation); a client that needs per-request effort on 3.8
  uses the OpenAI or native dialect. Revisit only if Anthropic's API grows an effort
  field to mirror.
  From: Deferred from the chat-dialect and sampling-defaults arc (2026-08-19).

- [ ] [blocked] **Flash-Next's two closed gates: auto_fetch and supports_drafting.**
  TWO GATES DID NOT MOVE and this ledger keeps them: `auto_fetch()` stays false (a
  111 GB fetch is explicit-only) and `supports_drafting()` stays false (D6's missing
  speculative verify seam). The unconsumed `recommended_presence_penalty`, the Unsloth
  template divergences and the seven audio/TTS specials are untouched and still open,
  under «Flash-Next's unconsumed presence penalty, template divergences and audio
  specials» in "Tokenizer, chat and sampling". New work this arc left behind carries the
  `From:` line "Deferred from the qwen4exp cache-image arc (2026-08-30, P4)".
  From: Qwen3.8-Flash-Next port (decided 2026-08-25, blocked on release + upstream).
  Promoted from the item «Port Qwen3.8-Flash-Next» on 2026-09-06; dated 2026-08-30 there.

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

- [ ] [small] **The bench and parity scripts record into the same file and skew usage stats.**
  `scripts/*.ts` drive `generate`, `chat` and serve through the same surfaces everyone
  else uses, so a sweep's several hundred runs land in the history beside real use, and
  a `--by day` table taken after a retune reads as a day of heavy inference that nobody
  did. Recording them was the deliberate default (decisions.md: silent exclusion is the
  harder mistake to notice), so the fix is one of two shapes and neither is decided:
  the scripts export `XWEN_METRICS_FILE=off`, which loses the data; or records grow a
  tag field the scripts set and `xwen stats` filters on, which keeps it and costs a
  schema field plus a default filter decision. `--surface` and `--since` carve a bench
  session out by hand today.
  From: Deferred from the metrics arc (2026-09-05).

- [ ] [small] **A `[metrics]` table in serve.toml (path, enabled).** `XWEN_METRICS_FILE` is the
  only control and it reaches all four surfaces, which is why it shipped alone
  (decisions.md). A server is the one surface with a config file and the one that runs
  long enough for an operator to want its history somewhere specific — a per-deployment
  path, or recording off for a server that fronts a benchmark. The table would override
  the variable for serve only; keep the variable as the surface-wide answer.
  From: Deferred from the metrics arc (2026-09-05).

- [ ] [small] **`x-claude-code-agent-id` is not recorded.** Claude Code sends it on subagent
  requests. It was left out on purpose: recording it as the session would split one
  session into a row per agent, which is the wrong default for the question `--by
  session` answers. It is real information though, and "which subagent burned the
  tokens" is a question this history could answer. If it lands it wants its own field
  and its own `--by agent`, never a fallback inside `session_key`.
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

- [ ] [blocked] **The file grows forever; there is no rotation or compaction.** A line is ~250
  bytes, so this is a slow problem: a hundred runs a day is under 10 MB a year, and the
  full-scan reader stays comfortable well past that. `--since` bounds what a report
  reads over, not what the file holds. Nothing in xwen prunes, and that is deliberate
  for now (the durable-history choice is the whole reason it is not in the cache dir),
  but a machine left running for years wants either a size-triggered roll to
  `metrics.jsonl.1` or a compaction that folds records older than N months into daily
  summaries. Decide when the file is big enough to be worth measuring, not before.
  From: Deferred from the metrics arc (2026-09-05).

- [ ] [blocked] **One UTC offset is applied to every record in a report, so a daylight-saving
  change buckets runs onto the wrong local day.** `read_utc_offset` shells out to
  `date +%z` and the answer is read once per report, not once per record, so a report
  covering both sides of a clock change interprets the far side an hour off. Away from
  midnight nothing moves; within an hour of it a run lands in the neighbouring day's
  bucket, and `--since YYYY-MM-DD` picks its cutoff by the same offset. Reading the
  offset per record would mean a subprocess per line, which is not the fix; the fix is
  either a real tz lookup (a dependency this arc declined) or reading
  `/var/db/timezone`-style rules directly. Worth doing when someone is misled by it,
  which on a machine that mostly runs in one season is not yet.
  From: Deferred from the metrics arc (2026-09-05).

- [ ] [small] **The serve TUI does not show a job's model, though `JobRecord` now carries it.**
  The metrics work added `model` to the job record so a served run could name its
  checkpoint; the dashboard's job rows still do not display it. On a single-checkpoint
  server that is nothing, and on a server that lazy-swaps between checkpoints it is the
  one field that explains a row's rate. Cheap: a column in the job table, the field is
  already there.
  From: Deferred from the metrics arc (2026-09-05).

## Cache images, memory and context

- [ ] [measured] **The router gemv holds `ffn_gate_inp` twice, ~251 MB resident f32 on Flash-Next.**
  **The router gemv holds `ffn_gate_inp` TWICE, 2026-09-06.** `MoeBlock` keeps the
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

- [ ] [blocked] **YaRN long-context.** Native 262144; Qwen documents 1M via YaRN but ships no
    scaling keys in config or GGUF. laguna's YaRN rope code is retained; wire an
    opt-in flag only on demand. Note rope table memory at 262k is trivial (64 dims).
  From: Priority order (decided 2026-07-28; P1-P9 shipped by 2026-07-29).

- [ ] [blocked] **The disk tier serves only the default checkpoint.** A non-default checkpoint
  runs with every disk-tier call site handed `None`: the tier binds to one checkpoint
  id at startup and `verify()` permanently distrusts itself against any other. The
  segment layout is already per-checkpoint directories (`root/<checkpoint>/`), so the
  lift is opening one tier per checkpoint lazily rather than one at startup — do it
  when a workload actually alternates checkpoints and misses its warm conversations,
  not before. Until then a swap costs the outgoing checkpoint's warm slots and, with
  the tier on, keeps only the DEFAULT checkpoint's conversations across swaps.
  From: Deferred from the serve batch + multi-checkpoint arc (2026-08-11).

- [ ] [unpriced] **The 128k operational envelope is unmeasured, and four constants were sized at
  8192.** Every perf figure in CLAUDE.md is at max_ctx 8192; raising the default makes
  long contexts REACHABLE, not characterized. Known pressure points, none touched by
  the lazy-KV change itself: (a) the prefill mask is sized by absolute position, not
  max_ctx — `PrefillMask::from_host` materializes `[1, n_head, seq, pos+seq]` f16 per
  512-token chunk, ~3.0 GiB transient at position 128k on the 27B (2.0 on the 35B)
  plus an f32 host Vec filled by a scalar double loop (~8.6e9 stores over a full 128k
  prefill) — this, not KV, is the binding cost of long prefill; (b)
  `DEFAULT_QUEUE_TIMEOUT_SECS` = 300 while a 128k 27B prefill is 187-295 s at the
  measured 445-702 tok/s, so one long prefill can push a queued request into the
  saturation drop; (c) `DISK_FLUSH_GRACE` = 25 s was sized on a ~4.2 GiB / ~5 s page-out
  image, while a 128k 27B conversation images at ~8 GiB; (d) the drafter's
  `draft_ctx` = 8192 horizon means speculation covers the first 6% of a 131072
  conversation and goes plain past it with no log line — the shipped drafted tok/s
  figures describe conversations inside that window only; (e) only the serve path
  clamps `context_length` to the checkpoint's `n_ctx_train`
  (`resolve_context_length`) — the CLI's `--max-ctx` never consults it, harmless for
  the 262144-window blessed files but silently past-window for a checkpoint converted
  smaller. Measure a real long-context workload before trusting (a)-(d); none matters
  at yesterday's 8192.
  From: Deferred from the client-feedback arc (2026-08-11).

- [ ] [blocked] **Lazy KV moves the unaffordable-`max_ctx` failure from load time to
  mid-conversation.** Eager allocation failed fast at load; now the same misconfigured
  server starts fine and hits the allocation error at whatever depth exhausts the
  device — a growth step failing mid-request surfaces as that request's error (the
  state is safe and retries converge, `grow_kv_capacity`'s doc). `MEMORY_WARN_BYTES`
  (90 GiB) never fires for any blessed file even at the 262144 ceiling, so the warning
  is not the guard here. If this ever bites, the fix is a load-time advisory line
  ("ceiling X GiB exceeds device memory Y") rather than a return to eager allocation.
  From: Deferred from the client-feedback arc (2026-08-11).

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
  conversation ever gets there. That was already ledgered as a memory item under the P3
  P3 ledger and as one of the three shrinks in "Refuted: the ~15 GB of private
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

- [ ] [blocked] **Mid-message snapshots would let an edited prompt resume; ledger only.** Today
  `rewind_to` can only stop at the anchor, a turn boundary, a fork point or a page-out
  tail, so rewriting the last user message of a single-message prompt falls under every
  snapshot and re-prefills from zero (`cached_tokens: 0` at all three lengths measured).
  Periodic snapshots INSIDE a long message — every N thousand tokens — would give the
  edit somewhere to land, and the recurrent state makes it a snapshot problem rather
  than a matching problem: there is nothing to restore at a position nobody captured.
  The price is what makes this a ledger item and not a task: a Flash-Next image is
  ~30 KiB/token plus the ~113 MiB DeltaNet floor per snapshot, so periodic stops inside
  a 32k message cost hundreds of MB of host RAM to save prefill for a client that edits
  prompts in place — a workload nobody here has. Revisit if one shows up; the knob would
  be an interval, defaulted off.
  From: Deferred from the first Flash-Next serve benchmark (2026-08-30).

## Parity, provenance and tooling

- [ ] [small] **27B dense bring-up — MOSTLY DONE 2026-07-28 via P7.** The parity gate ran the
    27B end to end: first forward correct, all gated tiers pass (strict is
    near-vacuous on the dense model — see P7c). Remaining: an interactive
    generate/chat smoke run, decode/prefill perf numbers for the 27B (nothing
    measured yet; 64 layers dense will be much slower per token than the A3B), and
    the deferred conv threadgroup-sizing check when P8 lands.
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

- [ ] [unpriced] **Bit-identity claims ride an unpinned Metal compile axis (candle compiles Fast, vendored kernels do not).**
  **The bit-identity claims ride an unpinned compile axis** (outside-model review,
  2026-07-29). candle rev 21cca0b compiles its kernels with BOTH
  `MTLMathMode::Fast` and `MTLMathFloatingPointFunctions::Fast`
  (candle-metal-kernels `kernel.rs:191-192`); `pipelines.rs` compiles the vendored
  sources with default options, and the fp pragmas pin only the math-mode axis.
  Kernels calling `fast::exp`/`fast::divide` explicitly (the router) are immune;
  the epilogue's bare `exp(-g)` in its sigmoid is the one spot where a toolchain
  that lowers the two axes differently could split bits. Empirically identical on
  this machine today — the bitwise ops tests and the strict parity tier are the
  tripwire, and any future failure there should suspect this first. The clean fix
  is constructing `MTLCompileOptions` to mirror candle's exactly, but that changes
  the compile of EVERY vendored kernel and needs a full bitwise-suite + parity
  re-run as its own arc, not a drive-by.
  From: Priority order (decided 2026-07-28; P1-P9 shipped by 2026-07-29).
  Promoted from the item «MoE block glue fusion — SHIPPED 2026-07-29» on 2026-09-06; dated 2026-07-29 there.

- [ ] [small] **The attention and full-stack decode/prefill benches were deleted, not
  ported.** `attention.rs`'s `tests::decode_bench` + `tests::prefill_bench`
  modules (~1300 lines) and `moe.rs`'s `full_stack_decode_bench` measured
  Laguna's attention chain — 48/72 heads, head dim 128, SWA rings, softplus
  per-head gate — none of which exists in Qwen 3.6, and they were the only
  consumers of the SWA-geometry test scaffolding. The MoE-side benches
  (`moe_decode_ffn_bench`, the expert-gather attribution set) survive unchanged.
  Rebuild the attention-side equivalents at Qwen geometry (16 Q / 2 KV heads,
  head dim 256, double-width `attn_q`, uniform causal) when the decode budget is
  next attacked; the deleted versions are in git history at the fork point.
  From: Deferred from the P2-P4 model-core retarget (2026-07-28).

- [ ] [small] **The `mtl_size!` rationale in dispatch.rs is factually wrong.**
  `src/ops/dispatch.rs:21-24` justifies the macro with "xwen does not depend on
  objc2-metal directly, and a function cannot return the unnameable type" — but
  `Cargo.toml:26-28` pins the objc2 crates as direct dependencies, `src/gguf.rs:141`
  already named `objc2_metal::MTLDevice`, and `check_delta_simd_width` now names
  `objc2_metal::MTLComputePipelineState` in that same file. `objc2_metal::MTLSize` is
  therefore nameable and the macro may be unnecessary. Left alone rather than
  rewritten: correcting the stated reason means either inventing a rationale nobody
  has verified or reworking the grid helpers, and neither belongs in a pass whose
  contract was that no computed value moves. Decide whether the macro earns its keep
  (candle's `get_block_dims` round-trip vs a plain struct literal) and rewrite or
  delete the comment to match.
  From: Deferred from the DeltaNet-kernel hardening pass (2026-07-29).

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

- [ ] [small] **`glance` the copied scripts/ for maxuna-isms** beyond the mechanical rename
  (bench prompt fixtures, hardcoded model names, parity-gate assumptions). Still unswept:
  `classify.ts`, and `tests/fixtures/bench-prompts` (never opened).
  From: Deferred from the fork bootstrap (2026-07-28).

- [ ] [measured] **The 35B's perplexity delta grew with the fused DeltaNet scan and the floor's
  margin shrank.** `PPL_NLL_DELTA_MAX = 0.002` stands and is not re-derived from the
  fused measurement (RESOLVED 2026-07-28 by the parity owner;
  [rationale and trip-wire](docs/parity.md#perplexity-gate)).
  Still open as a WATCH item: the fused scan sits at 0.000791 on the 35B, so a
  further ~2.5x rise fails the gate, and the sign is systematic (candidate worse in
  all four measurements across both architectures — the fused scan widened the gap
  ~+50% on each). This is the single most sensitive number the gate reports about the
  fused scan; the cosine tiers barely moved (35B mm actually improved, 0.999540 →
  0.999631).
  From: Deferred from the fork bootstrap (2026-07-28).

- [ ] [blocked] **Partition-parity drift never measured.** The q8/f16 dual-storage split makes
  cached state depend on call partitioning (see decisions.md "Kernel policy" entry,
  2026-07-28). Accepted by decision, but the drift magnitude at the 8↔9 boundary on
  real weights has never been quantified — measure it (same prompt, cache on/off,
  compare state and downstream logits) if a near-tie flip is ever suspected in
  production, before blaming sampling.
  From: Deferred from the fork bootstrap (2026-07-28).

- [ ] [blocked] **Quant-vendor comparison never measured.** ggml-org was chosen over
  unsloth/bartowski on provenance (converter authors, inspectable custom mix, dflash
  sidecars), not on quality. Now that the perplexity gate exists, pointing it at a
  competing Q4_K_M is cheap — run it if output quality ever comes into question.
  From: Deferred from the fork bootstrap (2026-07-28).

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
  From: Deferred from the Qwen3.8-27B + API-naming arc (2026-08-14).

- [ ] [small] **`spec-equivalence.ts`'s sampled mode grades itself with a heuristic that
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

- [ ] [small] **No ppl reference fixture for Qwen3.8-27B.**
  **2026-08-29 — no ppl reference fixture for Qwen3.8-27B.** Re-grading the
  oracle bump (e9fa0781 → `6fe749801`) turned this up: the 3.8-27B parity run
  can only grade strict/mm/decode, because
  `tests/fixtures/reference-ppl-Qwen3.8-27B-Q4_K_M.json` does not exist and
  never has — a full-tier run bails with "ppl reference fixture missing", so
  the checkpoint has been shipping without a perplexity floor since it was
  added. Nothing regressed; the tier was simply never calibrated for this
  file. Fix: `--regen-ppl-ref` against the 3.8 hub file, then grade ppl and
  record the floor in docs/parity.md beside the 3.6 pair's. Until then the
  3.8's parity coverage is 5 checks where the others get 6.
  From: Qwen3.8-Flash-Next port (decided 2026-08-25, blocked on release + upstream).
  Promoted from the item «Port Qwen3.8-Flash-Next» on 2026-09-06; dated 2026-08-29 there.

- [ ] [small] **The parity harness cannot run on qwen4exp.**
  **2026-08-29 — P4: the parity harness cannot run on qwen4exp (from U7).**
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
  From: Qwen3.8-Flash-Next port (decided 2026-08-25, blocked on release + upstream).
  Promoted from the item «Port Qwen3.8-Flash-Next» on 2026-09-06; dated 2026-08-29 there.

- [ ] [small] **parity-gate.ts accepts --model-size flash-next with no fixtures behind it.**
  `scripts/hf.ts`'s flash-next entry widens what `--model-size` the parity gate
  accepts, with nothing behind it. The entry exists so `bench.ts` can resolve
  the checkpoint (b54046b), but the gate reads the same table, so
  `parity-gate.ts --model-size flash-next` is now spellable and will fail deep
  rather than at argument validation — the harness cannot run on this checkpoint
  at all and there are no fixtures for it. Low priority precisely because the run
  fails either way; fix by gating the gate's accepted set on fixture existence.
  Same entry: its **`shards` key is dead** — nothing reads it, the loader finds
  the shard set from any one file. Delete it or make it load-bearing.
  From: Qwen3.8-Flash-Next port (decided 2026-08-25, blocked on release + upstream).
  Promoted from the item «Port Qwen3.8-Flash-Next» on 2026-09-06; dated 2026-08-29 there.

- [ ] [small] **Three upstream reports owed, none filed.**
  **2026-08-29 — Upstream reports owed (three, none filed).** **(1) candle
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

- [ ] [small] **`XWEN_GDN_PROFILE`'s decode line overstates a step by roughly its dispatch
  round trip, and the dispatch-floor correction does not recover it.** The scan measured
  3.79-7.19 ms/token corrected in that line against 1.43 ms/token in an amortized bench
  of the same work at the same geometry, and the same inflation applies to every step in
  the line (its raw mixer total, 78 ms, is more than three whole unprofiled tokens). The
  line is still useful for RANKING steps within one run — which is how the 27B prefill
  work used it — but a decode figure from it must not be quoted as a cost, and the
  shares it prints are shares of an inflated total. Either bracket the whole block once
  and attribute by difference, or run every step under `XWEN_GDN_REPS` and say so on
  the line; until then treat it like `XWEN_STACK_PROFILE`'s decode stages (CLAUDE.md
  already says those rank, not time).
  From: Deferred from the DeltaNet decode-scan pass (2026-08-30).

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

- [ ] [small] **`strided_sum`'s reduce-order replay refuses extents above 5, so a wider indexer geometry fails at `select`.**
  The reason is candle's reducer, which
  folds through a 4-lane `simd_sum` there so the bit-identity breaks (1 ulp at
  extent 6). Both production extents are 4; a checkpoint with `ratio` or indexer
  head count above 5 would fail at `select` and needs either the plain `sum` (bounded,
  not bitwise) or a widened replay.
  From: Deferred from the prefill-chunk pass (2026-08-30).
  Promoted from the item «Decode on Flash-Next steps down ~11 ms/token the moment the context crosses the 2048-token QSA budget, then slopes gently» on 2026-09-06; dated 2026-08-30 there.

- [ ] [small] **The fused QSA gather is Metal-only and alignment-restricted; the CPU/oracle path silently takes the `index_select` chain.**
  A non-Metal source (the CPU/oracle attention
  path) takes that chain with no switch. The kernel refuses a view
  whose start or head stride is not a multiple of 4 elements (vec4 loads).
  From: Deferred from the prefill-chunk pass (2026-08-30).
  Promoted from the item «Decode on Flash-Next steps down ~11 ms/token the moment the context crosses the 2048-token QSA budget, then slopes gently» on 2026-09-06; dated 2026-08-30 there.

## Tokenizer, chat and sampling

- [ ] [small] **`top_k = 0` means greedy here, "top-k disabled" in llama.cpp.** The sampler
  maps every `top_k <= 1` to argmax, where llama.cpp treats `k <= 0` as a no-op
  filter (the whole vocabulary stays eligible). Pre-existing, harmless at the
  default of 20, but the serve layers forward client-supplied values verbatim, so
  a llama.cpp-reared client sending `top_k: 0` gets deterministic output instead
  of unrestricted sampling. Surfaced by outside-model review 2026-07-29. A
  semantics decision like the temperature-order item below, not a bug fix.
  From: Deferred from the sampler-tail pass (2026-07-28).

- [ ] [small] **Temperature is applied before the top-k/top-p cut; llama.cpp's default
  chain cuts first.** Found 2026-07-29 while transcribing `top_p`: llama.cpp's
  default sampler chain is top_k → typ_p → top_p → min_p → temp → dist
  (common/sampling.cpp), so its truncation sees raw-logit probabilities, while
  xwen (like HF's default warper order) scales by temperature first, so the cut
  sees the sharpened/flattened distribution. At the model's default temp 1.0 the
  two are identical; the divergence only bites when `--temp` is overridden.
  Convention question like the top-p one was — llama.cpp and HF disagree with
  each other here, so there is no single ground truth to defer to. Needs a
  decision, not a patch.
  From: Deferred from the sampler-tail pass (2026-07-28).

- [ ] [small] **Qwen3.6 vocab is 248320 padded / 248077 real, and constrain.rs will trip on
  it.** `constrain.rs:90` asserts `tok_trie().vocab_size() == expected_vocab` and
  `:264` feeds it the tokenizer's id space (~248070 via HF tokenizer), while the
  model's logits width is 248320 — the equality fails against a real model. Decide:
  pad the trie to logits width (padding ids permanently masked) or relax the check to
  trie ≤ logits with the tail force-masked. Also check the ban-string path against
  [PADnnnnnn] ids (type 5, unreachable but present). tokenizer.rs now exposes both
  sizes distinctly (chat-tok phase).
  From: Deferred from the fork bootstrap (2026-07-28).

- [ ] [small] **Qwen3.8's tokenizer adds seven ids the embedded tokenizer does not know.**
  248070-248076 (`<|audio_start|>`, `<|audio_end|>`, `<tts_pad>`, `<tts_text_bos>`,
  `<tts_text_eod>`, `<tts_text_bos_single>`, `<|audio_pad|>`) exist in 3.8's
  tokenizer.json and not in the vendored 3.6 one; base vocab and merges are identical,
  so text tokenizes the same and only these ids are affected. Unresolved: whether the
  text-only checkpoint can emit one at all (its lm_head covers the padded 248320 rows
  either way), and what `decode` does with it if it does — the likely answer is an empty
  string or a lossy replacement, silently, mid-reply. Cheapest honest fix if it ever
  matters is not a second 12.8 MB embed but treating unknown-but-in-range ids as a stop
  or a logged anomaly. Reopen if a 3.8 reply ever ends strangely for no visible reason.
  2026-08-26: Qwen3.8-Flash-Next ships this exact tokenizer (hash-verified: base
  identical, added tokens through 248076), so the qwen4exp port arc makes a third
  checkpoint carry these seven ids — the question stops being 3.8-27B-only and the
  answer should be settled once, in that arc's P4, for all of them.
  From: Deferred from the Qwen3.8-27B + API-naming arc (2026-08-14).

- [ ] [unpriced] **The cards' recommended penalties (presence_penalty 1.5) are not implemented, and
  the reason is the speculative verify path, not laziness.** The official model cards
  recommend `presence_penalty` 1.5 for instruct (non-thinking) mode on ALL THREE
  checkpoints, and ALSO for thinking mode on the 35B-A3B alone — the 27B and 3.8-27B
  thinking recommendations say 0.0. Sources: HF README.md of Qwen/Qwen3.6-27B (~lines
  633-639), Qwen/Qwen3.6-35B-A3B (~661-667), Qwen/Qwen3.8-27B (~250-255);
  generation_config.json carries NO penalty keys, so anyone reading only the config
  files misses this entirely. Not implemented because (1) the sampler has no penalty
  machinery at all (`repetition_penalty` and `min_p` are likewise absent), and (2) a
  penalty makes the target distribution history-dependent, which entangles speculative
  decoding: the batched verify forward (`forward_all_logits`) scores every draft
  position in one pass, and each position's distribution would need the penalty applied
  over ITS history prefix — per-position penalty state, on both the drafted and the
  plain arm, or `--draft` and `--no-draft` sample from different distributions and the
  spec-equivalence gate is broken by design. llama.cpp does carry penalties through its
  verify, so there is a reference when this is taken; it is sampler + verify + gate work
  as one unit. Until then the OpenAI dialect accepts and DROPS
  `presence_penalty`/`repetition_penalty`/`min_p` (decisions.md "Serving" for why
  dropping sampling params is acceptable where dropping template kwargs is not), and the
  35B-A3B's thinking-mode sampling is the one place the shipped defaults knowingly
  deviate from the full card recipe. Related but separate: the 3.6 pair's cards list a
  third "thinking, precise coding" set (temp 0.6 / top_p 0.95 / top_k 20) — not
  auto-selectable (nothing in a request says "coding"), achievable as an explicit
  `--temp 0.6`, recorded here so nobody rediscovers it as a gap.
  From: Deferred from the chat-dialect and sampling-defaults arc (2026-08-19).

- [ ] [small] **Flash-Next's unconsumed presence penalty, template divergences and audio specials.**
  Three P4 leftovers. `Model::recommended_presence_penalty()` returns the card's 1.5 for
  non-thinking Flash-Next and **nothing consumes it** — threading the request's
  resolved checkpoint through openai/native/anthropic prepare is the same
  wiring needed to stop accept-and-dropping request penalties (the 2026-08-19
  penalties item in this section); the parity-harness fixes are «The parity harness
  cannot run on qwen4exp» in "Parity, provenance and tooling" and the drafter is
  «Flash-Next still ships no drafter, and `supports_drafting()` stays false» in
  "Drafting"; the embedded chat template is Unsloth-modified and diverges
  from `reference/chat_template-qwen38.jinja` for **tool calls, the developer
  role, multiple leading system messages and `effort=high`** (plain chat and
  thinking render byte-identical, which is why P2 could ship on it); and the
  checkpoint's tokenizer adds seven audio/TTS specials at 248070-248076 that
  the embedded 3.6 tokenizer does not carry — harmless for text, unhandled.
  From: Qwen3.8-Flash-Next port (decided 2026-08-25, blocked on release + upstream).
  Promoted from the item «Port Qwen3.8-Flash-Next» on 2026-09-06; dated 2026-08-29 there.

## Research candidates

- [ ] [blocked] **Port Qwen3.8-Flash-Next.** A PORT, not a registry entry — QSA sparse
  attention and the n-gram embedding subsystem are new. The closed narrative of this item
  is in the archive under its `From:` heading; the parts that stay open are their own
  items across the area sections, each naming that same heading.
  From: Qwen3.8-Flash-Next port (decided 2026-08-25, blocked on release + upstream).

- [ ] [blocked] **Same-machine MLX arm for the 35B-A3B comparison.** Landscape research (2026-08-30,
  session xwen-da): xwen's Flash-Next numbers have no public Apple Silicon peer (best
  published same-chip: llama.cpp 33.0 decode on a smaller IQ4_XS; MLX 4-bit needs
  ~163 GB and does not fit 128 GB), but the 35B-A3B class IS contested — MLX 4-bit on
  M4 Max measures ~91 decode (one Qwen3.5 sweep says 130). Before ever claiming a lead
  there, run mlx-lm on THIS machine with the closest 4-bit build of Qwen3.6-35B-A3B,
  same prompts, thermal protocol. Sources and the full engine survey are in the session
  log only; re-research before citing (aggregator-tier numbers were discarded as
  unreliable).
  From: Deferred from the landscape research (2026-08-30).

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

- [ ] [unpriced] **Per-resource barrier scoping and dependency-filtered cross-encoder fence waits**
  (candle patch). candle's `auto_barrier` emits a whole-scope barrier over the full
  window since the last one (`encoder.rs:104-149`), and every new encoder waits on every
  live fence rather than only the ones it depends on. This is the same pair the
  2026-08-08 prefill-residual entry left standing as the unconfirmed remainder after
  `XWEN_CHUNK_SYNC` and the cadence sweep both cleared — and that entry's closing
  condition was "do not re-propose without an instrument that can see inside a chunk",
  which a patched candle IS. Estimate unknown; the residual it targets is +350-560
  µs/token on 27B prefill.
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
