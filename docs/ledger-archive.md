# Ledger archive

Closed text of [TODO.md](../TODO.md), moved here verbatim. The section headings are the
ledger's own headings from before it was regrouped by area on 2026-09-06 (the arc that
deferred each item), so an item's `From:` line in TODO.md and a section name quoted
anywhere both resolve here. Since the regroup the unit that moves is the item or the
lettered sub-item, and each moved block opens with a one-line note saying what it is and
where its open remainder lives. Retired items go under `Retired: <area>` headings with
the reason and the reopen condition on their retirement line. Nothing here is planned; the
live ledger is TODO.md, and a retired item is picked up again whenever its reopen line
says so. Text is never deleted: it moves here.

## FIRST: why is Flash-Next so far from its ceilings, and rewrite this ledger from the answer (2026-09-05)

1. **Measure achievable bandwidth** so bytes-vs-time arguments have a real peak: a
   Metal kernel streaming a multi-GB buffer, batched dispatches per sync, several
   repetitions, read and read+write. Record the figure in AGENTS.md and retire the
   "never measured" caveat there and in the benching rules.
2. **Decode budget on today's binary.** Count dispatches per token per stage from the
   code (not from old profiles), and price the floor: dispatch count × the 8.41 µs fit
   (re-fit it if the candle rev or the encoder cadence changed). Then bytes per stage
   from the tensor tables. Budget = floor + bytes/measured-bandwidth + the serial scan;
   the residual against 22 ms is what nobody has explained. Use amortized benches or
   GPU timestamps; the sync-bracketing profilers rank, they do not price.
3. **Prefill budget at the 2048 chunk on today's binary.** Per-stage time from
   amortized runs or GPU timestamps, plus a bytes-moved audit per layer per chunk
   (activation passes, weight re-reads, readbacks). Decompose the ffn glue: router,
   rescale chain, SwiGLU, combine, shared expert, hc down/up — each with its own
   number. The 2026-08-30 composition finding stopped at "the glue is the majority".
4. **Rewrite the perf ledger below from the two budgets.** Every live decode and
   prefill item gets re-ranked by measured share of the gap, items the budgets show to
   be small are annotated down (never deleted), and the structural levers the budgets
   expose get items with a priced upper bound: for decode, fewer launches per token
   (whole-block fusion, or candle's per-dispatch locking on the pinned rev) rather than
   per-kernel bandwidth; for prefill, whole-chain glue fusion and the hc weight
   re-read at chunk granularity. Expected from the composition already known:
   +15-20 tok/s decode if the launch floor or count halves, +20-30% prefill from the
   glue — both UNPRICED until step 2 and 3 exist.

Entry points: `XWEN_STACK_PROFILE` / `XWEN_GDN_PROFILE` / `XWEN_PLE_PROFILE` for stage
names only; the amortized bench pattern in `src/ops/*` `#[ignore]` tests (e.g.
`ops::ple::tests::ple_tail_bench`) for pricing; `scripts/bench.ts` and the 2026-08-30
FFN-glue log entry for the interleaved end-to-end protocol; TODO.md items (14) and (15)
below for the dispatch-count facts to re-verify. Thermal protocol per AGENTS.md.

[Section prose of the pre-regroup ledger, moved verbatim on 2026-09-06.]

Opened by the owner after the PLE device tail shipped. Every perf item below was priced
bottom-up from a stage profile; nobody has worked top-down from what the machine can do.
The arithmetic says the gap is large on both axes, and the ledger has no item that
explains it, so the diagnosis is the next unit of work, not another kernel.
**RESULT, same day — steps 1-3 DONE, step 4 is the section that follows this one.**
Bandwidth 537-565 GB/s; decode 6.33 GB and 1740 dispatches per token, bytes-only ceiling
81-86 tok/s; prefill 13.7 TFLOP/s at 3851, expert gemms 14-43% of wall. Full record:
[ceiling-diagnosis.md](records/ceiling-diagnosis.md), plus decisions.md "Ceilings"
and decisions.md "Achievable bandwidth is MEASURED".
**The ceilings, as estimated 2026-09-05 (SUPERSEDED by the result above — kept as the
record of what was assumed; every number here was wrong by 1.3-2.5x).** A decode token moves ~2.5-3 GB of weights
(~1.5 GB Q4_K experts, ~0.7 GB Q8 hyper-connection, the rest attention/GDN/PLE/lm_head);
at 22 ms/token that is ~120-140 GB/s effective against a 614 GB/s part whose achievable
GPU read bandwidth has never been measured here (one shipped kernel was priced at
510 GB/s amortized). Bandwidth-only bound: 180-220 tok/s; we run 46. Prefill needs
~9-10 GFLOP/token for ~4-5B active parameters; at 1140 tok/s @3851 that is ~10-11 TFLOP/s
against a dense gemm kernel that runs 28-36 TFLOP/s in isolation on this machine.
Gemm-only bound: ~3000 tok/s; realistic with unavoidable glue 2000-2500; we run 1140.
**Working hypotheses to confirm or refute, one per axis.** Decode: the binding
constraint is dispatch latency, not bytes — ~1000 dispatches/token (576 MoE, 288 GDN,
the rest attention/PLE/QSA/hc glue, assembled from stale per-stage counts) at the fitted
8.41 µs/dispatch floor is 8-9 ms of the 22 ms, plus ~1.4 ms of genuinely serial GDN scan.
Prefill: the binding constraint is unfused, memory-bound glue — the ffn stage is "mostly
not the expert gemms" (2026-08-30), six elementwise passes over the routed activations
per layer, ~40 GB of hc weight re-reads per 2048-token chunk, 14 non-gemm dispatches
per MoE layer — but the last full prefill stage table is from 2026-08-29 at 530 tokens,
BEFORE the hc fusion and the 2048 chunk, so every share in it is stale.
**Steps.**

## Flash-Next perf ledger, re-ranked from the measured budgets (2026-09-05, step 4)

7. **Per-kernel bandwidth work on the big planes: ≈ 0.** The 28 MB gemv streams at
   95-97% of a pure read. (14)'s "+11-15 tok/s if the layer reached bandwidth" stays
   withdrawn; the vendored mv path for hc (P3 (3)) is bytes-at-rate already.

**Prefill (3.41 s @3851: expert gemms 0.46-1.44 s bracketed, of which the 0.30 s of
weight re-reads is a floor INSIDE that time rather than a separate term; ~0.25 s hc
activation traffic estimated; GDN chunked scan, attention and glue ranked only).**
4. **Weight re-reads: 9%, structural.** Every expert is touched per 2048-token chunk
   (~40 rows each), so a chunk reads the whole 82.5 GB trunk. A 4096 chunk would halve
   it in principle, but 4096 was MEASURED SLOWER on 2026-08-30 (745 vs 824 tok/s at
   2048; decisions.md "The prefill chunk is per architecture"), so chunk size alone does not recover it.
5. **Dispatch count: <1%.** Nothing to gain from launch-count work at prefill.

[Section prose of the pre-regroup ledger, moved verbatim on 2026-09-06.]

Every item below exists elsewhere in this file; this section is the ranking, priced
from the log's decode and prefill budgets, and it supersedes the ranking prose in the
P3 ledger and the prefill-chunk section without deleting either. Prices: a removed
decode dispatch ≈ 4 µs (≈0.02% of a token), a removed sync ≈ 0.3 ms (≈1.4%), bytes at
537-565 GB/s. Ceilings that may be quoted: decode 81-86 tok/s bytes-only at the
measured rate (we run 47); prefill 2300-3000 tok/s gemm-only at 28-36 TFLOP/s (we run
1129-1140 @3851).
**Decode (21.3 ms/token = 12.0 bytes + ~7.4 dispatch fixed cost + 0.9 syncs + 1.0 scan
beyond its bytes; the fixed-cost term is the residual, attributed at ~4 µs/dispatch).**

[Closed parts of the item «Hyper-connection carrier: 672 dispatches/token (35% of all launches), the largest population», moved verbatim on 2026-09-06; its open remainder is in TODO.md under "Decode performance".]

   **(e) MEASURED and CLOSED 2026-09-06 (pinned cf7c579): the 2..8 window is the largest
   win the gate has.** Medians 149.7 vs 93.2 tok/s at n = 8, 108.9 vs 69.5 at 4, 68.1 vs
   38.6 at 2, i.e. +57-76%; nothing above n = 8 has been measured.
   [Record](records/hc-gate-ragged-and-probe-decode.md).

[Closed parts of the item «MoE FFN: 576 dispatches/token (30%)», moved verbatim on 2026-09-06; its open remainder is in TODO.md under "Decode performance".]

   **PROBED 2026-09-06 in decode mode (`XWEN_DUP_DECODE`), and the shape of the item
   changes.** The shared expert floors at 0.43 ms of a 19.65 ms token (2.2%); the router
   projection prices at zero, which at decode means it overlaps itself and runs at low
   occupancy, never that it is free.
   **REFUTED by reading, same day: this ledger's own "fold the router projection into
   `kernel_moe_router`" (−1/layer)** — that kernel is one threadgroup per token. So the
   −192 is the shared expert alone (−4/layer) and the router's lever is a wide-grid gemv.
   [Record](records/hc-gate-ragged-and-probe-decode.md).
   **DONE 2026-09-06, same day (b7cd358, plus 0ed20ea for the fused|classic host line):
   the shexp fusion landed at −4/layer, −192/token on Flash-Next and −160 on the 35B, ON
   BY DEFAULT — 35B 113.2 → 115.0 tok/s (+1.6%), Flash-Next 51.2 → 51.5 (+0.6%), a fifth
   of the +3.5-4% this item priced; both checks pass.**
   [Record](records/fused-moe-shared-expert.md).

   (c) The review fix commit LANDED 2026-09-06 as 2c56d16: cross-path bound 1e-6 / 2e-6
   measured through the planed route, every admitted shape and partition dispatched, all
   bindings offset, predicate asks alignment / plane bound / routed width.

   **DONE 2026-09-06, same day (24c4069): the router gemv landed ON BY DEFAULT, the
   largest decode lever of the day — 35B 115.1 → 127.0 tok/s (+10.3%), Flash-Next 50.5 →
   52.9 (+4.8%), both checks pass; 127.0 is the new 35B plain-decode figure and the
   schema-v12 `router_mv` pin is LOAD-BEARING (the router runs before the top-k choice).**
   [Record](records/router-gemv.md). The review fix commit LANDED 2026-09-06 as
   `moe: router review fixes`, the commit after ed39b70.

[Closed parts of the item «Expert gemm efficiency: 14-43% of wall, bracketed by two in-situ A/Bs», moved verbatim on 2026-09-06; its open remainder is in TODO.md under "Prefill performance".]

   **BUILT AND RUN, same day (ab43499, `XWEN_DUP_STAGE`): expert gemms 0.96-1.09 s of
   3.4 s (28-32%), MoE glue 0.40, hc gates 0.39, GDN 0.23, shared expert ~0, 38% unpriced.
   The bracket is settled at its upper half and the "minority" reading refuted.**
   [Log](log.md#2026-09-05--duplicate-dispatch-probe-prices-flash-next-prefill-in-situ-expert-gemms-109-s-of-342-s-3851-32-moe-glue-040-hc-gates-039-gdn-023-shared-expert-0).

[Trimmed from the open item «Hyper-connection carrier: 672 dispatches/token (35% of all launches), the largest population» on 2026-09-06: closed (e), the 2..8 window.]

   (e) the 2..8
   token window now runs the fused gate's numerics (closer to the oracle than the QMatMul
   window it replaces; `XWEN_HC_GATE_FUSED_MAX_N=1` restores) — a deliberate change,
   recorded in decisions.md, not a bug — but its THROUGHPUT there is UNMEASURED: every
   A/B was n = 1, and at 2..8 tokens kernel A re-stages the carrier row per token per
   threadgroup. An A/B pinning `XWEN_HC_GATE_FUSED_MAX_N=1` against 8 on a serve-style
   ragged forward would price it (Qwen review, 2026-09-06).

[Trimmed from the open item «Expert gemm efficiency: 14-43% of wall, bracketed by two in-situ A/Bs» on 2026-09-06: probe build note, probe now run.]

   **FIRST, build the instrument that settles the
   bracket: an in-situ duplicate-dispatch probe** — a presence switch that encodes a
   stage's kernels twice (the expert gemms first, then the hc gates, the GDN chunked
   scan) so the wall delta IS that stage's in-situ time; no math change when unset,
   Flash-Next replay check anyway. Duplicate only the kernel dispatches, never the
   surrounding block — re-running the router, the gathers or an allocation would put
   their cost into the delta too (Qwen review, 2026-09-05).

## Priority order (decided 2026-07-28; P1-P9 shipped by 2026-07-29)

[Sub-item (c) of the open item «The DFlash drafter's per-token cache sync, its unre-derived draft-ctx horizon, and the deferred ring-buffer cache» shipped 2026-09-06, moved verbatim. Its open remainder — (b), the per-token cache sync — stays in TODO.md under "Drafting".]

   - **(c) `DEFAULT_DRAFT_CTX` (8192) was NOT re-derived and its inherited rationale
     is now half wrong.** Laguna's argument was O(depth) drafter forwards plus
     collapsing proposal quality with depth. The O(depth) half no longer holds: every
     sidecar layer but the last is windowed (2048 on the 27B, 4096 on the 35B) and
     `attention` narrows the cache to the window, so only one layer of five or six
     grows with the context. The memory argument stands (40 KiB/token on the 27B,
     48 on the 35B, imaged per cache slot). Re-derive by measuring drafter cost and
     acceptance at 4k/8k/16k/32k on the Qwen sidecars before changing it.

1. **Mechanical fork — DONE 2026-07-28.** cp-based copy of ../laguna, maxuna→xwen
   rename, MAXUNA_*→XWEN_* env prefix, Qwen tokenizer/chat-template/configs vendored
   into reference/, dflash KEPT (reversal — see log 2026-07-28). Gate passed:
   `cargo build` green (verified independently), zero maxuna references, src/scripts
   byte-identical to laguna modulo rename+rustfmt (proven by re-derivation diff).
   Expected carryover: 6 tokenizer tests assert Laguna vocab (P5 owns them);
   config.rs still asserts arch "laguna" (P2 owns it); scripts/ref-dump.sh +
   build-llamacpp.sh point at the absent laguna llama.cpp fork (P7 owns them).

2. **config.rs + gguf.rs retarget — DONE 2026-07-28** (see log entry; interval-driven
   layer pattern, real-header cross-checks, ENDOFTEXT hardcode rationale in config.rs).
   Original scope: Parse `qwen35`/`qwen35moe` metadata into a
   `XwenConfig` with `LayerKind::{Full, Linear}` per layer (`(i+1) % 4 == 0` rule;
   honor `qwen35.attention.recurrent_layers` array if present), reject other archs.
   Loader name table per CLAUDE.md cheat sheet. Traps: no ffn_norm (post_attention_norm
   is pre-MLP), no ssm_in (attn_qkv/attn_gate), `ssm_dt.bias` suffix, `ssm_a` no
   suffix, double-width attn_q. Keep ExpertStack (256 experts → one allocation) and
   dual-storage attention planes as-is. Validate rope.dimension_sections == [11,11,10,0]
   and error otherwise.

3. **DeltaNet reference implementation — DONE 2026-07-28** (linear_attn.rs frozen
   oracle; forms pinned in decisions.md "Model math"; hand-computed + streaming +
   ordering tests green). Original scope: New module (linear_attn.rs): composed candle
   ops, recurrent form only, fp32 state, exactly llama.cpp delta-net-base.cpp
   autoregressive semantics (see CLAUDE.md cheat sheet for the update equations). This
   is the frozen oracle — correctness first, speed irrelevant. Includes conv-state
   handling (last kernel−1 columns of the fused stream) and the gated RMSNorm ordering
   (norm → ×weight → ×silu(z)). Unit-test against hand-computed small cases AND against
   llama.cpp-dumped activations once P7 lands.
   (a) Prefill via the recurrent form is O(T) sequential small ops and will be slow;
   acceptable for bring-up. Chunked scan is P8.

4. **Full-attention + MoE layer adaptation — DONE 2026-07-28** (637 lib tests green;
   flash.metal unreachable at head dim 256 → prefill uses materialized-mask sdpa, see
   the deferred item; rollback trail memory cost recorded in decisions.md). Original
   scope: Attention: strided q/gate split
   (per-head interleaved), QK-norm [256], partial NEoX rope n_rot 64 theta 1e7 (rope
   tables only over 64 dims; dims 64..255 pass through), sigmoid output gate before
   o_proj, uniform causal masking (no SWA anywhere — flash.metal's in-kernel mask path
   simplifies). MoE: softmax→top8→renorm router (keep the 6.1e-5 clamp), drop laguna's
   sigmoid/bias/scale router path, shared expert via ffn_*_shexp + scalar sigmoid gate.
   Dense 27B FFN: plain SwiGLU (already exists as DenseMlp). model.rs: per-layer
   Full/Linear dispatch; KV cache only on full-attn layers; recurrent state (conv +
   delta, fp32) on linear layers with checkpoint/rollback/snapshot/export mirroring the
   KV cache's machinery (spec decode and the prefix cache depend on all four).

5. **chat.rs rewrite (ChatML) + tokenizer swap — DONE 2026-07-28** (all five template
   vectors byte-exact; 20k-conversation differential fuzz vs the reference jinja found
   zero divergences; constrain trie width bug found+fixed; design calls recorded in
   decisions.md). Original scope: Port the official template per the
   decisions.md entry; keep content/structure separation. Fixtures: the rendered test
   vectors from the bootstrap research (minimal, thinking on/off, historic-thinking
   stripping, parallel tool calls + grouped responses). Typed errors for the template's
   raise cases; reject vision content items. Gen loop: two stop ids, open-think
   seeding, think-split by token id 248069.

6. **hub.rs + CLI repoint — DONE 2026-07-28** (`--model-size 27b|35b`, 35b default;
   filenames verified against the HF API; drafter constants on the dflash sidecars,
   still opt-out). Original scope: Default repo/files → ggml-org Q4_K_M per CLAUDE.md;
   `--model-size 27b|35b` (or similar) selector; drafter constants → dflash sidecars
   (wired but inert until P9). Sampling defaults 1.0/0.95/20; stop ids from
   gguf/generation config.

7. **Parity harness vs upstream llama.cpp — DONE 2026-07-28** (see log entry;
   oracle pinned at llama.cpp `e9fa0781`, floors + drift profile + tap table in
   docs/parity.md). Both checkpoints agree with upstream: Track A shows smooth
   monotonic drift with no cliff, and Track B passes every tier at floors an order
   of magnitude tighter than laguna's (strict 0.9998, mm 0.999). The 27B's first
   forward was correct — no bring-up bisection was needed. Original scope: Build
   ggml-org/llama.cpp master (pin the commit in parity.md), repoint
   scripts/parity-gate.ts + logits-dump taps at the qwen35 graphs, recalibrate all
   tier floors on the Q4_K_M checkpoints, fill in parity.md's TBDs.
   (a) **Track A cannot localize inside a layer.** The tap set is the inherited
   laguna one (attn_norm / mixer out / ffn_inp / ffn_norm / ffn_out / l_out), so a
   divergence resolves to a layer and a stage, not a sub-op. The Qwen graphs expose
   far more: DeltaNet core out + `new_state` + the conv/beta/alpha/gate chain,
   `ffn_moe_logits`/`ffn_moe_weights_norm`, `shared_expert_gate{,_sigmoid}`,
   `Qcur_normed`/`Kcur_normed`/`gate_sigmoid`. Adding them needs tap plumbing in
   `linear_attn.rs` / `moe.rs` / `attention.rs` (model-math files, deliberately not
   touched during the harness work). The llama.cpp names to match are tabulated in
   docs/parity.md "Tap names".
   (b) **`provenance.flash` is a label, not a fact, on these checkpoints.** It is
   env-derived, and `flash.metal` is compiled at head dim 128 while Qwen 3.6 is 256,
   so the candidate reporting `flash: "fused"` actually ran candle sdpa with a
   materialized mask. Consistent, so no gate is weakened, but the field cannot be
   read as evidence the flash kernel ran. Fix when flash is instantiated at BD 256
   (pairs with the deferred prefill-mask item below).
   (c) **The strict tier is near-vacuous on the dense 27B.** With no routed experts,
   `--moe-impl reference` and `--moe-impl fused` run the same `DenseMlp`, and the
   strict env pins everything else classic on both sides — hence the measured
   bitwise 1.000000000. It confirms determinism, not expert-kernel fidelity. The
   27B's real signal is the mm/decode/ppl tiers. Consider a dense-specific strict
   variant (e.g. reference = f32 attention, candidate = classic mv only) if the
   dense path ever needs its own regression detector.
   (d) **The `_Q8` widenings were not recalibrated, and one of them is load-bearing.**
   `NORM_RATIO_MAX_Q8` (1.5) and `NEAR_TIE_MARGIN_Q8` (1.0) fire on every Qwen
   candidate (`attn_decode == "q8"`), but their derivation is laguna's measured
   1.3075 l2 ratio and 0.848 logit swing. Measured on the 35B decode dumps
   (2026-07-28): the worst per-step l2 deviation across all three fixtures is
   **1.0211** — the 1.5 band has ~24x more headroom than needed and could be tightened
   toward ~1.06 (still ~3x margin) to actually catch a scale bug. The near-tie window
   is the opposite case: text-mixed step 15 excused a mismatch at **0.5567** below the
   reference top1, i.e. it needed the widened 1.0 and would have hard-failed at the
   standard 0.5. So tighten the l2 band; leave the near-tie window at 1.0 and record
   0.5567 as its anchor. Both need more than three fixtures' worth of evidence first.

12. **MTP exploration — DEFERRED by decision.** Sidecar reuses parent arch, one extra
    full-attn block + nextn.* tensors, plain KV cache, eh_proj over
    [norm(emb);norm(h)]. Evaluate as drafter only after P9 lands or fails (see
    decisions.md "Speculative decoding").
    - ANNOTATED 2026-07-29: P9 landed, and the trigger is still not met. DFlash's
      acceptance is 85-95% — a better drafter would not help. What limits xwen's
      speculation is the verify forward's cost (P9a) and the drafter cache sync
      (P9b), and an MTP drafter would pay both identically. Do not open this until
      P9a lands and the win is measured with a fast verify.

[Section prose of the pre-regroup ledger, moved verbatim on 2026-09-06.]

Live items as of 2026-07-29, by value: **P9(a)** K-snapshot fused verify (the
unlock for spec decode's real 27B win — measured 39 ms/verified-position on the
reference-scan fallback today); the **P8c prefill residual** (+350-560 µs/token
outside every measured stage, now the largest 27B-prefill unknown; diagnosis
first, in-situ per-layer timing in `run_stack`); the **P8c attention glue**
(lowest risk first step: route the main attention block through the existing
bit-identical `attn_gate`/`permute_01`/`cast_*` kernels, then re-measure).
P9(b) drafter inject fusion and P10 serve adaptation follow.
UPDATE (later 2026-07-29): **P9(a) is DONE and `--draft` flipped to opt-out** —
see the P9 annotations and log.md. Live items now, by value: the **P8c prefill
residual**; the **P8c attention glue**; the **verify round's ~149 ms fixed
cost** (new, "Deferred from the K-snapshot verify pass" below); P9(b); P10.
UPDATE 2026-08-08: the **P8c prefill residual** was diagnosed, not fixed — it is
real (+410-438 µs/token reproduced), it is not inside any stage (per-stage syncs
find only +103 of it), and both cross-chunk accumulation and command-buffer
batching are refuted as its mechanism; it is now blocked on an instrument that
can count barriers and fence waits inside a chunk. The **P8c attention glue** is
DOWNGRADED — its premise was inverted (the glue kernels are already wired in) and
its ~42 ms/layer bounty never existed. Live items by value: the **verify round's
~149 ms fixed cost**; P9(b); P10; the attention glue's surviving remnant (a fused
sigmoid gate, ~2-3 dispatches) and the head-dim-256 flash instantiation.
UPDATE 2026-08-08 (later): the **verify round's ~149 ms fixed cost** is RESOLVED for
its dominant term — `mul_mv_ext` shipped and measured, verify forward 0.40x at span 2,
27B drafted decode +11.6-13.2%; ~89 ms of intercept remains with two named
non-coverages. The **`p_min`/`pause_margin` retune** is consequently UNBLOCKED and is
now the best-motivated live item (the controller already shifted behavior on its own:
27B pauses 16 vs 28 on code, 14 vs 32 on chat). Live items by value: the retune; P9(b);
P10; attention-projection `mv_ext` coverage; the attention glue's surviving remnant and
the head-dim-256 flash instantiation.
UPDATE 2026-08-08 (later still): both of the top two are DONE. **Attention-projection
`mv_ext` coverage shipped** (verify forward −12.0% at span 8, −5 to −6% at spans 4-6,
a wash at span 2), and **the retune ran** — `draft_p_min` is now per-checkpoint, 0.5 on
the 27B (+11-13% over the shipped 0.3 within-sweep) and 0.3 on the 35B-A3B, with
`pause_margin` confirmed at a shared 1.0 by its first real sweep. Live items by value:
P9(b) drafter inject fusion; P10 serve adaptation; the ~89 ms verify-forward intercept
(now with its two named non-coverages resolved, so it needs a fresh decomposition
rather than another subtraction); the P8c prefill residual, still blocked on a
barrier/fence instrument; the attention glue's surviving remnant and the head-dim-256
flash instantiation. New small items from this pass: the span-2 Proj window floor
option, and `DEFAULT_DRAFT_CTX`, which the retune deliberately did not sweep.
UPDATE 2026-08-09: **`xwen batch` shipped** — a surface arc rather than a perf one, so
the live perf order above is unchanged. It does move P10: the serve tree now owes a
`/xwen/v1/batch` endpoint on top of its template adaptation, and the batch core was
written transport-agnostic so that endpoint is a handler, not a port. Its own deferrals
are in "Deferred from the batch + scored-classification arc (2026-08-09)" below; the one
with parity implications is the missing Track-B case for snapshot-replay-vs-scratch,
today an at-ship manual A/B.
UPDATE 2026-08-11: **`/xwen/v1/batch` shipped, ahead of the rest of P10** — and with
it the engine became checkpoint-aware: every request names its model, `--model` is only
the default, the engine swaps lazily with one model resident at a time (log.md
2026-08-11, decisions.md "Serving"). The endpoint did not need the dialect adaptation
P10 was gating on, because the batch core renders its own prompts — P10's remaining
scope (ChatML tool-call parsing in the dialect layers, thinking semantics, prefix-cache
snapshots carrying recurrent state) is unchanged. New deferrals in "Deferred from the
serve batch + multi-checkpoint arc (2026-08-11)" below.

[Closed parts of the item «DeltaNet Metal kernels — (a) DONE 2026-07-28, (b) still open», moved verbatim on 2026-09-06; its open remainder is in TODO.md under "Prefill performance".]

   - **(a) SHIPPED, and it covered prefill too:** four kernels in `src/ops/delta.metal`
     take a DeltaNet layer to 8 dispatches at any sequence length (was ~65 per decoded
     token), and on the 35B-A3B decode goes 57.8 → 91.2 at 596 tokens and prefill
     305 → 2183 (7.15x). Kill-switch `XWEN_DELTA_CLASSIC=1`.
     [Record](records/fused-deltanet-kernels.md), decisions.md "Model math".

   - The refuted re-decomposition is kept runnable, not deleted:
     `XWEN_DELTA_SCAN_V2=1` selects `kernel_delta_scan_v2` (llama.cpp's shape) plus
     the `kernel_delta_l2norm` dispatch it needs, on the `XWEN_MOE_DUAL` precedent.

   - **The fused scan is bounded, not bit-identical**, so `XWEN_DELTA_CLASSIC=1` is
     now pinned on BOTH sides of the strict parity tier and a `delta` provenance
     field (parity_schema v6, grandfather "classic") proves which path each dump ran.
     Cached pre-v6 reference dumps stay valid. docs/parity.md "Provenance pins".
   - **Greedy output is not preserved at longer prompts, by construction.** At 596
     prompt tokens fused and classic produce byte-identical greedy output; at 1929
     they share 69 words and then fork at a near-tie. That is the expected
     consequence of reassociated f32 sums and is what the decode tier's near-tie rule
     exists to grade — it is not a kill-switch bug.
   - The fused path requires head dim 128 (both checkpoints) and a Metal device;
     anything else silently keeps the reference scan. A `seq > 1` chunk under an
     armed rollback checkpoint also stays on the reference scan (single tokens do
     not) — see decisions.md.

[Closed parts of the item «DFlash adaptation to the Qwen sidecars — ADAPTED 2026-07-29, but speculation is a 27B-only win and stays opt-in», moved verbatim on 2026-09-06; its open remainder is in TODO.md under "Drafting".]

   - **(a) The K-snapshot fused verify is the precondition for speculation to pay,
     not an optimization of it — the top open item under P9. DONE 2026-07-29,** built
     exactly as sketched below: verify marginal cost 9.42 → 3.57 ms/position (fixed
     ~171 → ~149 ms), end-to-end 27B +19.3-21.0% code, 35B +18.1-19.8% code (was
     -11.5/-12.7% — the pause controller stopped pausing, see (d)).
     [Log](log.md#2026-07-29--k-snapshot-fused-verify-lands-spec-decode-goes-from-single-digits-to-8-21-the-35b-flips-from--12-to-13-20-and---draft-becomes-opt-out).
     The retired fallback's successor items live under "Deferred from the K-snapshot
     verify pass (2026-07-29)".
     Original scope, kept: Under an armed
     rollback trail a multi-token chunk takes the frozen reference scan
     (linear_attn.rs:194-205), which walks tokens one at a time, so the 48-of-64
     (27B) and 30-of-40 (35B) DeltaNet layers get NO batching win inside a verify
     forward: measured 245 ms for a ~6-position 27B verify against a 43 ms plain
     step, i.e. 39 ms per verified position. That is why the gains are single-digit
     percent rather than the 1.39-2.29x reported elsewhere on Apple silicon. The
     structural provision is already in place: both scan kernels
     (`kernel_delta_scan`, `kernel_delta_scan_v2` in `src/ops/delta.metal`) hold each
     thread's slice of the state in registers across the whole timestep loop, so
     emitting the last K per-token states is one guarded store inside the loop plus a
     wider output buffer — mirror llama.cpp's `n_rs_seq + 1` most-recent-first
     snapshot planes (ggml-metal.metal, the `K > 1` branch of
     `kernel_gated_delta_net_impl`). Landing it retires the `seq > 1 && trail_armed`
     fallback, the ~1-2.3 GB verify-walk trail (decisions.md "Speculative decoding"),
     and P8b(b)'s last live rationale.

   - **(d) The `--draft` opt-out flip is DEFERRED with numbers.** The flip was
     conditional on the controller holding a never-materially-slower property on both
     checkpoints; it does not, because the 35B's 12% loss lands on rounds the
     controller has already paused (see (b)). Re-evaluate after (a) and/or (b): the
     bar is the 35B at or above plain, not merely closer to it.
     RESOLVED 2026-07-29: **flipped — drafting is now the default** (`--no-draft` opts
     out). (a) alone met the bar with margin (35B +18.1-19.8% code / +12.6-12.8% chat),
     and (b) stays open as a lever rather than the gate: its ~1.2 ms/token cache sync is
     only fatal on PAUSED rounds, and the controller stopped pausing.
     [Log](log.md#2026-07-29--k-snapshot-fused-verify-lands-spec-decode-goes-from-single-digits-to-8-21-the-35b-flips-from--12-to-13-20-and---draft-becomes-opt-out).

   - **`--draft` is not byte-identical to `--no-draft` in general, but the sampler
     stream is in lockstep.** `bun scripts/spec-equivalence.ts` runs two modes.
     Greedy: 11 of 12 comparisons match exactly, the twelfth forking on the 27B chat
     prompt at a near-tie, because the batched verify forward reassociates its f32
     sums differently from the single-token forward — same class as the
     fused-delta-scan divergence under P8a, not a verify-walk bug. Sampled
     (temperature 0.8, fixed seed, `p_min` 0, auto-pause off — the only mode that can
     see the RNG, since argmax never draws from it): the 35B is identical on both
     prompts over 360-435 drafted tokens and the 27B on the code prompt over 315, so
     the spec loop draws exactly as many times as the plain loop. The 27B chat prompt
     forks at line 12, deep enough to be the near-tie signature rather than a desync.
     Both modes refuse to pass a run that drafted nothing. Deliberately not a
     `cargo test` gate: near-tie forks are expected and would make it flaky.
   - ANNOTATED 2026-07-28, RESOLVED 2026-07-29: drafting was turned OFF by default
     (`DEFAULT_DRAFT_ENABLED = false`) because the inherited opt-out default aborted
     every zero-flag `xwen generate` and `xwen serve` at startup, and this item was
     to flip it back. It does NOT flip — see (d) for the blocking numbers. The
     CLI/config help text in `bin/xwen/main.rs` (`DraftArgs`, `ServeArgs`) and
     `serve/config.rs` (`DraftToml`, the `[draft]` `--init` template block) no longer
     says drafting is unavailable; it now gives the measured reason for opt-in.
   - CLOSED 2026-07-29: the three load blockers are gone. The shipped sidecars carry
     no `dflash.decoder_arch` key (the requirement is deleted), no `enc.aux_norm`
     (the per-tap norm-and-scale is deleted — the encoder is concat → `fc` →
     `enc.output_norm`, dflash.cpp:109-123) and no `blk.N.attn_gate` (the softplus
     output gate is deleted). `dflash::tests::real_file_load_and_shapes` and
     `real_file_bf16_alias_load_and_forward` were the suite's only red tests and are
     now green against both sidecars' real weights.
   - DONE 2026-07-29: `DRAFT_KV_BYTES_PER_TOKEN` in `serve/config.rs` derives from
     the new `hub::Model::draft_kv_bytes_per_token()` (40 KiB/token on the 27B's five
     drafter layers, 48 on the 35B-A3B's six), alongside
     `hub::Model::kv_bytes_per_token`.
   - ANNOTATED 2026-07-29: the K-snapshot plan for the verify walk's recurrent-state
     rollback is now (a) above, promoted from a nice-to-have to the item's top
     blocker by the verify-cost measurement.

[Closed parts of the item «serve adaptation», moved verbatim on 2026-09-06; its open remainder is in TODO.md under "Serve, batch and CLI".]

    - ANNOTATED 2026-08-19: **the thinking-flags half is now covered** (commits
      a2e02d0/205d9ba) — enable_thinking, preserve_thinking and the template
      `reasoning_effort` are all surfaced per dialect, with the Anthropic dialect
      deliberately having no per-request field (2026-08-19 deferred section).
      [Record](records/chat-dialects.md). What this item still holds open:
      nothing about thinking flags.

[Closed parts of the item «27B dense bring-up — MOSTLY DONE 2026-07-28 via P7», moved verbatim on 2026-09-06; its open remainder is in TODO.md under "Parity, provenance and tooling".]

    - The conv threadgroup-sizing worry (below, "the 27B linear-attn conv runs over
      10240 channels") turned out not to bind: the fused conv kernel is a flat
      one-thread-per-element launch through the shared `dispatch_linear` helper, so
      channel count only sets the grid size. Closed by construction, not measurement.

[Closed parts of the item «MoE block glue fusion — SHIPPED 2026-07-29», moved verbatim on 2026-09-06; its open remainder is in TODO.md under "Decode performance".]

    - **The two router matmuls stay candle dispatches** — MLX's `gemv_t` accumulation
      order is not reproducible from a differently-shaped hand-written gemv, and it
      depends on the output width, so concatenating the shexp gate row onto the router
      weight would have changed that gate's bits. Costs one dispatch of the ten saved.

    - **The dual-weight gate|up gather is built, bitwise, and switched OFF** — it
      measured slower (99.5 vs 102.8 tok/s). `XWEN_MOE_DUAL=1` opts in. See
      decisions.md "Refuted perf directions" for the mechanism; the short version is
      that merging two bandwidth-bound dispatches halves the threadgroup count and the
      memory-level parallelism with it.
    - **Still open on the MoE decode path:** the remaining 14 dispatches are 8 matmuls
      plus `ffn_norm`, the two router gemvs, the routed `silu_mul`, the fused router
      and the fused epilogue. The next real lever is the shared expert — its three
      q8_0 `QMatMul` projections plus its gate gemv are 4 of the 14, and nothing has
      priced whether they are worth a dedicated fused SwiGLU the way the routed
      experts got one.

[Trimmed from the open item «DeltaNet Metal kernels — (a) DONE 2026-07-28, (b) still open» on 2026-09-06: the original scope preamble, rewritten.]

- [ ] [measured] **DeltaNet Metal kernels — (a) DONE 2026-07-28, (b) still open.** Original scope:
   (a) fused recurrent decode step (one dispatch per layer
   per token; state stays resident, fp32); (b) chunked prefill scan, chunk 64,
   llama.cpp's chunked form as the spec (cumsum → tri decay mask → solve_tri →
   per-chunk state update) — needs tri-solve which candle lacks; vendored kernel.
   Kill-switches XWEN_DELTA_CLASSIC / XWEN_DELTA_CHUNK_CLASSIC falling back to the P3
   reference. Gate: bitwise-or-bounded vs reference per parity.md tiering.

[Trimmed from the open item «DFlash adaptation to the Qwen sidecars — ADAPTED 2026-07-29» on 2026-09-06: stale opt-in and 27B-only claim.]

   , but speculation is
   a 27B-only win and stays opt-in.

[Trimmed from the open item «DFlash adaptation to the Qwen sidecars — ADAPTED 2026-07-29» on 2026-09-06: stale pointer to archived (d).]

     Fixing either could flip `--draft` to opt-out; see (d).

[Trimmed from the open item «serve adaptation» on 2026-09-06: thinking flags and snapshots landed.]

    thinking-mode flags
    (enable_thinking / preserve_thinking) surfaced per dialect, prefix-cache + disk
    tier snapshots extended with recurrent state (48–96 KiB conv + 2–6 MiB delta per
    snapshot depending on model).

[Trimmed from the open item «Three leftovers from the MoE glue fusion: the residual add, the prefill combine, and `mul_mv_id_dual`'s unchecked ids» (until 2026-09-06 titled «MoE block glue fusion — SHIPPED 2026-07-29») on 2026-09-06: the shipped headline, retitled away.]

- [ ] [measured] **MoE block glue fusion — SHIPPED 2026-07-29.** An MoE layer went from 24
    dispatches per decoded token to 14 (960 → 560 across the 40 layers) and 35B-A3B
    decode from 92.6 to 102.8 tok/s (+11.0%), on three fusions all bit-identical to the
    candle chains they replace, behind `XWEN_MOE_GLUE_CLASSIC=1`, with both parity gates
    passing at pre-change numbers.
    [Record](records/fused-moe-glue.md), decisions.md "Kernel policy".

[Trimmed from the open item «The 27B's interactive generate/chat smoke run was never done, and its numbers carry a ±10% spread» (until 2026-09-06 titled «27B dense bring-up — MOSTLY DONE 2026-07-28 via P7») on 2026-09-06: the mostly-done headline, retitled away.]

- [ ] [small] **27B dense bring-up — MOSTLY DONE 2026-07-28 via P7.** The parity gate ran the
    27B end to end: first forward correct, all gated tiers pass (strict is
    near-vacuous on the dense model — see P7c). Remaining: an interactive
    generate/chat smoke run, decode/prefill perf numbers for the 27B (nothing
    measured yet; 64 layers dense will be much slower per token than the A3B), and
    the deferred conv threadgroup-sizing check when P8 lands.

[Trimmed from the open item «The DFlash drafter's per-token cache sync, its unre-derived draft-ctx horizon, and the deferred ring-buffer cache» (until 2026-09-06 titled «DFlash adaptation to the Qwen sidecars — ADAPTED 2026-07-29») on 2026-09-06: the adapted headline, retitled away.]

- [ ] [measured] **DFlash adaptation to the Qwen sidecars — ADAPTED 2026-07-29; (b), (c) and
   (e) still open.**

[Item closed 2026-09-06 as already shipped: the `<function=...>` tool-call parser landed 2026-07-28 (log.md "Serve integration fixes: the tool-call parser was reading `:` and `;` as span markers") and lives in `src/serve/engine.rs`; the item was stale.]

- [ ] [unpriced] **serve adaptation.** Tool-call parsing for the `<function=...>` XML-ish format in
    both API dialects (string args raw, non-string JSON) — the one piece of the original
    scope still not adapted (AGENTS.md "serve"). Estimated-prefill scheduling unchanged.
  From: Priority order (decided 2026-07-28; P1-P9 shipped by 2026-07-29).

## Deferred from the P2-P4 model-core retarget (2026-07-28)

- [ ] **The DeltaNet rollback trail costs one retained delta state per verify
  token per layer.** `LayerCache::Linear` records the state after every token
  while a checkpoint is live (llama.cpp's K-snapshot-slots equivalent), which at
  block_size 16 is 16 x 2 MiB x 30 layers ~= 1 GB on the 35B, 16 x 3 MiB x 48
  ~= 2.3 GB on the 27B, held only for the duration of a verify walk. Measure it
  against the spec-decode win when P9 lands; a chunked scan (P8) that can replay
  a prefix cheaply would let the trail be dropped entirely.
  ANNOTATED 2026-07-29 (P9a): the footprint is unchanged in magnitude but changed
  in shape — the trail's delta entries are now unmaterialized views into one
  `[seq, v_heads, 128, 128]` snapshot buffer per layer per verify forward
  (~48 MiB/layer at 16 planes on the 27B) instead of per-token materialized
  copies; still walk-scoped, dropped at rollback. The chunked-scan replay
  rationale is dead (P8b refuted + P9a landed). Spec decode's win is now
  double-digit on both checkpoints, so the memory is earning its keep; measure
  only if verify walks ever run concurrent per-seq in serve.

## Deferred from the sampler-tail pass (2026-07-28)

- [x] **xwen's top-p convention follows candle, not llama.cpp.** RESOLVED
  2026-07-29: switched to the llama.cpp/HF convention. `truncate_top_p` now
  renormalizes the top-k survivors and keeps the shortest prefix whose cumulative
  mass reaches `top_p`, crossing token included (`cum_sum >= top_p`, llama.cpp's
  comparison); `top_p >= 1.0` is a no-op as it is there. `min_keep` is not
  carried — llama.cpp's default is 0. Sampled outputs changed, accepted. See
  decisions.md "Top-p renormalizes over the top-k survivors". The perf half of
  this item did NOT ship and is restated as its own entry below.
  Original context, verbatim: The cut is
  measured against full-vocabulary probability mass and is skipped entirely when
  the top-k set holds less than `top_p` of the total; llama.cpp and HF both
  renormalize over the k survivors first and therefore trim in cases where xwen
  does not. Preserved deliberately through the perf rewrite (decisions.md
  "Thinking budget and sampling controls"), but it is a real divergence from the
  project's declared ground truth for everything else, and `--top-p` does not
  mean what a llama.cpp user would expect. Decide it as a semantics question:
  switching is a few lines and removes the need for the full-vocabulary softmax
  on the fast path (the k-wide softmax would drop the readback to whatever the
  selection needs), so it is also worth ~0.1 ms. Needs a decision, not a patch.
  Second reason to want the switch (2026-07-29): comparing absolute mass is what
  makes the truncation sensitive to which backend ran the softmax. The fast path
  softmaxes on the device and the `SampleControl` path on the CPU, and an input
  sitting within an ulp of the threshold can therefore truncate differently on
  the two. Renormalizing over the k survivors never compares absolute mass, so
  the whole boundary question dissolves rather than being documented around.
- [x] **The WP-G1 expert-gather comment block in moe.rs still quotes laguna's
  numbers.** `moe.rs` above `tiled_stack_dt` reasoned from "~2.4 GB over 47
  layers" and "~13.7 ms/token (LPM)" — measurements of laguna's geometry, left
  in place because replacing them means inventing numbers nobody has taken at
  Qwen width. Re-run `moe_decode_ffn_bench` (its constants are correct now) and
  rewrite the block from what it reports.
  - RESOLVED 2026-07-29 (MoE-glue arc). The block now states the byte floor,
    which is arithmetic and needs no bench: at 35B-A3B geometry (hidden 2048,
    expert_ff 512, top_k 8) the three q4_K projections gather ~14 MB per layer
    and ~570 MB per token across the 40 MoE layers. The laguna timing claims are
    gone rather than replaced — the ~365 GB/s lm_head anchor they were compared
    against is kept as the reference point, so the benches still say what a
    reading means without asserting a measurement nobody has taken here.

[Duplicate of the open item «Top-k selection still crosses the bus at full vocabulary width.», folded and moved verbatim on 2026-09-06.]

- [ ] [measured] **The fast path still softmaxes the full vocabulary it no longer needs.**
  Split out of the top-p convention item when that resolved (2026-07-29). The cut
  now renormalizes over the k survivors, which is arithmetically a k-wide softmax,
  so nothing downstream of the selection depends on the full-vocabulary
  denominator any more — the ~0.1 ms it was worth is unclaimed only because the
  selection itself still runs CPU-side over the whole row. Pairs with «Top-k selection
  still crosses the bus at full vocabulary width» in this section: land that and the
  device softmax collapses to the candidates along with the readback.
  From: Deferred from the sampler-tail pass (2026-07-28).

[Item closed 2026-09-06 by decision: xwen keeps the HF/vLLM order — decisions.md "Thinking budget and sampling controls", the paragraph on sampler order.]

- [ ] [unpriced] **Temperature is applied before the top-k/top-p cut; llama.cpp's default
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

[Item shipped 2026-09-06 with the presence-penalty arc (log.md "2026-09-06 — Presence penalty: the cards' recipe, through the speculative verify, on by default"): top_k 0 is no cut, 1 is greedy. Moved verbatim.]

- [ ] [unpriced] **`top_k = 0` means greedy here, "top-k disabled" in llama.cpp.** The sampler
  maps every `top_k <= 1` to argmax, where llama.cpp treats `k <= 0` as a no-op
  filter (the whole vocabulary stays eligible). Pre-existing, harmless at the
  default of 20, but the serve layers forward client-supplied values verbatim, so
  a llama.cpp-reared client sending `top_k: 0` gets deterministic output instead
  of unrestricted sampling. Surfaced by outside-model review 2026-07-29. A
  semantics decision, like the temperature-order one settled on 2026-09-06 (decisions.md
  "Thinking budget and sampling controls"), not a bug fix.
  From: Deferred from the sampler-tail pass (2026-07-28).

## Deferred from the dense-FFN prefill gemm pass (2026-07-29, P8c)

- [x] **The dense-FFN gemm diff shipped WITHOUT an independent review pass** — the
  only arc of 2026-07-29 that did (an agent-spawning moratorium was in effect;
  every other arc got a two-model-family review). Both parity gates pass and the
  test suite is green, so this is process debt, not a known defect. When it runs,
  the author's own pointers at the two places a subtle bug could hide: (1)
  `src/ops/dense_mm.metal` is `src/ops/f16_t.metal` with exactly ONE intended
  substitution (block-quant tile dequant replacing the half widen-copy in A-tile
  staging) — any other divergence between the two files is a bug; (2) the dequant
  sub-block indexing (`xb = base + k_pos/(16*nl)`, `il = (k_pos/16) % nl`) assumes
  each dequant call returns the 16 contiguous elements at super-block offset
  `[16*il, 16*il+16)` — pinned by a test against `QTensor::dequantize`, but an
  off-by-one here is the classic quant-kernel failure and deserves adversarial
  eyes.
  RESOLVED 2026-07-29: the two-model-family review ran (Claude + Codex
  `gpt-5.6-sol` at xhigh, both adversarial, both read-only) and found ZERO
  correctness bugs at any severity. Pointer (1) holds semantically, not
  literally: the kernel body is line-identical to `f16_t.metal` outside A-tile
  staging, but beyond the claimed substitution the file also adds the required
  template-ization plus Q8_0/Q5_K/Q6_K dequantizers alongside Q4_K — a
  support-matrix addition, not a divergence in executable behavior. Pointer (2)
  verified two independent ways: Claude diffed the four block structs and
  dequant functions verbatim against `reference/llama.cpp` (ggml-common.h,
  ggml-metal.metal) and Codex re-derived the nibble/scale arithmetic from
  scratch (il = 4g + r case analysis, get_scale_min_k4 6-bit packing, the
  folded `d/16` high-nibble path); the empirical pin remains
  `dense_q4k_matches_oracle_production_shapes`. Secondary probes (seq-gate
  boundary 32/33, ragged tiles, buffer offsets, threadgroup sizing, barrier
  placement, pipeline-cache keying) all clean. One doc nit found and fixed:
  the `dense_mm.metal` header said `seq >= DENSE_MM_MIN_SEQ` where the gate is
  strictly `>`.

- [ ] **The host-built materialized causal mask is not today's problem but becomes
  first-order at 8k+.** `flash.metal` is genuinely unreachable at head dim 256
  (`ops::flash_attn` hard-bails at `head_dim != FLASH_BD` (128), dispatch.rs:3324 and
  3361, and has zero production callers), so prefill runs candle sdpa against a
  host-built mask — a scalar `Vec<f32>` loop (kv_cache.rs:89-98) then uploaded, broadcast
  to 24 heads and cast to f16, to carry what is one bit per position. **REFUTED as a
  meaningful part of the 27B gap**, and the refutation turns on a detail worth not
  re-deriving: the mask is **HOISTED** — built once per chunk in `model.rs` `run_stack`
  and shared across all 16 full-attention layers, NOT rebuilt per layer. The profiling
  pass's own first run multiplied by 16 and produced a ~682 ms scare; corrected, it is
  51.22 ms, **0.37% of wall at 3851** and 0.15% at 880. Mask + sdpa together grow ~1.2
  percentage points across the two lengths against a 16% observed throughput drop.
  The 402 MB figure is DERIVED (Σ over chunks of `n_head × seq × k_seq × 2`), not
  measured; with the 51.22 ms measured time it implies ~8 GB/s, slow in isolation and
  irrelevant at 0.4% of wall. But both quadratic terms roughly quadruple from 4k to 8k,
  so this becomes first-order at long context. The real fix is the existing ledger item —
  make `flash.metal` reachable at head dim 256, removing the mask rather than making it
  cheaper. Pairs with the head-dim-256 flash item under P8.

[Section prose of the pre-regroup ledger, moved verbatim on 2026-09-06.]

All three come out of the per-stage prefill profile that root-caused the 27B gap.
They are what that profile found and did NOT fix; the profile itself is transcribed
in log.md 2026-07-29.

[Trimmed on 2026-09-06 from the item «Attention glue: ~10 unfused eager passes per layer, inside a 57.13 ms/layer attention block», since retitled «A fused sigmoid-gate kernel at the attention gate site, and the head-dim-256 flash instantiation» and living under Prefill performance: the downgraded glue premise, inverted. The block below is the item as it stood, checkbox and tag included.]

- [ ] [unpriced] **Attention glue: ~10 unfused eager passes per layer, inside a 57.13 ms/layer
  attention block.** MEASURED (profiling pass, amortized, T=3851): the whole attention
  layer is 57.13 ms, growing monotonically with position within the prefill (5.746 ms at
  a 512-token chunk at position 0 → 9.176 at position 3072). The brief that opened the
  arc put ~42.43 ms/layer of that in the glue — the permute/cast/gate copies around the
  projections and sdpa, each a separate candle dispatch over a full
  `[T, n_head, head_dim]` tensor — but that split appeared in the briefing rather than in
  the raw profile, so **re-measure the sub-term before sizing any work against it**.
  `ops::attn_gate` already exists and does exactly the fused-gate job, but it is wired
  only into the DFlash path. Two steps, in order: route the main block through the
  existing `attn_gate`/`permute_01`/`cast_*` kernels (bit-identical to the chains they
  replace, so this is `XWEN_ATTN_GLUE_CLASSIC` territory and needs no new parity tier),
  then measure what is left. Do not start by writing a new fused kernel — the existing
  ones may cover most of it.
  ANNOTATION 2026-08-08: **DOWNGRADED — the premise is inverted and the ~42 ms/layer of
  glue never existed** (the glue kernels have been wired into the main block since the
  fork; the attention block's length-growth is +53.5 µs/token, ≈ the sdpa quadratic).
  [Record](records/27b-prefill-residual.md).

[Duplicate of the open item «Prefill runs candle sdpa with a materialized mask, not the vendored flash kernel.», folded and moved verbatim on 2026-09-06.]

- [ ] [unpriced] **A fused sigmoid-gate kernel at the attention gate site, and the
  head-dim-256 flash instantiation.** The gate kernel is worth ~2-3 dispatches; the flash
  instantiation is what would remove the mask (the item «Prefill runs candle sdpa with a
  materialized mask, not the vendored flash kernel» in this section). Neither is sized
  against a measured bounty. These two are all that survived this item's original premise
  — ~42 ms/layer of unfused attention glue — which was DOWNGRADED 2026-08-08 as inverted;
  that narrative is in the archive. [Record](records/27b-prefill-residual.md).
  From: Deferred from the dense-FFN prefill gemm pass (2026-07-29, P8c).

## Deferred from the K-snapshot verify pass (2026-07-29, P9a)

- [ ] **The verify round's ~149 ms fixed cost is the new spec-decode ceiling — RESOLVED
  2026-08-08 for its dominant term; ~89 ms of intercept remains, see the closing
  annotation.** Fit
  over spans 2-32 on the 27B (`n_past` 512): ~149 ms fixed + 3.57 ms/position,
  against a ~40-43 ms plain step — the fixed part is ~113 ms above a plain forward
  and ~60% of a typical round, and it is NOT the DeltaNet scan any more. Candidates,
  none priced: checkpoint materialization (one conv+delta copy per layer on arm),
  rollback restore, the trail's host-side conv slices (~1 cat + seq materializes per
  DeltaNet layer), full-span logits computation + readback, command-buffer syncs.
  Price the stages before attacking any of them — this item is a profile, not a fix.
  ANNOTATION 2026-08-08: **the profile ran, and EVERY candidate listed above is
  refuted as the owner. The fixed cost is inside the verify FORWARD, and it is the
  dense FFN's matmuls running at small M.** Conditions: 27B Q4_K_M, `n_past` 512,
  `lowpowermode 0`, `target/release/spec-verify-bench` grown per-stage sync brackets
  plus per-span stack-profile dumps, medians over 20 reps.
  - **The armed machinery is cheap.** Checkpoint arm: 5.7 ms fixed — the ~157 MB of
    per-round materializes cost almost nothing. Rollback: 2.6 ms fixed + 0.43 ms/tok,
    and a keep-4 vs keep-0 branch comparison shows no difference. Full-span logits +
    readback: 0.12 ms + 0.099 ms/tok (a last-row-materialize variant reads a flat
    ~0.4 ms). Together they are a rounding error against ~149 ms.
  - **It is the forward itself, and it is present UNARMED.** Fit over spans 2-32 puts
    the forward's own fixed cost at ~161 ms, and a span-2 UNARMED forward measures
    152 ms against a ~40-43 ms plain step. Nothing about speculation causes this; a
    2-token forward is simply ~3.7x a 1-token forward.
  - **Stage decomposition (span-2 forward vs a plain seq-1 step, both stack-profiled
    under an identical sync regime):** dense FFN **131.8 vs 33.9 ms = +97.8 of the
    +111.7 ms total, 87.6%**; lm_head +4.4 (3.2x); mixer_delta +5.9; mixer_full_attn
    +2.7; every other bucket under 1 ms.
  - **Mechanism: candle's `mul_mm` collapses at small M.** At seq 2..=32 every
    quantized matmul takes the tiled path, whose grid degenerates to `ne01/64`
    threadgroups — ~73 GB/s effective, against ~280 GB/s on the seq==1 mat-vec path.
    Corroborated by two refutation rounds: forcing the vendored dense gemm onto small
    spans (`XWEN_DENSE_MM_MIN_SEQ=1`) moved the fixed intercept only −3.3 ms because
    the cooperative-tensor gemm has the SAME small-M occupancy collapse (its marginal
    did improve, 2.40 → 1.63 ms/tok), and `XWEN_MM_ID_MIN_SEQ=1` on the 35B was
    strictly WORSE at spans 2-8 (+4.1-4.4 ms). So this is not a threshold to retune —
    no kernel currently in the tree wants these shapes.
  - **Fix in flight (its own arc, which will document what ships):** vendor
    llama.cpp's `mul_mv_ext` multi-row mat-vec — dequantize once, reuse the result
    across 2-5 output rows (ggml-metal-ops.cpp:2120-2223, `ne11_mm_min` 8). By byte
    arithmetic it should win at spans 2-8 and wash by ~16.
    ANNOTATION 2026-08-08: **the kernels are IN and routed on by default**
    (`src/ops/mv_ext.metal`, q4_K/q6_K/q8_0 x r1ptg 2..5, window 2..=8,
    `XWEN_MV_EXT_CLASSIC` reverts, provenance `mv_ext` at schema v8). Correctness is
    gated by oracle tests; the THROUGHPUT claim above is still unmeasured — no model
    has been run against it. See "Deferred from the small-batch mat-vec pass" below
    for what the measurement owes and what the window still does not cover.
  - **This inverts the retune item below.** "Longer drafts amortize better" was
    reasoned off a dominant fixed cost; if `mul_mv_ext` lands, short spans get
    cheaper and the fixed cost stops dominating, so the tuning conclusion has to be
    re-derived rather than carried. Cross-referenced there.
  - `src/ops/dispatch.rs:330-334` documents this exact gap (ggml's `mul_mv_ext`
    kernels for ne11 2..8, "not vendored — see TODO.md"). Its pointer was dangling —
    no such item existed. **It resolves here.**
  ANNOTATION 2026-08-08 (round 6, supersedes the "still unmeasured" note nested under
  the fix-in-flight bullet above): **`mul_mv_ext` shipped and the dominant term is
  gone.** Verify forward on the 27B at `n_past` 512, default vs `XWEN_MV_EXT_CLASSIC=1`,
  interleaved, 2 reps/arm means: span 2 **61.45 vs 153.44 ms (0.40x, −92.0)**, span 4
  85.87 vs 176.91 (0.49x), span 6 125.89 vs 197.97 (0.64x), span 8 **161.16 vs 220.11
  (0.73x, −59.0)**. Spans 12-32 match between arms within 1.2-4.2% — the window is
  2..=8 and above it the ext path is inactive. End-to-end drafted decode (P9a protocol,
  greedy, `-n 128`, 3 reps, medians): 27B code **31.7 vs 28.4 tok/s (+11.6%)**, 27B chat
  **30.9 vs 27.3 (+13.2%)**; 35B code 131.1 vs 125.8 (+4.2%), 35B chat 119.5 vs 119.5
  (+0.0%, a real dead heat — that cell is pause-dominated at 25-26 of 44 rounds). The
  35B's verify gain is only 3.2-4.3 ms at spans 2-8 and zero beyond, as predicted: just
  its shared expert and lm_head route through `QLinear`.
  - **Caveat on the span-2 point estimate.** The default arm's per-rep spread was large
    and one-directional (rep 1 faster by 15-30%, the known pattern, biggest yet).
    Bounded by the per-rep extremes the span-2 win is **−87.5 to −96.4 ms** — sign and
    magnitude survive; only the point estimate is soft.
  - **Cross-round caveat.** The classic arm on this binary reads ~9-15% slower at
    mid-spans than round 3's binary did (fixed intercept 172.9 vs 161.0 ms). Different
    binaries, machine-state variance. Only within-round ratios are trustworthy.
  - **What remains: ~89 ms of fit intercept at the spans-2-8 arm**, and two known
    non-coverages explain part of it. The attention projections are NOT in the window —
    on the default path they are f16 or q8_0 planes (`ops::matmul_f16` / `matmul_q8`),
    never `QLinear` — and the single-row lm_head goes through `forward` rather than
    `forward_all_logits`. Both are ledgered under "Deferred from the small-batch
    mat-vec pass" below. Anything beyond those needs a fresh decomposition.
  - See log.md 2026-08-08 "`mul_mv_ext` ships", decisions.md "The small-batch matmul
    window routes from ONE decision point", docs/parity.md "Provenance pins" for the
    `mv_ext` field.
  ANNOTATION 2026-08-08 (later the same day): **one of the two named non-coverages is
  closed, and what it collected off the intercept is SPAN-DEPENDENT rather than a flat
  subtraction.** The attention/DeltaNet projections joined the window
  (`Proj::DenseF16Q8`); measured against a HEAD binary on the same bench, it took
  **−21.0 ms at span 8, −8.5 at span 6, −4.3 at span 4, and nothing at span 2** (+1.4,
  which is a wash inside the arm-ordering bias — see that item for why). So it flattens
  the arm's slope more than it lowers its intercept, and the ~89 ms figure — which is a
  fit intercept, i.e. the extrapolation to span 0 — is not reduced by 21 ms or by any
  single number. This is consistent with the earlier finding that **~40 ms of the
  intercept is ordinary per-forward fixed cost** (a plain seq-1 step is 40-43 ms), and
  it means the projections were never intercept: they were per-token weight re-reads,
  which is exactly what the displaced gemv does. Remaining named non-coverage: the
  single-row lm_head bypass, which is a strict-tier anchor and is closed-by-analysis
  under "PART A of the brief" below rather than open work. Anything further needs a
  fresh decomposition against the new arm, not another subtraction from this one.
- [x] **`p_min` 0.3 and `pause_margin` 1.0 were tuned against the reference-scan
  cost curve and are now stale — DONE 2026-08-08. Swept, and `p_min` is now
  PER-CHECKPOINT: 0.5 on the 27B, 0.3 on the 35B-A3B; `pause_margin` stays a shared
  1.0, confirmed by its first real sweep.** Two independent 120-run sweeps of the new
  `scripts/retune-draft.ts`, machine otherwise idle, `lowpowermode 0` recorded at start
  and end of each. Winners replicated in BOTH runs on every knob that moved.
  - **27B `p_min` 0.5** — mean-of-medians 37.3 / 37.2 tok/s against 33.0 / 33.5 at the
    shipped 0.3 and 36.0 / 36.5 at 0.7; +46-52% over plain (24.9-25.3). Mechanism: at
    0.5 the chat prompt stops pausing entirely (13-18 paused rounds at 0.2/0.3 → 0) and
    acceptance goes 57% → 78%, taking that cell 29.4 → 36.8-36.9. The code cell already
    ran pause-free at 0.3 in five of six reps and moves only 36.5-37.6 → 37.5-37.9.
  - **35B `p_min` stays 0.3** — 127.9 / 128.4 against 125.2 / 125.3 at 0.5, i.e.
    installing the 27B's winner globally would have cost the 35B ~2.5%. Its cheaper
    target forward still profits from drafting deeper at lower acceptance.
  - **`pause_margin` 1.0** — 35B: 129.2 / 128.7, ahead of 0.8 and 1.2 in both runs. 27B
    at p_min 0.5: a genuine wash, 1.0 and 1.2 within 0.1 tok/s in both runs with the
    runs' nominal winners disagreeing (1.2, then 0.0) across a ~0.5 tok/s spread —
    expected, since the controller never pauses at that floor. **This was the first time
    `pause_margin` was actually swept**; P9 validated 1.0 against 0.0 only.
  - **Installed:** `Model::draft_p_min_default()` (src/hub.rs), one const arm per
    checkpoint; `DraftArgs.draft_p_min` is `Option<f32>` resolved through it; serve
    resolves it through a new `CliOverrides.model_size`; `DEFAULT_DRAFT_P_MIN` deleted;
    `SpecParams::default()` documented as a base every real caller overwrites. Tests:
    `hub::tests::the_drafting_floor_is_per_checkpoint`,
    `serve::config::tests::draft_p_min_defaults_per_checkpoint`. Suite green, 722 + 69.
  - **Note on the sweep's own conflict text.** Both raw sweep logs print "the constants
    made per-model, which is a TODO.md item, not a retune". No such ledger item ever
    existed — a dangling pointer of the same class as `dispatch.rs:330-334`'s. It is
    moot rather than resolved: `draft_p_min` now HAS a per-model home, and the script's
    recommendation block was rewritten to point at `hub.rs` for `p_min` and the three
    shared sites for `pause_margin`.
  - **`DEFAULT_DRAFT_CTX` (c) was NOT swept** and still interacts; it stays open under
    P9(c).
  - The harness is the standing methodology now — protocol, the no-cell-reuse rule and
    the preserved P9 qualification criterion are in decisions.md "Measurement
    discipline". `SHIPPED_P_MIN` in the script must be edited alongside `hub.rs` or the
    next sweep grades against a status quo that no longer ships. See log.md 2026-08-08
    and decisions.md "Speculative decoding".
  Original context, verbatim: The curve they were fitted to (39 ms/position
  marginal) no longer exists; with 3.6 ms/position marginal and a dominant fixed
  cost, longer drafts amortize better and pausing is less often right (the 35B now
  pauses 0-of-20 rounds on code with the OLD tuning — the win may grow with a
  retune, and `DEFAULT_DRAFT_CTX` (c) interacts). Same protocol as the P9 tuning
  sweep: both models, both prompt kinds, interleaved, two independent runs.
  ANNOTATION 2026-08-08: **do not run this sweep until the small-M matmul work
  settles.** The item above found the ~149 ms fixed cost to be the dense FFN's
  matmuls at small M, and the `mul_mv_ext` fix in flight targets exactly spans 2-8.
  If it lands, the cost curve this retune would fit changes shape at the short end —
  short spans get cheaper and the fixed cost stops dominating — which reverses the
  "longer drafts amortize better" reasoning above. Retuning against today's curve
  would fit a curve that is about to move.
  ANNOTATION 2026-08-08 (later, round 6): **`mul_mv_ext` landed, the curve moved as
  predicted, and this sweep is now UNBLOCKED — it is also better motivated than before.**
  The block above is lifted. Short spans got much cheaper (verify forward 0.40x at span
  2, 0.73x at span 8), so this is the THIRD cost curve `p_min` 0.3 / `pause_margin` 1.0
  have been fitted against and wrong about: the reference scan's 39 ms/position, P9a's
  fixed-cost-dominated curve, and now a curve that is cheap at the short end and
  unchanged above span 8. The motivating evidence is that the controller ALREADY shifted
  behavior on its own: the 27B default arm pauses far less than the classic arm — **16
  vs 28 rounds on code, 14 vs 32 on chat** — and drafts more, without anyone retuning
  anything. That is the controller finding the new economics by accident, which is
  exactly the signal that its fitted constants are stale. Note also that the 35B chat
  cell is pause-dominated (25-26 of 44 rounds) and showed +0.0% — a pause-side retune is
  the most likely way to move it. Protocol unchanged from the P9 sweep: both models,
  both prompt kinds, interleaved, two independent runs; `DEFAULT_DRAFT_CTX` (c) still
  interacts. See log.md 2026-08-08 "`mul_mv_ext` ships".

[Section prose of the pre-regroup ledger, moved verbatim on 2026-09-06.]

All three come out of the measurement pass that closed P9(a); raw per-rep data in the
session logs referenced by log.md "K-snapshot fused verify lands".

[Closed parts of the item «Every verify arm goes superlinear at span 48», moved verbatim on 2026-09-06; its open remainder is in TODO.md under "Drafting".]

  Separate new observation from the same sweeps, unexplained and NOT arming-dependent:
  **lm_head roughly doubles at span 48 (7.0 → 13.1 ms)** in both the armed and the
  unarmed profiled runs. Recorded so a future contradiction has a trail.

[Item closed 2026-09-06 by decision, option (c): accept and document — decisions.md "Serving", the paragraph on plane-less slots.]

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

## Deferred from the small-batch mat-vec pass (2026-08-08)

- [x] **Nothing about this arc has been MEASURED — DONE 2026-08-08 (round 6).** The
  numbers exist and the docs it owed are written: `docs/log.md` 2026-08-08
  "`mul_mv_ext` ships", `docs/decisions.md` "The small-batch matmul window routes from
  ONE decision point" plus its two companion entries, `docs/parity.md` "Provenance
  pins" for the `mv_ext` field. Headline: verify forward 0.40x at span 2 (61.45 vs
  153.44 ms) rising to 0.73x at span 8; drafted decode +11.6% / +13.2% on the 27B,
  +4.2% / +0.0% on the 35B. The protocol note was followed — interleaved A/B against
  `XWEN_MV_EXT_CLASSIC=1`, and the full measurement is annotated on the "~149 ms fixed
  cost" item above, including the two variance caveats that came out of it (a large
  one-directional per-rep spread in the default arm, and a ~9-15% cross-round shift in
  the classic arm's own baseline).
  Original context, verbatim: Correctness is pinned by oracle
  tests against `QTensor::dequantize` at production reduction lengths, but no model
  has been run: the predicted spans-2-8 win, the crossover against candle's `mul_mm`,
  and the effect on the verify round's ~161 ms forward are all still arithmetic. The
  measurement owes `docs/log.md` a dated entry and `docs/decisions.md` a "The
  small-batch matmul window" entry — deliberately NOT written yet, because both want
  the numbers. Protocol notes: this is a decode-adjacent path, so use the interleaved
  A/B (`XWEN_MV_EXT_CLASSIC=1` is the other arm) and calibrate against the classic
  arm's known baseline before believing absolutes (CLAUDE.md "Benching rules").
- [x] **The window's upper edge is inherited, not measured — CLOSED-REFUTED 2026-08-08.
  It is measured now, and 8 is the right ceiling.** `XWEN_MV_EXT_MAX_SEQ=32` makes
  spans 16 / 24 / 32 **worse than classic by 1.11x / 1.42x / 1.69x**, with span 12 a
  wash at 0.98x. The degradation is monotonic in span — the multi-row mat-vec stops
  paying once the token count fills the tiled path's threadgroup grid — so ggml's
  `ne11_mm_min` 8 envelope was not merely untested inheritance and the default window
  stays 2..=8. Do not re-raise the ceiling without a kernel that changes that shape;
  recorded in decisions.md "Refuted perf directions". The K-quant divergence the item
  asked the measurement to check is retained deliberately and is now written up as a
  decision (decisions.md, "Two deliberate divergences from ggml's own gating") rather
  than left as an open question.
  Original context, verbatim: 2..=8 is ggml's tested
  envelope; whether the kernel also beats candle's `mul_mm` at 9..32 (the rest of the
  verify-span range, which is where the fixed cost actually lives) is open.
  `XWEN_MV_EXT_MAX_SEQ=<n>` raises the ceiling without a rebuild; above 8 the plan
  uses r1ptg 4, which is xwen's extension rather than ggml's tuning (ggml aborts
  there). Note ggml additionally restricts its K-quants to ne11 4..=8 while xwen
  routes them from 2 — deliberate, because our fallback at 2..3 is the 73 GB/s
  `mul_mm` rather than ggml's tuned alternative, but it is another inherited-gate
  divergence the measurement should check.
- [x] **Attention projections are NOT in the window — DONE 2026-08-08 (later the same
  day). They are now, and the verify forward at span 8 fell 12.0%.** `Proj::DenseF16Q8`
  routes seq 2..=8 to the already-vendored q8_0 `mul_mv_ext` over a `QuantPlane` VIEW on
  the same buffer and `base_off` the gemv used, so the coverage costs no extra memory.
  One `Proj` variant reaches seven tensors on every layer of both checkpoints:
  `attn_q`/`attn_k`/`attn_v`/`attn_output` on the full-attention layers (16 of the 27B's
  64) and `attn_qkv`/`attn_gate`/`ssm_out` on the DeltaNet layers (the other 48, via the
  same type from `linear_attn.rs`). The `mv_ext_window` plan and `mv_ext_supported` are
  threaded verbatim so `XWEN_MV_EXT_CLASSIC` reverts this site identically; added on top
  is a 16-byte activation-alignment guard (the ext kernel reads the activation as
  `float4`, the gemv takes any offset). Two documented asymmetries:
  `XWEN_MV_EXT_MAX_SEQ` cannot widen this site past 8, because the enclosing
  `Q8_DECODE_MAX_SEQ` arm already sends seq > 8 to the dense f16 plane; and the
  alignment guard is a per-call fallback the env-derived `mv_ext` provenance field
  cannot see, which is fine only while every production activation here is offset-0 —
  a strided caller must be preceded by recording what ran.
  - **Measured** (27B `spec-verify-bench`, `n_past` 512, `lowpowermode 0`, interleaved
    against a HEAD-commit binary built in a scratch clone, 5 reps/arm pooled from two
    A/B sessions, medians): span 8 **175.20 → 154.19 ms (−21.0, −12.0%)** with
    non-overlapping per-rep ranges (165.9-183.4 vs 145.2-159.3), span 6 140.39 → 131.92
    (−6.0%), span 4 87.66 → 83.32 (−5.0%), spans 12-48 unchanged.
  - **Span 2 reads +1.4 ms (+2.3%) and that is a WASH, not a regression.** The
    interleave put the coverage arm second in every pair, and at spans 12/16/24 — where
    the kernel provably cannot run — the second arm still reads slower in all five pairs,
    pairwise medians +2.3% / +2.0% / +1.6%. The span-2 pairwise median is +2.8%, the same
    magnitude, over a much wider spread (−11.9% to +5.6%). The
    mechanism that would explain a real span-2 loss (at t=2 only one gemv pass is saved
    and the fixed nsg=2/nxpsg=8 geometry may not pay for it) survives as a hypothesis
    and is ledgered below as its own item.
  - **The original item's size estimate was read off the wrong bucket AND the wrong
    span.** It sized the opportunity as `mixer_full_attn`'s +2.7 ms of +111.7, but three
    quarters of the tensors this site reaches sit in the DeltaNet layers, which that
    profile charged to `mixer_delta` (+5.9); and that whole profile was a SPAN-2
    comparison, where this change is measurably a wash. The displaced gemv costs one
    full weight pass per token, so the opportunity grows with span — which is why the
    number worth quoting is span 8's 21 ms and not anything derived from the span-2
    stage profile.
  - Reviews: Claude (no findings, independently confirmed the guard arithmetic and that
    no production activation is strided) and Codex gpt-5.6-sol (no Critical/High; 2 Low
    + 1 Nit, all fixed). Both parity gates ALL PASS at pre-change numbers; no schema
    change (the `mv_ext` field records an env-derived mode, not a site list, so v8
    stands). See log.md 2026-08-08 "the small-batch window reaches the attention and
    DeltaNet projections", decisions.md "The small-batch matmul window routes from ONE
    decision point" (EXTENDED clause) and its accuracy entry's QUALIFIED clause.
  Original context, verbatim: The brief assumed
  `QLinear::forward` would catch them; it does not, because on the default path the
  attention weights are dense f16 planes (`ops::matmul_f16`) or raw q8_0
  (`ops::matmul_q8`), never `QLinear` — only the `XWEN_ATTN_F32` parity path uses
  `QLinear`, and that one must keep QMatMul. So at spans 2..8 every attention
  projection still re-reads its weights once per token (`F16_MM_MIN_SEQ` 8 and
  `Q8_DECODE_MAX_SEQ` 8 both send that range to a gemv). The stage profile put
  `mixer_full_attn` at only +2.7 ms of the +111.7, so this is small — but ggml
  instantiates `kernel_mul_mv_ext_f16_f32_r1_*` and the q8_0 variant is already
  vendored here, so both are cheap to add if the measurement says the window helps.
  ANNOTATION 2026-08-08 (round 6): **the measurement says the window helps, so the
  conditional above is discharged and this is now a live cheap follow-up.** The window
  is worth 0.40x on the verify forward at span 2 and +11.6-13.2% on 27B drafted decode,
  so extending the same treatment to the attention projections has a measured
  motivation rather than a hypothetical one. It stays SMALL by the same stage profile
  (`mixer_full_attn` was +2.7 ms of +111.7) — size it against that before spending much
  on it. Cheapness is unchanged: the f16 ext variant exists in ggml and the q8_0 one is
  already vendored here. Part of the ~89 ms of intercept still unaccounted for on the
  fixed-cost item above.
- [ ] **PART A of the brief (multi-row plain mv at the lm_head) was NOT done, and
  the reason is that the site cannot use it.** The brief called for extending the
  vendored plain mat-vec from seq==1 to 2..=3 at the lm_head bypass
  (`XwenModel::forward`). That bypass always operates on ONE row — `forward` narrows
  to the last position before the projection — so `run_plain_mv` would never see the
  multiple rows the byte arithmetic was about; the `seq == 1` condition there selects
  a PHASE, not a row count. Relaxing it would only switch a prefill/verify chunk's
  last-row logits from QMatMul to the vendored gemv, and that is the exact tensor the
  strict tier compares (`result_output`), so it would move a bitwise anchor for no
  bandwidth gain. The genuinely multi-row lm_head is in `forward_all_logits`, and
  that one now takes `mul_mv_ext` at spans 2..8 — which is the better path anyway
  (one weight pass for 2..5 rows beats 2..3 re-reads). Nothing to do unless the
  premise changes.
- [ ] **q5_K has no ext kernel** (sanctioned in the brief). ggml instantiates one; no
  supported checkpoint stores a weight in q5_K on a path this kernel serves — the
  retired unsloth UD file's experts were the only q5_K, and experts go through the
  mm_id/mv_id gather, not here. Add only if such a checkpoint returns.

[Section prose of the pre-regroup ledger, moved verbatim on 2026-09-06.]

The `mul_mv_ext` port itself shipped (kernels, routing, kill-switch, provenance v8,
oracle tests). These are the pieces it did not carry.
UPDATE 2026-08-08 (round 6): the measurement landed — the first two items below are
closed by it, and the docs it owed are written. The rest stand.

## Deferred from the batch + scored-classification arc (2026-08-09)

- [x] **No `/xwen/v1/batch` HTTP endpoint. DONE 2026-08-11.** The core
  (`batch::run_batch`) is transport-agnostic — it takes a `Generator` and a request
  struct, and the CLI subcommand is a thin stdin/stdout wrapper around it — so serving
  it is a handler plus a dialect decision (native only, or an OpenAI-batch-shaped
  alias). Deferred behind P10: the serve tree still needs its Qwen template adaptation
  (tool-call parsing, thinking semantics, recurrent-state prefix snapshots), and adding
  an endpoint to a dialect layer that has not been adapted yet means adapting it twice.
  Nothing about the core needs to change when it lands.
  OUTCOME: shipped native-only, ahead of the P10 gate — it turned out not to need the
  dialect adaptation at all, because the batch core renders its own prompts and never
  touches the dialect layers. The prediction held: the core changed only by growing
  hooks (progress callback + cancellation poll), and the endpoint is a handler
  (`serve/batch.rs`) plus a second `Job` variant. Same document both transports; the
  request's `model` is honored per request and the engine swaps checkpoints lazily.
  Log.md 2026-08-11, decisions.md "Serving".
- [x] **`escape` is opener-level and formatting-confounded for bare literals — DONE
  2026-08-11.** Forced by the first external client (multi-field first fields pinned at
  0.999-1.000, mean escape stuck at 1/fieldCount; their one-token-early hypothesis was
  checked and refuted — the mass was ` true`/` false`, the answer in space-led
  spelling). Shipped as the first candidate refinement grown to the whole row:
  `escape_mass` classifies every encodable id by decoded text (whitespace-stripped for
  unquoted fields, verbatim for quoted; pure-whitespace tokens excluded and
  renormalized away), via `Generator::last_probs` + `LagunaTokenizer::decoded_vocab`.
  First-field escape 0.9999 → 0.00197 measured, scores bit-identical. The
  sequence-level escape (the second candidate) remains unbuilt and unneeded so far.
  decisions.md "Batch" (2026-08-11) has the full story.

[Section prose of the pre-regroup ledger, moved verbatim on 2026-09-06.]

`xwen batch` shipped with its prefix cache, `include_score` scored assembly and the
nine-item demo (log.md 2026-08-09, decisions.md "Batch"). These are the pieces it
deliberately did not carry.

## Deferred from the fork bootstrap (2026-07-28)

- [x] **Decide what to do with laguna's parity fixtures and tests/parity.rs — DONE
  2026-07-28 (P7).** `tests/fixtures/parity-prompts.json` regenerated with Qwen ids
  from the oracle's own `llama-tokenize --no-bos` (fixture `long-swa` renamed
  `long-mixed`: there is no sliding window in Qwen 3.6, so the long fixture now
  stresses the DeltaNet recurrence instead); the Laguna `reference-ppl.json` deleted
  and replaced by per-checkpoint `reference-ppl-<basename>.json`;
  `committed_ppl_reference_fixture_stays_valid` retargeted to validate every
  per-checkpoint fixture present. `tests/parity.rs`'s comparison plumbing was
  model-agnostic and needed only the floor recalibration — no gutting required.
- [x] **The 27B linear-attn conv runs over 10240 channels at hidden 5120 — CLOSED
  2026-07-28 (P8a), no sizing problem.** The fused `kernel_delta_conv` is a flat
  one-thread-per-output-element launch through the same `dispatch_linear` helper the
  other glue kernels use (up to 256 threads per group, bounds-checked tail), so the
  channel count only sets the grid extent. Both conv widths (10240 on the 27B, 8192
  on the 35B) are covered bitwise by `conv_matches_reference_bitwise`.

- [x] **flake.nix description still says "maxuna engine"** — DONE 2026-07-28, the
  fork agent renamed all three occurrences (description + two rationale comments).

[Trimmed from the open item «`glance` the copied scripts/ for maxuna-isms» on 2026-09-06: the P7 half already done.]

  Partly
  done 2026-07-28 (P7): `hf.ts` repointed at the two ggml-org repos with a
  `--model-size`-style selector, `parity-gate.ts` / `parity.ts` / `ref-dump.sh` /
  `build-llamacpp.sh` / `bench.ts` retargeted.

[Trimmed from the open item «The 35B's perplexity delta grew with the fused DeltaNet scan and the floor's margin shrank» on 2026-09-06: the RESOLVED floor-derivation half.]

  `PPL_NLL_DELTA_MAX = 0.002` was derived as
  `max(3 x measured, 0.002)` from a measured 0.000511; the fused scan moved that to
  **0.000791** (27B: 0.000221 → 0.000330). The gate still passes with ~2.5x headroom,
  but 3 x 0.000791 = 0.00237 now EXCEEDS the constant, so the recipe that produced it
  no longer reproduces it. **RESOLVED 2026-07-28 (parity owner): keep 0.002, do NOT
  re-derive from the fused measurement** — the recipe is a one-time floor-SETTING
  heuristic anchored to the reference-scan baseline, and re-fitting it to the change
  under test ratchets the bound outward forever and catches nothing.
  [Rationale and trip-wire](parity.md#perplexity-gate).

[Item closed 2026-09-06 as already shipped: `GrammarTrie::embedded()` builds at `LagunaTokenizer::PADDED_VOCAB` and a unit test pins the padded width above the tokenizer's; grammar-masked batch has run on the real model since 2026-08-09.]

- [ ] [small] **Qwen3.6 vocab is 248320 padded / 248077 real, and constrain.rs will trip on
  it.** `constrain.rs:90` asserts `tok_trie().vocab_size() == expected_vocab` and
  `:264` feeds it the tokenizer's id space (~248070 via HF tokenizer), while the
  model's logits width is 248320 — the equality fails against a real model. Decide:
  pad the trie to logits width (padding ids permanently masked) or relax the check to
  trie ≤ logits with the tail force-masked. Also check the ban-string path against
  [PADnnnnnn] ids (type 5, unreachable but present). tokenizer.rs now exposes both
  sizes distinctly (chat-tok phase).
  From: Deferred from the fork bootstrap (2026-07-28).

[Item shipped 2026-09-06, moved verbatim.]

- [ ] [small] **`glance` the copied scripts/ for maxuna-isms** beyond the mechanical rename
  (bench prompt fixtures, hardcoded model names, parity-gate assumptions). Still unswept:
  `classify.ts`, and `tests/fixtures/bench-prompts` (never opened).
  From: Deferred from the fork bootstrap (2026-07-28).

## Deferred from the serve batch + multi-checkpoint arc (2026-08-11)

- [x] **The batch route inherits axum's default request-body limit (~2 MB) — DONE
  2026-08-11.** A real batch tripped it (a 377 KB story split one batch into 14
  POSTs), which is exactly the condition this item deferred on. Now an explicit
  100 MB `DefaultBodyLimit` over the whole API router; the 413 stays the framework's
  (still not the native envelope — accepted, a client at 100 MB has bigger problems).
  decisions.md "Serving", log.md 2026-08-11 client-feedback entry.
- [x] **Startup drafter resolution still trusts `--model-size`, not the file. DONE
  2026-08-11 (same day, review fix).** `run_serve` resolved the official-sidecar path
  via the flag before the GGUF was ever opened, so `--model-size 27b -m <35b.gguf>`
  (or a config-file `model` disagreeing with the flag) selected the 27B sidecar for a
  35B target — not silent, `validate_model` refused to start, but the error blamed the
  drafter when the real mistake was the flag/path mismatch. Fixed by deriving the size
  from the served GGUF's architecture (metadata-only read) before `resolve_draft`. The
  one-shot CLI commands deliberately keep the flag's double duty — there the flag and
  the payload are the intent. Pre-existing; surfaced by the 2026-08-11 review.

[Section prose of the pre-regroup ledger, moved verbatim on 2026-09-06.]

`/xwen/v1/batch` and per-request checkpoint selection shipped (log.md 2026-08-11,
decisions.md "Serving"). These are the pieces deliberately not carried.

[Item shipped 2026-09-06, moved verbatim.]

- [ ] [small] **Neither `/health` nor the TUI says which checkpoint is loaded.** `/health`
  reports `model_loaded` as a bare bool and the TUI vitals were built around one model
  id for the process lifetime. Post-swap, both are truthful but incomplete: nothing
  outside the log line says the resident model changed. Cheap: a `model` field on
  `/health` from a shared `AtomicU8`-style cell the engine stamps at load, and a vitals
  line. Do it with the first operational confusion, or sooner if the TUI gets touched.
  From: Deferred from the serve batch + multi-checkpoint arc (2026-08-11).

## Deferred from the Qwen3.8-27B + API-naming arc (2026-08-14)

- [x] **DONE 2026-08-15 (MTP arc, stages A+B): Qwen3.8-27B drafts with its MTP head.**
  `src/mtp.rs` implements the head, `src/drafter.rs` the two-kind seam, and
  `generate.rs` the chain round; `hub.rs` names the sidecar so a zero-flag run fetches
  it. First live smoke on the 27B, `lowpowermode 0` on AC: 39.3 tok/s drafted against
  24.5 plain in the same session at 93.5% acceptance, and `--draft` is byte-identical to
  `--no-draft` at temp 0 and at temp 0.8 seed 42 over 192-256 tokens.
  ANNOTATION 2026-08-15 (Stage C): both of those numbers are superseded, and neither was
  wrong so much as under-measured. The +60% was one run at the then-shipped (0.5, 3); the
  qualification sweep puts the shipped configuration — now (0.7, 4) — at +44-45% code /
  +37-38% chat, and 93.5% acceptance was a single code run against the sweep's 80.0%. The
  byte-identical claim holds for GREEDY (re-verified at 128 and 256 tokens, and again
  after the defaults moved) but NOT for sampled, which diverges at some seeds on the 3.8
  and on the shipped 3.6-27B alike — the pre-existing near-tie class, not a regression.
  See the Stage C log entry and the spec-equivalence items. What follows below was the
  case before any of this. ORIGINAL:
  **Qwen3.8-27B decodes plain: no DFlash sidecar exists, and its MTP sidecar is
  unread.** ggml-org ships `mtp-Qwen3.8-27B-*.gguf` (18 tensors, 1 layer,
  DeepSeek-style: `norm(embed) ⊕ norm(hidden) → fc → one transformer layer → the
  target's shared lm_head`), which is a different drafter shape from DFlash's
  block-diffusion sidecar — a new drafter implementation, not a config entry. The cost
  of not doing it is the whole speculative win on this checkpoint: 3.8 decodes at 23.8
  tok/s plain (one greedy run, 2026-08-14) where the same-geometry 3.6-27B runs 37-38
  DRAFTED against its own 24.8-25.3 plain, so this is the largest single tok/s item on
  the ledger for anyone who actually runs 3.8. The verify machinery
  (K-snapshot fused verify, rollback, auto-pause) is drafter-shape-agnostic and would be
  reused; what is new is the drafter forward and its cache. MTP sidecars also exist for
  both 3.6 checkpoints, so an MTP drafter is testable against a checkpoint that already
  has a measured DFlash baseline to beat.
- [x] **DONE 2026-08-19 (commits a2e02d0/205d9ba): Qwen3.8's chat-template semantics are
  implemented as a per-checkpoint dialect.** The design question at the bottom of this
  item was answered once, as asked: `ChatDialect { Qwen36, Qwen38 }` on `ChatOptions`,
  from `Model::chat_dialect()`, with `ChatOptions::for_dialect` carrying each template's
  defaults. (a) shipped: the xhigh/low preambles render verbatim (pinned against the
  vendored template character-for-character), medium injects nothing, the system block
  is synthesized when the conversation has none, and the preamble precedes the `# Tools`
  header; the OpenAI `reasoning_effort` field now drives the think budget AND the
  template level (nearest-mapping — the one-field-or-two question this item posed was
  answered "one", decisions.md "Serving"). (b) shipped: preserve_thinking defaults true
  under the 3.8 dialect, and is per-request on the native and OpenAI dialects.
  (c) was WRONG as recorded: xwen HAD implemented the inline `<think>`-in-content
  fallback (`split_reasoning`, running unconditionally), so 3.8 turns were getting the
  3.6 reading rather than a free pass — it is now gated to the Qwen36 dialect, and a 3.8
  turn renders such content verbatim. TOKENIZATION_RULES_VERSION went 2 → 3 for the
  encoding change. See log.md 2026-08-19 and the new deferred section below for what the
  arc deliberately did not carry. ORIGINAL:
  **Qwen3.8's chat-template semantics are vendored but not implemented, and TWO of
  them make every default 3.8 conversation diverge from the official rendering.** The
  template is at `reference/chat_template-qwen38.jinja` and cross-checked by chat.rs's
  tests; its behaviors are not.
  (a) `reasoning_effort` — with thinking on and no effort named, the template resolves
  to `xhigh` and prepends "Reasoning effort is set to xhigh. Please think carefully
  through the task, validate key assumptions, consider plausible alternatives, and
  prioritize correctness, consistency, and clarity in the final answer." to the system
  block (creating one when the request has none); `low` prepends its own sentence and
  `medium` prepends nothing. Since xhigh is the DEFAULT, every 3.8 conversation xwen
  renders today is missing a system instruction the model was trained to see — what we
  render equals the official `medium` rendering. Note the OpenAI dialect ALREADY takes a
  `reasoning_effort` field and maps it to a think budget, so implementing this means
  deciding whether one field drives both or they are separate knobs.
  (b) `preserve_thinking` defaults to TRUE, the opposite of 3.6's and of what serve does
  today, so a 3.8 conversation drops reasoning blocks its own template would have kept.
  (c) The inline `<think>`-in-content parsing fallback was removed, which costs nothing —
  xwen never implemented it.
  All are per-checkpoint prompt behavior on a renderer that is currently
  checkpoint-blind: the design question is where the checkpoint enters `ChatOptions`, and
  it should be answered once for all of them rather than three times. Until then the
  divergence is documented, not silent (decisions.md "Tokenization, chat, tool calls").

[Section prose of the pre-regroup ledger, moved verbatim on 2026-09-06.]

Qwen3.8-27B shipped as a registry entry and the APIs went to full model names only
(log.md 2026-08-14, decisions.md "Defaults and CLI surface" / "Serving"). These are the
pieces deliberately not carried.

[Closed parts of the item «No parity-gate or retune arm for Qwen3.8-27B», moved verbatim on 2026-09-06; its open remainder is in TODO.md under "Parity, provenance and tooling".]

  ANNOTATION 2026-08-15 (MTP stage B): the drafter half of this is no longer hypothetical
  — 3.8 now HAS a drafted arm, so `retune-draft.ts`'s exclusion and `SHIPPED_P_MIN`'s
  missing 3.8 entry both become Stage C's, alongside the parity-gate run.
  [Record](records/mtp-stage-b.md).

## Deferred from the MTP drafting arc (2026-08-15, stages B and C)

- [x] **DONE 2026-08-15 (Stage C, C1): the MTP chain length is a per-checkpoint knob.**
  `hub::Model::draft_max_default` returns 15 for the DFlash checkpoints and 3 for the
  3.8's MTP head, exactly as `draft_p_min_default` works; `--draft-max` and the serve
  config's `draft.max` override it, and both are now `Option` so "unset" is
  distinguishable from "set to the old shared default". `MtpDrafter::max_chain_len` stays
  as a sanity ceiling (16) so a mistyped `--draft-max 500` costs a bad round rather than
  five hundred forwards. The `serve --init` template comments `max` out and explains the
  per-kind split, like `p_min`. ORIGINAL:
  **The MTP chain length is a compile-time 3, not a knob.** `MtpDrafter::max_chain_len`
  returns llama.cpp's fitted `n_max` default and a round takes `min(--draft-max, 3)`. It
  is deliberately not on the flag: `--draft-max`'s default of 15 is a block drafter's
  number and both kinds read the same flag, so honouring it would draft 15-step chains by
  default. The consequence is that a Stage C sweep cannot explore chain depth without
  editing the constant. Promote it the way `p_min` was promoted — a per-checkpoint default
  on `hub::Model` with the flag overriding — when the sweep needs it.
- [x] **DONE 2026-08-15 (Stage C, C3): `scripts/spec-equivalence.ts` has a 3.8 arm.** It
  and `retune-draft.ts` were wired up together, both through `scripts/hf.ts`, whose
  `drafter: null` for 3.8 was the single thing excluding it from both harnesses. The
  default model list is now `27b,35b,3.8-27b`.
  **The stage-B claim it was meant to re-run does NOT fully reproduce, and the ledger
  says so rather than the harness quietly passing.** GREEDY is clean on both fixtures at
  128 and at 256 tokens. SAMPLED (temp 0.8, fixed seed, `p_min` 0, `pause_margin` 0)
  diverges on 3.8 at some seeds and not others — seed 42 code forks at line 7, seed 7
  forks at line 1 on both fixtures, seed 99 code and seed 1 chat are byte-identical —
  where stage B recorded it byte-identical at 192 tokens, seed 42. That claim was one
  hand-run pair and did not survive being re-run.
  It is NOT an MTP regression, and the control is why: the shipped DFlash 27B diverges in
  sampled mode too, on the chat fixture at every one of seeds 42/7/99 (lines 9-13), while
  its code fixture is always clean. So sampled-mode divergence is the pre-existing
  near-tie class the script's own header documents — the batched verify forward
  reassociates its f32 sums differently from the single-token forward, and at temperature
  a near tie resolves to a different token. A structural sampler-stream bug is separately
  ruled out: it would fire on every seed, and two 3.8 seed/fixture pairs came back
  byte-identical over 128 sampled tokens, which is impossible if the spec loop drew a
  different number of times than plain.
  What is left open is the SCRIPT'S OWN CRITERION, not the engine — split out as its own
  item below. Until it is fixed: GREEDY is the gate, and a sampled divergence needs the
  control run beside it before it means anything.

[Section prose of the pre-regroup ledger, moved verbatim on 2026-09-06.]

The MTP head ships and drafts (see the closed item above). These are the pieces stages B
and C deliberately did not carry.

[Trimmed from the open item «No fitted draft floor for a custom drafter on Qwen3.8-27B» (until 2026-09-06 titled «DONE 2026-08-14 (review round): drafting is resolved per checkpoint, not per process») on 2026-09-06: the DONE narrative and its stray checkbox.]

- [ ] [blocked] [x] **DONE 2026-08-14 (review round): drafting is resolved per checkpoint, not per
  process.** Filed as deferred in the first pass and fixed in the review round, because a
  sidecar-less DEFAULT checkpoint silently disabled drafting for every OTHER checkpoint
  that server could load (-46 to -52% on the 27B, invisible). `ServeSettings.draft` is
  now a `DraftMode` (`Off` / `Official` / `Custom(path)`) resolved when each checkpoint
  loads. [Record](records/serve-target-review.md), and the rules it settled are in
  [decisions/serving.md](decisions/serving.md).
  What remains: nothing about the shape, but the fallback floor it exposes is worth a
  measurement.

[Duplicate of the open item «The MTP head cannot follow a rewind, so a serve conversation that rewinds stops speculating until it prefills from zero.», folded and moved verbatim on 2026-09-06.]

- [ ] [unpriced] **A stored MTP cache image resumes only at the position it ends at.** Same root as
  the item above, seen from the disk tier: `DrafterImage` carries one carry hidden, so
  `MtpDrafter::import_cache` refuses a `pos` short of `image.pos` rather than restoring a
  head that cannot take another token. A page-in that resumes at an earlier snapshot
  therefore loses the drafter planes and runs that conversation plain — which is the
  regime `Engine::rejects_image` already documents as acceptable (a drafter refusal costs
  speculation, not the conversation), but it is more common for this kind than for
  DFlash, whose images take any prefix. Fixed by whichever fix the item above gets.
  From: Deferred from the MTP drafting arc (2026-08-15, stages B and C).

## Deferred from the chat-dialect and sampling-defaults arc (2026-08-19)

- [x] **`--min-think`/`--max-think` are not guarded against `--no-think`.** The same
  distortion class as the guarded `--raw` combos (`--show-thinking`, `--no-think`,
  `--reasoning-effort` with `--raw` are all startup errors): with thinking off the
  prompt closes the `<think>` block itself, so a min/max think budget describes a span
  that will never open — the flags are inert, and inert-but-accepted is the shape this
  CLI otherwise refuses. Cheap fix in `main.rs` next to the existing guards; the only
  care needed is serve, where `thinking.default_budget` is a server-wide setting that
  legitimately coexists with per-request thinking-off (there it means "when a request
  thinks, cap it", so serve is NOT in scope for this guard).
  - DONE 2026-08-19, same day (the arc's review pass): both combinations are startup
    errors in both gen and chat arms (`ThinkArgs::check_think_budgets`, unit-tested).
    One correction to the text above: the flags were never merely inert — the CLI arms
    the ThinkBudget machinery unconditionally, so an armed `--max-think` against a
    no-think reply would have injected the wrap-up sentence and a stray `</think>`
    into the answer (serve guards this via `max_think.filter(|_|
    starts_in_thinking)`; the CLI path had no such filter). Serve stays out of scope,
    as argued.

[Section prose of the pre-regroup ledger, moved verbatim on 2026-09-06.]

The chat template became a per-checkpoint dialect and sampling defaults went mode-keyed
(log.md 2026-08-19; commits a2e02d0/205d9ba). These are the pieces deliberately not
carried.

[Item shipped 2026-09-06 (log.md "2026-09-06 — Presence penalty: the cards' recipe, through the speculative verify, on by default"), moved verbatim.]

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

## Deferred from the qwen4exp cache-image arc (2026-08-30, P4)

- [x] **A qwen4exp serve run has never been benchmarked.** Every perf number for this
  checkpoint in CLAUDE.md and the port doc comes from `generate` — prefill ~796 tok/s,
  decode ~45, both measured on the one-shot path. Serve adds the queue, the prefix
  cache, page-out and per-request template rendering, and on a 111 GB resident trunk the
  page-out itself is the interesting cost: a `HostFullKv` for this checkpoint carries
  the QSA planes on top of the K/V ones, and `snapshot_bytes` is now what sizes it. Do
  not quote the one-shot figures as serve figures until a serve run has been measured
  under the usual protocol (interleaved rounds, medians, power mode stated).

  **DONE 2026-08-30 — the run happened; the figures are in log.md ("serve on
  Flash-Next, first benchmark") and the rule it uncovered is in decisions.md
  "Serving".** Read-only bench at f949b1d, `xwen serve --no-tui` on defaults
  (`--ctx 262144`, 2 slots, 4 snapshots, disk tier off, no drafter), OpenAI dialect
  streaming with `include_usage`, thinking off, `max_tokens 64`, the qsa-c fixtures as
  prompts, `pmset -g` printing `powermode 0` with no `lowpowermode` key. Serve decode is
  at parity with `generate`: **45.2-46.9 tok/s at 2k, 44.1-45.5 at 7.6k, 42.1-43.5 at
  32k**, TTFT-derived prefill ~800-940 / 627-696 / 500-511 at the same lengths. Cached
  resubmits return in 95-233 ms and a grown conversation takes its next turn in 348 ms
  at 7.6k / 489 ms at 32k. Footprint at rest after the 32k runs: 16 GB phys, 21 GB peak,
  43 GB clean mapped weights; load 23.2 s; 20 runs, no errors. Two things the run left
  behind are new items in the 2026-08-30 serve-benchmark section at the end of this
  file: serve's 4-7% deficit at 32k, and mid-message snapshots. Page-out cost was NOT
  measured here — the disk tier was off and no slot swap was forced — so that half of
  this item's question is still open and lives on in the ~113 MiB floor item above.

[Section prose of the pre-regroup ledger, moved verbatim on 2026-09-06.]

Snapshots, page-out, rewind and the disk tier learned the qwen4exp recurrent state, and
`xwen serve` and `xwen batch` opened for Flash-Next (log.md 2026-08-30, decisions.md
"Qwen3.8-Flash-Next"). These are the pieces that arc deliberately did not take.

## Deferred from the prefill-chunk pass (2026-08-30)

- [ ] **The f16 rescale chain at prefill** (`moe.rs` `needs_rescale`, the L2 guard that
  keeps the down-projection input inside f16 range on the f16-tile prefill variants).
  At 2048 rows per chunk the guard is a band of elementwise dispatches per layer that
  decode never pays; fold it into the gemm epilogue or the following norm, whichever the
  profiler ranks higher (rank, do not price — the profilers are sync-inflated). Promoted
  to first 2026-08-30: it is part of the non-gemm majority of `ffn`. **SHIPPED
  2026-08-30** (log.md "FFN glue"): `ops::silu_mul_l2` folds the whole band —
  silu*mul, Σact², sqrt, clamp, ×32768, divide — into one dispatch
  (`XWEN_ACT_L2_CLASSIC` reverts; 3.574e-7 max-rel vs the chain); +4.8% Flash-Next
  prefill @3803 within-sweep, and the 35B mm/ppl tiers moved within floors
  (0.999618 / 0.001179).
- [ ] **Route the hyper-connection and shared-expert gemms onto `dense_mm`.** The P8c
  gemm (`src/ops/dense_mm.metal`) was 2.2-2.7x on the 27B's dense FFN; whether the
  Flash-Next hc mix and the `shexp` gemms take it at prefill today has not been checked,
  and at 2048 rows per chunk they are squarely in its `seq > 32` envelope. Same gate
  as `dense_mm` (Q4_K/Q8_0 source, graded by mm/ppl), and the same accuracy trade.
  Promoted to second 2026-08-30: the shared expert is in the non-gemm-expert majority of
  `ffn`. **SHIPPED 2026-08-30** (log.md "FFN glue"): both routed through
  `QLinear::forward_gemm` above seq 32 (`XWEN_SHEXP_QMATMUL` /
  `XWEN_HC_GEMM_QMATMUL` revert; hc planes dense_mm-only, decisions.md). The
  surprise inverted the ranking: shexp ≈0 end-to-end, hc `up` (k=320, the shape
  flagged "may not win") +7-11%; Flash-Next 872 → 962-977 @3803, 766 → 860 @7606,
  35B 2755 → 3090 @3803.
- [ ] **A narrower `mm_id` token tile (NR1 32 → 16).** At the 2048 chunk each Flash-Next
  expert sees ~40 rows per gemm, which a 32-row tile covers as one full tile plus a
  quarter-empty one; a 16-row tile would waste less of the second tile and lets the
  1024 chunk (~20 rows) stop paying for a half-empty tile too. Untested; the win, if
  any, is bounded by the expert gemm's share of prefill, so profile that share first.
  **REFUTED by a code read, 2026-08-30, before any bench**: the `_t` kernel dequantizes
  the expert's whole weight tile once per TOKEN tile (mm_id.metal ~590-625, indexed by
  expert and out-row only), and it is dequant-bound, so passes per expert =
  ceil(rows/NR1) and a narrower tile RAISES the dominant cost (Flash-Next 1.88 → 2.97
  passes; 35B 2.5 → 4.5). The lever runs the other way — NR1 64 (1.0 / 1.5 passes,
  +6% / +20% MMA slots, 16 KB smem) — and the larger waste is the grid: sized for one
  expert owning every row, ~97% of launched threadgroups early-return at the 2048 chunk
  (down: 1,310,720 launched, 40,960 useful). Both are being implemented as a work-list
  grid (map0 emits (expert, tile) pairs; host bound ceil(t*top_k/NR1)+n_expert, no
  readback) plus a templated NR1 64 on the `_t` family, each behind a switch.
  **SHIPPED THE OTHER WAY, 2026-08-30** (log.md "mm_id tiles", decisions.md "mm_id
  tiles"): work-list grid on all three families + `_t64` with the ≥ 24-rows rule,
  bit-neutral, isolated +17-23% (FN gate/up 416k → 512k tok/s, FN down q5_1 202k →
  236k, 35B gate/up 628k → 751k, 35B down 260k → 281k), end-to-end at 3803 tokens
  nothing claimable (Flash-Next 841/799 → 827/862, 35B 2657 → 2708 in one round), the
  `ffn` stage falling only 3-5% because the gemms are its minority. Any further
  mm_id tile work sits below the two items above.

[Section prose of the pre-regroup ledger, moved verbatim on 2026-09-06.]

The chunk went 512 → 2048 on the MoE checkpoints, on every surface, and stayed 512 on
the dense ones (decisions.md "The prefill chunk is per architecture", log.md 2026-08-30 "prefill chunk"). The
A/B named four things it did not take. **Re-ranked 2026-08-30 after the mm_id tile pass**
(log.md "mm_id tiles"): the expert gemms are a MINORITY of the prefill `ffn` stage (a
17-23% isolated gemm gain moved `ffn` 3-5% and prefill wall not at all), so the two
non-gemm items come first. [CONTESTED 2026-09-05: the "minority" reading came off the
stage profiler, which inflates prefill 2.2x, and the same amortized bench's rates put the
expert gemms at ~44% of wall — [record](records/ceiling-diagnosis.md).]
[SETTLED 2026-09-05 by the duplicate-dispatch probe: expert gemms 28-32% of wall and 73%
of `ffn`, glue 11.5%, and the re-ranked ledger at the top of this file puts the expert
gemm first for prefill — [log](log.md#2026-09-05--duplicate-dispatch-probe-prices-flash-next-prefill-in-situ-expert-gemms-109-s-of-342-s-3851-32-moe-glue-040-hc-gates-039-gdn-023-shared-expert-0).]

[Closed parts of the item «Decode on Flash-Next steps down ~11 ms/token the moment the context crosses the 2048-token QSA budget, then slopes gently», moved verbatim on 2026-09-06. That item is now fully closed and no longer appears in TODO.md; its surviving scope was promoted into items under Decode performance, Prefill performance and Parity, provenance and tooling, each naming it in a "Promoted from" line.]

- [ ] **Decode on Flash-Next steps down ~11 ms/token the moment the context crosses the
  2048-token QSA budget, then slopes gently: 46.1 tok/s at 1963 tokens, 30.8 at 2045
  (the run crosses 2048 mid-decode), 30.6 at 2107, 29.4 at 3810, 27.3 at 7620** (2026-08-30,
  `--no-draft --raw -n 64`, interleaved, `pmset -g` said `powermode 0`). Not KV scaling:
  below the budget `QsaIndexer::select` short-circuits to Dense (indexer.rs:294-296);
  above it, attention itself is CAPPED at 2051 gathered keys and every added cost is in
  the indexer, 12 layers per step: (1) the pooled block keys are recomputed from ALL raw
  keys every step (indexer.rs:316-320), and the `mean(1)` over [n_blocks,4,128] misses
  candle's contiguous-reduce test and takes `fast_sum_f32_strided`, which launches one
  2-thread threadgroup per output — ~1.5 M threadgroups per step at 4k, ~12.6 M at 32k
  (est. 8-10 ms at 4k, 65-85 ms at 32k: ~10 tok/s at Claude-Code contexts); (2) one
  host readback sync per layer for the scores (indexer.rs:346), 12 pipeline drains per
  step, ~3 ms flat; (3) the rope chain + k_norm + score matmul over all blocks, linear.
  Fix, in order: cache the pooled+normed+roped key per FULL block (immutable once the
  block is complete; only the tail block is recomputed per step), which kills (1) and
  (3); then a device-side top-k or fused score+select writing row indices, which kills
  (2). Bench at 2k/4k/16k/32k. Stack profile (attribution only, sync-inflated) puts the
  +13.3 ms/token growth at 1919 → 3810 in three stages: `qsa_select` +5.0 ms,
  `mixer_full_attn` +4.5 ms (the gather path — 24 `index_select` dispatches plus a
  `stack` per layer, attention.rs:702-713, which only runs above the budget; capped is
  not cheap) and `ple` +3.2 ms (unexplained, possibly bleed from the adjacent syncs), so
  the fix also needs a single-dispatch gather (or attention reading the row list
  directly). The shipped checkpoints have no indexer and are unaffected. No runtime QSA kill switch exists (`force_dense_qsa` is cfg(test)).
  **2026-08-30, later: steps A and B SHIPPED** (block-key cache in `IndexerCache`, fused
  `ops::qsa_gather`, kill switch `XWEN_QSA_CLASSIC`): 3.8k 30.5 → 32.9, 7.6k 30.3 → 33.5,
  greedy byte-identical, ~8.5 ms/token of the cliff left.
  [Log](log.md#2026-08-30-qsa-decode-steps-ab--block-keys-cached-per-complete-block-and-the-kv-row-gather-fused-flash-next-decode-above-the-2048-budget-305--329-at-38k-303--335-at-76k),
  decisions.md "Block keys are cached per complete block". **Step C is next**: a
  device-side top-k (or fused score+select writing row indices) that removes the 12
  per-step score readbacks.
  **2026-08-30, later still: step C SHIPPED** (`kernel_qsa_select`, radix select with no
  readback, kill switch `XWEN_QSA_HOST_TOPK`): 3.8k 33 → 41-44, 7.6k 33 → 44-45, 16k
  32.0 → 41.7, 32k 33.8 → 45.3 against a 46.7 below-budget anchor, greedy
  byte-identical. **The cliff is closed.**
  [Log](log.md#2026-08-30-qsa-decode-step-c--block-selection-moved-onto-the-device-flash-next-decode-above-the-2048-budget-33--44-45-toks-at-38k-32k-the-cliff-closed),
  decisions.md "Decode selection runs on the device". Remaining sub-items:

[Duplicate of the open item «Above the 2048 indexer budget: +165 dispatches», folded and moved verbatim on 2026-09-06.]

- [ ] [unpriced] **`kernel_qsa_select`'s threshold walk is serial on thread 0.**
  The walk covers 256 bins × 4 passes on that one thread; a cooperative
  256-thread walk (per-bin prefix via scan) is the obvious next shape. Measure the
  kernel's share of the step with an amortized bench before acting — at 44-45 vs 46.7
  there is at most ~1 ms/step left in the whole above-budget path.
  From: Deferred from the prefill-chunk pass (2026-08-30).
  Promoted from the item «Decode on Flash-Next steps down ~11 ms/token the moment the context crosses the 2048-token QSA budget, then slopes gently» on 2026-09-06; dated 2026-08-30 there.

## Deferred from the technique survey (2026-08-30)

- [x] **`CANDLE_METAL_COMPUTE_PER_BUFFER` default (50, per DISPATCH not per op —
  `commands.rs:18,162`): REFUTED 2026-08-30, keep 50.** The decode-side A/B (Flash-Next,
  3 rounds rotated, 60 s idles, anchors clean, `powermode 0`): 1000 lost every cell,
  monotonically with context — decode −3.6% @1937, −6.2% @3803, −6.8% @7606 (prefill
  −6.4% there too); 200 a wash short, −1.6 to −2.0% long (35B same direction). Greedy
  byte-identical across arms, so it is pure performance, and the 2026-08-08 prefill
  sweep (10-1000 within 0.9%) plus this decode result close the knob in both directions:
  frequent rollovers are FREE-to-beneficial (plausibly by keeping the in-flight pool and
  the `prev_ce_outputs` fence map small). No default change; nobody should set the var.

Hazard that applies to every item above: candle's pooled-buffer recycle fires at
`strong_count == 1` with no in-flight check (`device.rs:488-503`), so a cadence or
concurrency change can hand a still-live buffer back to the pool. Grade these with the
parity gate plus greedy equivalence, never with tok/s alone — a corruption from this
mechanism is intermittent and looks like a sampling difference.

[Section prose of the pre-regroup ledger, moved verbatim on 2026-09-06.]

Research pass, no code (log.md "technique survey"; the three refutations it produced are
in decisions.md "Refuted perf directions"). **Every item here is UNPRICED** — the survey
ranked candidates against published evidence, it did not measure anything on this
machine, so no item carries an expected win. Each says what would have to be measured
first. The candle-side items (3) and (4) are patches to the pinned rev 21cca0b, which
nothing in this repo has done before; treat that as part of their cost.

[Duplicate of the open item «+350 to +560 µs/token of prefill cost lives OUTSIDE every measured stage, and it grows with prompt length.», folded and moved verbatim on 2026-09-06.]

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

[Duplicate of the open item «Expert gemm efficiency: 14-43% of wall, bracketed by two in-situ A/Bs», folded and moved verbatim on 2026-09-06.]

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
  cap is lifted to "unpriced" — [record](records/ceiling-diagnosis.md).]
  [PRICED 2026-09-05: 28-32% of prefill wall by the duplicate-dispatch probe.]
  From: Deferred from the technique survey (2026-08-30).

## Next Flash-Next perf work (2026-09-05)

[Section prose of the pre-regroup ledger, moved verbatim on 2026-09-06.]

**DONE 2026-09-05, decode: the fused hc gate** (dd50397, decode item 1 of the re-ranked
ledger above, +9% plain decode; [record](records/fused-hc-gate.md)). Its own "next
candidate" ranking was amended 2026-09-06 — the router fold refuted, the −192 shexp alone
— and then superseded by the two entries below. Prefill: the
duplicate-dispatch probe has priced the stages (log "Duplicate-dispatch probe"); the
expert gemm (0.96-1.09 s of 3.4 s) is the item, the down plane's dequant the first
experiment, and `XWEN_DUP_STAGE=experts_down` is how to price any change to it in situ.
**DONE 2026-09-06, decode: the fused shared expert** (b7cd358 + 0ed20ea, ledger item 2;
35B 113.2 → 115.0 tok/s (+1.6%), Flash-Next 51.2 → 51.5 (+0.6%), both checks pass;
[record](records/fused-moe-shared-expert.md)). With it the −192 above is spent, and
it changed how the next candidate is chosen: the launch budget only prices launches that
carry under ~2 MB and sit on the dependent chain (decisions.md "Ceilings"; item 2).
**DONE 2026-09-06, decode: the router-projection gemv** (24c4069, ledger item 2; 35B
115.1 → 127.0 tok/s (+10.3%), Flash-Next 50.5 → 52.9 (+4.8%), both checks pass;
[record](records/router-gemv.md)). It is the largest decode lever of the day and it
did not come out of the launch budget at all.
**The day's three stacked levers: 35B 113.2 → 115.0 → 127.0, Flash-Next 51.2 → 51.5 →
52.9** (the Flash-Next levels drift between sessions, so only the within-session ratios
are claims). **The launch-budget order below is unchanged** (the MoE glue kernels, then
the token-id readback sync at item 3, then the QSA tail at item 5, then the GDN glue at
item 4), **but a new class now sits ahead of it to be surveyed first: OCCUPANCY.** The
router projection was priced by neither the probe nor the byte budget and was worth +10%,
so the next instrument is the threadgroup-count-against-bytes audit of every decode
dispatch (item 2(e)), and it should run before the next launch-count lever is picked.
**DONE as opt-in 2026-09-05, (a) default flipped the same day** (`XWEN_PLE_DEVICE=1`,
multi-token Metal forwards; +12.8% prefill @3851, +12.9% @880, decode flat;
`XWEN_PLE_TAIL_CLASSIC=1` is the kill switch, and the Flash-Next check is codified as
`scripts/flashnext-replay.ts` with an engine-side near-tie rule).
[Record](records/ple-device-tail.md), and parity.md "Limitations" for the rule.
Live follow-ups: (b) decode tail on device is UNPRICED (0.13 ms/token of host work against the extra
readback it would need; the batched readback already lands the carrier); (c) the conv kernel's
per-element `exp` under safe math and the armed-path strided window rebuild are unpriced and
unreachable-at-default respectively.
The next candidate was **PLE's remaining host gate and conv**, item (5) in the P3
ledger below. Start at `src/qwen4exp/ple.rs::PleLayer::forward`: measure the host
`gate`/`conv` work at decode and 2048-token prefill after the readback collapse,
then price a device implementation with an amortized bench before wiring it in.
The old decode three-readback estimate is superseded; detailed results from this
pass are in [the log](log.md#2026-09-05--ple-batches-its-three-device-to-host-readbacks).
A device port must preserve the PLE conv window, n-gram history, checkpoint/partial
commit and cache-image semantics. `ref_ple.rs` and `ref_hc.rs` remain frozen oracles;
the PLE tests already cover those state transitions. Keep a classic arm, run the
Flash-Next forced-replay workflow in `docs/parity.md`, and apply AGENTS.md's thermal
protocol to end-to-end measurements. No gain has been established for this remaining
work. The hc Q8 decode substitution is not an alternative default ready to enable:
its up shape needs a new geometry and its down shape remains unqualified (item (3)).

## Deferred from the DeltaNet-kernel hardening pass (2026-07-29)

[Section prose: the heading exists because TODO.md items name it in their `From:` line.
The first item closed under it is the `mtl_size!` one below, on 2026-09-06.]

[Item shipped 2026-09-06, moved verbatim.]

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

## Deferred from the client-feedback arc (2026-08-11)

[Section prose of the pre-regroup ledger, moved verbatim on 2026-09-06.]

The escape fix, `shared_prefix`, the 100 MB body cap and lazy KV / the 131072 CLI
default shipped (log.md 2026-08-11 client-feedback entry; decisions.md "Batch",
"Serving", "Defaults and CLI surface"). What the arc deliberately did not do:

[Item shipped 2026-09-06, moved verbatim. The envelope is measured (docs/perf-state.md "Long context"), (a)-(e) all landed, and what it opened instead lives in the long-context envelope arc's own heading below.]

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

## Qwen3.8-Flash-Next port (decided 2026-08-25, blocked on release + upstream)

[Section prose of the pre-regroup ledger, moved verbatim on 2026-09-06.]

Alibaba's ModelScope countdown page
(https://modelscope.cn/models/Qwen/Qwen3.8-Flash-Next) drops the model card
~2026-08-26: an open-weight preview of the Qwen4 architecture. Teased specs from the
since-trimmed model-card highlights: multimodal MoE, 125B main params + 51B additional
n-gram embedding params ("fast local token lookups" — reportedly a hashed table read a
few rows per token, never through a matmul), 6B active per token, built on "GDN and
QSA" mechanisms, ~1/9th the training cost of Qwen3.7-Plus at comparable capability.
Decision: we WILL port it, targeting Q4_K on this machine.

[Closed parts of the item «Port Qwen3.8-Flash-Next», moved verbatim on 2026-09-06. Its open remainder was promoted into items across five areas of TODO.md — Decode performance, Prefill performance, Serve/batch/CLI, Cache images/memory/context, Parity/provenance/tooling and Tokenizer/chat/sampling — each carrying a "Promoted from" line naming this item.]

  and the n-gram embedding subsystem are new; the transferable assets are the Gated
  DeltaNet implementation (GDN) and the 35B-A3B MoE machinery (router / top-k renorm /
  shared expert). Blocked on three things, in dependency order: (1) the actual model
  card + transformers modeling code (drops ~2026-08-26); (2) llama.cpp arch support —
  our ground-truth chain and parity oracle need it, and it will lag release; (3) a
  GGUF path — ggml-org GGUF, or convert ourselves once llama.cpp's convert script
  supports the arch.

  - Capacity checked 2026-08-25: 128 GB RAM, 757 GB free disk. ~105 GB estimated at Q4
    (~75 GB MoE + ~30 GB n-gram table); the n-gram table can stay file-backed/CPU-side
    since it's sparse row lookups. Worst case (no day-one GGUF) is ~350 GB BF16
    safetensors + self-conversion, which fits disk. The "one large model process at a
    time" rule becomes absolute at this size.

  - First moves when the card drops: read the modular_*.py modeling code + config.json
    to turn "architectural upgrades" (attention/residual/embedding/optimization) into
    a concrete delta list; watch llama.cpp for the arch PR; decide vision-tower
    separability (expect to ignore it like mmproj-*).

  - Research findings 2026-08-25: release timestamped 2026-08-26T15:00Z; planned
    artifacts are safetensors + FP8 only, no GGUF planned; nothing in flight in
    llama.cpp or transformers (closest precedent for the n-gram table: open unmerged
    llama.cpp PR #19167, LongCat-Flash-Lite n-gram embeddings — ggml has no shipped op
    for it either). The trimmed highlights (125B/51B/6B) survive only as forum
    copy-pastes; "GDN and QSA" specifically is community paraphrase, so GDN carrying
    over is NOT yet established. Planning assumption (2026-08-25): Unsloth publishes
    within ~1 h of release — that covers weights/quants and their usual
    tokenizer/template fixes, but an Unsloth GGUF still requires llama.cpp arch
    support to exist first (GGUF is the container, not the graph), and xwen needs the
    graph port regardless; a day-one GGUF would only hand us ground-truth authority
    #2 (the tensor table) early and skip self-conversion. Whatever file we bless,
    parity floors get calibrated to ITS quant mix — Unsloth dynamic mixes are not
    ggml-org's Q4_K_M mix. Don't wait on llama.cpp to START the port: vLLM/SGLang
    support is typically contributed by Qwen themselves and lands day-one (as it did
    for Qwen3-Next), so their model code + the transformers modular file are the
    executable references for the math in the llama.cpp gap — enough to write the
    graph and float-level taps against, even though the GGUF-vs-GGUF parity oracle
    still has to wait for a llama.cpp arch + blessed file.

  - Architecture priors (researched 2026-08-25, pre-card — verify against the real
    modeling code before building on any of this): QSA has no paper; the strongest
    prior is DeepSeek-DSA-shaped (small-dim indexer, relu(k·q) scores, top-k kv
    selection, sparse mask over ordinary attention) — our pinned llama.cpp clone
    already carries three implementations (`ggml_lightning_indexer`,
    `src/models/glm-dsa.cpp:343-383`), so that half would be cheap. The n-gram table
    is almost certainly Engram-shaped (DeepSeek, arXiv 2601.07372, code published):
    multiplicative-XOR hash over suffix bi/trigrams of NORMALIZED token ids (NFKC +
    lowercase compression — a silent-garbage trap), K hash heads per order,
    prime-sized tables, rows concatenated → projected → scalar-gated → added
    residually at a couple of MID-STACK layers (not the input embedding). Indices
    depend only on token ids → host-side precompute + get_rows + gated add; no new
    GPU op needed, and the table is the one component where file-backed costs
    nothing (Engram paper: 100B-param host-offloaded table = 2.8% throughput).
    GDN = Gated DeltaNet, no published evolution — our DeltaNet path is the piece
    most likely to survive unchanged. BIGGEST structural risk (low confidence, high
    impact): "Residual" upgrades = hyper-connections (ggml `ggml_dsv4_hc_*`,
    arXiv 2512.24880), which would invalidate the `x + f(norm(x))` skeleton.
    llama.cpp precedent: qwen3-next took ~78 days release→merge (#15940/#16095);
    expect a similar-or-longer tail here, and no day-one ggml-org GGUF (the convert
    pipeline needs the arch first). Multimodal: Qwen practice is a separable
    mmproj-* tower and one report says no vision exercised in this release — prior
    ~80% we can ignore it as usual.

  - CARD DROPPED 2026-08-26 — confirmed spec (from config.json, transformers
    `modular_qwen4_exp.py` on main, and Unsloth's GGUF metadata; full digests in the
    session that wrote this entry). Arch `qwen4_exp` (GGUF `qwen4exp`), 48 layers,
    hidden 2560, 125B MoE + 51.2B PLE table + 4B MTP = 180B on disk, A6B, ctx 262144
    (YaRN→1M). Same `(i+1)%4==0` attention cadence (12 attn / 36 GDN). BF16 repo:
    131 shards, 360 GB, no trust_remote_code. Prior grades: GDN TRUE except the
    gated-norm output gate is `sigmoid(z)` NOT `silu(z)` (`output_gate_type`) —
    otherwise byte-identical geometry to our 27B DeltaNet (16K/48V, dim 128, inner
    6144, conv 4, silu over fused stream, same tiled-vs-interleave converter rule).
    QSA VARIANT of DSA: MQA indexer (4 q-heads, 1 k-head, dim 128, RMSNorm then
    64-dim partial rope), scores relu(q·k).sum(heads)/√128 over 4-token mean-pooled
    BLOCKS (fp32 pool, k_layernorm, rope at block-first position; keys cached RAW),
    top-512 blocks = 2048-token budget, incomplete tail always visible, no sliding
    window. Residual skeleton FALSE — hyper-connections: 4 streams (10240-wide
    carrier), rank-320 read bottleneck silu(down/4)→sigmoid(up), per-stream mean
    read, write gate 2·sigmoid(inject/4), write-back onto the UN-NORMED stream; NO
    input_layernorm/post_attention_layernorm/final norm tensors exist (hc_norm =
    grouped RMSNorm(group 2560) replaces all); tail `hyper_connection_mixer` then
    lm_head. PLE (n-gram) Engram-VARIANT: orders {2,3}×8 heads, row dim 160, 16
    prime tables (~20M rows each) as ONE padded [320001536,160] tensor (HF: 128
    shard_N tensors; GGUF: single `per_layer_token_embd.weight`, IQ4_NL 28.8 GB),
    splitmix64-derived odd multipliers SHIPPED as I64 buffers (read, don't
    recompute), hash over RAW token ids (NO NFKC/lowercase), shift-right that never
    crosses an eos boundary where eos = SCALAR 248044 (not 248046 — wrong id
    silently corrupts lookups at turn boundaries), ONE layer (ple_layer_ids [2],
    ONE-indexed → decoder idx 1, a GDN layer), injection = key_proj→4 streams +
    value_proj, per-stream dot-product gate ÷√2560 with SIGNED SQRT then sigmoid,
    plus depthwise conv k=4 DILATION 3 (state 9) — adds to the 10240 stream. A PLE
    layer carries THREE recurrent states: GDN conv (10240×3), PLE conv (10240×9),
    2-token raw history. MoE: 512 experts top-10, softmax-all-then-topk-then-renorm,
    NO 6.1e-5 clamp (that's a 3.6-35B-only detail); shexp 640-wide with
    `shared_expert_gate` shape [1,2560]. Attention/rope/tokenizer carry over:
    double-width interleaved q/gate, QK-RMSNorm(256), sigmoid out-gate, theta 1e7,
    n_rot 64, mrope [11,11,10] interleaved ≡ NEoX-64 for text; tokenizer =
    Qwen3.8-27B's exactly (base hash-identical to vendored 3.6 + audio specials to
    248076); template = near-3.8 dialect + vision items (vision in system raises);
    stops [248046,248044] via generation_config. Sampling: thinking 1.0/0.95/20,
    non-thinking 0.7/0.80/20 with PRESENCE_PENALTY 1.5 — our serve accepts-and-drops
    penalties, now a real gap for this checkpoint. Vision separable: inline
    `model.visual.*` ViT (no mmproj file), masked_scatter at image_pad 248056,
    deepstack empty — text-only drop is clean. MTP head present (4B, QSA layer +
    MoE, separate fc_embedding/fc_hidden projections NOT 3.8's concat eh_proj;
    reuses target embd/lm_head) but transformers ships no MTP class — semantics
    need vLLM/SGLang or the tech report. layer_types in config.json says
    "full_attention" but the config class REWRITES it to qwen_sparse_attention —
    trusting the file builds dense attention that runs and is silently worse.
    Toolchain: NOT in llama.cpp master; open DRAFT PR #27742 (Unsloth; Qwen's
    #27739 closed in its favor), zero new ggml ops, conversion moved to a
    `conversion/` package upstream — our pinned oracle (e9fa0781) predates all of
    it. No ggml-org GGUF repo. Unsloth GGUF: split-file (gguf-split, metadata-only
    first shard, `split.*` keys, 1224 tensors), UD-IQ1_S up (72.5 GB), indexer BF16,
    hc Q8_0, sampling keys baked into metadata, `general.name` "Qwen3.8 Flash
    Next". xwen work, descending: hyper-connections; QSA; PLE; split-GGUF loader;
    IQ4_NL dequant (table ships IQ4_NL even in Q4 mixes — self-converting the table
    to Q4_K instead would dodge it); per-checkpoint sampling defaults + presence
    penalty; registry/dialect entry. First testable milestone per Orvar: Unsloth
    Q4-class file.

  - **2026-08-29 — P4/P3: Flash-Next prefill is 3.5x slower than llama.cpp
    (from U7).** 203.5 tok/s against 713.4 at 530 prompt tokens on the identical
    file in the same hour; 2.60 s reproduced to the centisecond across two
    independent runs, so it is not first-forward Metal pipeline compilation.
    Decode is within ~8% (37.7-38.1 vs 40.9-41.5 tok/s) and unremarkable. Two
    known contributors to look at first: the 43 Q5_1-down layers prefill through
    the per-token `mul_mv_id` fallback (D18 — the mm_id item above), and the
    dense-FFN prefill gemm was this same shape of problem on the 27B (P8c) and
    took a vendored kernel to close. Caveats on the absolutes: `lowpowermode 0`
    with no high-power claim, shared machine, llama.cpp thermal-boosts harder —
    the RATIO is the trustworthy part.
    **CLOSED 2026-08-29 (P3): prefill 795.7 tok/s against llama.cpp's 789 (1.01x) and
    decode 43.1 vs 41.4 (1.04x), same file and hour, over three commits — 8112733 (the
    Q5_1 `mm_id` arm), 8aeed73 (four fused hc kernels) and 2c8d3b3 (the split norm
    launch). The second suspect was wrong in its specifics: the other third of the
    prefill wall was the hyper-connection GLUE, not a gemm.**
    [Record](records/flash-next-p3-kernels.md). What stays open is in the P3 ledger
    below, not here.

  - **2026-08-29 — P3 perf ledger for Flash-Next (everything deferred from P2).**
    P2 was correctness-first by decision (decisions.md), so every one of these is
    a known cost taken deliberately, not a discovery.
    **Gain estimates for what remains (2026-08-30, from the floor-corrected
    decode attribution at 6fbc7e8: 22.66 ms/token = 44.1 tok/s; mixer_delta
    10.8 ms / 39%, ffn 7.9 / 28%, ple 1.1-3.8, mixer_full_attn 2.3, hc reads
    1.9, lm_head 1.2, hc writes / qsa_select / embed under the profiler floor).
    These are attributions and byte counts, NOT measurements of the fixes;
    peak bandwidth has never been measured on this machine, so every ceiling
    below is against the nominal figure and may be optimistic.** [MEASURED
    2026-09-05: 537-565 GB/s streaming read, a 6.33 GB whole-token byte floor and an
    81-86 tok/s bytes-only ceiling, so the re-ranked ledger at the top of this file
    supersedes the ranking prose here — [record](records/ceiling-diagnosis.md).]

    (5) PLE readback collapse (three `to_vec1` → one): saves ~0.3 of the
    0.52 ms readback → **+0.5-0.7 tok/s**; PLE gate/conv/readback all on
    device (proj stays): PLE 1.06 → ~0.45 ms → **+1.2 tok/s decode, +5-6%
    prefill (~40 ms of host gate+conv per 512-token chunk, ~+45 tok/s)**; if
    the stack profiler's 3.75 ms `ple` charge is real (it brackets the hash
    and the carrier add the sub-step timer does not; unreconciled), the upside
    is **up to +6 tok/s**. (1a) vendored `mv_id` Q5_1 arm: the down plane is
    ~40% of the ~1.5 GB of expert bytes per token; a 1.2-1.5x kernel over
    candle's baked one on ~3 ms → **+1-2 tok/s**; (1c) per-stack `use_mm` is
    prefill-only and now moot on this file. (3) hc decode gemv through xwen's
    vendored mv path instead of candle `QMatMul` (0.7 GB/token, floor ~1.2 ms
    against 1.9 measured for the whole read) → **+0-1.2 tok/s**; in-place
    `hc_write` is under the floor → **~0**. (10) bimodal decode (42 vs 44):
    **+0-1 tok/s at the median** if the fast mode can be held. NOT YET
    LEDGERED and the largest by far: (14) `mixer_delta` — 36 GDN layers at
    10.8 ms/token against a projection byte floor of ~2.1 GB/token (attn_qkv +
    attn_gate + ssm_out, Q8_0) ≈ 3.5-4 ms at nominal bandwidth plus ~0.4 ms
    of delta-state traffic → **up to +11-15 tok/s** if the layer reached
    bandwidth, which needs a per-op breakdown of the stage first (projections
    vs conv vs delta step vs gnorm; no number exists); (15) MoE decode
    efficiency — `ffn` moves ~1.5 GB/token in 7.9 ms (≈190 GB/s effective,
    the same rate the 35B-A3B shows), so routing glue and dispatch count, not
    bytes, are the cost → **+3-8 tok/s** plausible, shared with the shipped
    checkpoints. Whole-token byte floor ≈ 5.5 GB (experts 1.5, GDN 2.1, attn
    0.6, hc 0.7, lm_head 0.6) ≈ 9 ms at nominal → a ceiling near 100-110
    tok/s that nobody should quote as reachable; llama.cpp sits at 41.

    **ANNOTATED 2026-08-30 after the GDN mixer arc (ae82696, 5526213, f89972f, 0261e17):
    the 10.8 ms above is SYNC-INFLATED and so is every share derived from it — two figures
    off that line were re-priced 2-3x lower, so "up to +11-15 tok/s if the layer reached
    bandwidth" in (14) is WITHDRAWN as a target and the lever is DISPATCH COUNT, not
    bytes.** [Record](records/gdn-mixer-arc.md), decisions.md "How to read
    `XWEN_GDN_PROFILE`".

    - **`ba_proj` — SHIPPED 0261e17:** the beta|alpha gemv folded into
      `kernel_delta_ba_fused`, one dispatch fewer per DeltaNet layer per token,
      **Flash-Next decode 44.4-44.5 → 46.5-46.7 (+4.6-4.8%), 35B-A3B 105.1 → 114.4
      (+8.8%)**, prefill unchanged, every gate re-passed. `XWEN_DELTA_BA_CLASSIC=1`
      restores the chain.

    - **`attn_qkv` — RETIRED, there was nothing there:** at DRAM it is the FASTEST of the
      three GDN projections (510 GB/s against `attn_gate` 464 and `ssm_out` 465), the
      profiler's ordering being inverted, and the shipped `(NR0, NSG)` already wins.

    - **The scan — kept OPT-IN, a wash.** `kernel_delta_scan_decode` behind
      `XWEN_DELTA_DECODE_KERNEL=1`; the general kernel already moves the state
      at 525-564 GB/s marginal, within 1.4x of a candle copy of the same bytes
      (its own ledger section below).

    **(15) MoE decode efficiency gets the same lens and it is the smaller
    target than it looks.** Counted from `src/moe.rs` on the decode path
    (`MoeBlock::forward` → `FusedExperts::project` + `ops::moe_epilogue`; the
    fused-glue predicate holds for this checkpoint, `use_mm` is false at seq 1):
    **12 dispatches per MoE layer and ZERO host syncs** — router matmul,
    `kernel_moe_router`, gate/up/down expert gather-matvecs (one per PLANE, not
    per expert), `kernel_moe_silu_mul`, the four shexp dispatches, the shexp
    gate matmul, and `kernel_moe_epilogue`. All 48 layers carry an MoE FFN, so
    **576 MoE dispatches per token** — twice the GDN block's 288, and the
    largest single dispatch population in the model [recounted 2026-09-05: the GDN
    block is 252 now, and the hc carrier at 672 on the decode split arm is the
    largest population, MoE second]. At 8.41 µs that is ~4.8 ms
    of a ~21 ms token in launch cost alone [~2.3 ms at the ~4 µs average the
    2026-09-05 budget closes at], which is most of why `ffn` reads
    ≈190 GB/s effective. But the glue is already fused (24 → 14 dispatches in
    2026-07-29's pass, and again since), and `XWEN_MOE_GLUE_CLASSIC` costs ~21
    per layer, so what is left is the six matvec/matmul dispatches per layer
    plus four glue ones. The **+3-8 tok/s** estimate above stands only if a
    real fusion exists there; the dual gate|up kernel that would have been the
    obvious one is REFUTED on this device (decisions.md, `XWEN_MOE_DUAL`).
    Next step is a count-reducing shape nobody has proposed yet, not a rate
    argument.

    **ANSWERED 2026-09-06: the shape is the shared expert, 5 → 1** (b7cd358), which takes
    the 12 dispatches per layer to 8 and is worth +1.6% on the 35B and +0.6% on
    Flash-Next, not the +3-8 tok/s this item estimated, because those launches were
    byte-bound rather than launch-bound.
    [Record](records/fused-moe-shared-expert.md), decisions.md "Ceilings". The
    remaining count-reducing shape here is the glue kernels, which do qualify under the
    refined rule.

    **STATUS after P3's first pass (2026-08-29): (1) partly, (2), (3) and (6)
    done; (4), (5), (8) untouched; (7) closed earlier; (9) retired; (10)-(13)
    added.** In rough order of expected
    payoff: **(1)** the Q5_1 expert kernels and per-stack `use_mm` — its own
    bullet above, and the first thing to try against the prefill gap
    (**item (b) SHIPPED 8112733**; (a) and (c) still open); **(2)**
    **prefill is 3.5x behind llama.cpp** — its own bullet above, **CLOSED
    2026-08-29** at 795.7 vs 789 tok/s; **(3)** a fused
    `hc_mix` kernel: the hyper-connection read/write is ~15 dispatches per
    layer-pair built from candle primitives, across all 48 layers, and was
    flagged as the top fusion candidate before any of it was written —
    **SHIPPED 2026-08-29 in 8aeed73 (four kernels, D21) plus 2c8d3b3 (the split launch
    below 32 tokens, D22): it was 34.3% of prefill wall, now 5+1 dispatches per
    layer-pair (~2128 → ~600 per forward), prefill 443 → 765-781 tok/s, decode 43.1.**
    [Record](records/flash-next-p3-kernels.md). Three follow-ups, none blocking:

    dilated conv and silu run on the HOST in f32 over a `[n,10240]` copy of the
    stream, 40 KB/token plus one device→host sync per forward at layer 1 (D17) —
    move them to device. **QUANTIFIED 2026-08-29 (`XWEN_PLE_PROFILE`): the host gate plus
    conv are ~40 ms of a 512-token chunk, and at decode the layer's fixed floor is
    ~0.85 ms, of which the three readbacks are 0.50 and the projections 0.33.**
    [Record](records/flash-next-p3-kernels.md).
    **DONE 2026-09-05 in two steps — the decode readback collapse**
    ([record](records/ple-readbacks.md); multi-token batching stays unqualified and
    disabled) **and gate, signed sqrt, conv and silu on device for multi-token forwards,
    +12.8% prefill @3851** ([record](records/ple-device-tail.md); the default flip
    and the live decode tail are in the 2026-09-05 section at the top of this file).
    Note the rest of the decode cost is NOT this — it is table page faults, item (6).

    **(6)** PLE prefetch: at prefill every row address is
    computable from token ids before layer 0 runs (hash, dedupe, batch-fault on a
    background thread), and at decode the moment token t is sampled position
    t+1's ~16 rows are known — touch them while the trunk runs. Never gate the
    fetch on the PLE gate value: it is computed mid-forward, acting on it
    serializes the lookup and kills the prefetch, and unconditional retrieval is
    cheap. Test `madvise(MADV_RANDOM)` on the table mapping (default readahead
    turns a 90-160 B row into a large window) and MEASURE cold vs warm fault cost
    rather than assuming the page cache wins. **SHIPPED 2026-08-29 in ac40526 (D23), and
    the measurement came first: the decode gather is page faults with only 4.7%
    page-cache hits, so a background thread now touches one byte per distinct page for
    the position about to be forwarded — median decode gather 0.002 ms with prefetch
    against 0.97-1.02 without, decode 45.0 vs 43.2 tok/s.**
    [Record](records/flash-next-p3-kernels.md); `XWEN_PLE_NO_PREFETCH` /
    `XWEN_PLE_NO_RANDOM` for the A/B.

    prefill overlay materializes a `[n_q, n_kv]` mask~~ **CLOSED 2026-08-29 in
    the review round (643a411)**: prefill masks are now one f16 plane broadcast
    across heads on ALL checkpoints, a layout change with no math change, worth
    ~800 MB/layer at 4k on the 27B; **(8)** `IndexerCache`

    was a SCALING GUESS from the 35B-A3B, never a measurement — the real first
    number is 37.5-38.1, so either close the gap or retire the guess.
    **RETIRED 2026-08-29: decode is 43.1 tok/s measured against llama.cpp's 41.4 on the
    same file in the same hour, so the guess is not a target and should not be quoted.**
    [Record](records/flash-next-p3-kernels.md).

    **RESOLVED 2026-08-30: made load-bearing** — `officialModel` now requires every entry
    in `shards` and names the missing ones, so an interrupted 111 GB fetch no longer
    resolves as a cache hit and then fails deep inside the load.
    [Log](log.md#2026-08-30--flash-next-becomes-the-default-checkpoint-serve-falls-back-to-the-35b-a3b-and-says-so).

    The parity-gate half of this item is untouched and still open. **(13) NEW
    2026-08-29 — two review-noted low items in the fused hc path, knowingly not
    fixed.** `n == 0` is not bailed on in every fused entry point — no zero-token
    forward is reachable from the stack today, so this is defensive only. And the
    bit-identity assertions compare RAW BIT PATTERNS (`f32::to_bits`), which makes
    `-0.0` and `+0.0` different values: a reordered FMA that yields `-0.0` where
    the candle chain yielded `+0.0` would fail `split_matches_single_bitwise` and
    the write/activation bitwise tests as a mismatch, with nothing numerically
    wrong. That strictness is the right default — it is what makes "bit-identical"
    mean something — but if one of those tests ever fails on a zero, read the bit
    patterns before assuming a real divergence. Neither item is a live defect;
    both are recorded so the next person does not have to re-derive that they
    were seen and judged.

  - **2026-08-29 — P4 ledger for Flash-Next (what "experimental" currently
    means).** **Serve is REFUSED for this checkpoint** — as of 643a411 that
    refusal is enforced in code (`Model::servable()` false: startup refusal for
    both the registry entry and a custom qwen4exp GGUF, never listed, 400 on a
    request naming it; `auto_fetch()` and `supports_drafting()` false too), so
    this bullet is now the P4 STARTING POINT rather than a warning.
    **2026-08-30 annotation: Flash-Next is now the PLAIN DEFAULT**
    (`Model::default()`), so this refusal is what a zero-flag `xwen serve` hits
    — it falls back to `Model::default_servable()` (Qwen3.6-35B-A3B) and logs
    one line saying which and why. The three gates are unchanged; only the
    default moved. Closing this item makes `default_servable()` return
    `default()` and retires both the fallback and its line, so P4's definition of
    done now includes deleting them (the hub test asserts the two converge once
    the default is servable).
    **2026-08-30 second annotation: `xwen batch` IS IN THE SAME BOAT and is
    gated with serve.** It was ledgered as a mode that could run the checkpoint;
    it cannot. A batch prefills the items' shared prefix once and takes a cache
    snapshot there (`batch.rs` `run_batch`), and an enum-scored field snapshots
    and restores around every option it scores (`score_field`) — both
    `refuse_state_transfer` on qwen4exp, so a zero-flag batch would have failed
    after a 111 GB download and a full prefill. Until this item closes,
    `BatchRequest::model()` resolves an absent `"model"` to
    `Model::default_servable()` (with serve's own fallback line on stderr) and
    refuses a payload naming Flash-Next up front (`Model::unbatchable_message`).
    `XWEN_BATCH_NO_CACHE` is NOT a way around it: it skips the shared prefix and
    leaves the per-option snapshots. So closing this item also retires batch's
    fallback and its refusal, and `Model::servable()` — which now gates both
    surfaces — becomes true in one place for both. The narrower fix, if P4 slips,
    is teaching batch to run without either snapshot (cold items, and scored
    fields re-prefilled from the item's own prefix), which costs the prefill dedup
    that is the whole point of the mode. A qwen4exp
    target would 500 on the snapshot path, because prefix-cache
    snapshots, host snapshots and the disk tier do not carry the new recurrent
    state (indexer raw-key caches, PLE conv window, the 2-id token history) — D15
    took that decoupling deliberately in P2. Closing it means teaching
    snapshot/page-out/rewind about all three, INCLUDING new disk LAYER_* tags
    that correctly reject on old readers, and the 2-id history is sequence-level
    (store beside `CacheSnapshot::pos`, not per layer) and is u32 in an all-f32

    **SHIPPED 2026-08-30 — the cache images carry the qwen4exp state, and both gated
    surfaces are open: `refuse_state_transfer` and every refusal message are deleted,
    `Model::servable()` is true for every registry checkpoint and `default_servable()`
    returns `default()`, so `xwen serve` and `xwen batch` both run Flash-Next with no
    flags.** [Record](records/flash-next-serve.md), decisions.md "Qwen3.8-Flash-Next".

[Duplicate of the open item «GDN: 252 dispatches (13%), the three fusion candidates in the P3 ledger» (survivor's From: Flash-Next perf ledger, re-ranked from the measured budgets (2026-09-05, step 4)), folded and moved on 2026-09-06. It is filed here, under the heading the duplicate text came from, not under the survivor's.]

- [ ] [measured] **GDN dispatch count: three unstarted fusion candidates.** (The text below is verbatim from the parent item and may open or close mid-sentence.)
  The GDN block issues **288 dispatches per decoded token** [252 since the
  beta|alpha fold; recounted 2026-09-05] and the sweep's
  fit prices a dispatch at **8.41 µs of fixed cost regardless of size**, so
  36 dispatches ≈ 0.3 ms ≈ **~+1.5%** of a ~21.4 ms token [RE-PRICED
  2026-09-05: the launch floor measured on byte-free dispatches is 2.4-2.7 µs
  and the decode budget closes at ~4 µs average, so −36 is ≈ 0.15 ms ≈ +0.7%;
  the 8.41 intercept is the gemv's own ramp]. On the 35B-A3B the
  same arithmetic roughly doubles (30 layers against an 8.7 ms token), which
  is what the ba fold's +8.8% there against +4.6-4.8% here already showed. That arithmetic,
  not a bandwidth headroom argument, is what sizes what remains — three
  candidates, each its own kernel change, none started:
  - **conv+silu+state into the scan** (−36 dispatches) → **+1-2%**.
  - **gnorm+zgate into `out_proj`'s prologue** (−36) → **+1-2%**; the two are
  already 0.08 ms of profiled work, so this is dispatches only.
  - **the three Q8_0 projections (`attn_qkv`, `attn_gate`, `ssm_out`) as one
  multi-plane launch** (−72) → **+2-4%**; note the `XWEN_MOE_DUAL`
  precedent (decisions.md) — merging dispatches that were already
  saturating bandwidth in parallel LOSES, so this one needs an A/B before
  it is believed, and these three planes are bandwidth-saturating.
  Ranges are deliberately narrower than the ba fold's measured +4.8%: that
  fold displaced a dispatch that was ALSO doing real work badly (a candle f32
  gemv at 33 GB/s), which is not true of any of the three above.
  From: Qwen3.8-Flash-Next port (decided 2026-08-25, blocked on release + upstream).
  Promoted from the item «Port Qwen3.8-Flash-Next» on 2026-09-06; dated 2026-08-30 there.

[Duplicate of the open item «Hyper-connection carrier: 672 dispatches/token (35% of all launches), the largest population» (survivor's From: Flash-Next perf ledger, re-ranked from the measured budgets (2026-09-05, step 4)), folded and moved on 2026-09-06. It is filed here, under the heading the duplicate text came from, not under the survivor's.]

- [ ] [unpriced] **Hyper-connection follow-ups: in-place hc_write and the Q8_0 decode gemms.** (The text below is verbatim from the parent item and may open or close mid-sentence.)
  **(a)** `hc_write` is out-of-place; an
  in-place FMA would drop a full-carrier write per layer-pair; **(b)** at
  decode the two Q8_0 bottleneck gemms go through `QMatMul`, which has **no
  `mv_ext` plane** at the `hc.rs` qlinear site (gguf.rs:1631-1648) — try
  xwen's own vendored mv path there, the same move that paid on the 27B's
  projections. **SCREENED 2026-09-05, no route changed:** up needs a different
  kernel geometry; down still needs qualification. Results in
  [the PLE arc log](log.md#2026-09-05--ple-batches-its-three-device-to-host-readbacks).
  **(c)** decode is BIMODAL
  round over round and unexplained (its own item below); **(4)**
  From: Qwen3.8-Flash-Next port (decided 2026-08-25, blocked on release + upstream).
  Promoted from the item «Port Qwen3.8-Flash-Next» on 2026-09-06; dated 2026-08-29 there.

[Duplicate of the open item «`IndexerCache` still allocates at `max_ctx` up front and has no growth path, and page-in is now a second reason to care» (survivor's From: Deferred from the qwen4exp cache-image arc (2026-08-30, P4)), folded and moved on 2026-09-06. It is filed here, under the heading the duplicate text came from, not under the survivor's.]

- [ ] [measured] **IndexerCache allocates at max_ctx with no growth path.** (The text below is verbatim from the parent item and may open or close mid-sentence.)
  allocates at `max_ctx` with no growth path, ~1.6 GB across the 12 QSA layers
  at the checkpoint's 262144 ctx, paid whether or not the conversation gets
  there; **(9)** the ~50 tok/s decode figure in the port doc's P0-pause notes
  From: Qwen3.8-Flash-Next port (decided 2026-08-25, blocked on release + upstream).
  Promoted from the item «Port Qwen3.8-Flash-Next» on 2026-09-06; dated 2026-08-29 there.

[Orphan fragment of item (5) of the P4 numbered list in «Port Qwen3.8-Flash-Next» above, left dangling on the promoted item «QSA top-k runs on the host via arg_sort» and trimmed from it on 2026-09-06. It came from the numbered list, not from that item.]

; **(5)** the PLE gate, signed sqrt,

[Orphan fragment of item (7) of the P4 numbered list in «Port Qwen3.8-Flash-Next» above, left dangling on the promoted item «Parallelize the 16 page faults inside one PLE gather» and trimmed from it on 2026-09-06. It came from the numbered list, not from that item.]

; **(7)** ~~QSA mask memory — the

[Orphan fragment of item (12) of the P4 numbered list in «Port Qwen3.8-Flash-Next» above, left dangling on the promoted item «The PLE prefetcher spawns one thread per PleTable» and trimmed from it on 2026-09-06. It came from the numbered list, not from that item.]

 **(12) NEW 2026-08-29 —

[Orphan fragment of item (11) of the P4 numbered list in «Port Qwen3.8-Flash-Next» above, left dangling on the promoted item «Flash-Next decode is bimodal round over round» and trimmed from it on 2026-09-06. It came from the numbered list, not from that item.]

 **(11) NEW 2026-08-29 (Opus-2 review #5) — the PLE prefetcher

[Orphan fragment of the P4 cache-image bullet in «Port Qwen3.8-Flash-Next» above, left dangling on the promoted item «Flash-Next's unconsumed presence penalty, template divergences and audio specials» and trimmed from it on 2026-09-06. It came from that bullet, not from the item named.]

  plane world, so it needs its own plane type and validator. Also P4:

[Closed on 2026-09-06: the port shipped and Flash-Next is the default checkpoint, so this stub held no open scope. Its live sub-items were promoted into TODO.md across Decode performance, Prefill performance, Serve/batch/CLI, Cache images, Parity and Tokenizer; each names this heading in its From: line.]

- [ ] [blocked] **Port Qwen3.8-Flash-Next.** A PORT, not a registry entry — QSA sparse
  attention and the n-gram embedding subsystem are new. The closed narrative of this item
  is in the archive under its `From:` heading; the parts that stay open are their own
  items across the area sections, each naming that same heading.
  From: Qwen3.8-Flash-Next port (decided 2026-08-25, blocked on release + upstream).

[Duplicate of the open item «Above the 2048 indexer budget: +165 dispatches», folded and moved verbatim on 2026-09-06.]

- [ ] [unpriced] **QSA top-k runs on the host via arg_sort.** A device partial-top-k kernel
  is the intended replacement: D16 says selection is computed with candle ops in P2, and
  says explicitly that the top-k kernel is P3.
  From: Qwen3.8-Flash-Next port (decided 2026-08-25, blocked on release + upstream).
  Promoted from the item «Port Qwen3.8-Flash-Next» on 2026-09-06; dated 2026-08-29 there.

[Duplicate of the open item «Flash-Next still ships no drafter, and `supports_drafting()` stays false.», folded and moved verbatim on 2026-09-06.]

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

[Duplicate of the open item «No parity-gate or retune arm for Qwen3.8-27B.», folded and moved verbatim on 2026-09-06.]

- [ ] [small] **No ppl reference fixture for Qwen3.8-27B.**
  **2026-08-29.** Re-grading the
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

[Duplicate of the open item «The parity harness cannot run on qwen4exp.», folded and moved verbatim on 2026-09-06.]

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

## Deferred from the DeltaNet decode-scan pass (2026-08-30)

[Section prose of the pre-regroup ledger, moved verbatim on 2026-09-06.]

`kernel_delta_scan_decode` landed as an OPT-IN arm (`XWEN_DELTA_DECODE_KERNEL=1`,
the general kernel still runs seq == 1 by default) and measured as a wash
(log.md 2026-08-30 "later still", decisions.md "A decode-specialized scan kernel is a
WASH"). These are the two things it deliberately did not take.

[Item closed 2026-09-06 as a doc change: docs/benching.md now carries the caveat that the line ranks steps and does not time them.]

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

## Deferred from the first Flash-Next serve benchmark (2026-08-30)

[Section prose of the pre-regroup ledger, moved verbatim on 2026-09-06.]

Read-only bench at f949b1d; numbers and protocol in log.md ("serve on Flash-Next, first
benchmark"), the prefix-reuse rule it uncovered in decisions.md "Serving". Nothing was
changed, so everything here is a follow-up rather than a leftover.

## Deferred from the landscape research (2026-08-30)

[Nothing closed here yet; the heading exists because TODO.md items name it in their From: line.]

## Deferred from the metrics arc (2026-09-05)

[Section prose of the pre-regroup ledger, moved verbatim on 2026-09-06.]

Shipped that day: per-run JSONL records on every surface and `xwen stats` over them
(log.md "Per-run metrics on disk"; the choices in decisions.md "Metrics"). Nothing here
blocks use of the feature; item (a) is the one with a known trigger.

[Item shipped 2026-09-06 (7f0659e: the field and `--by agent`), moved verbatim.]

- [ ] [small] **`x-claude-code-agent-id` is not recorded.** Claude Code sends it on subagent
  requests. It was left out on purpose: recording it as the session would split one
  session into a row per agent, which is the wrong default for the question `--by
  session` answers. It is real information though, and "which subagent burned the
  tokens" is a question this history could answer. If it lands it wants its own field
  and its own `--by agent`, never a fallback inside `session_key`.
  From: Deferred from the metrics arc (2026-09-05).

[Item shipped 2026-09-06, moved verbatim.]

- [ ] [small] **The serve TUI does not show a job's model, though `JobRecord` now carries it.**
  The metrics work added `model` to the job record so a served run could name its
  checkpoint; the dashboard's job rows still do not display it. On a single-checkpoint
  server that is nothing, and on a server that lazy-swaps between checkpoints it is the
  one field that explains a row's rate. Cheap: a column in the job table, the field is
  already there.
  From: Deferred from the metrics arc (2026-09-05).

[Item shipped 2026-09-06, moved verbatim.]

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

## Deferred from the presence-penalty arc (2026-09-06)

[Nothing closed here yet; the heading exists because TODO.md items name it in their From: line.]

## Retired: Prefill performance

[Retired 2026-09-06: the chunked scan is refuted twice as a prefill lever (the fused sequential scan is 3% of 27B prefill, and llama.cpp's own Metal decomposition measured slower here) and its rollback-replay rationale is superseded by the K-snapshot verify. Reopen if: a per-stage profile contradicts the 3% figure, or a rollback design genuinely needs chunk-boundary replay.]

- [ ] [measured] **DeltaNet Metal kernels — (a) DONE 2026-07-28, (b) still open.** Original
   scope for (b): chunked prefill scan, chunk 64, llama.cpp's chunked form as the spec
   (cumsum → tri decay mask → solve_tri → per-chunk state update) — needs tri-solve which
   candle lacks; vendored kernel. Kill-switch XWEN_DELTA_CHUNK_CLASSIC falling back to the
   P3 reference. Gate: bitwise-or-bounded vs reference per parity.md tiering.
   - **(b) The chunked scan (chunk 64, tri-solve) remains open**, and its case is now
     weaker than it looked: the single-dispatch sequential scan already put prefill at
     ~2000 tok/s, so the chunked form is competing against that rather than against
     the 300 tok/s reference. Its real remaining argument is the rollback trail (see
     the archived P2-P4 model-core retarget section): a chunked scan that can replay a
     prefix cheaply would let the per-token trail be dropped entirely. Measure before building.
     ANNOTATION 2026-07-29: measured, and the picture splits by model — the weak-case
     reading holds on the 35B, while on the 27B the sequential scan looked like the cause
     of a 1.8-2.1x prefill loss to llama.cpp, making the chunked form's bounty ~2x there.
     [Log](log.md#2026-07-29--first-llamacpp-head-to-head-xwen-wins-decode-on-both-models-loses-27b-prefill-2x-to-the-sequential-deltanet-scan).
     ANNOTATION 2026-07-29 (later the same day): **the ~2x bounty is WITHDRAWN — that
     reading was wrong.** The fused scan is 3% of 27B prefill, so making it FREE moves
     prefill ~297 → ~307 tok/s against llama.cpp's 486; the gap is in the dense
     projections and needs its own item.
     [Log](log.md#2026-07-29--the-deltanet-scan-is-3-of-27b-prefill-llamacpps-decomposition-measured-slower-and-the-premise-behind-p8b-refuted).
     ANNOTATION 2026-07-29 (P8c): **that item was opened, root-caused and CLOSED the same
     day — the gap was the dense FFN's gemm (66-85% of 27B prefill wall through candle's
     `kernel_mul_mm_q4_K_f32`), and `src/ops/dense_mm.metal` fixed it.**
     [Record](records/dense-ffn-prefill-gemm.md), decisions.md "The dense-FFN
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

[Retired 2026-09-06: it only matters for short chunks and ragged tails and nothing has shown those shapes hot. Reopen if: a prefill probe shows the shexp or hc route hot at short chunks.]

- [ ] [small] **`DENSE_MM_MIN_SEQ` is unfitted for the shexp/hc shapes.** The 32-token
  floor was fitted on the 27B FFN (k 5120/17408); the shexp (k 2560/640, and
  2048/512 on the 35B) and hc (k 10240/320) routes inherit it unmeasured
  (2026-08-30). Only matters for short chunks and ragged tails; sweep
  `XWEN_DENSE_MM_MIN_SEQ` over those shapes if they ever show up hot.
  From: Deferred from the prefill-chunk pass (2026-08-30).

[Retired 2026-09-06: one sync per chunk, not per token, and the item itself rates it low value. Reopen if: a prefill probe prices the per-chunk readback above 1% of wall.]

- [ ] [unpriced] **QSA prefill still reads the scores back once per chunk per layer to build the mask on the host.**
  Prefill is `n > 1`, and it assembles the `[n_q, n_kv]` mask on the host rather than on
  the device. A device-side mask build (top-k per query row,
  then a fill kernel) would remove it; low value — one sync per chunk, not per token.
  From: Deferred from the prefill-chunk pass (2026-08-30).
  Promoted from the item «Decode on Flash-Next steps down ~11 ms/token the moment the context crosses the 2048-token QSA budget, then slopes gently» on 2026-09-06; dated 2026-08-30 there.

## Retired: Cache images, memory and context

[Retired 2026-09-06: Qwen ships no YaRN scaling keys in config or GGUF, the native window is 262144, and no workload here has asked for more. Reopen if: a real workload needs context beyond the native 262144 window.]

- [ ] [blocked] **YaRN long-context.** Native 262144; Qwen documents 1M via YaRN but ships no
    scaling keys in config or GGUF. laguna's YaRN rope code is retained; wire an
    opt-in flag only on demand. Note rope table memory at 262k is trivial (64 dims).
  From: Priority order (decided 2026-07-28; P1-P9 shipped by 2026-07-29).

[Retired 2026-09-06: periodic snapshots inside a 32k message cost hundreds of MB of host RAM to serve a prompt-editing workload nobody here runs. Reopen if: a client that edits prompts in place shows up, or the DeltaNet snapshot floor drops far enough to make an interval knob cheap.]

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

[Retired 2026-09-06: a workload that alternates checkpoints and misses its warm conversations does not exist here. Reopen if: a workload alternates checkpoints and loses warm conversations to the swap.]

- [ ] [blocked] **The disk tier serves only the default checkpoint.** A non-default checkpoint
  runs with every disk-tier call site handed `None`: the tier binds to one checkpoint
  id at startup and `verify()` permanently distrusts itself against any other. The
  segment layout is already per-checkpoint directories (`root/<checkpoint>/`), so the
  lift is opening one tier per checkpoint lazily rather than one at startup — do it
  when a workload actually alternates checkpoints and misses its warm conversations,
  not before. Until then a swap costs the outgoing checkpoint's warm slots and, with
  the tier on, keeps only the DEFAULT checkpoint's conversations across swaps.
  From: Deferred from the serve batch + multi-checkpoint arc (2026-08-11).

[Retired 2026-09-06: no blessed file hits it at any ceiling and the state is safe when it does. Reopen if: a misconfigured `max_ctx` fails mid-conversation for someone; the fix is a load-time advisory line.]

- [ ] [blocked] **Lazy KV moves the unaffordable-`max_ctx` failure from load time to
  mid-conversation.** Eager allocation failed fast at load; now the same misconfigured
  server starts fine and hits the allocation error at whatever depth exhausts the
  device — a growth step failing mid-request surfaces as that request's error (the
  state is safe and retries converge, `grow_kv_capacity`'s doc). `MEMORY_WARN_BYTES`
  (90 GiB) never fires for any blessed file even at the 262144 ceiling, so the warning
  is not the guard here. If this ever bites, the fix is a load-time advisory line
  ("ceiling X GiB exceeds device memory Y") rather than a return to eager allocation.
  From: Deferred from the client-feedback arc (2026-08-11).

## Retired: Drafting

[Retired 2026-09-06: outside the production regime (block_size 16 caps real verify spans near 17) and the 2026-08-08 evidence attributes it to armed-trail memory pressure, not a kernel threshold. Reopen if: a drafter whose block_size pushes real verify spans past 32, or an armed-trail memory regression in production.]

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
  [Log](log.md#2026-08-08--verify-round-diagnosis-the-149-ms-fixed-cost-is-the-dense-ffns-matmuls-at-small-m-and-none-of-the-armed-machinery-it-was-blamed-on).
  From: Deferred from the K-snapshot verify pass (2026-07-29, P9a).

[Retired 2026-09-06: custom-GGUF-only edge on a design target of blessed checkpoints whose sidecars always match, and `--no-draft` is the workaround. Reopen if: someone runs a custom GGUF whose sidecar fails preflight and wants plain decode instead of an error.]

- [ ] [blocked] **The draft-by-default flip makes a mismatched custom GGUF fail at startup.**
  With drafting opt-out, `xwen serve --model <custom.gguf>` whose geometry fails the
  drafter preflight (`DflashConfig::check_against_target`) now hard-errors where it
  previously ran plain; `--no-draft` is the workaround. Recommended shape when it
  bites someone: an IMPLICITLY-defaulted drafter that fails preflight should degrade
  to plain decoding with a warning, while an EXPLICIT `--draft` keeps the hard error.
  Not built now: the design target is the two blessed checkpoints, whose sidecars
  always match, so the edge is custom-GGUF-only.
  From: Deferred from the K-snapshot verify pass (2026-07-29, P9a).

[Retired 2026-09-06: the truncation argument holds for every shape shipped (shared prefix below the snapshot position), and the harder case only arrives with a prefix tree or concurrent sequences. Reopen if: the multi-level prefix tree or concurrent-sequence serving lands.]

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

[Retired 2026-09-06: no custom drafter for Qwen3.8-27B exists and the retune script cannot sweep one. Reopen if: someone runs a custom drafter on Qwen3.8-27B.]

- [ ] [blocked] **No fitted draft floor for a custom drafter on Qwen3.8-27B.** Exposed by the
  per-checkpoint `DraftMode` work of 2026-08-14
  ([record](records/serve-target-review.md),
  [decisions/serving.md](decisions/serving.md)), which settled the shape and left
  this floor behind. A custom drafter attached to a checkpoint with no fitted floor of
  its own falls back to `SpecParams::default().draft_p_min`, which is the 35B-A3B's fitted 0.3
  wearing a neutral name — an arbitrary value for that pair. If anyone actually runs a
  custom drafter on Qwen3.8-27B, fit a floor for it (`scripts/retune-draft.ts` cannot:
  it sweeps official sidecars only).
  From: Deferred from the MTP drafting arc (2026-08-15, stages B and C).

[Retired 2026-09-06, sub-item (e) of the open item «The DFlash drafter's per-token cache sync, its unre-derived draft-ctx horizon, and the deferred ring-buffer cache» in "Drafting": only worth it if `draft_ctx` grows a lot under (c), and (c) is taken with the 128k envelope work. Reopen if: `draft_ctx` is raised well past 8192 and the flat `[n_kv, max_ctx, hd]` allocation shows in the footprint.]

   - **(e) A ring-buffer drafter cache is deferred.** The per-layer cache stays a flat
     `[n_kv, max_ctx, hd]` array; windowing lives in `attention`'s narrow-plus-mask.
     A ring would cap the allocation at the window rather than at `draft_ctx`, but it
     would also stop `DrafterImage` being a straight prefix copy of the committed
     rows, which is what makes export/import and the disk tier simple. Only worth it
     if `draft_ctx` grows a lot under (c).

## Retired: Decode performance

[Retired 2026-09-06: worth at most ~1.4 ms at span 2 and possibly nothing, and the only evidence sits inside a known arm-ordering bias the item itself identifies. Reopen if: a bias-controlled A/B at spans 2-24 runs for another reason; add the two window floors as arms.]

- [ ] [measured] **Option: floor the `Proj::DenseF16Q8` window at t >= 3, leaving span 2 on the
  gemv.** Added 2026-08-08 from that pass's attention-projection coverage A/B. At t=2 the
  ext kernel saves only ONE gemv weight pass, and its geometry is fixed (nsg=2, nxpsg=8)
  rather than tuned for a two-row batch, so there is a plausible mechanism for it not to
  pay there. The evidence does NOT currently show a loss: span 2 measured +1.4 ms
  (+2.3%), but the same
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

[Retired 2026-09-06: outside the production regime, arming-independent, and the attention-projection coverage A/B came back a clean negative on it. Reopen if: a real verify span reaches 48, or the `mul_mv_ext` window routing is next changed.]

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
  [Record](records/small-batch-window-projections.md).
  From: Deferred from the small-batch mat-vec pass (2026-08-08).

[Retired 2026-09-06: in-place moves the same 3.1 MB in and out, the scan is already within 1.4x of that floor arm, and the only prizes need an aliasing promise no op-level function can make. Reopen if: a decode profile shows pool-allocation cost, together with a caller-supplied unaliased flag plumbed from `LinearAttnBlock::forward_fused`.]

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

[Retired 2026-09-06: the shipped prefetcher already reads a median gather of 0.002 ms against 0.97-1.02 without, so the overlap window this is conditioned on is not short. Reopen if: a gather profile shows the window is too short at longer contexts.]

- [ ] [unpriced] **Parallelize the 16 page faults inside one PLE gather.** Follow-up on the
  shipped PLE prefetcher, if its A/B shows the overlap window is too short: the 16 faults
  inside a single gather are taken serially on one thread — parallelize them across the
  window rather than deepening the lookahead.
  From: Qwen3.8-Flash-Next port (decided 2026-08-25, blocked on release + upstream).
  Promoted from the item «Port Qwen3.8-Flash-Next» on 2026-09-06; dated 2026-08-29 there.

[Retired 2026-09-06: unreachable while every published qwen4exp file asserts n_ple == 1. Reopen if: a multi-PLE checkpoint appears.]

- [ ] [small] **The PLE prefetcher spawns one thread per PleTable.** That is harmless on
  every published qwen4exp file, because upstream hard-asserts `n_ple == 1`, but the code
  does not depend on that assert: a checkpoint with several PLE layers would get a
  prefetch thread each, all faulting the same table. If a multi-PLE file ever appears,
  share one prefetcher across tables rather than one per layer.
  From: Qwen3.8-Flash-Next port (decided 2026-08-25, blocked on release + upstream).
  Promoted from the item «Port Qwen3.8-Flash-Next» on 2026-09-06; dated 2026-08-29 there.

[Retired 2026-09-06: each of its three parts is a deliberate non-take with no number and no user waiting, and the item itself states the condition under which each becomes worth taking. Reopen if: the epilogue learns to write both `ffn_out` and `l_out` when taps are on, a prefill probe prices the combine above 1% of wall, or the dual path ships on by default.]

- [ ] [measured] **Three leftovers from the MoE glue fusion: the residual add, the prefill
    combine, and `mul_mv_id_dual`'s unchecked ids.** The fusion itself shipped 2026-07-29
    ([record](records/fused-moe-glue.md), decisions.md "Kernel policy"); these are
    what it deliberately did not take.
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

## Retired: Research candidates

[Retired 2026-09-06: no comparative claim is being published, and the arm costs a second large-model process on a machine whose standing rule is one at a time. Reopen if: before publishing any claim of a lead in the 35B-A3B class on Apple silicon.]

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

## Retired: Serve, batch and CLI

[Retired 2026-09-06: the literature puts single-level collapse at roughly 1% of achievable reuse and no measured workload here loses on it. Reopen if: a measured workload where the single level demonstrably loses, or a system prompt shared across successive batch requests becomes a real cost.]

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

[Retired 2026-09-06: no caller needs machine-readable progress from a batch, and the single-document output is what makes `jq` over a batch trivial. Reopen if: a caller needs per-item results before the batch ends.]

- [ ] [small] **Results are not streamed.** `xwen batch` prints one JSON document when the last
  item finishes; progress goes to stderr as unstructured lines. A long batch therefore
  gives a caller nothing machine-readable until the end. NDJSON on stdout (one
  `ItemResponse` per line, `BatchStats` last) is the obvious shape and would not change
  the core, which already completes items in request order. Wants a flag rather than a
  format change — the current single-document output is what makes `jq` over a batch
  trivial.
  From: Deferred from the batch + scored-classification arc (2026-08-09).

[Retired 2026-09-06: no long batch runs over HTTP here, and the same shape as «Results are not streamed» would solve both. Reopen if: a long batch over HTTP is cut off by an idle-timeout proxy or a client needs progress.]

- [ ] [small] **Batch-over-HTTP gives no progress until the last item.** The CLI shows stderr
  progress lines; the HTTP client gets one JSON document at the end and nothing before
  it (a proxy that times out idle responses will cut a long batch off). The engine-side
  hooks already emit per-item progress into the server log, so an SSE or NDJSON variant
  of the route is wiring, not design. Related to the existing «Results are not
  streamed» item in this section — solve both with one shape when picked up.
  From: Deferred from the serve batch + multi-checkpoint arc (2026-08-11).

[Retired 2026-09-06: no client asks for token-level evidence; the scored path answers the label-level question that is asked. Reopen if: a client needs `logprobs`/`top_logprobs` on emitted tokens.]

- [ ] [small] **Per-token logprobs are not exposed in any dialect.** `include_score` reports
  confidence over a field's ALLOWED OPTIONS, which is a different quantity from
  OpenAI's `logprobs`/`top_logprobs` (raw log-softmax over the vocabulary at each emitted
  position, top-k of it). The machinery for both now exists — `Generator::last_logprobs_for`
  is the log-softmax over an encodable slice — but the two must not be conflated in the
  surface: a client asking for `logprobs` wants token evidence, not label evidence.
  Independent of the scored path; belongs with the serve adaptation.
  From: Deferred from the batch + scored-classification arc (2026-08-09).

[Retired 2026-09-06: an accuracy refinement of the scored classifier that nobody has asked for; away from near-ties the channels agree. Reopen if: a scored near-tie on a boolean is read as a confident answer, or a consumer needs the channels summed.]

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

[Retired 2026-09-06: the README definition stands and consumers can compare escape across categories with and without first fields; the split wants the channel-summing refinement derived alongside it. Reopen if: a consumer needs value-escape separated from format drift, together with the channel-summing refinement.]

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

[Retired 2026-09-06: each of the four lifts is a scope decision with no schema waiting on it. Reopen if: a schema someone needs is refused by the shape guard.]

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

[Retired 2026-09-06: both checkpoints are cached on this machine and the download resumes on retry. Reopen if: a request names an uncached checkpoint and races the watchdog, or the hf-hub progress bar draws over the TUI.]

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

[Retired 2026-09-06: deliberate: the native generate surface is modelless by design and the batch route selects. Reopen if: a native-API consumer wants the non-default checkpoint through `/xwen/v1/generate`.]

- [ ] [blocked] **`/xwen/v1/generate` carries no model field.** Deliberate — the native generate
  surface documents itself as modelless and the batch route is the native surface that
  selects — but it is now the only route that cannot reach the non-default checkpoint.
  Add the field if a native-API consumer ever wants it; it is a two-line change in
  `prepare` plus tests.
  From: Deferred from the serve batch + multi-checkpoint arc (2026-08-11).

[Retired 2026-09-06: exposure is bounded by one short item plus one prefill and no workload has a span long enough to care. Reopen if: a workload has scored spans or prefills long enough that cancel latency matters.]

- [ ] [blocked] **Mid-batch cancellation does not reach the scored path's forced spans, nor any
  prefill.** The cancel poll runs between items and per decoded token inside an item's
  free decode; a scored item's teacher-forced assembly checks only at item boundaries,
  and neither the shared-prefix prefill nor an item's own tail prefill polls at all
  (`prefill_tokens` chunks internally but takes no callback). Items are short (≤192
  tokens in the demo), so the exposure is bounded by one item's latency plus one
  prefill — thread the poll through `assemble_scored` and the prefill chunk loop only
  if a real workload makes either span long enough to care.
  From: Deferred from the serve batch + multi-checkpoint arc (2026-08-11).

[Retired 2026-09-06: on a single-user box the age limit bounds the damage and fairness under mixed traffic is not a workload here. Reopen if: scheduling fairness under real mixed traffic matters.]

- [ ] [blocked] **The batch scheduling estimate is bytes-based and can read zero.** A batch of
  items with empty message content (schema-only probes) estimates zero prompt tokens
  and schedules as free; the real cost floor is the rendered template per item. Fold a
  per-item constant into `size_estimates` (or estimate from the rendered skeleton)
  when scheduling fairness under real mixed traffic matters; on a single-user box the
  age limit already bounds the damage.
  From: Deferred from the serve batch + multi-checkpoint arc (2026-08-11).

[Retired 2026-09-06: a single-user machine does not interleave checkpoints in its queue. Reopen if: a real workload interleaves checkpoints and pays the swap per pickup.]

- [ ] [blocked] **The scheduler does not group queued jobs by checkpoint.** `shortest-prefill`
  scores by prefill cost alone, so a queue holding jobs for both checkpoints can pick
  them interleaved and pay a ~3 s swap per pickup where checkpoint-grouped ordering
  would pay two. The cost model could add the swap (a job for the non-resident
  checkpoint costs its prefill plus a load-equivalent), which also naturally batches
  same-checkpoint work without starving the other (the age limit already guards
  starvation). Do it when a real workload actually interleaves checkpoints; a
  single-user machine mostly will not.
  From: Deferred from the serve batch + multi-checkpoint arc (2026-08-11).

[Retired 2026-09-06: the default bind is loopback on a single-user machine. Reopen if: the server fronts a LAN under `api_key`.]

- [ ] [blocked] **100 MB bodies are buffered with no concurrency bound.** The batch handler
  buffers and serde-parses the whole body (typically 2-5x the text in tree form)
  BEFORE `submit_batch` can answer 429, and nothing caps concurrent connections — N
  clients can each hold ~100 MB + parse tree against 19-37 GB of resident weights.
  Accepted for now: the default bind is loopback on a single-user machine, and the
  compat dialects never need large bodies. If the server ever fronts a LAN under
  `api_key`, add a concurrency-limit layer (or move the cap per-route: 100 MB for
  `/xwen/v1/batch`, default for the dialects) before raising anything else.
  From: Deferred from the client-feedback arc (2026-08-11).

[Retired 2026-09-06: the dialect has no natural field for it and the OpenAI and native dialects carry it. Reopen if: Anthropic's API grows an effort field to mirror.]

- [ ] [blocked] **The Anthropic dialect has no per-request template-effort knob.** Its API shape
  has no natural field: `thinking.budget_tokens` is a budget, not a level, and inventing
  a nonstandard field on a compat dialect defeats the point of speaking the dialect.
  Requests get the server-wide `[thinking] effort` default (which `count_tokens` also
  renders, so counts match generation); a client that needs per-request effort on 3.8
  uses the OpenAI or native dialect. Revisit only if Anthropic's API grows an effort
  field to mirror.
  From: Deferred from the chat-dialect and sampling-defaults arc (2026-08-19).

[Retired 2026-09-06: `XWEN_METRICS_FILE` reaches every surface and no server here needs its history somewhere specific. Reopen if: a server deployment needs a per-deployment metrics path or recording off for a benchmark front.]

- [ ] [small] **A `[metrics]` table in serve.toml (path, enabled).** `XWEN_METRICS_FILE` is the
  only control and it reaches all four surfaces, which is why it shipped alone
  (decisions.md). A server is the one surface with a config file and the one that runs
  long enough for an operator to want its history somewhere specific — a per-deployment
  path, or recording off for a server that fronts a benchmark. The table would override
  the variable for serve only; keep the variable as the surface-wide answer.
  From: Deferred from the metrics arc (2026-09-05).

[Retired 2026-09-06: under 10 MB a year at a hundred runs a day and the full-scan reader is comfortable well past that. Reopen if: the metrics file is large enough to be worth measuring (tens of MB).]

- [ ] [blocked] **The file grows forever; there is no rotation or compaction.** A line is ~250
  bytes, so this is a slow problem: a hundred runs a day is under 10 MB a year, and the
  full-scan reader stays comfortable well past that. `--since` bounds what a report
  reads over, not what the file holds. Nothing in xwen prunes, and that is deliberate
  for now (the durable-history choice is the whole reason it is not in the cache dir),
  but a machine left running for years wants either a size-triggered roll to
  `metrics.jsonl.1` or a compaction that folds records older than N months into daily
  summaries. Decide when the file is big enough to be worth measuring, not before.
  From: Deferred from the metrics arc (2026-09-05).

[Retired 2026-09-06: nothing moves away from midnight and the machine mostly runs in one season. Reopen if: a report is misread because of a clock change.]

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

## Retired: Parity, provenance and tooling

[Retired 2026-09-06: empirically identical on this machine, and the bitwise ops tests plus the strict parity tier are the tripwire. Reopen if: a bitwise ops test or the strict tier fails with no other cause; suspect the compile axis first.]

- [ ] [unpriced] **Bit-identity claims ride an unpinned Metal compile axis (candle compiles Fast, vendored kernels do not).**
  Found by outside-model review, 2026-07-29. candle rev 21cca0b compiles its kernels with BOTH
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

[Retired 2026-09-06: nothing has needed an attention-side bench at Qwen geometry, and the deleted versions are in git history at the fork point. Reopen if: the 27B decode budget is next attacked and needs an attention-side bench.]

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

[Retired 2026-09-06: a watch item whose tripwire is the ppl gate itself; nothing to do until it moves. Reopen if: the ppl tier fails or the 35B delta rises past 0.0015 (it sits at 0.000791 against a 0.002 floor).]

- [ ] [measured] **The 35B's perplexity delta grew with the fused DeltaNet scan and the floor's
  margin shrank.** `PPL_NLL_DELTA_MAX = 0.002` stands and is not re-derived from the
  fused measurement (RESOLVED 2026-07-28 by the parity owner;
  [rationale and trip-wire](parity.md#perplexity-gate)).
  Still open as a WATCH item: the fused scan sits at 0.000791 on the 35B, so a
  further ~2.5x rise fails the gate, and the sign is systematic (candidate worse in
  all four measurements across both architectures — the fused scan widened the gap
  ~+50% on each). This is the single most sensitive number the gate reports about the
  fused scan; the cosine tiers barely moved (35B mm actually improved, 0.999540 →
  0.999631).
  From: Deferred from the fork bootstrap (2026-07-28).

[Retired 2026-09-06: accepted by decision and no near-tie flip has been suspected in production. Reopen if: a near-tie flip is suspected in production; measure it before blaming sampling.]

- [ ] [blocked] **Partition-parity drift never measured.** The q8/f16 dual-storage split makes
  cached state depend on call partitioning (see decisions.md "Kernel policy" entry,
  2026-07-28). Accepted by decision, but the drift magnitude at the 8↔9 boundary on
  real weights has never been quantified — measure it (same prompt, cache on/off,
  compare state and downstream logits) if a near-tie flip is ever suspected in
  production, before blaming sampling.
  From: Deferred from the fork bootstrap (2026-07-28).

[Retired 2026-09-06: ggml-org was chosen on provenance and output quality has not come into question. Reopen if: output quality comes into question; point the ppl gate at a competing Q4_K_M.]

- [ ] [blocked] **Quant-vendor comparison never measured.** ggml-org was chosen over
  unsloth/bartowski on provenance (converter authors, inspectable custom mix, dflash
  sidecars), not on quality. Now that the perplexity gate exists, pointing it at a
  competing Q4_K_M is cheap — run it if output quality ever comes into question.
  From: Deferred from the fork bootstrap (2026-07-28).

[Retired 2026-09-06: both production extents are 4 and no checkpoint with a wider indexer geometry exists. Reopen if: a checkpoint with `ratio` or indexer head count above 5.]

- [ ] [small] **`strided_sum`'s reduce-order replay refuses extents above 5, so a wider indexer geometry fails at `select`.**
  The reason is candle's reducer, which
  folds through a 4-lane `simd_sum` there so the bit-identity breaks (1 ulp at
  extent 6). Both production extents are 4; a checkpoint with `ratio` or indexer
  head count above 5 would fail at `select` and needs either the plain `sum` (bounded,
  not bitwise) or a widened replay.
  From: Deferred from the prefill-chunk pass (2026-08-30).
  Promoted from the item «Decode on Flash-Next steps down ~11 ms/token the moment the context crosses the 2048-token QSA budget, then slopes gently» on 2026-09-06; dated 2026-08-30 there.

[Retired 2026-09-06: the CPU/oracle path is not graded on Flash-Next today, so the silent fallback has no consumer. Reopen if: the parity harness runs on Flash-Next (the «The parity harness cannot run on qwen4exp» item), which grades the CPU/oracle path.]

- [ ] [small] **The fused QSA gather is Metal-only and alignment-restricted; the CPU/oracle path silently takes the `index_select` chain.**
  A non-Metal source (the CPU/oracle attention
  path) takes that chain with no switch. The kernel refuses a view
  whose start or head stride is not a multiple of 4 elements (vec4 loads).
  From: Deferred from the prefill-chunk pass (2026-08-30).
  Promoted from the item «Decode on Flash-Next steps down ~11 ms/token the moment the context crosses the 2048-token QSA budget, then slopes gently» on 2026-09-06; dated 2026-08-30 there.

## Retired: Tokenizer, chat and sampling

[Retired 2026-09-06: text tokenizes identically and no 3.8 or Flash-Next reply has ended strangely; the item already names its own reopen condition. Reopen if: a 3.8 or Flash-Next reply ends strangely for no visible reason, or one of ids 248070-248076 is observed in a sample.]

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

[Retired 2026-09-06: a pointer item: its penalty half is taken by «The cards' recommended penalties (presence_penalty 1.5) are not implemented», its harness and drafter halves are their own items, the audio specials retire with the 3.8 tokenizer item, and the Unsloth template divergences (tool calls, developer role, multiple leading system messages, `effort=high`) have no user rendering those shapes. Reopen if: a Flash-Next chat uses tools, a developer role, several leading system messages or `effort=high` and renders differently from `reference/chat_template-qwen38.jinja`.]

- [ ] [unpriced] **Flash-Next's unconsumed presence penalty, template divergences and audio specials.**
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

## Deferred from the long-context envelope arc (2026-09-06)

[Opened 2026-09-06 by the arc recorded in docs/records/long-context-envelope.md. Nothing closed here yet; the heading exists so this arc's items have somewhere to land.]

[Duplicate of the open item «QSA prefill selection round-trips through the host per sparse layer per chunk, and it is the Flash-Next long-context tax», folded and moved verbatim on 2026-09-06; the memory half is carried there, its throughput conclusion is contested there.]

- [ ] [measured] **The QSA indexer's host-built mask is 42 GB of Flash-Next's peak footprint at
  131072.** **2026-09-06.** Moving the CAUSAL prefill mask to the device took the 35B's
  131072 peak from 42-69 GB to a flat 17 GB, and moved Flash-Next's not at all (59 GB
  either way): its indexer builds its own `n x n_kv` f32 mask on the host through
  `Tensor::from_vec` (`qwen4exp/indexer.rs`, `select_with`), per sparse layer per chunk,
  and above the 2048 budget that is every chunk — so no two allocations ask the pool for
  the same size and none is recycled. The fix is the same shape that worked for the
  causal one: build the -inf plane on the device and scatter zeros at the selected
  columns, same values, behind a kill switch. Do NOT expect throughput from it — the
  causal mask's own A/B was a dead heat on time on both checkpoints, because candle fills
  the next chunk's mask while the GPU is still on this one. This is a memory item.
  [Record](records/long-context-envelope.md).
  From: Deferred from the long-context envelope arc (2026-09-06).

[Shipped 2026-09-06 by the arc recorded in docs/records/qsa-device-mask.md: the prefill selection and the mask now build on the device (`kernel_qsa_select_mask`, kill switch `XWEN_QSA_HOST_MASK`), and the round trip was priced first with `XWEN_QSA_TIMER`. No open remainder here; the decode-side cousins stay folded into «Above the 2048 indexer budget: +165 dispatches» in TODO.md "Decode performance".]

- [ ] [measured] **QSA prefill selection round-trips through the host per sparse layer per chunk, and it is the Flash-Next long-context tax.** Flash-Next prefill falls 925 → 584 → 403 → 231 tok/s from 8k to
  128k (2026-09-06, [record](records/long-context-envelope.md)), i.e. 1.08 → 4.33
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
