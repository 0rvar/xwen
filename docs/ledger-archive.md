# Ledger archive

Closed items of [TODO.md](../TODO.md), moved here verbatim with their annotations when the
arc that closed them ended (first sweep 2026-09-06). Section headings are the ledger's own,
so a section name quoted anywhere still greps here. Nothing in this file is actionable; the
live ledger is TODO.md. Items are never deleted: they move here.

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

## Priority order (decided 2026-07-28; P1-P9 shipped by 2026-07-29)

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

