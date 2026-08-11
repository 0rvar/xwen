# Deferred work ledger

Items are never deleted, only annotated with their outcome in the item's own bold
header (`DONE <date>`, `CLOSED-REFUTED <date>`, …) plus the measurement that closed
them and a pointer to the log.md entry with the full arc. Sub-items are lettered under
a numbered parent.

## Priority order (decided 2026-07-28; P1-P9 shipped by 2026-07-29)

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

8. **DeltaNet Metal kernels — (a) DONE 2026-07-28, (b) still open.** Original scope:
   (a) fused recurrent decode step (one dispatch per layer
   per token; state stays resident, fp32); (b) chunked prefill scan, chunk 64,
   llama.cpp's chunked form as the spec (cumsum → tri decay mask → solve_tri →
   per-chunk state update) — needs tri-solve which candle lacks; vendored kernel.
   Kill-switches XWEN_DELTA_CLASSIC / XWEN_DELTA_CHUNK_CLASSIC falling back to the P3
   reference. Gate: bitwise-or-bounded vs reference per parity.md tiering.
   - **(a) SHIPPED, and it covered prefill too.** `src/ops/delta.metal` +
     `src/ops/delta.rs`: four kernels — conv+silu+next-window, the fused beta/decay
     head over a load-time-concatenated `[hidden, 2*v_heads]` beta|alpha projection,
     the gated output RMSNorm, and one scan kernel that runs the WHOLE recurrence for
     T timesteps in a single dispatch with the head's state slice resident in
     registers across the scan. A DeltaNet layer is 8 dispatches at any sequence
     length (was ~65 per decoded token, ~8·T per prefill chunk). 35B-A3B, low-power
     off (`lowpowermode 0`; no `highpowermode` key is exposed on this machine, so the
     High Power tier is neither confirmed nor available — these are not laguna's
     "full power" anchors), warm, interleaved A/B, median of 3: decode 57.8 → 91.2 at a
     596-token prompt and 56.6 → 88.0 at 1929; prefill 305 → 2183 (7.15x) and
     300 → 2274 (7.57x). Kill-switch `XWEN_DELTA_CLASSIC=1`. See log 2026-07-28 and
     decisions.md "Model math".
   - **(b) The chunked scan (chunk 64, tri-solve) remains open**, and its case is now
     weaker than it looked: the single-dispatch sequential scan already put prefill at
     ~2000 tok/s, so the chunked form is competing against that rather than against
     the 300 tok/s reference. Its real remaining argument is the rollback trail (see
     the P2-P4 deferred item): a chunked scan that can replay a prefix cheaply would
     let the per-token trail be dropped entirely. Measure before building.
     ANNOTATION 2026-07-29: measured, and the picture splits by model. On the 35B the
     weak-case reading holds (prefill near llama.cpp parity at steady state). On the
     27B the sequential scan — fused or not — is the measured cause of a 1.8-2.1x
     prefill loss to llama.cpp (269 vs 486 @925, 236 vs 502 @4k, and xwen DEGRADES
     with length while llama.cpp improves): 48 layers at inner 6144 amplify what 30
     layers at 4096 hide. The chunked form's bounty is therefore ~2x on 27B prefill,
     not a marginal 35B win. See log.md 2026-07-29 head-to-head.
     ANNOTATION 2026-07-29 (later the same day): **the ~2x bounty is WITHDRAWN — that
     reading was wrong, and the measurement that would have caught it had never been
     taken.** The fused scan is 3% of 27B prefill: 48 layers × 1.97 ms is 95 ms of a
     2.96 s prefill at 880 tokens, 48 × 8.56 ms is 411 ms of 14.2 s at 3851. Making
     the scan FREE moves 27B prefill from ~297 to ~307 tok/s against llama.cpp's 486,
     so no scan form — chunked, re-decomposed, or absent — can be the 1.8-2.1x gap.
     The gap is in the dense projections and needs its own item.
     ANNOTATION 2026-07-29 (P8c): **that item was opened, root-caused and CLOSED the
     same day — the gap was the dense FFN's gemm, and it is fixed.** The profiling
     pass put 66-85% of 27B prefill wall time in the dense SwiGLU FFN (64 layers,
     17408-wide, Q4_K) running through `QLinear` → candle `QMatMul` →
     `kernel_mul_mm_q4_K_f32` at ~12-13 TFLOP/s, against 28-36 TFLOP/s for the same
     shapes on the Metal-4 cooperative-tensor gemm. (A band because that budget row
     is derived from an isolated rate ~7-8% pessimistic against a real forward.) The
     gap is kernel efficiency, not bandwidth: the Q4_K arm moves 3.6x FEWER weight
     bytes and takes 2.4x LONGER, and a bandwidth-bound arm moving fewer bytes would
     be the faster one. `src/ops/dense_mm.metal`
     is the dense cooperative-tensor gemm reading Q4_K directly with an in-kernel
     tile dequant, gated at `seq > DENSE_MM_MIN_SEQ` (32). Kernel-level 2.4-3.0x at a
     512-token chunk; end-to-end numbers, the rejected dequant-to-scratch
     alternative, and the precision cost are in log.md 2026-07-29 and decisions.md
     "The dense-FFN prefill gemm dequantizes in-kernel".
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
   - The refuted re-decomposition is kept runnable, not deleted:
     `XWEN_DELTA_SCAN_V2=1` selects `kernel_delta_scan_v2` (llama.cpp's shape) plus
     the `kernel_delta_l2norm` dispatch it needs, on the `XWEN_MOE_DUAL` precedent.
   - `XWEN_DELTA_CHUNK_CLASSIC` was never created — there is no chunked path to
     switch off yet. It belongs with (b).
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

9. **DFlash adaptation to the Qwen sidecars — ADAPTED 2026-07-29, but speculation is
   a 27B-only win and stays opt-in.** Both sidecars load, draft and verify correctly
   at 85-95% acceptance; the two deliberately-red tests are green and the suite is
   760/0. Measured (`lowpowermode 0`, warm, greedy, 128 tokens, interleaved, 3 reps):
   27B **+4.8 to +6.8%** on a code prompt (26.4-26.7 vs 25.0-25.2 tok/s) and **+1.5 to
   +7.4%** on a chat prompt, quoted as ranges across two independent runs because the
   27B's between-run level shifts even though its within-run reps are tight; 35B-A3B
   **-11.5%** and **-12.7%** (93.0/92.1 vs 105.1/105.5, repeating to within 1%).
   `draft_p_min` retuned 0.5 → 0.3 (only value ahead of plain on both
   prompt kinds in every run); `pause_margin` stays 1.0. See log.md 2026-07-29 and
   decisions.md "Speculative decoding". Original scope: repoint drafter arch check
   (arch `dflash`, decoder arch qwen35/qwen35moe), tap indices from `target_layers`
   metadata, mask_token_id, sliding-window pattern; verify the fc.weight geometry
   (5×hidden / 8×hidden concat). Needs P4's recurrent-state rollback. Re-tune
   auto-pause and draft-ctx horizon for this drafter's cost curve.
   - **(a) The K-snapshot fused verify is the precondition for speculation to pay,
     not an optimization of it — the top open item under P9. DONE 2026-07-29** (same
     day, see log.md "K-snapshot fused verify lands"): built exactly as sketched
     below — `delta_scan_with_trail` widens the state output to most-recent-first
     planes in both scan kernels, plane 0 stays the unchanged after-loop store
     (planes = 1 bitwise-identical, tested), the armed clause is gone from the fused
     gate, `XWEN_DELTA_CLASSIC` unchanged as kill switch. Two-model review clean,
     both parity gates pass with pre-change numbers. Measured: verify marginal cost
     9.42 → 3.57 ms/position (fixed ~171 → ~149 ms); end-to-end 27B +19.3-21.0%
     code / +7.6-8.4% chat, 35B +18.1-19.8% / +12.6-12.8% (was -11.5/-12.7% — the
     pause controller stopped pausing, see (d)). The retired fallback's successor
     items live under "Deferred from the K-snapshot verify pass (2026-07-29)".
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
   - **(d) The `--draft` opt-out flip is DEFERRED with numbers.** The flip was
     conditional on the controller holding a never-materially-slower property on both
     checkpoints; it does not, because the 35B's 12% loss lands on rounds the
     controller has already paused (see (b)). Re-evaluate after (a) and/or (b): the
     bar is the 35B at or above plain, not merely closer to it.
     RESOLVED 2026-07-29: **flipped — drafting is now the default** (`--no-draft`
     opts out). (a) alone met the bar with margin: 35B +18.1-19.8% code /
     +12.6-12.8% chat over plain, both prompt kinds, two independent runs. The (b)
     attribution was measured right but read wrong — the ~1.2 ms/token cache sync
     is only fatal on PAUSED rounds, and with verify cheap the controller stopped
     pausing (35B code: 54-of-66 rounds paused → 0-of-20). (b) stays open as a
     lever, no longer as the gate. Zero-flag `generate`/`serve` now load the
     dflash sidecar; help/config text updated with the measured reason.
   - **(e) A ring-buffer drafter cache is deferred.** The per-layer cache stays a flat
     `[n_kv, max_ctx, hd]` array; windowing lives in `attention`'s narrow-plus-mask.
     A ring would cap the allocation at the window rather than at `draft_ctx`, but it
     would also stop `DrafterImage` being a straight prefix copy of the committed
     rows, which is what makes export/import and the disk tier simple. Only worth it
     if `draft_ctx` grows a lot under (c).
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

10. **serve adaptation.** Tool-call parsing for the `<function=...>` XML-ish format in
    both API dialects (string args raw, non-string JSON), thinking-mode flags
    (enable_thinking / preserve_thinking) surfaced per dialect, prefix-cache + disk
    tier snapshots extended with recurrent state (48–96 KiB conv + 2–6 MiB delta per
    snapshot depending on model). Estimated-prefill scheduling unchanged.

11. **27B dense bring-up — MOSTLY DONE 2026-07-28 via P7.** The parity gate ran the
    27B end to end: first forward correct, all gated tiers pass (strict is
    near-vacuous on the dense model — see P7c). Remaining: an interactive
    generate/chat smoke run, decode/prefill perf numbers for the 27B (nothing
    measured yet; 64 layers dense will be much slower per token than the A3B), and
    the deferred conv threadgroup-sizing check when P8 lands.
    - ANNOTATED 2026-07-28 (P8a): the 27B perf gap is now filled. Low-power off
      (`lowpowermode 0`; the High Power tier is not exposed on this machine), warm,
      batch 1, interleaved A/B, median of 3: decode 19.0 tok/s at a 596-token
      prompt and 17.9 at 1929; prefill 290.4 and 209.3 tok/s. It is ~4.7x slower per
      decoded token than the 35B-A3B, as expected from 64 dense layers at hidden
      5120. The fused DeltaNet kernels bought it 1.25-1.33x decode and 2.7-3.8x
      prefill; its per-token budget is dominated by the dense SwiGLU, not by dispatch
      count, so the next 27B lever is the FFN, not more glue fusion. Still open: the
      interactive smoke run.
    - CAVEAT on those 27B numbers: its per-rep spread is materially wider than the
      35B's (the 596-token fused decode walked 21.7/19.0/17.9 across three reps as
      the machine heated, against a 35B classic arm that repeated to within 0.8%).
      Treat the 27B figures as ±10% and re-measure off an idle machine before using
      them as a baseline for anything. See decisions.md "Measurement discipline".
    - The conv threadgroup-sizing worry (below, "the 27B linear-attn conv runs over
      10240 channels") turned out not to bind: the fused conv kernel is a flat
      one-thread-per-element launch through the shared `dispatch_linear` helper, so
      channel count only sets the grid size. Closed by construction, not measurement.

12. **MTP exploration — DEFERRED by decision.** Sidecar reuses parent arch, one extra
    full-attn block + nextn.* tensors, plain KV cache, eh_proj over
    [norm(emb);norm(h)]. Evaluate as drafter only after P9 lands or fails (see
    decisions.md "Speculative decoding").
    - ANNOTATED 2026-07-29: P9 landed, and the trigger is still not met. DFlash's
      acceptance is 85-95% — a better drafter would not help. What limits xwen's
      speculation is the verify forward's cost (P9a) and the drafter cache sync
      (P9b), and an MTP drafter would pay both identically. Do not open this until
      P9a lands and the win is measured with a fast verify.

13. **MoE block glue fusion — SHIPPED 2026-07-29.** An MoE layer went from 24
    dispatches per decoded token to 14 (960 → 560 across the 40 layers), and 35B-A3B
    decode from 92.6 to 102.8 tok/s (+11.0%, `lowpowermode 0`, warm, interleaved,
    median of 5, arms non-overlapping). Three fusions, all bit-identical to the candle
    chains they replace, behind `XWEN_MOE_GLUE_CLASSIC=1`: `kernel_moe_router`
    (softmax → bitonic arg-sort → gather → sum → clamp → renormalize, 7 dispatches → 1),
    `kernel_moe_epilogue` (weighted combine + shared-expert sigmoid gate + its multiply
    + the routed+shared add, 4 → 1), and the shared expert's `silu(g)*u` moved onto the
    existing `ops::silu_mul` (2 → 1). Both parity gates pass with numbers identical to
    the pre-change run, so the schema is untouched. See log 2026-07-29 and decisions.md
    "Kernel policy".
    - **The two router matmuls stay candle dispatches** — MLX's `gemv_t` accumulation
      order is not reproducible from a differently-shaped hand-written gemv, and it
      depends on the output width, so concatenating the shexp gate row onto the router
      weight would have changed that gate's bits. Costs one dispatch of the ten saved.
    - **The residual add was NOT fused in, on purpose.** The briefed design folded
      `model.rs`'s `x + ffn_out` into the epilogue for a twelfth dispatch. That would
      delete the `ffn_out` tap, which docs/parity.md lists with published per-layer
      floors on both checkpoints — or force the gate onto the classic path, where it
      would never exercise the fused epilogue at all. Worth ~40 dispatches per token
      (one per layer) if someone later teaches the epilogue to write both `ffn_out` and
      `l_out` when taps are on; not worth a provenance hole.
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
    - **Prefill is untouched and was not attacked.** Above `MM_ID_MIN_SEQ` the epilogue
      declines (the f16-tile projection carries an L2 rescale it has no term for) and
      only the router kernel and the shared-expert activation fuse; measured +0.6% at
      925 tokens and +0.2% at 4k, i.e. nothing. Fusing the prefill combine would mean
      an epilogue variant carrying the rescale — cheap to write, but prefill is
      compute-bound at ~2100-2500 tok/s and there is no evidence it would show.
    - **The bit-identity claims ride an unpinned compile axis** (outside-model review,
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
    - **`mul_mv_id_dual`'s wrapper trusts its ids buffer.** It validates rank, dtype,
      contiguity and dims but not that each id is < n_expert (values live on the GPU;
      checking means a readback) nor that ids/gate/up share x's device. Fine for the
      router-produced ids of its only caller, loose for a `pub` API. Harden if the
      dual path ever ships on by default.

14. **YaRN long-context.** Native 262144; Qwen documents 1M via YaRN but ships no
    scaling keys in config or GGUF. laguna's YaRN rope code is retained; wire an
    opt-in flag only on demand. Note rope table memory at 262k is trivial (64 dims).

## Deferred from the P2-P4 model-core retarget (2026-07-28)

- [ ] **The attention and full-stack decode/prefill benches were deleted, not
  ported.** `attention.rs`'s `tests::decode_bench` + `tests::prefill_bench`
  modules (~1300 lines) and `moe.rs`'s `full_stack_decode_bench` measured
  Laguna's attention chain — 48/72 heads, head dim 128, SWA rings, softplus
  per-head gate — none of which exists in Qwen 3.6, and they were the only
  consumers of the SWA-geometry test scaffolding. The MoE-side benches
  (`moe_decode_ffn_bench`, the expert-gather attribution set) survive unchanged.
  Rebuild the attention-side equivalents at Qwen geometry (16 Q / 2 KV heads,
  head dim 256, double-width `attn_q`, uniform causal) when the decode budget is
  next attacked; the deleted versions are in git history at the fork point.
- [ ] **Prefill runs candle sdpa with a materialized mask, not the vendored
  flash kernel.** `flash.metal` is compiled at `BD == 128` and Qwen 3.6 is head
  dim 256, so the in-kernel mask path is unreachable and `model.rs` materializes
  the `[1, n_head, seq, k_seq]` f16 mask on every prefill again — the allocation
  laguna's flash path was written to avoid (1.5-2.3 GB at 4k on laguna's head
  count; ~1/3 of that here with 16 heads). Either instantiate flash at BD 256 or
  accept the mask. Pairs with P8.
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

- [ ] **Top-k selection still crosses the bus at full vocabulary width.** The draw
  now costs 0.406 ms/token, of which 0.199 ms is the GPU→CPU copy of the 993 KB
  probability row and ~0.11 ms is the CPU streaming top-k. A Metal top-k (or a
  block-wise partial reduction that ships candidates, not the whole row) would
  leave only ~20 values to read back, and most of the 0.199 ms is command-buffer
  sync rather than copy, so the win is a fraction of it — measure before
  building. Pairs with P8; the sampler now has a bench that would show it
  (`cargo test --release sampler_decode_bench -- --ignored --nocapture`).
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
- [ ] **The fast path still softmaxes the full vocabulary it no longer needs.**
  Split out of the top-p convention item when that resolved (2026-07-29). The cut
  now renormalizes over the k survivors, which is arithmetically a k-wide softmax,
  so nothing downstream of the selection depends on the full-vocabulary
  denominator any more — the ~0.1 ms it was worth is unclaimed only because the
  selection itself still runs CPU-side over the whole row. Pairs with the Metal
  top-k item above: land that and the device softmax collapses to the candidates
  along with the readback.
- [ ] **The `SampleControl` path still softmaxes on the CPU.** Adjusted draws
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
- [ ] **`top_k = 0` means greedy here, "top-k disabled" in llama.cpp.** The sampler
  maps every `top_k <= 1` to argmax, where llama.cpp treats `k <= 0` as a no-op
  filter (the whole vocabulary stays eligible). Pre-existing, harmless at the
  default of 20, but the serve layers forward client-supplied values verbatim, so
  a llama.cpp-reared client sending `top_k: 0` gets deterministic output instead
  of unrestricted sampling. Surfaced by outside-model review 2026-07-29. A
  semantics decision like the temperature-order item below, not a bug fix.
- [ ] **Temperature is applied before the top-k/top-p cut; llama.cpp's default
  chain cuts first.** Found 2026-07-29 while transcribing `top_p`: llama.cpp's
  default sampler chain is top_k → typ_p → top_p → min_p → temp → dist
  (common/sampling.cpp), so its truncation sees raw-logit probabilities, while
  xwen (like HF's default warper order) scales by temperature first, so the cut
  sees the sharpened/flattened distribution. At the model's default temp 1.0 the
  two are identical; the divergence only bites when `--temp` is overridden.
  Convention question like the top-p one was — llama.cpp and HF disagree with
  each other here, so there is no single ground truth to defer to. Needs a
  decision, not a patch.
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

## Deferred from the DeltaNet-kernel hardening pass (2026-07-29)

- [ ] **The `mtl_size!` rationale in dispatch.rs is factually wrong.**
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

## Deferred from the dense-FFN prefill gemm pass (2026-07-29, P8c)

All three come out of the per-stage prefill profile that root-caused the 27B gap.
They are what that profile found and did NOT fix; the profile itself is transcribed
in log.md 2026-07-29.

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

- [ ] **+350 to +560 µs/token of prefill cost lives OUTSIDE every measured stage, and it
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
  ANNOTATION 2026-08-08: **the diagnosis ran; the residual is real, it is NOT inside any
  stage, and two candidate mechanisms are refuted.** The named next step was built —
  `src/stack_profile.rs` / `XWEN_STACK_PROFILE`, in-situ per-stage timing by device
  sync, plus a `XWEN_CHUNK_SYNC` probe flag (design and reading discipline in
  decisions.md "Measurement discipline"; both flags stripped in `parity-gate.ts`'s
  `baseEnv()`). Conditions throughout: `lowpowermode 0` (no `powermode` key on this
  machine, high-power never claimable), warm, `XWEN_BENCH=1`, interleaved arms, medians
  of 3, 27B Q4_K_M, prefill-925 (880 tok) and prefill-4k (3851 tok).
  - **Reproduced.** Plain-arm length delta +410.3 µs/token (round 1) and +437.9
    (round 2), squarely inside the ledgered band.
  - **Under per-stage serialization the same delta is only +102.8 µs/token**:
    mixer_full_attn +53.5 (which matches the ~+69 sdpa+mask quadratic already
    estimated), ffn +42.2, residual_ffn +16.8, mask_upload +7.9, mixer_delta −9.1
    (flat). So **~335 µs/token exists ONLY when stages pipeline** — it is not in any
    stage's kernels, it is in how consecutive stages interact when the queue runs ahead.
  - **Refuted, by direct A/B, as the mechanism:** (a) cross-chunk accumulation —
    `XWEN_CHUNK_SYNC` prunes candle's buffer pool, clears its fence map and drops the
    encoder's barrier history at every chunk boundary, and the length delta is
    unchanged (+431.1 vs +437.9, a −6.8 difference; the flag itself costs +9.2 µs/token
    at 925 and +2.4 at 4k, a per-chunk price); (b) command-buffer batching —
    `CANDLE_METAL_COMPUTE_PER_BUFFER` at 10/200/1000 against the default 50, at 4k, all
    within 0.9%. See decisions.md "Refuted perf directions".
  - **Surviving hypotheses, both intra-chunk and both unconfirmed:** barrier storms from
    buffer-pointer recycling (candle rev 21cca0b emits a full `MTLBarrierScope::Buffers`
    barrier when a pool-recycled pointer is reused within an encoder session) and
    fence-wait pileup (every new encoder waits on every fence in the growing
    `prev_ce_outputs` map). Both re-develop inside a chunk regardless of boundary
    cleanup, which is exactly what the `XWEN_CHUNK_SYNC` result requires of the real
    mechanism. Supporting candle facts: the pool prunes ONLY inside
    `wait_until_completed`/`flush_and_wait_current`; `Tensor::from_vec` bypasses the
    pool entirely and allocates a fresh exact-size-keyed `MTLBuffer` plus a
    residency-set commit per call (the per-chunk mask upload is on that path);
    `find_available_buffer` scans O(total cached buffers).
  - **Next step: an instrument that can see INSIDE a chunk.** A barrier/fence counter
    needs either a candle patch or a Metal capture; `XWEN_STACK_PROFILE` cannot separate
    the two survivors because syncing is what makes the cost disappear.
  - **Second thread, tracked here rather than split out:** the **ffn stage's +42.2
    µs/token of in-stage growth is unexplained**. The dense SwiGLU is length-INDEPENDENT
    per token by construction, so a stage that grows with prompt length under
    serialization is an anomaly on its own terms; the signature is allocator pressure,
    not arithmetic.
  - Today's plain baselines for the record: 755-767 tok/s @925, 574 @4k, against the
    ledger's 702/445 of 2026-07-29. Machine-state variance, not a code change — nothing
    in this arc touched a production path, the 27B's ±10% between-run caveat applies
    (P11), and a compile load preceded round 1. See log.md 2026-08-08.

- [ ] **Attention glue: ~10 unfused eager passes per layer, inside a 57.13 ms/layer
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
  ANNOTATION 2026-08-08: **DOWNGRADED — the premise is inverted, and the ~42 ms/layer of
  glue never existed.** Read the code before sizing any of this. `permute_01`,
  `permute_01_f16`, `cast_f16`, `cast_f32` and `rope_neox` have been wired into the MAIN
  attention block since the fork (`attention.rs`'s `fused_glue` paths), so step one of
  the plan above is already done and was never undone. `ops::attn_gate` has **zero
  production call sites** and cannot serve Qwen as written: it computes a
  scalar-per-(token, head) softplus gate where Qwen needs a head_dim-wide sigmoid
  (attention.rs:360). DFlash uses no glue kernels at all — the "wired only into the
  DFlash path" reading was wrong in both directions. And the number is gone too: the
  ~42.43 ms/layer figure came from the briefing rather than the raw profile, and the
  in-situ synced differential (see the residual item above) puts the whole attention
  block's length-growth at +53.5 µs/token, which is ≈ the already-known sdpa quadratic
  with nothing left over for glue. **What remains live:** a fused sigmoid-gate kernel at
  the gate site, worth ~2-3 dispatches, and the head-dim-256 flash instantiation that
  would remove the mask (the item below). Neither is sized against a measured bounty.
  See log.md 2026-08-08.

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

All three come out of the measurement pass that closed P9(a); raw per-rep data in the
session logs referenced by log.md "K-snapshot fused verify lands".

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
- [ ] **Serve slots persisted without drafter planes silently decode plain forever
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
- [ ] **The draft-by-default flip makes a mismatched custom GGUF fail at startup.**
  With drafting opt-out, `xwen serve --model <custom.gguf>` whose geometry fails the
  drafter preflight (`DflashConfig::check_against_target`) now hard-errors where it
  previously ran plain; `--no-draft` is the workaround. Recommended shape when it
  bites someone: an IMPLICITLY-defaulted drafter that fails preflight should degrade
  to plain decoding with a warning, while an EXPLICIT `--draft` keeps the hard error.
  Not built now: the design target is the two blessed checkpoints, whose sidecars
  always match, so the edge is custom-GGUF-only.
- [ ] **Every verify arm goes superlinear at span 48.** Fused 264 → 548 ms between
  spans 32 and 48; classic 472 → 846; reproduced across reps and arms. NOT the
  dense-mm threshold (`XWEN_DENSE_MM_CLASSIC=1` arm shows the same jump). Outside
  the production regime — drafter `block_size` 16 caps real verify spans near 17 —
  so unchased, but unexplained. Also from the same pass, one anomalous
  `XWEN_DENSE_MM_CLASSIC=1` sweep read ~17% high at every span including span 2
  where that kernel cannot run, against four mutually-consistent fused sweeps and
  an immediate re-run that matched them; single unreplicated outlier, recorded
  here so a future contradiction has a trail.
  ANNOTATION 2026-08-08: **new evidence — the overshoot is ARMING-dependent, so it
  is trail memory pressure rather than a kernel threshold.** Every checkpoint-on run
  overshoots its own spans-2-32 extrapolation by **1.54-1.65x**; both no-checkpoint
  runs come in UNDER at **0.80-0.91x**. It does not move with the dense-mm or mm_id
  knobs (consistent with the original `XWEN_DENSE_MM_CLASSIC=1` finding). The
  profiled armed-minus-unarmed `mixer_delta` delta grows **6.8 → 160.6 → 304.5 ms at
  spans 2 / 32 / 48**, which tracks the K-snapshot plane buffer: ~3.15 MB per plane
  per layer, i.e. ~100-150 MB/layer at spans 32-48. Still outside the production
  regime (`block_size` 16 caps real spans near 17), so still unchased — but the
  suspect is now named and it is the trail, not the scan kernel.
  Separate new observation from the same sweeps, unexplained and NOT arming-dependent:
  **lm_head roughly doubles at span 48 (7.0 → 13.1 ms)** in both the armed and the
  unarmed profiled runs. Recorded so a future contradiction has a trail.

## Deferred from the small-batch mat-vec pass (2026-08-08)

The `mul_mv_ext` port itself shipped (kernels, routing, kill-switch, provenance v8,
oracle tests). These are the pieces it did not carry.
UPDATE 2026-08-08 (round 6): the measurement landed — the first two items below are
closed by it, and the docs it owed are written. The rest stand.

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
- [ ] **Option: floor the `Proj::DenseF16Q8` window at t >= 3, leaving span 2 on the
  gemv.** Added 2026-08-08 from the coverage A/B above. At t=2 the ext kernel saves only
  ONE gemv weight pass, and its geometry is fixed (nsg=2, nxpsg=8) rather than tuned for
  a two-row batch, so there is a plausible mechanism for it not to pay there. The
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
- [ ] **The lm_head roughly doubles at span 48 (7.0 → 13.1 ms) and nobody knows why.**
  Added 2026-08-08 from the verify-round sweeps; first recorded inside the annotation on
  "Every verify arm goes superlinear at span 48" above and promoted here because it is
  a distinct phenomenon from that item's trail-memory finding and belongs with the
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
  coverage A/B is a clean negative on it.** Span 48 is unchanged between the two arms —
  both read ~505-540 ms across all ten runs, medians 526.39 (HEAD) vs 521.97 (coverage),
  a 0.8% difference in the direction of the coverage arm and well inside the run-to-run
  spread. That is expected (the window ends at 8) and is recorded because it rules out
  the projection routing as a contributor: whatever doubles the lm_head at span 48 is
  untouched by everything shipped so far.
- [ ] **q5_K has no ext kernel** (sanctioned in the brief). ggml instantiates one; no
  supported checkpoint stores a weight in q5_K on a path this kernel serves — the
  retired unsloth UD file's experts were the only q5_K, and experts go through the
  mm_id/mv_id gather, not here. Add only if such a checkpoint returns.

## Deferred from the batch + scored-classification arc (2026-08-09)

`xwen batch` shipped with its prefix cache, `include_score` scored assembly and the
nine-item demo (log.md 2026-08-09, decisions.md "Batch"). These are the pieces it
deliberately did not carry.

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
- [ ] **Prefix grouping is single-level, and there is no cross-batch pinned snapshot.**
  One batch computes one LCP over all items; items that share more with each other than
  with the batch as a whole get no credit for it, and a system prompt shared across
  successive batch requests is re-prefilled every time. The literature says the first
  costs little: BatchLLM measured single-level collapse at roughly 1% of achievable reuse
  against a full prefix tree, and a tree brings eviction, invalidation and per-node
  snapshot accounting with it. The pinned cross-batch snapshot is the cheaper of the two
  and is the one to build first if it is built. Revisit only with a measured workload
  where the single level demonstrably loses, not on principle.
- [ ] **Results are not streamed.** `xwen batch` prints one JSON document when the last
  item finishes; progress goes to stderr as unstructured lines. A long batch therefore
  gives a caller nothing machine-readable until the end. NDJSON on stdout (one
  `ItemResponse` per line, `BatchStats` last) is the obvious shape and would not change
  the core, which already completes items in request order. Wants a flag rather than a
  format change — the current single-document output is what makes `jq` over a batch
  trivial.
- [ ] **Per-token logprobs are not exposed in any dialect.** `include_score` reports
  confidence over a field's ALLOWED OPTIONS, which is a different quantity from
  OpenAI's `logprobs`/`top_logprobs` (raw log-softmax over the vocabulary at each emitted
  position, top-k of it). The machinery for both now exists — `Generator::last_logprobs_for`
  is the log-softmax over an encodable slice — but the two must not be conflated in the
  surface: a client asking for `logprobs` wants token evidence, not label evidence.
  Independent of the scored path; belongs with the serve adaptation.
- [ ] **Snapshot-replay-vs-scratch has no Track-B parity case.** The equivalence was
  exercised at ship time by hand (`XWEN_BATCH_NO_CACHE=1` as the A/B arm, same request
  both ways) and the finding is recorded — values identical except one genuine near-tie,
  scores differing in the third to fourth decimal, both explained by the `mv_id`/`mm_id`
  partition split. That is a measurement, not a gate. The decode tier is the right home
  for it (greedy replay with the near-tie rule, which is exactly the rule this divergence
  class needs); it wants a fixture batch with a long shared prefix and enough items to
  make one near-tie likely. Until then a regression in the restore path would be caught
  only by someone running the demo.
- [ ] **`escape` is opener-level and formatting-confounded for bare literals.** It is the
  mass on tokens that open no option at the field's first choice-point token. For a
  quoted enum the forced opening quote filters formatting out and the number is
  meaningful; for a boolean, whose choice point sits after `:`, whitespace tokens a
  pretty-printer would emit compete with `true`/`false` and the escape reads near 1 while
  the answer is near-certain — observed at 0.998 escape beside a 0.9986 answer score. Two
  candidate refinements: whitespace-normalize the opener set (cheap, narrow), or make the
  escape sequence-level rather than opener-level (correct, but needs a bound on the
  non-option continuation space). Do not act on the current number for bare literals; it
  is documented on the field.
- [ ] **v1's scored-schema limits are refusals, and each has a known lift.** The shape
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
- [ ] **DFlash draft-slot handling across snapshot/restore was never checked against
  SGLang's pattern.** Batch replay syncs the drafter by truncation (`sync_drafter_to`),
  which is correct here because every item shares every token below the snapshot
  position — see decisions.md "Batch". The serving-SOTA research surfaced SGLang's
  snapshot/promote handling of speculative draft slots across cache reuse, which solves a
  strictly harder problem (concurrent sequences, divergent branches) and may name a case
  the truncation argument does not cover. Read it against `sync_drafter_to` and
  `DrafterImage` before the multi-level prefix tree or the serve endpoint lands, since
  both break the shared-prefix premise truncation rests on.

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
- [ ] **`glance` the copied scripts/ for maxuna-isms** beyond the mechanical rename
  (bench prompt fixtures, hardcoded model names, parity-gate assumptions). Partly
  done 2026-07-28 (P7): `hf.ts` repointed at the two ggml-org repos with a
  `--model-size`-style selector, `parity-gate.ts` / `parity.ts` / `ref-dump.sh` /
  `build-llamacpp.sh` / `bench.ts` retargeted. Still unswept: `classify.ts`, and
  `tests/fixtures/bench-prompts` (never opened).
- [ ] **Qwen3.6 vocab is 248320 padded / 248077 real, and constrain.rs will trip on
  it.** `constrain.rs:90` asserts `tok_trie().vocab_size() == expected_vocab` and
  `:264` feeds it the tokenizer's id space (~248070 via HF tokenizer), while the
  model's logits width is 248320 — the equality fails against a real model. Decide:
  pad the trie to logits width (padding ids permanently masked) or relax the check to
  trie ≤ logits with the tail force-masked. Also check the ban-string path against
  [PADnnnnnn] ids (type 5, unreachable but present). tokenizer.rs now exposes both
  sizes distinctly (chat-tok phase).
- [x] **The 27B linear-attn conv runs over 10240 channels at hidden 5120 — CLOSED
  2026-07-28 (P8a), no sizing problem.** The fused `kernel_delta_conv` is a flat
  one-thread-per-output-element launch through the same `dispatch_linear` helper the
  other glue kernels use (up to 256 threads per group, bounds-checked tail), so the
  channel count only sets the grid extent. Both conv widths (10240 on the 27B, 8192
  on the 35B) are covered bitwise by `conv_matches_reference_bitwise`.

- [ ] **The 35B's perplexity delta grew with the fused DeltaNet scan and the floor's
  margin shrank.** `PPL_NLL_DELTA_MAX = 0.002` was derived as
  `max(3 x measured, 0.002)` from a measured 0.000511; the fused scan moved that to
  **0.000791** (27B: 0.000221 → 0.000330). The gate still passes with ~2.5x headroom,
  but 3 x 0.000791 = 0.00237 now EXCEEDS the constant, so the recipe that produced it
  no longer reproduces it. **RESOLVED 2026-07-28 (parity owner): keep 0.002, do NOT
  re-derive from the fused measurement.** The recipe is a one-time floor-SETTING
  heuristic anchored to the reference-scan baseline, not an invariant to maintain
  against whatever the candidate currently measures — re-fitting it to the change
  under test ratchets the bound outward forever and catches nothing. The constant
  deliberately no longer reproduces from `3 x measured`, and that is the correct
  state: it is a tighter, more sensitive bound than the recipe would now give, and
  the fused path clears it with 2.5x headroom. Widening it later needs evidence the
  rise is benign, corroborated by greedy agreement and cosine — perplexity alone
  cannot show that. Rationale and the trip-wire are in docs/parity.md "Perplexity
  gate". Still open as a WATCH item: the fused scan sits at 0.000791 on the 35B, so a
  further ~2.5x rise fails the gate, and the sign is systematic (candidate worse in
  all four measurements across both architectures — the fused scan widened the gap
  ~+50% on each). This is the single most sensitive number the gate reports about the
  fused scan; the cosine tiers barely moved (35B mm actually improved, 0.999540 →
  0.999631).
- [x] **flake.nix description still says "maxuna engine"** — DONE 2026-07-28, the
  fork agent renamed all three occurrences (description + two rationale comments).
- [ ] **Partition-parity drift never measured.** The q8/f16 dual-storage split makes
  cached state depend on call partitioning (see decisions.md "Kernel policy" entry,
  2026-07-28). Accepted by decision, but the drift magnitude at the 8↔9 boundary on
  real weights has never been quantified — measure it (same prompt, cache on/off,
  compare state and downstream logits) if a near-tie flip is ever suspected in
  production, before blaming sampling.
- [ ] **Quant-vendor comparison never measured.** ggml-org was chosen over
  unsloth/bartowski on provenance (converter authors, inspectable custom mix, dflash
  sidecars), not on quality. Now that the perplexity gate exists, pointing it at a
  competing Q4_K_M is cheap — run it if output quality ever comes into question.

## Deferred from the serve batch + multi-checkpoint arc (2026-08-11)

`/xwen/v1/batch` and per-request checkpoint selection shipped (log.md 2026-08-11,
decisions.md "Serving"). These are the pieces deliberately not carried.

- [ ] **The disk tier serves only the default checkpoint.** A non-default checkpoint
  runs with every disk-tier call site handed `None`: the tier binds to one checkpoint
  id at startup and `verify()` permanently distrusts itself against any other. The
  segment layout is already per-checkpoint directories (`root/<checkpoint>/`), so the
  lift is opening one tier per checkpoint lazily rather than one at startup — do it
  when a workload actually alternates checkpoints and misses its warm conversations,
  not before. Until then a swap costs the outgoing checkpoint's warm slots and, with
  the tier on, keeps only the DEFAULT checkpoint's conversations across swaps.
- [ ] **Batch-over-HTTP gives no progress until the last item.** The CLI shows stderr
  progress lines; the HTTP client gets one JSON document at the end and nothing before
  it (a proxy that times out idle responses will cut a long batch off). The engine-side
  hooks already emit per-item progress into the server log, so an SSE or NDJSON variant
  of the route is wiring, not design. Related to the existing "Results are not
  streamed" item in the 2026-08-09 section — solve both with one shape when picked up.
- [ ] **Neither `/health` nor the TUI says which checkpoint is loaded.** `/health`
  reports `model_loaded` as a bare bool and the TUI vitals were built around one model
  id for the process lifetime. Post-swap, both are truthful but incomplete: nothing
  outside the log line says the resident model changed. Cheap: a `model` field on
  `/health` from a shared `AtomicU8`-style cell the engine stamps at load, and a vitals
  line. Do it with the first operational confusion, or sooner if the TUI gets touched.
- [ ] **A cache-miss checkpoint downloads inside the request that named it.** ~20 GB
  on a miss, inside one HTTP request, racing the watchdog deadline if one is configured
  (the download resumes in place, so a retry eventually completes — hf-hub semantics).
  Both checkpoints are cached on this machine, so this is theoretical here; if it ever
  bites, the fix is a 503-with-progress answer while a background fetch runs, not a
  longer deadline. Also: hf-hub's own byte-level progress bar writes to raw stderr
  (`ApiBuilder::with_progress(true)`), bypassing `ServeLogger` — under `--tui` it
  draws over the dashboard, the same hazard class the batch runner's `eprintln!`s
  were converted to hooks for. Route or suppress it when this item is picked up.
- [ ] **`/xwen/v1/generate` carries no model field.** Deliberate — the native generate
  surface documents itself as modelless and the batch route is the native surface that
  selects — but it is now the only route that cannot reach the non-default checkpoint.
  Add the field if a native-API consumer ever wants it; it is a two-line change in
  `prepare` plus tests.
- [ ] **Mid-batch cancellation does not reach the scored path's forced spans, nor any
  prefill.** The cancel poll runs between items and per decoded token inside an item's
  free decode; a scored item's teacher-forced assembly checks only at item boundaries,
  and neither the shared-prefix prefill nor an item's own tail prefill polls at all
  (`prefill_tokens` chunks internally but takes no callback). Items are short (≤192
  tokens in the demo), so the exposure is bounded by one item's latency plus one
  prefill — thread the poll through `assemble_scored` and the prefill chunk loop only
  if a real workload makes either span long enough to care.
- [ ] **The batch route inherits axum's default request-body limit (~2 MB).** The
  endpoint's contract is "the same document the CLI reads on stdin", but the CLI has
  no size limit and the route silently does — an oversized batch gets a framework 413
  that is not in the native error envelope. Decide a deliberate limit (and error
  shape) when a real batch first trips it; document it either way.
- [ ] **The batch scheduling estimate is bytes-based and can read zero.** A batch of
  items with empty message content (schema-only probes) estimates zero prompt tokens
  and schedules as free; the real cost floor is the rendered template per item. Fold a
  per-item constant into `size_estimates` (or estimate from the rendered skeleton)
  when scheduling fairness under real mixed traffic matters; on a single-user box the
  age limit already bounds the damage.
- [x] **Startup drafter resolution still trusts `--model-size`, not the file. DONE
  2026-08-11 (same day, review fix).** `run_serve` resolved the official-sidecar path
  via the flag before the GGUF was ever opened, so `--model-size 27b -m <35b.gguf>`
  (or a config-file `model` disagreeing with the flag) selected the 27B sidecar for a
  35B target — not silent, `validate_model` refused to start, but the error blamed the
  drafter when the real mistake was the flag/path mismatch. Fixed by deriving the size
  from the served GGUF's architecture (metadata-only read) before `resolve_draft`. The
  one-shot CLI commands deliberately keep the flag's double duty — there the flag and
  the payload are the intent. Pre-existing; surfaced by the 2026-08-11 review.
- [ ] **The scheduler does not group queued jobs by checkpoint.** `shortest-prefill`
  scores by prefill cost alone, so a queue holding jobs for both checkpoints can pick
  them interleaved and pay a ~3 s swap per pickup where checkpoint-grouped ordering
  would pay two. The cost model could add the swap (a job for the non-resident
  checkpoint costs its prefill plus a load-equivalent), which also naturally batches
  same-checkpoint work without starving the other (the age limit already guards
  starvation). Do it when a real workload actually interleaves checkpoints; a
  single-user machine mostly will not.
