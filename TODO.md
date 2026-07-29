# Deferred work ledger

Items are never deleted, only annotated with their outcome in the item's own bold
header (`DONE <date>`, `CLOSED-REFUTED <date>`, …) plus the measurement that closed
them and a pointer to the log.md entry with the full arc. Sub-items are lettered under
a numbered parent.

## Priority order (decided 2026-07-28, next session starts at P1)

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

9. **DFlash adaptation to the Qwen sidecars.** Repoint drafter arch check (arch
   `dflash`, decoder arch qwen35/qwen35moe), tap indices from `target_layers`
   metadata, mask_token_id, sliding-window pattern; verify the fc.weight geometry
   (5×hidden / 8×hidden concat). Needs P4's recurrent-state rollback. Re-tune
   auto-pause and draft-ctx horizon for this drafter's cost curve.
   - ANNOTATED 2026-07-28: drafting is now OFF by default (`DEFAULT_DRAFT_ENABLED
     = false`) because the inherited opt-out default aborted every zero-flag
     `xwen generate` and `xwen serve` at startup. **Flip it back to `true` as part
     of this item**, along with the CLI/config help text in `bin/xwen/main.rs`
     (`DraftArgs`, `ServeArgs`) and `serve/config.rs` (`DraftToml`, the `[draft]`
     `--init` template block), all of which currently say drafting is unavailable.
   - The first failure is NOT the `decoder_arch == "laguna"` check: the shipped
     sidecars carry no `dflash.decoder_arch` key at all, so `from_gguf` fails on the
     missing key before reaching it. Two more blockers behind it — the shipped
     tensor tables have no `enc.aux_norm` and no `blk.N.attn_gate`, both of which
     `DflashDrafter::build` requires. `dflash::tests::real_file_load_and_shapes` and
     `real_file_bf16_alias_load_and_forward` fail on exactly this and are the
     suite's only red tests; they go green when this item lands.
   - `DRAFT_KV_BYTES_PER_TOKEN` in `serve/config.rs` describes the 35B-A3B sidecar
     only (6 layers; the 27B has 5). It feeds an `--init` comment and nothing that
     allocates — make it per-model here, alongside `hub::Model::kv_bytes_per_token`.

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
