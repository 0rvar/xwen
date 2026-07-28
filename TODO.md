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

8. **DeltaNet Metal kernels.** (a) fused recurrent decode step (one dispatch per layer
   per token; state stays resident, fp32); (b) chunked prefill scan, chunk 64,
   llama.cpp's chunked form as the spec (cumsum → tri decay mask → solve_tri →
   per-chunk state update) — needs tri-solve which candle lacks; vendored kernel.
   Kill-switches XWEN_DELTA_CLASSIC / XWEN_DELTA_CHUNK_CLASSIC falling back to the P3
   reference. Gate: bitwise-or-bounded vs reference per parity.md tiering.

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

12. **MTP exploration — DEFERRED by decision.** Sidecar reuses parent arch, one extra
    full-attn block + nextn.* tensors, plain KV cache, eh_proj over
    [norm(emb);norm(h)]. Evaluate as drafter only after P9 lands or fails (see
    decisions.md "Speculative decoding").

13. **YaRN long-context.** Native 262144; Qwen documents 1M via YaRN but ships no
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
- [ ] **The 27B linear-attn conv runs over 10240 channels at hidden 5120** — check
  dispatch.rs threadgroup sizing assumptions when P8 lands (laguna never ran a
  depthwise conv).
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
