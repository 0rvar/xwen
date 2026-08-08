# Decisions

This file is the WHY, by topic: every deliberate choice, default, policy, and refuted
direction, with its evidence. Refuted directions are recorded as decisions too — "we
measured it and we will not do X" is load-bearing. Each entry ends with a parenthesized
date pointing at the `docs/log.md` entry that tells the whole story; decisions.md gets
amended in place, the log entry preserves what was believed at the time.

Inherited rule from laguna, kept because it bit that project three separate times: an
observed identity must never be promoted to a claimed one. When this file says
"bit-identical", it says why.

## Scope

**xwen is a fork of ../laguna (crate `maxuna`) adapted to Qwen 3.6, serving exactly two
checkpoints: Qwen3.6-27B (dense) and Qwen3.6-35B-A3B (MoE).** Same design target as the
parent: maximum tok/s on this one machine (M5 Max, Metal), batch 1, GGUF weights, no
portability hedging. The 35B-A3B is the bring-up model — 20.4 GB Q4_K_M, 3B active,
fastest iteration loop; the 27B dense follows as a variant (its FFN is a strict subset
of the MoE machinery) (2026-07-28).

**The dependency set is laguna's, verbatim, and is not relitigated.** The candle git pin
(rev 21cca0b) ships the quantized indexed MoE matmuls and the residency-set APIs the
mmap loader needs; the objc2 crates stay `=`-pinned to what that rev resolves or cargo
duplicates them and the ObjC types stop interoperating (2026-07-28).

**Text-only.** Qwen 3.6 is multimodal upstream, but the GGUF conversions are text-only
(no vision tensors in the qwen35/qwen35moe arch lists; mmproj is a separate CLIP file we
do not load). The chat template's vision content items are rejected, not rendered
(2026-07-28).

## Defaults and CLI surface

**Default checkpoints are ggml-org's Q4_K_M files.** `ggml-org/Qwen3.6-27B-GGUF` and
`ggml-org/Qwen3.6-35B-A3B-GGUF` are HF's own llama.cpp org — the closest thing to
official GGUFs (Qwen published safetensors/FP8 only). Q4_K_M over Q8_0 because decode is
bandwidth-bound and the Q4_K_M mix (attention/ssm/shared-expert Q8_0, expert stacks
Q4_K, lm_head Q6_K) keeps the quality-critical planes at 8-bit anyway. Single files, no
sharding (2026-07-28).

**Sampling defaults follow generation_config.json: temp 1.0, top_p 0.95, top_k 20.**
Stop tokens are the generation_config list `[248046 <|im_end|>, 248044 <|endoftext|>]` —
config.json's single `eos_token_id: 248044` is wrong for chat and runs straight past
turn boundaries (2026-07-28).

## Ground truth and parity methodology

**Upstream llama.cpp master replaces the poolside fork as parity ground truth.** The
reference implementation is `src/models/qwen35.cpp`, `src/models/qwen35moe.cpp`, and the
shared `src/models/delta-net-base.cpp` (llama.cpp was refactored; per-arch graphs no
longer live in llama-model.cpp), running the identical ggml-org GGUF. The three-tier
gate methodology (strict oracle cosine / fork-equivalence / greedy replay + perplexity
bound) carries over from laguna; floors must be recalibrated for the Qwen checkpoint
quant mix and are not laguna's numbers (2026-07-28).

**The oracle is PINNED to a commit, not tracked.** `reference/llama.cpp` must be a
checkout at the exact sha recorded in docs/parity.md. A different oracle build can move
the achieved cosines, which would silently invalidate the floors calibrated against it,
so moving the pin is a deliberate act paired with a re-calibration — never a `git
pull`. The harness itself is indifferent to HOW the checkout is materialized; what it
requires is that the sha be recorded and deliberate.

**The oracle checkout is a git SUBMODULE** (shallow, declared in .gitmodules), with
only `reference/llama.cpp/build` ignored. Resolved by the owner the same night it
arose: the P7 session had materialized it as a plain gitignored clone while a
submodule gitlink was staged concurrently; the owner picked the submodule, whose
gitlink makes the oracle sha reviewable in the diff — moving the pin is a staged
change the owner approves, which is exactly the "deliberate act" the paragraph above
requires (2026-07-28).

**Floors are calibrated across BOTH checkpoints and all fixtures, then set under the
WORST observed value.** The gate constants are global (one `COS_MIN_MM` for whatever
file is gated), so a floor derived from one checkpoint or one prompt would be a
coin-flip on the others. The measured spread is in docs/parity.md "Floors": the
strict floor sits ~1e-4 under the worst classic-path value and the mm floor ~5.4e-4
under its worst, roughly 1.7x the observed prompt-to-prompt spread. Per-checkpoint
applicability remains enforced by the procedure, not by code — the ledger item for
binding a floor to a checkpoint hash is still open (2026-07-28).

**Tap-name translation lives in the harness, not in the engine.** Our tap names and
llama.cpp's `cb()` node names disagree on three of nine taps, and the mixer output has
two different llama.cpp names depending on layer kind. `scripts/parity.ts` owns the
mapping (`refTapNames`) rather than renaming engine taps, so the parity harness can
follow upstream's naming churn without touching model code — and so a mapping mistake
is a harness bug, not a silent change to what the engine records (2026-07-28).

**HF transformers is the secondary reference, with two conversion deltas that make
diffing against it hazardous:** GGUF norm weights are pre-baked `w+1` (HF Qwen3.5 norms
are Gemma-style zero-centered `(1+w)`) for every norm EXCEPT the DeltaNet `ssm_norm`,
and GGUF V-head ordering is tiled (llama.cpp permutes V-side weights at conversion to
suit `ggml_repeat`) where HF safetensors are grouped (`repeat_interleave`). We read
GGUF, so we use the pre-baked/tiled forms directly and never "fix" them (2026-07-28).

## Model math: the forms that could have gone the other way

Every entry here is a place where two defensible readings existed, the code had to pick
one, and picking wrong would have produced a model that runs, emits fluent text, and is
quietly incorrect. Each is pinned by a unit test, because "we checked once" does not
survive a refactor.

**silu runs over the WHOLE fused DeltaNet stream, before the q/k/v split — so q and k
are silu'd before their L2 normalization, not just v.** The natural misreading of the
recurrence is that silu is the value-path activation. It is not: `qwen35.cpp:397`
applies `ggml_silu` to the entire `[conv_dim, T]` conv output and the q/k/v views are
taken from the result afterwards (`:400-423`), matching HF's
`causal_conv1d_fn(..., activation="silu")`. Getting this wrong changes every q and k
that enters the delta rule and is invisible in any shape check (2026-07-28).

**The q/k L2 normalization uses ggml's clamp form `x / max(‖x‖, eps)`, NOT HF's
`x · rsqrt(Σx² + eps)`.** `ggml_compute_forward_l2_norm_f32` computes
`scale = 1.0f/fmaxf(sqrtf(sum), eps)` (`ggml/src/ggml-cpu/ops.cpp:4204`, read directly
from the vendored tree, not taken on report); HF/FLA computes the rsqrt form
(`modular_qwen3_next.py:222-224`). The two agree to rounding for any vector whose norm
clears eps, which a silu'd conv output always does — so this cannot move parity today.
It is still decided rather than left to chance: llama.cpp is the parity ground truth, a
strict tier may one day ask for bitwise agreement, and an "it cannot matter" difference
is exactly the kind that turns out to matter later. eps is `rms_norm_eps` read from the
checkpoint, not a hardcoded 1e-6 — llama.cpp passes `hparams.f_norm_rms_eps`, and only
the shipped checkpoints make those the same number (2026-07-28).

**The `1/√128` scale is applied to the readout only.** llama.cpp scales q once, before
the recurrence (`delta-net-base.cpp:319-321`), and q enters the recurrent form at
exactly one place — the `o = q·S` readout (`:365-366`). Scaling q up front and scaling
the output are therefore algebraically identical, and the chunked path applies the same
scale at the same point so the two forms agree. There is no second scale anywhere: not
on k, not folded into beta (2026-07-28).

**q and k are broadcast from K-heads up to V-heads by TILING — output head `j` reads
K-head `j % n_k_heads` — never by interleaving.** This was the single highest-risk
assumption in the DeltaNet port, because the usual way to write this broadcast in ggml
(reshape to `[d, 1, n_k, T]`, then repeat) yields interleave semantics, and both forms
type-check, run, and produce plausible output. `qwen35.cpp:442-443` repeats directly on
the natural `[head_k_dim, num_k_heads, T, S]` layout, and
`ggml_compute_forward_repeat_f32` (`ggml/src/ggml-cpu/ops.cpp:1723-1739`) writes
destination head `i1*ne01 + k1` from source head `k1` — tiled. This is deliberate, not
incidental: the converter pre-permutes every V-side weight from HF's grouped order into
tiled order precisely so ggml's repeat can replace an expensive interleaved one
(`conversion/qwen.py:355-378`). Reading GGUF, tiling is correct; reading HF safetensors
directly it would be `j / ratio`, and we do not read those (2026-07-28).

**The DeltaNet output norm is `rms_norm → × ssm_norm.weight → × silu(z)`, with the gate
LAST.** `build_norm_gated` (`qwen35.cpp:246-255`) normalizes, applies the weight, and
only then multiplies by `silu(z)`; current HF agrees and carries the literal comment
`# Norm before gate` (`modular_qwen3_next.py:76`). Older FLA `FusedRMSNormGated`
variants gate first, which is why this needs recording: a reader who reaches for the
wrong upstream finds a form that disagrees with both llama.cpp and current
transformers. Folding the gate in before the norm would change the statistic the norm
divides by, so it is not a reordering that washes out (2026-07-28).

## Weights and loading

**GGUF is the only weight format**, loaded by laguna's mmap/no-copy path
(`newBufferWithBytesNoCopy` over the page cache, batch-registered residency, classic
full-copy fallback under `XWEN_LOAD_CLASSIC`). Nothing in that loader is
architecture-specific; only the tensor-name table changes (2026-07-28).

**Loader name traps, recorded here because each one silently produces a working-looking
model:** there is no `ffn_norm` — `blk.N.post_attention_norm.weight` is the PRE-MLP norm
(HF semantics), not a Gemma-style post-norm. There is no `ssm_in` — the DeltaNet QKV
projection ships as `blk.N.attn_qkv.weight` and the z-gate as `blk.N.attn_gate.weight`.
`blk.N.ssm_a` (no `.weight` suffix) already holds `-exp(A_log)`. `blk.N.ssm_dt.bias`
uses a `.bias` suffix. Full-attention `attn_q` is double-width with per-head interleaved
`[q_h, gate_h]` layout (2026-07-28).

**beta and alpha ship as two separate tensors; there is no fused `ssm_ba` on these
architectures.** The loader briefly carried a fallback that split a
`[2·v_heads, hidden]` `ssm_ba` on the theory that either conversion might exist. The
shipped headers settle it: both files have `blk.N.ssm_beta.weight` and
`blk.N.ssm_alpha.weight`, `[hidden, v_heads]` Q8_0 each, mapped one-to-one from HF's
`in_proj_b` / `in_proj_a`. `LLM_TENSOR_SSM_BETA_ALPHA` → `blk.%d.ssm_ba` does exist in
llama.cpp's arch table, but is referenced only by the `qwen3next` arch, which maps HF's
fused `in_proj_ba`. The fallback was removed rather than kept as insurance: a branch
that can never fire is a claim about the world that nothing will ever check
(2026-07-28).

**Persistent state is partition-dependent in its low bits, and that is accepted, not
denied.** The dual-storage attention planes (f16 GEMM above `Q8_DECODE_MAX_SEQ`=8,
raw-q8 GEMV at or below it) are deliberately not bit-identical per weight element, and
the same `Proj` feeds cache-mutating paths — KV writes, the DeltaNet conv window and
recurrence. So the same tokens partitioned differently into forward calls (prefix-cache
snapshot stops, verify batches of 9+ vs one-token decode) produce state differing in
low bits. A second-model review caught model.rs's rollback docstring promising bitwise
identity with a differently-partitioned counterfactual — an observed identity promoted
to a claimed one, the exact failure mode this file's preamble warns about. The
docstring now states the real guarantee: restores are bit-exact replays of recorded
bytes; cross-partition agreement is numeric and parity-gated; `XWEN_ATTN_DEQUANT` pins
one canonical representation when bitwise partition-independence matters. The
alternative — one representation always — roughly doubles the attention-projection
bytes streamed per decoded token (halving those bytes is why dual storage exists;
laguna measured the win when it shipped it), a real cost not paid for a low-bit
property nothing currently depends on (2026-07-28).

## Kernel policy

**Laguna's kernel policy is inherited wholesale:** vendored `.metal` sources runtime-
compiled via include_str!, ggml-geometry dispatch, `fp contract(off)` +
`fp reassociate(off)`, a `XWEN_*_CLASSIC` kill-switch and provenance string per kernel,
and the rule that nothing upstream of the MoE router is reimplemented unless
bit-identical — laguna measured per-op-correct-to-1.6e-7 kernels moving final logits by
1.3e-3 through router near-tie flips. Qwen3.6-35B-A3B has 256 experts and a softmax
router; the same chaos-amplifier reasoning applies unchanged (2026-07-28).

**DeltaNet gets a frozen reference first, kernels second.** A composed-candle-ops
implementation of the recurrence (recurrent form, fp32 state) is the correctness oracle,
mirroring laguna's `ReferenceExperts`; vendored Metal kernels (chunked prefill scan,
fused decode step) land only after the reference passes parity, and the reference is
never optimized (2026-07-28).

**The fused DeltaNet scan is bounded, not bit-identical — so it is pinned OFF in the
strict tier.** Every other vendored kernel reproduces its candle chain's rounding
boundaries exactly, which is why their `*_CLASSIC` pins are pure provenance discipline.
The scan cannot: the reference contracts k and q against the state with a candle gemm
and normalizes q/k with a candle reduce, while the kernel partitions both across
threads (the whole point — the state stays in registers across all T timesteps).
Reassociating an f32 sum is not something a kernel can undo. So `XWEN_DELTA_CLASSIC=1`
is pinned on both sides of the strict tier, and the fused path is graded by the
bounded mm/decode/ppl tiers instead, with a `delta` provenance field (schema v6) proving
which side ran what. Measured agreement vs the reference at both shipped geometries:
relative L2 under 1e-5 on both the per-token output and the state left behind, at
sequence lengths from 1 to 512 (2026-07-28).

**A multi-token chunk under an armed rollback checkpoint stays on the reference scan.**
**SUPERSEDED 2026-07-29 (P9a)** — the revisit clause below fired: the verify walk was
chunk-shaped and hot (39 ms/verified position, the number that capped spec decode's 27B
win). The scan kernels now spill per-token states themselves: `delta_scan_with_trail`
widens the state output to `[planes, v_heads, 128, 128]`, most-recent-first (plane s =
state after token `seq-1-s`), mirroring llama.cpp's `kernel_gated_delta_net` K>1
snapshot slots so the CPU oracle stays diffable. Plane 0 is still the unchanged
after-loop store, so `planes = 1` — every unarmed prefill and decode call — is
byte-identical to the old kernel; the in-loop guarded store only touches slots ≥ 1.
The armed clause is gone from the fused gate; the trail's delta entries are
unmaterialized views into the snapshot buffer, the conv entries the same host-side
stream slices the reference records. `XWEN_DELTA_CLASSIC=1` still routes everything,
armed chunks included, to `forward_classic`. Both parity gates re-passed with numbers
identical to the pre-change run. Original entry, kept for the reasoning:
The one-dispatch scan can only report the state after the LAST token, and an armed
DeltaNet layer needs the state after every token (`LayerCache::Linear`'s trail — the
equivalent of llama.cpp's K snapshot slots). Rather than teach the kernel to spill T
intermediate states, `LinearAttnBlock::forward` sends `seq > 1 && trail_armed` to
`forward_classic`. Single tokens still take the fused path even when armed, because
their only state IS the state after the last token — so spec decode's per-token verify
steps keep the win and only a batched verify forward pays. Revisit if the verify walk
ever becomes chunk-shaped and hot (2026-07-28).

**The three glue kernels ARE bit-identical, and that was worth the block-scope
pragmas.** `delta.metal` carries `fp contract(off)` / `reassociate(off)` at BLOCK scope
on the conv, beta/decay and gated-norm kernels while deliberately leaving the scan
free to contract into fma — its two inner loops are the entire prefill cost. File-scope
pragmas (the sibling glue files' convention) would have doubled the scan's inner
instruction count; a second library would have cost another runtime compile. The conv
kernel is bitwise against the reference's cat + per-tap broadcast chain + silu, and the
beta/decay kernel bitwise against candle's `usigmoid` and the stable softplus chain
(2026-07-28).

**One kill-switch covers the whole MoE block glue, because all of it is bit-identical.**
`XWEN_MOE_GLUE_CLASSIC=1` reverts the fused router, the fused block epilogue and the
shared expert's fused activation together, the way `XWEN_ATTN_GLUE_CLASSIC` covers the
attention glue family. Splitting them would imply the pieces have independent
correctness stories; they do not — each reproduces its candle chain's rounding
boundaries exactly, so the switch is a safety handle and a provenance anchor, not a
tier. The routed experts keep their older, narrower switches (`XWEN_ACT_CLASSIC`,
`XWEN_COMBINE_CLASSIC`), which still apply on the classic branch. Because nothing here
can move a dump, the parity schema is untouched: no `moe_glue` provenance field, no pin,
no grandfather clause (2026-07-29).

**The MoE router matmul stays candle's; the fusion starts at the logits.** The obvious
version of the router fusion swallows the gemv too, and it was rejected on evidence
rather than effort. candle lowers a `[1, hidden] × [hidden, n_expert]` f32 matmul to
MLX's `gemv_t_float32_bm1_bn2_sm8_sn4_tm4_tn4`, whose K-partition is strided across
lanes and whose cross-lane combine is a specific shuffle-down tree; reproducing it means
lifting the kernel verbatim AND relying on the compiler contracting `result += vc * inter`
into an fma identically, which the source text does not pin. Decisive on top of that:
its accumulation order depends on the OUTPUT WIDTH, so the tempting trick — concatenating
the shared-expert gate row onto the router weight at load time, the way `ba_wt` folds
beta and alpha in the DeltaNet block — would have changed that gate's bits, because a
`[hidden, 1]` matmul hits a different MLX kernel family entirely than a `[hidden, 257]`
one. So both matmuls stay candle dispatches and the kernel takes over at the softmax.
The cost is one dispatch out of ten; the alternative was a bounded router, and a bounded
router is exactly the thing the chaos-amplifier rule above forbids (2026-07-29).

**Reproducing candle's arg-sort tie order is not optional, and it is not the stable
one.** candle's Metal `arg_sort_last_dim` is llama.cpp's bitonic network, whose
comparators are strict and never consult the index — so equal probabilities do NOT come
out in ascending expert order, and the CPU backend (a stable `sort_by`) disagrees with
the Metal one. Any top-k selection shortcut would therefore diverge on ties, and a tie
flip is not a rounding difference: it swaps a whole expert's contribution into the
output. The fused router runs the network verbatim, ties and all
(`router_ties_match_candle_bitwise`). Worth knowing before anyone "simplifies" it: the
all-equal case comes out as the identity permutation, which makes the non-stability easy
to miss in a casual test (2026-07-29).

**The dense-FFN prefill gemm dequantizes in-kernel; it does not dequantize to a
scratch plane.** The 27B's prefill was 1.8-2.1x behind llama.cpp, and the profiling pass
found **66-85%** of its wall time in the dense SwiGLU FFN — 64 layers, 17408-wide, Q4_K —
running through `QLinear` → candle's `QMatMul` → `kernel_mul_mm_q4_K_f32` at ~12-13
TFLOP/s, where the same shapes hit 28-36 TFLOP/s on the Metal-4 cooperative-tensor f16
gemm. (A band, not a point: the FFN row of that budget is derived from an isolated T=512
rate that is ~7-8% pessimistic against a real forward, which is why the @880 budget sums
to 106.8% of wall. See log.md 2026-07-29.)

**The gap is kernel efficiency, not a memory wall, and the argument needs no peak-
bandwidth number.** Do not write "both are far below the machine's peak" — nobody has
measured this machine's peak. The airtight form: at T=512 the Q4_K arm reads 50.14 MB of
weights against the f16 arm's 178.32 MB, so it moves **3.6x fewer weight bytes and takes
2.4x longer** (10.9 vs 66.0 GB/s). If either arm were bandwidth-bound, the one moving
fewer bytes would be the faster one. It is not (2026-07-29).

Two ways to close it. (a) Dequantize each weight to a transient f16 scratch buffer per
layer per chunk and feed the existing `f16_t.metal` gemm. (b) Port the dense
cooperative-tensor kernel to read Q4_K directly and dequantize each tile in registers.
**(b) shipped**, on arithmetic that is not close: a `[17408, 5120]` f16 plane is 178 MB
written and 178 MB read back against the 50 MB of Q4_K the kernel would otherwise
stream — about 8x the weight traffic, ~0.68 ms per projection at 600 GB/s against the
gemm's own 3.2 ms, so roughly a fifth of the win handed back plus a 178 MB scratch
allocation. The tile dequant reads each super-block once and never leaves registers.

The port itself was small because the pieces already existed: `f16_t.metal` is the
dense tensor gemm (64-row x 128-token tiles, the f32 activation read straight from
device as a cooperative tensor, no B staging), and `mm_id.metal` has ggml's Q4_K tile
dequant. `dense_mm.metal` is the first with the second substituted into its A-tile
phase — which is also exactly how ggml writes it, templated over
`(block_q, nl, dequantize_func)`. The 128-wide token tile is what keeps it from being
dequant-bound the way the MoE gather is at 32: each dequantized element feeds 128 MACs
(2026-07-29).

**The dense-FFN gemm crosses over at exactly 33 tokens, and that is a tile boundary, not
a tuned number.** `DENSE_MM_MIN_SEQ = 32`, exclusive (`seq > 32`), following
`F16_MM_MIN_SEQ` and ggml's `ne11_mm_min` rather than `MM_ID_MIN_SEQ`'s inclusive
convention. candle's kernel tiles tokens 32 wide and the vendored one 128, so up to 32
tokens both sit on the same launch-latency floor — measured 1.01-1.05x, a wash — and at
33 candle takes a second token tile while the vendored gemm does not: 1.20x there,
1.6-1.9x at 128, and 2.4-3.0x at a 512-token chunk. Below the boundary there is no
throughput reason to take the vendored kernel and a positive reason not to (next entry),
so the threshold sits where the win starts rather than where it would be tidy
(2026-07-29).

**The dense-FFN gemm is LESS accurate than the `QMatMul` chain it replaces, and that is
the trade being made.** Against a dequantize-then-f32 oracle at the 27B FFN shapes it
lands ~4.1e-4 rel_l2 where candle's kernel lands ~1.9e-4; the two differ from each other
by ~3.7e-4. Both stage the weight tile as half and accumulate in f32 — the extra ~2e-4
is matmul2d's reduced-precision tensor-core path, which is where the throughput comes
from (the `false` variant of that descriptor flag runs at classic speed). This is not a
new precision class and not a new decision: the attention prefill gemm made the
identical trade, docs/parity.md §3b already names the ~2e-4 band the fork's prefill
precision class, and llama.cpp sets the same flag for its own dense FFN prefill. So the
kernel is graded like the DeltaNet scan rather than like the glue kernels —
`XWEN_DENSE_MM_CLASSIC=1` is pinned on BOTH sides of the strict tier, a `dense_mm`
provenance field (schema v7, grandfather `classic`) proves which side ran what, and the
bounded mm/decode/ppl tiers carry the signal against frozen floors. Worth stating
plainly because the reflex is to treat a faster kernel as free: this one costs precision,
the gate is what decides the cost is acceptable, and the kill-switch is what makes the
question answerable later (2026-07-29).

**The FFN prefill planes cost no device memory, which is the only reason they exist.**
17.1e9 FFN parameters as f16 is 34 GB — the permanent-plane approach the attention
projections use (`Weights::attn_proj`'s dual f16/q8_0 storage) is arithmetically
impossible here. `Weights::qlinear_with_plane` instead reuses `qlinear_with_buffer`'s
construction: read the quantized bytes once, upload once via `QStorage::from_data`,
retain the buffer, then wrap that same storage in the `QTensor`. The `QLinear` decode
path and the prefill gemm index one allocation. A checkpoint whose FFN dtype has no
vendored kernel, or a non-Metal load, gets `None` and runs `QLinear` at every seq
(2026-07-29).

**The small-batch matmul window routes from ONE decision point, and it is
`QLinear::forward`.** At seq 2..=8 no kernel in the tree wanted the shape: candle's
`mul_mm` grid collapses to `ne01/64` threadgroups at small M (~73 GB/s against ~280 on
the seq==1 mat-vec path), and the vendored cooperative-tensor gemm has the same
occupancy collapse — forcing it down with `XWEN_DENSE_MM_MIN_SEQ=1` moved the verify
forward's fixed intercept only −3.3 ms. `src/ops/mv_ext.metal` is llama.cpp's
`kernel_mul_mv_ext`: dequantize a weight block once, reuse it across 2-5 output rows
(Q4_K/Q6_K/Q8_0 × r1ptg 2-5). Routing it from the single `QLinear::forward` site rather
than per-call-site is the deliberate part — that one point covers the 27B dense FFN, the
35B shared expert and `forward_all_logits`' lm_head, so the window has one definition,
one kill switch (`XWEN_MV_EXT_CLASSIC`) and one provenance field. Measured at 27B
`n_past` 512: the verify forward goes 153.44 → 61.45 ms at span 2 (0.40x) and 220.11 →
161.16 at span 8 (0.73x); drafted decode gains +11.6% / +13.2% on the 27B (2026-08-08).

**Two deliberate divergences from ggml's own gating, both because our fallback is worse
than theirs.** (1) ggml restricts its K-quants to ne11 4..=8; xwen admits them from 2,
because at spans 2-3 our alternative is the 73 GB/s `mul_mm` rather than the tuned path
ggml falls back to — the comparison that sets the gate is against what we actually have.
(2) `nxpsg=8` is baked rather than tuned per shape. Both are recorded as divergences
rather than oversights so a future reader diffing against ggml does not "fix" them
(2026-08-08).

**This kernel is MORE accurate than what it replaces — the opposite of the `dense_mm`
trade — and the tests assert the direction, not a band.** rel_l2 4e-7..8e-6 against the
`QMatMul` mm's ~1.8e-4, i.e. 20-400x better, because it is f32 end to end where candle's
tiled mm stages weight tiles as half. The oracle tests assert `rel <= rel_classic` at
1.0x rather than an absolute tolerance: the property worth pinning is "never worse than
the path it replaced", and freezing a number would break on a legitimate kernel change.
Worth stating next to the `dense_mm` entries above precisely because the reflex they
install — a faster kernel costs precision — does not hold here (2026-08-08).

**Provenance treatment mirrors `dense_mm`, even though the gate cannot exercise this
kernel.** `mv_ext` at schema v8, grandfather `classic`, pinned classic on both sides of
strict. No parity fixture ever produces a 2..8 forward (prefill chunks are 512, decode
is 1), so the tiers are structurally blind to it and the `mv_ext.rs` oracle tests carry
the correctness claim instead. The pin is still right: it costs nothing, it keeps the
field meaningful if a future fixture ever does enter the window, and a dump that cannot
say which path it ran is worth less than one that can (2026-08-08).

**The recurrent state is fp32, non-negotiable.** `mamba_ssm_dtype: "float32"` upstream;
llama.cpp hardcodes F32 for both conv and delta states. State per layer per sequence:
`(d_conv−1)·conv_dim` floats conv + `128·128·H_v` floats delta (2 MiB on the 35B)
(2026-07-28).

## Refuted perf directions — do not reopen without new evidence

Laguna's refuted list (death-by-dispatch, encoder takeover, all-f16 activation chains,
mixed-operand matmul2d as a tensor speedup, sub-32-seq mm_id) transfers as *prior
evidence, not law* — same machine, same candle, same kernel geometry, different model.
Anything re-opened here needs a measurement, and its entry moves into this section with
the number that killed or revived it (2026-07-28).

**The dual-weight expert gather — fusing gate, up and their SwiGLU into one dispatch —
is REFUTED on this device.** `kernel_mul_mv_id_q4_K_f32_dual` (mv.metal) does exactly
that, and is bitwise against the three dispatches it replaces: it calls the same
`mul_mv_q4_K_sums` body once per projection, factored out of the verbatim ggml
`mul_mv_q4_K_impl`, so neither accumulation order moves, and the epilogue is
`silu_mul.metal`'s expression with its roundings pinned. It still measured slower —
99.5 tok/s decode against 102.8 for the split chain, 35B-A3B, interleaved, median of 5,
five reps apart with no overlap between the arms.

The mechanism is worth recording because it generalizes. It is NOT register pressure:
the single and dual pipelines both report `max_total_threads_per_threadgroup` of 1024.
It is the grid. Two independent `mul_mv_id` dispatches expose twice the threadgroups,
and the routed-expert gather is bandwidth-bound, so that parallelism is what keeps
memory requests outstanding. The dual kernel keeps ONE dispatch's grid while walking both
weight planes serially per thread, halving the memory-level parallelism to save two
launches. Dispatch count is the right thing to attack in the MoE glue — 24 to 14 bought
11% — but it stops being the right thing to attack the moment the dispatches being
merged were already saturating bandwidth in parallel.

Kept behind an opt-in `XWEN_MOE_DUAL=1` rather than deleted: it is a measured, gated,
bitwise-proven artifact, and the conclusion is device-specific. Re-price it before
reusing it, do not assume it (2026-07-29).

**The DeltaNet scan decomposition: llama.cpp's Metal shape is REFUTED here, and the
scan was never the 27B prefill gap in the first place.** Two claims, measured together,
because the second is what makes the first not worth revisiting.

llama.cpp's `kernel_gated_delta_net_impl` (ggml-metal.metal) gives ONE SIMDGROUP each
state value-column end to end: both key-dim contractions become `simd_sum`, the timestep
loop holds no barrier at all, and the grid is `(S_v/nsg) × H × n_seqs` — 1536
threadgroups at the 27B geometry against our 192, eight times the parallelism. It is
transplanted here as `kernel_delta_scan_v2` plus the hoisted `kernel_delta_l2norm` the
shape requires, adapted to our transposed state layout and our end-of-scan `1/sqrt(128)`.
It is SLOWER, at every geometry and every length. Isolation timing of one layer's whole
recurrence (`delta_scan_timing`, plateau ms, both arms inclusive of their q/k norm):

| geometry | seq 880 | seq 4096 |
|---|---|---|
| 27B (16 K / 48 V), shipped | 1.97 | 8.56 |
| 27B, llama.cpp shape | 2.73 | 14.81 |
| 35B-A3B (16 / 32), shipped | 1.57 | 6.31 |
| 35B-A3B, llama.cpp shape | 1.88 | 8.93 |

The mechanism is the same one the dual-weight gather taught, running the other way.
Parallelism was not the binding constraint; q/k READ AMPLIFICATION is. Our kernel stages
one normalized q and k vector into threadgroup memory per threadgroup per timestep, so
that traffic scales with the 192 threadgroups. Giving every simdgroup its own column
also gives every simdgroup its own copy of those reads — 6144 of them per timestep at
the 27B geometry, 32x the L2 traffic — and at head dim 128 each lane owns only 4 state
entries, so two `simd_sum` reductions sit on top of 8 useful FMAs per lane per timestep.
The barriers and the redundant in-register norm that the shape removes are cheaper than
that. Hoisting the norm out on its own is a loss too, for a related reason: the separate
dispatch reads and rewrites the whole q|k plane (0.52 ms at seq 880, 1.80 ms at 4096,
27B) to save a reduction that was already amortized over a threadgroup.

The second claim is the one that closes the direction. **The scan is 3% of 27B prefill,
so no decomposition of it can be the 1.8-2.1x llama.cpp prefill gap.** 48 DeltaNet
layers × 1.97 ms is 95 ms of a 2.96 s prefill at 880 tokens, and × 8.56 ms is 411 ms of
a 14.2 s prefill at 3851. Deleting the scan outright would move 27B prefill from ~297 to
~307 tok/s against llama.cpp's 486. The end-to-end A/B corroborates the isolation
numbers to within noise — the llama.cpp shape costs 48 × 6.25 ms ≈ 300 ms at 4096, which
is 1.9% of the wall time, and the measured arms differ by 1.7%. On the 35B the scan is a
larger slice (30 layers, 11-13% of prefill) but that model is already near llama.cpp
parity. Whatever the 27B loses, it loses in the dense projections, not here (2026-07-29).

`kernel_delta_scan_v2` and `kernel_delta_l2norm` are kept behind `XWEN_DELTA_SCAN_V2=1`
rather than deleted, on the `XWEN_MOE_DUAL` precedent and for a sharper reason: the
kernel they mirror is sitting in `reference/llama.cpp` inviting the same proposal, so the
refutation is worth more as a runnable arm than as a paragraph. `delta_scan_timing`
(src/ops/delta.rs, `#[ignore]`d) is the instrument — re-run it before believing any
future claim about the scan's cost, including this one.

**Chunk-boundary device syncs and command-buffer batching granularity are both REFUTED
as levers on the 27B prefill residual.** The residual — +350 to +560 µs/token of
length-dependent prefill cost outside every measured stage — was reproduced in situ at
+410.3 and +437.9 µs/token across two rounds, and per-stage serialization accounts for
only +102.8 of it (mixer_full_attn +53.5, ffn +42.2, residual_ffn +16.8, mask_upload
+7.9, mixer_delta −9.1). So ~335 µs/token exists ONLY when stages pipeline, and the two
obvious pipelining suspects were tested directly and both cleared.

*Cross-chunk accumulation.* `XWEN_CHUNK_SYNC` waits for each 512-token chunk's forward
to complete before enqueueing the next, which prunes candle's buffer pool, clears its
fence map and drops the encoder's barrier history — everything a chunk boundary could
reset. The length delta with it on is +431.1 µs/token against +437.9 without, a
difference of −6.8. Whatever accumulates re-develops inside a single chunk. (The flag
itself costs +9.2 µs/token at 925 and +2.4 at 4k — a per-chunk fixed price, not a
length-dependent one.)

*Command-buffer batching.* `CANDLE_METAL_COMPUTE_PER_BUFFER` swept over 10 / 200 / 1000
against candle's default 50 at 4k: all four means within 0.9%. A 100x range of batching
granularity moves nothing, so the cost is neither submission overhead nor a per-buffer
fixed price.

What is left standing is intra-chunk, and unconfirmed: barrier storms from
buffer-pointer recycling (candle's encoder emits a full `MTLBarrierScope::Buffers`
barrier when a pool-recycled pointer is reused within one encoder session) and
fence-wait pileup (every new encoder waits on every fence in the growing
`prev_ce_outputs` map). Both are consistent with the `XWEN_CHUNK_SYNC` result, and
separating them needs a counter candle does not expose — a patched candle or a Metal
capture. Do not re-propose either refuted lever without an instrument that can see
inside a chunk (2026-08-08).

**Extending the `mul_mv_ext` window past seq 8 is REFUTED — ggml's `ne11_mm_min` 8 is
the right ceiling and is now measured rather than inherited.** The obvious follow-on to
the small-batch win was to raise the ceiling: spans 9-32 are where the verify round's
fixed cost actually lives, and the window's upper edge was ggml's tested envelope rather
than anything anyone here had priced. `XWEN_MV_EXT_MAX_SEQ=32` prices it, and the
multi-row mat-vec loses: spans 16 / 24 / 32 come in **worse than classic by 1.11x /
1.42x / 1.69x**, with span 12 a wash at 0.98x. The degradation is monotonic in span,
which is the signature of the mechanism — a mat-vec reusing one dequantized block across
r1ptg rows stops paying once the token count is large enough for the tiled path's
threadgroup grid to fill, and past that it is strictly the worse decomposition. The
crossover is where ggml put it. Do not re-raise the ceiling without a kernel that
changes this shape (2026-08-08).

## Speculative decoding

**dflash.rs stays in the fork — a removal decision was made and reversed within the
bootstrap session.** The drafter was believed to be Laguna-specific; then the GGUF
survey found ggml-org ships official DFlash sidecar drafters for BOTH Qwen 3.6 models
(arch string `dflash`, block_size 16, mask_token_id, sliding-window pattern; 27B: 5
layers, taps [2,17,32,47,62]; 35B: 6 layers, taps [2,7,12,17,23,28,33,38]). The
subsystem is directly adaptable; adaptation (tap wiring, decoder_arch check, mask token)
is tracked in TODO.md. llama.cpp additionally implements recurrent-state rollback
specifically for qwen35/qwen35moe, confirming spec decode is viable on the hybrid
(2026-07-28). LANDED 2026-07-29: the adaptation shipped and both sidecars load, draft and
verify correctly — see the entries below for what it cost and why it is still opt-in.

**A DeltaNet layer's spec-decode rollback is a recorded per-token trail, not a
truncation — and it costs about a gigabyte while a verify walk is in flight.** A
full-attention layer rolls back for free: it writes each position to its own slot, so
discarding a rejected tail is a length assignment. A recurrent layer has no such
structure — every step overwrites the state — so no image of a single moment
reconstructs an intermediate one. `LayerCache::checkpoint` therefore ARMS the layer and
the verify forward records the state after each token as it goes; `rollback(commit)`
reads the entry for the last accepted token. This mirrors llama.cpp, which keeps K
most-recent-first snapshot slots for exactly this reason. The recurrent reference
produces a fresh state tensor per step, so recording the trail costs handles rather
than copies — but the states are real allocations: at block_size 16 that is roughly
16 × 2 MiB × 30 layers ≈ 1 GB on the 35B and 16 × 3 MiB × 48 layers ≈ 2.3 GB on the
27B, held only for the duration of a verify walk. Accepted for now because correctness
came first and P9 has not measured the spec-decode win yet; a chunked scan that can
replay a short prefix cheaply would let the trail be dropped entirely (2026-07-28).

**MTP sidecars are a second drafter option, deferred.** The MTP GGUFs reuse the parent
arch as one extra full-attention block (`blk.64`/`blk.40` + `nextn.*` tensors) with a
plain KV cache. Evaluate only after DFlash adaptation lands or fails (2026-07-28).

**Drafting is OPT-IN, reversing laguna's opt-out default.** Laguna shipped
`DEFAULT_DRAFT_ENABLED = true`: no flag meant the official drafter. On xwen that made
every zero-flag invocation abort, because the shipped sidecars carry no
`dflash.decoder_arch` key and two more blockers sat behind that one. The load-time checks
stay strict — asking for a drafter that cannot load should fail loudly — but nothing asks
by default. Naming one opts in three ways: `--draft <gguf>`, `--draft official`, or
`draft.path` in the config, the last of which enables on its own rather than needing
`enabled = true` beside it (2026-07-28).
REVISED 2026-07-29 (P9): the load blockers are gone and both sidecars draft well, but
**opt-in stays**, now for a measured reason rather than a broken one. The flip to opt-out
was conditional on the auto-pause controller holding a never-materially-slower property on
both checkpoints, and it cannot: on the 35B-A3B an attached drafter costs ~12% of decode
on rounds where it drafts NOTHING (see the next entry), which is a cost the controller has
no lever over. The 27B gains 1.5-7.4% depending on prompt and run. A default that helps
one checkpoint by single digits and takes 12% off the other is not a default. Revisit when
the fused verify and a cheaper inject land.

**The drafter's per-token cache sync is what decides whether speculation pays, not its
acceptance rate.** Both Qwen sidecars propose well — 85-95% acceptance, and 100% at
`p_min` 0.9 — yet speculation is a 4.8-7.4% win on the 27B and an 11.5-12.7% loss on the
35B-A3B. The discriminator is a fixed cost, not a probabilistic one: every committed token
must run `encode` plus the drafter's per-layer K/V injection to keep its cache in step with
the target's, about 14 small Metal dispatches for ~1.2 ms. Measured directly by an arm
that can never draft (`--draft-p-min 1.1`, 119 of 127 rounds paused): 92.6 tok/s against
105.1 plain on the 35B, indistinguishable from the best real drafting arm. That is 12% of
a 9.5 ms plain step and 2.8% of the 27B's 43 ms one. The sync is mandatory while a drafter
is attached — a drafter whose cache falls out of step can never resume speculating
(`drafter_span_rows` returns 0) — so the only fixes are to make it cheaper (it is
dispatch-bound, like the pre-fusion MoE glue) or to let the controller detach entirely
rather than merely pause. Both are ledgered under TODO.md P9 (2026-07-29).

**Speculative decoding's batching win does not currently exist in the DeltaNet layers, and
that is the ceiling on P9.** **SUPERSEDED 2026-07-29 (P9a, same day)** — the K-snapshot
fused verify landed and the predicted unlock was measured. The batched verify's marginal
cost fell from 9.42 to 3.57 ms/position on the 27B (fit over spans 2-32; at the span-6
operating point, 41.0 → 31.2 ms/position with the fixed cost included), and the
end-to-end wins moved from single digits to **27B +19.3-21.0% code / +7.6-8.4% chat, 35B
+18.1-19.8% code / +12.6-12.8% chat** — the 35B flipped from a 12% regression to a
double-digit win because the pause controller stopped pausing (35B code: 54-of-66 rounds
paused → 0-of-20) once verify got cheap, not because the ~1.2 ms drafter cache sync
(P9b) got any cheaper. The new ceiling is the verify round's FIXED cost: ~149 ms on the
27B (~113 ms above a plain step, ~60% of a typical round), no longer the DeltaNet scan —
pricing it is the successor ledger item. Original entry, kept for the reasoning:
Under an armed rollback trail a multi-token chunk takes the
frozen reference scan (linear_attn.rs:194-205), which walks tokens one at a time in candle
ops. So the 48-of-64 (27B) and 30-of-40 (35B) layers that are DeltaNet cost the same per
position inside a verify forward as they would as separate decode steps: 245 ms for a
~6-position 27B verify against a 43 ms plain step, i.e. 39 ms per verified position.
Speculation only wins on the attention and FFN layers, which is why the measured gains are
single-digit percentages rather than the 1.39-2.29x reported elsewhere on Apple silicon.
Accepted for this arc deliberately: the alternative was building the K-snapshot fused
verify inside the adaptation, and the adaptation had to be verified first. The consequence
to carry forward is that **the K-snapshot work is the precondition for speculative
decoding to pay here, not an optimization of it** (2026-07-29).

**The drafter's sliding window is implemented as a cache narrow plus a ≤15-column mask,
not as a full-width score mask.** The sidecars window every layer but the last (2048
positions on the 27B, 4096 on the 35B) and llama.cpp masks `p1 - p0 >= n_swa`, keeping
`[p - window + 1, p]` on the past side. The block's 16 queries have floors spanning at
most 15 positions, so their windows' union is one contiguous range: `attention` narrows
the cache to it and masks only the columns between the individual floors, or not at all
while the context still fits inside the window. The alternative — one additive mask over
the full `[16, context]` score row — is simpler but leaves every windowed layer costing
O(context), which throws away the only thing the window is for. With the narrow, five of
the 27B's six drafter layers cap at 2048 positions per round and only the final full layer
grows with depth; that retires half the argument behind `DEFAULT_DRAFT_CTX` being 8192 and
is why re-deriving that cap is now a ledger item rather than a settled number. A
ring-buffer cache would go further and is ledgered; the flat position-indexed cache stays
because it is what makes `DrafterImage` a straight prefix copy (2026-07-29).

**A drafter is checked against its target in exactly one place, because two places
drifted.** Nothing in a DFlash sidecar's metadata names the checkpoint it belongs to, so
pairing one with a target is only ever the caller's assertion, and there were two callers
asserting it differently: `Generator::attach_drafter` on the CLI path and serve's
`check_draft_geometry` at startup. Serve's lacked the hidden-size comparison, and the tap
bound does not separate the two shipped sidecars in one direction — the 35B-A3B drafter's
translated taps top out at 37, inside the 27B's 64 layers — so `xwen serve --model-size
27b --draft <35B drafter>` passed startup validation and failed the first job, which is
precisely what startup validation exists to prevent. Both callers now go through
`DflashConfig::check_against_target`, so a check added for one is a check the other gets.
It covers what can be cross-checked (hidden size, tap bounds, mask id against the target's
vocabulary) plus what can only be checked for internal consistency (head counts and their
divisibility, an even head dim, a block size of at least 2 — a block of 1 is the anchor
alone and could never carry a draft). The drafter's own depth, FFN width and head counts
describe the drafter and have nothing to be compared against (2026-07-29).

**Three drafter-graph forms came from the oracle, not from the inherited code, and all
three contradicted it.** `reference/llama.cpp/src/models/dflash.cpp` is the executable
reference and the laguna branch it was forked from no longer exists, so where the two
disagree the oracle wins. (1) The noise block is NON-CAUSAL —
`llama_set_causal_attn(ctx_dft, false)` at common/speculative.cpp:1004, with the causal
branch of the mask builder guarded by `if (causal)` at llama-kv-cache.cpp:1793. This is a
block-diffusion drafter: it denoises the whole block in one forward, so a later block
position informs an earlier one, and `the_noise_block_attends_to_itself_in_both_directions`
pins it. (2) The KV-injection path applies NO `attn_norm` (dflash.cpp:252-253 projects the
raw encoder output), while the query path does — the two paths deliberately disagree, and
`enc.output_norm` is the injection path's only norm. (3) The encoder is three ops,
concat → `fc` → `enc.output_norm`, with no per-tap norm or scale; the `enc.aux_norm`
tensor the inherited code required is absent from both shipped tensor tables (2026-07-29).

**Cache sizing figures are derived per checkpoint on `hub::Model`, not carried as
constants.** `serve/config.rs` had inherited laguna's geometry verbatim:
`FULL_KV_BYTES_PER_TOKEN = 12 full layers × 8 KV heads × 128 head_dim × 2 × 2` = 48
KiB/token, and a 72 MiB snapshot described as "deep copies of the 36 SWA rings". Every
factor is wrong for Qwen and the model has no SWA layer at all. The real figures:
20 KiB/token on the 35B-A3B (10 full layers × 2 KV heads × 256 head_dim × K+V × f16)
and 64 KiB on the 27B (16 × 4 × 256 × 2 × 2); a snapshot is DeltaNet recurrent state —
f32 conv window plus f32 delta state over the linear layers — at a fixed 62.8 MiB
(35B-A3B) or 149.6 MiB (27B) whatever position it covers. The consumption sites are all
display (the `--init` template) plus the `MAX_CHAIN_BYTES` justification, so
`Model::kv_bytes_per_token()` and `Model::snapshot_bytes()` derive them from a per-model
geometry table with a test pinning the arithmetic; anything holding a real `XwenConfig`
should measure from that instead. One consequence surfaced: a 27B conversation filling
the trained context while retaining dozens of snapshots can exceed the 24 GiB
`MAX_CHAIN_BYTES` and be refused, which is the cap working as designed — a refused chain
costs a re-prefill, an allocation failure at twice the chain size takes the process down
(2026-07-28).

## Serving

**The serve/ tree (Anthropic + OpenAI + native dialects, TUI, queue, prefix cache, disk
tier) is inherited as-is**; it is architecture-agnostic. The KV export/import and disk
tier must additionally carry the recurrent state for the 3-of-4 linear layers — KV cache
alone no longer reconstructs a prefix. Native endpoint moved `/maxuna/v1/*` →
`/xwen/v1/*` (2026-07-28).

## The prefix cache and the disk tier

Inherited from laguna; correctness now depends on snapshotting (KV cache for the 10–16
full-attention layers) + (conv + delta state for the linear layers) as one unit. Sizing
is favorable: the 35B keeps KV for only 10 layers with 2 KV heads — the hybrid's state
is far smaller per token than a uniform transformer's (2026-07-28).

**`CONTAINER_VERSION` stays at 2 for the DeltaNet snapshot variant — deliberately, not
by oversight.** The version discriminates FRAMING (header fields, directory layout,
record tags) and nothing else, because two mechanisms already cover the payload and
leave a bump nothing to catch. The checkpoint binding (hash plus file length, checked in
`read_header`) means an image can only be read back beside the exact file that wrote it,
so a laguna-era image cannot reach a Qwen build at all. Within a checkpoint,
`kv_cache`'s per-layer kind tags (`LAYER_FULL`/`LAYER_SWA`/`LAYER_LINEAR`) give each
kind its own field layout and dtype, and `check_restorable` rejects a layer whose kind
or shape disagrees with the live cache. The recurrent-state snapshot is therefore a new
per-layer tag inside unchanged framing. Bump the version only when the framing itself
changes; the invariant is recorded at the constant so the next reader does not have to
re-derive it (2026-07-28).

## Tokenization, chat, tool calls

**The Qwen tokenizer.json (12,807,982 bytes, byte-identical between the two model
repos, sha256 5f9e4d49…) is vendored at reference/tokenizer.json and embedded via
include_bytes!, following laguna's embedded-tokenizer decision.** Qwen2 byte-level BPE,
NFC normalizer, no BOS ever prepended (`add_bos_token: false`, no post-processor). The
split regex differs from Qwen3 by `\p{M}` handling — do not reuse a Qwen3 regex
(2026-07-28).

**chat.rs is a hand-written Rust port of the official chat_template.jinja (7764 bytes,
byte-identical across both repos), keeping laguna's content/structure separation** so
pasted text discussing control tokens can never become control tokens. The subtle rules,
verified by rendering the real template: string tool-arguments render RAW (non-strings
JSON-encode); OpenAI-style JSON-string `arguments` must be parsed into a map first
(template raises on strings); thinking blocks are kept only for turns strictly after the
last user turn that is not wholly a `<tool_response>` wrapper (or all, under
`preserve_thinking`); generation prompt opens an unclosed `<think>\n` (thinking on) or
emits a closed empty block (thinking off); consecutive tool results collapse into one
user turn. Rendered test vectors from the bootstrap research are the fixture set
(2026-07-28).

**One deliberate divergence from template byte-parity: a tool result as the FIRST
message is refused** (`ChatError::ToolResultOpensConversation`). The reference template
hits undefined `loop.previtem` there and emits a turn that closes without ever opening
(`<|im_end|>` with no `<|im_start|>user`); byte-parity would mean handing the model a
malformed boundary. Refusal ordering otherwise follows the template exactly:
NoMessages → NoUserQuery → SystemNotFirst (2026-07-28).

**Bodies are stripped with Python's str.strip() whitespace set (29 codepoints,
including U+001C–U+001F), not Rust's `trim`.** Jinja's `|trim` is str.strip(), and the
difference decides real behavior: whether a body reads as a bare `<tool_response>`
wrapper (which moves last_query_index and therefore which turns keep their reasoning)
and whether an assistant turn counts as empty when its first tool call picks its
separator. Verified against an exhaustive Unicode sweep (2026-07-28).

**Constrained decoding's control-token safety is a compile-time property enforced by a
test, not a runtime force-mask.** toktrie marks every `<…>`-shaped added token special
regardless of the tokenizer's special flag, so no grammar byte can ever match a control
marker; a per-draw 250-id mask sweep would duplicate that guarantee on the hot path and
need an EOG carve-out. The guarantee rests on toktrie's bracket HEURISTIC, not on
special:true — a future marker spelled without angle brackets would be
grammar-reachable, which is why `no_control_token_is_ever_offered` sweeps the full
control range at every step and asserts the mask stayed wide. The constrain trie is
sized to the model's logit width 248320 (padded tail unreachable by construction), and
`new()` refuses a checkpoint with a different width — this sizing fixed a latent bug
where every constrained serve request died on a short mask (2026-07-28).

**tokenizer.rs is the single owner of every token id in the crate**, including the
hardcoded second stop id 248044 that no GGUF key advertises; config.rs imports it. Two
vocab sizes are exposed deliberately: `vocab_size()` = 248070 encodable id space,
`PADDED_VOCAB` = 248320 logit width — callers pick by which side of the sampler they
are on (2026-07-28). The serve engine's tool-call span parser is now the same rule
rather than an exception — see the entry below for what the exception cost.

**The serve engine parses Qwen's real call format, and a span it cannot read degrades to
text.** The inherited parser was laguna's twice over. Its span markers were literal ids
`25`/`26`, which in Qwen's vocabulary are `:` and `;` (the real `<tool_call>` pair is
248058/248059), so every colon in ordinary prose opened a phantom span and every
semicolon closed one — truncating replies into a discarded span or reporting a
fabricated call, while genuine `<tool_call>` tokens passed through as text. Its interior
grammar was laguna's `<arg_key>`/`<arg_value>`, strings absent from Qwen's vocabulary
and never emitted by chat.rs. The parser now sources both ids from `LagunaTokenizer` and
reads the format chat.rs renders: `<function=NAME>` then per-argument
`<parameter=KEY>\nVALUE\n</parameter>` then `</function>`, one function per span, with
the newlines around a value treated as framing rather than content. Two rules follow
from the template rather than from taste. The `</tool_call>` token is structural
wherever it lands, mid-value included — chat.rs writes a literal `</tool_call>` inside
an argument as ordinary content, so it never encodes to the added token, and reading the
token as content would let one malformed value swallow the rest of the reply. And a span
that never names a callable tool degrades: its raw text, markers included, goes to the
client as answer text with a logged warning, instead of being silently dropped as the
old parser dropped it. Never discard, never fabricate. The class of bug is closed by
construction in the tests: they drive the emitter over ids from the real embedded
tokenizer, round-tripping conversations that chat.rs rendered, and one hostile case
feeds prose full of `:`, `;` and `<function=` text and asserts zero calls with
byte-identical output (2026-07-28).

**`--ban-string` protects the stop ids the decode loop actually uses.** `scan_banned`
guarded the compile-time `LagunaTokenizer::EOG` while the loop stops on
`Sampler::eog_ids()`, which `XwenConfig` derives from the checkpoint's metadata. The two
agree on both shipped files, so nothing was broken — but they are independent sources,
and a checkpoint declaring a different `eos_token_id` would leave its real stop token
bannable, letting `--ban-string` remove the only token that can end a reply. The
protected set is now passed in from the sampler, so the guarantee holds by construction
rather than by the two sources happening to match (2026-07-28).

## Thinking budget and sampling controls

Laguna's `GenEvent` thinking/answer split carries over; Qwen 3.6 seeds generation inside
an open `<think>` block, so the generation loop starts in thinking state whenever the
template opened one, and `</think>` (token 248069) is the split marker. `<think>` /
`</think>` are single tokens but `special: false` in the tokenizer — encoding user text
never produces them via the special-token path; the loop must treat them by token id
(2026-07-28).

The sampler is in-crate rather than candle's `LogitsProcessor`, because at vocab 248320
the processor's shape costs ~0.6 ms of CPU per token: a temperature pass, a full CPU
softmax, a `to_vec1`, and a `select_nth_unstable_by` over 248320 indices behind an
indirect comparator. The replacement keeps the distribution and changes the execution:
the full-vocabulary softmax runs on the device holding the logits (one Metal kernel,
not 248320 CPU `expf` calls) and the candidate set comes from a single-pass streaming
top-k. 0.819 → 0.406 ms/token measured at real width by `sampler_decode_bench`
(2026-07-28).

**Top-p renormalizes over the top-k survivors before the cut, following llama.cpp and
HF, NOT candle.** `truncate_top_p` is `llama_sampler_top_p_apply`: `top_p >= 1.0` is a
no-op, otherwise the survivors are rescaled to sum to one and the shortest prefix whose
cumulative mass *reaches* `top_p` is kept — the comparison is `cum_sum >= top_p` and the
token that crosses the threshold is included, so the kept mass is at least `top_p` and
never just short of it. llama.cpp's other knob, `min_keep`, is not carried: its default
is 0 (disabled), and the loop's own guarantee — the first iterate can only cut at index
1 or later — is the only floor that default produces. HF's `TopPLogitsWarper` after
`TopKLogitsWarper` is the same rule (2026-07-29).

The convention this replaced, kept on the record because the divergence explains the
shape of the surrounding code: candle's `TopKThenTopP` softmaxes over the whole
vocabulary, truncates to k, and applies the cut to the survivors *without* renormalizing
them, so `top_p` was a threshold on full-vocabulary mass and the cut was skipped outright
whenever the top-k set held less than `top_p` of the total. The 2026-07-28 perf retarget
preserved that deliberately — it was a performance change, and switching sampling
conventions inside it would have been an unreviewable behavior change riding along — and
ledgered the question instead. Resolved here as a semantics question: llama.cpp is the
project's declared ground truth everywhere else, and `--top-p` now means what a llama.cpp
user expects. Sampled outputs change: a seeded stochastic run draws a different (equally
valid) token stream than a pre-2026-07-29 build, and the change is one-directional —
renormalizing only ever cuts the same or more. Greedy decoding is untouched, so the
parity gate is unaffected. Two things follow from the switch. The fast path no longer
*needs* a full-vocabulary softmax — renormalizing over the k survivors is exactly a
k-wide softmax, so a Metal top-k could ship ~20 values instead of the 993 KB row (still
a TODO, not done here). And the truncation stopped being sensitive to which backend ran
the softmax: the shared denominator now divides back out of the cut as well as the draw,
so the device fast path and the CPU `SampleControl` path truncate identically instead of
being able to disagree by an ulp at the threshold (2026-07-29).

One residual ULP-level divergence from llama.cpp is known and accepted (2026-07-29,
found by outside-model review with a reproduced counterexample): llama.cpp truncates the
raw logits to k and re-softmaxes the survivors, while xwen divides the full-vocabulary
softmax's survivors by their sum. The quotients are algebraically identical but not
bit-identical — at an exact f32 boundary the cumulative walk can land one ulp apart and
keep one candidate more or fewer (verified: logits [-10.193466, -19.933178, -2.5489683],
k=2, p=0.9995216131210327 keep 2 here and 1 there). Not worth restructuring the device
fast path over, since it reads back probabilities, not logits; the `llamacpp_filtered`
test oracle shares xwen's ordering and is therefore blind to exactly this class, which
is why the bound is documented rather than tested.

Consequence recorded so it is not mistaken for a regression: seeded stochastic runs
produce different (equally valid) token streams than pre-2026-07-28 builds. candle's
candidate list came out of `select_nth_unstable_by` in unspecified order and this one is
sorted descending; a weighted draw maps its single uniform through the cumulative
weights, so the same seed lands on a different token. Greedy decoding is bit-identical
(argmax over the CPU copy, ties to the lowest id, no RNG touched), which is why the
parity gate — greedy end to end — is unaffected (2026-07-28). The top-p convention
switch moved the seeded streams a second time, and for a second reason: the candidate
set itself is now narrower wherever the cut bites (2026-07-29).

**A NaN in the logit row fails the draw; ties at the top-k boundary go to the lowest
id.** Two contracts the in-crate sampler states rather than inherits. NaN loses every
ordered comparison, so both a scan that skips it (what the rewrite first did) and one
that lets it win (candle's argmax pins index 0) turn a corrupt forward into a plausible
token; the sampler errors on it instead, on every path including greedy, which is what
the parity gate runs. `-inf` is a separate thing — it is how the controls exclude an id
— and stays skippable. The tie contract is the one place the sampler is deliberately
*stronger* than candle rather than equal to it: `select_nth_unstable_by` leaves which of
several equal entries survives unspecified, while the streaming top-k's strict `>`
against the floor keeps the lowest ids, so the candidate set is a function of the
probabilities and not of the traversal that built it. Equivalence with candle is
therefore claimed as distribution equality for untied inputs plus deterministic low-id
selection at exact boundary ties, and — since the top-p conventions diverge — only where
no top-p cut applies. The softmax denominator no longer reaches the outcome at all: it
cancels out of the weighted draw and, once the survivors are renormalized, out of the
cut too, so the two backends agree bit-for-bit on which candidates survive. llama.cpp is
the oracle for the truncation itself (2026-07-29).

**Ids past the tokenizer's vocabulary are not drawable.** The output layer is padded
(248320 rows against 248070 encodable ids) and the rows in between decode to nothing, so
the sampler carries the encodable bound, checks it against every logit row it is handed,
and narrows the row to it before the softmax. Narrowing rather than masking is what keeps
the padding out of the denominator too, which is what lets the device fast path and the
CPU `SampleControl` path softmax the same values. The bound is passed in from the
tokenizer at construction, never written as a literal: the two vocabulary sizes are a
per-checkpoint fact and `PADDED_VOCAB` vs `vocab_size()` is the distinction that decides
which callers belong on which side (2026-07-29).

## Measurement discipline

Inherited unchanged from laguna: state the power mode with every number, never report
first-forward prefill as steady-state, bench via the scripts with warmup, and one
~20–70 GB process at a time (2026-07-28).

**A/B perf comparisons must INTERLEAVE the two arms, and a sequential matrix is not a
valid A/B.** Measured 2026-07-28 while benching the fused DeltaNet kernels: a
back-to-back matrix of eight `xwen generate` runs drifts **20–35% slower** end to end,
uniformly across both arms and both checkpoints, over roughly ten minutes of continuous
GPU load. `pmset -g therm` records nothing while it happens, so there is no flag to
check — the only tell is that the control arm moves too. Two consequences, both learned
the expensive way. Run every arm of a comparison adjacently (F, C, F, C, …) and report
the median of each, so both arms sample the same thermal envelope; the ratio survives
drift even when the absolutes do not. And treat any absolute tok/s figure as a
warm-machine number unless it came off an idle machine — the first pass of that matrix
reported the 27B at 13.9 tok/s decode and a cooled, interleaved re-run put the same
build at 19.0.

**This does NOT touch the parity gate, and it is worth saying so explicitly.** Every
tier grades logits, agreement counts or mean NLL — all arithmetic, all thermally
invariant. A throttled run produces bit-identical dumps, just later. The only
thermally sensitive figures in docs/parity.md are the wall-clock runtimes ("42 s
warm"), which are scheduling guidance and not gate criteria. So the interleaving
protocol is a bench-work rule; the gate needs no equivalent (2026-07-28).

**`pgrep -f` is not a usable "is a model running?" guard — test what the process is
EXECUTING, not what its command line mentions.** The pattern string appears in the argv
of whatever runs the check, so `pgrep -f "logits-dump"` matches its own wrapper and
aborts over a model process that does not exist. This bit both
`scripts/parity-gate.ts`'s preflight and an ad-hoc bench guard within the same hour.
Note the failure mode is not "the wrapper is a shell" — a bun/python/make wrapper, or a
`git diff -- src/bin/logits-dump.rs`, matches just as well — so excluding `sh -c` is a
heuristic, not a fix. Two structural fixes, both sound: match the process NAME exactly
(`pgrep -x logits-dump`), or keep `-f` and filter on `argv[0]`, which is what
parity-gate ships (`isModelProcess`, unit-tested offline against captured lines — a
property `-x` cannot offer since it is opaque to the caller). `-x` does work —
`pgrep -x bun` matches 3 live processes where `pgrep -f bun` matches 15, which is
exactly the argv-only false-positive class being eliminated.

Three traps cost both agents time while establishing that, all worth knowing before
probing process state in an agent sandbox: background processes do not survive
(`nice(5) failed: operation not permitted`), so a positive-case test needs a process
started some other way; `ps -p <pid>` returns nothing even for pids `pgrep` can see, so
"is it alive?" comes back empty and reads as "it isn't"; and `pgrep zsh` returns 0 in
EVERY form (`-f`, `-l`, `-x`, bare), so probing `-x` against a shell looks like `-x` is
broken when the target is simply invisible. Pick a probe target you can independently
confirm is running and visible — `bun` works here. A guard whose failure mode is a
concurrent 20 GB load deserves both halves tested against real processes, permissive
and restrictive, not just its matcher unit-tested (2026-07-28).

**Per-stage forward timing is done IN SITU, by device sync, and only its
length-DIFFERENTIALS may be read as measurements.** `src/stack_profile.rs`
(`XWEN_STACK_PROFILE`) decomposes a chunk's wall clock into the stages `run_stack`
actually runs, on the real weights in the real dispatch order, because a stage budget
assembled from synthetic microbenchmarks cannot see what the wall clock holds that no
stage claims — which is the entire question it was built for. Five design rules, each
answering a way the instrument could lie:

- Stages are bracketed by `Device::synchronize`, so a stage's total is completed GPU
  work rather than enqueue time. Adjacent stages SHARE the sync between them, so the
  brackets add one sync per stage and not two.
- Host-side gaps go to their own `inter_stage_host` bucket. The sync closing one stage
  opens the next, so an unbucketed gap would be silently charged to whichever kernel
  follows it, and per-token cost living in the glue would be mis-attributed to the
  stage after it.
- `unaccounted == 0` is an ENFORCED bracket-integrity invariant, not a result. Every
  interval inside a chunk's bracket belongs to some bucket by construction; a nonzero
  value means the brackets are wrong. Anything real shows up as a bucket.
- Phase (prefill vs decode) is DECLARED by the caller (`XwenModel::set_phase`), never
  inferred from token count. Inference has two failure modes that both occur in
  practice: a prompt whose length is ≡ 1 mod 512 ends prefill with a one-token chunk,
  and a speculative verify forward feeds a whole span while being decode.
- `XWEN_BENCH`'s warm-up pass is excluded by resetting the accumulators after it, so a
  dump never averages a cold chunk with warm ones.

**The reading discipline is the load-bearing part.** Per-stage syncs roughly DOUBLE the
prefill wall — absolute synced numbers are ~2x plain and mean nothing on their own. What
survives is the differential between two prompt lengths: the per-stage sync overhead is
approximately constant per token, so it cancels out of a length delta. Any claim taken
off this instrument must be a length-differential, never an absolute. It is built for
plain `--no-draft` generation (the speculative and server paths accumulate correctly but
print no dumps), and with the variable unset the cost at each instrumented site is one
`Option` check. `XWEN_STACK_PROFILE` and `XWEN_CHUNK_SYNC` are both stripped in
`parity-gate.ts`'s `baseEnv()` (2026-08-08).

**A warm-up pass that reads faster than the timed pass following it is a thermal
ordering artifact, not evidence.** The warm-up runs on the cooler chip. This looked like
evidence about allocator-pool state during the residual diagnosis and was not; the
profiler excludes warm-up from its dumps for exactly this reason (2026-08-08).

## Process

Inherited unchanged: multi-reviewer review with external model families on evidence
(reviewers recorded as wrong with disproofs, not just as right); a reviewer reads the
path you wrote, a live check walks the path you forgot; let the existing suite arbitrate
a proposed fix; docs drift is tracked work. Every shipped arc updates log.md (dated
entry) + README if the surface changed + this file if a decision was made, changed, or
refuted — a TODO.md update alone is not sufficient (2026-07-28).
