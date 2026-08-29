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

**Qwen3.8-27B is a registry entry, not a port (2026-08-14).** Its `config.json` is
byte-identical to Qwen3.6-27B's — same graph, hparams, tokenizer ids, generation config
— and its GGUF declares the same `qwen35` architecture, rope sectioning and ssm keys, so
it needs no model math, no new geometry and no parity run: it is the same forward pass
over different weights. `ggml-org/Qwen3.8-27B-GGUF` for the same reason the 3.6 files
were chosen. Three things about it are genuinely new. It ships **no DFlash sidecar**, so
speculation is absent rather than configurable — every drafter accessor is `Option` and
a zero-flag run logs one line and decodes plain (~25 tok/s, against the 27B's drafted
37-38); the repo's MTP sidecar is unread (TODO.md). [SUPERSEDED 2026-08-15 by the MTP
arc: the sidecar is read, the accessors resolve, and a zero-flag run drafts. The
`Option` shape stayed — it is about a checkpoint that ships no sidecar, not about this
one.] Its Q4_K_M mix puts the 16
`attn_output.weight` tensors at **Q6_K** where 3.6 had Q8_0 — upstream's
`output.weight=q6_k` rule substring-catches `attn_output`; nothing asserts on that
plane's quant and lm_head already exercises Q6_K. And its tokenizer.json is NOT
byte-identical to 3.6's: it adds seven audio/TTS specials at 248070-248076 over an
identical base vocab and merge table, which the embedded 3.6 tokenizer therefore
tokenizes text identically to (see "Tokenization" for what was and was not decided).

**Sampling defaults follow generation_config.json: temp 1.0, top_p 0.95, top_k 20.**
[SUPERSEDED 2026-08-19 for the sampling half: defaults are now keyed to thinking mode —
see the mode-keyed entry below. generation_config.json's values ARE the thinking set,
which stays the default for thinking runs and for every mode-less path.] Stop tokens
are the generation_config list `[248046 <|im_end|>, 248044 <|endoftext|>]` —
config.json's single `eos_token_id: 248044` is wrong for chat and runs straight past
turn boundaries (2026-07-28).

**Sampling defaults are mode-keyed: thinking temp 1.0 / top_p 0.95, non-thinking 0.7 /
0.80, top_k 20 both, identical across all three checkpoints (2026-08-19).** The
evidence is the official model cards, which key their recommendation to thinking on/off
and nothing else: the HF READMEs of Qwen/Qwen3.6-27B, Qwen/Qwen3.6-35B-A3B and
Qwen/Qwen3.8-27B all give thinking 1.0/0.95/20 and instruct 0.7/0.80/20 — three cards,
two sets, no per-checkpoint variation, which is why
`SamplerOptions::recommended(thinking)` takes a bool and not a `Model`.
generation_config.json carries only the
thinking set, which is why the old fixed defaults were that set and why `Default` stays
it (raw prompts and benches have no chat mode and keep sampling as they always did).
The resolution order everywhere is explicit value → mode recommendation: CLI sampling
flags became Option-valued (a mode-dependent default cannot live on a clap
`default_value_t` — the DraftArgs precedent), and serve's fixed
`DEFAULT_TEMPERATURE`/`TOP_K`/`TOP_P` constants are gone because a server-wide constant
cannot know a request's mode; `ServeSettings` sampling keys are Options resolved per
request AFTER thinking is known, request over config over recommendation. A pinned
config value deliberately pins one number for both modes — that is what an operator
writing a number means — and unset gives each request its mode's own. The cards'
remaining recommendations did not ship: the penalties (see "Thinking budget and
sampling controls") and the 3.6 pair's third "thinking, precise coding" set
(0.6/0.95/20), which is not auto-selectable — nothing in a request says "coding" — and
is reachable as an explicit `--temp 0.6`.

**`--reasoning-effort` on a 3.6 checkpoint is a startup error, not a no-op
(2026-08-19).** The flag names a parameter of the 3.8 chat template; the 3.6 template
has none, so a supplied level would change nothing. Ignoring it would train the
operator to believe it did something — the `--model-size` rule again: flags
cross-check instead of shrugging, and the check runs before the 20 GB load. Unset is
allowed everywhere because the default level (xhigh) renders nothing on 3.6 anyway.
The serve-side `[thinking] effort` / `--reasoning-effort` default is the deliberate
exception: it is a server-wide setting on a server that may load any checkpoint per
request, so it is documented as inert on 3.6 rather than refused (refusing would make
a 3.8-tuned config invalid the day a 3.6 request arrives). Both `--no-think` and
`--reasoning-effort` are rejected with `--raw`, the same class as the existing guarded
combos: they describe the chat template, which a raw prompt never renders.

**`max_ctx` is a ceiling, not an allocation; the CLI defaults to 131072 and serve to
the trained window (2026-08-11).** Full-attention KV buffers start at 8192 positions
(`model::KV_INITIAL_CTX`) and double on demand up to `max_ctx`
(`LayerCache::ensure_full_capacity`; growth is monotonic for a model's life and logged
per step). Growth copies the WHOLE old buffer, deliberately not just the committed
rows: the grown buffer is the old one plus a zeroed extension, so every property that
held of the fixed allocation — `LayerSnapshot::Full`/`LayerCheckpoint::Full` carrying
no data because restore and rollback are length truncations, rows above a rewound
`len` still holding what they held — carries over with nothing to argue about which
engine flow depends on rows past `len`. A committed-rows-only copy was the first cut
and was replaced for exactly that argument's sake; the cost difference is a bounded
device blit, O(log) times per lifetime. Page-in grows through `import_full_kv` exactly
as a prefill would, and the host-image pre-flight no longer bounds a restore position
by allocated slots (max_ctx, checked at model level, is the real bound). What "reset" means is
dropping the model: the serve idle unload therefore shrinks a grown cache for free, and
no in-life shrink mechanism exists or is wanted (capacity is a high-water mark of real
usage). The alternative — preallocating at max_ctx, the pre-2026-08-11 behavior — made
big defaults expensive (16 GiB of idle KV on a 27B serve at 256k) and kept the CLI
default pinned at a timid 8192; lazy growth makes the 131072 CLI default cost 0.5 GiB
(27B) / 0.16 GiB (35B) until a prompt actually grows past 8k. A growth pass ends in
one `wait_until_completed`: candle's Metal pool frees the replaced buffers only at a
sync, and without one the whole pass holds old and new allocations side by side. Rope
tables still build to max_ctx at load — 64 MB at 262144, not worth the machinery
(2026-08-11).

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

EXTENDED 2026-08-08 (later the same day): **there is now a SECOND routing site,
`Proj::DenseF16Q8`, and it is a second site rather than a second window.** The q8_0
attention and DeltaNet projections never touch `QLinear` on the default path — they are
dual-storage planes read by `ops::matmul_q8` — so the one-decision-point property could
not reach them and the choice was to add a site or leave 64 layers' worth of projections
re-reading their weights once per token. The site takes the plan from `mv_ext_window`
verbatim and checks `mv_ext_supported`, so `XWEN_MV_EXT_CLASSIC` reverts it identically;
what it adds is a 16-byte activation-alignment guard, because the ext kernel reads the
activation as `float4` where the gemv reads scalars at any offset. Two asymmetries are
deliberate and documented at the call site: `XWEN_MV_EXT_MAX_SEQ` cannot widen this site
past 8, since the enclosing `Q8_DECODE_MAX_SEQ` arm already routes seq > 8 to the dense
f16 plane; and the alignment guard is a silent per-call fallback that the env-derived
`mv_ext` provenance field structurally cannot report, which is acceptable only while
every production activation here is offset-0 — a strided caller must be preceded by
recording what actually ran, the way the `delta` field does. Sites deliberately left
out: `ssm_beta`/`ssm_alpha` (dense f32 `[5120,96]`, nothing to stream), the MoE routed
experts (the `mv_id` family, already refuted at these spans), the `seq == 1` lm_head
bypass (a strict-tier bitwise anchor), and ggml's f16 ext variant (the q8_0 alias over
the same weight streams half the bytes, so an f16 port would be strictly worse at every
site that has both). Measured on the 27B verify forward at `n_past` 512, interleaved
against a HEAD binary, 5 reps/arm pooled over two sessions, medians: span 8 **175.20 →
154.19 ms (−12.0%)**, span 6 −6.0%, span 4 −5.0%, spans 12-48 unchanged. Span 2 reads
+2.3%, but the coverage arm ran second in every pair and the same protocol reads
pairwise medians of +1.6 to +2.3% at spans 12/16/24 where the kernel cannot run — so
that cell is a wash rather than a regression, and flooring the window at t>=3 is
ledgered as needing an A/B designed to separate the effect from that ordering bias
(2026-08-08).

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
QUALIFIED 2026-08-08 (later the same day): **the direction claim is about the DISPLACED
path, so it is per-site, and at the new `Proj::DenseF16Q8` site the two paths are
LEVEL.** What that site displaces is the vendored q8_0 gemv, which narrows nothing
either — it multiplies raw int8 quants by an f32 delta and accumulates in f32, exactly
as this kernel does — so both sit at ~1e-6 and their ratio is pure reduction-order
noise, measured within 1-2% of each other with the better one varying by shape. Hence a
separate test constant: `GEMV_MULTIPLE` 2.0 rather than `CLASSIC_MULTIPLE` 1.0. The 2x
band is wide enough not to grade noise and narrow enough that an f16 staging path
creeping in (~1e-4, two orders out) still fails. "Never the further from exact" is
therefore true at the `QLinear` sites and not a property of the kernel; the `mv_ext`
provenance doc-comment in `tests/parity.rs` was corrected to say so.

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
REVISED 2026-08-15 (MTP arc): **landed, and for a checkpoint that did not exist when this
was written.** DFlash adaptation landed (P9) and the trigger recorded here was never met
on the 3.6 pair — a better drafter would not have helped them. What re-opened it was
Qwen3.8-27B, which ships no DFlash sidecar at all, so the choice there was MTP or plain
decode rather than MTP or DFlash. See the entries below for the selection, the head's
shape and the fitted defaults.

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

**`draft_p_min` is PER-CHECKPOINT — 0.5 on the 27B, 0.3 on the 35B-A3B — while
`pause_margin` stays a single shared 1.0.** Both were fitted together on 2026-08-08 by
two independent 120-run sweeps of `scripts/retune-draft.ts`, and they came out shaped
differently, which is why one knob moved home and the other did not. The 27B's target
forward is expensive, so it wants short, confident drafts: at 0.5 its chat prompt stops
pausing entirely (13-18 paused rounds at 0.2/0.3 → 0), acceptance goes 57% → 78%, and
mean-of-medians reads 37.3 / 37.2 tok/s against 33.0 / 33.5 at the shipped 0.3 — +46-52%
over plain. The 35B-A3B's forward is cheap enough that drafting deeper at lower
acceptance still pays, and 0.5 costs it ~2.5% (125.2-125.3 against 127.9-128.4 at 0.3).
Both winners replicated across both runs. A single shared value would therefore have to
pick which checkpoint to be wrong for, so the default moved to
`Model::draft_p_min_default()` in `src/hub.rs` — one const arm per checkpoint, resolved
by the CLI (`DraftArgs.draft_p_min` is now `Option<f32>`) and by serve's merge (via
`CliOverrides.model_size`); `DEFAULT_DRAFT_P_MIN` is gone and `SpecParams::default()` is
documented as a base every real caller overwrites. `pause_margin` did NOT earn the same
treatment: 1.0 wins both 35B runs outright, and on the 27B at p_min 0.5 the margin is a
wash — 1.0 and 1.2 within 0.1 tok/s in both runs, and the two runs' nominal winners
disagree while spanning ~0.5 tok/s — because a controller that never pauses is
insensitive to its pause threshold. Note this is the FIRST time `pause_margin` was
actually swept; P9 validated 1.0 only against 0.0. Two tests pin the split
(`hub::tests::the_drafting_floor_is_per_checkpoint`,
`serve::config::tests::draft_p_min_defaults_per_checkpoint`), and the sweep script's
`SHIPPED_P_MIN` table must be edited alongside `hub.rs` or the next sweep grades against
a status quo that no longer ships (2026-08-08).

**The pause machinery is not free even in a regime where it never pauses, and the two
prompt kinds pay for it differently.** The `m=0` never-pause arm is a permanent
diagnostic in stage 2 for this reason. On the 27B at p_min 0.5 it is simultaneously the
fastest code cell in either sweep (medians 39.7 / 40.5 tok/s, ahead of the shipped 1.0's
37.9 / 38.2) and the slowest drafted chat cell (34.3 / 35.2 against 37.1 / 37.4). The
mechanism is the forced plain round every 32 that margin > 0 schedules to keep the
controller's cost EMA fresh: removing those rounds changes the drafter's round alignment
enough to move chat acceptance from 78% to 73.5%. So m=0 is not a candidate value — it
trades a prompt kind against another rather than being uniformly better — but it stays
in the grid because the asymmetry it exposes is the only direct read on what the pause
apparatus costs when it is not pausing (2026-08-08).

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

**Qwen3.8-27B drafts with its own MTP head, chosen over DSpark, EAGLE3 and transferring
the 3.6 DFlash head.** The checkpoint ships no DFlash sidecar, so the real alternative was
plain decode, and four candidates were surveyed before one was built.

*Transferring the 3.6-27B DFlash head* was the cheapest and was MEASURED, because the two
configs are byte-identical and `--model-size 3.8-27b --draft <3.6 sidecar>` simply works.
It partially transfers and does not pay: acceptance 64-76% where the same head on its own
3.6 target proposes 81-92%, giving 0.99x/0.95x on the 3.8 (1.02x/0.86x with auto-pause
disabled) against the native pair's 1.33-1.65x in the same session. The controller
correctly paused 72-89% of rounds. A head that proposes well below its native rate does
not clear its own overhead; that experiment set the bar MTP had to beat rather than
providing an interim default (docs/log.md 2026-08-15, Phase 0).

*DSpark* has exactly one head for this target, `RadixArk/Qwen3.8-27B-DSpark` — third
party, published for SGLang, and acceptance-tuned against `Qwen/Qwen3.8-27B-FP8` rather
than the Q4_K_M GGUF served here. (The draft's own weights are BF16; an earlier
telling of this decision said "FP8-trained draft", which is wrong — the FP8 is the target
it was aligned against. The distribution-mismatch argument survives the correction, the
precision one does not.) A third-party GGUF conversion exists but nothing first-party
does. *EAGLE3* is ruled out on availability alone: no EAGLE3 checkpoint has been
published for Qwen3.8-27B, and the pinned clone's supported list tops out at Qwen3-32B
(reference/llama.cpp/docs/speculative.md:35-51). No EAGLE3-on-Metal speedup figure is
cited here: the "1.05x on Apple Silicon" number this decision was briefed with traces to
an mlx-lm prototype discussion measuring a 4-bit Llama-3.1-8B on an M3 Ultra, whose
author states it was LLM-produced and not independently verified. It is neither
llama.cpp, nor this model, nor this machine, so it grades nothing.

*MTP* won on being first-party in the blessed repo — `ggml-org/Qwen3.8-27B-GGUF` ships
`mtp-Qwen3.8-27B-Q8_0.gguf` (3.16 GB, 18 tensors) beside the target — and on a step cost
that made the arithmetic work before any of it was built: an MTP step measured 7.1-8.5%
of a target decode forward across two runs, bracketing the 8.19% the byte budget predicts
(451.3 MB of Q8_0 head weights plus the target's 1042.9 MB Q6_K lm_head against ~18.25 GB
for a target forward). Under 10% is the band where depth 2-3 pays. Counter-evidence
considered and NOT accepted: llama.cpp issue #23752 reports MTP as a net throughput loss
at every configuration on Metal (M1 Max, Qwen3.5-9B, -11% to -24%), attributed to
per-step dispatch overhead. It is one unconfirmed report on other hardware and another
checkpoint, and the mechanism it blames is the one xwen's on-GPU chain and fused verify
exist to avoid; this repo's own measurement on this machine is the opposite sign. Worth
re-reading if a future revision regresses (2026-08-15).

**The MTP head's `h` input is the target's POST-final-norm hidden, not a pre-norm layer
output.** The trunk's `output_norm` runs before the hidden is handed to `hnorm`, which
makes the MTP tap a different tensor from every DFlash spec tap — those are pre-norm layer
outputs, and reusing one here produces a head that runs and drafts noise. This follows
llama.cpp's `graph_mtp` and upstream commit 166fe294, which made the choice deliberately;
`XwenModel` therefore grows an accessor for this tensor rather than reusing a tap
(2026-08-15).

**The draft chain stays on the GPU; only its final result is read back.** A per-step CPU
readback measured +1.45 to +2.96 ms of pure synchronization over the same op batched
(1.3-1.7x), against a step that is itself only ~2-9 ms — so on a 3-step chain a
read-per-step pattern spends most of what drafting saves, and it is clock-independent
overhead that a faster machine does not shrink. Each step's argmax and probability are
therefore reduced on device, the next step's embedding is gathered BY DEVICE INDEX, the
hidden is carried forward as a tensor, and one readback ends the chain. The accepted
consequence: the `p_min` walk runs host-side afterwards, so a chain that will be cut at
step 1 has already paid for steps 2 and 3. At depth 3 that is the cheaper side of the
trade; at a much larger depth it would not be, which is a thing to re-measure if the
depth default ever grows (2026-08-15).

**`draft_p_min` is a FULL-VOCAB probability in xwen and deliberately not llama.cpp's
top-10-renormalized one.** llama.cpp's MTP path builds a draft sampler with a hardcoded
`top_k = 10` and compares the argmax's probability AFTER renormalizing over those ten
survivors (common/speculative.cpp:1314-1336, :1589-1609). xwen compares against the full
softmax. The same numeric threshold is therefore a strictly stricter gate here than there
— renormalizing over a truncated set can only raise the top probability — and the two are
not interchangeable. This is not a defect to fix: a full-vocab probability is the quantity
that actually means "how sure is the drafter", and truncating first makes the floor depend
on a `top_k` nobody chose. It is recorded because EVERY fitted `draft_p_min` in this repo
rests on the definition, and because any future cross-check against llama.cpp must run
BOTH sides at `p_min` 0 or compare gates that are not the same gate (2026-08-15).

**A failed drafter reset or import leaves the head untouched, allocating before it
clears.** `MtpDrafter::reset` builds the zero carry tensor — a device allocation, and so a
fallible one — BEFORE clearing the cache and the committed count. Clearing first would, on
an allocation failure, leave a head reporting zero committed positions while still holding
the previous conversation's carry hidden, and the next row 0 would be built from a hidden
belonging to somebody else's text: a silently poisoned draft context rather than a visible
error. Failing with everything untouched is the only post-state a caller can reason about.
The same rule governs `import_cache`, which validates kind, position and layer count
before it believes any of the image's bytes (2026-08-15).

**A stored MTP cache image is usable at EXACTLY the position it ends at, where a DFlash
image backs any resume at or below its own.** The head's row at `p` is built from the
target's hidden at `p - 1`, and an image carries exactly one such hidden — the one for its
final position. So a partial cover is not a shorter-but-valid prefix, it is an image whose
carry belongs to the wrong position, and `drafter_planes_usable` refuses it rather than
resuming a head that cannot take another token. A DFlash image has no such constraint
because each of its rows is a function of that position's taps alone. The cost is
speculation for that conversation, not the conversation — the regime `Engine::rejects_image`
already treats as acceptable — but it arises far more often for this kind, and it is the
disk-tier face of the live rewind limitation (TODO.md). `an_mtp_image_backs_only_the_position_it_ends_at`
and `drafter_planes_are_usable_only_when_they_reach_the_resume_point` pin both halves
(2026-08-15).

**Chain depth is a per-DRAFTER-KIND default, and on the MTP head it is the knob that
matters — not the confidence floor.** `Model::draft_max_default()` returns 15 for a
DFlash block drafter, which proposes its whole block in one forward and for which 15 is
the structural ceiling rather than a fitted value, and 4 for the MTP head, which pays a
forward per step. 4 was fitted here (Stage C, 2026-08-15) and is not llama.cpp's 3. A 3x3
p_min-by-depth sweep, 128 greedy tokens, interleaved, medians of 3, had all nine arms
qualifying and depth-4 ahead of depth-3 at every floor, driven almost entirely by the
chat fixture (+36.7 to +39.2% over plain against +27.5 to +32.9%) while code was a wash.
The optimum is bracketed rather than sitting on the grid edge: a follow-up probe at
p_min 0.7 read 34.9 / 34.0 / 32.6 / 25.4 tok/s mean-of-medians at depths 4 / 5 / 6 / 8.
Depth 8 is where the auto-pause controller starts firing in earnest (34-80 rounds paused)
and drafting stops paying at all, which is the controller doing its job.

The floor was fitted in the same sweep to 0.7 and is **held far more weakly**, which the
record states rather than letting a bare number imply otherwise: at fixed depth 4 the
three floors spanned 33.2-33.8 mean-of-medians (1.8%), where depth spanned 12%. What the
floor unambiguously changes is wasted work — acceptance at depth 4 is 65.5% at 0.3
against 80.0% at 0.7 — which costs nothing measurable at batch 1 here because the target
forward dominates, and would matter wherever the drafter competes for the same silicon.
Sweeping the two together rather than in sequence is why this is visible at all: fitting
a floor at the shipped depth and then a depth at the fitted floor would have found each
against the other's stale value (2026-08-15).

**The auto-pause controller costs 3-6% on a checkpoint it never pauses, and the shared
`pause_margin` was NOT changed on that evidence.** Stage C's margin sweep on the 3.8-27B
made the never-pause arm the winner: `margin 0` read 35.9 tok/s mean-of-medians against
34.8 at the shipped 1.0, with `margin 0.8` collapsing to 28.8 (it pauses 32-87 rounds).
Pausing cannot explain the top of that: BOTH the 0 and 1.0 arms recorded ZERO paused
rounds. The mechanism is the controller's instrumentation, not its decisions —
`PauseController` forces a plain round every `FORCE_PLAIN_EVERY` (32, and every 4 until
its plain warm-up is met) to keep `ema_plain_ms` from going stale, and a forced-plain
round commits one token where a drafting round commits about four. In a 128-token run of
~40 rounds that is roughly three rounds' worth of speedup given up, which is the size of
the gap observed.

It was not installed, for a reason that is about the SHAPE of the constant rather than
the size of the win: `pause_margin` is one shared value at three sites, only one
checkpoint's stage 2 was run, and decisions.md already records the controller earning its
keep on the 3.6 pair. Installing 0 on one checkpoint's evidence would silently change the
other two to a value this sweep never graded for them — exactly the conflict the retune
script warns about — and would remove the safety net that the depth-8 arm proves still
works. The finding is real and is ledgered as an optimization (make the plain-baseline
cadence adaptive, or recover the baseline from the verify forward instead of spending a
round on it) rather than as a default change (2026-08-15).

**The MTP graph is confirmed end-to-end against llama.cpp, by identical text rather than
by similar acceptance.** Both implementations were run on the same raw fixture with the
same target and sidecar at depth 3, `p_min` 0 on both sides, greedy: acceptance came out
73.3% against 75.0% (code) and 45.7% against 47.1% (chat), and the 128-token
continuations were BYTE-IDENTICAL on both fixtures. The identical text is the load-bearing
half — it means acceptance is being compared over the very same continuation rather than
over two texts that merely resemble each other, and it independently exercises the trunk,
since two unrelated implementations agreed on every greedy argmax for 128 tokens. The
residual 1-2 points is xwen proposing slightly more drafts near the token budget's end
(120 against 116, 162 against 157), which is round bookkeeping.

Two harness traps this cost, both worth knowing before anyone repeats it. `llama-cli` in
this revision embeds llama-server and runs CONVERSATION mode regardless of `-no-cnv`,
silently applying the chat template and enabling thinking; a first attempt compared
xwen's raw continuation against llama.cpp's chain-of-thought and produced a spurious
11.5-point chat gap that looked like a graph bug. Drive the comparison through
`llama-server`'s `/completion` endpoint, which takes the prompt verbatim, and read
`timings.draft_n` / `draft_n_accepted`. And both sides MUST run at `p_min` 0, because the
two `p_min` definitions differ (see above) (2026-08-15).

## Serving

**The serve/ tree (Anthropic + OpenAI + native dialects, TUI, queue, prefix cache, disk
tier) is inherited as-is**; it is architecture-agnostic. The KV export/import and disk
tier must additionally carry the recurrent state for the 3-of-4 linear layers — KV cache
alone no longer reconstructs a prefix. Native endpoint moved `/maxuna/v1/*` →
`/xwen/v1/*` (2026-07-28).

**One server serves both checkpoints; `--model` is only the default (2026-08-11).**
Every job (generation or batch) names the checkpoint it needs, and the engine's pickup
compares that against what is resident: a mismatch images the live conversation out
through the same path an idle unload takes, drops the state, and lazy-loads the named
checkpoint — one model resident at a time, by construction, which is this machine's
memory invariant, not a scheduling choice. The alternatives were refused deliberately:
keeping both models loaded risks GPU OOM (CLAUDE.md operational hazards), and a
temp-generator-per-foreign-batch would reload on every consecutive same-model batch
where the swap design keeps the second one warm. Which checkpoint the default GGUF is
comes from its `general.architecture` (`Arch::model()`), never from `--model-size` —
the flag and a `-m` path can disagree, the file cannot. The same rule now covers the
startup drafter resolution (review fix, same day): `run_serve` reads the served GGUF's
architecture before resolving the official sidecar, so a config-file `model` (or `-m`)
that disagrees with `--model-size` gets the sidecar for the model actually served
rather than a geometry error blaming the drafter. The one-shot CLI commands keep the
flag's old double duty deliberately — there the flag and the payload are the intent.
Selection strictness followed
each surface's existing character: the batch route 400s an unknown model (the field
exists to select), the compat dialects fall back to the default (SDKs send their own
model ids and must keep working), the native generate endpoint stays modelless.
SUPERSEDED in two places on 2026-08-14 — the architecture stopped identifying a
checkpoint (see "The GGUF names itself" below), and the compat dialects stopped falling
back (see "On the wire a checkpoint has exactly one name").

**On the wire a checkpoint has exactly one name — its full name — and an unknown one
is a 400 on every surface (2026-08-14).** Two bugs shared a root: the model vocabulary
was the CLI's. `/v1/models` listed the served file's basename AND every checkpoint's
short alias, so one model appeared under two ids (a live server listed
`Qwen3.6-35B-A3B-Q4_K_M`, `27b` and `35b` for two checkpoints), which is not a listing
a client can pick from; and the compat dialects' fall-back-to-default meant an SDK's
own id (`gpt-4o`) was answered by whatever checkpoint the server happened to default
to, indistinguishably from a correct request. Both are fixed by naming: the APIs accept
and echo `Model::full_name()` only — the string ggml-org names the repo with and the
GGUF carries as `general.name`, quant-independent — while `--model-size` keeps the
short aliases (and now also accepts full names, so a `/v1/models` id pastes into the
CLI). Absent or empty `model` still means the served checkpoint; anything unrecognized
is a 400 in the dialect's own error format, listing the valid names. The one real cost
is deliberate: a client that used to get an answer for `"model": "35b"` now gets a 400
telling it what to send, which is the SDK-default surprise turned into a message
instead of a silently wrong model. Quant is not part of the name — one server serves
one file per checkpoint, and the response's job is to say which MODEL answered.

**The GGUF names itself; the architecture is only a fallback (2026-08-14).** Adding
Qwen3.8-27B broke `Arch::model()`'s one-to-one claim: two releases now ship the dense
`qwen35` graph with byte-identical configs, so the architecture can no longer say which
checkpoint a file is. The identification chain is now explicit `--model-size` (the
operator naming a custom file), then `general.name` (both blessed files carry their
exact full name; a substring pass catches a re-quantized conversion), then the file
name, then `Arch::model()` as the last resort with a logged warning. `Arch::model()`
survives as exactly that fallback and its doc says so. It matters because the identity
picks the hub repo and the sidecar: guessing 3.6 for a 3.8 file would attach the wrong
drafter to a graph that accepts it, costing acceptance rather than failing.
SUPERSEDED in its ordering the same day (review round): `--model-size` is NOT the first
link and does not override the file — see "`--model-size` is a tie-break, not an
override" below. The rest of this paragraph stands.

`Model::identify` uses the architecture only to NARROW the candidates, never to answer —
including for the MoE graph, which one official checkpoint ships. Answering from the
arch there would have been safe for the engine and wrong for the API: any conversion
onto that graph would then be reported by `/v1/models` and every response under
`Qwen3.6-35B-A3B`, which is a claim about weights nobody checked. Unidentified files
therefore keep reporting under their own file name (unchanged behavior) while the engine
still runs them as the architecture's checkpoint — the identity and the id are two
questions, and only the first has a safe default.

Both name sources are read by ONE rule (review round, same day): an exact full name, or a
whole full name found inside the name, case-insensitively — which accepts the shapes real
files take (`Qwen3.8-27B-Q8_0`, `Qwen3.6-27B-Instruct`) while requiring that a complete
checkpoint name actually appear. `general.name` is consulted before the file name because
it is what the converter wrote INTO the file, but it earns no looser matching for it.
Matching a bare release series was tried and refused twice over: it identifies
`My-Qwen3.6-14B-finetune.gguf` as the official 27B, and — since `Arch::Moe` has exactly
one candidate, so no ambiguity check can save it — `MyMoE-3.6` as Qwen3.6-35B-A3B. Either
one answers an official name with weights nobody checked, which is the single thing this
function exists to prevent. A name matching more than one checkpoint identifies as none
rather than as whichever `MODELS` lists first. The blessed files are unaffected: their
`general.name` is the exact full name, verified on all three.

**`--model-size` is a tie-break, not an override (2026-08-14, review round).** It names
the checkpoint a file that says nothing about itself holds. Against a file that DOES say,
a contradicting flag is a startup error naming both sides, and it must agree with the
architecture too. It had silently won, which meant the server started clean and then
failed `EngineState::load`'s own arch/identity checks on every request — a 500 per
request for a mistake that was fully knowable at startup. Those load-time checks remain
as a backstop for the case they can still catch: a file replaced under a running server.

**A job names a FILE, not just a checkpoint (2026-08-14, review round).** `Target` is a
checkpoint plus "is this the served file". The distinction only exists on a server whose
GGUF identifies as none of the official checkpoints, and there it is the whole ballgame:
the official checkpoint of the same architecture is a DIFFERENT FILE that happens to size
its caches identically, so a bare `Model` could not say which was meant. With it: the
served file answers for its own advertised id (which the resolver now accepts — it was
400ing the one id `/v1/models` published), an official name resolves that checkpoint's
real hub file and swaps to it like any other checkpoint, and the batch document is
labeled with the id that answered rather than with the arch fallback's full name. Two
things fall out for free: `Target` equality is the engine's swap check (two files, two
targets), and the disk tier's binding is a `Target` comparison rather than a checkpoint
one — the tier is bound to `settings.model`, which is a file.

**The disk tier stays bound to the default checkpoint.** `DiskCache::open` scans and
binds at startup against `settings.model`, and `verify()` deliberately disables the
tier for the rest of the process if weights with a different checkpoint id load. Under
a non-default checkpoint every engine call site is therefore handed `None` instead of
the tier: feeding it foreign images would poison the store with bytes that claim the
default binding, and verifying it against foreign weights would permanently disable it.
The segment layout is already per-checkpoint directories, so a tier-per-checkpoint is a
straightforward lift when a workload wants it (TODO.md, 2026-08-11 arc).

**Speculation is decided per checkpoint, not per process (2026-08-14, review round).**
`ServeSettings.draft` was one resolved `Option<PathBuf>`, which was correct while
"official sidecar" named one file and became silently wrong when a checkpoint shipping
none arrived: a server whose DEFAULT checkpoint was Qwen3.8-27B ran every OTHER
checkpoint plain too, costing the 27B its measured +46 to +52% with nothing in the log
about it. It is now a `DraftMode` — `Off`, `Official`, or `Custom(path)` — and
`checkpoint_paths` resolves `Official` when a checkpoint loads, so each drafts with its
own sidecar and a sidecar-less one decodes plain with its own line. A `Custom` path still
belongs to the checkpoint it was validated against and never transfers (2026-08-11,
unchanged); any other checkpoint falls back to its official sidecar rather than borrowing
it. Startup validation follows the same split: a custom drafter is judged at startup as
before, an official sidecar when the checkpoint that owns it attaches it, and the served
checkpoint's sidecar is still PREFETCHED at startup so a first request does not stall
behind a 3.5 GB download. The dashboard's drafting cell tracks the loaded checkpoint
(`ModelLoaded` clears it, `DrafterLoaded` sets it) instead of the setting, for the same
reason the setting stopped being the answer.

**`draft.p_min` resolves at drafter attach, not in the config merge (2026-08-11).**
The merge used to bake `--model-size`'s per-checkpoint default into the settings, which
was correct while one process meant one checkpoint and silently wrong the moment it
did not — the fitted floor is a property of the loaded checkpoint. Unset now stays
`None` through the merge and each load applies `Model::draft_p_min_default()`; an
explicit value pins one floor for every checkpoint served, which is exactly what the
CLI flag means. A non-default checkpoint always speculates with its official sidecar —
a custom `draft.path` belongs to the default checkpoint alone, since sidecars never
transfer between checkpoints.

**`/xwen/v1/batch` is the CLI's document over HTTP, run as one queue job
(2026-08-11).** Same request JSON as `xwen batch` stdin, same response document as its
stdout — one surface, two transports; the core was written transport-agnostic for
exactly this. The job rides the ordinary `JobQueue` (so it serializes with chat
requests, honors shutdown, and can be swept when its client leaves) as a second `Job`
variant whose single terminal event carries the whole response. Scheduling and the
watchdog deadline use a bytes/3 token overestimate, which errs the right way twice:
small chat requests keep jumping ahead of a batch still in the queue (once picked, a
batch runs to completion — the batch=1 engine preempts nothing), and the deadline errs
loose. The
runner's stderr `eprintln!`s became a `BatchHooks` progress callback — the CLI prints
the same lines as before, the server routes them into its log — and the same hooks
carry cancellation: client-gone/deadline/shutdown fold into the job's cancel token,
polled between items and per decoded token, and items the cancellation reached report
it in their own `error` field so a deadline still yields a truthful partial document.
The batch marks the engine dirty up front and the live conversation is paged out before
it runs: the runner owns the whole cache, and the existing post-job reset machinery is
what puts the cache back.

**The request-body cap is an explicit 100 MB, replacing axum's implicit 2 MB
(2026-08-11).** The implicit cap was never a decision — nobody chose 2 MB, axum's
default arrived with the framework — and it bound first in practice: a real client
split one batch over a 377 KB story into 14 POSTs to fit under it, re-prefilling the
shared prefix each time. The wire is the wrong layer to police cost: the queue's
bytes/3 token estimates and max_ctx judge what a request actually costs, and both keep
doing so at any body size. 100 MB is far past any request the engine can serve while
still bounding a hostile stream; it covers every dialect on the API router (`/health`
carries no body). Two accepted edges, both recorded in the ledger: `Router::layer`
wraps only the routes registered before it, so a POST route added after that line
silently gets axum's 2 MB default back — the layer call carries the warning; and
bodies are buffered and parsed BEFORE the queue can answer 429, with no concurrency
bound on connections, which is acceptable exactly because the default bind is loopback
on a single-user machine (2026-08-11).

**One `reasoning_effort` field drives both the think budget and the 3.8 template
preamble, with off-scale levels nearest-mapped instead of raised (2026-08-19).** The
OpenAI dialect's field kept its budget mapping (none=off, minimal=1024, low=4096,
medium=16384, high/xhigh/max=uncapped — the budget scale is this server's own) and now
also selects the template level a 3.8 prompt renders: minimal→low, high/max→xhigh. One
knob rather than two because a client saying "low effort" means low effort, not "cap
the tokens but instruct the model to think hard" — the split-knob reading was the
"conflicting field" the 2026-08-14 ledger item worried about, and it dissolves once the
field drives both. The nearest-mapping is a deliberate divergence from llama.cpp, which
passes the raw string into the jinja and lets the template raise: `minimal`, `high` and
`max` are real levels of this API's vocabulary that the template happens not to define,
and answering them with the nearest defined level serves the request where upstream
turns it into a template error. The level is dropped whenever thinking resolves off, because the template
renders it only inside the thinking guard. Clients that want the raw template parameter
have it: `chat_template_kwargs.reasoning_effort` takes exactly the template's three
levels, and the top-level field wins over the kwarg — llama.cpp's precedence, kept so a
client speaking both shapes gets upstream's answer. The server-wide default
(`[thinking] effort` / serve `--reasoning-effort`) fills in when a request names
nothing, and `count_tokens` renders under the same default so a count matches the
generation it predicts. The native dialect exposes the raw parameter directly
(`reasoning_effort`, three levels only) plus `preserve_thinking`; the Anthropic dialect
exposes no per-request effort knob — its API has no natural field for it, so the
server-wide default applies (TODO.md).

**`chat_template_kwargs` is validated strictly — the one exception to the compat
dialects' accept-and-drop permissiveness (2026-08-19).** The OpenAI dialect accepts and
drops sampling parameters this engine has no equivalent for (penalties, logprobs,
`min_p`) because a client sending them still gets a correct completion, merely sampled
without them. Template kwargs are a different category: they steer the rendered prompt
itself, so a kwarg silently ignored means the client got a DIFFERENT PROMPT than it
believes it asked for, with nothing anywhere saying so. An unknown key, a wrong type,
or a `reasoning_effort` outside the template's three levels is therefore a 400 naming
the offender (the error for an off-scale level points at the top-level field, where the
wider none/minimal/high/max vocabulary belongs). The accepted keys are the three
parameters the vendored templates actually take — `enable_thinking`,
`preserve_thinking`, `reasoning_effort` — the official Qwen card's (and vLLM's) request
shape.

Extended 2026-08-19 (the arc's review pass): a request-level TEMPLATE effort on a 3.6
target is a 400 by the same argument. `chat_template_kwargs.reasoning_effort` (and the
native dialect's `reasoning_effort` field — the same raw parameter) name a parameter
only the 3.8 template defines; on a 3.6 checkpoint they would render nothing, which is
precisely the silently-different-prompt outcome strict validation exists to prevent,
and it contradicted the CLI, where `--reasoning-effort` on a 3.6 checkpoint had been a
startup error since the arc landed. llama.cpp would silently ignore an unused kwarg —
a deliberate divergence, consistent with this repo's cross-check-instead-of-shrug flag
policy. Both `prepare`s take the resolved `Target` and the error names the model. The
boundaries, each deliberate: the OpenAI TOP-LEVEL `reasoning_effort` field stays
accepted on 3.6 (it carries budget semantics on every checkpoint, and the error points
clients at it); kwargs `enable_thinking`/`preserve_thinking` stay accepted on 3.6
(real parameters of that template); and the server-wide `[thinking] effort` default
stays inert-but-legal on 3.6 — it is an operator setting covering whatever checkpoints
a server serves, not a request asking this model for a level.

Extended again 2026-08-19 (later still): the batch surface applies the same refusal
with the module's own failure shape. A `reasoning_effort` on a batch item or on the
request's `defaults` against a 3.6 checkpoint fails PER ITEM (message naming the
checkpoint and the 3.8-template provenance), not as a request-level error — batch
validation failures land on the item so the other N-1 keep their prefill, and a
defaults-level effort fails every item identically because the defaults reach the
renderer only through the items. Effort with thinking off (batch's default) stays
accepted and inert on 3.8, as everywhere else.

**Normalization passes assistant reasoning through in native tools mode; retention is
decided once, in the renderer, per dialect (2026-08-19, the arc's review pass).** Both
compat normalizers used to strip `reasoning` from every assistant turn before the
trailing assistant/tool run. That rule predates the dialect arc, when the renderer
dropped exactly those turns anyway (`preserve_thinking || index > last_query`, with
preserve always false), so the early strip was invisible. The 3.8 template made it a
bug: its `preserve_thinking` default is TRUE — the 3.8 card recommends preserved
thinking for agent workloads — so the dialect asked the renderer to keep reasoning the
normalizers had already destroyed, the OpenAI kwarg `preserve_thinking: true` was
defeated on every checkpoint, and the three dialects disagreed (native replayed
everything, the compat APIs didn't). Now native tools mode passes every turn's
reasoning through and the renderer's dialect rule is the single owner; the
`trailing_run_start` predicates are gone. The debug tools modes keep dropping
reasoning everywhere — they render the history as if tools had never existed, which is
their documented point. One nuance recorded so nobody "fixes" it back: Anthropic's
real Messages API strips non-trailing thinking blocks, and this dialect deliberately
does NOT emulate that — it is a wire-compatibility layer over Qwen checkpoints, and
what renders must follow the checkpoint's template, not the API vendor's serving
policy. A 3.6 request still renders without superseded reasoning, but because the 3.6
template says so, not because the API layer pre-judged it.

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

## Batch

**`xwen batch` snapshots at the EXACT longest common token, not at a chunk-64
boundary.** The chunk-boundary plan was inherited from Marconi, whose constraint is
real for it: a chunked prefill kernel cannot materialize the recurrent state mid-chunk,
so the only cheap snapshot positions are chunk multiples, and the prefix is truncated
down to one. xwen has no such constraint — the generation layer materializes full state
at whatever position a prefill stops, which is what serve's `PrefixCache` has always
relied on to snapshot at arbitrary stops. Rounding down would therefore have meant
building alignment logic to give up reuse that costs nothing to keep. Notable because
the argument that first recommended the boundary — keep it simple — is the argument
that refuted it (2026-08-09).

**Grouping is single-level, the cache is RAM-only, and it dies with the request.** One
batch computes one longest common prefix across all items and that is the whole cache
hierarchy: no prefix tree, no disk tier, no TTL, nothing that outlives the process. The
run is one command with all its items in hand, so a request-scoped snapshot needs no
eviction policy, no cross-request invalidation and no checkpoint binding — the three
places the serve-side prefix cache spends its complexity. Multi-level trees and a
pinned cross-batch system-prompt snapshot are ledgered, not built (2026-08-09).

**Three floors keep the snapshot honest, and one of them is a correctness precondition
rather than a heuristic.** `MIN_SHARED_PREFIX` = 64: below that the snapshot/restore
bookkeeping costs more than re-prefilling the tokens it saves, so every item runs cold
instead. A single-item batch never snapshots — there is nothing to share it with. And
the shared span is capped at `min_len − 1`, so the longest item's prefix can never be
the whole of the shortest item: an item with no tail of its own would decode from the
snapshot position itself and write into positions the snapshot covers, which
`XwenModel::restore_cache_snapshot` explicitly forbids (model.rs:716 — full-attention
layers are restored by truncation, valid only while the slots below the position still
hold the same keys). The cap is not a margin; it is the restore precondition expressed
as arithmetic (2026-08-09).

**The drafter rides along by truncation, not by a `DrafterImage` export.** Every item
shares every token below the snapshot position, so restoring the target's cache and
calling `sync_drafter_to(pos)` leaves the drafter holding exactly the rows it would
have built by prefilling that prefix itself. Exporting and re-importing a drafter image
per item would reconstruct the same rows at the price of a copy per item. Truncation is
correct here only because of the single-level grouping — a prefix tree with divergent
branches would need the image (2026-08-09).

**Batch sampling defaults to greedy and thinking defaults to OFF, and both are
deliberate divergences from the chat surface.** A batch request is a
classification/extraction surface: two runs of the same payload should agree token for
token, and a caller comparing scores across items needs the sampler out of the way, so
`BATCH_SAMPLING` is temperature 0 with the chat defaults parked on the inert
`top_k`/`top_p`/`seed` fields for a request that raises the temperature alone. Thinking
is off for a blunter reason, observed directly during bring-up: items carry tight token
budgets, and an unrequested reasoning block eats a 64-token budget before the answer
starts, so the item finishes on `length` with no JSON at all. Either default is one
field away for a request that wants it (2026-08-09).

**A scored field's value is ASSEMBLED by the engine, not decoded under a grammar mask
with the choice points detected after the fact.** The alternative was to keep the
llguidance path and read confidence off the mask at the positions where the grammar
offered a choice, which needs the grammar to say where those positions are. llguidance
1.7.6 exposes no structural-position API: captures do not exist for grammars compiled
from JSON schema, and capture semantics are on-completion anyway — the wrong shape for
a per-step reading. Assembly inverts the problem instead of working around it. The
runner teacher-forces the JSON skeleton it wrote itself, so it knows exactly where each
value begins, and selection is exact rather than inferred: every allowed option is
scored in full and the answer is chosen from the scores. It needs no llguidance change,
and the response is the same document the grammar path would have produced. What it
costs is one forward per option token instead of one per answer token. v1 scope is a
flat all-required object of enum/boolean fields; `include_score` PRESENCE routes the
item, so `false` is an annotation like any other and differs from absent — a schema
cannot silently switch engines by toggling a boolean (2026-08-09).

**An option's score includes the one delimiter token that follows it, which is what
makes the option set prefix-free.** Without it a strict-prefix option can never lose:
score is a sum of log-probabilities, every added token contributes something negative,
so `score("low_priority") = score("low") + (negative)` and `"low"` wins every time
regardless of what the model believes. Scoring the terminator — the closing quote for a
string enum, the delimiter after a bare literal — makes the two option sequences
diverge at a token they cannot share, and the comparison becomes a real one. Found by
the outside-model review pass (codex, `gpt-5.6-sol`) and confirmed live: on the
adversarial pair `["low", "low_priority"]` the long option now takes 0.9988 when the
prompt asks for it (2026-08-09).

**Seam canonicality is CHECKED at plan time, not assumed.** The assembled document is
scored as `encode(segment) ++ encode(option) ++ encode(segment)`, which is only the
document's own tokenization when the tokenizer agrees — BPE is free to merge across a
seam and produce a different sequence for the same bytes. `check_seams` re-encodes every
(segment, option) and (option, segment) pair and refuses the schema when the merged
encoding differs from the concatenation. In practice this rejects values that fuse with
their delimiter (trailing punctuation or whitespace) and values needing JSON escapes,
where the escape sequence rather than the label would be what gets scored. Refusal was
chosen over scoring-the-escape-anyway: an option whose score is not the score of the
value the caller named is worse than an error message. Lifting the limits is ledgered
(2026-08-09).

**`escape` was opener-level mass, confounded by formatting for bare literals —
SUPERSEDED 2026-08-11 by the whole-row classification below.** As shipped 2026-08-09 it
reported the probability on tokens that open no option at the first choice-point token.
For a quoted enum the forced opening quote filtered formatting out; for a boolean, whose
choice point sits after `:`, whitespace tokens a pretty-printer would emit competed with
`true`/`false` and the escape read near 1 beside a near-certain answer score. Kept then
with the refinement ledgered; the first external client hit it (every multi-field
item's first field at 0.999-1.000, pinning mean escape at 1/fieldCount) and the
refinement shipped.

**`escape` is a whole-row classification by token TEXT, with formatting factored out —
because the mass the opener set missed was mostly the ANSWER in the model's preferred
spelling.** The client's one-token-early hypothesis was checked and refuted first: a
row dump at the exact read shows the first boolean slot holding 54.9% ` true` / 44.9%
` false` — single space-led tokens, the model wanting `{"k": true` against the compact
skeleton — with bare `true`/`false` at ~5e-5, and the style pinning to compact (bare
`false` 99.8%) from the second field on, which is why only first fields read ≈1.
`escape_mass` now classifies every encodable id by its raw BYTES: stripped of leading
JSON whitespace at an unquoted field (verbatim at a quoted one, where leading
whitespace is string content), a nonempty prefix of some option's bytes is INSIDE;
nothing but JSON whitespace at an unquoted field is FORMATTING, excluded from both
sides; escape = outside / (inside + outside). Bytes, not decoded text, because
byte-level BPE cuts multi-byte characters across tokens and `decode` of such an id is
lossy (U+FFFD) — a text-level match would misread the canonical opener of any
non-ASCII option as escape (found by the second-model review, codex `gpt-5.6-sol`;
`LagunaTokenizer::decoded_vocab` reverses the byte-level alphabet instead, pinned
against `encode` by test). JSON whitespace means the four bytes JSON allows — an NBSP
would invalidate the document and honestly counts outside (same review). Prefix
matching is one-way — `yesterday` is not headed for `yes`, so a token that begins with
an option and carries on counts outside; the canonical tokenizations carrying real
mass never fuse across the value's edge (check_seams refuses the plans where they
would) — and canonical-spelling-only for quoted fields (`\/` and `\uXXXX` alternates
count outside, the check_seams stance; negligible mass). Measured: first-field escape
0.9999 → 0.00197, and the residue is genuine vocabulary-gap signal (mostly ` "`, a
string where a boolean belongs); scores are untouched by construction (they never read
the classification) and verified bit-identical. Costs one ~248k-entry vocab walk
cached per tokenizer plus one full-row softmax readback per field
(`Generator::last_probs`, normalized by the same code path as the scores) (2026-08-11).
The residual first-field elevation (absolute 0.010-0.035 without thinking, 0.042-0.109
with, against later fields at 4.3e-7 to 1.7e-4 — 35B measurements; the client's 27B
data tails higher, max 0.109 without thinking) was re-reported by the client post-fix and
confirmed by a position-vs-identity controlled row dump: it follows document position,
not field identity, and its mass is concentrated on plausible alternative openers —
mostly ` "` and the space-led capitalized booleans. It is conditioning signal, kept
as-is; consumers comparing escape across categories should aggregate with and without
first fields (2026-08-12, log.md same date).

**`shared_prefix` is a wire-size field, deliberately NOT a prefill feature.** The
runner has prefilled the items' shared TOKEN prefix once since batch shipped; what
repeated was the request body — a 377 KB story per item forced a real client into 14
POSTs under the old 2 MB cap. The field is prepended verbatim to every item's first
message before rendering, so the resulting prompts (and answers, and scores) are
byte-identical to spelling the document per item — pinned by test and verified live
over both transports. Alternatives refused: request-level shared MESSAGES would change
prompt structure (separate turns) and so change answers against the inline spelling;
placeholder interpolation buys nothing over prepending. An item with no messages
fails as an item, an empty string means absent (2026-08-11).

**Scores are not bit-stable between the cached and cold arms, and consumers must
compare with tolerance.** Replaying an item from the snapshot prefills its tail as a
short span, which routes the MoE through `mv_id` where the cold arm's single long
prefill takes `mm_id` — the same math at different precision. Measured across the demo
batch: chosen values identical on every item but one, whose two candidates sat at 0.502
and 0.493 and swapped, and scores differing in the third to fourth decimal throughout.
This is the partition-dependence class already recorded above ("Persistent state is
partition-dependent in its low bits"), reached through a new door, not a new phenomenon;
`XWEN_MM_ID_MIN_SEQ=1` forces both arms onto one kernel and makes them byte-identical,
which is how it was diagnosed. `XWEN_BATCH_NO_CACHE=1` is the A/B lever (2026-08-09).

## Tokenization, chat, tool calls

**The Qwen tokenizer.json (12,807,982 bytes, byte-identical between the two model
repos, sha256 5f9e4d49…) is vendored at reference/tokenizer.json and embedded via
include_bytes!, following laguna's embedded-tokenizer decision.** Qwen2 byte-level BPE,
NFC normalizer, no BOS ever prepended (`add_bos_token: false`, no post-processor). The
split regex differs from Qwen3 by `\p{M}` handling — do not reuse a Qwen3 regex
(2026-07-28).

**Qwen3.8's tokenizer.json differs from 3.6's by seven added tokens and nothing else,
and the embedded 3.6 file is what still ships (2026-08-14).** Compared structurally, not
by hash alone: `model.vocab` (248044 entries), `model.merges` (247587), the normalizer,
pre-tokenizer, post-processor and decoder are byte-identical; 3.8 adds
`<|audio_start|>`, `<|audio_end|>`, `<tts_pad>`, `<tts_text_bos>`, `<tts_text_eod>`,
`<tts_text_bos_single>`, `<|audio_pad|>` at ids 248070-248076, above every id the chat
path uses. Text therefore tokenizes identically under the embedded file, and client text
spelling one of those markers encodes as plain BPE — which is the safer behavior for
client content anyway. What is NOT decided here: whether a text-only checkpoint can emit
one of those ids at all, and what the embedded tokenizer would decode it to. Left as a
ledger item rather than improvised into a per-checkpoint tokenizer, since a second
12.8 MB embed for seven ids nothing renders is the kind of thing to decide deliberately.

**Qwen3.8 ships a different chat template; it is vendored beside 3.6's and the renderer
is unchanged — which means every default 3.8 conversation renders differently from the
official template, by one sentence (2026-08-14).** [SUPERSEDED 2026-08-19: the renderer
is now dialect-parameterized and the divergence is closed — see the next entry. The
template facts below all stand.] `reference/chat_template-qwen38.jinja`
(8952 bytes, verbatim from Qwen/Qwen3.8-27B). Diffed hunk by hunk against 3.6's: a
`reasoning_effort` system preamble, `preserve_thinking` defaulting to true instead of
false, the inline `<think>`-in-content parsing fallback removed, and an empty-arguments
guard on tool calls. The generation prompt — the block that decides what the model is
handed to continue — is byte-identical, and so is the `# Tools` prose, which is why one
hand-written renderer still serves every checkpoint.

The divergence is not hypothetical and is worth stating in full, because the defaults
make it universal rather than opt-in: with thinking ON (the default) and no
`reasoning_effort` given, 3.8's template resolves the effort to `xhigh` and prepends
"Reasoning effort is set to xhigh. Please think carefully through the task, validate key
assumptions, consider plausible alternatives, and prioritize correctness, consistency,
and clarity in the final answer." to the system block — creating one if the request has
no system message. xwen rendered neither that sentence nor the `low` variant, so every
default 3.8 conversation this server rendered was missing a system instruction the model
was trained to see. `medium` is the one effort level that injects nothing, so what xwen
rendered then was exactly the official `reasoning_effort="medium"` rendering. Accepted
knowingly for the arc that added the checkpoint (it is prompt semantics, not model math,
and the serve layer already had a conflicting `reasoning_effort` field of its own to
reconcile), but nobody should read "the generation prompt is byte-identical"
as "the prompts are the same".

Both vendored templates are cross-checked by chat.rs's tests (the fixed prose must
appear in each, and the generation-prompt block must match between them), so a future
release that moves either one fails a test rather than a reply.

**The renderer is parameterized by `ChatDialect`, and the 3.8 divergences above are
implemented behavior, not an accepted gap (2026-08-19).** `Model::chat_dialect()` maps
the 3.6 pair to `Qwen36` and the 3.8 to `Qwen38`; `ChatOptions::for_dialect` carries
each template's own defaults, and every prompt-building path (CLI gen/chat, all three
serve dialects, count_tokens, batch) reaches its options through it. The dialect was
kept a two-value enum on the options rather than a second renderer because the
templates' turn rendering and generation prompt are byte-identical — the differences
are confined to the system block and two defaults, and each is pinned by a test rather
than asserted in prose:

- The `reasoning_effort` preamble renders under `Qwen38` with thinking on: `xhigh`
  (the template's default) and `low` prepend their sentences — held as constants
  asserted verbatim, length included, against the vendored 3.8 template and asserted
  ABSENT from the 3.6 one — while `medium` injects nothing, making a medium render
  byte-equal to a 3.6 render of the same conversation. With no system message the
  dialect synthesizes a system block to carry the sentence (the template's own
  behavior); the block anchors the prefix cache like a client's and the preamble stays
  out of the client-content spans, since it is template prose, not client content.
  With tools it opens the block ahead of the `# Tools` header.
- `preserve_thinking` defaults true under `Qwen38` (template line 116's `is undefined
  or is true`), false under `Qwen36`.
- An empty system message emits no block under `Qwen38` where `Qwen36` emits the empty
  block its template unconditionally writes.
- `split_reasoning` — the inline `<think>`-in-content fallback — runs under `Qwen36`
  only. The 2026-08-14 record (and the ledger item it fed) claimed xwen "never
  implemented that fallback"; that was WRONG — chat.rs had it and ran it
  unconditionally, so a 3.8 turn replaying reasoning inside content was getting the
  3.6 reading. It is now dialect-gated, and a 3.8 turn renders such content verbatim,
  as its template does.

`TOKENIZATION_RULES_VERSION` went 2 → 3 with this: the same 3.8 conversation encodes
differently under the current rules, and a stale disk image must fail the stamp check
rather than longest-common-prefix-match a stream these rules would never produce.

**chat.rs is a hand-written Rust port of the official chat_template.jinja (7764 bytes,
byte-identical across both Qwen 3.6 repos), keeping laguna's content/structure separation** so
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

**The cards' recommended penalties are refused for now, not quietly half-done
(2026-08-19).** The official cards recommend `presence_penalty` 1.5 for instruct mode
on ALL three checkpoints, and for thinking mode on the 35B-A3B alone (the 27B and
3.8-27B thinking recommendations say 0.0) — HF README.md of Qwen/Qwen3.6-27B (~lines
633-639), Qwen/Qwen3.6-35B-A3B (~661-667), Qwen/Qwen3.8-27B (~250-255);
generation_config.json carries none of them, so the files and the cards disagree and
the cards are the fuller recipe. Not implemented, for a reason beyond "the sampler has
no penalty machinery": a penalty makes the target distribution HISTORY-DEPENDENT, and
the speculative verify path assumes it is not — `forward_all_logits` scores a whole
draft in one batched forward against per-position distributions that would each need
the penalty applied over a different history prefix, and `spec-equivalence`'s
greedy gate would have to hold under that per-position application on both the drafted
and plain arms. That is a real design (llama.cpp does it) but it is sampler + verify +
gate work as one unit, and shipping the penalty on the plain path alone would make
`--draft` and `--no-draft` sample from different distributions — the exact property the
equivalence gate exists to forbid. Until then the OpenAI dialect keeps accepting and
dropping `presence_penalty`/`repetition_penalty`/`min_p` (they degrade sampling, not
the prompt — see the kwargs entry under "Serving" for the line between the two), and
the mode-keyed defaults ship the cards' temp/top_p/top_k, which ARE mode-pure.
Ledgered with the values and sources (TODO.md).

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
EXTENDED 2026-08-08: **filtering on `argv[0]` is only correct if you first establish
which lines ARE records.** `pgrep -fl` prints one record per line as `<pid> <argv0>
<args…>`, but an argv that embeds newlines prints as extra lines carrying no pid — and
agent harnesses produce exactly that, wrapping commands as `zsh -c "cd <repo>\n<cmd>"`.
The continuation line `cd /Users/…/xwen` then has `xwen` as its second token, which
`execName` reads as argv0, and the guard aborted both parity gates over a model process
that did not exist. The fix is one predicate ahead of the matcher: a line is a record
only if it leads with a pid, and fragments are dropped because the record they belong to
is checked on its own first line anyway. Same lesson one level down from the original —
the matcher was right and the tokenization under it was not.

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

**Retuning the drafting controller is a scripted protocol now, `scripts/retune-draft.ts`,
and its load-bearing rule is that NO cell is reused between stages.** The constants have
now been wrong about three successive cost curves (P9's reference scan, P9a's
fixed-cost-dominated verify, and the post-`mul_mv_ext` one), so the protocol is a script
rather than a procedure someone re-derives from the last log entry. The shape: stage 1 sweeps `draft_p_min` at the shipped margin, stage
2 sweeps `pause_margin` at stage 1's winner, both against a plain `--no-draft` baseline
arm, rep-outermost interleave, greedy `-n 128` under `XWEN_BENCH=1`, 3 reps, medians.
The qualification criterion is P9's and is preserved deliberately: an arm qualifies only
if it is ahead of plain's median on BOTH prompt kinds in EVERY rep, the winner is the
highest mean-of-medians among qualifiers, and a tie resolves toward the shipped value.
The no-reuse rule is the part that is easy to get wrong: carrying stage 1's measurement
of the shipped margin into stage 2 looks like a free saving and is the interleaving
error one level up — it grades two arms from different thermal epochs. Every stage-2 arm
is re-measured. Supporting requirements, each of which a review found missing: cell
identities must not collide under rounding, the child env is clean with the HF cache
root resolved once to an absolute path and `HF_TOKEN` never passed down or serialized,
there is a contention guard and a per-run timeout, and the raw JSON dump is written 0600
and atomically after every run so a sweep that dies mid-way still yields its data. The
script prints recommendations and never edits a default (2026-08-08).

**A warm-up pass that reads faster than the timed pass following it is a thermal
ordering artifact, not evidence.** The warm-up runs on the cooler chip. This looked like
evidence about allocator-pool state during the residual diagnosis and was not; the
profiler excludes warm-up from its dumps for exactly this reason (2026-08-08).

**`--stats` reports drafting against a plain baseline measured INSIDE the same run, and
its plain bucket deliberately excludes wasted drafter time.** Every round is bucketed as
plain (no draft block verified — paused, empty-draft, serial-thinking, or past the
drafter's context; one token) or drafted (a block was verified; the round's full time,
draft phase included; accepted plus bonus tokens), with full-accept as a subset of
drafted marking the ceiling a longer block could reach. The load-bearing choice is what
the plain bucket folds: `round_ms - draft_ms`, the same quantity the pause controller's
plain comparator uses, so a round that ran the drafter and then committed plain
contributes only its target forward. The bucket is therefore not "what those rounds
cost" — it is what plain decode costs on this run's text, which makes `plain_rounds /
plain_ms` an interleaved plain arm sampled in the same thermal envelope as the drafted
rounds it is compared against. That is the whole reason to build it this way: the
interleaving rule above says a cross-session ratio is not evidence, and the 27B's
between-session level shift has already forced that warning into a log entry. A
breakdown that carried in an external plain number would reproduce exactly the error the
rule exists to prevent. `est. net ±Y.Y%` follows from the same partition — all committed
tokens priced at the run's own plain rate, against `plain_ms + spec_ms + (draft_ms -
spec_draft_ms)`, three terms that cover decode-loop model time once each. It is gated on
`plain_rounds >= 8`, because a rate off a handful of rounds is noise, and it is an
estimate rather than a measurement: it assumes the plain rate the run did exhibit is the
rate a fully plain run would have exhibited, which a real `--no-draft` A/B still owns.
The per-round averages it replaced divided by all rounds and inverted the reading of a
mostly-paused run; `draft` now divides by the rounds that actually drafted (2026-08-09).

## Qwen3.8-Flash-Next (qwen4exp)

The fourth checkpoint and the first that is a genuine second architecture rather than a
registry entry over an existing graph: 48 layers of gated DeltaNet and sparse attention
carried on a 4-stream hyper-connection residual, every layer MoE (512 experts, top-10),
plus a 51.2B n-gram embedding table. `docs/qwen4exp-port.md` is the arc's detailed
working doc — spec, traps, phase plan, running log — and stays that. **This section is
the authority for the decisions themselves**; the port doc's own Decisions list is
retained as the record of what was decided when, and points here.

**Third arch, composition over forking.** `Arch::Qwen4Exp` gets its own graph module;
shared blocks (DeltaNet, attention internals, MoE glue, rope) are reused by composition
and parameterized only where the math actually differs. The qwen35/qwen35moe forward
paths are not edited. Why: the existing three checkpoints keep their throughput BY
CONSTRUCTION — their code is untouched — and a divergent copy of DeltaNet would rot.
The parameterizations that came out of it are deliberately tiny: a `ZGate {Silu,
Sigmoid}` enum on `LinearAttnBlock` resolved at construction (qwen4exp's gated norm
gates on `sigmoid(z)`, ours on `silu(z)`, a silent-garbage difference), and MoE's renorm
clamp becoming a `sum_floor` field (existing checkpoints keep 6.103515625e-5, qwen4exp
passes 0.0 — the clamp is a 3.6-35B detail, not universal). The trunk seam is the same
principle one level up: `XwenModel` gained an `Option<Qwen4ExpParts>` and a one-line
dispatch in `run_stack` rather than a second model type, because `Generator` holds a
concrete `XwenModel` across 87 call sites over 26 methods, all of which a 4-stream model
needs identically (2026-08-26, seam widened 2026-08-29).

**The PLE table lives on the CPU side.** The `[320001536, 160]` n-gram table is mmap'd
file-backed; hashing, the 16-row gather and row dequant happen host-side per token, and
the result (2560 floats/token) feeds the GPU graph. Why: it is pure `get_rows` — no
matmul ever touches it — row addresses depend only on token ids and so are
prefetchable, and the page cache handles hot/cold better than any policy we would write.
GPU residency stays reserved for the ~80 GB trunk. Independently confirmed twice since:
llama.cpp implements it Gemma-3n-style as one host-side gather table, CPU-resident on
CUDA automatically; and LMSYS measured −0.07% throughput offloading it (2026-08-26).

**Refuted: building a cache or eviction policy for the PLE table.**
`garnermccloud/Qwen3.8-Flash-Next-NVFP4-SSD-Stream` ships the streaming version of that
design — the table pulled out as a flat FP8 sidecar, streamed per step with io_uring —
and it deliberately has NO cache and NO eviction, just fixed 64 MiB pools. That is the
right shape: 16 deterministic rows per token with poor reuse are better served by
issuing the exact page reads early than by an LRU, and on unified memory there is no
host→device staging at all, so "touch the pages early" is the whole mechanism. What
transfers is the sizing (2.5 KiB/token of payload, ≤ ~64 KiB/token in 4 KiB pages after
dedup, hidden by one decoder block of overlap) and one negative: they avoid mmap
readahead amplification, so `madvise(MADV_RANDOM)` on our mapping is the thing to test —
default readahead would turn a 90-160 B row into a large window. Their throughput
numbers are not a citation for anything; the baseline looks slow rather than the SSD
fast (2026-08-29).

**Reference-first for every new component.** Hyper-connections, the QSA indexer and PLE
each got a frozen CPU f32 reference with fixture tests before any device work, mirroring
the `ReferenceExperts` pattern; fixtures come from the transformers modeling code, the
one executable ground truth that existed at the time. It earned its keep immediately —
the fixtures settled the QSA whole-blocks-plus-tail question against llama.cpp, retracted
a wrong "PLE gate clamp" divergence we had recorded, and caught that a tail-0 context can
mask the query's own token (2026-08-26).

**Text-only; MTP deferred; serve after CLI.** Vision is dropped (masked_scatter
injection, empty deepstack — a clean cut, and mrope collapses to NEoX-64 for text
exactly as on 3.6/3.8). The MTP head has no transformers implementation and its forward
semantics are unconfirmed (separate `fc_embedding`/`fc_hidden` projections, NOT 3.8's
concat `eh_proj`), so it waits for vLLM/SGLang or the tech report rather than being
guessed at. Serve integration follows CLI bring-up — and as of the P2 review it is not
merely deferred but actively refused: a qwen4exp target would 500 on the snapshot path,
so `xwen serve` rejects the checkpoint until P4 (2026-08-26, sharpened 2026-08-29).

**Weights: Unsloth first, and `UD-Q4_K_XL` specifically.** Dev and first testing run
against Unsloth's Q4-class UD file; a self-converted file with a mix we control remains
the eventual parity target, because floors are calibrated per quant mix and the UD mixes
are per-layer heterogeneous. `UD-Q4_K_XL` is the chosen first target: it is the only
Q4-class trunk whose quant types are ones xwen already has kernels for (Q4_K / Q8_0 /
F32, with IQ4_NL confined to the PLE table). `UD-IQ4_XS` would be roomier — 64.9 GB
trunk against 82.5 — but needs IQ4_XS matmul kernels we do not have. 82.5 GB of wired
trunk plus a demand-paged table plus KV is tight on 128 GiB, but it is the same file
llama.cpp reported 24-25 tok/s decode on for a 128 GB DGX Spark (2026-08-29).

**The 640-column rule, and why Q5_1 became unavoidable.** `ffn_down_exps` is
`[640, 2560, 512]`, and 640 % 256 = 128, so it fails every K/IQ type's block-size
requirement and llama.cpp's generic `tensor_type_fallback()` demotes that plane to a
32-block type on EVERY publisher's file: Q4_K→Q5_0, Q5_K→Q5_1, Q6_K→Q8_0, IQ*→IQ4_NL.
`ffn_gate_exps`/`ffn_up_exps` (ncols 2560) keep their K-quants; `per_layer_token_embd`
(ncols 160) is 32-block-only forever, which is how ggml-org shipped a Q8_0 trunk with a
Q4_0 table. On our target that means Q5_1 down on 43 layers and Q8_0 on 5, against Q4_K
gate/up on 47 and Q5_K on layer 2. **Q5_1 support is therefore not optional and also not
blocking**: `ExpertStack` carries dtype per tensor with no whitelist, `FusedExperts::new`
never compares dtypes across planes or layers, decode falls through to candle's baked
`kernel_mul_mv_id_q5_1_f32`, and prefill drops the affected layer to per-token
`mul_mv_id`. So it runs correctly today and the kernel work is reclassified as P3 perf,
not P2 scope. The `IQ4_NL` matmul deferral stands unchanged — that was always about
IQ4_NL specifically, not about new matmul dtypes generally (2026-08-29).

**One oracle, no vendored copies.** `reference/llama.cpp` is a single submodule, bumped
e9fa0781 → `6fe749801` once PR #27742 merged, and it gates every checkpoint including
qwen4exp. Why this reversed: while the PR was unmerged, its files were vendored
read-only under `reference/qwen4exp/` as reading material, on the reasoning that an
unreviewed AI-drafted branch is not a frozen correctness oracle. The merge settled that,
and a proposed SECOND clone (so the 3.6/3.8 floors could stay frozen at the old pin) was
refuted in favour of moving the one pin and re-confirming: all three existing checkpoints
re-passed at `6fe749801` the same day, floors unchanged and not re-derived, so there was
nothing for a second clone to protect. `scripts/build-llamacpp.sh` needs no target
argument. Only `PROVENANCE.md` and the semantic `bea3b12d` → `6fe749801` diff survive in
`reference/qwen4exp/`, as history (2026-08-26, reversed 2026-08-29).

**The loader owns GGUF tensor-table parsing.** candle's `GgmlDType` cannot even PARSE a
file containing an IQ tensor — `Content::read` fails on the unknown dtype before any
kernel question arises — so the split-GGUF loader, already xwen-owned code, also owns
the tensor-table and dtype parsing, and the pinned candle stays unpatched. IQ4_NL work
splits three ways: metadata visibility (loader), CPU row dequant (needed only for the
PLE table), and Metal matmul kernels (needed only if a matmul weight is IQ4_NL —
deferred). Worth recording that this was DECIDED in P0 and not implemented until P2:
`gguf::open` on the real file failed with "unknown dtype for tensor 20" right up until
unit U0 landed it (2026-08-26, implemented 2026-08-29).

**QSA rides the existing attention block through an overlay, and decodes by row
gather.** `AttnBlock::forward` gained a trailing `Option<&QsaSelection>` whose `None`
path is byte-identical for existing checkpoints; prefill merges a `Mask` into the
existing `PrefillMask` path, decode passes `Rows` and gathers the selected K/V rows into
a packed contiguous view for a maskless sdpa. The gather exists because candle's sdpa
VECTOR kernel — the `seq == 1` route — is compiled WITHOUT mask support and SILENTLY
IGNORES a mask tensor, so a masked decode through stock sdpa would run dense attention
with no error at all (2026-08-26).

**QSA pooled keys stay f32; no round-back to the cache dtype.** The block key is
mean-pooled in f32 and goes straight into the k-norm and rope. HF rounds it back to the
raw-key cache dtype first, which at a BF16 indexer cache strips the pooled key to 8
mantissa bits before it is ever scored; llama.cpp pools through `ggml_get_rows` into f32
and never rounds back. We follow llama.cpp, because llama.cpp is this port's parity
oracle — the same rule that settles every other divergence in the 3.6/3.8 graphs. **The
consequence is recorded so it is not rediscovered as a bug: exact index-set parity
against an HF tap at real geometry is not attainable and is not a goal.** Measured at
real geometry, the bf16 round-back perturbs scores by ~1.2e-2 against a top-k cut margin
of ~2e-3, so roughly 0.5 of the 512 selected blocks per query differ at every context
length above budget. Grade the Metal path against the f32 oracle, not against HF
(2026-08-29).

**P2 keeps the new recurrent state out of `LayerCache`, and refuses what it cannot
carry.** Indexer raw-key caches, the PLE conv state and the 2-id token history live in
`Qwen4ExpParts` with their own checkpoint/rollback mirroring `LayerCache`'s, rather than
growing a fourth variant across five enums and ~15 match sites. Prefix-cache snapshots,
host snapshots and the disk tier do not carry them: a qwen4exp target refuses
snapshot save and restore with a loud error. Why: it decoupled three parallel units from
`kv_cache.rs` entirely. The cost is honest and now scheduled — it is exactly why serve
is refused until P4 (2026-08-29).

**PLE in P2 is host-hybrid, knowingly.** Hash, table row gather and IQ4_NL row dequant
run on the CPU from `MmapSource::bytes`; `key_proj`/`value_proj` run on device; the
per-stream gate, signed sqrt, dilated conv and silu run on the host in f32 over a
`[n, 10240]` copy of the stream — 40 KB/token and one device→host sync per forward at
layer 1. A known P3 cost taken deliberately for correctness first (2026-08-29).

**Refuted: the pre-release architecture priors.** The port was planned five days before
the card dropped, from a trimmed model-card and forum copy-pastes. Grading them against
the real config: GDN carried over (true, and byte-identical in geometry to our 27B
block, except the gated norm's z-gate is `sigmoid` not `silu`); the n-gram table is
Engram-shaped (true in structure, wrong in three details — raw token ids with NO
NFKC/lowercasing, ONE layer not "a couple of mid-stack layers", and a per-stream
dot-product gate rather than Engram's scalar one); hyper-connections were flagged as the
biggest structural risk at LOW confidence and are in fact present in every layer, which
is the single largest structural difference from anything we ship; and QSA being
DeepSeek-DSA-shaped was right in outline. The lesson is the one the priors themselves
warned about: they were useful for sizing the work and worthless as ground truth — every
one of them was re-derived from `config.json`, the transformers modular file and the
shipped GGUF headers before a line was written (2026-08-25, graded 2026-08-26).

**Phased, correctness before speed.** P0 scaffold (split-GGUF loader, config parse,
registry) → P1 CPU references and fixtures → P2 graph assembly, real file, greedy smoke,
oracle agreement → P3 Metal and perf → P4 serve, sampling defaults, harness extension.
P2 closed 2026-08-29 with the graph agreeing with the llama.cpp oracle at 189/192
forced-replay steps and zero hard mismatches (2026-08-26).

## Process

Inherited unchanged: multi-reviewer review with external model families on evidence
(reviewers recorded as wrong with disproofs, not just as right); a reviewer reads the
path you wrote, a live check walks the path you forgot; let the existing suite arbitrate
a proposed fix; docs drift is tracked work. Every shipped arc updates log.md (dated
entry) + README if the surface changed + this file if a decision was made, changed, or
refuted — a TODO.md update alone is not sufficient (2026-07-28).
