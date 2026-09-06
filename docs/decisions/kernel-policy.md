# Kernel policy

One topic of [docs/decisions.md](../decisions.md), the index of decisions by topic; moved
here verbatim on 2026-09-06. Dated paragraphs, newest additions appended within their topic.


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
no grandfather clause (2026-07-29). That uniformity ended 2026-08-30, and the
additions got their own switches and provenance fields precisely because they CAN
move a dump: the rescale branch's activation glue folds into `ops::silu_mul_l2`
(bounded — its sum order is not candle's; `XWEN_ACT_L2_CLASSIC`, provenance
`act_l2` at schema v9), and the shared expert's three projections can take the
dense prefill gemm (`XWEN_SHEXP_QMATMUL`, provenance `shexp_gemm`).
`XWEN_MOE_GLUE_CLASSIC` still covers only the bit-identical trio.

**The shared expert is ONE dispatch at decode, and it ships on by default even though it
bought a fifth of what the launch budget promised.** Five launches per MoE layer (gate,
up, silu*mul, down, gate logit) become two kernels in `src/ops/moe_glue.metal`:
`kernel_moe_shexp_gate_up`, which takes both Q8_0 gemvs, the SwiGLU and the
`ffn_gate_inp_shexp` logit, and `kernel_moe_epilogue_shexp`, the block epilogue with the
down gemv folded in and separate accumulators for the routed combine and the down dot.
Net −4 dispatches per layer, −192 per token on Flash-Next and −160 on the 35B-A3B.
Measured on a pinned binary, three interleaved rounds each: 35B decode 113.2 → 115.0
tok/s (+1.6%, fused ahead in every round), Flash-Next 51.2 → 51.5 (+0.6%, per-round
−0.4/+0.4/+0.8%). The prediction from the ~4 µs launch budget was +3.5-4% on Flash-Next
and it failed; the reason is the "Ceilings" refinement below. Shipped anyway because it is
positive on both checkpoints, loses nothing anywhere, and both correctness checks pass:
the Flash-Next replay check with `XWEN_MOE_SHEXP_CLASSIC=1` as control, and the 35B parity
gate (log.md, this date). Like the fused hc gate it copies, it is BOUNDED rather than
bitwise (~1e-6 relative L2 from an f32 host oracle at both shipped geometries at n = 1, 3
and 8, both accumulations reassociated), so it gets its own switches and a provenance
field rather than living under `XWEN_MOE_GLUE_CLASSIC`: `XWEN_MOE_SHEXP_CLASSIC` restores
the five-dispatch chain, `XWEN_MOE_SHEXP_FUSED_MAX_N` (8, inclusive) bounds the window,
`moe_shexp` at schema v11 records which side ran, the strict tier pins classic and the
mm/decode/ppl tiers grade fused. `kernel_moe_epilogue` itself is unchanged and still
bit-identical. Known limitation, ledgered: the provenance field is written from the env
predicate rather than from observed execution, so it records intent, and the one-time
"moe: shared expert fused|classic at N token(s)" host line (0ed20ea) is what a bench uses
to prove a run was not a silent fallback (2026-09-06).

**hc planes are dense_mm-only.** The loader's plane predicate is
`dense_mm_supported || mv_ext_supported`, so giving the hyper-connection bottleneck
planes for the prefill gemm would also have opened the mv_ext 2..8-token window
inside `QLinear::forward` — on every hc path including `XWEN_HC_CLASSIC`, and
asymmetrically: `down` (k 10240) is mv_ext-eligible where `up` (k 320) is not.
Ragged chunks and serve resumes live in that window. `QLinear::without_mv_ext`
therefore pins the hc projections' `forward` to QMatMul at every token count the
gemm doesn't take — bitwise the pre-plane behavior, test-pinned
(`without_mv_ext_keeps_small_batch_on_qmatmul`) — while `forward_gemm` keeps the
plane. The shexp planes predate the change and keep their window:
`XWEN_SHEXP_QMATMUL` restores that surface's immediate pre-change route, mv_ext
included (2026-08-30; all three external/internal reviewers converged on the find).

**Rejected: keeping the hc dense-gemm route off by default until a qwen4exp oracle
tier exists** (a Codex review recommendation). The whole qwen4exp port is graded
without an oracle tier — forced replay, greedy equivalence, and switch A/Bs — so the
condition would gate this one lever on infrastructure nothing else waits for. The
route is A/B'd at the dense_mm precision class (3.66-3.69e-4 rel_l2 against the
QMatMul route over identical bytes), its greedy forks bisect to near-ties (every
lever alone forks the same token; the all-classic arm is deterministic), and it is
worth +7-11% prefill on the default checkpoint. `XWEN_HC_GEMM_QMATMUL` remains the
lever if evidence turns (2026-08-30).

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

**But the projection's DISPATCH is now a vendored gemv at 1..8 rows, and its pin on the
oracle is load-bearing (2026-09-06).** The paragraph above still holds of the FUSION: the
kernel takes over at the softmax, both matmuls stay outside it, and no bounded router
exists. What changed is what runs underneath the projection. candle lowers
`[1, hidden] × [hidden, n_expert]` f32 to the mlx `gemv_t` kernel, which covers the whole
plane with 8 threadgroups (5.24 MB per layer on Flash-Next, 2.1 MB on the 35B), and at
2..8 rows to a 32×32-tile gemm on 16 threadgroups with 24 of its 32 M-rows idle. Every
cheap route out was closed by reading candle at rev 21cca0b: the tile choice is hardwired
with no knob, a `QMatMul` over an F32 QTensor dequantizes and lands back on the same gemv,
and candle's own `kernel_mul_mv_f32_f32` receives a zero row stride from its host code. So
`kernel_mul_mv_f32_f32_v` (`src/ops/f32.metal`, the f16 gemv of `src/ops/f16.metal` with
float weight loads) is vendored, and runs 256 threadgroups on Flash-Next's 512 experts. It
is taken at 1..=8 rows inclusive (`XWEN_ROUTER_MV_MAX_N`, 0 for classic) unless
`XWEN_ROUTER_MV_CLASSIC`; candle's matmul still runs above the ceiling, so prefill is
untouched. The route is reassociation only, ~1e-6 rel_l2 at both geometries and at t = 1,
3 and 8, but this switch is unlike every earlier bounded one: top-k routing is discrete and
the router runs BEFORE the routing decision, so a near-tie flips a whole expert rather than
a few output bits. The strict tier pins classic AND the reference oracle runs classic, and
that second pin is the load-bearing one; schema v12 records `router_mv`. The cost is that
the router plane is held twice, transposed for candle and as loaded for the gemv, ~251 MB
on Flash-Next and ~8 MB on the 35B, which is open and ledgered. Worth +10.3% decode on the
35B-A3B and +4.8% on Flash-Next (log.md "Router projection on a 256-threadgroup gemv").

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

**The beta|alpha projection folds into its own head at up to 32 tokens — a dispatch removed,
not a kernel made faster (2026-08-30).** `kernel_delta_ba_fused` (delta.metal) reads
`x_normed` and the `[hidden, 2 · v_heads]` f32 weight and writes `beta` and the log-decay
directly, replacing a candle gemv (~30 µs per layer for a ~1 MB weight, i.e. ~33 GB/s)
plus `kernel_delta_ba`; the projection output the two-dispatch arm materializes never
reaches memory. It removes one dispatch per DeltaNet layer per token and nothing else,
and it pays on both DeltaNet checkpoints (530-token prompt, 128 decoded, unprofiled,
interleaved): **Flash-Next 44.4-44.5 → 46.5-46.7 tok/s (+4.6-4.8%, 36 layers), 35B-A3B
105.1 → 114.4 (+8.8%, 30 layers)**, prefill unchanged on both (796-798 / 2248-2268),
because a prefill chunk takes the gemv either way. The 35B gains nearly twice as much for
the reason that generalizes to every remaining fusion on this path: the saving is a fixed
~0.7-0.8 ms, and it lands against a 9.5 ms token there against Flash-Next's 22.

Geometry: one threadgroup owns `DELTA_BA_COLS` = 8 output columns, FITTED (8 → 12
threadgroups at 7.4 µs; 16 → 6 at 9.9; 4 → 24, tying at decode and losing by half at seq
8-32; below 8 a threadgroup's weight run stops filling a cache line on its own), each
column's dot split across `DELTA_BA_ROWS` = 128 row chunks folded in a tree, and a `_t4`
specialization tiling `DELTA_BA_TOKS` = 4 tokens so a short verify chunk reads the weight
once per tile rather than once per token.

Three things about it are load-bearing. **The ceiling is deliberately short of the
crossover.** `DELTA_BA_MAX_SEQ` = 32 covers decode (1) and a DFlash verify block (16);
the measured crossover is farther out (18.8 µs fused against the chain's 71.7 at seq 32,
`delta_ba_timing`), and it is not taken, because the fused kernel reads the whole weight
once per token TILE where candle's gemm reads it once per CHUNK — prefill chunks are
hundreds of tokens, that regime has never been measured, and the gemm's reuse is why
prefill is shaped the way it is. Prefill takes the gemv unchanged. **The epilogue is
shared so the arms cannot drift.** The beta sigmoid and the softplus decay against the
pre-baked `ssm_a` and the dt offset are two `static inline` helpers (`delta_ba_beta`,
`delta_ba_logdecay`) called from both kernels, which is what keeps the plain
`kernel_delta_ba` BITWISE against the reference while the fused one exists.
**It is bounded, not bitwise**, because it reassociates the dot product against candle's
gemv: graded at 2e-6 across every shipped geometry at seq 1/3/4/5/16/32, the widest
(hidden 5120) measuring 1.05e-6 on the decay — the same class as `XWEN_DELTA_CLASSIC`,
and tight enough that a swapped beta/alpha column block or a mis-tiled token still fails
by orders of magnitude. `XWEN_DELTA_BA_CLASSIC=1` restores the two-dispatch chain as a
kill switch (not a bit-identity anchor); `parity-gate.ts` strips it from the run env with
the other kernel switches, and it sits inside the arm `XWEN_DELTA_CLASSIC` already
switches away from, so it appears in no parity row (parity.md).

**State the text claim precisely: byte-identical over the window that was GRADED, not
forever.** All three shipped checkpoints re-passed their gates at 0261e17 (35B-A3B six
graded, 27B five, 3.8-27B five), and Flash-Next forced replay reads 185/192 with 7
near-ties (0.0002-0.288 logit, rank 2-3), 0 hard, against 186/192 with 6 ties at fd46c7a
— the extra flip is a 0.0002-logit tie. The greedy text is byte-identical to the classic
arm over 64 tokens and **forks at about step 124 of a 128-token free run** on the
530-token prompt. That is what a 2e-6 bound predicts and not a defect, but "byte-identical"
without the window attached is the kind of claim that gets quoted into a bitwise
assumption later. Under teacher forcing over 128 steps the two arms pick the same top-1 at
every step and drift apart by at most 0.32 logit (log.md 2026-08-30); the fork is that drift
crossing a boundary on the free-run path, not a step either arm gets wrong.

Follow-up ab5b322 added a device threadgroup-capability check to
`delta_ba_fused_applies` (the predicate now refuses a device whose
`maxTotalThreadsPerThreadgroup` cannot cover `DELTA_BA_COLS * DELTA_BA_ROWS`), brought a
tail geometry under test, and corrected the comments; no arithmetic moved, so the
0261e17 parity grade stands for it.

This is the arc's one shipped win out of three units, and the reason it paid generalizes:
see "How to read `XWEN_GDN_PROFILE`" and the `attn_qkv` refutation below. The dispatch it
displaced was also doing real work badly, which is NOT true of the remaining fusion
candidates on this path (TODO.md item (14)) — expect 8.41 µs × 36 layers ≈ 0.3 ms from
those on Flash-Next, not 1 ms.

**The prefill chunk is per architecture — 2048 on the MoE checkpoints, 512 on the dense
ones — on every surface, and 4096 is not better anywhere.** Prompts are fed to the model
`XwenModel::prefill_chunk()` tokens at a time: `Arch::prefill_chunk_default` unless
`XWEN_PREFILL_CHUNK` overrides it, and serve, generate, chat, batch and the logits-dump
ppl pass all read that one accessor (so a serve prefill split for a departed-client check
still costs no extra GPU passes, and the ppl tier scores the shape generate runs). It was
a flat 512 from the fork, chosen for the dense 27B's attention working set. On an MoE
checkpoint the chunk is also the expert batch: Flash-Next routes top-10 of 512 experts,
so a 512-token chunk hands each expert ~10 rows per `mm_id` gemm and the gemm runs at
gemv-like intensity; at 2048 each expert sees ~40 rows. Measured 2026-08-30
(`powermode 0`, interleaved arms, medians; log.md): Flash-Next at 3851 tokens
748 → 814 → 824 → 745 tok/s for chunks 512 / 1024 / 2048 / 4096 (three rounds), at 1962
tokens 883 → 933 → 951 → 957; 35B-A3B at 3851 tokens 2429 → 2634 for 512 → 2048 (+8.4%,
two rounds); the dense 27B at 3851 tokens 650/599 → 608/571 (round 1 / round 2), 2048
being 5-6% SLOWER in both rounds. Decode was unchanged in every arm, and greedy output
over 64 tokens byte-identical between 512 and 2048. Peak phys_footprint rose 17.4 → 19.5
GB at 2048 and 22.5 GB at 4096 on Flash-Next, 9.4 → 11.3 GB on the 35B, 41 → 44-46 GB on
the 27B. The dense result is the explanation for the 4096 one: what grows with the chunk
is not only the expert batch — the sdpa mask and the attention score tile grow with the
SQUARE of the chunk — so a checkpoint with no expert batch to feed pays the quadratic
cost and collects nothing, and on the MoE ones that cost outruns the rows-per-expert gain
past 2048 (at 1962 tokens 4096 and 2048 are the same single chunk, which is why that
prompt cannot separate them). Hence a per-architecture default rather than one number:
2048 where there are experts, 512 where there are none, 1024 the MoE fallback if the
+2 GB ever matters (2026-08-30).

**mm_id tiles: the pass-2 grid is a work list, and the token tile is 64 wide when
experts average ≥ 24 rows.** The vendored two-pass `mm_id` launched pass 2 on ggml's
`(t/32, n_out/64, n_expert)` grid — sized for one expert owning every row — so at the
2048 chunk ~97% of threadgroups early-returned; and its `_t` kernel is dequant-bound
with the expert tile dequantized once per token tile, so passes per expert are
ceil(rows/NR1) and the ledgered NARROWER tile (16) would have raised the dominant cost
(Flash-Next 1.88 → 2.97 passes; 35B 2.5 → 4.5). Hence the opposite: map0 emits a flat
(expert, tile) work list bounded at ceil(t*top_k/NR1) + n_expert with no readback, pass
2 walks it, and the `_t` family is templated on NR1 with 64 chosen by
`t*top_k/n_expert ≥ 24` (Flash-Next 40, 35B 64 at the 2048 chunk). Both are bit-neutral
by measurement (`work_list_and_nr1_64_match_full_grid_nr1_32_bitwise`; the 32/64
identity is test-pinned, not structural — different `matmul2d` instantiations). Measured
2026-08-30: isolated, the expert gemms run +17-23% (FN gate/up 416k → 512k tok/s, FN
down q5_1 202k → 236k, 35B gate/up 628k → 751k, 35B down 260k → 281k); end-to-end at
3803 tokens NOTHING claimable (Flash-Next arms inside one arm's round-to-round spread;
35B +1.9% in one round), the profiler ranking showing why — `ffn` fell only 3-5%
because the gemms are a minority of that stage next to the router, rescale chain,
[REFUTED 2026-09-05, "Ceilings" below: the profiler that ranked them inflates prefill
2.2x; the duplicate-dispatch probe prices the expert gemms at 28-32% of prefill wall and
73% of the `ffn` stage in situ]
SwiGLU, combine and shared expert. Shipped anyway as the correct shape of the kernel
(less idle launch, fewer dequant passes, no accuracy trade), with `XWEN_MM_ID_FULL_GRID`
and `XWEN_MM_ID_NR1` as kill switch and A/B knob, and the finding re-ranks the prefill
ledger toward the non-gemm parts of `ffn` (2026-08-30).

**Dispatch-floor levers ranked (2026-08-30).** The 8.41 µs fixed cost every dispatch
pays is not a Metal constant and it is not ours to fix in a `.metal` file — the ranked
candidates for it are all in candle's encoder layer, and a read of the pinned rev
(21cca0b) put them in order: whole-scope barriers that could be per-resource scoped
(`encoder.rs:104-149`), cross-encoder fence waits that are unfiltered by dependency
(every new encoder waits on every live fence), and the CPU-side locking each dispatch
does to bind (an `EntryState` mutex, 4-6 acquisitions and `HashSet` inserts per bind) —
that last being the only one plausible as a component of the floor itself. All three are
candle patches, none is priced, and the commit cadence
(`CANDLE_METAL_COMPUTE_PER_BUFFER`, default 50, counted per dispatch) is the one knob
that needs no patch. Full anchors and the reasoning in log.md "technique survey"; queue
in TODO.md. Nothing here ships on a tok/s reading alone: candle's pooled-buffer recycle
has no in-flight check (`device.rs:488-503`), so a concurrency or cadence change is
graded by the parity gate plus greedy equivalence.

**The causal prefill mask is built on the device, and it is a MEMORY change (2026-09-06).**
`PrefillMask::causal_on_device` replaces the host `Vec<f32>` fill — a scalar double loop
over `seq x (pos + seq)`, ~8.6e9 stores over a full 131072 prefill — and its upload with
two `arange` vectors, a broadcast compare and a `where`. Bit-identical by construction
and by test: `the_device_built_causal_mask_equals_the_host_fill` compares the additive
f32 plane AND the f16 sdpa copy at every chunk shape the prefill walk produces, on the
real Metal device. `XWEN_HOST_MASK=1` restores the host path, which a non-Metal device
takes anyway; it is the `--control` arm for `scripts/flashnext-replay.ts` and is stripped
by `parity-gate.ts`.

The ledger item that asked for this called the host fill "the binding cost of long
prefill". That is REFUTED on time: at 131072 the two arms are a dead heat on both
checkpoints (35B 667.8 against 659.2 tok/s, Flash-Next 230.8 against 230.9), because
candle is asynchronous and the host fills chunk N+1's mask while the GPU is still on
chunk N, so ~69 GB of stores and uploads hide behind a prefill of 197 s and 569 s
respectively. Nobody should quote this change as a throughput win.

What it does buy is 25-52 GB of peak footprint on the dense-attention path: the 35B's
131072 peak goes from 42-69 GB to a flat 17 GB, and the device arm's two passes agree
exactly where the host arm's differ by 27 GB. The mechanism is buffer handling, not
arithmetic — the host path reaches the device through `Tensor::from_vec`, one fresh
exact-size buffer per chunk with no two chunks asking for the same size, while `where`
allocates through the pooled builder and gets recycled. `ops::chunk_sync`'s doc comment
had already named that shape ("a pool that only ever adds entries because each chunk's
mask upload asks for a fresh exact-size buffer") before anyone measured it.

Flash-Next does not move, at 59 GB either way, because its QSA indexer builds its own
`n x n_kv` f32 mask through the same `Tensor::from_vec`, per sparse layer per chunk, and
above the 2048 budget that is every chunk. Moving THAT one is the follow-up, and it is
ledgered with a memory number rather than a time one — the time criterion it was given
(10% of the 131072 prefill wall) fails on the causal mask's own 0%.
