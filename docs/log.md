# Engineering log

Reverse-chronological. Heading convention: `## YYYY-MM-DD — headline stating what
shipped, ideally with the number`. Same-day entries disambiguate in the heading text.
Superseded entries are marked in the headline, never deleted.

## 2026-07-29 (latest) — First llama.cpp head-to-head: xwen wins decode on both models, loses 27B prefill 2x to the sequential DeltaNet scan

**Context.** Three perf arcs (sampler tail, fused DeltaNet, fused MoE glue) had moved
xwen's own numbers, but nobody had ever measured llama.cpp on this machine. Same GGUF
files fed to both engines (the blessed Q4_K_M pair), llama.cpp at the pinned oracle
build e9fa078 with `-fa 1`, strictly interleaved arms per the measurement-discipline
entry, 6-7 reps per cell, decode at matched context depth 630. Power mode: AC,
`lowpowermode 0`, no `highpowermode` key — Automatic, the whole run.

**Results** (median tok/s, ratio = xwen/llama.cpp):

| cell | llama.cpp | xwen | ratio |
|---|---|---|---|
| 35B decode tg256@630 | 98.0 | 103.1 | 1.05 |
| 35B prefill @925 / @4096 | 2725.7 / 2497.9 | 2546.6 / 2316.6 | 0.93 / 0.93 |
| 27B decode tg256@630 | 19.2 | 19.6 | 1.02 |
| 27B prefill @925 / @4096 | 486.2 / 501.8 | 268.9 / 235.6 | 0.55 / 0.47 |

**Reading.** Decode is won on both models — the design target metric. The 35B prefill
deficit is mostly thermal-boost asymmetry: llama.cpp's early reps boost to ~3140 and
settle to ~2500-2700 (-17%) while xwen only drifts -5%; restricted to settled reps the
ratios are 0.96 @925 and 0.90 @4k, i.e. near parity at 925. The 27B prefill gap is
real, large, and length-growing: xwen *degrades* 269→236 from 925 to 4k while llama.cpp
*improves* 486→502. The cause is the one CLAUDE.md's P8 note predicted — the 27B runs
48 sequential DeltaNet reference scans at inner width 6144 (vs the 35B's 30 at 4096),
and llama.cpp prefills those layers with its chunked (chunk=64) form. P8b (chunked
scan) is the named fix and now has a measured bounty: ~2x on 27B prefill.

**Sampler asymmetry checked.** llama-bench's tg loop runs no sampler; xwen's decode
number carries full 1.0/0.95/20 sampling. Greedy control run: 104.9 vs 103.1 — the
sampler costs ~1.5%, so the decode win is not an artifact of what each side skips.

**Also confirmed.** The 2026-07-29 MoE-glue numbers reproduce independently (35B decode
103.1 vs 102.8 quoted, 27B 19.6 vs 19.0). Raw per-rep data and harness scripts are in
the session scratchpad (results-{35b,27b}.txt), not the repo.

## 2026-07-29 — Top-p switched to llama.cpp's renormalizing convention

**Context.** The 2026-07-28 sampler rewrite kept candle's top-p rule — cut against
full-vocabulary mass, skip the cut entirely when the top-k set holds less than `top_p`
of the total — and ledgered the divergence rather than fixing it inside a perf change.
llama.cpp is this project's ground truth for everything else, and `--top-p 0.95` did not
mean what a llama.cpp user would read it as.

**Change.** `truncate_top_p` is now `llama_sampler_top_p_apply`. `top_p >= 1.0` returns
without touching anything (llama.cpp builds an empty sampler there); otherwise the top-k
survivors are renormalized to sum to one and the walk keeps the shortest prefix whose
cumulative mass reaches `top_p`, comparing `cum_sum >= top_p` with the crossing token
INCLUDED. `min_keep` is not carried — llama.cpp's default is 0, disabled, and the loop's
own at-least-one guarantee is all that default gives. The two branches of
`candidate_set` collapsed into one: the "is the top-k mass already under `top_p`" guard
existed only to express the absolute-mass rule and has no analogue here. The
renormalization is CPU-side over the ≤20 candidates and costs nothing; the device-side
full-vocabulary softmax stays for now.

**What this changes.** Sampled outputs, one-directionally: renormalizing cuts the same
or more, never less, so a seeded stochastic run draws a narrower candidate set than a
pre-2026-07-29 build wherever the cut bites. Accepted. Greedy is untouched and the
parity gate is greedy end to end, so nothing there moves. A side effect worth naming:
the cut no longer compares absolute mass, so the softmax denominator divides back out of
it, and the device fast path and the CPU `SampleControl` path can no longer truncate
differently by an ulp at the threshold.

**Tests.** `top_p_measures_absolute_mass_not_renormalized_mass` is replaced by
`top_p_measures_mass_renormalized_over_the_candidate_set`, built on two hand-computed
rows where the conventions give different answers — one where the old rule's cut ran and
stopped short, one where it was skipped outright because the top-k set held only 0.76 of
the mass. A `llamacpp_filtered` transcription joins the candle transcription as an
oracle, and the candle-equivalence matrix now claims candle agreement only at `top_p`
1.0, where no cut applies; below that it is llama.cpp the candidate set is checked
against, at every shape and every p. 29 sampler tests pass, clippy clean.

**Not done.** The perf half of the ledger item — the fast path no longer needs a
full-vocabulary softmax, since renormalizing over k survivors IS a k-wide softmax — is
split out as its own TODO entry. It is gated on a Metal top-k, not on this change.

## 2026-07-29 — Fused MoE glue: an MoE layer goes from 24 dispatches per token to 14, and 35B decode from 92.6 to 102.8 tok/s

**Context.** With the DeltaNet layers fused, the 35B-A3B's remaining decode cost is the
MoE half, and it was launch-bound rather than bandwidth-bound: 24 dispatches per layer
at seq==1, 960 per token across the 40 layers, of which exactly 8 are real matmuls.
Everything else is glue — a softmax, an arg-sort, a gather, a sum, two clamp halves that
each upload a fresh 4-byte scalar buffer, a divide, a sigmoid, three elementwise
multiplies and three adds.

**Change.** Three fusions, each bit-identical to the candle chain it replaces, behind one
kill-switch (`XWEN_MOE_GLUE_CLASSIC=1`).

`kernel_moe_router` (new `src/ops/moe_glue.metal`) takes the router logits and returns
the selected ids and their renormalized weights in one threadgroup per token, replacing
seven dispatches. `kernel_moe_epilogue`, in the same file, computes
`Σ_k down[k]·w[k] + shexp·sigmoid(gate)` in one pass over the uncombined down
projection, replacing the weighted combine, the shared-expert gate sigmoid, its
broadcast multiply and the routed+shared add — four dispatches. And the shared expert's
`silu(g)*u` now calls the existing vendored `ops::silu_mul` instead of candle's two
elementwise passes. The routed experts' `ExpertFfn` grew a `project` method returning
the down projection BEFORE the combine, so the epilogue can own the combine; the
reference oracle declines it (it scatters as it goes) and so does the f16-tile prefill
branch (its projection carries an L2 rescale the epilogue has no term for), and both
fall back to the untouched classic composition.

**The router matmul stays candle's, deliberately.** It lowers to MLX's
`gemv_t_float32_bm1_bn2_sm8_sn4_tm4_tn4`, whose K-partition is strided across lanes
(thread t owns `{i·32 + 4t + tm}`) and whose combine is a specific shuffle-down tree —
reproducible only by lifting the kernel verbatim, and even then resting on the compiler
contracting `result += vc * inter` to an fma the same way. Worse, its accumulation order
depends on the output width, so concatenating the shared-expert gate row onto the router
weight to save a dispatch would have changed that gate's bits. Both matmuls stayed
candle's; the fusion starts at the logits. This is branch (b) of the briefing's decision
tree, and it costs one dispatch out of the ten saved.

**Bit-identity held everywhere, including the two places it looked unlikely.** candle's
`softmax_last_dim` is not a max/exp/sum triple but the online Welford form, with `fast::exp`
in the merge, `fast::divide(1, d)` hoisted once per thread, and a five-step
`simd_shuffle_down` tree over `MD{m, d}` pairs (the merge op is not a built-in simd op,
so candle takes the shuffle tree, not `simd_sum`); the kernel reproduces its
compile-time BLOCKSIZE switch over a runtime width. And candle's Metal arg-sort is
llama.cpp's bitonic network, which is deterministic but NOT stable — its comparators are
strict and never consult the index, so equal probabilities do not come out in ascending
expert order. That matters more than any rounding here: a tie flip swaps an entire
expert. The network is reproduced verbatim rather than replaced by a selection
algorithm, and `router_ties_match_candle_bitwise` pins it against candle's on four
tie families (all-equal, a twelve-way plateau with eight slots to fill, a
sixteen-value block repeated across the row, and a single winner over thirty-two tied
runners-up).

**Perf.** 35B-A3B Q4_K_M, `lowpowermode 0`, warm, batch 1, `--no-draft`, 630-token
prompt, 256 decoded tokens. Interleaved arms, median of 5, per docs/decisions.md
"Measurement discipline":

| | classic | fused | |
|---|---|---|---|
| decode | 92.6 tok/s | 102.8 tok/s | +11.0% |
| prefill 925 | 2475.1 tok/s | 2488.9 tok/s | +0.6% |
| prefill 4k | 2101.4 tok/s | 2105.8 tok/s | +0.2% |

The decode arms never overlap across the five reps (classic 91.9–93.7, fused 102.2–104.3),
and an earlier independent run on a cooler machine put the same ratio at +11.1%. Prefill
is unmoved, as it should be: above `MM_ID_MIN_SEQ` the epilogue declines and only the
router kernel and the shared expert's activation fuse, against a chunk that is
compute-bound rather than launch-bound.

**A fourth fusion was built, measured, and switched off.**
`kernel_mul_mv_id_q4_K_f32_dual` computes the gate and up expert matvecs and their
SwiGLU activation in one dispatch instead of three, and is bitwise against that chain —
it runs the same `mul_mv_q4_K_sums` body twice, factored out of the verbatim ggml
`mul_mv_q4_K_impl` so neither projection's accumulation order moves. It measured
SLOWER: 99.5 against 102.8 tok/s decode, five interleaved reps apart with no overlap. It
is not register pressure — both pipelines report `max_total_threads_per_threadgroup` of
1024. It is the grid: two independent `_v` dispatches expose twice the threadgroups, and
a bandwidth-bound gather turns that into outstanding memory requests, so folding them
into one grid that walks both weight planes serially per thread costs more than the two
saved launches buy. Kept behind an opt-in `XWEN_MOE_DUAL=1` as a measured artifact worth
re-pricing on another device, not because anything runs it.

**Parity provenance needs no schema change, and that is the point of the bitwise work.**
Every shipped kernel here reproduces its chain's rounding boundaries exactly, so there
is no path for the fused/classic split to change a dump — no new provenance field, no new
pin, no grandfather clause. Both gates confirm it, reporting numbers identical to the
pre-change run: 35B ALL PASS (strict cos=1.000000, mm cos=0.999631, decode 63/63/62 of
64 with 0 mismatches, ppl Δnll=0.000791), 27B ALL PASS (strict and mm cos=1.000000,
decode 64/64 on all three fixtures, ppl Δnll=0.000330). The dense 27B never enters this
code — `DenseMlp` keeps the free `swiglu` helper untouched — and was re-gated only
because the ledger says a shared file was edited.

`XWEN_MOE_GLUE_CLASSIC=1` joined `parity-gate.ts`'s `referenceEnv()` on hygiene, so the
oracle stays a pure candle composition, and it is deliberately NOT on the strict
candidate: the strict tier now grades the FUSED glue against a reference that ran the
unfused one. Regenerating the reference under the new pin (`--regen-ref`) and re-running
returns `cos=1.000000 top5=5/5` with every other tier unchanged — the full-model
confirmation that the ops-level bitwise tests generalize past synthetic shapes.

**Tests.** 679 lib tests pass (the two deliberately-red dflash `real_file` tests
unchanged), plus 63 in the parity binary. Eleven are new: eight in `ops::moe_glue`
(bitwise router across eight geometries including a padded bitonic network and a softmax
width below candle's shared-memory threshold, the tie families, offset views, a clamp
whose floor actually binds, the bitwise epilogue across three sequence lengths with gate
logits in both sigmoid saturation tails, a `#define`-vs-Rust geometry cross-check, and
the shape/dtype contracts), two in `ops::mv_id` for the dual kernel, and
`moe::fused_block_matches_candle_bitwise`, which runs a whole synthetic MoE block
through `MoeBlock::forward` and compares it against a reference built entirely from
candle ops — the wiring test, where a swapped operand or a gate that picked up a stray
sigmoid would show up and the per-kernel tests would not.

## 2026-07-28 — Fused DeltaNet kernels: a layer goes from ~65 dispatches per token to 8, and 35B prefill from 305 to 2183 tok/s

**Context.** Three of every four Qwen 3.6 layers are gated DeltaNet, and all of them
ran the P3 reference: composed candle ops, one scan step per token. Decode spent about
65 Metal dispatches on a layer containing seven matmuls. Prefill was worse in kind, not
just degree — the scan is a Rust `for t in 0..seq` loop issuing eight dispatches per
timestep per layer, so a 512-token chunk on the 35B cost roughly 123k dispatches. This
is P8's decode-side package. The chunked (chunk 64, tri-solve) scan was explicitly out
of scope and stays open as P8b.

**What landed.** `src/ops/delta.metal` + `src/ops/delta.rs`, four kernels wired into a
`forward_fused` beside the untouched reference (renamed `forward_classic`, math
verbatim — it is still the oracle):

- `kernel_delta_conv` — causal depthwise conv, silu, and the next conv window, reading
  the carried window and the fresh qkv rows as two buffers. That kills the `cat`, and
  writing the window directly kills the `zeros_like` + `slice_set` materialization.
- `kernel_delta_ba` — `beta = sigmoid(b_raw)` and `g = ssm_a * softplus(a_raw +
  dt_bias)` from ONE `[hidden, 2·v_heads]` projection, built at load time by
  concatenating `ssm_beta` and `ssm_alpha`. Two gemvs become one. It emits the LOG
  decay and lets the scan exponentiate, folding away another pass.
- `kernel_delta_scan` — the whole recurrence, all T timesteps, one dispatch.
- `kernel_delta_gnorm` — the gated output RMSNorm.

Eight dispatches per layer at any sequence length.

**The scan's shape is the whole trick.** Value-dim columns of the state are completely
independent — `sk[j]`, `d[j]`, the rank-1 update of column `j` and `o[j]` all touch only
column `j` — so the only cross-thread folds are the two key-dim contractions. A
threadgroup owns one V-head and 32 of its 128 columns; thread `(r, jl)` holds 32 state
rows of one column IN REGISTERS for the entire scan. The state is read once and written
once no matter how long the chunk is, which is what turns prefill's per-timestep
dispatch storm into a single launch. Consecutive threads in a simdgroup share `r` and
take consecutive `j`, so both state passes are contiguous 32-float runs. q and k are
read straight out of the conv output with the tiled K-head mapping (V-head `h` reads
K-head `h % k_heads`) and L2 clamp-normalized in the load stage, so the reference's
materialized tile-and-broadcast disappears too.

**Measured, warm (`XWEN_BENCH=1`), batch 1, greedy (`--temp 0 --seed 7`), `-n 128`.**
Power: `pmset -g` reports `lowpowermode 0` and exposes NO `highpowermode` key on this
machine, so low-power mode is confirmed off but the High Power tier is neither
confirmed nor available — do not read these as laguna's "full power" anchors. Protocol:
fused and classic runs INTERLEAVED, three reps per arm, median reported — see the
measurement-discipline note below, which is the reason these numbers are not the ones a
naive sequential matrix produces.

35B-A3B Q4_K_M:

| prompt | prefill classic | prefill fused | decode classic | decode fused |
|---|---|---|---|---|
| 596 tokens | 305.4 tok/s | **2183.2** (7.15x) | 57.8 tok/s | **91.2** (1.58x) |
| 1929 tokens | 300.3 tok/s | **2274.1** (7.57x) | 56.6 tok/s | **88.0** (1.55x) |

27B dense Q4_K_M, which had no perf numbers at all before today:

| prompt | prefill classic | prefill fused | decode classic | decode fused |
|---|---|---|---|---|
| 596 tokens | 77.3 tok/s | **290.4** (3.76x) | 14.3 tok/s | **19.0** (1.33x) |
| 1929 tokens | 77.9 tok/s | **209.3** (2.69x) | 14.3 tok/s | **17.9** (1.25x) |

The 27B's decode gain is smaller (1.25-1.33x vs the 35B's 1.55-1.58x) because 64 dense
SwiGLU layers at hidden 5120 dominate its per-token budget; dispatch count was never
its problem. Its per-rep spread is also visibly wider than the 35B's — the 35B's
classic arm repeated at 303.2/305.4/305.6 tok/s prefill, while the 27B's fused 596-token
decode walked 21.7/19.0/17.9 across its three reps as the machine heated.

**A measurement finding worth more than the numbers.** The first pass of this matrix ran
the eight cells sequentially and reported the 27B at 13.9 tok/s decode and 131.3 tok/s
prefill. A cooled, interleaved re-run of the identical binary put the same cells at 19.0
and 290.4. The whole matrix drifts 20-35% slower over roughly ten minutes of continuous
GPU load, uniformly across BOTH arms and both checkpoints, and `pmset -g therm` records
nothing while it happens — the only tell is the control arm moving too. So a sequential
A/B on this machine is not an A/B; the arms have to be interleaved. Recorded in
decisions.md "Measurement discipline", along with the sibling trap found the same hour:
`pgrep -f "logits-dump"` matches the argv of the shell running the check, so it reports
a model process that does not exist (this bit parity-gate's own preflight too — use
`pgrep -x`).

**This is the first vendored kernel family that is not bit-identical, and that changed
the parity gate.** Every earlier fused kernel reproduced its candle chain's rounding
boundaries exactly, so its `*_CLASSIC` pin was pure provenance discipline. The scan
cannot: the reference contracts k and q against the state with a candle gemm and
normalizes q/k with a candle reduce, and the kernel partitions both across threads —
that is the point. Reassociating an f32 sum is not something a kernel can undo. So
`XWEN_DELTA_CLASSIC=1` is now pinned on BOTH sides of the strict tier (with the fused
scan on, strict stops being a bitwise tier), and a `delta` provenance field
(parity_schema v6, grandfather `classic`) proves which path each dump ran. The cached
reference dumps were written at v5 and grandfather correctly — every reference in both
gate runs was reused, no regeneration.

The other three kernels ARE bitwise, using block-scope `fp contract(off)` /
`reassociate(off)` so the scan stays free to contract into fma — its two inner loops
are the entire prefill cost, and file-scope pragmas (the sibling glue files'
convention) would have doubled their instruction count.

**Parity: ALL PASS on both checkpoints.** 35B-A3B — strict 1.000000, mm 0.999631
(was 0.999540 pre-change: slightly *better*), decode 63/64, 63/64, 62/64 agreements
with 1, 1 and 2 excused near-ties and zero non-excused mismatches, ppl Δnll 0.000791.
27B — strict 1.000000, mm 1.000000, decode 64/64 on all three fixtures with zero
excusals, ppl Δnll 0.000330.

The 35B result reproduced **three times across two independent builds**, matching to
every digit each time — twice here, and once accidentally on the parity owner's build
(an unguarded `import.meta.main` in parity-gate.ts fired a whole four-tier run as an
import side effect, since fixed). The fused kernels are deterministic: same dumps, same
cosines, same agreement counts, same Δnll, regardless of who compiled them.

The one number that moved in the wrong direction is perplexity, and it is worth stating
precisely because it is the only place the kernel's fidelity cost is visible at all.
The 35B's ppl delta went 0.000511 → 0.000791 and the 27B's 0.000221 → 0.000330 —
+55% and +49%, proportionally the same on two different architectures. And the SIGN is
systematic: the candidate is worse (higher NLL) in all four measurements, so this is
bias, not symmetric rounding noise. Everything else about the two candidates was
identical, which makes the attribution clean — that is a real, measured cost of the
fused scan, still comfortably inside the 0.002 bound.

The floor stays at 0.002 (parity owner's call, and the right one): `max(3 × measured,
0.002)` is a one-time floor-SETTING heuristic anchored to the reference-scan baseline,
not an invariant to maintain against whatever the candidate currently measures.
Re-fitting it to 0.000791 would widen the bound to fit the change under test, and a
bound re-fitted to each new implementation ratchets outward forever and catches
nothing. So the constant deliberately no longer reproduces from `3 × measured` — it is
tighter and more sensitive than the recipe would now give. Trip-wire: from 0.000791, a
further ~2.5x rise fails the gate. Since the 35B mm cosine went the OTHER way
(0.999540 → 0.999631), perplexity — not cosine — is the number to watch on further
DeltaNet kernel work.

The 35B's long-mixed fixture also picked up two near-tie excusals it did
not have before (margins 0.0161 and 0.1391 — both would clear even the standard 0.5
window), which is exactly where you would expect them: long-mixed is the fixture that
carries the DeltaNet recurrence over 600+ tokens.

**Greedy output is not preserved at longer prompts, by construction.** At a 596-token
prompt, fused and classic produce byte-identical 128-token greedy continuations on both
models. At 1929 tokens the 35B shares 69 words and then forks at a near-tie; the 27B
stays identical at both lengths (dense, no router, far less tie-prone). This is the
expected consequence of reassociated f32 sums, not a kill-switch bug — and it is why
the decode tier grades against the llama.cpp-anchored oracle with a near-tie rule
rather than against the previous build.

**One deliberate carve-out.** A `seq > 1` chunk under an armed rollback checkpoint
stays on the reference scan: the one-dispatch scan can only report the state after the
LAST token, and an armed layer needs one per token for the trail. Single tokens still
take the fused path even when armed, since their only state IS the final one — so spec
decode's per-token verify steps keep the win and only a batched verify forward pays.

**Tests.** 660 lib pass / 2 known-red (the dflash `real_file` pair, P9's), 63 parity
pass (was 60). Eleven new: four per-kernel tests at both shipped geometries (16/48 and
16/32 at head dim 128) with the conv and beta/decay ones asserting `f32::to_bits`
equality against the reference chain, a no-mutation and a streaming test for the scan,
shape/geometry rejection, offset-view handling, a block-level fused-vs-reference test
that grades the kernels as a package, and three parity rejection tests for the new
`delta` pin. The scan test was mutation-checked: flipping the K-head mapping from tiled
to interleaved moves its relative L2 from ~1e-6 to 1.37.

**Hardening follow-up, 2026-07-29.** Guards, an assertion mechanism, and doc
corrections on top of the kernels above. Nothing here moves a computed value; the
kernel math, barriers and indexing were re-read and stand.

- *Geometry, asserted twice over.* The scan's threadgroup shape lived as `#define`s in
  delta.metal (which index the state slice) and as independent Rust constants in
  dispatch.rs (which size the grid), with nothing tying the two together — drift meant
  silent out-of-bounds device writes. delta.metal now carries `static_assert`s for the
  three relations, and `scan_geometry_matches_metal` parses the `#define`s out of the
  source and compares them against dispatch.rs's copies.
- *Simd width.* Both the scan (`red[2][DELTA_D/32]`, indexed by simdgroup index) and the
  gated norm assume 32-wide simdgroups. `check_delta_simd_width` reads
  `threadExecutionWidth` at pipeline setup, so a device that ever differed fails at load
  instead of quietly folding the wrong lanes.
- *Empty chunks.* `delta_ba` and `delta_gnorm` accepted `seq == 0` and encoded a
  zero-dimension grid; they now bail like `delta_conv` and `delta_scan` already did.
- *Provenance is observed, not assumed.* `LinearAttnBlock::forward` also falls back to
  the reference scan on a non-128 head dim, a non-Metal device, or an armed multi-token
  chunk — none of which the environment shows — so an env-derived `delta` field could
  stamp "fused" on a dump that never dispatched a delta kernel, and the bounded tiers
  would grade the reference against itself and pass on nothing. Two `AtomicU64`s now
  count layer forwards down each path (`linear_attn::delta_path_counts`), and
  logits-dump derives the field from them, refusing to write a dump whose observed path
  contradicts the environment or splits across both. The field's value vocabulary is
  unchanged, so parity_schema and the gate's checks are untouched.
- *Docs that overclaimed.* TWO of the four kernels are bitwise, not three: the gated
  norm reassociates its sum of squares through `simd_sum` and grades at 2e-6, in the
  same class as the scan. The `math_mode(fast)` pragma pins the math mode but NOT the
  fast-vs-precise math-function compile option (this source compiles with no
  `MTLCompileOptions`), so what holds the intrinsics to candle's rounding is the pair of
  on-device bitwise tests, not the pragma. `ba_matches_reference_bitwise`'s decay
  assertion exponentiates through candle on both sides, so it grades `g` — not the
  scan's fast-math `exp`. And docs/parity.md's manual env table omitted
  `XWEN_DELTA_CLASSIC=1` from the reference and strict-candidate rows, so the documented
  by-hand procedure would have failed the tier the script passes.

One new lib test (`scan_geometry_matches_metal`), taking the delta family to nine, plus
four empty-chunk rejections folded into `shape_and_geometry_errors`; 63 parity
unchanged, since the `delta` field's value vocabulary did not move. The `static_assert`s
were mutation-checked out of band: halving `DELTA_TG_COLS` fails two of the three at
compile time.

## 2026-07-28 — Sampler tail: 0.82 → 0.41 ms/token, by moving the softmax off the CPU

**Context.** The per-token sampling tail was suspected of costing multiple milliseconds
of the ~16.9 ms decode budget. The bench that would have shown it (`sampler_decode_bench`
in moe.rs) was still carrying laguna's shapes — hidden 3072, vocab 100352, top-k 10,
47 MoE layers — so it had never measured Qwen geometry. Fixed first: the whole
`decode_bench` constant block now reads 35B-A3B (hidden 2048, expert_ff 512, top-k 8,
vocab 248320, 40 MoE layers), and the tiled `[VOCAB, HIDDEN]` synthetic tables tile at
512 rows because 1024 does not divide 248320.

**Measured, then fixed.** Baseline at real width: 0.819 ms/token for the whole draw, of
which 0.204 ms is the GPU→CPU copy of the 993 KB logit row and 0.615 ms is CPU work.
The CPU work was candle `LogitsProcessor::sample_topk_topp`: a full-vocabulary
temperature divide, a full softmax (three more passes plus a 993 KB allocation), a
`to_vec1`, and a `select_nth_unstable_by` over 248320 indices with an indirect
comparator. Micro-benched in isolation, the exp pass alone is 0.347 ms and the
`select_nth` 0.270 ms — together they are the tail.

**Change.** sampler.rs no longer uses `LogitsProcessor`; it owns its RNG and its
filtering. Two things moved. The full-vocabulary softmax now runs on whatever device
holds the logits — one Metal kernel instead of 248320 CPU `expf` calls — and the draw
reads back probabilities instead of logits, so the number of bus crossings is
unchanged at one. The candidate set comes from a streaming top-k (one comparison per
entry against the running k-th best, insertion on the order of `k·ln(n/k)` times)
instead of `select_nth` over an index vector. After: 0.406 ms/token, CPU work 0.206 ms.
0.41 ms/token back, ~2.4% of the budget, ~1.5 tok/s at the 59 tok/s baseline.

**The op-order finding.** candle's `TopKThenTopP` is temperature → full softmax → top-k
→ top-p, and its top-p cut is measured against FULL-vocabulary probabilities: it
compares the running cumulative mass of the top-k survivors to `top_p` without
renormalizing them first, and skips the cut entirely when the top-k mass is already at
or below `top_p`. llama.cpp and HF transformers both do the opposite — truncate to k,
renormalize over the survivors, then cut. The two disagree whenever the top-k set does
not hold nearly all the mass. This rewrite deliberately preserves candle's order, so no
distribution changed; it is why the fast path still needs a full-vocabulary softmax
rather than a 20-wide one. The divergence from llama.cpp is a ledger item, not a fix
made in passing.

**What did change: seeded token streams.** candle's candidate list came out of
`select_nth_unstable_by` in unspecified order; this one is sorted by descending
probability. A weighted draw maps its single uniform through the cumulative weights, so
the same uniform now lands on a different token. The distribution is identical, the
seeded sequence is not. Nothing depended on it — the parity gate is greedy end to end,
and the argmax path is untouched (still a CPU first-maximal scan). Pinned by two new
tests: the candidate set is compared id-for-id and weight-for-weight against a literal
transcription of candle's filtering, and 40000 draws are compared against the real
`LogitsProcessor` as a live oracle.

**Verdict.** The bench measures Qwen now, which was the precondition for any of this.
The remaining 0.41 ms is 0.20 ms of readback (mostly command-buffer sync, not copy) and
0.11 ms of streaming top-k; taking the selection to the GPU so only k values cross the
bus is the next lever, and a ledger item.

**Review follow-up, 2026-07-29.** Two reviewers went over the replacement. Nothing they
found moves the distribution on a well-formed row; three of the four are about rows that
are not well-formed, and the fourth is a claim the tests were making too broadly.

- *The padded tail was drawable.* The sampler read the whole logit width, but the output
  layer is wider than the tokenizer (248320 rows against 248070 encodable ids) and the
  rows between them decode to nothing. A padded row winning a slot puts a textless id
  into the stream. `Sampler::new` now takes the encodable bound — `tok.vocab_size()` at
  the one production construction site, never a literal — validates it against the width
  of every row it is handed, and narrows the row to it before anything else runs. The
  narrow is a view, so this is cheaper than what it replaced rather than an extra pass,
  and it keeps the padding out of the softmax denominator as well as out of the
  selection, which is what leaves the two softmax backends (device on the fast path, CPU
  on the controlled one) looking at identical values.
- *NaN was being skipped.* NaN loses every ordered comparison, so the argmax scan walked
  past a corrupt row and returned the best of the survivors — a silent answer where
  `LogitsProcessor` had at least pinned index 0. Every path now errors on a NaN, greedy
  included, which is the one the parity gate runs. `-inf` is not corruption (it is how
  the controls exclude an id) and stays skippable. The streaming top-k catches its NaNs
  inside the branch it already takes for a genuine improvement, so the hot loop still
  costs one comparison per entry.
- *The tie-break contract was overstated.* `top_k_desc`'s strict `>` against the floor
  gives lowest-id-wins at an exact top-k boundary tie; candle's `select_nth_unstable_by`
  leaves the same case unspecified. The determinism is the better contract — it is what
  makes the candidate set a function of the probabilities rather than of the traversal —
  so it is now claimed as such: distribution equivalence for untied inputs, deterministic
  low-id selection at exact boundary ties, and a test pinning the tie behavior itself
  without reference to what candle does with it.
- *Equivalence was mostly tested against a transcription.* The broad oracle is a
  hand-copy of candle's filtering, so a shared misreading would pass; only one narrow
  case ran against the real `LogitsProcessor`. That comparison is now a matrix — widths
  64 / 2048 / the checkpoint's own 248320, k of 1 / 20 / 64, top_p of 0.5 / 0.95 / 1.0,
  over flat, peaked and exactly-tied rows, 81 cases, ~4 s in release. Untied rows assert
  containment (candle must never draw an id this sampler excluded) plus matching
  frequencies; tied rows assert only eligibility, since candle's answer there is
  unspecified. Mutation-checked: shifting the top-p threshold by 30% and flipping the
  tie-break direction each fail it.

Five new lib tests, 666 passing. The one behavior change a user could observe is the
padded tail no longer being drawable, which was never a legitimate outcome.

## 2026-07-28 (later still) — Second-model review: one finding, and it was our own docstring overclaiming

**Context.** The external-model reviewer (GPT-5.6 via codex, ~35 min, xhigh) ran over
the full retarget alongside the two Claude reviewers. It cleared every model-math trap
independently (interleaved gate split, partial rope, rollback indexing, router order,
chunk carry) and missed the tool-call parser bug the integration reviewer caught — the
two reviewer families found disjoint issues, which is the argument for running both.

**Change.** Its one finding (MEDIUM): `kv_rollback`'s docstring promised caches
"byte-identical to ones that only ever advanced over the committed tokens" — but the
q8/f16 dual-storage split makes cache-mutating projections partition-dependent
(a verify batch of 9+ tokens runs the f16 plane where one-token decode runs q8 GEMV),
so the counterfactual-identity half of that promise is false in exactly the
speculative case it describes; the parity gate cannot see it because it runs one fixed
partitioning. Resolution: the mechanism is kept (the bytes-halving is why dual storage
exists), the CONTRACT was corrected — restore is a bit-exact replay of recorded bytes,
cross-partition agreement is numeric and parity-gated, `XWEN_ATTN_DEQUANT` pins one
representation when partition-independence matters. decisions.md records it under
Kernel policy; drift magnitude at the 8↔9 boundary is a ledger item, unmeasured.

**Verdict.** No code behavior changed; a claimed identity was demoted to the observed
one, per this project's own first rule. Review pass complete: three reviewers, one
critical fix (tool-call parser), one contract correction, model math clean.

## 2026-07-28 (late night) — Serve integration fixes: the tool-call parser was reading `:` and `;` as span markers

**Context.** An adversarial review of the inherited serve/ tree against the Qwen
retarget turned up four integration defects. One is severe and had no chance of being
caught by the suite that covered it.

**The headline: `serve/engine.rs` opened tool-call spans on ordinary punctuation.** The
span parser carried laguna's token ids as literals, `TOOL_CALL_OPEN: u32 = 25` and
`TOOL_CALL_CLOSE: u32 = 26`. In Qwen's vocabulary 25 is `:` and 26 is `;`; the real
`<tool_call>` pair is 248058/248059, and `tokenizer.rs` has held those constants since
the fork. So for any request carrying tools, every colon the model wrote in prose opened
a span and every semicolon closed one. What followed a colon stopped being answer text
and started being parsed as a call — delivered as a fabricated tool call if it happened
to parse, silently discarded by the heal path if it did not, which is to say the reply
truncated at the first colon. Genuine `<tool_call>` tokens meanwhile fell through to the
`_` arm and reached the client as literal text. The interior grammar was laguna's too:
`<arg_key>`/`<arg_value>`, strings that are not in Qwen's vocabulary and that chat.rs
has never emitted — so even a correctly-framed span parsed to nothing.

**Why the suite missed it.** There were seventeen tests over this parser and they all
passed. Every one of them scripted the token stream by hand as `(TOOL_CALL_OPEN,
"<tool_call>")` — the constant paired with the text a correct constant would decode to.
That pairing is the bug, asserted as a fixture. The tests agreed with the parser about
what id 25 meant, and neither had ever been asked what the tokenizer thought. A test
that builds its input from the same wrong constant as the code cannot fail when the
constant is wrong; it can only fail when the code stops being self-consistent. The fix
is not more assertions, it is a different input source, so the tests now drive the
emitter over ids produced by the real embedded tokenizer — round-tripping conversations
that `chat.rs` rendered, and one hostile case that feeds prose full of `:`, `;` and
`<function=` text and asserts zero calls with byte-identical output. Under the old
constants that case produces a tool call named `name`.

**Change.** The ids come from `LagunaTokenizer` now; the interior parser reads what
chat.rs writes (`<function=NAME>`, `<parameter=KEY>\nVALUE\n</parameter>`,
`</function>`, one function per span, framing newlines stripped from values). Two
behaviors changed on purpose rather than by translation. `</tool_call>` is structural
wherever it lands, mid-value included — the template writes a literal one inside an
argument as content so it never encodes to the added token, and treating the token as
content is what let a malformed value swallow the rest of a reply. And a span that never
names a callable tool now degrades: raw text, markers included, to the client as answer
text with a logged warning (`ServeLog::ToolSpanDegraded`, counted separately from
`healed` in the per-request report). Never discard, never fabricate.

**Three smaller items.** Drafting was on by default, inherited from laguna, which made
every zero-flag `xwen generate` and `xwen serve` abort at startup — `xwen serve` before
the listener bound. The error turns out to be `missing GGUF key dflash.decoder_arch`
rather than the `decoder_arch == "laguna"` mismatch the review predicted: the shipped
sidecars have no such key, so the failure precedes the check adaptation was expected to
repoint, with `enc.aux_norm` and `blk.N.attn_gate` absent behind it. Default flipped to
off; asking for a drafter still fails loudly. `serve/config.rs` was sizing caches with
laguna's geometry — 48 KiB/token from 12 full layers × 8 KV heads × 128 head_dim, and a
72 MiB snapshot described as copies of 36 SWA rings, in a model with no SWA layer at
all. Real figures are 20 KiB/token (35B-A3B) and 64 KiB (27B), with snapshots of
DeltaNet recurrent state at a fixed 62.8/149.6 MiB; they are derived on `hub::Model` now
rather than carried as constants. And `scan_banned` was protecting the compile-time EOG
set while the decode loop stopped on the GGUF-derived one — harmless today, since the
two match on both checkpoints, but only by coincidence.

**Verdict.** 645 lib tests pass, up from 642. Two pre-existing failures remain, both
`dflash::tests::real_file_*`, both the unadapted-drafter story above; they cannot pass
until TODO.md P9 lands and were failing before this arc. The tool-call parser is the
first inherited subsystem found to be not merely unadapted but actively wrong on Qwen
input, and the way it survived — tests built from the same constant as the code — is
worth carrying into the review of every other inherited dialect layer.

## 2026-07-28 (late night) — Parity harness live (P7): both checkpoints match upstream llama.cpp, floors an order of magnitude tighter than laguna's

**Context.** Everything before this rested on cited source lines, hand-computed unit
tests, and one greedy eyeball. The engine had never been compared against another
implementation on the same weights, and the 27B had never been run at all. P7 was the
item standing between "it produces fluent text" and "it computes the right thing".

**Change.** Upstream `ggml-org/llama.cpp` shallow-cloned into `reference/llama.cpp`
and PINNED at `e9fa0781f1c25fc4fe8c86be1edc6970661ad6f0` (recorded in docs/parity.md).
This session materialized it as a plain clone and gitignored the path, reasoning that
a submodule puts the reference tree in the index and the index is the human's review
surface; a `.gitmodules` entry declaring the same path as a submodule was staged
concurrently from elsewhere, and the owner settled it that night in favour of the
submodule — the gitlink makes the oracle sha reviewable in the diff, so moving the pin
becomes a staged change someone approves, which is the property the pin exists to
have. `scripts/build-llamacpp.sh` retargeted at the path.
`tests/fixtures/parity-prompts.json` regenerated with Qwen ids from the oracle's own
`llama-tokenize --no-bos`; the SWA-specific `long-swa` fixture replaced by
`long-mixed`, 612 tokens of prose that stresses the DeltaNet recurrence instead of a
sliding window that does not exist here. `hf.ts` repointed at the two ggml-org repos
with a size selector; `parity-gate.ts` gained `--model-size 27b|35b` and namespaces
every artifact by checkpoint basename (two official files, two architectures, two
sets of floors — they must never share a parity dir or a frozen ppl fixture).
`ref-dump.sh` retargeted. `scripts/parity.ts` learned the tap-name mapping
(`refTapNames`) rather than renaming engine taps. Laguna's `reference-ppl.json`
deleted; the committed-fixture test now validates every per-checkpoint fixture.

**Two latent parser bugs in the inherited `parity.ts`, both silent corrupters.**
First, the node-header regex captured names as `(\S+)`, so headers with spaces
(`cache_r_l0 (reshaped)`, `(view)`) were skipped and their value rows were attributed
to the previous node — which kept that node's `sum` but replaced its sampled row with
an unrelated tensor's. Symptom: `attn_norm-0` reporting `rowRelL2 = 2.29e+6` while its
values were in fact digit-identical to the oracle's. Second, `FLOAT_RE.test(line)` on a
shared `/g` regex advanced `lastIndex`, dropping every other value row. Neither would
ever produce a false PASS, but both produced convincing false divergences — the first
one cost a real detour before it was pinned down.

**Result.** Both checkpoints agree with upstream on identical GGUF weights. The final
four-tier run with the calibrated constants in place is `ALL PASS (6 graded)` on each —
42 s warm for the 35B, 2.0 min for the 27B, since the Reference dumps are cached and
only the candidates regenerate.

Track A (first-divergence bisection, 35B, code-short, 242 taps compared): no cliff.
The sampled-row rel-L2 profile is smooth and *flattens* rather than compounding —
`l_out` runs 1.8e-3 at layer 0, 1.2e-3 at 7, 2.4e-2 at 23, 1.4e-2 at 39 — and the
final-logits sampled cosine is 0.999995. Individual `sumRelErr` spikes up to 1.9e-1
are near-cancelling residual sums, not divergences; their own `rowRelL2` stays in the
neighbourhood's band.

Track B, `bun scripts/parity-gate.ts`, 35B-A3B Q4_K_M:

| tier | fixture | result |
|---|---|---|
| strict | code-short | PASS, cosine 0.999999861, top-1 = ref, top5 5/5 |
| mm | code-short | PASS, cosine 0.999539782, top-1 = ref, top5 5/5 |
| decode | code-short | PASS, 63/64 agree, 1 excused near-tie (0.0040 logit) |
| decode | text-mixed | PASS, 62/64 agree, 2 excused (0.5567, 0.2606) |
| decode | long-mixed | PASS, **64/64 agree**, 0 excused |
| ppl | — | PASS, Δmean_nll 0.000511 (fused 1.694170 vs reference 1.693659 over 4218 tokens) |

**The 27B's first forward was correct.** No bisection was needed: it loaded (18.2 GB
resident), prefilled 58 tokens in 9.6 s cold, and produced a top-5 that tracks the
35B's on the same prompt. Its parity numbers are *better* than the 35B's across the
board — strict bitwise 1.000000000, mm ≥ 0.999993294, decode **64/64 with zero excused
near-ties on all three fixtures**, ppl Δ 0.000221 (fused 1.748093 vs reference
1.747872) — with the caveat that on a dense model the
strict tier is near-vacuous, since `--moe-impl reference` and `fused` run the same
`DenseMlp` and the strict env pins everything else classic on both sides. The 27B's
real signal is the mm/decode tiers, which exercise the f16 attention path and the
fused glue.

**Floors, calibrated across both checkpoints and all three fixtures** (the constants
are global, so they are set under the WORST observed value): `COS_MIN_STRICT = 0.9998`
(worst achieved 0.999894, 35B long-mixed) and `COS_MIN_MM = 0.999` (worst achieved
0.999540, 35B code-short — ~1.7x the observed prompt-to-prompt spread). Both are an
order of magnitude tighter than laguna's 0.9955 / 0.985: these kernels track the
oracle much more closely on this architecture, largely because the Qwen Q4_K_M mix
keeps attention, ssm and shared-expert weights at q8_0.

**Verdict.** The engine is parity-validated. This is the checkpoint that makes every
later kernel change checkable instead of hopeful, and it is the gate P8's DeltaNet
kernels will be graded against. Four things are explicitly NOT proven and are in the
ledger: Track A cannot localize inside a layer (the tap set is still laguna's six —
no DeltaNet core, router logits, or shared-expert gate taps, which would need plumbing
in the model-math files); `provenance.flash` says `"fused"` while `flash.metal` is
compiled at head dim 128 and cannot serve Qwen's 256, so prefill is really candle sdpa
with a materialized mask; the dense strict tier as noted above; and the `_Q8` widened
bands were inherited, not recalibrated — measured worst per-step l2 deviation is
1.0211 against a 1.5 band (far too loose), while the near-tie window genuinely needed
its widened 1.0 (text-mixed step 15 excused at 0.5567).

## 2026-07-28 (night) — First real-weights decode: correct output, clean stop, 59 tok/s on the 35B-A3B

**Context.** Everything to this point rested on cited source lines and hand-computed
tests; nothing had touched real weights. The 20.4 GB ggml-org Q4_K_M finished
downloading and the release binary was fresh.

**Change.** None — this is the first execution of the P2-P6 stack:
`xwen generate -p "What is 2+2? Answer with just the number." --temp 0 --no-draft`.

**Result.** Correct end to end on the first attempt: a coherent five-step thinking
block, `</think>`, the answer `4`, and a natural stop on `<|im_end|>` (143 tokens
emitted with 369 of budget left — the stop list works). Load 2.8-3.0s via the mmap
path, 19.2 GB resident as predicted (weights 19.0 + KV 0.2 + recurrent state 0.1 at
max_ctx 8192). Warm numbers, power mode unmeasured: prefill 23 tokens at 167.6 tok/s,
decode 58.6-59.0 tok/s. The 8.76s cold-run prefill was Metal pipeline compilation, not
the DeltaNet scan — the "will look like a hang" warning in the P2-P4 entry overstated
short-prompt cost; long-prompt prefill through the per-token scan remains unmeasured
and P8 still owns it. Coherence at temp 0 is strong indirect evidence for the whole
trap set (tiled V-heads, norm baking, gate split, router order): any of those wrong
produces fluent garbage, not correct arithmetic under a formatting constraint.

**Verdict.** The engine runs Qwen 3.6 correctly by the greedy-eyeball half of the
fallback gate. Not yet parity-validated — P7 is unchanged as the next honest
checkpoint. Decode at ~59 tok/s pre-kernel-work is 2.7x laguna's decode on 2.7x
fewer active params, i.e. exactly the bandwidth story, and the DeltaNet reference's
~240 extra dispatches per token are not yet visibly the bottleneck at this length.

## 2026-07-28 (evening) — ChatML port, tokenizer swap, and a latent constrain-width bug (P5+P6): suite fully green, 642/0

**Context.** The fork still rendered Laguna's angle-bracket template over Qwen's vocab,
tokenizer.rs carried Laguna ids, and two tests failed on vocab assumptions. hub/CLI
still resolved poolside checkpoints.

**Change.** chat.rs rewritten against the vendored chat_template.jinja (ChatML, tools
preamble, XML-ish call format with string-args-raw, collapsed tool-result turns, the
last_query_index thinking-retention rule, open-`<think>` generation seeding), keeping
laguna's content/structure separation; typed `ChatError` mirroring the template's raise
cases plus one deliberate divergence (tool-result-first refused — the template emits a
malformed boundary there). tokenizer.rs repointed at Qwen ids and made the single owner
of every token id, exposing both vocab sizes (248070 encodable / 248320 logit width).
hub.rs got a per-model checkpoint table (`--model-size 27b|35b`, 35b default,
API-verified filenames); sampling defaults aligned to 1.0/20/0.95. The constrain trie
is now sized to the logit width — which surfaced and fixed a latent bug where every
constrained serve request would have died on a 248,096-bit mask against 248,320 logits.
Control-token safety under grammars turned out to rest on toktrie's angle-bracket
heuristic rather than the special flag; documented and pinned by a full-range sweep
test. Design calls recorded in decisions.md "Tokenization, chat, tool calls".

**Result.** 662 lib tests: 642 pass, 0 fail, 20 ignored (perf benches). The five
template vectors are byte-exact; a differential harness ran 20,000 fuzzed conversations
plus exhaustive role-shape and Unicode-whitespace sweeps against the reference jinja
with zero divergences. Adversarial review by a second model family found two real bugs
pre-merge (silent second-system drop; trim whitespace set), both fixed.

**Verdict.** The prompt/token surface is done and independently verified. What remains
before trusting the whole stack is a forward pass against real weights — the 35B
Q4_K_M was downloading as this entry was written.

## 2026-07-28 — Model core retargeted to Qwen 3.6 (P2-P4): config, loader, DeltaNet reference, attention and MoE

**Context.** The mechanical fork built green but still computed Laguna: 48 uniform
attention layers, a sliding-window ring every fourth layer, a softplus per-head output
gate, and a sigmoid/bias/scale MoE router. P2-P4 replaced that core wholesale with the
Qwen 3.6 hybrid, on the critical path for everything downstream. Two research agents ran
alongside the implementation — one extracting llama.cpp master's `qwen35.cpp` /
`qwen35moe.cpp` / `delta-net-base.cpp` graphs, one range-parsing the shipped ggml-org
GGUF headers — so every load-bearing form was written first from the CLAUDE.md cheat
sheet and then confirmed or corrected against a cited source line.

**Change.** `LagunaConfig`/`LagunaModel` → `XwenConfig`/`XwenModel`. config.rs parses
both archs, rejects anything else, and derives per-layer `LayerKind::{Full, Linear}`
from the `full_attention_interval` key rather than a hardcoded 4. New `linear_attn.rs`
holds the gated-DeltaNet layer as a frozen oracle in the `ReferenceExperts` sense:
composed candle ops, recurrent form only, fp32 state, one sequential scan step per
token. attention.rs gained the double-width `attn_q` with its per-head interleaved gate
(strided split, not a halving of the row), QK-RMSNorm over 256 dims, partial NEoX rope
at n_rot 64 / theta 1e7, sdpa scale 1/√256, and an ELEMENTWISE `sigmoid(gate)` — 4096
independent values per token, not one scalar per head. moe.rs swapped the router for
softmax-over-all-256 → top-8 → renormalize with the f16 floor, no selection bias and no
weight scale, and gave the shared expert its scalar sigmoid gate. kv_cache.rs grew a
third `LayerCache::Linear` variant carrying conv window and delta state in f32 through
checkpoint, rollback, snapshot and the on-disk framing; the SWA ring machinery is left
in place but nothing on the model path constructs it. model.rs dispatches per layer and
loads the pre-MLP norm from `post_attention_norm`. The mmap/no-copy loader, ExpertStack,
QLinear and the dual-storage attention planes were not touched — only the name table.

**Result.** `cargo build` and `cargo test --no-run` green. 659 lib tests: 637 pass, 20
ignored, 2 fail — `generate::ban_scan_catches_every_em_dash_token` and
`constrain::walks_a_valid_document_and_completes`, both in tokenizer/constrain code this
arc never touched, and the second is the already-logged `<think>`-is-not-special trap.
23 tests are new, including the three the recurrence needed: the delta rule walked
against a scalar f64 reimplementation of the update equations at head dim 2 over three
tokens, the gated-norm ordering pinned with a non-uniform weight and opposite-sign gate
factors, and conv-state continuity checked by feeding seven tokens one at a time versus
one batch. Six corrections came out of the ground-truth pass: no fused `ssm_ba` (two
separate tensors), `ssm_conv1d.weight` is 2-D not 3-D, `full_attention_interval` exists
and should be read, `value_length` exists and should be asserted square, neither
`eot_token_id` nor `eom_token_id` exists so the second stop id is a named constant, and
the L2 norm is ggml's clamp form rather than HF's rsqrt form. The tiled-versus-
interleaved k-head broadcast — the one assumption that could have silently corrupted
every DeltaNet layer — came back tiled, as the cheat sheet said, traced to
`ggml_compute_forward_repeat_f32`'s destination index.

One finding is a real regression rather than a correction: **the vendored flash kernel
is unreachable on this architecture.** `flash.metal` is compiled at `BD == 128` and
Qwen 3.6 is head dim 256, so prefill falls back to candle's sdpa with a materialized
`[1, n_head, seq, k_seq]` f16 mask — precisely the allocation laguna's flash path was
written to avoid. Correct but slower, and now a ledger item alongside the deleted
attention benches and the DeltaNet rollback trail's ~1 GB verify-walk cost.

**Verdict.** The engine computes Qwen 3.6. Nothing has been run against real weights
yet, and it should not be mistaken for validated: the fallback gate (reference unit
tests plus a greedy eyeball) has only had its first half satisfied, and prefill through
a per-token sequential scan will make that first real load slow enough to be
mistaken for a hang. The next honest checkpoint is P7's parity harness; until it lands,
every claim here rests on cited source lines and hand-computed tests, not on a number
measured against llama.cpp.

## 2026-07-28 — Fork bootstrap: laguna mapped, Qwen 3.6 architecture pinned down, mechanical fork started

**Context.** xwen is a manual fork of ../laguna (crate `maxuna`, ~72k lines: candle+Metal
GGUF inference engine for poolside Laguna S 2.1) retargeted at Qwen3.6-27B and
Qwen3.6-35B-A3B. Bootstrap ran as five parallel agent workstreams: laguna codebase map,
Qwen 3.6 architecture research, GGUF header survey (range-request parsing of ggml-org
files, no downloads), llama.cpp reference extraction, and the cp-based mechanical fork.

**Findings that set the design.** Qwen 3.6 is not Qwen3 — it is the Qwen3-Next-derived
hybrid: 3 gated-DeltaNet linear-attention layers per full-attention layer (full at
indices 3,7,11,…), sigmoid-gated attention output fused into a double-width q_proj
(per-head interleaved [q_h, gate_h]), QK-RMSNorm over head_dim 256, partial RoPE (n_rot
64 of 256, theta 1e7; IMROPE in llama.cpp, but provably identical to NEoX rope for
text-only), MoE with 256 experts / top-8 / softmax-then-renorm plus a sigmoid-gated
shared expert (35B), MTP head shipped as sidecar GGUFs. candle has zero support for any
of this — the DeltaNet recurrence is new code. Full config/tensor tables live in
CLAUDE.md's cheat sheet; conversion traps (norm +1 baking, tiled V-heads, ssm_a =
-exp(A_log), no ffn_norm, no ssm_in) are recorded in decisions.md "Weights and loading".

**The dflash reversal.** The plan dropped dflash.rs as Laguna-specific (a diffusion
drafter bound to a poolside checkpoint). The GGUF survey then found ggml-org publishes
DFlash sidecar drafters for both Qwen 3.6 models under the same `dflash` architecture —
the subsystem is portable after all. The removal was cancelled mid-flight; the fork
keeps dflash.rs and all drafter wiring, with Qwen adaptation tracked as its own TODO
item. Lesson, same one laguna's decisions.md preaches: check the artifact before
deleting the code that consumes it.

**Fork state.** Mechanical fork (cp-based copy, maxuna→xwen rename, MAXUNA_*→XWEN_* env
prefix, Qwen tokenizer/template vendored into reference/) running as this entry is
written; build gate is `cargo build` on the unmodified-logic tree. Docs
(this file, decisions.md, parity.md, CLAUDE.md, TODO.md) written fresh, mirroring
laguna's documentation system.

**Verdict.** Research phase complete with high confidence on every load-bearing fact
(all numbers read from shipped GGUF headers, llama.cpp master source, and live HF
repos, not from memory or blogs). Implementation fan-out next: config/loader, DeltaNet
reference, attention/MoE adaptation, chat.rs, parity harness — in that order.
