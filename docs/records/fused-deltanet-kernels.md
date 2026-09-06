# 2026-07-28 — Fused DeltaNet kernels: a layer goes from ~65 dispatches per token to 8, and 35B prefill from 305 to 2183 tok/s

Moved verbatim from [docs/log.md](../log.md) on 2026-09-06; the log keeps this entry's
opening paragraph and links here.


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
