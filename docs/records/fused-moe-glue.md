# 2026-07-29 — Fused MoE glue: an MoE layer goes from 24 dispatches per token to 14, and 35B decode from 92.6 to 102.8 tok/s

Moved verbatim from [docs/log.md](../log.md) on 2026-09-06; the log keeps this entry's
opening paragraph and links here.


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
