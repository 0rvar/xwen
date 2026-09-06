# 2026-09-06 — The fused hc gate is +57-76% on 2..8-token forwards, and the duplicate-dispatch probe learns decode mode: the shared expert floors at 0.43 ms, the router projection overlaps itself

Moved verbatim from [docs/log.md](../log.md) on 2026-09-06; the log keeps this entry's
opening paragraph and links here.


Two questions the 2026-09-05 entries below left open. The fused hyper-connection gate
also fires at 2..8 tokens, and every A/B of it so far was a single-token decode; the
duplicate-dispatch probe priced prefill only, because `ops::dup` returned early at one
token. Both are answered here. Neither run changed model math: run 1 is a switch A/B on
cf7c579, run 2 adds a probe stage and a probe opt-in (d5daa18) and nothing else.

**Protocol, both runs.** A pinned binary built in a detached worktree under /tmp
(cf7c579 for run 1, d5daa18 for run 2), so no `cargo build` in the main tree could swap
the binary or its `include_str!` kernels under a running harness. Arms interleaved with
the order reversed each round, three rounds, 60 s idle between rounds, medians reported.
`pmset -g` printed `lowpowermode 0` at the start and at the end of both sessions. The
machine was otherwise idle, with no second model process.

### Run 1 — the fused gate at 2..8 tokens is the largest win it has

`XWEN_HC_GATE_FUSED_MAX_N` admits the fused gate up to 8 rows inclusive, but every
measurement of it was at n = 1, and kernel A re-stages the carrier row per token per
threadgroup, so its throughput in that window was unknown and could plausibly have been
a regression. The A/B forces every prefill forward to exactly n tokens with
`XWEN_PREFILL_CHUNK=n` over the 880-token `tests/fixtures/bench-prompts/prefill-925.txt`
fixture, which makes the reported prefill tok/s the n-token-forward throughput itself; a
serve batch of n concurrent decodes does the same per-row gate work. Arms: fused (the
default) against `XWEN_HC_GATE_FUSED_MAX_N=1`, which leaves n = 1 fused and sends 2..8
down the split seven-dispatch path. `xwen generate --no-draft --raw -n 8`, log
/tmp/ceil/ragged-ab.log.

| chunk | fused tok/s (rounds) | median | split tok/s (rounds) | median | fused/split |
|---|---|---|---|---|---|
| 8 | 149.7 145.2 149.9 | **149.7** | 93.6 93.2 93.2 | **93.2** | **+61%** |
| 4 | 109.6 108.9 108.8 | **108.9** | 69.7 69.5 69.5 | **69.5** | **+57%** |
| 2 | 68.8 68.1 66.3 | **68.1** | 38.6 38.6 38.7 | **38.6** | **+76%** |

Not a regression, and a larger win than the +9% the same two kernels buy at one token.
The mechanism is the path they displace rather than the fusion arithmetic: the hc planes
are dense_mm-only (`QLinear::without_mv_ext`), so in the 2..8 window the split path runs
both bottleneck gemms through candle `QMatMul`'s tile matmul kernel, which is a gemm
shape with half-precision activation tiles at a gemv-sized problem. The fused kernels
replace it with two wide-grid gemv-style launches. This is the window whose numerics
decisions.md records as deliberately changed, and the change now has a throughput result
behind it. Decode item 1's sub-item (e) is closed as measured.

**One observation, ledgered rather than explained.** The 8-token decode tail that follows
each ragged prefill read lower on every fused arm (47.9-52.1 tok/s) than on every split
arm (55.4-57.6), across all nine pairs, even though n = 1 decode runs the identical fused
gate on both arms: the ceiling only moves the 2..8 window. A 128-token recheck at chunk 8
(/tmp/ceil/ragged-decode.log, two rounds) is NOT comparable and carries no reading, because
the fused arm's generation hit a stop token after 11 tokens in both rounds, at different
tokens each time; it reads 43.8 and 43.9 tok/s over those 11 tokens against 50.0 and 50.8
over 128 on the split arm. Yesterday's 128-token n = 1 A/B (51.2 against 47.0) stays the
decode measurement. The open question is whether anything about a fused-gate prefill leaves
the first decode steps slower, and a recheck needs a prompt that cannot stop early or a
token budget that ignores stop ids. TODO.md decode item 1, sub-item (f).

### Run 2 — the probe in decode mode, and what a decode delta means

Two gaps in the instrument. The MoE router projection (`route_logits`, a candle f32
matmul `[1,2560]x[2560,512]` reading 5.24 MB of F32 weight per layer, 251 MB per token)
sat one line BEFORE the `moe_glue` wrapper and was therefore inside no stage at all, so
yesterday's prices never included it and it belonged to the 38% unpriced. And `ops::dup`
returned early at n == 1, so no decode stage could be priced by it. Commit d5daa18 adds a
`router_proj` stage covering exactly that projection and nothing else (`moe_glue` keeps
its recorded coverage) and an `XWEN_DUP_DECODE` opt-in that lets the probe repeat
single-token calls.

`xwen generate --no-draft -n 128` on the 596-token `decode-630` fixture,
`XWEN_DUP_DECODE=1 XWEN_DUP_STAGE=<stage>`, three rounds. Log /tmp/ceil/dup-decode.log.

| arm (stage duplicated once) | decode tok/s (rounds) | median | ms/token | delta vs base | prefill median (596 tok) |
|---|---|---|---|---|---|
| base | 50.1 51.2 50.9 | 50.9 | 19.65 | | 1167.2 |
| `router_proj` | 50.8 51.0 50.8 | 50.8 | 19.69 | +0.04 ms | 1157.0 |
| `shexp` | 49.7 49.8 50.0 | 49.8 | 20.08 | **+0.43 ms (2.2%)** | 1169.7 |
| `moe_glue` | 51.0 51.0 50.8 | 51.0 | 19.61 | 0 | 1048.6 |

**What a decode delta is, and this is the part to carry forward.** At decode a duplicated
launch has no buffer hazard against its original: it writes a fresh output from the same
inputs, so candle's encoder inserts no barrier and the GPU is free to run the two
concurrently. A stage that leaves the GPU idle prices at about zero. So in decode mode a
delta above zero is a FLOOR for that stage, and a delta of about zero means "this stage
overlaps with itself", never "this stage is free". Prefill carries the same caveat, but at
decode it is the common case rather than the corner one, which is why the probe cannot
price a latency-bound decode stage at all.

Read that way, the router projection and the glue kernels overlap fully. The shared expert
does not, because its five launches per layer (gate, up, silu*mul, down, gate logit; 240
per token) are a dependent chain, and it floors at 0.43 ms per token, 2.2%, against the
0.77-0.96 ms the per-launch budget of 3.2-4 µs gives 240 launches. The `moe_glue` prefill
delta at 596 tokens is 0.058 s, which is what yesterday's 0.40 s at 3851 tokens scales to
(0.062 s expected), a consistency check on the instrument across two prompt lengths.

The router projection is therefore still UNPRICED at decode. What the run does say about it
is that it runs at low occupancy, since its copy was free, and that is the gemv hypothesis
exactly: a single-row mlx gemm over a 5.2 MB plane. Its price is the A/B that replaces it
with a wide-grid gemv at n <= 8 (candle `QMatMul` over the F32 `ffn_gate_inp` tensor,
reaching ggml's `kernel_mul_mv_f32_f32`), which is now the experiment. No number is claimed
for it here.

The same reading refutes, by argument rather than by measurement, the ledger's own idea of
folding the router projection into `kernel_moe_router`: that kernel is one threadgroup per
token, which is the shape `kernel_hc_norm_inject` lost 6% of decode with (decisions.md
"Below 32 tokens the hc norm splits"), so a fold would stream 5.2 MB of router weight on a
single core. The gemv replacement is the correct form of that lever. The other MoE decode
experiment is the shared expert's five launches folded to one with the down gemv in the
epilogue, which is −4 per layer and −192 per token, and it is in progress.
