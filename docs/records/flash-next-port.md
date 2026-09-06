# 2026-08-29 — Qwen3.8-Flash-Next runs: P0 through P2 in one day, agreeing with llama.cpp at 189/192 forced-replay steps, 37.5-38.1 tok/s decode

Moved verbatim from [docs/log.md](../log.md) on 2026-09-06; the log keeps this entry's
opening paragraph and links here.


**Context.** The qwen4exp arc had been paused since 2026-08-26 with P0 complete and no
runnable weights. Three things unblocked it at once: Unsloth published the full quant
ladder, llama.cpp merged qwen4exp support, and the machine had disk free. The day ran
P1 and P2 end to end. `docs/qwen4exp-port.md` is the arc doc; the decisions moved to
decisions.md "Qwen3.8-Flash-Next (qwen4exp)" as this entry was written.

**Picking a file.** Header-parsing every published GGUF turned up a rule nobody had
written down: `ffn_down_exps` is `[640, 2560, 512]`, and 640 % 256 = 128, so it fails
every K/IQ type's block-size requirement and llama.cpp's generic `tensor_type_fallback()`
silently demotes that plane to a 32-block type on every publisher's file. Q4_K becomes
Q5_0, Q5_K becomes Q5_1, Q6_K becomes Q8_0. `per_layer_token_embd` (ncols 160) is
32-block-only for the same reason, which is why ggml-org's Q8_0 file carries a Q4_0 PLE
table. `UD-Q4_K_XL` was chosen — 111.33 GB, 4 shards, 82.53 GB of trunk — as the only
Q4-class file whose types we already have kernels for, with the consequence that 43 of
its 48 layers carry Q5_1 experts. That looked like a blocker for half a day and was not:
`ExpertStack` types each tensor from its own GGUF info with no whitelist, nothing
compares dtypes across planes or layers, decode reaches candle's baked
`kernel_mul_mv_id_q5_1_f32`, and prefill drops the layer to per-token `mul_mv_id`.
Correct today, slower than it should be, and now a P3 ledger item rather than P2 scope.

**The oracle moved.** PR #27742 merged 2026-08-27 as `6c84c7d5d`, with a graph-splits
follow-up `6fe749801` the next day. The vendored copies under `reference/qwen4exp/` were
re-vendored at the new pin, and then deleted outright: the submodule was bumped
e9fa0781 → `6fe749801` instead, one oracle for every checkpoint. The proposed second
clone — keeping the old pin frozen so the 3.6/3.8 floors stayed valid — was refuted by
just re-running the gate: **all three existing checkpoints re-passed at the new pin the
same day**, floors unchanged and not re-derived (35B-A3B six graded checks, strict cos
1.000000, mm 0.999631, ppl Δnll 0.000791; 27B clean throughout; 3.8-27B five checks,
its ppl tier skipped because that checkpoint has never had a reference fixture — a gap
that has been shipping quietly and is now ledgered). The semantic diff of the vendored
move survives as `reference/qwen4exp/UPSTREAM-DIFF-2026-08-29.md`; the one finding in it
that changed our plans is that upstream had already moved PLE conv state onto its own
recurrent row, which refuted a note we had written about upstream wasting it across all
36 recurrent layers.

**P1: three frozen references, three reviewers.** `ref_hc`, `ref_ple` and `ref_qsa` —
hyper-connections plus both norm flavours, the n-gram hash plus its injection layer, the
QSA indexer — graded against five golden fixtures, 38 tests. The review round changed
real things: the grouped norm now accumulates in f64 like ggml's CPU path and is one
shared implementation instead of three near-copies with weaker asserts, every matvec
asserts its full shape, the PLE conv dilation is derived rather than loaded, and the
gate propagates NaN. The fixtures also settled a question the port doc had recorded
wrongly: HF DOES clamp inside the signed sqrt, so a "PLE gate clamp divergence" we had
written down was retracted.

**P2: eight units, and first light.** U0 loader-owned GGUF header parsing (the real file
had never opened — `gguf::open` failed with "unknown dtype for tensor 20" until this
landed; 1223 candle tensors plus one raw IQ4_NL plane), U1 registry, U2 device
hyper-connections, U3 the QSA indexer and the attention overlay, U4 PLE and IQ4_NL row
dequant, U5 the `ZGate`/`sum_floor` wiring, U6 the stack itself. Suite 957/957. First
smoke answered "The capital of France is **Paris**." and stopped cleanly; the second,
with thinking on, produced 400 coherent tokens — reasoning inside `<think>`, a clean
`</think>`, then a working Python function with a unittest.

U3 cost a day's worth of confusion to a candle bug worth reporting upstream: **Metal
`index_select` is silently wrong on strided sources.** No error, just wrong rows. Worked
around by gathering per head.

**U7: it agrees with the oracle.** The parity gate itself cannot run on this checkpoint
at all — every tier's reference side is `--moe-impl reference`, and
`ReferenceExperts::forward` panics at `src/moe.rs:198` with "the len is 512 but the
index is 1073971200". That index is `0x40038000`, the f32 bit pattern of 2.0547, so an
f32 buffer of routing data is reaching a `to_vec1::<u32>()` read as expert ids on the
512-expert / top-10 geometry. It reproduces identically through the fused router kernel
and the candle chain, so it is downstream of the router branch. The fused runner is
unaffected, which is what made the rest of the measurement possible.

So U7 rebuilt the decode tier by hand with llama.cpp standing in for the blocked
reference runner: `llama-server /completion` fed a token-id array (bypassing both the
chat template and llama.cpp's tokenizer, so both engines provably see the same
sequence), then xwen teacher-forced along the oracle's own trajectory with `--replay`.
Tokenization was verified first and is byte-identical over the whole 4218-token corpus,
first differing index −1. Result: **189/192 agreeing, zero hard mismatches**, the three
divergences being near-ties at 0.2876, 0.0348 and 0.0097 logit — far inside the
`NEAR_TIE_MARGIN_Q8` band, and in every case llama.cpp's pick was xwen's own rank-2.
Full-vocab logprobs against the oracle's `n_probs` matched at top-1 and top-5 in exact
order, first differing at rank ≥5 among entries at logprob −12 to −14. The perplexity
comparison is protocol-limited and deliberately not called a grade: 0.5413 nats for
llama.cpp's eight independent 512-token windows against 0.3697 for xwen's one continuous
4218-token context, a gap the protocols predict. Full record in
`docs/qwen4exp-parity-2026-08-29.md`.

**Review round.** Claude, Qwen and Fable reviewed the arc; Codex was out of credits.
Three findings mattered. **PLE rollback ignores commit** — a state-machine bug in the
new recurrent path, the class of thing that only shows up after a rewind. **Serve would
500** on a qwen4exp target, because the snapshot path cannot carry the new state, so
Flash-Next is now explicitly CLI-only and `xwen serve` refuses it until P4 — the honest
version of "serve integration follows CLI bring-up". And a **name-fold collision** in
the space↔hyphen identification added for this checkpoint (the file calls itself
"Qwen3.8 Flash Next").

**Numbers, with the usual caveats.** Plain decode 37.5-38.1 tok/s against llama.cpp's
40.9-41.5 on the same file in the same hour — within ~8%, unremarkable. Prefill 203.5
tok/s against 713.4 at 530 tokens: **3.5x slower, reproduced to the centisecond across
two runs**, so not pipeline compilation. Two suspects, both already ledgered: the 43
Q5_1-down layers prefilling through the per-token fallback, and the dense-FFN prefill
gemm that was exactly this shape of problem on the 27B and took a vendored kernel to
close. `lowpowermode 0` with no high-power claim, single runs, a machine shared with
other agents' builds all day, and llama.cpp thermal-boosts harder than xwen — the ratio
is the trustworthy part. One more finding worth understanding: xwen dirties ~15 GB of
private memory where llama-server dirties 751 MB on the identical file, i.e. ~15 GB of
weights materialized rather than aliased from the mapping.

**Verdict.** The graph is correct, on the evidence of three independent instruments, and
it is honestly slow in one specific place. P2 is closed and the arc pauses here. What it
is NOT: it has no drafter, no snapshots, no serve, no parity harness of its own, and no
perplexity floor — and the frozen ppl corpus looks contaminated for this checkpoint
(PPL 1.45 on WikiText-2 test where the 3.6 pair scores 1.69 nats, with llama.cpp
independently agreeing the model is that good), so it will need a fresh corpus before it
gets one. All of that is in TODO.md with dates and context, none of it silently dropped.
