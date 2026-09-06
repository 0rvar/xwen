# 2026-07-29 — DFlash adapted to the Qwen sidecars (P9): both drafters load and accept 85-95%, and speculation is a 27B-only win because the verify forward runs the per-token reference scan

Moved verbatim from [docs/log.md](../log.md) on 2026-09-06; the log keeps this entry's
opening paragraph and links here.


**Context.** The DFlash subsystem came over from laguna whole and inert: `dflash.rs`
described a six-layer, hidden-3072, 72-head drafter with a per-head softplus attention
gate and a per-tap encoder norm, gated behind a `dflash.decoder_arch == "laguna"` check.
The ggml-org sidecars for Qwen 3.6 are a different model — 5 layers at hidden 5120 (27B)
or 6 at 2048 (35B-A3B), 32 Q / 8 KV heads, no gate tensor, no `enc.aux_norm`, no
`decoder_arch` key at all, and sliding-window attention on every layer but the last. Two
tests (`real_file_load_and_shapes`, `real_file_bf16_alias_load_and_forward`) were the
suite's only red ones and asserted the laguna geometry on purpose, as the arc's gate.

**What the oracle actually says.** Three of the graph's differences were not in the
briefed list and were found by reading `reference/llama.cpp/src/models/dflash.cpp`
against the shipped headers:

- **The noise block is non-causal.** `common/speculative.cpp:1004` calls
  `llama_set_causal_attn(ctx_dft, false)`, and the causal branch of the KV mask builder
  is guarded by `if (causal)` (llama-kv-cache.cpp:1793). The drafter is a block-diffusion
  model: it denoises `[id_last, MASK × 15]` in one forward, so every block position sees
  every other in both directions. The inherited code masked the block causally.
- **The injection path applies no `attn_norm`.** dflash.cpp:252-253 projects the raw
  encoder output into `wk`/`wv`; `enc.output_norm` is the only norm on that path. The
  query path in `draft_forward` does apply `attn_norm`, so the two deliberately disagree.
  The inherited code normed both, citing the laguna branch — which no longer exists.
- **The encoder is three ops.** dflash.cpp:109-123: concatenate the taps tap-major,
  `fc`, `enc.output_norm`. No per-tap RMS-norm, no per-tap scale.

Everything else the mapping pass had established held up: taps are the residual
*entering* the named target block, so the `t - 1` translation to our `l_out` capture
points is unchanged; rope is plain NEoX over all 128 dims (no `rope.dimension_count` key
→ `n_rot = n_embd_head_k`), theta 1e7 from the GGUF; QK-norm before rope; SwiGLU with a
real `ffn_norm` tensor.

**Sliding windows, implemented as a narrow rather than a mask.** llama.cpp masks
`p1 - p0 >= n_swa`, i.e. a query at position `p` keeps `[p - window + 1, p]` on the past
side and — being non-causal — everything on the future side. The obvious implementation
is an additive mask over the full score row, but the block's windows are a contiguous
union: query 0 has the deepest floor, and the 16 queries' floors span at most 15
positions. `attention` therefore narrows the cache to `[lo, committed + n_block)` and
masks only the ≤15 columns between the individual floors, falling back to no mask at all
while the context still fits inside the window. A windowed layer costs O(window) per
round instead of O(context), which retires half of the argument behind
`DEFAULT_DRAFT_CTX` being small: only the final full-attention layer — one of five or six
— still grows with depth.

**Measurements** (`lowpowermode 0`, warm, greedy, 128 decoded tokens, interleaved arms
within one process-per-run sweep, 3 reps, medians; scripts in the session scratchpad,
`scripts/spec-equivalence.ts` committed):

| | plain | `--draft` (p_min 0.3) | acceptance |
|---|---|---|---|
| 27B, code prompt | 25.0-25.2 tok/s | 26.4-26.7 (+4.8 to +6.8%) | 87.4% |
| 27B, chat prompt | 20.6-21.5 tok/s | 20.9-23.1 (+1.5 to +7.4%) | 65.9-73.7% |
| 35B-A3B, code prompt | 105.1 tok/s | 93.0 (-11.5%) | 81.3% |
| 35B-A3B, chat prompt | 105.5 tok/s | 92.1 (-12.7%) | 82.8% |

The 27B rows are ranges across two independent interleaved runs rather than one run's
median, because that model's run-to-run spread is wide enough to matter (docs/decisions.md
"Measurement discipline", and the 27B caveat under TODO P11): within a run the reps are
tight — 26.9/26.7/26.7 against 24.8/25/25 on the code prompt — but the level shifts
between runs. The sign is stable; the magnitude is ±2 points. The 35B rows repeated to
within 1%.

The drafter proposes well on both models — 85-95% acceptance at `p_min` 0.5-0.9, and a
27B run at `p_min` 0.9 accepted 54 of 54. The 35B's loss is not a drafting failure.

**The 35B loses ~12% before it drafts anything.** An arm with `--draft-p-min 1.1`, where
no drafter token can ever clear the threshold and 119 of 127 rounds pause, still decodes
at 92.6-92.7 tok/s against 105.1-105.5 plain — the same loss as the best drafting arm.
The cost is the drafter's per-round cache sync: every committed token runs `encode` (an
8-tap concat through a [2048, 16384] `fc`) plus six layers of `wk`/`wv` projections,
QK-norm, rope and two `slice_set`s, about 14 small Metal dispatches for ~1.2 ms. That is
12% of a 9.5 ms plain step and 2.8% of the 27B's 43 ms one, which is the whole difference
between the two models' verdicts. It is dispatch-bound, not FLOP-bound — the same disease
the MoE glue fusion cured — and it is mandatory while a drafter is attached, because a
drafter whose cache falls out of step with the target's can never resume speculating.

**The verify forward gets almost no batching win, which is the real ceiling.** Under an
armed rollback trail a multi-token chunk falls back to the frozen reference scan
(linear_attn.rs:194-205), and that scan walks tokens one at a time in candle ops. So the
layers that are 48 of the 27B's 64 and 30 of the 35B's 40 cost the same per position in a
16-token verify as in 16 single-token steps: measured 245 ms for a ~6-position verify on
the 27B against a 43 ms plain step, i.e. 39 ms per verified position. Speculative decoding
is a bet that verifying N tokens costs far less than decoding them one at a time; on this
architecture that bet currently pays only in the attention and FFN layers. **The
K-snapshot fused verify is therefore not an optimization of P9 but the precondition for
it** — TODO.md P9 carries it with the structural note that both scan kernels already hold
each thread's state slice in registers across the timestep loop, so emitting per-token
snapshots is one guarded store plus a wider output buffer.

**Tuning.** `draft_p_min` swept over {0.2, 0.3, 0.5, 0.7, 0.9} on the 27B: 0.3 is the only
value that came out ahead of plain on both prompt kinds in every run (0.5 lost on the chat
prompt twice), so the default moves 0.5 → 0.3 in `SpecParams`, the CLI and the serve
config. Lower `p_min` drafts longer at lower acceptance, and that wins precisely because
the verify cost is near-linear in its span — with no batching penalty to pay, a longer
span amortizes the round's fixed cost. That reasoning is fitted to the reference-scan cost
curve and should be re-run when the fused verify lands. `pause_margin` stays 1.0: the
controller earns it on the 27B, where always-drafting (`--draft-pause-margin 0`) measured
21.8 tok/s against the controller's 25.8 and plain's 23.3. It cannot help the 35B, whose
loss is charged to rounds the controller has already paused.

**`--draft` stays opt-in.** The flip to opt-out was conditional on the controller holding
a never-materially-slower property on both models. It does not: no setting of `p_min` or
`pause_margin` can recover a 12% loss that is incurred on paused rounds. The CLI, serve
config and `--init` template text all lose their "not adapted yet, fails at load" wording
and gain the measured reason instead.

**Equivalence, in two modes.** `scripts/spec-equivalence.ts` diffs `--draft` against
`--no-draft`. Greedy mode (temperature 0) checks the verify walk's token selection; it
found 11 of 12 comparisons byte-identical, the twelfth forking on the 27B chat prompt at an
adjective ("accessible" vs "educational"). That is the batched verify forward reassociating
its f32 sums differently from the single-token forward and flipping a near-tie — the same
class the decode parity tier's near-tie rule grades, and the same one already recorded for
the fused delta scan under P8a.

Greedy mode has a structural blind spot, though, raised by the second-family review: at
temperature 0 the argmax path never draws from the seeded RNG, so no amount of greedy
agreement can show that the spec loop advances the SAMPLER STREAM the same number of times
the plain loop does — one extra or missing draw would reroute every subsequent token. A
`sampled` mode now covers it: temperature 0.8 at a fixed seed with `--draft-p-min 0` and
`--draft-pause-margin 0`, so every round drafts a full block and nothing pauses (auto-pause
is what makes a temperature>0 run irreproducible from a seed, so it has to be off).
Result: the 35B is byte-identical on both prompts with 360 and 435 drafted tokens over 384
and 464 verified positions, and the 27B is identical on the code prompt with 315 drafted.
**The sampler stream is in lockstep.** The 27B chat prompt forks in sampled mode too, at
line 12 — deep enough that the stream was demonstrably in step for over a hundred tokens,
which is the near-tie signature rather than the desync one (a desync reroutes the first
sampled token, and the script says so when a sampled-mode fork lands on line 1).

Both modes now also refuse to report OK on a run that drafted nothing: a comparison that
paused into plain decoding exercises no verify and no rollback, so its agreement means
nothing, and the script says NO COVERAGE instead. It rebuilds before comparing and checks
the binary is not older than the newest source, so it cannot bless a stale build.

**Deleted, added, changed.** Deleted: the `decoder_arch` requirement, the `enc.aux_norm`
tensor and its per-tap norm-and-scale, the `attn_gate` tensor and its softplus output
gate, the `softplus` helper, and the within-block causal mask. Added: `sliding_window` +
`swa_layers` on `DflashConfig` with a `layer_window(il)` accessor, `value_bool`, the
narrow-plus-mask windowed attention, `Model::draft_kv_bytes_per_token()` (40 KiB/token on
the 27B's five layers, 48 on the 35B's six — `serve/config.rs`'s hardcoded 35B-only
constant now derives from it), and `scripts/spec-equivalence.ts`. `attach_drafter` also
gained the mismatched-pairing check the CLI path never needed while every drafter load
failed: nothing in a sidecar's metadata names its target, so `--draft <path>` can pair the
27B drafter with the 35B, which used to reach `set_spec_taps`'s `assert!` and panic. It is
now an error naming both numbers, and the hidden-size mismatch is caught at attach rather
than at the first forward. Review caught that serve had the same hole one level earlier:
`check_draft_geometry` compared head counts but not hidden size, and the tap check alone
does not separate the two sidecars in one direction — the 35B-A3B drafter's translated
taps top out at 37, inside the 27B's 64 layers — so `xwen serve --model-size 27b --draft
<35B drafter>` passed startup validation and failed the first job instead. The check is
now threaded through `read_draft_config` from the target's `XwenConfig`, which is what
`validate_model` exists to do. Tests: the two red ones
rewritten against real sidecar geometry and parameterized over both models (the alias test
now injects 2100 positions, past the 27B's 2048 window, so the narrow-plus-mask path runs
on real weights), plus three new ones — a windowed forward graded against the scalar
reference and against an unwindowed twin, a perturbation test proving the last block row
informs the first, and a config test for the window keys. `ops/bf16.rs`'s `DRAFTER_SHAPES`
covers both sidecars' twelve production matmul shapes. Suite: **760 passing, 0 failing**,
the two deliberately-red tests among them.
