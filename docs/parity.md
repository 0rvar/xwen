# Parity verification

Proves the engine computes what upstream llama.cpp computes for the identical GGUF.
This file is the single authoritative location for tiers, tap names, floors, and
runbook commands — other docs point here and do not restate the numbers.

## The oracle

Upstream `ggml-org/llama.cpp` master, NOT a vendored fork: Qwen 3.6 support is
in-tree (`src/models/qwen35.cpp`, `qwen35moe.cpp`, `delta-net-base.cpp`). The
checkout lives at `reference/llama.cpp` as a git SUBMODULE (shallow; see
.gitmodules) — the gitlink makes the oracle sha reviewable in the diff, and moving
the pin shows up as a staged change the owner must approve. Only
`reference/llama.cpp/build` is gitignored.

**Pinned commit: `6fe74980162af0ed5e559870d5deccafaa034e7c`** (2026-08-28,
"model: qwen4exp: reduce number of graph splits (#27880)") — bumped 2026-08-29
to get qwen4exp support into the one oracle. Previous pin
`e9fa0781f1c25fc4fe8c86be1edc6970661ad6f0` (2026-07-28, "model: Add
Laguna-S-2.1 LLM_TYPE (#26233)"), cloned and built 2026-07-28, where every floor
below was calibrated.

Re-confirmation at the new pin, 2026-08-29 — **the floors hold for the 3.6
pair; nothing was re-measured or moved.**

- **Qwen3.6-35B-A3B: ALL PASS**, six graded checks. strict cos 1.000000, top5
  5/5; mm cos 0.999631; decode 63/64, 63/64, 62/64 agreeing with 1/1/2 excused
  and 0 mismatch; ppl Δnll 0.000791.
- **Qwen3.6-27B: ALL PASS.** strict cos 1.000000; mm cos 1.000000; decode 64/64
  three times, 0 excused; ppl Δnll 0.000243.
- **Qwen3.8-27B: run in progress at time of writing — result not yet recorded.**
  Until this line says otherwise, the bump is confirmed for the 3.6 pair only.

```bash
just init                               # git submodule update --init --recursive
bash scripts/build-llamacpp.sh          # cmake from an ephemeral nix shell, system CLT SDK
```

Moving the pin is a deliberate act, not a `git pull`: a different oracle build can
move the achieved cosines, so re-record the sha here and re-run the calibration in
"Floors" below. `scripts/build-llamacpp.sh` documents the fetch/checkout pair.

Binaries the harness uses, all in `reference/llama.cpp/build/bin`:
`llama-eval-callback` (Track A), `llama-tokenize` (fixture ids + cross-check),
`llama-server` (raw greedy oracle), `llama-cli` (chat-wrapped continuation only —
see "Raw greedy oracle").

## Two tracks

**Track A — first-divergence bisection (`scripts/parity.ts` vs
`llama-eval-callback`).** eval-callback runs the real graph and prints, for every
node, a full-tensor `sum` plus first-3/last-3 sampled values. That is enough to walk
the graph in execution order and find the *first* node where our intermediate math
drifts. It is not enough for a real full-vector cosine on the logits.

**Track B — full-logit gate (`tests/parity.rs`, dump vs dump).** Two `logits-dump`
JSON files in the same schema compared on the full last-position logit vector:
cosine, top-1, top-5, plus scale and finiteness guards.

Neither track needs the other: run Track A to *locate* a bug, Track B to *gate* a
release.

## The dump format (`logits-dump`)

`src/bin/logits-dump.rs` feeds raw token ids through one forward pass and writes
JSON. Unchanged from the Laguna original except for what the model itself supplies
(vocab 248320, the Qwen tap set, `attn_decode: "q8"` on both official files).

```jsonc
{
  "model": "…/Qwen3.6-35B-A3B-Q4_K_M.gguf",
  "prompt": "def fibonacci(n):",     // provenance only; may be null. The tool never tokenizes.
  "moe_impl": "reference",           // reference (oracle) or fused
  "provenance": { … },               // how the dump was produced — see "Provenance"
  "tokens": [727, 73111, …],         // input token ids (u32); NO BOS, ever
  "n_tokens": 58,
  "vocab": 248320,
  "logits": [ …vocab f32… ],         // FULL last-position logits
  "top1": 248044,
  "top5": [[248044, 21.02], …],      // (token_id, logit), descending
  "taps": [
    {
      "name": "attn_norm-0",         // OUR tap name — see the mapping table below
      "shape": [58, 2048],           // candle dims, outer..inner (last dim = feature)
      "sum": -112.58,                // whole-tensor sum — comparable to eval-callback `sum`
      "mean": …, "std": …, "l2": …,
      "first8": [ … ],               // first 8 of the last-position row
      "last_row": [ …feature f32… ] | null   // full last-position row; null above 16384 elems
    }
  ]
}
```

Why "last-position row": eval-callback truncates every tensor to first-3/last-3
along each dim, and its last printed row is the last token. So the last-position
feature vector is the one row comparable in detail on both sides.

Vocab note: 248320 is the PADDED logits width; real tokens end at 248076. The
padding rows are part of the compared vector on both sides (llama.cpp computes them
too), so no masking is applied anywhere in the gate.

## Tap names

Our taps are the engine's own names; llama.cpp names its graph nodes with
`cb(tensor, name, il)` and eval-callback prints `"{name}-{il}"` for `il >= 0`, bare
`"{name}"` for `il == -1` (`src/llama-context.cpp:2469-2475`). `scripts/parity.ts`
translates between them (`refTapNames`); nothing in the engine is renamed.

| our tap | llama.cpp node | layers | what it is |
|---|---|---|---|
| `attn_norm-{il}` | `attn_norm-{il}` | all | pre-mixer RMSNorm output |
| `attn_o_proj-{il}` | `attn_output-{il}` | full-attn only (`(il+1) % 4 == 0`) | attention out after the sigmoid gate and `wo` |
| `attn_o_proj-{il}` | `linear_attn_out-{il}` | DeltaNet only | DeltaNet out after the gated norm and `ssm_out` |
| `ffn_inp-{il}` | `attn_residual-{il}` | all | `x + mixer(...)`, the residual feeding the FFN block |
| `ffn_norm-{il}` | `attn_post_norm-{il}` | all | the PRE-MLP norm (there is no `ffn_norm` tensor) |
| `ffn_out-{il}` | `ffn_out-{il}` | all | dense SwiGLU (27B) or `moe_out + gated shexp` (35B) |
| `l_out-{il}` | `l_out-{il}` | all | layer output (post-FFN residual) |
| `result_norm` | `result_norm` | — | final RMSNorm output at the last position |
| `result_output` | `result_output` | — | final logits |

**`attn_o_proj` must be resolved by layer kind, not by name presence.**
`attn_output-{il}` exists on a DeltaNet layer too, where it names the *pre-gate
DeltaNet core output* (`delta-net-base.cpp:552`) — a different tensor of a different
shape. Matching it there reports a bogus first divergence (measured: `sumRelErr`
0.95 on `attn_output-0`).

**`h_nextn` is deliberately NOT mapped.** Ours is the pre-final-norm residual
stream (the DFlash capture point); llama.cpp's `h_nextn` is the POST-final-norm
hidden state (`qwen35.cpp:211` cbs the norm's `MUL`). The post-norm value is
compared as `result_norm` instead.

Nodes llama.cpp emits that we have no tap for (so `parity.ts` skips them): the
attention internals `Qcur_full` / `Qcur_normed` / `Kcur_normed` / `gate_reshaped` /
`attn_pregate` / `gate_sigmoid` / `attn_gated`; the DeltaNet internals
`linear_attn_qkv_mixed`, `z`, `beta`/`beta_sigmoid`, `alpha`/`a_softplus`/`gate`,
`conv_states*`, `conv_output_raw`/`conv_output_silu`, `q_conv`/`k_conv`/`v_conv`
and their `*_predelta` forms, `state_predelta`, `new_state`, `final_output`; the MoE
internals `ffn_moe_logits` / `ffn_moe_probs` / `ffn_moe_topk` /
`ffn_moe_weights{,_sum,_sum_clamped,_norm}` / `ffn_moe_{gate,up,swiglu,down,weighted,out}`;
and the shared-expert set `ffn_shexp`, `shared_expert_gate`,
`shared_expert_gate_sigmoid`, `ffn_shexp_gated`. Adding taps for the DeltaNet core
output, the router logits and the shared-expert gate scalar would let Track A
localize *inside* a layer instead of only between layers; it needs tap plumbing in
`linear_attn.rs` / `moe.rs` and is a TODO.md ledger item, not a gap in the gate.

Two more llama.cpp facts worth knowing when reading a trace: the fused
`GATED_DELTA_NET` op is NOT cb'd (it prints as `node_N`), so the whole DeltaNet
recurrence is one opaque node; and `norm-{il}` names three or four *different*
nodes per layer (mixer norm, q-norm, k-norm, post norm) — only stream position
disambiguates them, which is why our tap set uses the scaled `attn_norm` instead.

## eval-callback output format

`common_debug_cb_eval` (`common/debug.cpp:171`) prints, per node:

```
common_debug_cb_eval:              attn_norm-0 = (f32)        MUL(norm-0{2048, 58, 1, 1}, blk.0.attn_norm.weight{2048, 1, 1, 1}}) = {2048, 58, 1, 1}
    [
        [
            [     -0.1063,       0.8386,      -1.9896,    ...,      -0.5064,      -0.2383,       1.3577  ],
            ...,
            [      0.6019,       0.1526,       1.4085,    ...,       1.5960,      -0.4132,      -0.3865  ],
        ],
    ]
    sum = -112.476746
```

- **Header:** `<fn>: <name> = (<dtype>) <OP>(<src0>{ne}, <src1>{ne}}) = {ne}`. `ne` is
  ggml order — `ne[0]` is the innermost/feature dim, `ne[1]` the token dim (the
  transpose of our candle `shape`).
- **`sum`** is over the *entire* tensor, computed before the truncated print, so it
  is a real full-tensor signal. It is the backbone of the divergence walk.
- **Values are truncated** to first-3/last-3 along each dim (`n = 3`, hardcoded at
  `common/debug.cpp:186`). The final printed innermost row is the last position.
- **Token ids are echoed.** eval-callback prints `number of input tokens = N`
  followed by N id lines; `ref-dump.sh` extracts them, so our dump runs on the
  identical sequence.
- Everything goes to stdout+stderr via the common logger — capture with `2>&1`.

Two parser traps, both found and fixed in `scripts/parity.ts` on 2026-07-28 (they
were latent in the inherited Laguna version and each silently corrupted the walk):

1. **Node names contain spaces and parentheses** (`cache_r_l0 (reshaped)`,
   `(view)`). A `(\S+)` name capture drops those headers, and every value row under
   them is then attributed to the PREVIOUS node — which keeps that node's `sum`
   (first-wins) but replaces its sampled row with an unrelated tensor's. Symptom:
   `attn_norm-0` reporting `rowRelL2 = 2.29e+6` while its `sum` was fine and its
   values were in fact digit-identical to ours.
2. **`FLOAT_RE.test(line)` on a shared `/g` regex** advances `lastIndex`, so every
   other value row was skipped and `lastRowSamples` ended up on an arbitrary earlier
   row. The row test now goes through `parseFloats` alone.

## Runbook

Prerequisites: the oracle is built (above) and the checkpoint is in the Hugging Face
cache (`xwen fetch --model-size 27b|35b`, or `bun scripts/hf.ts model 27b` to print
the path).

**The one-command path: `scripts/parity-gate.ts`.** It produces the Reference dumps,
produces the Fused candidate dumps, and runs every tiered gate:

```bash
bun scripts/parity-gate.ts                                   # 35B, all tiers
bun scripts/parity-gate.ts --model-size 27b                  # 27B dense
bun scripts/parity-gate.ts --tiers strict,mm                 # just the full-logit gates
bun scripts/parity-gate.ts --tiers decode --fixtures long-mixed
bun scripts/parity-gate.ts --regen-ref                       # rebuild the Reference dumps too
bun scripts/parity-gate.ts --tiers ppl --regen-ppl-ref       # re-freeze the committed ppl reference
```

Flags: `--tiers strict,mm,decode,ppl` (default all); `--fixtures
code-short,text-mixed,long-mixed` (default all; strict/mm always grade code-short,
decode grades all three, ppl has no fixture axis); `--model-size 27b|35b` (default
`35b`, matching the CLI) or `--model <path>` / `$XWEN_MODEL` for a file not in the
hub (mutually exclusive); `--regen-ref`; `--regen-ppl-ref`; `--parity-dir DIR`;
the experiment hooks `--sdpa-f32` / `--attn-mm-classic` / `--flash-classic`; and
`--expect-attn-decode f16|q8` (default `q8` — both ggml-org Q4_K_M files store
attention weights q8_0).

**Everything is namespaced by checkpoint basename.** The parity dir defaults to
`/tmp/xwen-parity-<basename>` and the frozen ppl reference to
`tests/fixtures/reference-ppl-<basename>.json`. The two official checkpoints are
different architectures with different floors; their artifacts must never mix.

The script enforces the model-run hazards: it `pgrep`s for a running model process
before every model invocation, runs strictly serial (ONE large process at a time),
streams all model output to log files under the parity dir, and never pipes through
a pager. Candidate dumps are always regenerated (the thing under test); Reference
dumps are reused when their provenance proves the pinned oracle environment. It
prints a per-tier PASS/FAIL summary with the key metric and exits nonzero on any
failure.

The preflight exits 3 and names the offending process — including your own
`xwen generate` in another terminal, which is the common case. It deliberately does
NOT kill anything.

**The preflight tests `argv[0]`, not the command line.** `pgrep -f` matches the whole
argv, so any wrapper that merely QUOTES one of these commands — an `until …; do sleep;
done; ./target/release/logits-dump …` waiter, a `zsh -c` one-liner, even
`git diff -- src/bin/logits-dump.rs` — used to abort a run over a model process that
did not exist. Both agents working this repo hit it within an hour of each other, so
it is a footgun rather than a curiosity. A wrapper's `argv[0]` is `/bin/zsh`, so
matching on the executable actually being run rejects it structurally instead of
guessing at shell syntax. `isModelProcess` is exported and unit-tested offline against
captured `pgrep -fl` lines, both the real processes it must catch and the wrappers it
must ignore. (`pgrep -x` gets the same structural property and does work here —
measured, `pgrep -x bun` returns 3 against `pgrep -f bun`'s 15. argv[0] is preferred
only because it is a pure function of a `pgrep -fl` line, so the matcher is testable
offline against captured incidents, which a `pgrep` invocation is not.)

**`main()` is behind `import.meta.main`.** Without it, importing anything from
`parity-gate.ts` runs the entire gate as an import side effect — ~40 s of model time
warm, and cold it would launch 20 GB loads nobody asked for. Found by doing exactly
that while unit-testing the matcher.

**Which cargo commands are safe while a gate is running:** `cargo test --lib` /
`--no-run --lib` do NOT relink the binaries and are safe. `cargo test --test <name>`,
`--tests`, and bare `cargo test` DO pull in the bin targets and relink
`target/release/{xwen,logits-dump}` under a running gate, splitting its dumps across
two builds. Verified by mtime on both sides. Treat the shared `target/` dir as part of
whoever holds the GPU, not just the GPU itself.

The manual per-step commands below are what the script automates; they remain the
reference and the fallback.

**1. Produce the reference side** (per prompt). `ref-dump.sh` runs eval-callback,
extracts the authoritative token ids, cross-checks against `llama-tokenize`, and
optionally greedy-decodes:

```bash
scripts/ref-dump.sh -m "$(bun scripts/hf.ts model)" --fixture code-short -o /tmp/ref-code --gen 24
scripts/ref-dump.sh -m "$(bun scripts/hf.ts model 27b)" -p "def fib(n):" -o /tmp/ref-27b
```

Outputs in the out dir: `eval-callback.txt` (the trace, ~5 MB for a 58-token prompt
on the 35B), `tokens.txt` (authoritative ids, comma-separated), `llama-cli.txt`
(chat-wrapped continuation, with `--gen`), `ref-cmd.txt` (the exact next commands).

Fixture texts must NOT end in a newline: `ref-dump.sh` passes the text through a
shell command substitution, which strips trailing newlines, so a fixture ending in
one tokenizes to a different sequence than the oracle's own echo.

**2. Produce our side** on the identical ids:

```bash
./target/release/logits-dump \
  --model "$(bun scripts/hf.ts model)" \
  --tokens "$(cat /tmp/ref-code/tokens.txt)" \
  --taps \
  --output /tmp/ref-code/ours.json
```

**3a. Track A — bisection:**

```bash
bun scripts/parity.ts \
  --ours /tmp/ref-code/ours.json \
  --ref  /tmp/ref-code/eval-callback.txt \
  --report /tmp/ref-code/parity-report.json
```

Prints the first divergent node (with `sumRelErr` / `rowRelL2`), the final-logits sum
and a sampled cosine. `--threshold` defaults to rel error `1e-2`. The report JSON
carries the WHOLE walk under `comparisons`, in graph order — read that, not just
`firstDivergence`: what disqualifies is a cliff, not a single number over the line
(see "Pass criteria").

**3b. Track B — full-logit gate:** put `candidate.json` and `reference.json` in a
directory and run the tier matching how the candidate was produced:

```bash
# strict — candidate is the CLASSIC mv fallback path with legacy f32 attention
XWEN_PARITY_DIR=/tmp/ref-code XWEN_PARITY_TIER=strict \
  cargo test --release --test parity logit_parity -- --exact --ignored --nocapture
# mm — candidate is the default tiled mm_id prefill path
XWEN_PARITY_DIR=/tmp/ref-code XWEN_PARITY_TIER=mm \
  cargo test --release --test parity logit_parity -- --exact --ignored --nocapture
```

Env combinations that define each side (what `parity-gate.ts` sets for you):

| side | env |
|---|---|
| Reference (all tiers) | `XWEN_ATTN_F32=1 XWEN_ATTN_MM_CLASSIC=1 XWEN_COMBINE_CLASSIC=1 XWEN_ATTN_GLUE_CLASSIC=1 XWEN_FLASH_CLASSIC=1 XWEN_ACT_CLASSIC=1 XWEN_DELTA_CLASSIC=1 XWEN_DENSE_MM_CLASSIC=1 XWEN_MOE_GLUE_CLASSIC=1`, `--moe-impl reference` |
| strict candidate | the first eight, PLUS `XWEN_NO_MM_ID=1 XWEN_MV_CLASSIC=1`, `--moe-impl fused` |
| mm / decode / ppl candidate | none (the shipped defaults), `--moe-impl fused` |

`XWEN_DELTA_CLASSIC=1` and `XWEN_DENSE_MM_CLASSIC=1` are the load-bearing ones in that
list — the gate checks both as provenance on both sides of the strict tier (see
"Provenance pins"), so a manual dump that omits either fails the tier rather than
merely shifting numbers.

`XWEN_MOE_GLUE_CLASSIC=1` is the opposite case and is deliberately NOT on the strict
candidate: the fused MoE router and block epilogue are bit-identical to the candle
chains, so the strict tier grades the FUSED glue against a reference that ran the
unfused one, and that is the point. Regenerating the reference under this pin and
re-running the gate returns `cos=1.000000 top5=5/5` with every other tier's number
unchanged, which is the full-model confirmation of the ops-level bitwise tests. A
reference cached before the pin existed is still valid for the same reason (2026-07-29).

**3c. Decode gate — greedy agreement under forced replay:**

```bash
DIR=/tmp/decode-code-short; mkdir -p "$DIR"
TOKENS="$(bun -e 'const j=await Bun.file("tests/fixtures/parity-prompts.json").json();
  console.log(j.prompts.find(p=>p.id==="code-short").tokens.join(","))')"

# the reference side runs under the oracle env of the 3b table, in full:
XWEN_ATTN_F32=1 XWEN_ATTN_MM_CLASSIC=1 XWEN_COMBINE_CLASSIC=1 XWEN_ATTN_GLUE_CLASSIC=1 \
XWEN_FLASH_CLASSIC=1 XWEN_ACT_CLASSIC=1 XWEN_DELTA_CLASSIC=1 XWEN_DENSE_MM_CLASSIC=1 \
XWEN_MOE_GLUE_CLASSIC=1 ./target/release/logits-dump --model "$M" --moe-impl reference \
  --tokens "$TOKENS" --greedy 64 --output "$DIR/reference-greedy.json"
# the candidate runs the shipped defaults — no env
./target/release/logits-dump --model "$M" --moe-impl fused \
  --replay "$DIR/reference-greedy.json" --output "$DIR/candidate-greedy.json"

XWEN_PARITY_DIR="$DIR" XWEN_PARITY_TIER=decode \
  cargo test --release --test parity greedy_parity -- --exact --ignored --nocapture
```

The reference free-runs greedy N tokens; the candidate is teacher-forced along that
exact sequence, recording its OWN argmax at each step before being forced. Free-run
comparison is useless here — the moment the two engines pick different tokens their
histories diverge and every later position is incomparable.

## Pass criteria

**Track A**: what disqualifies is a divergence *cliff* — one node whose deviation
jumps orders of magnitude above its layer neighbourhood. Smooth monotonic drift
across layers is expected candle-Metal vs ggml-Metal kernel noise on identical
Q4_K_M weights.

Measured drift profile, 35B-A3B, code-short, shipped fused path vs the pinned oracle
(2026-07-28) — `rowRelL2` of the sampled last-position row:

| tap | L0 | L3 | L7 | L15 | L23 | L31 | L39 |
|---|---|---|---|---|---|---|---|
| `attn_norm` | 8.1e-5 | 1.1e-4 | 1.1e-3 | 2.2e-2 | 1.7e-2 | 1.8e-2 | 1.0e-2 |
| `attn_residual` | 5.0e-3 | 2.2e-3 | 2.8e-3 | 2.8e-2 | 2.7e-2 | 2.1e-2 | 1.0e-2 |
| `ffn_out` | 5.9e-3 | 8.9e-3 | 3.3e-3 | 9.1e-3 | 1.9e-2 | 2.9e-2 | 1.8e-2 |
| `l_out` | 1.8e-3 | 2.3e-3 | 1.2e-3 | 1.3e-2 | 2.4e-2 | 2.2e-2 | 1.4e-2 |

Smooth, no cliff, and it flattens rather than compounding. `sumRelErr` is noisier
(single values up to ~1.9e-1 at `attn_output-23` and ~1.5e-1 at `l_out-31`) because
these residual sums nearly cancel — judge a sum outlier against its own
neighbourhood and against the same node's `rowRelL2`, never alone. Final-logits
sampled cosine on the same run: **0.999995**.

Same measurement on the 27B dense (386 taps compared over 64 layers) is flatter still —
the whole profile stays inside one band with no depth trend at all:

| tap | L0 | L3 | L15 | L31 | L47 | L63 |
|---|---|---|---|---|---|---|
| `attn_norm` | 3.3e-4 | 8.9e-4 | 6.9e-4 | 8.1e-4 | 1.0e-3 | 2.7e-4 |
| `attn_residual` | 1.6e-3 | 1.6e-3 | 6.4e-4 | 7.5e-4 | 1.2e-3 | 5.1e-4 |
| `ffn_out` | 3.9e-3 | 2.2e-3 | 1.4e-3 | 1.5e-3 | 1.0e-3 | 7.6e-4 |
| `l_out` | 1.2e-3 | 2.7e-3 | 4.4e-4 | 7.6e-4 | 5.0e-4 | 4.9e-4 |

Worst `rowRelL2` anywhere in the 27B walk is 8.95e-3 (`ffn_out-2`); worst `sumRelErr`
is 1.69e-1 (`attn_post_norm-35`, again a cancelling sum whose own row error is
~1e-3). Final-logits sampled cosine 1.000000, `result_output` sum rel error 2.02e-3.
The dense model drifting less than the MoE one is expected: no router, so no
top-8 selection that a ulp-level difference could flip.

**Track B** is three tiers keyed by change kind via `XWEN_PARITY_TIER`:

- **strict** (classic fallback path, `XWEN_NO_MM_ID=1 XWEN_MV_CLASSIC=1
  XWEN_ATTN_F32=1` + the classic pins): cosine at or above the strict floor, top-1
  agreement, top-5 overlap `>= 4/5`. Gates the legacy kill-switch mat-vec path,
  whose per-(token,slot) matvec accumulates in the same order as the oracle's
  per-row matmul, so it tracks the oracle tightly with near-zero headroom.
- **mm** (the shipped tiled prefill): cosine at or above the mm floor, top-5 overlap
  `>= 4/5`, and top-1 matches the Reference OR the candidate's top-1 is the
  Reference's top-1/top-2 while the Reference's own top-1/top-2 margin is `< 0.5`
  logit (a genuine near-tie). mm_id sums over K in 8x8 simdgroup tiles — a different
  but equally valid f32 accumulation ORDER — so it drifts further than strict.
- **decode** (the shipped decode kernels): greedy agreement vs the Reference oracle
  under teacher-forced replay, plus the perplexity-delta bound. Full-logit cosine is
  reported as a diagnostic only: every decode lever reorders f32 accumulation and
  strict passes with essentially zero headroom, so no correct decode change could
  clear the strict cosine.

**Scale-sensitive hard checks (every tier).** Cosine, top-1 and top-5 overlap are all
scale-INVARIANT — a uniform rescale sails through at cosine 1.0, and a NaN slips past
(`NaN < cos_min` is false). So the gate additionally hard-fails on any non-finite
candidate logit and on a candidate/reference L2-norm ratio outside
`[1/NORM_RATIO_MAX, NORM_RATIO_MAX]`.

**Provenance pins.** The tier is caller-selected with no cross-check against how the
dump was produced, so every dump carries a `provenance` object and the gate pins each
field per side and tier: `moe_impl`, `attn_dtype`, `attn_mm`, `attn_glue`, `sdpa`,
`flash`, `act`, `combine`, `attn_decode`, `delta`, `dense_mm`, `mv_ext`, plus
`seq_len`/`mm_min_seq`/`no_mm_id` for
"was the mm_id path actually active". A field missing at or after its introduction
version is a stale binary and hard-fails; `src/parity_schema.rs` is the single source
of truth for the version and the grandfather table. What the Qwen checkpoints report
on the shipped path: `attn_dtype f16`, `attn_mm tensor`, `attn_decode q8`,
`combine fused`, `attn_glue fused`, `sdpa f16`, `flash fused`, `act fused`,
`delta fused`, `dense_mm fused`, `mv_ext fused`, `mm_variant tensor`.

**`delta` and `dense_mm` are the two path pins that are load-bearing rather than
blessed-anchor discipline.** The other `*_CLASSIC` kernels are bit-identical to the candle chains
they replace, so pinning them on the reference side only anchors provenance. The
fused gated-DeltaNet scan is not: it partitions the k- and q-contractions across
threads and folds the q/k L2 norm through `simd_sum` where the reference runs a
candle gemm and a candle reduce, so it is bounded-close, not bitwise. Two
consequences. (1) `XWEN_DELTA_CLASSIC=1` is pinned on BOTH sides of the **strict**
tier — with the fused scan on, strict is not a bitwise tier at all. (2) The oracle
must never run it, or the bounded tiers would compare the kernels to themselves; the
`delta` field is what proves it did not. Introduced at schema version 6 with
grandfather `classic`, so every cached pre-v6 reference dump stays valid without
regeneration (no binary of that era had a fused path). The fused kernels are graded
by **mm**, **decode** and **ppl**, where they run by default.

`dense_mm` is the same story with a different mechanism, and the arithmetic is worth
stating because it is the one place where a shipped kernel is knowingly less accurate
than what it replaced. The vendored dense-FFN prefill gemm (`src/ops/dense_mm.metal`,
seq > 32 on the dense checkpoint) runs matmul2d's reduced-precision tensor-core path:
against a dequantize-then-f32 oracle at the 27B FFN shapes it lands ~4.1e-4 rel_l2
where candle's `kernel_mul_mm_q4_K_f32` lands ~1.9e-4, and the two differ from each
other by ~3.7e-4. That is the same ~2e-4 band §3b already names for the attention
prefill gemm, and llama.cpp sets the same descriptor flag for its own dense FFN — so
it is fork-equivalent, not novel. It is still bounded rather than bitwise, so
`XWEN_DENSE_MM_CLASSIC=1` is pinned on both sides of **strict** and the fused gemm is
graded by **mm**, **decode** and **ppl** against frozen floors. Introduced at schema
version 7 with grandfather `classic` (no pre-v7 binary had the gemm), so cached
references stay valid. Note the field is env-derived and the 35B-A3B has no dense FFN
layer at all — on that checkpoint `dense_mm` labels the configured path, not an
executed one, exactly as `flash` does below.

**`mv_ext` is pinned even though no gate fixture can exercise it, and the reason is
worth writing down.** The vendored multi-row mat-vec (`src/ops/mv_ext.metal`, routed at
seq 2..=8 from `QLinear::forward` and, since 2026-08-08, from `Proj::DenseF16Q8` for the
q8_0 attention and DeltaNet projections) never runs during a gate: prefill chunks are 512 tokens
and decode is a single token, so no fixture produces a forward inside the window. The
tiers are structurally blind to this kernel, and its correctness claim rests entirely on
the `mv_ext.rs` oracle tests against `QTensor::dequantize` at production reduction
lengths — not on anything in this document. It still gets the full treatment:
`XWEN_MV_EXT_CLASSIC=1` pinned on both sides of **strict**, an `mv_ext` field at schema
version 8 with grandfather `classic` (no pre-v8 binary had the kernel), so cached
references stay valid. Three reasons the pin earns its place despite the blindness: it
costs nothing, it becomes load-bearing the moment a fixture or a serve path does enter
the window, and a dump that cannot say which path produced it is worth less than one
that can. Note the accuracy direction depends on which path a site displaces, so it is
not a property of the kernel. Against the `QMatMul` mm at the `QLinear` sites this kernel
is the opposite of `dense_mm` — f32 end to end, 4e-7..8e-6 rel_l2 against ~1.8e-4, 20-400x
BETTER, and the oracle tests assert `rel <= rel_classic` rather than an absolute band.
Against the vendored q8_0 gemv at the `Proj` sites the two are LEVEL (both ~1e-6, within
1-2% of each other, the better one varying by shape), which is why that comparison gets
its own 2x band. Either way the pin is provenance discipline rather than a bounded-kernel
guard.

**`flash: "fused"` is currently a provenance label, not a fact, on these
checkpoints.** The field is env-derived, and `flash.metal` is compiled at head dim
128 while Qwen 3.6 is 256 — so the vendored flash kernel is unreachable and prefill
actually runs candle sdpa with a materialized mask. The label is still *consistent*
(reference and strict pin `classic`, candidates report `fused`), so no gate is
weakened, but do not read it as evidence the flash kernel ran. Tracked in TODO.md.

## Floors

Floors are a property of the CHECKPOINT'S QUANT MIX as much as of our kernels, so they
never carry across a quant mix. Laguna's floors do not transfer and were not reused.

The gate constants are GLOBAL — one `COS_MIN_MM` applies to whatever file is gated —
so they are measured on every checkpoint and every fixture and then set below the
WORST observed value. Calibrating on one checkpoint or one prompt would be a coin flip
on the others.

Derivation discipline, inherited from laguna: **strict** is anchored just under the
worst achieved classic-path value (a regression detector, near-zero headroom by
design); **mm** sits far enough under its worst achieved to absorb prompt-to-prompt
and run-to-run variation without accepting a real regression; the perplexity bound is
`max(3 x |measured delta|, 0.002)` nats.

### Calibration record — 2026-07-28, oracle `e9fa0781`, ggml-org Q4_K_M

**Still the live record, and still an `e9fa0781` measurement**: the submodule was
bumped to `6fe749801` on 2026-08-29 and the 3.6 pair re-passed these floors
unchanged there (see "The oracle"). The numbers below were not re-derived.

**Measured with the REFERENCE DeltaNet scan on both sides**, before the fused delta
kernels existed (`src/ops/delta.metal` postdates the last dump in this table by half
an hour). Every dump below is `delta: "classic"` in the schema-v6 sense — they predate
the field, so they carry no `delta` key and grandfather to exactly that value.

This matters for reading the table now that the fused scan is the mm/decode/ppl
default. The **strict** anchors stay directly comparable, because
`XWEN_DELTA_CLASSIC=1` is pinned on both sides of that tier (see "Provenance pins") —
so a strict number that moves means something OTHER than the delta kernel changed. The
**mm / decode / ppl** rows are the pre-fused BASELINE: the fused scan is bounded-close
rather than bitwise, so it will drift against these, and the drift is the measurement.
Note the 35B mm headroom is only 5.4e-4 (achieved 0.999540 against a 0.999 floor), so
that cell is where a fused-scan regression surfaces first.

Re-deriving these floors from a run of the change under test would be circular — the
floor is what the new kernel has to clear. If a fused-path tier cannot clear one, the
options are to fix the kernel or to re-run this whole cross-checkpoint sweep and record
the widening WITH its evidence here; not to relax the constant in `tests/parity.rs`.

Raw last-position cosine of each candidate against the f32 Reference oracle, all
three fixtures, both checkpoints:

| fixture | 35B strict | 35B mm | 27B strict | 27B mm |
|---|---|---|---|---|
| code-short | 0.999999861 | 0.999539782 | 1.000000000 | 0.999999806 |
| text-mixed | 0.999983987 | 0.999723867 | 1.000000000 | 0.999999765 |
| long-mixed | 0.999894418 | 0.999862836 | 1.000000000 | 0.999993294 |

Top-1 matched the Reference on every cell, top-5 overlap 5/5 on every cell. The
candidate/reference L2-norm ratio stayed in `[0.996937, 1.003992]`.

**Floors set from this table** (`tests/parity.rs`):

| constant | value | derivation |
|---|---|---|
| `COS_MIN_STRICT` | `0.9998` | ~1e-4 under the worst achieved 0.999894 (35B long-mixed) |
| `COS_MIN_MM` | `0.999` | ~5.4e-4 under the worst achieved 0.999540 (35B code-short), ~1.7x the observed 0.99954..0.99986 spread |

Both are an order of magnitude tighter than laguna's (0.9955 / 0.985): these kernels
track the oracle much more closely on this architecture, largely because the Qwen
Q4_K_M mix keeps attention, ssm and shared-expert weights at q8_0 rather than pushing
everything to q4_K.

**The 27B column is near-vacuous for STRICT by construction.** The dense model has no
routed experts, so `--moe-impl reference` and `--moe-impl fused` run the same
`DenseMlp`, and the strict env pins everything else classic on both sides — the
bitwise 1.000000000 confirms determinism, not expert-kernel fidelity. The 27B's real
signal is the mm tier (f16 attention path + fused glue) and the decode/ppl tiers.

**Decode tier, 64 forced-replay steps per fixture:**

| fixture | 35B agreements | 35B excused | 27B agreements | 27B excused |
|---|---|---|---|---|
| code-short | 63/64 | 1 (candidate 3555 vs reference 36497, 0.0040 logit below top1) | 64/64 | 0 |
| text-mixed | 62/64 | 2 (0.5567 and 0.2606 below top1) | 64/64 | 0 |
| long-mixed | 64/64 | 0 | 64/64 | 0 |

Zero non-excused mismatches anywhere. Worst per-step candidate/reference L2 deviation
across the 35B's three fixtures: **1.0211**.

Two notes on the inherited `_Q8` widened bands, which fire on every Qwen candidate
(`attn_decode == "q8"`). `NEAR_TIE_MARGIN_Q8 = 1.0` is load-bearing: text-mixed step
15 excused at 0.5567 and would have hard-failed at the standard 0.5. `NORM_RATIO_MAX_Q8
= 1.5` is not: the measured worst deviation is 1.0211, so the band has ~24x more
headroom than the data needs. Neither was re-derived here — see the TODO.md ledger
item.

**Perplexity tier, 4218-token corpus (4217 scored), all runs `nonfinite == 0`.** The
candidate's DeltaNet path is called out because it moved the number:

| checkpoint | candidate delta path | reference NLL | candidate NLL | signed delta |
|---|---|---|---|---|
| 35B-A3B | classic (reference scan) | 1.693659 | 1.694170 | +0.000511 |
| 35B-A3B | fused | 1.693659 | 1.694450 | +0.000791 |
| 27B | classic (reference scan) | 1.747872 | 1.748093 | +0.000221 |
| 27B | fused | 1.747872 | 1.748201 | +0.000330 |

**The sign is the finding, not the magnitude.** The candidate is worse — higher NLL —
in all four measurements, across two architectures and two different candidate
implementations. This is a systematic bias, not symmetric rounding noise, and it is
what a scale-sensitive instrument is supposed to expose. Switching the delta path from
the reference scan to the fused kernels widened the gap by **+55% on the 35B and +49%
on the 27B** — proportionally the same on both models, which is what makes it credible
as a real fidelity cost of the fused scan rather than run-to-run variation. Everything
else about the two candidates was identical, so the attribution is clean.

**`PPL_NLL_DELTA_MAX` stays at 0.002, anchored to the REFERENCE-SCAN baseline.** The
recipe `max(3 x |measured|, 0.002)` was applied once, to the classic-scan measurement
(`max(3 x 0.000511, 0.002)` = 0.002, the floor binding). Re-applying it to the fused
measurement would give `3 x 0.000791` = 0.00237 and widen the bound — but that fits the
bound to the change under test, and a bound re-fitted to each new implementation
ratchets outward forever and catches nothing. The recipe is a one-time floor-SETTING
heuristic against the oracle, not an invariant to maintain against the candidate.

So the constant deliberately no longer reproduces from `3 x measured`, and that is the
correct state: it is now a TIGHTER bound than the recipe would give, which makes it
more sensitive, and the fused path still clears it with 2.5x headroom. Widening it
later requires evidence that the increase is benign — which perplexity alone cannot
show, so corroborate with greedy agreement and the cosine tiers before touching it.

**Trip-wire for future kernel work:** the fused scan sits at 0.000791 on the 35B. A
further ~2.5x rise fails the gate. Read that as the instrument working; the cosine
tiers are much less sensitive here (the 35B mm cosine actually *improved* with the
fused scan, 0.999540 → 0.999631), so perplexity is the number to watch.

Runtime, for planning a gate run. The expensive half is the Reference oracle, and it
is cached: a COLD four-tier run (every reference generated) is 10-15 min per
checkpoint — the 35B ppl reference pass alone is 3.3 min, and each decode reference is
~80 s. A WARM run reuses every reference and only regenerates the candidates, which is
**42 s for the 35B and 2.0 min for the 27B** across all six graded tiers. That is the
number that matters, because it is what re-gating a model-math change actually costs.
`--regen-ref` forces the cold path.


Re-calibrate whenever the anchor checkpoint, its quant mix, or the oracle pin
changes. The floors are global constants in `tests/parity.rs` applied to whatever
file is gated — per-checkpoint applicability is enforced by THIS procedure, not by
code.

## Raw greedy oracle

`llama-cli -st -no-cnv` APPLIES THE CHAT TEMPLATE. On these checkpoints its output
opens with the Qwen thinking block (`[Start thinking]`), so `llama-cli.txt` is a
chat-wrapped continuation, not a raw-completion oracle, and it will "diverge" at
token 1 against a raw run. For raw greedy parity use `llama-server /completion` with
a token-id ARRAY as the prompt (no template, no tokenizer):

```bash
reference/llama.cpp/build/bin/llama-server -m "$(bun scripts/hf.ts model)" -ngl 999 -c 4096
curl -s localhost:8080/completion -d '{
  "prompt": [727, 73111, 1393, …], "n_predict": 64, "temperature": 0, "top_k": 1
}'
```

Force `top_k: 1` (or ignore the emitted tokens and read the argmax from `n_probs`
logprobs): a `temperature: 0` request has historically still dist-sampled the emitted
token on some builds.

## Perplexity gate

Greedy agreement catches token flips but is blind to how the fused path reshapes the
*distribution*. The perplexity-delta bound over a frozen held-out corpus is the
scale-sensitive complement: mean next-token NLL of the Fused runner against the
Reference oracle on the identical corpus, failing if they drift past the frozen
bound.

**Corpus** (`tests/fixtures/ppl-corpus.txt`, attribution in
`ppl-corpus-README.md`): the head of the WikiText-2 raw *test* split — held out (not
the parity prompts), mixed-register prose, truncated at a paragraph boundary.
**4218 tokens under the Qwen tokenizer** (`add_special_tokens=false`, nothing
prepended — the vocabulary has no BOS), 4217 scored.

**Protocol** (identical on both runners, so protocol quirks cancel in the delta): one
continuous chunked-prefill pass, 512-token chunks, positions continuous, KV cache
fresh at the start and never reset between chunks. At every position `p`,
`log_softmax(logits[p])[tokens[p+1]]` in f64 with a stable logsumexp;
`mean_NLL = -mean(logprob)`.

**Gate** (`ppl_parity`): enforces runner provenance on both sides, zero non-finite
logprobs, identical scored token streams (count + FNV `token_hash` + full ids), then
`|mean_NLL(fused) - mean_NLL(reference)| <= PPL_NLL_DELTA_MAX`.

The reference dump is a keepable artifact: the blessed copy is
`tests/fixtures/reference-ppl-<basename>.json` per checkpoint, so a routine check
only regenerates the fused side. Resizing or regenerating the corpus invalidates both
the fixture and the frozen bound — recalibrate.

## Limitations

- The gate compares against llama.cpp's implementation, not against HF transformers;
  the two differ by design in conversion-baked forms (norm +1, tiled V-heads — see
  decisions.md "Ground truth"). Bugs shared with llama.cpp's qwen35 implementation
  are invisible to this gate.
- fp32-state DeltaNet means chunked-vs-recurrent prefill equivalence is numeric, not
  bitwise; the chunked kernel gets its own reference-vs-kernel test independent of
  llama.cpp (P8).
- Track A's per-layer resolution stops at the layer boundary: with no taps inside the
  DeltaNet or MoE blocks, a divergence localizes to a layer and a stage
  (mixer / FFN), not to a sub-op. llama.cpp does not cb() the fused
  `GATED_DELTA_NET` node either, so even the oracle side is opaque there unless the
  fused op is disabled.
- eval-callback exposes only per-node sums plus first-3/last-3 samples, so Track A's
  logit comparison is coarse. Full-vector cosine is Track B's job.
- eval-callback's samples are the last *innermost* row. For rank-2 tensors that is
  the whole last-position vector; for rank > 2 it is one head of the last position,
  so `parity.ts` skips the sampled-row check there and relies on the exact
  full-tensor `sum`.
- eval-callback computes logits for the last position only, so per-position logit
  parity is not available; both tracks compare the last position.
- Run eval-callback with `-ngl 999` (the `ref-dump.sh` default) so the oracle uses
  the Metal path, closest to our engine.
