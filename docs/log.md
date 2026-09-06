# Engineering log

Reverse-chronological. Heading convention: `## YYYY-MM-DD — headline stating what
shipped, ideally with the number`. Same-day entries disambiguate in the heading text.
Superseded entries are marked in the headline, never deleted.

## 2026-09-06 — Drafting off by default on the 35B-A3B

A policy change, no math touched. Two independent measurements the same day read the
35B-A3B's drafted arm below plain at every length — -8% at 1046 tokens deepening to -37%
at 16409 on long-document prose, and -4% on a 256-token code prompt — because plain
decode gained 10.3% from the router gemv while the drafted defaults were fitted on
2026-08-08 against a level three improvements older. A zero-flag run on that checkpoint
now decodes plain and says so; `--draft official`, or a serve config that names the
official drafter, still attaches the sidecar, and the 27B and 3.8-27B are unchanged. What
silence means is `Model::draft_default_on()`, read by the CLI and by serve's new
`DraftMode::Default`. The fitted `p_min` 0.3 and depth 15 are left alone on purpose: the
retune (Front item 1) is what decides whether the default comes back, and it needs more
than one context length to see the loss.
[Decision](decisions/speculative-decoding.md).

## 2026-09-06 — The 128k envelope measured, and the prefill mask is a 25 GB memory fix

The oldest open ledger item closed. `scripts/longctx.ts` is the new instrument: it cuts
repo prose to a token target against the checkpoint's own GGUF vocab, interleaves
lengths A B A B, and samples `phys_footprint_peak` per run. Two results reframe things.
**Flash-Next decode is flat across context where the 35B's is not** — 47.1 to 46.9 tok/s
from 8k to 64k and 41.9 at 128k, against the 35B's 96.4 falling to 36.8 — so the 35B's
headline 127.0 describes short conversations only. **Prefill is the wall**: 197 s for a
maximal 35B prefill and 569 s for Flash-Next, which is what `queue_timeout` now derives
from (300 s flat was dropping queued requests while the server worked normally; the
default is 2640 s at `context_length` 262144, and it is logged at startup).

The prefill mask went to the device, bit-identical, `XWEN_HOST_MASK=1` restoring the host
fill. The ledger called the host loop "the binding cost of long prefill" and that is
**refuted on time** — both checkpoints are a dead heat at 131072, because candle fills
chunk N+1's mask while the GPU is still on chunk N. What it actually buys is **the 35B's
131072 peak footprint falling from 42-69 GB to a flat 17 GB**, which is where the "~28 GB
that is neither weights nor KV" went: `Tensor::from_vec` hands the pool a fresh
exact-size buffer per chunk and no two chunks ask for the same size. Flash-Next stays at
59 GB because its QSA indexer builds its own host mask per sparse layer per chunk, which
is the ledgered follow-up. `DEFAULT_DRAFT_CTX` stays at 8192 and the reason inverted:
drafting on the 35B reads **below** plain at every length measured, -8% at 1046 tokens
(80.6% acceptance) deepening to -37% at 16409, so raising the horizon would extend a
loss — which corroborates the same day's presence-penalty finding on a code prompt from
the other direction. The horizon also stopped being silent, `--max-ctx` became one
constant clamped to `n_ctx_train` on every surface, and the disk flush budget follows the
bytes it has queued. Tables, the A/B and what was left undone:
[record](records/long-context-envelope.md).

## 2026-09-06 — Chores: health and TUI name the checkpoint, tagged bench records, mtl_size, scripts sweep

Five ledger chores, none of them touching model math. `/health` grew a `model` field and
the dashboard a resident cell and a HISTORY `model` column, all three reading one
`ResidentModel` the engine stamps at load and clears at every unload — the bool and the
name come from a single read, so they cannot disagree, and the TUI polls the cell rather
than folding events, so an unload cannot leave a stale name on screen. Run records grew
an optional `tag`, set from `XWEN_METRICS_TAG` inside `RunRecord::new`; the scripts that
drive the binary export it (`bench`, `parity`, `demo`), `xwen stats` reports untagged
runs alone and its footer always says how many it left out, `--tag`/`--all-tags` ask for
the rest (decisions.md "Metrics" — the new paragraph supersedes 2026-09-05's). The
`mtl_size!` macro is gone for a `const fn` returning `objc2_metal::MTLSize`, whose round
trip through candle's `get_block_dims` had every field overwritten and so computed
nothing: 69 call sites, no value moved. And the scripts sweep closed: `classify.ts` and
the bench fixtures carry no laguna leftovers, but the fixture NAMES are laguna's token
counts — 880, 3851 and 596 under this tokenizer — so the two 27B prefill rows in
docs/perf-state.md are relabelled and the mapping is written down.

## 2026-09-06 — Presence penalty: the cards' recipe, through the speculative verify, on by default

The cards' `presence_penalty` 1.5 is live: non-thinking on every checkpoint, thinking on
the 35B-A3B alone, explicit values winning on every surface (CLI, serve.toml, OpenAI,
native, batch). On the device before the softmax, so plain decode is unchanged; through
the verify rows with a round-end rollback truncation, so the greedy equivalence gate passes
at 1.5 and at 0. Acceptance on the 35B drops 63.0% to 59.4%, drafted decode -1.2%; sampled
mode diverges at 1.5 in the known near-tie class. `top_k` 0 is now "no cut", 1 greedy.
Batch's non-thinking output changes. [Record](records/presence-penalty.md).

## 2026-09-06 — Metrics: the Claude Code agent id is recorded, and `xwen stats --by agent`

`x-claude-code-agent-id` now lands on the run record as its own `agent` field, read
beside the session header on every serve route, and `--by agent` groups on it. The
session key is untouched, so one session is still one row under `--by session`; this is
the view that splits it by subagent. Absent field reads back as `None`, no schema bump.
Reverses the 2026-09-05 decision on the owner's say (decisions.md "Metrics"). 7f0659e.

## 2026-09-06 — Ledger triage: 98 open items to 49, and an intake rule

The first triage pass under the rules the restructure entry below wrote, and the ledger
loses half its length without losing a line of text. 34 items and one lettered sub-item
are retired, each with the reason and the reopen condition that brings it back; 10
duplicates are folded into the partner item that already owned the question, with a
pointer line wherever the fold carried something the partner did not say; and 5 are
closed — two that shipped and went stale in place, two by decision (decisions.md
"Serving" on plane-less slots, "Thinking budget and sampling controls" on sampler order)
and one by a doc line in docs/benching.md. Everything moved verbatim to
[docs/ledger-archive.md](ledger-archive.md). The cause the pass exposed is the intake
path, so the rule changed with it: AGENTS.md now says a deferred scope or a review
finding enters the ledger only when it carries a number or a user who is waiting, and
everything else is a "not taken now" line in the arc's record. Record: [docs
restructure](records/docs-restructure.md).

## 2026-09-06 — Docs restructured: decisions by topic, records behind log stubs, perf-state and benching split out, and the ledger regrouped by area with a ten-item Front

Two passes in one day, both script-driven with every moved line accounted for. The first
split decisions.md into 17 topic files behind an index, moved 32 long log entries into
docs/records behind short stubs, archived 44 closed ledger items verbatim, trimmed ~50
annotations to three lines and a link, and took AGENTS.md from 524 to 388 lines by
moving the figures to docs/perf-state.md and the measurement rules to docs/benching.md;
scripts/docs-check.ts now asserts links, anchors, titles and quoted names. The second
regrouped TODO.md by area with a capped Front, made the lettered sub-item the unit that
closes, added state tags and `From:` provenance lines, and gave the ledger a second exit
beside shipping (retired, with a reopen condition; decisions.md "The ledger is a memory
and a backlog"). Sizes and the reasoning are in the record:
[docs restructure](records/docs-restructure.md).

## 2026-09-06 — Router projection on a 256-threadgroup gemv: 35B decode 115.1 → 127.0 (+10%), Flash-Next 50.5 → 52.9 (+5%); the lever was occupancy, not launches

The third decode lever of the day and the largest, and it is not a launch-count item at
all. The morning's two entries left the MoE router projection as the one decode candidate
the duplicate-dispatch probe could not price: its duplicate cost nothing, which at decode
means the stage overlaps itself rather than that it is free. What the zero did say was
that it runs at low occupancy. This entry replaces the kernel underneath it and measures
+10.3% on the 35B-A3B and +4.8% on Flash-Next, more than the fused hc gate and the fused
shared expert bought on Flash-Next between them.

Full record, moved verbatim: [records/router-gemv.md](records/router-gemv.md) (128 lines).

## 2026-09-06 — Fused MoE shared expert: 5 launches per layer become 1, 35B decode +1.6%, Flash-Next +0.6%, and the probe says why: the shared expert was bytes, not launches

The other half of the morning's decode item 2. The probe had just priced the shared
expert's five-launch chain at a 0.43 ms floor and the ledger had it as the next −192
launches; this entry ships the fusion, measures it on both MoE checkpoints, and finds
that the launch budget over-predicted it by a factor of five. The number is small and
positive on both checkpoints. The reading behind it is the part that changes how the
next candidate gets ranked.

Full record, moved verbatim: [records/fused-moe-shared-expert.md](records/fused-moe-shared-expert.md) (152 lines).

## 2026-09-06 — The fused hc gate is +57-76% on 2..8-token forwards, and the duplicate-dispatch probe learns decode mode: the shared expert floors at 0.43 ms, the router projection overlaps itself

Two questions the 2026-09-05 entries below left open, neither run changing model math
(cf7c579, d5daa18). Forcing every prefill forward to exactly n tokens, the fused
hyper-connection gate beats the split seven-dispatch path by +76% at 2 tokens, +57%
at 4 and +61% at 8 (three interleaved rounds, medians, `lowpowermode 0`), because in
that window the split path sends both bottleneck gemms through candle's tile matmul.
In decode mode the probe floors the shared expert at 0.43 ms/token and reads the
router projection and the MoE glue at about zero. The reading to carry forward: a
duplicated decode launch has no buffer hazard against its original, so a delta above
zero is a FLOOR and a delta of about zero means the stage overlaps itself, never that
it is free. The router projection is therefore still unpriced, and its zero says low
occupancy. Left open: the 8-token decode tail read lower on every fused arm, and the
recheck needs a prompt that cannot stop early.

Full record, moved verbatim: [records/hc-gate-ragged-and-probe-decode.md](records/hc-gate-ragged-and-probe-decode.md) (107 lines).

## 2026-09-05 — Fused hyper-connection decode gate: 7 dispatches per gate become 3, Flash-Next plain decode 47.0 → 51.2 tok/s (+9% median), the first in-situ confirmation of the dispatch-count lever

The ceiling diagnosis (below) put ~7.4 ms of a 21.3 ms token into per-dispatch fixed
cost at ~4 µs across ~1740 launches, naming the hyper-connection gates as the largest
population. Commit dd50397 replaces six of the seven launches a gate makes at decode
with two kernels, taking hc launches from 672 to 288 per token. Flash-Next plain
decode goes 47.0 to 51.2 tok/s, +8.9% at the median over three interleaved rounds on
the 596-token fixture, with the honest range +5-10% because both arms are bimodal;
prefill is unchanged. The budget predicted +7.8% for 384 dispatches at ~4 µs, so this
is the first end-to-end confirmation that the per-dispatch residual is real and
recoverable. The kernels are bounded, not bitwise; the replay check passes with 0
hard mismatches and `XWEN_HC_GATE_CLASSIC` restores the split path. Still ledgered:
the write fold, the unswept threadgroup shape, and the QSA tail and MoE glue.

Full record, moved verbatim: [records/fused-hc-gate.md](records/fused-hc-gate.md) (103 lines).

## 2026-09-05 — Duplicate-dispatch probe prices Flash-Next prefill in situ: expert gemms 1.09 s of 3.42 s @3851 (32%), MoE glue 0.40, hc gates 0.39, GDN 0.23, shared expert ~0

The instrument the ceiling diagnosis asked for (below). `XWEN_DUP_STAGE=<names>` makes
the multi-token path encode a named stage's kernel launches `XWEN_DUP_REPS` extra times
(default 1) with the copies' results dropped, so the prefill wall delta against a plain
run is that stage's GPU time in situ — no syncs, no host work duplicated, only launchers
that are pure functions of tensors the caller already holds (`ops::dup`; the router,
gathers, cache advances and allocations of the surrounding blocks run once). Stages:
`experts` (the three mm_id gemms), `experts_down`, `moe_glue` (router, activation,
epilogue), `shexp`, `hc` (norm, both bottleneck gemms, silu, mix, write), `hc_gemm`,
`gdn` (conv, beta/decay head, scan, gated norm), `gdn_scan`. Commit ab43499; parity.md
carries the switch row. The delta is a LOWER bound for a stage that leaves the GPU idle
(candle's encoder barriers only on buffer hazards, so a copy writing a fresh buffer may
overlap the original); the two-copy experts arm below reads 1.03 s per copy against 1.09
for one, so at least that stage has little overlap to hide.

**Protocol.** `xwen generate --no-draft --raw -n 4 --stats` on the 3851-token
`prefill-4k` fixture, Qwen3.8-Flash-Next at the 2048 chunk, ten arms interleaved with the
order reversed every round, three rounds, 60 s idle between rounds, medians;
`pmset -g` said `lowpowermode 0` (automatic) throughout. The binary was a detached
worktree build of ab43499 at /tmp/xwen-bench — an earlier attempt on the main tree's
binary aborted mid-session because a coding agent's `cargo build` had swapped
`target/release/xwen` (and its `include_str!` kernels) under the harness; a first,
unpinned session of five arms is quoted only as a replicate.

| arm (stage duplicated once) | tok/s | wall s | delta s | share of 3.42 s | unpinned replicate |
|---|---|---|---|---|---|
| base | 1126.4 | 3.419 | — | — | 1131.1 (3.405 s) |
| experts (gate, up, down mm_id) | 854.3 | 4.508 | 1.089 | 31.8% | 0.959 (28.2%) |
| experts, two copies (`XWEN_DUP_REPS=2`) | 703.7 | 5.473 | 2.054 = 1.027/copy | — | — |
| experts_down | 987.2 | 3.901 | 0.482 | 14.1% | 0.423 (12.4%) |
| moe_glue (router, activation, epilogue) | 1009.8 | 3.814 | 0.395 | 11.5% | — |
| shexp (shared expert, both gemms) | 1123.3 | 3.428 | 0.009 | 0.3% | — |
| hc (norm, down, silu, up, mix, write) | 1011.5 | 3.807 | 0.388 | 11.3% | 0.346 (10.2%) |
| hc_gemm (the two bottleneck gemms) | 1082.8 | 3.557 | 0.138 | 4.0% | — |
| gdn (conv, beta/decay, scan, gated norm) | 1055.9 | 3.647 | 0.228 | 6.7% | 0.222 (6.5%) |
| gdn_scan | 1077.2 | 3.575 | 0.156 | 4.6% | — |

Round-to-round spread on the duplicated arms was under 1% except the experts arms
(±2-4%); the 4-token decode column is noise at that length and is not quoted.

**What it settles.**

- **The expert gemms are 28-32% of prefill wall (0.96-1.09 s of 3.4 s), inside the
  ceiling diagnosis's 14-43% bracket** and closer to the amortized bench's 1.44 s than
  to the full-grid A/B's 0.46. Gate+up together are 0.61 s (18%), down 0.48 s (14%) —
  the down plane is 44% of the expert time, not "about half". The 0.30 s of weight
  re-reads sits inside this.
- **The 2026-08-30 "gemms are a minority of `ffn`" reading is REFUTED.** In situ the
  `ffn` stage is experts 1.09 + glue 0.40 + shexp 0.01 = 1.50 s (44% of wall) and the
  gemms are 73% of it. The glue is real — 0.40 s, as much as every hc gate together — but
  it is the smaller half, and the shared expert is free at prefill (two dense gemms at
  2048 rows, 0.3%).
- **The hyper-connection gates are 11% (0.39 s), of which the bottleneck gemms are 4%
  and the four glue kernels plus the write 7% (0.25 s).** The ledger's "~8% activation
  traffic" estimate was about right.
- **GDN kernels are 7% (0.23 s): the chunked scan 4.6%, conv + beta/decay + gated norm
  2.1%.** The stage profiler's 20% for `mixer_delta` includes its three projections,
  which the probe does not cover.
- **Priced: 2.11 s = 62% of wall. Unpriced 1.31 s (38%):** the GDN and attention
  projections, full attention and QSA selection (12 layers), PLE, embedding, lm_head, and
  whatever the concurrent encoder hides between stages. Those are the next probe stages if
  the ranking needs them; attention and QSA mutate caches around their kernels, so the
  wrap has to sit inside the block.

**How the ledger reads now (TODO.md, prefill).** Expert gemm efficiency stays first at a
measured 0.96-1.09 s; MoE glue (0.40 s) and the hc glue (0.25 s) are second and third and
are now priced rather than estimated; the GDN scan (0.16 s) is fourth; the shared expert
drops off the list. Every one of these is a fusion or kernel-efficiency lever, none is a
launch-count lever (prefill dispatch floor <1%, unchanged).

## 2026-09-05 — Ceiling diagnosis for Flash-Next: achievable bandwidth measured at 537-565 GB/s, a decode token is 57% weight bytes and ~33% per-dispatch fixed cost, prefill is not launch-bound

TODO.md's "FIRST" item asked why decode and prefill sit under their ceilings. Steps
1-3 are here; every run was at `lowpowermode 0`, automatic power mode. Achievable
bandwidth is now measured: streaming read 537-565 GB/s at the median, 87-94% of the
614 nominal, with a 2.4-2.7 µs per-dispatch floor for a dependent chain. A decode
token reads 6.33 GB of weights, 11.7-12.3 ms of its 21.3, so the bytes-only ceiling
is 81-86 tok/s and the residual ~7.4 ms is ~1740 dispatches at ~4 µs average; decode
is neither CPU-bound nor command-buffer-bound. Prefill at 3851 tokens runs 13.7
TFLOP/s on 12.07 GFLOP/token, its dispatch floor is under 1%, and two in-situ A/Bs
bracket the expert gemms at 14-43% of wall, which CONTESTS the 2026-08-30 "gemms are
a minority of `ffn`" reading. The lever is dispatch COUNT for decode and the expert
gemms plus hc glue for prefill, never per-kernel bandwidth.

Full record, moved verbatim: [records/ceiling-diagnosis.md](records/ceiling-diagnosis.md) (264 lines).

## 2026-09-05 — PLE gate and conv move to device for multi-token prefill: Flash-Next prefill +12-13% at 880 and 3851 tokens, on by default (`XWEN_PLE_TAIL_CLASSIC` restores the host tail)

Picked up from TODO.md's "Next Flash-Next perf work" item (P3 item (5)), started in a
Codex session and finished here. Two Metal kernels take the PLE gate and the dilated
conv off the host for multi-token forwards, where `XWEN_PLE_PROFILE` had put them at
41 + 198 ms per 2048-token chunk; the conv state stays host-owned and its layout is
untouched. Flash-Next prefill goes 1010 to 1139.7 tok/s at 3851 tokens (+12.8%) and
1117.6 to 1261.5 at 880 (+12.9%), interleaved arms with 60 s idles, `lowpowermode 0`,
decode flat. `XWEN_PLE_TAIL_CLASSIC` restores the host tail, which decode keeps: the
pair costs ~0.13 ms/token there and nobody has turned that into a qualified gain. The
forced replay showed one hard mismatch, and a control established it is not the
kernel: reversing the summation order of one f32 dot product in the classic path
reproduces the same mismatch at the same step. So `flashnext-replay.ts` gained an
engine-near-tie excusal rule, and the run passes at 185/192 with 0 hard.

Full record, moved verbatim: [records/ple-device-tail.md](records/ple-device-tail.md) (123 lines).

## 2026-09-05 — PLE batches its three device-to-host readbacks

The next Flash-Next perf candidate from TODO.md was the PLE readback collapse.
`PleLayer::forward` called `to_vec1` separately for key, value and carrier, and
candle's `MetalStorage::to_cpu` allocated, blitted and waited each time;
`readback_inputs` now encodes all three copies into one staging buffer and waits
once. The all-length sweep carried a drift flag and no established prefill gain, so
batching ships at seq == 1 only and multi-token prefill keeps its original transfers.
Decode reads 44.78 to 46.06 tok/s, +2.85% central, over a CB/BC pair at 3677 tokens
with 60 s idles and anchors inside the 3% flag; the isolated transfer transaction
goes 0.50-0.59 to 0.17-0.23 ms. Correctness is equivalence to the pre-change engine
rather than a fresh oracle run: 192/192 exact replays across three fixtures, plus all
six 35B parity checks. A second candidate, hc's two Q8_0 decode projections, was
screened and left unchanged, its down projection still unresolved on the ledger.

Full record, moved verbatim: [records/ple-readbacks.md](records/ple-readbacks.md) (105 lines).

## 2026-09-05 — Shared agent instructions live in AGENTS.md

Moved the root agent context from `CLAUDE.md` to `AGENTS.md`, preserving the
instructions. `CLAUDE.md` imports it with `@AGENTS.md` so both entry points use the
same source. Updated the README and justfile references to the canonical file.

## 2026-09-05 — Per-run metrics on disk and `xwen stats`: every surface appends a JSONL record, aggregated by day/model/surface/session

Nothing in xwen remembered a run, so `src/metrics.rs` and an `xwen stats` subcommand
give every surface a place to write one line and give that file a reader. Each
finished run appends one JSON object to `$HOME/.local/state/xwen/metrics.jsonl`:
schema version, timestamp, surface, checkpoint, the token counts and the seconds each
phase took, `ok`, and the optionals a surface may know, an absent optional meaning
not measured and never 0. `stats` groups by day, week, month, model, surface, client,
session or all, and every rate is sum(tokens)/sum(seconds) over the bucket rather
than a mean of per-run rates. A failed run is a record, not a gap, and `ok` means the
run reached its own end. Twenty review findings came back and all twenty are fixed,
five material. What is NOT covered: no model was run end to end for a smoke, because
another process held the GPU, and that is the first ledger item.

Full record, moved verbatim: [records/metrics.md](records/metrics.md) (170 lines).

## 2026-08-31 — Serve dialects: a mid-conversation system turn demotes to a user turn, and Claude Code runs against xwen end to end

**Context.** A config-repo session pointed Claude Code at xwen's Anthropic endpoint
(the rewritten `qwen` launcher; `qwen --review` drives the headless read-only
reviewer). First real harness traffic immediately hit a wall: Claude Code injects its
token-budget reminders as `system`-role messages mid-conversation, and both wire
dialects carried them through to the renderer, whose template-fidelity check refuses a
system message past index 0.

**The fix, and why not the other one.** `push_turn` in serve/anthropic.rs and
serve/openai.rs now demotes a system turn past the head of the conversation to a user
turn, in place, merging into adjacent user text. Relaxing the validation instead was
considered and refused without needing an experiment: the official templates themselves
hard-raise `System message must be at the beginning.` in the message loop (both
vendored dialects; chat_template-qwen38.jinja:106), so a mid-stream `<|im_start|>system`
block is formatting the checkpoint's own template forbids and llama.cpp would 500 on.
The renderer stays faithful, chat.rs's refusal stays as the backstop, and the wire
dialects absorb the harness quirk — the same normalizer-adapts/renderer-never-lies split
the serve tree already runs on. Demotion is positional on purpose: an
"N tokens left" reminder means what it means at the turn it arrived in, and in-place
demotion reproduces exactly how Claude Code embeds `<system-reminder>` blocks in user
turns anyway. Leading consecutive system messages still merge into the one head system
block. One test per dialect; serve suite 420/420, chat 38/38; no model math touched, so
no parity run owed. Verified end to end: a full `qwen --review` session against a live
server, Qwen answering from actually-read files.

**Also shipped (39db173): `just install`.** Plain `cargo install --path .` ignores
Cargo.lock — a re-resolved metal/objc2 crate set produced a binary whose Metal-4 kernels
failed to compile at runtime (dense_mm.metal, mpp::tensor_ops identifiers undeclared).
The recipe pins `--locked`; the trap is in CLAUDE.md's operational hazards.

## 2026-08-30 (technique survey) — Perf landscape and technique survey (research, no code): no public Apple Silicon runtime is a peer on Flash-Next, four techniques survive the cut, and candle turns out to already implement MLX-style concurrent encoding

Two research passes with nothing built: where xwen sits against every public Apple
Silicon runtime, and which of their techniques are worth taking here. On Flash-Next
there is no public peer. The best published same-chip figure is llama.cpp Metal on an
M5 Max at 33.0 tok/s decode / 966 prefill, but on a smaller IQ4_XS build, so it
crosses files and sessions and is not the claim to make; the honest headline stays
the same-file same-hour +13% decode / +24% prefill. The one contested class is the
35B-A3B, where other machines report ~91 and 130.2 against our 114, so no lead is
claimed until the ledgered same-machine arm runs. Four techniques survive the cut and
every one is unpriced. The finding that re-ranks the queue is that candle at the
pinned rev ALREADY implements the MLX-style concurrent encoding the survey meant to
adopt, just coarsely, so the residual levers are all candle-side: cadence,
whole-scope barriers, cross-encoder fence waits, per-dispatch locking.

Full record, moved verbatim: [records/technique-survey.md](records/technique-survey.md) (83 lines).

## 2026-08-30 (FFN glue) — the L2 fold and the shexp/hc gemms onto dense_mm: Flash-Next prefill +12% at 3803 and 7606 tokens, 35B +12%

The mm_id tile pass ended with the attribution that reset this queue: the expert gemms
are a minority of the prefill `ffn` stage. At the 2048 chunk the stage is 22
dispatches/layer (8 gemms, 14 glue); per layer the rescale chain makes six elementwise
passes over the [t,10,640] activation (~367 MB), the shared expert runs three Q8_0
QMatMuls through candle's 32-token-tile mul_mm (~334 MB), and the hc `down`/`up`
(Q8_0 [320,10240] / [10240,320]) ran plain QMatMul outside the stage entirely (~40 GB
of weight re-reads per chunk). Three levers, each behind its own switch:

1. **The L2 fold** (`ops::silu_mul_l2`, `XWEN_ACT_L2_CLASSIC` reverts): the rescale
   branch's seven dispatches — silu_mul, sqr, sum_keepdim, sqrt, clamp, affine,
   broadcast_div — as one threadgroup-per-row kernel (row in registers, fixed
   sequential-then-tree sum). Bounded, not bitwise: 3.574e-7 max-rel act_s / 2.287e-7
   col_l2 against the chain (`l2_fold_matches_candle_chain`); the strict tier is
   structurally blind to it (mv_id has no rescale branch), mm/decode/ppl grade it.
2. **shexp onto dense_mm** (`XWEN_SHEXP_QMATMUL` reverts): the three projections
   through the new `QLinear::forward_gemm` above `dense_mm_min_seq` (32) — the P8c
   kernel, its precision class measured 3.68-3.70e-4 rel_l2 vs the QMatMul route at
   the shexp shapes (`forward_gemm_matches_qmatmul_at_prefill_and_is_forward_below`).
3. **hc down/up onto dense_mm** (`XWEN_HC_GEMM_QMATMUL=down|up|both` reverts per
   arm): planes via `qlinear_with_buffer` (each ~3.3 MB, sharing the QMatMul
   allocation), routed from the fused hc read only; 3.66-3.69e-4 at both shapes.

**Bench** (3803/7606-token prompts, `XWEN_BENCH=1 generate --no-draft --raw -n 64
--stats`, two rounds with arm order reversed, 60 s idles; `pmset -g` before/mid/after:
`powermode 0`, no `lowpowermode` key; anchors p1937: 1133.7 → 1119.6 → 1148.5
prefill, ±1.3%). Cumulative arms, prefill tok/s (decode flat at 43.0-44.1 in every arm):

| arm | p3803 r1/r2 | p7606 (1 round) |
|---|---|---|
| all-classic | 872.1 / 865.0 | 766.5 |
| +fold | 918.3 / 902.3 | 771.9 |
| +fold+shexp | 918.2 / 898.2 | 767.0 |
| +fold+shexp+hc-down | 896.7 / 913.1 | 775.5 |
| +hc-up = default | 962.2 / 976.7 | 860.5 |

Attribution within-sweep: the fold +4.8%, shexp ≈0 end-to-end (its 334 MB was small
against the stage), hc-down ≈0, **hc-up +7-11%** — the k=320 shape briefed as "may not
win" is the biggest single lever; its [10240,320] output side is where QMatMul's
32-token tiling re-read the most weight bytes. Total +11.6% @3803, +12.3% @7606. The
35B (no hc): 2754.8/2746.3 → 3089.9/3081.1 prefill (+12%), decode 106.9-108.7 flat.
The free A/B there, `XWEN_MM_ID_TENSOR_HP=1` (f32 activation tiles, which have no
rescale chain at all): 2286-2314 prefill — the f32-tile gemm loses ~25% against the
f16 tensor default, so deleting the rescale work does not begin to pay for it.
Refuted (decisions.md).

**Greedy** (temp-0, 64 tokens, p3803): forks on both checkpoints, and the bisect says
near-ties rather than a guilty lever. The all-classic arm is deterministic (rerun
byte-identical, and byte-identical to the prior session's default), yet EVERY lever
alone forks Flash-Next at the same decode token ~28 ("due to bureaucracy" → "due to
warfare"), four of five arms producing one identical alternative continuation. On the
35B the fold and the shexp route each fork at byte 208 with identical outputs,
combined at byte 214. No tool reports the top-2 logit margin at a decode step
(`logits-dump` is prompt-only), so the tie can be argued but not priced — ledgered.

**Gate** (35B): ALL PASS, six graded — the first re-pass where graded numbers moved:
mm cos 0.999631 → 0.999618, ppl Δnll 0.000791 → 0.001179, both within the frozen
floors; strict bit-exact, all three levers structurally off it. Provenance schema v9
adds `act_l2` / `shexp_gemm` / `hc_gemm`, all grandfathered "classic".

**The review find that mattered** (Opus, Codex and Qwen converged on it): the hc
planes as first implemented also opened the mv_ext 2..8-token window on every hc path
— the plane predicate is `dense_mm_supported || mv_ext_supported`, hc `down`
(k 10240) passes the mv_ext one where `up` (k 320) does not, so ragged chunks and
serve resumes would have changed numerics asymmetrically, `XWEN_HC_CLASSIC` included.
The rule now: **hc planes are dense_mm-only** — `QLinear::without_mv_ext` keeps
`forward` on QMatMul at every count the gemm doesn't take, bitwise the pre-plane
behavior (`without_mv_ext_keeps_small_batch_on_qmatmul`), while `forward_gemm`
keeps the plane. The shexp planes predate this change and keep their window:
`XWEN_SHEXP_QMATMUL` restores the immediate pre-change route, mv_ext included.

## 2026-08-30 (mm_id tiles) — work-list grid and NR1 64: expert gemms +17-23% in isolation, prefill unchanged end-to-end at 3803 tokens, and the ffn stage turns out to be mostly not the gemms [CONTESTED 2026-09-05: the "mostly not the gemms" reading came off the 2.2x-inflated stage profiler; two in-situ A/Bs bracket the expert gemms at 14-43% of prefill wall, and the tile work reads −2.8% end to end — see "Ceiling diagnosis"]

The prefill-chunk pass left a ledger item for a NARROWER `mm_id` token tile (32 to
16). A code read refuted it before any bench: `kernel_mul_mm_id_t` dequantizes the
expert's whole weight tile once per TOKEN tile and is dequant-bound, so a narrower
tile RAISES the dominant cost. The read also found that the `(t/32, n_out/64,
n_expert)` grid is sized for one expert owning every row, so ~97% of launched
threadgroups early-return. Two changes shipped behind switches: a work-list grid, and
NR1 64 on the `_t` family. Isolated at t = 2048 the expert gemms gain +17-23%, all
four launch shapes bitwise against the old grid. End to end at 3803 tokens the
arm-to-arm spread sits inside a single arm's round-to-round spread, so no prefill win
is claimable. Greedy output is byte-identical and the 35B gate is ALL PASS. The
attribution drawn at the time, that the gemms are a minority of `ffn`, came off the
sync-inflated stage profiler and is contested above.

Full record, moved verbatim: [records/mm-id-tiles.md](records/mm-id-tiles.md) (82 lines).

## 2026-08-30 (serve on Flash-Next, first benchmark) — serve decodes at parity with `generate` through 32k (42-47 tok/s), a 32k conversation resumes its next turn in 0.5 s, and an edited prompt re-prefills from zero by design

Every Flash-Next figure in CLAUDE.md was `generate`'s, and TODO.md's "a qwen4exp serve
run has never been benchmarked" said not to quote them as serve figures. This is that
run: read-only, at f949b1d, nothing in `src/` touched.

**Setup.** `xwen serve --no-tui --port 8099` on its defaults — `--ctx 262144`, 2 cache
slots, 4 snapshots, disk tier off, no drafter (Flash-Next ships none). OpenAI dialect,
streaming, `stream_options.include_usage` on for the cached-token counts,
`chat_template_kwargs {enable_thinking: false}`, `max_tokens 64`. Prompts are the qsa-c
fixtures plus ~50 tokens of chat template. Load 23.2 s. `pmset -g` printed `powermode 0`
and no `lowpowermode` key (verbatim; no high-power claim either way). 20 runs, no errors.
Scripts under the session scratchpad, `serve-bench/`.

Three passes per prompt, in order: **cold** (nothing in the cache), **cached** (byte-
identical prompt resubmitted), **partial-edit** (the last user message rewritten, the
prefix ahead of it untouched). Each length ran twice (r2).

| prompt tokens | TTFT cold | TTFT cached | TTFT partial-edit | decode cold / cached / partial | cached_tokens cold / cached / partial |
| --- | --- | --- | --- | --- | --- |
| 1984 | 2305 ms | 95 ms | 2451 ms | 45.19 / 46.87 / 43.00 | 0 / 1977 / 0 |
| 1982 (r2) | 3957 ms | 97 ms | 2813 ms | 45.41 / 46.76 / 43.85 | |
| 7655 | 12336 ms | 126 ms | 10942 ms | 44.11 / 45.45 / 43.76 | 0 / 7648 / 0 |
| 7653 (r2) | 11120 ms | 137 ms | 11416 ms | 44.65 / 44.62 / 43.75 | |
| 32108 | 63949 ms | 233 ms | 64584 ms | 43.47 / 43.53 / 41.92 | 0 / 32101 / 0 |
| 32107 (r2) | 62933 ms | 215 ms | 64534 ms | 42.92 / 42.09 / 43.14 | |

Derived prefill (the bench script's own approximation: TTFT minus its render/encode
delta, first decode step included — NOT tokens over TTFT, which gives 861 for the 2k r1
cold cell and 501 for the 3957 ms r2 outlier): **~800-940 tok/s at 2k, 627-696 at 8k,
500-511 at 32k** — the same shape `generate` shows, prefill falling with length while
decode barely moves. Treat the prefill figures as approximate; `generate --stats` is the
prefill instrument.

**Thermal check.** The 1984-token anchor read 45.14 tok/s at the start and 42.53 right
after the two 32k prefills — −5.8%, over the 3% flag — and came back to 46.20 after a
90 s cooldown. That is the duty cycle, not a level shift; the 32k rows below were taken
under it and are the pessimistic end of their own spread.

**Against `generate`.** Parity at 2k and 7.6k (44-47 either way). At 32k serve reads
**4-7% lower** — under the 10% bar, so it was not chased and the
profiler was not run. One plausible contributor, untested: serve's
`--ctx 262144` logs `state 2.0GB` where `generate` at `max_ctx 8192` logs 0.2 GB, so the
recurrent-state allocation serve walks per step is 10x the one-shot path's. Ledgered.

**Multi-turn is where serve wins.** The Claude Code pattern — `[user, assistant, user]`,
the conversation grown by one exchange — resumes off the turn-boundary snapshot: 7650 →
7744 tokens cost 10467 ms cold and then **348 ms** for the follow-up (cached 7713 of
7744); 32108 → 32202 cost 67492 ms and then **489 ms** (cached 32171 of 32202). A 32k
agent conversation takes its next turn in half a second.

**Editing the last user message gets `cached_tokens: 0` and a full cold prefill, and
that is the design, not a miss.** `PrefixCache::plan` (src/serve/engine.rs:3727) takes
the longest common prefix with what the slot holds; because the edit is not at the
cache's end it cannot extend, so it asks `rewind_to` (engine.rs:3701) for a resume
point, and `rewind_to` quantizes DOWN to the nearest snapshot — the leading system block
(the anchor), a turn boundary, a fork point, or the tail a page-out took
(`plan_snapshot_stops`, engine.rs:2009). Below the shallowest of those it returns
`Cold`. The reason is state, not bookkeeping: DeltaNet and PLE state is recurrent, so it
is restorable only at positions where it was captured, and no snapshot sits inside a
message. Editing the last message of a single-message prompt therefore lands under every
snapshot and replays from zero. Mid-message snapshots would fix it and cost ~30 KiB of
image per token retained; ledgered, not taken.

**Footprint** at rest after the 32k runs: 16 GB phys, 21 GB peak, 43 GB of clean mapped
weights.

## 2026-08-30 (QSA decode, step C) — block selection moved onto the device: Flash-Next decode above the 2048 budget 33 → 44-45 tok/s at 3.8k-32k, the cliff closed

Steps A+B (below) left one named cost: each of the 12 QSA layers read its block scores
back to the host every decode step (`to_vec1` in `select_with`), ran `top_blocks` +
`expand_into` there and uploaded the rows. Each readback drains the pipeline, so the CPU
could not encode layer N+1 while the GPU ran layer N — the GPU idled 12 times per token.

**Mechanism.** `kernel_qsa_select` (`src/ops/qsa_select.metal`, `ops::qsa_select`,
`dispatch::run_qsa_select`): one threadgroup of 1024 threads per layer per step, each
thread a contiguous stripe of `ceil(nb / 1024)` scores. The threshold is found by
MSB-first radix select over a canonical integer key — four passes of a 256-bin threadgroup
histogram, thread 0 walking the bins downward to the one where the cumulative count
reaches what is still needed. Then a compaction: per-thread counts of keys above and equal
to the threshold, two threadgroup exclusive scans (simd prefix + per-simdgroup totals) that
rank the equal keys in index order and assign output slots, and a re-walk that emits every
above-threshold block and the lowest-indexed equal ones up to the quota. Stripes are
contiguous and thread-ordered, so the rows come out ascending. `keep == nb` is an
identity fill. `n_sel = keep * ratio + tail` is known on the host (`tail = (pos+1) %
ratio` at a single-token step), so the output buffer is allocated at the right size and the
step has no readback at all: the attention's gather reads the row buffer the kernel wrote.
Prefill (`n > 1`) keeps the host path — its overlay is a host-assembled mask anyway.

**The key.** Both arms rank by ONE function, `score_key` (a Rust copy in indexer.rs used
by `top_blocks`' comparator, a Metal copy in the kernel): a non-negative finite float
orders by its bit pattern, denormals included; a set sign bit or a NaN keys as 0. This
made the host comparator a true total order (it was `partial_cmp` with an `Equal`
fallback) and makes the two arms identical for every input, contract or not. The trap
on the way: the design's `as_type<uint>(max(score, 0.0f))` FAILED the tie sweep at
`nb 100 keep 50` — Metal's `max(-0.0f, 0.0f)` returned `-0.0`, whose bit pattern is
the LARGEST key, so a `-0.0` block was kept ahead of everything. The pure-bit key also
sidesteps flush-to-zero on the compare, which would have ranked a denormal equal to 0.

Kill switch `XWEN_QSA_HOST_TOPK` (presence-based; `XWEN_QSA_CLASSIC` implies it;
stripped by parity-gate.ts; a row in parity.md's qwen4exp table). Tests: the kernel against
`top_blocks` + `expand_into` bit for bit over nb ∈ {1, 5, 100, 511, 512, 513, 2000,
65536} × keep ∈ {1, nb/2, 512, nb} × tail 0..3 with heavily tied scores (exact zeros,
−0.0, denormals) and continuous ones; a tie-quota case spanning 500 stripes; NaN and
negative scores; the dispatch's bails; and the scripted-sequence test now runs THREE arms
(classic+host, cached+host, cached+device) with identical rows at every step.

**Bench** (thermal protocol; `--no-draft --raw -n 64 --stats`; before =
`XWEN_QSA_HOST_TOPK=1`, after = default; arm order alternated between rounds — r1
before-then-after, r2 after-then-before; 60 s between rounds; `pmset -g` printed
`powermode 0` at both ends; no other model process):

| prompt | before decode | after decode | before prefill | after prefill |
| --- | --- | --- | --- | --- |
| 1937 anchor (after arm) start → end | | 45.6 → 46.7 | | 960.2 → 972.3 |
| 3803 r1 / r2 | 33.1 / 33.0 | 41.1 / 44.1 | 835.7 / 834.8 | 832.0 / 837.3 |
| 7606 r1 / r2 | 33.9 / 33.3 | 44.2 / 45.0 | 718.0 / 686.5 | 695.3 / 716.5 |
| 15972 (one round; only 10 tokens decoded before EOS) | 32.0 | 41.7 | 589.2 | 596.6 |
| 32061 (one round) | 33.8 | 45.3 | 470.9 | 473.4 |

Anchor drift +2.4% (within the 3% flag). Above-budget decode now runs at 44-45 against
the 45.6-46.7 below-budget anchor at every length up to 32k: the QSA cliff is closed,
and the per-step cost no longer grows with context in the indexer. The 3803 r1 "after"
figure (41.1) is a 32-token sample (EOS) and the 16k rows are 10-token samples; the
prefill differences flip sign with arm order and are noise. Greedy (`--top-k 1`, 64
tokens) byte-identical between arms at 3803 and at 15972. The cooperative bin walk,
the prefill readback and the earlier profile's `ple` +3.2 ms above budget are the
ledgered follow-ups (TODO.md).

## 2026-08-30 (QSA decode, steps A+B) — block keys cached per complete block and the K/V row gather fused: Flash-Next decode above the 2048 budget 30.5 → 32.9 at 3.8k, 30.3 → 33.5 at 7.6k

The decode cliff at the QSA budget (TODO.md, measured earlier today: 46 tok/s below 2048
tokens, ~30 above) had three named costs, and this arc took the first two: the pooled
block keys were recomputed from EVERY raw key on every step in 12 layers, through a
`mean(1)` that candle routes to `fast_sum_f32_strided` (one two-thread threadgroup per
output — ~1.5 M per step at 4k), and the `Rows` gather ran 3 candle dispatches per
plane per layer. The third — one host readback of the scores per layer, 12 pipeline
drains per step — is step C and is still there.

**A. Block-key cache.** A complete block's key (pool → k_norm → rope at the block's
first position) depends only on its own `ratio` raw rows, which never change while they
sit below the cache length, so `IndexerCache` now carries a derived plane
`blocks: [max_ctx/ratio, 128] f32` plus `blocks_ready`. `select` builds only the
blocks `[blocks_ready, n_blocks)`, in one batch: at decode that is no key work on three
steps of four and one block on the fourth. Every `len` write goes through one private
`set_len` that clamps `blocks_ready` to `len / ratio` (truncate, rollback, reset);
`import_rows` resets it to 0 outright, because the rows below the import were replaced,
not kept. The plane is never exported — an image carries raw rows and the next `select`
rebuilds in one batch. Pooling and the per-head score sum no longer call candle's
reduce: `strided_sum` adds the narrows in the reduce's own two-thread order,
`(r0+r2)+(r1+r3)`, which is bit-identical to `mean(1)` / `sum(0)` at extent 4 (and
up to 5; refused above, where candle's simd fold orders differently).

**B. Fused gather.** `kernel_qsa_gather_{f16,f32}` (`src/ops/qsa_gather.metal`): one
threadgroup per (selected row, head), vec4 copies, strides passed in for the head-strided
cache view. Two dispatches per layer instead of six; a copy, so bitwise.

Kill switch `XWEN_QSA_CLASSIC` runs the old full recompute and the `index_select` chain;
a scripted test (`cached_block_keys_match_the_classic_recompute`: chunked prefill
crossing the budget, decode steps, a rollback, a truncate below a block boundary,
refills, export/import) holds both arms to identical selections and the plane to a
from-scratch recompute with 0 differing elements. Accounting: `indexer_bytes_per_token`
now takes the ratio and reports 640 B/token/layer (512 raw + 128 amortized block row);
`Model::kv_bytes_per_token` on Flash-Next 30720 → 32256.

**Bench** (thermal protocol: anchor arm first and last, arms interleaved classic-then-
fused per prompt, 60 s pauses between rounds; `--no-draft --raw -n 64 --stats`,
`pmset -g` printed `powermode 0` before and after):

| prompt | classic decode | fused decode | classic prefill | fused prefill |
| --- | --- | --- | --- | --- |
| 1937 (below budget) r1 / r2 | 45.9 / 45.9 | 45.7 / 45.4 | 985.8 / 997.5 | 961.3 / 960.2 |
| 3803 r1 / r2 | 30.5 / 30.4 | 32.8 / 32.9 | 841.4 / 833.1 | 837.6 / 824.7 |
| 7606 (one round) | 30.3 | 33.5 | 736.3 | 711.0 |
| anchor (1937, fused) start → end | | 45.6 → 46.2 | 988.9 → 962.4 | |

Anchor drift +1.3% on decode, within the 3% flag. Greedy text (`--top-k 1`, 3803
prompt, 64 tokens) byte-identical between arms. Caveat: every round ran classic then
fused, so the 1-3% lower prefill in the fused arm is order-confounded (the anchor's own
prefill fell 2.7% across the session with nothing changed) and is not yet a finding.
Per-step at 3.8k: 32.8 → 30.4 ms against 21.8 ms below budget, so ~8.5 ms of the cliff
remains, consistent with the 12 readback syncs step C removes.

## 2026-08-30 (prefill chunk) — the prefill chunk goes per architecture: 2048 on the MoE checkpoints (+10% Flash-Next, +8% 35B-A3B at 3.9k tokens), 512 stays on the dense 27B where 2048 is 5-6% slower

The prefill chunk had been a flat `const PREFILL_CHUNK = 512` in generate.rs and again in
serve/engine.rs since the fork, chosen for the dense 27B's attention working set. On an
MoE checkpoint the chunk is also the expert batch: Flash-Next routes top-10 of 512
experts, so a 512-token chunk gives each expert ~10 rows per `mm_id` gemm. The A/B:
`XWEN_PREFILL_CHUNK` (new, `ops::prefill_chunk_override`), `XWEN_BENCH=1`, `--no-draft
--raw --stats`, `-n 128`, the `prefill-4k.txt` fixture (3851 tokens) and its first half
(1962 tokens), arms interleaved within each round, one process at a time, first run after
the cold start discarded, `pmset -g` printing `powermode 0` (no high-power claim).

Flash-Next, three rounds, prefill tok/s (medians): at 3851 tokens **748 / 814 / 824 /
745** for chunks 512 / 1024 / 2048 / 4096; at 1962 tokens **883 / 933 / 951 / 957** (the
last two are the same single chunk there). Decode after the 4k prompt 27.4 / 27.5 /
27.9 / 27.1 — unchanged. 35B-A3B, two rounds, 3851 tokens: **2429 / 2429 at 512 → 2641 /
2627 at 2048** (+8.4%), decode 107-108 either way. Dense 27B, two rounds, 3851 tokens:
**650 / 599 at 512 → 608 / 571 at 2048** — 2048 is 6.5% and 4.7% slower within its own
round (the between-round drift is the usual duty-cycle settling; the arms were
interleaved so it cancels). Greedy output over 64 tokens on the 2k prompt is
byte-identical between 512 and 2048 on Flash-Next. Peak `phys_footprint` (polled at
1 s): Flash-Next 17.4 GB at 512 and 1024, 19.5 at 2048, 22.5 at 4096; 35B 9.4 → 11.3;
27B 41 → 44-46. No OOM.

The dense result explains the 4096 one. The sdpa mask and the attention score tile grow
with the square of the chunk; a checkpoint with no expert batch pays that and collects
nothing, and on the MoE ones it outruns the rows-per-expert gain past 2048. So the
default is per architecture — `Arch::prefill_chunk_default`: 2048 for `Moe` and
`Qwen4Exp`, 512 for `Dense` — read through `XwenModel::prefill_chunk` by every surface:
generate/chat/batch, serve (whose own constant is gone; the snapshot economics are
position-based, `SNAPSHOT_MIN_GAIN` 1024 tokens, and never assumed the chunk), and the
logits-dump ppl pass (which used to hard-code 512 to "match generate" and now reads the
same accessor). `parity-gate.ts` strips `XWEN_PREFILL_CHUNK` from the run env. The
35B's new chunk has not been re-graded by the parity gate; the last full run was at 512.
Deferred (TODO.md "prefill-chunk pass"): a 16-row `mm_id` token tile, the hc/shexp gemms
onto `dense_mm`, the f16 rescale chain at prefill, and the unattributed decode fall from
46.7 tok/s at a 530-token prompt to 27.4 at 3.9k context, seen in every arm alike.

## 2026-08-30 (later still) — the GDN mixer arc: a per-step profiler names three targets, two of them turn out to be its own brackets, and folding the beta|alpha projection buys +4.6-4.8% on Flash-Next and +8.8% on the 35B-A3B

`XWEN_GDN_PROFILE` (ae82696) is a per-step attribution of the gated-DeltaNet block.
Its decode line named three targets and two of them turned out to be the instrument.
A decode-specialized scan kernel is a wash end to end (44.7 against 44.6 tok/s,
byte-identical text) and stays opt-in, because an amortized bench prices the scan at
1.35-1.43 ms/token against a bytes-only floor of 0.98-1.02: it was already
bandwidth-bound. And `attn_qkv`, which the profiler ranked slowest of the three Q8_0
projections at 346 GB/s, streams at 510 in a rotate-arm sweep, the fastest of them.
What shipped is the smallest target on the line: folding the beta|alpha projection
into its own head (0261e17) is worth +4.6-4.8% on Flash-Next and +8.8% on the 35B-A3B
(105.1 to 114.4), prefill unchanged, the mechanism being one dispatch fewer per
layer. The lesson: that line RANKS steps, it does not PRICE them. The remaining GDN
work is dispatch-count fusion.

Full record, moved verbatim: [records/gdn-mixer-arc.md](records/gdn-mixer-arc.md) (249 lines).

## 2026-08-30 (P4, later the same day) — Flash-Next serves: the QSA indexer rows ride in `HostFullKv`, the PLE state rides on its own layer's snapshot entry, and the container goes to v4

P4 kept this checkpoint off two surfaces, and it was one question: what does a cache
image have to carry, with two answers. The QSA indexers' raw keys are
position-indexed, so a snapshot stores nothing and a rewind is an exact truncate;
only page-out moves bytes, beside the K/V planes they mirror. The PLE state has no
inverse, so it travels as data: a `PleImage` rides its own layer's entry through a
wrapper variant rather than a fourth kind, because that layer is ALSO a DeltaNet
layer and a flat variant would have dropped its recurrent state. The container goes 3
to 4, forced by the QSA half alone: its planes sit untagged inside the full-attention
record, so a v3 reader would report corruption on a file that is not corrupt. That
let `refuse_state_transfer` and the unservable messages go; `auto_fetch` and
`supports_drafting` stay closed. 1009 lib and 5 bin tests pass, but no serve-engine
harness runs a real model, so the real file through a real server is untried.

Full record, moved verbatim: [records/flash-next-serve.md](records/flash-next-serve.md) (160 lines).

## 2026-08-30 (same day, follow-up) — `xwen batch` was never able to run the new default; it now shares serve's fallback, and a `--model` file identifies itself on the one-shot path

Review of the default move found two holes in it.

**`xwen batch` cannot run qwen4exp, and the default move pointed it straight at it.**
The entry above lists `batch` among the modes that "can run it". They cannot: batch
prefills the items' shared prefix once and takes a cache snapshot there
(`batch.rs:515`), and an enum-scored field snapshots and restores around every option it
scores (`batch.rs:1518`/`:1527`). Both are `XwenModel::refuse_state_transfer` on
qwen4exp, for the same reason serve is refused. So a zero-flag `xwen batch` would have
downloaded 111 GB, loaded it, prefilled, and failed at the first snapshot with a message
about cache images. `XWEN_BATCH_NO_CACHE` is not an escape hatch either: it skips the
shared prefix and leaves the per-option snapshots, so it works right up until a schema
has an enum in it, which is worse than not working.

`BatchRequest::model()` now resolves an absent `"model"` to `Model::default_servable()`,
the same fallback serve uses, and refuses `Qwen3.8-Flash-Next` by name before the load.
The CLI prints the fallback line on stderr when the payload named nothing, mirroring
serve's. A hub test asserts the two defaults are the same checkpoint, so P4 cannot make
one servable and leave the other behind.

That made `Model::servable()` a predicate about two surfaces rather than one, and its doc
said the opposite in as many words ("the CLI one-shots (`generate`, `chat`, `batch`) run
the checkpoint fine"). Corrected, along with `hub.rs`'s `servable` doc and
`batch.rs:210`. `unservable_reason()` lost its trailing serve clause and is now the model
half only, so both refusals can quote it and add what they do with a cache image;
`unbatchable_message()` is batch's. The serve line quoted in the entry below therefore
reads slightly differently now. Neither message offers the other refused surface as the
way out, which the single message did (it sent an operator to `xwen batch`), and a test
pins that.

**A custom GGUF on `generate`/`chat`/`batch` was silently taking the default family.**
With `--model <gguf>` and no `--model-size`, the one-shots used `select.size()` for the
chat dialect, the drafter and the label, which after the default move meant Flash-Next
for any file at all. Serve has read this off the file since it started identifying
checkpoints. The rule moved into `XwenConfig::identify` (returning `Identity::Official`
or `Identity::Assumed`), `serve::engine::identify_checkpoint` became a thin mapping onto
`Target`, and `one_shot_checkpoint` in main.rs is the CLI's caller. `--model-size` stays
a cross-check that has to agree; a file that identifies as nothing still falls back to
`Arch::model()` with a line. Batch reads the payload's `"model"` as that cross-check,
since it has no size flag. The metadata open happens before the template knobs resolve
and before `resolve_model` can download anything, so a contradicting flag fails in
milliseconds.

Two smaller ones. `scripts/hf.ts`'s `officialModel` checked shard 1 only, so an
interrupted 111 GB fetch resolved as a cache hit and failed deep in the load; it now
requires every entry in `shards` and names the missing ones, which is what that key was
added for (TODO.md called it dead). And the README's model section still said "all
Q4_K_M" (Flash-Next is UD-Q4_K_XL), called the checkpoint "CLI-ONLY" (it is
`generate`/`chat` only), and claimed a status for "both checkpoints" from when there were
two.

998 lib tests and 5 bin tests pass.

## 2026-08-30 — Flash-Next becomes the default checkpoint; serve falls back to the 35B-A3B and says so

`#[default]` on `hub::Model` moved from `Qwen35BA3B` to `Qwen38FlashNext`. Every mode
that can run it now runs it with no flags: `generate`, `chat`, `batch`, `fetch`, and
`bun scripts/bench.ts`. Nothing about the checkpoint changed — it is the same
EXPERIMENTAL label, the same three P2 gates, the same 111 GB of shards.

Serve is the mode that cannot run it, so it does not. `Model::default_servable()` is the
new rule: `Model::default()` when the server can run it, otherwise `Qwen3.6-35B-A3B`. A
zero-flag `xwen serve` prints one line —

    xwen: Qwen3.8-Flash-Next cannot be served yet (the qwen4exp recurrent state (the QSA
    raw-key caches, the PLE conv window and its n-gram token history) is not carried by
    any cache image, and the server snapshots, rewinds and pages conversations out on
    its ordinary path); serving Qwen3.6-35B-A3B. Pass --model-size to choose.

— and serves the 35B-A3B, which is what it served yesterday. `--model-size flash-next`
is still a startup refusal: naming a checkpoint and quietly getting another one is the
failure `checkpoint_selectable` exists to prevent. The reason clause is now
`Model::unservable_reason()`, so the refusal and the fallback line cannot drift apart.

Two things that went differently from the plan. The fallback was specified as "the first
entry of `MODELS` that is servable" — that entry is the **27B**, not the 35B-A3B, because
`MODELS` is the order `/v1/models` prints in, not a preference order. Deriving it would
have cut every existing server's decode to about a quarter (25 tok/s against 104-107)
while looking principled, so `default_servable()` names its fallback. And the fallback
notice moved inside the "nothing named a model" branch: a config file with its own
`model` path is not falling back to anything and should not be told that it is.

`auto_fetch()` is deliberately unchanged, which leaves the one real cost: a zero-flag run
on a cold cache downloads 111 GB. That gate was always about a checkpoint arriving as a
side effect of a stranger's HTTP request, not about an operator's own zero-flag run;
`ensure_model` fetches all four shards after the same size notice every other checkpoint
prints, and resumes in place.

Drafting: the default checkpoint ships no sidecar, so the zero-flag path prints "decodes
without speculation (no drafter exists for its graph yet)" rather than the old "no
drafter available" line, which read as something having gone missing.

Scripts: `hf.ts`'s `defaultSize()` follows the binary to `flash-next`. `parity-gate.ts`
now pins `35b` explicitly — it has no llama.cpp oracle for the qwen4exp graph — and
`retune-draft.ts` was already excluded by `draftingSizes()`. `spec-equivalence.ts` keeps
its explicit three-checkpoint list.

Also removed: `hub::official_drafter()`, which had no callers (serve's `engine.rs` has
its own function of the same name).

994 lib tests and 4 bin tests pass. Two new hub tests pin the default and the fallback,
and the fallback test asserts the two must converge once the default becomes servable —
so P4 cannot land and leave serve on the older model by accident.

## 2026-08-29 (P3, later the same day) — Flash-Next prefill 239 → 796 tok/s and decode 37.8 → 45: a Q5_1 `mm_id` arm, four fused hyper-connection kernels, and a norm split across streams below 32 tokens

P2 closed hours earlier with prefill 3.3-3.5x behind llama.cpp on the identical file;
P3 opened on that gap. Everything below is the `UD-Q4_K_XL` file at a 530-token
prompt, arms interleaved, `pmset -g` reporting `powermode 0`, not a high-power claim.
A Q5_1 `mm_id` arm took prefill 239 to 443, the 43 layers whose down plane is Q5_1
having sent all three expert planes down the per-token fallback. Four fused hc
kernels replaced 17 glue dispatches and took prefill to 765-781, 1.75x. That fusion
cost 6% of decode by running one threadgroup per token, so below 32 tokens the norm
now splits across streams, bit-identical, taking decode 37.8 to 43.1. The PLE layer's
cost turned out to be page faults on a demand-paged 28.8 GB table, so a prefetch
thread rides sample time and decode reaches 45. End state against llama.cpp in the
same hour: prefill 795.7 against 789, decode 43.1 against 41.4, forced replay 186/192
with 0 hard.

Full record, moved verbatim: [records/flash-next-p3-kernels.md](records/flash-next-p3-kernels.md) (129 lines).

## 2026-08-29 — Qwen3.8-Flash-Next runs: P0 through P2 in one day, agreeing with llama.cpp at 189/192 forced-replay steps, 37.5-38.1 tok/s decode

The arc had been paused since 2026-08-26 with no runnable weights; Unsloth's quant
ladder, llama.cpp's merge and free disk unblocked it, and P1 and P2 ran in one day.
Header-parsing every published GGUF turned up an unwritten rule: `ffn_down_exps` is
640 columns, which fails every K/IQ block-size requirement, so the converter silently
demotes that plane on every publisher's file. `UD-Q4_K_XL` was chosen as the only
Q4-class file whose types we have kernels for, at the cost of 43 of 48 layers
carrying Q5_1 experts. The oracle moved to pin 6fe749801 and all three existing
checkpoints re-passed at it. The parity gate cannot run on this checkpoint, so U7
rebuilt the decode tier by hand with llama-server standing in: 189/192 agreeing, zero
hard mismatches, the three divergences near-ties. Decode 37.5-38.1 tok/s against
40.9-41.5 in the same hour; prefill 203.5 against 713.4, 3.5x slower and reproduced
to the centisecond. Correct, and honestly slow in one place.

Full record, moved verbatim: [records/flash-next-port.md](records/flash-next-port.md) (113 lines).

## 2026-08-19 (later still, same day) — Batch stops pinning the template effort at xhigh: `reasoning_effort` lands on items and defaults, refused per item on 3.6

The dialect arc gave every surface an effort knob except one. Batch's
`resolve_render` built its `ChatOptions` from `for_dialect` and overrode only
`enable_thinking`, so a batch item that enabled thinking on the 3.8 checkpoint always
rendered the xhigh preamble with no way to ask for `low` or `medium`. Spotted
reviewing the shipped arc's coverage; closed same day.

`reasoning_effort` is now a field on both the item and the request's `defaults`,
layering exactly as `thinking` does — item over defaults over the template's own
`xhigh`. On the wire it is a JSON string in the template's three spellings:
`ReasoningEffort` gained `Serialize`/`Deserialize` through its existing
`Display`/`FromStr`, so the JSON form accepts and rejects exactly what the CLI flag
does, and the key is known to the strict `deny_unknown_fields` wire types while
typos stay refused. On a 3.6 checkpoint a supplied effort — either layer — is
refused with the checkpoint named and the 3.8-template provenance stated, matching
the CLI startup error and serve's 400; batch's refusal lands PER ITEM, the module's
shape for validation failures (a defaults-level effort fails every item identically,
since the defaults reach the renderer only through the items — decisions.md
"Serving", the effort-refusal decision's second extension). Effort with thinking off
(batch's default) or alongside injected reasoning is accepted and inert/independent,
as on the other surfaces: the 3.8 template reads the level only inside
`enable_thinking`'s guard. Note the asymmetry the tests pin: an effort-ABSENT
thinking item renders the template's xhigh sentence; `medium` is the level that
renders no preamble at all.

**Verification.** `cargo build --release` clean; `cargo test --release` fully green —
849 lib + 3 CLI + 69 parity-harness tests passed, 0 failed (6 new batch tests).
`cargo fmt --check` clean. No model math touched; the parity gate was not run and did
not need to be.

## 2026-08-19 (later, same day) — The dialect arc's review pass lands three fixes: reasoning retention moves wholly to the renderer, --no-think rejects armed think budgets, and a request-level template effort on 3.6 is a 400

The arc below went through review before closing: one internal reviewer came back
clean, the external second-model reviewer (a different model family, per the working
rule that different families catch different bugs) found all three defects fixed here.
Nothing else changed; the docs entries this pass touches are annotated in place.

**Reasoning retention is decided ONCE, in the renderer, per dialect.** Both compat
normalizers (openai.rs, anthropic.rs) were stripping assistant `reasoning` from every
turn before the trailing assistant/tool run — a rule from before the dialect arc, when
the renderer dropped exactly those turns anyway, so stripping early was invisible.
After the arc it was load-bearing and wrong: the Qwen38 dialect's
`preserve_thinking=true` default (and the OpenAI kwarg `preserve_thinking: true` on
any checkpoint) asked the renderer to keep reasoning the normalizers had already
destroyed, and the three dialects disagreed — native replayed everything while the two
compat APIs silently didn't. Now NATIVE tools mode passes reasoning through on every
assistant turn and `chat.rs`'s dialect rule (`preserve_thinking || index > last_query`)
is the single owner of what renders; the `trailing_run_start` predicates are deleted.
The debug tools modes still drop reasoning everywhere, deliberately (they render the
pre-tools history). Noted in decisions.md: Anthropic's real API strips non-trailing
thinking and this dialect deliberately does NOT emulate that — it serves Qwen
checkpoints, whose 3.8 card recommends preserved thinking for agent workloads.
End-to-end tests in all three dialect files render a normalized multi-turn history
under both dialects: 3.6 drops the pre-last-query reasoning, 3.8 keeps it, and
`preserve_thinking: true` keeps it on 3.6 too.

**CLI: `--no-think` with a nonzero `--min-think`/`--max-think` is a startup error**
(both gen and chat arms, `ThinkArgs::check_think_budgets`). The ledger item had filed
the combination as "inert"; the review found it worse — the CLI arms the ThinkBudget
machinery unconditionally, so the ceiling would have waited for a `</think>` a
no-think reply never emits and eventually forced its wrap-up sentence and a stray
`</think>` into the answer. Serve already guards the same hazard by dropping the
budget when the prompt does not start in thinking; the CLI now refuses it up front,
same pattern as the `--raw` combos. TODO.md item annotated done, same day.

**A request-level TEMPLATE effort on a 3.6 target is a 400, not a silent no-op.**
`chat_template_kwargs.reasoning_effort` (OpenAI) and `reasoning_effort` (native)
reached the renderer but only the 3.8 dialect consumes them — on a 3.6 model they
changed nothing, contradicting both the CLI's startup error and the module's own
strict-kwargs stance. Both `prepare`s now take the resolved `Target` and refuse the
request with the model named; the OpenAI error points at the top-level
`reasoning_effort` field, which keeps its budget semantics on 3.6 and stays accepted,
as do kwargs `enable_thinking`/`preserve_thinking` (real 3.6 template parameters) and
the server-wide `[thinking] effort` default (an operator setting, inert-but-legal on
3.6 as before).

**Verification.** `cargo build --release` clean; `cargo test --release` fully green —
843 lib + 3 CLI + 69 parity-harness tests passed, 0 failed (5 new serve tests, 1 new
CLI test; the four tests that pinned the old normalize-level strip are rewritten to
pin the pass-through contract). `cargo fmt --check` clean. No model math touched; the
parity gate was not run and did not need to be.

## 2026-08-19 — The chat template becomes a per-checkpoint DIALECT: 3.8 gets its reasoning_effort preamble and preserve_thinking default, gen/chat gain --no-think and --reasoning-effort, and sampling defaults go mode-keyed (1.0/0.95/20 thinking, 0.7/0.80/20 instruct)

Two commits, one arc (a2e02d0, 205d9ba): the first makes the renderer dialect-aware,
the second surfaces the knobs on every entry point and re-keys the sampling defaults.
Together they close the 2026-08-14 divergence, where every default 3.8 conversation
rendered without the system instruction its template injects. `ChatDialect` comes
from `Model::chat_dialect()`, and four 3.8 behaviours are pinned by tests: the
verbatim `reasoning_effort` preamble, `preserve_thinking` defaulting true, no block
for an empty system message, and no inline `<think>` splitting. The CLI gains
`--no-think` and `--reasoning-effort`, and `TOKENIZATION_RULES_VERSION` goes 2 to 3.
Sampling defaults are keyed to thinking mode per the official cards, so serve's fixed
constants are gone and its keys resolve per request. On the wire one
`reasoning_effort` field drives both the think budget and the template level. 838 lib
and 69 binary tests pass; no model math changed.

Full record, moved verbatim: [records/chat-dialects.md](records/chat-dialects.md) (101 lines).

## 2026-08-15 (latest, same arc) — MTP Stage C: the graph is confirmed against llama.cpp by BYTE-IDENTICAL output, the sweep moves the 3.8's defaults to p_min 0.7 / depth 4 (+44-45% code, +37-38% chat over plain), and two stage-B claims do not survive being re-run

Stage C is the verification half; no implementation changed except two fitted
constants. The graph is confirmed by a stronger result than asked for: run against
llama.cpp at `p_min` 0 on both sides, the generated text is BYTE-IDENTICAL on both
fixtures, with acceptance 73.3% against 75.0% on code and 45.7% against 47.1% on
chat, so acceptance is compared over the very same continuation. Two harness traps
cost the first attempt: `llama-cli` runs conversation mode regardless of `-no-cnv`,
and the two `p_min` floors are different gates. The sweep then crossed `p_min` with
depth, 128 greedy tokens, interleaved, medians of 3 reps, and moved the defaults to
p_min 0.7 / depth 4: depth is the axis that pays, spanning 12% where the floor spans
1.8%, and a probe brackets 4 as a peak. At the shipped configuration, measured three
times in one session, 34.4-35.7 code and 33.1-34.0 chat against plain 23.7-24.8. Two
stage-B claims did not survive being re-run.

Full record, moved verbatim: [records/mtp-stage-c.md](records/mtp-stage-c.md) (115 lines).

## 2026-08-15 (superseded in part by Stage C above — its "+60%" is a single run at the old defaults, and its sampled-equivalence claim does not reproduce) — MTP drafting SHIPS on Qwen3.8-27B: 39.3 tok/s drafted against 24.5 plain in the same session (+60%) at 93.5% acceptance, and `--draft` is byte-identical to `--no-draft` under both equivalence modes

The checkpoint that had no drafter now has one: `src/mtp.rs` is the head, one extra
trunk-flavour full-attention layer with its own KV, `src/drafter.rs` the seam between
the two kinds, and `generate.rs` the chain round. The sync rule now lives in one
function: the head's KV row for position p is built from (token_p, hidden_{p-1}), and
`sync` owns the shift itself. The chain stays on the GPU, one readback at the end,
because Phase 0 priced a per-step CPU readback at +1.45-2.96 ms. Prefill pays for the
head and the cost is small, an interleaved A/B reading -4.2% at 839 prompt tokens and
-5.8% at 3286, the growth with position being the head's own mask. The first live
smoke read 39.3 tok/s drafted against 24.5 plain at 93.5% acceptance, single runs at
the old defaults, which Stage C supersedes. One limitation is ledgered rather than
hidden: the head cannot follow a rewind, keeping exactly one carry hidden.

Full record, moved verbatim: [records/mtp-stage-b.md](records/mtp-stage-b.md) (110 lines).

## 2026-08-15 — Phase 0 for MTP drafting on Qwen3.8-27B: the 3.6 DFlash head partially transfers but does not pay (0.86-1.02x vs the native head's 1.33-1.65x in the same session), and an MTP draft step costs 7-8.5% of a target forward

Measurement only, no shipped code changed, and everything is conditional on machine
state: `lowpowermode 1`, on battery, plain 3.8 decode reading 9.0-9.4 tok/s against
the 23.8 recorded the day before. No absolute here is comparable to another session;
both experiments rest on quantities a throttled machine cannot move. The 3.6-27B
DFlash head does attach to the 3.8 target and partially transfers, acceptance running
50-80% against the native head's 66-97% on the same prompts in the same hour, but
throughput lands at 0.99x code / 0.95x chat where the native pair runs 1.43x / 1.33x,
so it does not clear its own overhead. The second experiment: one MTP draft step
costs 7.1-8.5% of a target decode forward by timing and 8.19% by bytes, inside the
viability band. Two design inputs fall out: the lm_head is 51.8-53.6% of the step's
time and 69.8% of its bytes, and a per-step CPU readback costs +1.45-2.96 ms, so the
chain should stay on the GPU.

Full record, moved verbatim: [records/mtp-phase-0.md](records/mtp-phase-0.md) (93 lines).

## 2026-08-14 — Review round: a job now names a FILE, not just a checkpoint; drafting is resolved per checkpoint; a contradicting `--model-size` fails at startup

Two reviews of the entry below found one shared root behind most of their findings:
the arc had introduced a second kind of model identity, the served file's own id,
without giving the engine a way to represent it. A job now names a `Target`, a
checkpoint plus "is this the served file", and equality on it is file identity, which
the engine's swap check and the disk tier already wanted. That closed three real
bugs, among them an official name answered by unchecked weights and `/v1/models`
publishing an id its own resolver refused. `--model-size` becomes a startup error
against a file that identifies itself, instead of silently winning and then 500ing
every request; drafting resolves per checkpoint. A second pass found four more, two
introduced by the fixes and both the same mistake: a rule stated in a comment and
checked nowhere. 873 tests pass, with the live swap cycle verified by hand.

Full record, moved verbatim: [records/serve-target-review.md](records/serve-target-review.md) (119 lines).

## 2026-08-14 — Qwen3.8-27B added as a registry entry (same graph, no drafter), and the APIs go to full model names only: `/v1/models` stops listing one checkpoint three ways and an unknown `model` is a 400 instead of a silent default

Qwen3.8-27B released today, its `config.json` byte-identical to Qwen3.6-27B's, so it
is the same forward pass over different weights; the parity gate was deliberately not
run. `Arch::model()` said the GGUF architecture identifies the checkpoint one-to-one,
true until two releases shipped the same dense `qwen35` graph, so identification
became a chain: `--model-size`, then `general.name`, then the file name, then the
arch with a warning. The running server was listing `Qwen3.6-35B-A3B-Q4_K_M`, `27b`
and `35b`, three ids for two checkpoints, while an unrecognized `model` fell back to
the default. The APIs now speak `Model::full_name()` and nothing else; anything else
is a 400. The drafter became `Option<Drafter>`, most of the diff. Two facts were
checked rather than trusted: the tokenizers are NOT byte-identical, and the chat
template DOES differ and is now vendored. 869 tests pass, the checkpoint verified end
to end from an empty cache.

Full record, moved verbatim: [records/qwen38-27b-registry.md](records/qwen38-27b-registry.md) (84 lines).

## 2026-08-12 — The client's residual first-field escape report is reproduced and confirmed as conditioning signal: it follows position, not field identity, and the outside mass is almost entirely the answer in another spelling or a plausible alternative shape. No code change.

**Where it came from.** The escape-fix consumer graded 2e2280b: the 0.999 pin is gone
(their first-field median now 0.0015, in the expected range), but the first boolean
field of every multi-field item still reads 50-88× hotter on escape than its siblings
(their max 0.032 vs ~1e-5 later) — and since "first field" is whatever their TOML lists
first, it inflates per-category mean escape unevenly. They flagged it as a possible
position bias at the document's first choice point.

**Reproduced with the same instrument that settled the original bug** — a temporary
row dump at the exact `score_field` read (reverted after; scoring path untouched).
Three items, five boolean fields each, `include_score: "all"`, 505-token shared rubric
prefix, 35B-A3B default model, greedy. Items 1 and 2 classify a byte-identical email
and differ ONLY in field order (first two fields swapped) — the position-vs-identity
control the client's data can't run.

**It follows position, decisively.** No-think field-0 escapes: 0.0114 / 0.0349 /
0.0102; every later field 3.6e-6 to 1.7e-4. In the byte-identical-email control
(items 1 vs 2): swapping `urgent` out of slot 0 drops it 152× (0.0114 → 7.5e-5),
swapping `spam` in raises it 311× (1.1e-4 → 0.0349, just above the client's 0.032
max) — same email, same rubric, same options, only the order moved. Item 3, a
different email with item 1's order, independently repeats the shape (`urgent` 0.0102
at slot 0, `spam` 4.2e-5 at slot 1). The hot slot moved with the order, not the name.

**The outside mass is exactly what the 2026-08-11 entry said it was.** At every
field-0 read the top eight outside tokens carry 99.3-99.7% of the outside mass in the
no-think run (90-99.9% with thinking) and they are all plausible openers: ` "` dominates (52-73% of outside — a quoted string where a boolean
belongs), then ` True`/` False` (25-46% — the ANSWER in Python capitalization, which
would invalidate the JSON and honestly counts outside), then traces of ` {"`, ` [`,
` null`. Nothing diffuse, nothing unrelated. The contrast that is the whole mechanism (item 1's
numbers; the shape repeats on all three):
field 0's argmax is ` true` at 0.986 (space-led, INSIDE) with bare `true` at 0.0024;
by field 1 one compact `"k":v,` exists and bare `false` reads 0.9995 with the entire
outside class at 1.1e-4. Style unpinned at the first choice point, fully pinned by the
second — the 0.00197-vs-0.000051 repro pair from the fix, now at a second data point
with a different schema.

**The thinking path is exonerated, and thinking amplifies the signal.** The 2026-08-11
entry's standing suspect for field-0 anomalies (the `held` reasoning-tail
reconciliation, the one genuinely field-0-specific shape in the assembly loop) is
cleared two ways: the no-think run has no reasoning tail and shows the full elevation;
and with `thinking: true` (needs budget — max_tokens 1200 refuses because the
reasoning block can't close, 6000 runs) field-0 escape rises to 0.042-0.109 (three
raw per-item values, not an aggregate; 35B only) while
later fields stay ≤1.1e-4, with ` "` still 71-99% of outside. After 5-6 KB of prose
reasoning the model is LESS committed to the compact skeleton's style — the same
signal reading louder, not a defect.

**Verdict passed back to the client.** Not an xwen bug; a real property of the
measurement they can either embrace or column-split (their with/without-first-field
columns are the right call). One nuance ledgered as a candidate refinement: much of
the outside mass everywhere is format drift of the same answer, not value
disagreement — space-led ` True`/` False` are 25-46% of field-0 outside mass, and
bare `True`/`False` are 28-87% of the (much smaller) later-field outside — so escape
overstates value-level disagreement wherever it is large enough to matter, which
today means first fields.

**Client follow-up, same day.** Their data reproduces the think-vs-plain field-0
elevation on MEDIANS (4.7× on the 35B, 1.5× on the 27B) but not on means: the 27B's
no-think arm is heavy-tailed (max 0.109 — reaching without thinking what the 35B
needed thinking for; we never ran the 27B). Our per-item think/plain ratios
(9.2/3.1/4.1×) median to ~4.1×, corroborating their 35B figure. Their report records
medians as the figure to read; agreed — n=3 here says nothing about tails. Dump logs kept in the session scratchpad; working tree
verified clean after the revert. The 2e2280b parity-gate run is STILL pending — the
GPU was left free for the client's own benchmark runs (contended prefill reads 3× low;
see "Benching rules").

## 2026-08-11 — First client feedback lands: the scored-field escape stops lying about first fields, `shared_prefix` collapses 14 POSTs into one, the body cap goes to 100 MB, and max_ctx becomes a ceiling (lazy KV, 128k CLI default)

The first external consumer of the scored batch path reported a bug, asked for a
feature and asked for three operational changes. The bug: on multi-field items the
FIRST field's `escape` read 0.999-1.000 while non-first fields sat at 1e-7 to 4e-4. A
row dump found the model putting 54.9% on ` true` and 44.9% on ` false`, single
tokens carrying a leading space, against ~5e-5 on the bare spellings the opener set
held: the measure read the model's preferred spelling of the ANSWER as none of the
above. The fix reclassifies the whole next-token row by raw BYTES, and first-field
escape goes 0.9999 to 0.00197 with scores bit-identical. `shared_prefix` ships as a
wire-size field, not a prefill feature, and the body cap goes 2 MB to 100 MB. max_ctx
becomes a ceiling rather than an allocation, caches starting at 8192 positions and
doubling on demand, so the CLI default rises to 131072. 791 + 69 tests pass; the
parity gate refused to start on a held GPU and is pending.

Full record, moved verbatim: [records/client-feedback.md](records/client-feedback.md) (136 lines).

## 2026-08-11 — `/xwen/v1/batch` ships and the server stops being pinned to one checkpoint: every request names its model, the engine swaps lazily

The batch runner had been CLI-only since 2026-08-09 with the HTTP endpoint parked in
the ledger. The ask that unparked it also reshaped it: the endpoint should not be
pinned to the served model, so `--model` decides only the compat dialects' default
and any checkpoint lazy-loads. `EngineState` now records which checkpoint it holds,
every queued job names the one it needs, and a mismatch images the live conversation
out through the idle-unload path before the swap, keeping one model resident. `POST
/xwen/v1/batch` takes exactly the document `xwen batch` reads on stdin and rides the
ordinary queue as a second `Job` variant; the runner grew a progress callback and a
cancellation poll. A two-family review found six fixes, the one real correctness bug
being new to the arc and opened by it: a chat job's thinking budget leaked into the
next batch job, which before this could not exist. 781 + 69 tests pass, and the first
live exercise swapped 35B to 27B inside one server.

Full record, moved verbatim: [records/serve-batch-multi-checkpoint.md](records/serve-batch-multi-checkpoint.md) (124 lines).

## 2026-08-09 — `xwen batch`: one prefill for N items, and scored fields that report the model's confidence instead of a sampled token

Two research passes opened the arc, converging from opposite directions: the cheapest
batch prefills its shared prefix once, and the accurate way to label a document along
nine taxonomies is nine narrow questions. So `xwen batch` reads one JSON request on
stdin, prefills the items' longest common token prefix once, snapshots it and replays
it per item. The scored path is the other half: the engine writes the JSON skeleton
itself, teacher-forces it, and scores every allowed option at each choice point, so
the model only ranks values. The outside-model review found the flaw that mattered,
that scoring an option's own tokens makes a strict-prefix option unbeatable; the fix
scores one terminator token past the value. The demo then produced the arc's thesis
as a measurement: the scored boolean vector beat the free-decoded array over the
identical tag set on both checkpoints, and the gap did not close when the rubric was
sharpened, so it is structural.

Full record, moved verbatim: [records/xwen-batch.md](records/xwen-batch.md) (213 lines).

## 2026-08-09 — `--stats` splits decode by round class: plain / drafted / full-accept, with an in-run plain baseline and an estimated net drafting effect

**The old block was arithmetically correct and read backwards.** `draft` and `verify`
both divided by ALL rounds, so a run whose controller paused 93% of its rounds printed
`verify 12ms/round` next to a draft cost averaged over rounds the drafter never ran.
That reads as "verification is slower than plain decode", when per committed token the
verified rounds were the cheaper ones and the drafter had barely fired. Nothing in the
block answered the question the block exists for: did drafting help THIS run, and by how
much.

**What ships.** `SpecStats::bucket_round` folds every round into one of two classes as
the loop finishes it. **Plain**: no draft block was verified — paused rounds,
empty-draft fallbacks, serial thinking rounds, rounds past the drafter's context — one
committed token each. **Drafted**: a block was verified; the round's full wall time,
draft phase included; tokens are accepted plus the bonus. **Full-accept** is a subset of
drafted, the rounds where every drafted token survived, which is the ceiling a longer
block could reach. Two counters come with it: `spec_draft_ms`, so `draft_ms -
spec_draft_ms` is drafter time spent on rounds that ended plain, and `draft_rounds`, the
rounds the drafter forward actually ran. One shared bucketer is called from both the CLI
loop (`generate_spec`) and serve's (`decode_loop_spec`) — serve's printer is unchanged
and feeds the bucketer its own emit-excluded `round_ms`.

**The design decision worth naming: the plain bucket excludes wasted drafter time.** It
folds `round_ms - draft_ms`, which is exactly the quantity the pause controller's plain
comparator folds. It is deliberately not "what the round cost" — it is what a plain
decode step costs on this run's text, so `plain_rounds / plain_ms` is an interleaved
plain baseline measured inside the run being reported. That immunity to between-session
drift is the point: the entry below has to warn against differencing a drafted 27B
figure against a plain one from another session, because the 27B's level moves. This
baseline never crosses a session boundary.

**Derived lines.** `drafting: X.XXx vs plain on drafted rounds` is the drafted bucket's
token rate over the plain rate. `est. net ±Y.Y%` prices every committed token at the
run's own plain rate and compares that against the decode time actually spent — plain
forwards, plus full drafted rounds, plus the wasted drafter ms — three terms that
partition decode-loop model time with no double counting. Both need `plain_rounds >= 8`;
a handful of plain rounds is noise, not a rate. Zero-round buckets print nothing (a
margin-0 run has no plain rounds, a fully paused one no drafted ones). The misleading
denominator is fixed in the same pass: `draft` now averages over `draft_rounds` and says
`ms/draft`. `bucket_round_partitions_time_and_tokens` pins the partition offline.

**What it looks like** (35B, thinking-heavy code prompt, smoke run, `lowpowermode 0`):

```
decode:  768 tokens in 7.82s (98.2 tok/s)
         plain:         703 tok in 7.09s (99.2 tok/s over 703 rounds)
         drafted:        64 tok in 0.72s (88.5 tok/s over 16 rounds, 4.0 tok/round)
         full-accept:    31 tok in 0.25s (123.2 tok/s over 6 rounds)
         drafting:    0.89x vs plain on drafted rounds; est. net -1.0% overall
spec:    719 rounds (701 paused), 95 drafted, 48 accepted (50.5%), 47 rejected
         814 verified positions; draft 0.2s (10ms/draft), verify 7.7s (11ms/round)
```

One run, not a benchmark — but it is the shape the instrument exists to make visible:
701 of 719 rounds paused, and across the 16 that drafted the token rate came in below
this run's own plain rate.

## 2026-08-08 — the small-batch window reaches the attention and DeltaNet projections (verify span 8 −12.0%), and the first real controller sweep makes `draft_p_min` per-checkpoint: 0.5 on the 27B, +11-13% over the shipped 0.3

Two arcs, the second the one the first predicted. Arc 1 extends the `mul_mv_ext`
window to the q8_0 attention and DeltaNet projections, one `Proj` variant covering
seven tensors on every layer of both checkpoints. The verify forward on the 27B at
`n_past` 512 goes -12.0% at span 8, -6.0% at 6 and -5.0% at 4, pooled medians of two
sessions; span 2 is a wash rather than the small regression the table shows, because
the spans where the kernel cannot run carry the same +2% ordering bias. Both gates
ALL PASS at pre-change numbers. Arc 2 builds `scripts/retune-draft.ts` so the
protocol is a script rather than folklore, and runs two 120-run sweeps. They agree:
`draft_p_min` becomes per-checkpoint, 0.5 on the 27B (37.3 against 33.0 at the
shipped 0.3) and 0.3 on the 35B, the mechanism being that the 27B stops pausing
entirely at 0.5. `pause_margin` had never been swept and now has, staying a shared
1.0 for a measured reason.

Full record, moved verbatim: [records/small-batch-window-projections.md](records/small-batch-window-projections.md) (180 lines).

## 2026-08-08 — `mul_mv_ext` ships: the verify forward at span 2 goes 153 → 61 ms, drafted decode gains +11.6-13.2% on the 27B, and the kernel is 20-400x MORE accurate than the mm it replaces

The entry below put 87.6% of the verify round's ~149 ms fixed cost in the dense FFN's
matmuls at small M and priced the fix by byte arithmetic. This arc built and measured
it. `src/ops/mv_ext.metal` vendors llama.cpp's multi-row quantized mat-vec, which
dequantizes a weight block once and reuses it across 2-5 output rows, routed at seq
2..=8 from one decision point, `QLinear::forward`. The verify forward on the 27B goes
153.4 to 61.5 ms at span 2 and 220.1 to 161.2 at span 8, interleaved arms, and
drafted decode gains +11.6% on the 27B code prompt and +13.2% on chat, +4.2% on the
35B code prompt and nothing on 35B chat, which is pause-dominated. The accuracy
result runs the other way from `dense_mm`, worth stating because the reflex is now to
expect a precision cost: this kernel is 20-400x MORE accurate than the mm it
replaces, being f32 end to end. Extending the window past 8 is REFUTED: spans 16, 24
and 32 read 1.11x, 1.42x and 1.69x worse than classic.

Full record, moved verbatim: [records/mv-ext.md](records/mv-ext.md) (101 lines).

## 2026-08-08 — Verify-round diagnosis: the ~149 ms fixed cost is the dense FFN's matmuls at small M, and none of the armed machinery it was blamed on

**Diagnosis only.** The fix arc is in flight and will write the entry that ships it;
this records what was measured, so the refutations do not have to be re-derived.

**Context.** P9a left the verify round's ~149 ms fixed cost as the new spec-decode
ceiling, with five unpriced candidates: checkpoint materialization, rollback restore,
the trail's host-side conv slices, full-span logits + readback, and command-buffer
syncs. The ledger item's instruction was "price the stages before attacking any of
them". `spec-verify-bench` grew per-stage sync brackets and per-span stack-profile
dumps to do that. Conditions throughout: 27B Q4_K_M, `n_past` 512, `lowpowermode 0`,
medians over 20 reps.

**All five candidates are refuted.** Checkpoint arm costs 5.7 ms fixed — the ~157 MB
of per-round materializes are cheap. Rollback is 2.6 ms fixed + 0.43 ms/tok, and a
keep-4 vs keep-0 branch comparison shows no difference at all. Full-span logits and
readback are 0.12 ms + 0.099 ms/tok, with a last-row-materialize variant reading a flat
~0.4 ms. Against ~149 ms, the entire armed apparatus is a rounding error.

**The cost is inside the verify forward, and it is there unarmed.** Fitting spans 2-32
puts the forward's own fixed cost at ~161 ms, and a span-2 UNARMED forward measures 152
ms against a ~40-43 ms plain step. Speculation does not cause it: a 2-token forward is
just ~3.7x a 1-token forward. Stage decomposition of that span-2 forward against a plain
seq-1 step, both stack-profiled under an identical sync regime, puts **131.8 vs 33.9 ms
in the dense FFN — +97.8 of the +111.7 ms total, 87.6% of it.** lm_head adds +4.4 (3.2x),
mixer_delta +5.9, mixer_full_attn +2.7, and every other bucket is under a millisecond.

**Mechanism: candle's `mul_mm` collapses at small M.** At seq 2..=32 every quantized
matmul takes the tiled path, whose grid degenerates to `ne01/64` threadgroups — ~73 GB/s
effective against ~280 GB/s on the seq==1 mat-vec path. Two refutation rounds confirm
the shape of the problem rather than a mis-set threshold. Forcing the vendored dense
gemm onto small spans (`XWEN_DENSE_MM_MIN_SEQ=1`) moved the fixed intercept only −3.3
ms, because the cooperative-tensor gemm has the same small-M occupancy collapse — though
its marginal did improve, 2.40 → 1.63 ms/tok. And `XWEN_MM_ID_MIN_SEQ=1` on the 35B was
strictly worse at spans 2-8, by +4.1-4.4 ms. No kernel currently in the tree wants these
shapes, so there is nothing to retune toward.

**The fix in flight** is llama.cpp's `mul_mv_ext` multi-row mat-vec: dequantize once and
reuse across 2-5 output rows (ggml-metal-ops.cpp:2120-2223, `ne11_mm_min` 8). Byte
arithmetic says it wins at spans 2-8 and washes by ~16. `src/ops/dispatch.rs:330-334`
already documented this exact gap and pointed at a TODO.md item that did not exist; the
annotation on the fixed-cost item is now that item.

**One consequence worth flagging before the fix lands:** it inverts the `p_min` retune
item. "Longer drafts amortize better" was reasoned off a dominant fixed cost, and
`mul_mv_ext` attacks precisely the short spans, so that tuning conclusion has to be
re-derived rather than carried. Cross-referenced in the ledger, and the retune should
wait.

**Also from these sweeps: the span-48 superlinearity has a suspect.** It is
arming-dependent — every checkpoint-on run overshoots its spans-2-32 extrapolation by
1.54-1.65x while both no-checkpoint runs come in UNDER at 0.80-0.91x — and it does not
move with the dense-mm or mm_id knobs. The profiled armed-minus-unarmed `mixer_delta`
delta grows 6.8 → 160.6 → 304.5 ms at spans 2 / 32 / 48, tracking the K-snapshot plane
buffer (~3.15 MB per plane per layer, ~100-150 MB/layer at spans 32-48). So it is trail
memory pressure, not a kernel threshold. Still outside the production regime and still
unchased. A separate observation from the same sweeps stays unexplained and is NOT
arming-dependent: lm_head roughly doubles at span 48, 7.0 → 13.1 ms, in both armed and
unarmed profiled runs.

## 2026-08-08 — The 27B prefill residual is real and lives in the pipelining: per-stage syncs find only +103 of the +410-438 µs/token, and both cross-chunk accumulation and command-buffer batching are refuted as its mechanism

The dense-FFN gemm arc closed the 27B prefill gap and handed off one number it could
not attribute: +350 to +560 µs/token outside every measured stage, growing with
length. This arc built the instrument that item asked for and ran it, and shipped a
diagnosis, not a fix. `XWEN_STACK_PROFILE` splits each chunk's wall clock into the
stages `run_stack` runs, bracketed by device syncs, with a host bucket for the gaps.
The residual reproduces twice, +410.3 and +437.9 µs/token between the 880- and
3851-token fixtures. Under per-stage serialization the same length delta is only
+102.8, of which attention is +53.5 and the FFN an unexplained +42.2 on a stage that
is length-independent by construction. So about 335 µs/token exists only when the
stages pipeline, the central finding. Two candidate mechanisms are refuted by direct
A/B: cross-chunk accumulation, and command-buffer batching over a 100x range. Two
hypotheses survive, both needing a counter candle does not expose.

Full record, moved verbatim: [records/27b-prefill-residual.md](records/27b-prefill-residual.md) (140 lines).

## 2026-07-29 — K-snapshot fused verify lands: spec decode goes from single digits to +8-21%, the 35B flips from -12% to +13-20%, and --draft becomes opt-out

**Context.** P9 shipped speculation as a 27B-only single-digit win with the 35B losing
12%, and its own annotation named the cause: under an armed rollback trail a multi-token
verify chunk fell back to the frozen reference scan, token by token — 39 ms per verified
position on the 27B, so the 48-of-64 DeltaNet layers got no batching win at all inside a
verify forward. TODO.md P9(a) called the K-snapshot fused verify "the precondition for
speculation to pay, not an optimization of it". This arc built it.

**What shipped.** The two fused scan kernels (`kernel_delta_scan`, `kernel_delta_scan_v2`
in `src/ops/delta.metal`) now optionally spill per-token states: a new
`ops::delta_scan_with_trail(.., state_planes)` widens the state output to
`[planes, v_heads, 128, 128]`, most-recent-first (plane s = state after token
`seq-1-s`), mirroring llama.cpp's `kernel_gated_delta_net` K>1 snapshot slots
(ggml-metal.metal:2740-2749) so the CPU oracle stays diffable. Plane 0 is the unchanged
after-loop store — at planes = 1 (every unarmed prefill and decode call) the kernel is
byte-identical to before, proven by a bitwise test. The armed clause is gone from
`linear_attn.rs`'s fused gate; an armed chunk runs the fused scan with planes = seq and
builds the rollback trail from unmaterialized plane views (delta) plus the same
host-side conv-stream slices the reference records. `XWEN_DELTA_CLASSIC=1` still routes
everything, armed chunks included, to `forward_classic` (untouched, still the frozen
oracle). `n_planes` rides the args struct, not a specialized pipeline. Details in
decisions.md "Model math" (the superseded armed-chunk entry).

**Verification.** 700 lib + 66 parity-harness tests green (4 net new; the test encoding
the old fallback was rewritten to assert the new invariant, and the new trail test was
mutation-tested — flipping the kernel's slot mapping fails it while every pre-existing
scan test stays green, so it is not vacuous). Two-model-family adversarial review
(Claude + Codex gpt-5.6-sol at xhigh): zero correctness findings; one doc-accuracy nit
each, both fixed. Both parity gates re-ran and pass with numbers identical to the
pre-change run (35B mm cos 0.999631, Δnll 0.000791; 27B all tiers) — the schema is
untouched.

**Measured** (`lowpowermode 0` — this machine exposes no `powermode` key, so high-power
is never claimed; warm, one model process at a time, interleaved arms, greedy, 128 new
tokens, `p_min` 0.3; two independent end-to-end runs, per-rep values in the raw logs).
The verify A/B is same-day and same-harness: `XWEN_DELTA_CLASSIC=1` IS the pre-P9a
verify path, and that arm reproduced the historical 245 ms @ span ~6 baseline.

- 27B verify round (`n_past` 512): fit over spans 2-32, the marginal cost fell
  **9.42 → 3.57 ms/position** (2.6x) over a fixed cost of ~171 → ~149 ms. At span 6:
  244 → 187 ms/round. In-loop `--stats` corroborate (~248 ms/8.3-position round →
  ~182 ms/7.3).
- 27B end-to-end: code **+19.3 to +21.0%** (29.7-30.0 vs 24.8-24.9 tok/s median), chat
  **+7.6 to +8.4%** — up from +4.8-6.8% / +1.5-7.4%. Acceptance 83.2% / 75.2% (down
  ~4 points from pre-P9a: the batched verify reassociates sums and accepts a slightly
  different token set; throughput improved regardless).
- 35B-A3B end-to-end: code **+18.1 to +19.8%** (124.0-126.7 vs 105.0-105.8), chat
  **+12.6 to +12.8%** — from **-11.5/-12.7%**. The mechanism is the pause controller,
  not the drafter: 35B code went 54-of-66 rounds paused → **0-of-20 paused, 159
  drafted**. The ledger's attribution of the 35B loss to the drafter cache sync (P9b)
  was measured right but read wrong — the ~1.2 ms sync is only fatal when the
  controller pauses and pays it for nothing; cheap verify made it stop pausing.
- spec-equivalence: six of eight comparisons byte-identical; the two 27B-chat forks are
  at the SAME points with the SAME words as the pre-P9a run (the known near-tie class).
  No desync — the sampled stream stays in step for 100+ tokens before forking.

**The `--draft` default flipped to opt-out** — P9(d)'s pre-registered bar ("the 35B at
or above plain, not merely closer") is met with margin on both prompt kinds in both
runs. Zero-flag `generate`/`serve` now load the dflash sidecar; `--no-draft` opts out.

**New open items** (ledgered under P9/TODO): the verify round's ~149 ms fixed cost
(~113 ms above a plain step, now ~60% of a typical round and the new ceiling — price
checkpoint/rollback/readback before attacking); `p_min`/`pause_margin` retune against
the new cost curve (0.3 was fitted to the reference-scan curve); an unexplained
superlinear jump at span 48 in every verify arm (outside the production regime —
block_size 16 caps real spans near 17 — refuted as the dense-mm threshold, cause
unknown); `spec-verify-bench.rs` shipped broken (fixture id `long-swa` predates the P7
rename to `long-mixed` — it had never run in this repo; fixed this arc). The flip's own
review (Codex, clean on the diff) surfaced two default-on consequences, both ledgered:
serve slots persisted without drafter planes now silently decode plain forever while
reporting draft ON (the common hydration path against `--no-draft`-era slots, not the
flag-change edge the code comment was written for), and a custom `--model` GGUF that
fails the drafter preflight now hard-errors at startup where it previously ran plain.

## 2026-07-29 — The 27B prefill gap was the dense FFN's gemm, not the DeltaNet scan: a Q4_K cooperative-tensor kernel takes prefill from 263 to 702 tok/s @925

Two entries below, the DeltaNet arc refuted its own premise and handed off a
question: the 27B's prefill loss is not in the DeltaNet layers, and the next arc
should start from a per-stage profile. It did. The dense SwiGLU FFN is 66-85% of
prefill wall, a band because that row is derived from a rate that is 7-8%
pessimistic, and the mechanism is not subtle: the FFN runs candle's Q4_K mm at ~12
TFLOP/s where the same shapes through the Metal-4 cooperative-tensor gemm do 28-36.
That it is kernel efficiency and not a memory wall is settled without appealing to
any peak: the Q4_K arm moves 3.6x fewer weight bytes and takes 2.4x longer. So
`dense_mm.metal` is that gemm reading Q4_K directly. Prefill goes 263 to 702 tok/s at
925 tokens and 200 to 445 at 4k, decode unchanged. The cost, stated plainly: the
kernel is LESS accurate than the chain it replaces, ~4.1e-4 against ~1.9e-4, so it is
pinned off both sides of the strict tier and graded by mm, decode and ppl.

Full record, moved verbatim: [records/dense-ffn-prefill-gemm.md](records/dense-ffn-prefill-gemm.md) (198 lines).

## 2026-07-29 — DFlash adapted to the Qwen sidecars (P9): both drafters load and accept 85-95%, and speculation is a 27B-only win because the verify forward runs the per-token reference scan

The DFlash subsystem came over from laguna whole and inert, describing a drafter the
Qwen sidecars are not, with two deliberately-red tests as the gate. Reading the
oracle against the shipped headers found three graph differences nobody had briefed:
the noise block is non-causal, the injection path applies no `attn_norm`, and the
encoder is three ops. Both drafters load and propose well, 85-95% acceptance, but
speculation is a 27B-only win: +4.8-6.8% on code and +1.5-7.4% on chat there, against
-11.5% and -12.7% on the 35B-A3B. The 35B's loss is not a drafting failure, and the
arm that proves it drafts nothing: at `--draft-p-min 1.1` it still decodes at 92.6
against 105.1 plain. The cost is the mandatory per-round cache sync, ~1.2 ms, 12% of
the 35B's step and 2.8% of the 27B's. The deeper ceiling is that an armed multi-token
chunk falls back to the frozen reference scan, which makes the K-snapshot fused
verify a precondition for P9, not an optimization.

Full record, moved verbatim: [records/dflash-drafting.md](records/dflash-drafting.md) (160 lines).

## 2026-07-29 — The DeltaNet scan is 3% of 27B prefill: llama.cpp's decomposition measured slower, and the premise behind P8b refuted

**Context.** The head-to-head entry below named the sequential DeltaNet scan as the cause
of the 27B's 1.8-2.1x prefill loss, and a mapping pass then established that llama.cpp's
Metal path does NOT run the chunked form it advertises — its fused
`ggml_gated_delta_net` op pre-empts the chunked graph (delta-net-base.cpp:437-446), so it
runs the same sequential scan we do, under a far more parallel decomposition. That made
re-decomposing our kernel the obvious lever and demoted P8b's chunked scan. This arc
built the re-decomposition. It lost, and finding out why cost the premise as well.

**Built.** `kernel_delta_scan_v2` is llama.cpp's shape adapted to our layouts: one
SIMDGROUP owns one state value-column for the whole T loop, both key-dim contractions
collapse to `simd_sum`, no barrier appears anywhere in the timestep loop, and the grid is
1536 threadgroups at the 27B geometry against the shipped kernel's 192. It needs q and k
pre-normalized, so `kernel_delta_l2norm` hoists the L2 clamp-norm out into its own
dispatch. Both are bounded against the frozen reference scan exactly as the shipped
kernel is, at the same tolerances — no parity schema change, floors untouched.

**Measured, and it lost.** Isolation timing per DeltaNet layer (`delta_scan_timing`,
plateau ms, each arm inclusive of its q/k norm), and interleaved end-to-end (median of 3
rounds, arm order flipped each round, full power `lowpowermode 0` start and end,
`XWEN_BENCH=1`, `--no-draft`, committed fixtures):

| isolation, one layer | 27B @880 | 27B @4096 | 35B @880 | 35B @4096 |
|---|---|---|---|---|
| shipped | 1.97 | 8.56 | 1.57 | 6.31 |
| llama.cpp shape | 2.73 | 14.81 | 1.88 | 8.93 |

| end to end, tok/s | 27B shipped | 27B v2 | 35B shipped | 35B v2 |
|---|---|---|---|---|
| prefill @880 | 296.9 | 307.6 | 2504.9 | 2497.2 |
| prefill @3851 | 262.1 | 257.7 | 2312.3 | 2398.6 |
| decode n=256 @630 | 22.0 | 22.1 | 103.3 | 102.2 |

The end-to-end arms are a tie in both directions — which is the actual finding, not a
measurement failure. **The scan is 3% of 27B prefill.** 48 layers × 1.97 ms is 95 ms of a
2.96 s prefill at 880 tokens; × 8.56 ms is 411 ms of 14.2 s at 3851. A free scan buys
~297 → ~307 tok/s against llama.cpp's 486. The 1.7x isolation regression at 4096 shows up
end-to-end as exactly the 1.7% it should, which is the cross-check that the share is
right. The mechanism of the loss is q/k read amplification: our threadgroup stages one
normalized q and k vector per threadgroup per timestep, llama.cpp's shape gives every
simdgroup its own copy — 32x the L2 traffic — and at head dim 128 each lane owns 4 state
entries, so two `simd_sum`s ride on 8 useful FMAs. Hoisting the norm alone loses too: the
extra dispatch rewrites the whole q|k plane (0.52 / 1.80 ms) to save a reduction already
amortized over a threadgroup.

**Shipped: nothing.** The scan kernel is byte-identical to what it was before this arc,
which both gates confirm by reproducing every number exactly (35B strict 1.000000, mm
0.999631, decode 63/63/62 with 0 mismatches, ppl Δnll 0.000791; 27B strict and mm both
1.000000, decode 64/64/64 with 0 mismatches, ppl Δnll 0.000330). `parity-gate.ts`'s
`baseEnv()` strips `XWEN_DELTA_SCAN_V2` alongside the other presence-based switches —
unlike its siblings it has no provenance field, so a stray shell value would have applied
to both sides and passed the gate while grading the wrong kernel. The two new kernels stay
behind `XWEN_DELTA_SCAN_V2=1` as a runnable refutation — the kernel they mirror is
vendored in this repo and will invite the same proposal again — along with
`ops::delta_l2norm`, its bounded test, and the `delta_scan_timing` bench that produced
the table. 681 lib tests pass with the two dflash `real_file` tests staying deliberately
red, 63 parity tests pass (3 ignored, the ones the gate feeds), and the delta and
linear-attn suites pass under `XWEN_DELTA_SCAN_V2=1` as well as under the default.
`scan_matches_reference` picked up seq 67 — prime, and a multiple of no tile or simd
width the scan is built from.

**What this hands off.** The 27B prefill gap is still ~1.8-2.1x and is now known NOT to
be in the DeltaNet layers. It is in the dense projections; that is the next arc's
question, and it should start from a per-stage profile rather than from a reading of
llama.cpp's kernels. P8b's chunked scan keeps its rollback-replay rationale and loses its
prefill bounty (see TODO.md and decisions.md, "The DeltaNet scan decomposition").

## 2026-07-29 — Two-family review of the MoE-glue + top-p diff: kernels clean twice over, one real hole in the gate script

**Context.** Standard post-arc review of what became commit `bec5fa2`, one Claude
reviewer and one outside-model pass (Codex CLI, gpt-5.6-sol, xhigh effort) over the full
diff. Both independently cleared the fused kernels line by line — barriers, the bitonic
network's pad ordering, the epilogue's fp pragmas, wrapper validation. The findings that
survived were all at the edges.

**Fixed.** (1) `parity-gate.ts`'s `baseEnv()` stripped every presence-based kernel
switch EXCEPT the two new ones, so a stray `XWEN_MOE_GLUE_CLASSIC` in the shell (even
`=0`) would have silently classic-ed both sides and passed the gate without dispatching
either new kernel. Both are stripped now, and the 35B gate was re-run under an
explicitly clean env: ALL PASS, every number identical (strict 1.000000, mm 0.999631,
decode 63/63/62 with 0 mismatches, ppl Δnll 0.000791) — which also retroactively
validates the pre-fix run. (2) `top_p == 0.0` — reachable through the serve layers —
had no test; it now pins the keep-one behavior against the llama.cpp oracle.

**Documented, not changed.** The top-p cut is one ulp off llama.cpp at exact f32
boundaries (they re-softmax survivor logits, we divide full-softmax survivors by their
sum — algebraically equal, reproduced counterexample in decisions.md), and the
`llamacpp_filtered` oracle is structurally blind to that class. **Ledgered.** `top_k=0`
semantics (greedy here, disabled there), the unpinned
`MTLMathFloatingPointFunctions` compile axis under the epilogue's bare `exp` (the
bitwise suites are the tripwire), and `mul_mv_id_dual`'s trusting wrapper. Codex's one
discounted finding — the ragged q4_K tail overread — is the documented, deliberate
ggml-matching inheritance, predating this arc.

## 2026-07-29 — First llama.cpp head-to-head: xwen wins decode on both models, loses 27B prefill 2x to the sequential DeltaNet scan

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

With the DeltaNet layers fused, the 35B-A3B's remaining decode cost was the MoE half,
and it was launch-bound rather than bandwidth-bound: 24 dispatches per layer at seq
== 1, of which 8 are real matmuls. Three fusions ship behind one kill switch, each
bit-identical to the candle chain it replaces. `kernel_moe_router` returns the
selected ids and their renormalized weights in one threadgroup per token, replacing
seven dispatches; `kernel_moe_epilogue` folds the combine, the shared-expert gate and
the final add into one pass, replacing four. Bit-identity held in the two places it
looked unlikely, candle's online-Welford softmax and its non-stable bitonic arg-sort,
where a tie flip swaps an entire expert. Decode goes 92.6 to 102.8 tok/s, +11.0%,
arms never overlapping across five interleaved reps, prefill unmoved. Because
everything is bitwise, no provenance change was needed and both gates report
pre-change numbers.

Full record, moved verbatim: [records/fused-moe-glue.md](records/fused-moe-glue.md) (108 lines).

## 2026-07-28 — Fused DeltaNet kernels: a layer goes from ~65 dispatches per token to 8, and 35B prefill from 305 to 2183 tok/s

Three of every four Qwen 3.6 layers are gated DeltaNet and all ran the P3 reference,
about 65 dispatches per layer per token, with prefill worse in kind: the scan was a
Rust loop issuing eight dispatches per timestep per layer, ~123k for a 512-token
chunk on the 35B. Four kernels replace it with 8 dispatches per layer at any length,
the scan's shape being the trick: value-dim columns are independent, so a thread
holds its state slice in registers and the state is read and written once however
long the chunk. The 35B-A3B goes 305.4 to 2183.2 tok/s prefill and 57.8 to 91.2
decode at 596 tokens; the 27B goes 77.3 to 290.4 and 14.3 to 19.0. A measurement
finding came out worth more than the numbers: a sequential A/B here is not an A/B,
the matrix drifting 20-35% slower over ten minutes of load. This is also the first
vendored family that is not bitwise, so `XWEN_DELTA_CLASSIC` is pinned on BOTH sides
of strict. Both gates ALL PASS, and perplexity is now the number to watch.

Full record, moved verbatim: [records/fused-deltanet-kernels.md](records/fused-deltanet-kernels.md) (197 lines).

## 2026-07-28 — Sampler tail: 0.82 → 0.41 ms/token, by moving the softmax off the CPU

The per-token sampling tail was suspected of costing multiple milliseconds of the
~16.9 ms decode budget, and the bench that would have shown it still carried laguna's
shapes, so it had never measured Qwen geometry. Measured at real width, the whole
draw cost 0.819 ms/token, of which 0.204 was the logit-row copy and 0.615 CPU work,
the exp pass alone 0.347 ms and the `select_nth` over 248320 indices 0.270. So the
full-vocabulary softmax moved onto whatever device holds the logits and the candidate
set now comes from a streaming top-k, the draw reading back probabilities instead of
logits so the bus crossings are unchanged. After: 0.406 ms/token, ~2.4% of the
budget. candle's op order is deliberately preserved, so no distribution changed; what
did change is that seeded token streams differ. The review follow-up fixed four
things, three about rows that are not well-formed: the padded tail was drawable, and
NaN was silently skipped on the greedy path the gate runs.

Full record, moved verbatim: [records/sampler-tail.md](records/sampler-tail.md) (94 lines).

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

The engine had never been compared against another implementation on the same
weights, and the 27B had never been run at all, so everything rested on cited source
lines and hand-computed unit tests. Upstream llama.cpp is pinned at e9fa0781 as a
submodule, so moving the pin is a staged change someone approves; the fixtures were
regenerated with Qwen ids from the oracle's own tokenizer, the SWA fixture replaced
by prose that stresses the DeltaNet recurrence. Two latent parser bugs in the
inherited `parity.ts` were silent corrupters, each capable of a convincing false
divergence and neither of a false pass. Both checkpoints agree with upstream: ALL
PASS, six graded each. Track A's bisection finds no cliff, the rel-L2 profile
flattening rather than compounding. The floors are set under the worst observed value
across both checkpoints and all three fixtures, 0.9998 strict and 0.999 mm, an order
tighter than laguna's. Four things are explicitly NOT proven and are ledgered.

Full record, moved verbatim: [records/parity-harness.md](records/parity-harness.md) (91 lines).

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
