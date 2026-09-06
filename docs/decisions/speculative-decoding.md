# Speculative decoding

One topic of [docs/decisions.md](../decisions.md), the index of decisions by topic; moved
here verbatim on 2026-09-06. Dated paragraphs, newest additions appended within their topic.


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

**Drafting is off by default on Qwen3.6-35B-A3B as of 2026-09-06, and the fitted floor
and depth are left exactly where they are.** Two independent measurements that day read
the drafted arm below plain on that checkpoint, at every length either of them covered.
The presence-penalty A/B (code prompt, 256 tokens, 3 interleaved reps, pinned build) read
drafted 121.1 tok/s against plain 126.5 at penalty 0, and 119.6 against 126.9 at the
shipped 1.5, at 63.0% and 59.4% acceptance. The long-context sweep read the same
direction on long-document prose with a forced thinking decode, medians of 2, drafted
against plain: 111.9 vs 121.9 at 1046 tokens (80.6% acceptance), 85.3 vs 116.3 at 4117
(70.9%), 73.1 vs 104.2 at 8201 (58.5%), 62.8 vs 99.1 at 16409 (57.4%). So it is -8%
already at 1k tokens and inside the acceptance band the fits were made at, and -37% by
16k. Nothing about the drafter changed: PLAIN decode gained +10.3% from the router gemv
the same day (115.1 to 127.0), on top of the beta|alpha fold and the fused shared expert
before it, and the +26 to +28% in docs/perf-state.md was fitted on 2026-08-08 against a
level three improvements older. A default measured against a baseline that has moved is
no longer a default anyone measured.

Off by DEFAULT and not removed: `Model::draft_default_on()` is what silence resolves to,
and `--draft official` (or a serve config that says `enabled = true` or
`path = "official"`) still attaches the sidecar on this checkpoint exactly as before. The
serve merge distinguishes `DraftMode::Default` from `DraftMode::Official` for that
reason alone — collapsing them would make a server that named the official drafter
quietly not use it on one checkpoint, which is the one thing an explicit request exists
to rule out.

`draft_p_min_default` (0.3) and `draft_max_default` (15) are deliberately untouched. They
are the best values anyone has measured for this drafter, and the reason the arm loses is
not that they are wrong but that the target forward they were fitted against got faster —
refitting them by hand off two workloads that were measuring something else would be
guessing. The reopen condition is a retune (`bun scripts/retune-draft.ts` on the 35B, at
more than one context length, since a short-prompt-only sweep sees less than half of the
loss) whose drafted arm reads above plain from 1k through 16k. If it does, this arm goes
back to true with the refitted values; if it does not, the item is retired and the sidecar
stays a flag (2026-09-06).
