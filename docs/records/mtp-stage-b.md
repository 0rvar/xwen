# 2026-08-15 (superseded in part by Stage C above — its "+60%" is a single run at the old defaults, and its sampled-equivalence claim does not reproduce) — MTP drafting SHIPS on Qwen3.8-27B: 39.3 tok/s drafted against 24.5 plain in the same session (+60%) at 93.5% acceptance, and `--draft` is byte-identical to `--no-draft` under both equivalence modes

Moved verbatim from [docs/log.md](../log.md) on 2026-09-06; the log keeps this entry's
opening paragraph and links here.


The checkpoint that had no drafter now has one. `src/mtp.rs` is the head (one extra
trunk-flavour full-attention layer with its own KV), `src/drafter.rs` the seam between the
two drafter kinds, and `generate.rs` the chain round. Stage A built the head and the seam
and left two `bail!` stubs where the loops go; this entry covers both stages, since A
shipped no user-visible behaviour on its own.

**Machine state: `lowpowermode 0`, on AC power, charging.** Per CLAUDE.md this machine
emits no `powermode` key, so high-power mode is not claimed — only that low-power mode is
off, which is a different state from the Phase 0 entry below (`lowpowermode 1`, on
battery) and is why none of its absolute numbers are compared against these.

**What a round does.** Draft: chain the head `min(--draft-max, 3)` steps from the token the
target just sampled. Step 1 consumes that token and the target's post-final-norm hidden at
the position before it; every later step is fully self-feeding, consuming its own previous
token and its own post-`shared_head_norm` hidden. Greedy, and a step whose argmax
probability falls below `p_min` DISCARDS its token and ends the chain. Verify, accept and
roll back are the existing DFlash machinery unchanged — checkpoint, batched
`forward_all_logits`, `accept_drafts`, `kv_rollback`, the retention cap, the auto-pause
controller — which is the whole reason a second drafter kind was affordable at all. Then
sync the head over the kept rows.

**The sync rule is the part that had to be got exactly right, and it now lives in one
function.** The head's KV row for position `p` is built from `(token_p, hidden_{p-1})` —
shifted right by one — with position 0 taking a zero hidden, mirroring llama.cpp's initial
`pending_h`. Stage A's `sync` took an already-shifted `h` and left the shift at the call
site, of which there are three (prefill, the round's plain step, the round's verify). That
was reconciled: `sync` now takes the tokens and the hiddens at the SAME positions, exactly
as a forward hands them over, and owns the shift itself, carrying the one hidden the next
row will need. `the_sync_pairs_each_token_with_the_previous_positions_hidden` pins it
against a synthetic head whose attention and FFN contribute nothing, so each row's hidden
is directly readable off the output and an off-by-one cannot hide.

**The chain stays on the GPU.** Phase 0 measured a per-step CPU readback at +1.45-2.96 ms
of pure sync, which on a 3-step chain is most of what drafting saves. So each step's argmax
and probability are reduced on device, the next step's embedding is gathered BY DEVICE
INDEX (`XwenModel::embed_rows`), the hidden is carried forward as a tensor, and one
readback happens at the end of the chain. The per-step vocabulary projection — Phase 0's
"51.8-53.6% of the step's time" — goes through the vendored seq=1 mat-vec
(`XwenModel::lm_head_row`, extracted from `forward`'s decode bypass), not QMatMul. The
`p_min` walk then runs on the host over the read-back probabilities, which means a chain
that will be cut at step 1 has already paid for steps 2 and 3; that is the trade the
on-device accumulation buys, and at depth 3 it is the cheaper side.

**Prefill pays for the head, and the cost is small.** llama.cpp pays an 84 MB-per-4k-ubatch
device-to-host round trip here because its two contexts cannot see each other's tensors;
xwen is one process on one device, so the hiddens go from the trunk's final norm into the
head without leaving the GPU. Measured as an interleaved A/B (XWEN_BENCH=1, 5 reps per arm,
alternating, medians), MTP-attached against plain: **-4.2% at 839 prompt tokens** (median 694.8 vs 725.5 tok/s,
arms spanning 691-702 and 721-734) and **-5.8% at 3286** (424.8 vs 450.9, reps 3-5 alone
430.7 vs 456.8 for 0.943 — the same ratio). Mildly worse at the longer prompt, which is
the shape the mechanism predicts: the head builds its OWN `[1, n_head, seq, pos+seq]`
prefill mask per chunk, where the trunk builds one and hoists it across all sixteen of its
full-attention layers, so the head adds a second full-size mask rather than a sixteenth of
one and the cost grows with `pos`. That mask is reusable as-is — at sync time the head's
committed length and head count both equal the trunk's — and is ledgered.

Both figures are interleaved medians and neither is a clean-machine absolute: the 4k arm's
first two reps read 237-339 tok/s against the last three's 425-460 while the machine
settled out of a concurrent test run, which is the duty-cycle effect CLAUDE.md warns about
(a doubled `XWEN_BENCH` prefill at 3286 tokens is a long run). The RATIO held across that
swing, which is the only reason the number is quotable at all. An earlier reading of -31%
at 4k was contention plus a real bug, both since resolved.

That bug is worth recording because it only appears past 1024 tokens: the head's
`LayerCache` is allocated at `max_ctx.min(1024)` and grown on demand like the trunk's, and
nothing was calling `ensure_full_capacity`. Anything over 1024 tokens of context failed
with a `slice_set` shape mismatch. The 4k prefill measurement is what caught it — the
decode smoke and every unit test run below the boundary.

**Numbers, all within-session.** First live smoke, 200 greedy-ish tokens on a code prompt
at the shipped defaults (`p_min` 0.5, chain 3): 39.3 tok/s drafted against 24.5 tok/s
plain in the same session, **+60%**, at 93.5% acceptance over 55 rounds and 3.8 tokens per
round. With the confidence floor removed entirely (`--draft-p-min 0`, chains always 3)
acceptance falls to 82.0% and throughput to 36.4 tok/s, which is the expected direction and
is what makes 0.5 the right shipped default to start Stage C from. These are single runs,
not a qualification sweep — that is Stage C.

**Equivalence holds in both modes.** `--draft` and `--no-draft` produce byte-identical
output at temperature 0 over 256 tokens, and at temperature 0.8 seed 42 over 192 tokens
with `--draft-p-min 0 --draft-pause-margin 0` — the mode that catches the spec loop
advancing the sampler stream a different number of times than the plain loop. Both were run
by hand; `scripts/spec-equivalence.ts` still has no 3.8 arm (TODO.md).

**Two shapes had to grow a kind.** The disk-tier drafter record now stores which drafter
wrote it, and `CONTAINER_VERSION` goes 2 → 3 so a v2 record is refused rather than
misread — the checkpoint binding cannot help here, since the same target file can be served
with either drafter attached. The two records genuinely differ: DFlash is f32 across
several layers, MTP is f16 across one (it reuses the trunk's own `LayerCache`) and carries
the shift-right carry besides its KV. Same for `hub::Drafter`, whose per-token cache
arithmetic hardcoded DFlash's 8 heads at dim 128: the MTP head is 4 KiB/token against
DFlash's 40-48, an order of magnitude cheaper to give context to.

**One real limitation, ledgered rather than hidden.** The head cannot follow a rewind. Row
`pos` needs the target hidden at `pos - 1` and the head keeps exactly one such hidden, so
`truncate` to anything shorter than it holds resets it to zero rather than resume on
another position's hidden — speculation goes off for the rest of that serve conversation
until a prefill from zero. The DFlash drafter keeps its rows across the same rewind because
each of its rows is a function of that position's taps alone. Three ways out are costed in
TODO.md; the cheapest correct thing was chosen for stage B because syncing on a stale carry
would be a silently wrong draft context rather than a slow one.

886 tests green (817 lib passed plus 26 ignored, and 69 in the binary; the ignored set
is the perf benches and the fixtures that need a checkpoint on disk) (`the_confidence_walk_discards_from_the_first_shortfall`,
`the_sync_pairs_each_token_with_the_previous_positions_hidden`,
`an_mtp_image_round_trips_and_cannot_be_read_as_a_dflash_one`, and the hub's kind
assertions are new). Stage C — the llama.cpp acceptance cross-check and the qualification
sweep — is a separate brief.
