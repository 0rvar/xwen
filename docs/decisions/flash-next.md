# Qwen3.8-Flash-Next (qwen4exp)

One topic of [docs/decisions.md](../decisions.md), the index of decisions by topic; moved
here verbatim on 2026-09-06. Dated paragraphs, newest additions appended within their topic.


The fourth checkpoint and the first that is a genuine second architecture rather than a
registry entry over an existing graph: 48 layers of gated DeltaNet and sparse attention
carried on a 4-stream hyper-connection residual, every layer MoE (512 experts, top-10),
plus a 51.2B n-gram embedding table. `docs/qwen4exp-port.md` is the arc's detailed
working doc — spec, traps, phase plan, running log — and stays that. **This section is
the authority for the decisions themselves**; the port doc's own Decisions list is
retained as the record of what was decided when, and points here.

**Block keys are cached per complete block (2026-08-30).** The QSA indexer's block key
— mean over `ratio` raw keys, k_norm, rope at the block's first position — is a
function of that block's rows alone, and those rows are immutable while they sit below
the cache length (the cache is position-indexed; only a truncation followed by an append
rewrites a row, and that moves the length below the block first). So `IndexerCache`
keeps the keys of the first `blocks_ready` complete blocks in a derived plane and
`select` builds only the new ones, in one batch, instead of recomputing all of them from
every raw key on every step. Invalidation lives in ONE setter: every `len` write clamps
`blocks_ready` to `len / ratio`. The one non-clamp is `import_rows`, which resets to 0 —
the rows below an import are replaced by the image, so a key computed from the old rows
would silently survive into the new conversation; a full rebuild is one batched call.
The plane is device-only derived state: not exported, not part of the snapshot format,
and `indexer_bytes_per_token` reports it amortized (128 B/token at ratio 4) while
`indexer_plane_bytes` reports the exact allocation. The pool and the per-head score sum
use `strided_sum`, which replays candle's strided-reduce order — `(n/2)` threads
accumulating `i ≡ tid (mod width)` in order, then one add — so the fast path is
bit-identical to the `mean(1)` / `sum(0)` it replaced at extent 4; the identity holds up
to extent 5 (one or two threads) and is REFUSED above, where candle folds through a
4-lane `simd_sum` in an order the tree does not reproduce (1 ulp at extent 6, measured).
The K/V row gather for the decode selection is one Metal kernel per plane
(`ops::qsa_gather`), a copy and therefore bitwise; a non-Metal source takes the candle
chain with no switch. `XWEN_QSA_CLASSIC` restores both old paths, and the host top-k
below with them. Bench and the caveat on the prefill difference: log.md 2026-08-30 (QSA
decode, steps A+B).

**Decode selection runs on the device (2026-08-30).** A single-token step above the
budget selects its blocks in `kernel_qsa_select` and never reads the scores back; the
row buffer it writes is what the attention's gather reads. Four decisions inside that:

1. *A canonical integer key, on both arms.* Selection ranks by `score_key` — the bit
   pattern of a non-negative finite float, denormals included; a set sign bit or a NaN
   keys as 0 — with a Rust copy in `top_blocks`' comparator and a Metal copy in the
   kernel. Not a float compare: Metal's `max(-0.0f, 0.0f)` returned `-0.0` in the
   first cut and its bit pattern is the largest key (caught by the tie sweep), and a
   flush-to-zero compare would rank a denormal equal to zero where the host ranks it
   above. The shared key also made the host a true total order — it was `partial_cmp`
   with an `Equal` fallback, which under a NaN depends on the walk order.
2. *MSB-first radix select over a sort.* The step needs a threshold and a set, not an
   order; four 256-bin histogram passes over ≤ 65536 keys in one threadgroup find the
   threshold and the equal-quota, and two exclusive scans compact in index order, which
   is exactly the tie rule (lower index wins) for free. A sort would have to be stable
   and would cost its own kernel; candle's `arg_sort` would make the tie rule a property
   of its stability.
3. *Prefill stays on the host.* A chunk's overlay is a `[n_q, n_kv]` mask assembled on
   the host anyway, so its one readback per chunk buys the assembly; a device mask build
   is a separate, low-value item (TODO.md).
4. *Both arms identical by construction, so the switch is a fallback, not a parity row.*
   `XWEN_QSA_HOST_TOPK` runs the readback path; because both rank by one key the rows
   are the same for every input, and the tests hold the kernel to the host bit for bit
   (tie sweep, quota across stripes, NaN/negative, the three-arm scripted sequence).
   Bench: log.md 2026-08-30 (QSA decode, step C) — 33 → 44-45 tok/s at 3.8k-32k, the
   cliff closed.

**Prefill selection and its mask run on the device too (2026-09-06), and decision 3
above is superseded.** "Prefill stays on the host" rested on the readback buying the
mask assembly for free, and at 128k it does not: timed segment by segment with
`XWEN_QSA_TIMER` (an explicit drain, then the readback, the host top-k, the fill and the
upload, each on its own clock), the round trip idled the GPU for 102-106 s of a 563 s
131072-token prefill — 768 round trips, 105 GB of scores read back and 421 GB of masks
uploaded, the host top-k alone 41-44 s of it. `kernel_qsa_select_mask` is the decode
kernel's radix select run one threadgroup per query over the `[n, n_blocks]` score
plane, writing each query's row of the `[n, n_kv]` additive mask in place (`-inf`
first, a device-memory barrier, then zeros over the tail and the selected blocks); no
compaction scan, since a mask is written by position. The mask comes out of candle's
allocator (power-of-two sizes, any free buffer at least as large is reused) rather than
the exact-size, never-reused buffers `Tensor::from_vec` makes, which is where the host
arm's 42 GB came from. Same bits as the host fill for every row, held by
`device_mask_matches_host_mask_bitwise` (tail-only rows, identity rows, the shipped
512-block keep at 30k, 65k and 131k, a final partial chunk, NaN and negative scores);
`XWEN_QSA_HOST_MASK` restores the host arm and `XWEN_QSA_CLASSIC` implies it. +22-28% at 128k (569 → 445-466 s), +15.7% at 8k, peak 59 → 28 GB, a greedy 8k run
byte-identical on both arms; the tables are in
[the record](../records/qsa-device-mask.md).

**Third arch, composition over forking.** `Arch::Qwen4Exp` gets its own graph module;
shared blocks (DeltaNet, attention internals, MoE glue, rope) are reused by composition
and parameterized only where the math actually differs. The qwen35/qwen35moe forward
paths are not edited. Why: the existing three checkpoints keep their throughput BY
CONSTRUCTION — their code is untouched — and a divergent copy of DeltaNet would rot.
The parameterizations that came out of it are deliberately tiny: a `ZGate {Silu,
Sigmoid}` enum on `LinearAttnBlock` resolved at construction (qwen4exp's gated norm
gates on `sigmoid(z)`, ours on `silu(z)`, a silent-garbage difference), and MoE's renorm
clamp becoming a `sum_floor` field (existing checkpoints keep 6.103515625e-5, qwen4exp
passes 0.0 — the clamp is a 3.6-35B detail, not universal). The trunk seam is the same
principle one level up: `XwenModel` gained an `Option<Qwen4ExpParts>` and a one-line
dispatch in `run_stack` rather than a second model type, because `Generator` holds a
concrete `XwenModel` across 87 call sites over 26 methods, all of which a 4-stream model
needs identically (2026-08-26, seam widened 2026-08-29).

**The PLE table lives on the CPU side.** The `[320001536, 160]` n-gram table is mmap'd
file-backed; hashing, the 16-row gather and row dequant happen host-side per token, and
the result (2560 floats/token) feeds the GPU graph. Why: it is pure `get_rows` — no
matmul ever touches it — row addresses depend only on token ids and so are
prefetchable, and the page cache handles hot/cold better than any policy we would write.
GPU residency stays reserved for the ~80 GB trunk. Independently confirmed twice since:
llama.cpp implements it Gemma-3n-style as one host-side gather table, CPU-resident on
CUDA automatically; and LMSYS measured −0.07% throughput offloading it (2026-08-26).

**Refuted: building a cache or eviction policy for the PLE table.**
`garnermccloud/Qwen3.8-Flash-Next-NVFP4-SSD-Stream` ships the streaming version of that
design — the table pulled out as a flat FP8 sidecar, streamed per step with io_uring —
and it deliberately has NO cache and NO eviction, just fixed 64 MiB pools. That is the
right shape: 16 deterministic rows per token with poor reuse are better served by
issuing the exact page reads early than by an LRU, and on unified memory there is no
host→device staging at all, so "touch the pages early" is the whole mechanism. What
transfers is the sizing (2.5 KiB/token of payload, ≤ ~64 KiB/token in 4 KiB pages after
dedup, hidden by one decoder block of overlap) and one negative: they avoid mmap
readahead amplification, so `madvise(MADV_RANDOM)` on our mapping is the thing to test —
default readahead would turn a 90-160 B row into a large window. Their throughput
numbers are not a citation for anything; the baseline looks slow rather than the SSD
fast (2026-08-29).

**Reference-first for every new component.** Hyper-connections, the QSA indexer and PLE
each got a frozen CPU f32 reference with fixture tests before any device work, mirroring
the `ReferenceExperts` pattern; fixtures come from the transformers modeling code, the
one executable ground truth that existed at the time. It earned its keep immediately —
the fixtures settled the QSA whole-blocks-plus-tail question against llama.cpp, retracted
a wrong "PLE gate clamp" divergence we had recorded, and caught that a tail-0 context can
mask the query's own token (2026-08-26).

**Text-only; MTP deferred; serve after CLI.** Vision is dropped (masked_scatter
injection, empty deepstack — a clean cut, and mrope collapses to NEoX-64 for text
exactly as on 3.6/3.8). The MTP head has no transformers implementation and its forward
semantics are unconfirmed (separate `fc_embedding`/`fc_hidden` projections, NOT 3.8's
concat `eh_proj`), so it waits for vLLM/SGLang or the tech report rather than being
guessed at. Serve integration follows CLI bring-up — and as of the P2 review it is not
merely deferred but actively refused: a qwen4exp target would 500 on the snapshot path,
so `xwen serve` rejects the checkpoint until P4 (2026-08-26, sharpened 2026-08-29).

**Weights: Unsloth first, and `UD-Q4_K_XL` specifically.** Dev and first testing run
against Unsloth's Q4-class UD file; a self-converted file with a mix we control remains
the eventual parity target, because floors are calibrated per quant mix and the UD mixes
are per-layer heterogeneous. `UD-Q4_K_XL` is the chosen first target: it is the only
Q4-class trunk whose quant types are ones xwen already has kernels for (Q4_K / Q8_0 /
F32, with IQ4_NL confined to the PLE table). `UD-IQ4_XS` would be roomier — 64.9 GB
trunk against 82.5 — but needs IQ4_XS matmul kernels we do not have. 82.5 GB of wired
trunk plus a demand-paged table plus KV is tight on 128 GiB, but it is the same file
llama.cpp reported 24-25 tok/s decode on for a 128 GB DGX Spark (2026-08-29).

**The 640-column rule, and why Q5_1 became unavoidable.** `ffn_down_exps` is
`[640, 2560, 512]`, and 640 % 256 = 128, so it fails every K/IQ type's block-size
requirement and llama.cpp's generic `tensor_type_fallback()` demotes that plane to a
32-block type on EVERY publisher's file: Q4_K→Q5_0, Q5_K→Q5_1, Q6_K→Q8_0, IQ*→IQ4_NL.
`ffn_gate_exps`/`ffn_up_exps` (ncols 2560) keep their K-quants; `per_layer_token_embd`
(ncols 160) is 32-block-only forever, which is how ggml-org shipped a Q8_0 trunk with a
Q4_0 table. On our target that means Q5_1 down on 43 layers and Q8_0 on 5, against Q4_K
gate/up on 47 and Q5_K on layer 2. **Q5_1 support is therefore not optional and also not
blocking**: `ExpertStack` carries dtype per tensor with no whitelist, `FusedExperts::new`
never compares dtypes across planes or layers, decode falls through to candle's baked
`kernel_mul_mv_id_q5_1_f32`, and prefill drops the affected layer to per-token
`mul_mv_id`. So it runs correctly today and the kernel work is reclassified as P3 perf,
not P2 scope. The `IQ4_NL` matmul deferral stands unchanged — that was always about
IQ4_NL specifically, not about new matmul dtypes generally (2026-08-29).

**One oracle, no vendored copies.** `reference/llama.cpp` is a single submodule, bumped
e9fa0781 → `6fe749801` once PR #27742 merged, and it gates every checkpoint including
qwen4exp. Why this reversed: while the PR was unmerged, its files were vendored
read-only under `reference/qwen4exp/` as reading material, on the reasoning that an
unreviewed AI-drafted branch is not a frozen correctness oracle. The merge settled that,
and a proposed SECOND clone (so the 3.6/3.8 floors could stay frozen at the old pin) was
refuted in favour of moving the one pin and re-confirming: all three existing checkpoints
re-passed at `6fe749801` the same day, floors unchanged and not re-derived, so there was
nothing for a second clone to protect. `scripts/build-llamacpp.sh` needs no target
argument. Only `PROVENANCE.md` and the semantic `bea3b12d` → `6fe749801` diff survive in
`reference/qwen4exp/`, as history (2026-08-26, reversed 2026-08-29).

**The loader owns GGUF tensor-table parsing.** candle's `GgmlDType` cannot even PARSE a
file containing an IQ tensor — `Content::read` fails on the unknown dtype before any
kernel question arises — so the split-GGUF loader, already xwen-owned code, also owns
the tensor-table and dtype parsing, and the pinned candle stays unpatched. IQ4_NL work
splits three ways: metadata visibility (loader), CPU row dequant (needed only for the
PLE table), and Metal matmul kernels (needed only if a matmul weight is IQ4_NL —
deferred). Worth recording that this was DECIDED in P0 and not implemented until P2:
`gguf::open` on the real file failed with "unknown dtype for tensor 20" right up until
unit U0 landed it (2026-08-26, implemented 2026-08-29).

**QSA rides the existing attention block through an overlay, and decodes by row
gather.** `AttnBlock::forward` gained a trailing `Option<&QsaSelection>` whose `None`
path is byte-identical for existing checkpoints; prefill merges a `Mask` into the
existing `PrefillMask` path, decode passes `Rows` and gathers the selected K/V rows into
a packed contiguous view for a maskless sdpa. The gather exists because candle's sdpa
VECTOR kernel — the `seq == 1` route — is compiled WITHOUT mask support and SILENTLY
IGNORES a mask tensor, so a masked decode through stock sdpa would run dense attention
with no error at all (2026-08-26).

**QSA pooled keys stay f32; no round-back to the cache dtype.** The block key is
mean-pooled in f32 and goes straight into the k-norm and rope. HF rounds it back to the
raw-key cache dtype first, which at a BF16 indexer cache strips the pooled key to 8
mantissa bits before it is ever scored; llama.cpp pools through `ggml_get_rows` into f32
and never rounds back. We follow llama.cpp, because llama.cpp is this port's parity
oracle — the same rule that settles every other divergence in the 3.6/3.8 graphs. **The
consequence is recorded so it is not rediscovered as a bug: exact index-set parity
against an HF tap at real geometry is not attainable and is not a goal.** Measured at
real geometry, the bf16 round-back perturbs scores by ~1.2e-2 against a top-k cut margin
of ~2e-3, so roughly 0.5 of the 512 selected blocks per query differ at every context
length above budget. Grade the Metal path against the f32 oracle, not against HF
(2026-08-29).

**P2 keeps the new recurrent state out of `LayerCache`, and refuses what it cannot
carry.** Indexer raw-key caches, the PLE conv state and the 2-id token history live in
`Qwen4ExpParts` with their own checkpoint/rollback mirroring `LayerCache`'s, rather than
growing a fourth variant across five enums and ~15 match sites. Prefix-cache snapshots,
host snapshots and the disk tier do not carry them: a qwen4exp target refuses
snapshot save and restore with a loud error. Why: it decoupled three parallel units from
`kv_cache.rs` entirely. The cost is honest and now scheduled — it is exactly why serve
is refused until P4 (2026-08-29). **SUPERSEDED IN PART 2026-08-30 (P4; the port doc's
D15 carries the same annotation):** `kv_cache.rs` now DOES carry two of the three — the
PLE conv window and its n-gram history as an image on the layer's own snapshot entry,
the QSA raw keys as planes inside `HostFullKv` — and a qwen4exp target snapshots,
rewinds, pages out and stores like any other. What survives is the residency half of the
decision, which was never only about P2 scope: the state still LIVES in
`Qwen4ExpParts`, not in a `LayerCache`, and `LayerCache::restore`/`check_restorable`
unwrap straight to the inner layer and stay unaware of the PLE — `XwenModel` is what
pairs the two. The refusal was P2 scope and nothing else, which is why it could be
retired without moving where the state lives (see the entry below).

**PLE in P2 is host-hybrid, knowingly.** Hash, table row gather and IQ4_NL row dequant
run on the CPU from `MmapSource::bytes`; `key_proj`/`value_proj` run on device; the
per-stream gate, signed sqrt, dilated conv and silu run on the host in f32 over a
`[n, 10240]` copy of the stream — 40 KB/token and one device→host sync per forward at
layer 1. A known P3 cost taken deliberately for correctness first (2026-08-29).

**PLE decode readbacks share one staging buffer and one wait (2026-09-05).** The
host-hybrid computation remains the same. Its old implementation did THREE
`to_vec1` calls, each allocating a staging buffer, encoding a blit and flushing the
GPU; the earlier "one sync" description above counted the logical boundary rather
than the actual waits. `readback_inputs` now copies key, value and carrier through
one candle blit encoder at seq == 1, then calls `flush_and_wait_current` once. At
the shipped geometry the planes contain 10240 + 2560 + 10240 f32 elements: **90 KiB/token** in
total, not just the carrier's 40 KiB. It copies bytes without changing any math.

The sources remain owned through completion, including dtype/stride materializations;
source offsets are honoured and each unequal-length plane has its own destination
range. The staging allocation uses candle's shared-storage builder and its ordinary
fence-tracking blit API, then the same completed-buffer CPU copy as candle's own
readback. Direct CPU access to private GPU source buffers would not be valid.

The decode path requests 90 KiB of staging (rounded to 128 KiB by candle). The initial all-length experiment requested 180 MiB at 2048 tokens (rounded to 256 MiB),
versus an 80 MiB largest individual transfer before, and had no established
end-to-end prefill gain. **Multi-token prefill stays on the existing independent
transfers.** Its batching remains unqualified; the general helper retains full-chunk
correctness/bench coverage. `XWEN_PLE_READBACK_CLASSIC=1` restores independent
transfers at decode too for A/B measurement. The parity script strips the switch;
no numeric provenance field is needed for a bit-preserving copy. The device-side gate/conv landed the same day as an opt-in (next paragraph).

**PLE gate and conv run on device for multi-token forwards, the default since 2026-09-05
(`XWEN_PLE_TAIL_CLASSIC=1` restores the host tail).** The device path moves the per-stream gate, signed sqrt, gated norm, dilated conv and
silu onto two Metal kernels (`ops/ple.metal`) for Metal forwards with `n > 1`; decode
keeps the host tail because its 0.13 ms/token has no qualified gain behind it. The conv
window stays host-owned: only the last `min(n, 9)` normalized rows come back (all `n`
while a checkpoint is armed), so snapshots, rollback, images and the disk tier are
byte-for-byte what the classic path produces. Measured +12.8% Flash-Next prefill at 3851
tokens and +12.9% at 880, decode flat (log.md 2026-09-05, gate and conv). The kernels
match the host tail on real inputs to 6e-7 abs / 8e-7 rel L2.

It shipped as opt-in first because the Flash-Next forced-replay stand-in, as then
written, marked one hard mismatch at long-mixed step 4 (oracle margin 1.83). A control
showed the instrument, not the kernel: reversing the summation order of one host f32
dot product, with no other change, produced the identical hard mismatch, and the host
tail itself held that decision by only 0.31 logit. **The check now carries an
engine-side near-tie rule** (`scripts/flashnext-replay.ts --control`): a mismatch is
also excused when the control arm, the same binary with the change switched off, holds
the oracle's token over the candidate's pick by less than the band. The oracle-side band
already encoded "the reference was not sure"; this encodes "the engine was not sure
 either", which is the only way a rounding-level perturbation can change the answer.
The cap of 8 excuses per fixture and the hard rule for everything else stay. The direct
real-input comparison (6e-7) remains the primary instrument for a re-implementation of
host math; the replay is for changes whose only reference is llama.cpp. Design choices kept: safe math mode so the gate's `isnan` guard survives; reductions
partitioned across simdgroups while every scalar product keeps the oracle's order;
`partial[32]` sized for the most simdgroups a threadgroup can hold rather than for the
256-thread launch; the `gated` scratch dropped after encoding (private pool, fenced
read-after-write, no CPU writer), where a readback staging buffer must outlive the wait.

**Refuted: the pre-release architecture priors.** The port was planned five days before
the card dropped, from a trimmed model-card and forum copy-pastes. Grading them against
the real config: GDN carried over (true, and byte-identical in geometry to our 27B
block, except the gated norm's z-gate is `sigmoid` not `silu`); the n-gram table is
Engram-shaped (true in structure, wrong in three details — raw token ids with NO
NFKC/lowercasing, ONE layer not "a couple of mid-stack layers", and a per-stream
dot-product gate rather than Engram's scalar one); hyper-connections were flagged as the
biggest structural risk at LOW confidence and are in fact present in every layer, which
is the single largest structural difference from anything we ship; and QSA being
DeepSeek-DSA-shaped was right in outline. The lesson is the one the priors themselves
warned about: they were useful for sizing the work and worthless as ground truth — every
one of them was re-derived from `config.json`, the transformers modular file and the
shipped GGUF headers before a line was written (2026-08-25, graded 2026-08-26).

**Flash-Next is `generate`/`chat`-only until P4, by construction rather than by
convention.** (Written as "CLI-only"; corrected 2026-08-30 — `xwen batch` is gated with
serve, see the amendment below. **RETIRED 2026-08-30 when P4 shipped**: every surface
runs the checkpoint; kept as the record of what the gate was and what lifting it
required.)
`Model::servable()` is false for this checkpoint, so `xwen serve` refuses it at startup
— both the registry entry and a custom qwen4exp GGUF — never lists it, and 400s a
request that names it. `Model::auto_fetch()` is false and `Model::supports_drafting()`
is false, so `--draft` is refused rather than silently ignored. Why refusal rather than
partial support: serve's snapshot, page-out and rewind paths require state the qwen4exp
parts cannot snapshot yet — the indexer raw-key caches, the PLE conv window and the 2-id
token history all live outside `LayerCache` by decision, and the disk tier has no tags
for them. A server that accepted the checkpoint would 500 on the first page-out, which
is a worse failure than a startup refusal (2026-08-29).

**Flash-Next is the default checkpoint; serve falls back rather than refuses.** The
`#[default]` on `hub::Model` moved from `Qwen35BA3B` to `Qwen38FlashNext`: it is the
best model here, it is faster than the dense pair, and every mode that can run it should
run it with no flags. Serve is the one mode that cannot (the entry above), and the rule
there is a FALLBACK, not a refusal: a run that named no checkpoint asked for none in
particular, so `xwen serve` serves `Model::default_servable()` and prints one line naming
both checkpoints and the reason. An explicit `--model-size flash-next` is still refused
exactly as before — naming a checkpoint and getting a different one silently is the
failure mode the whole `checkpoint_selectable` rule exists to prevent. The fallback fires
only when nothing else named a model: a config with its own `model` path is not falling
back to anything, so it gets no line.

`default_servable()` NAMES `Qwen3.6-35B-A3B` rather than deriving "the first servable
entry of `MODELS`". `MODELS` is a display order (it is what `/v1/models` prints), and its
first servable entry is the 27B, which decodes at roughly a quarter of the 35B-A3B's
rate — so the derivation would have silently downgraded every existing server while
looking principled. The named constant keeps serve's behaviour exactly what it was
before the flip.

The `auto_fetch()` gate is UNCHANGED, and that is the one visible cost: a zero-flag
`generate`/`chat`/`batch`/`fetch` on a cold cache now downloads 111 GB. `auto_fetch` was
always about a checkpoint arriving as a side effect of a stranger's request, not about
the operator's own zero-flag run — `ensure_model` fetches all four shards after the same
size notice every other checkpoint gets, and it resumes in place. Refusing to fetch the
default would have made the default unusable, which is not a gate, it is a bug
(2026-08-30).

**AMENDED 2026-08-30 (same day): `xwen batch` is on serve's side of that line, not the
one-shots'.** The entry above (and `Model::servable`'s own doc) counted `batch` among the
modes that can run the default. It cannot: a batch prefills the items' shared prefix once
and snapshots the cache there, and a scored field snapshots and restores around every
option it scores, so it moves cache state on its ORDINARY path exactly as the server
does. `servable()` therefore gates two surfaces, and both resolve an unnamed checkpoint
through `default_servable()` — a hub test pins that the two agree, so P4 cannot free one
and leave the other. An explicitly named `Qwen3.8-Flash-Next` is refused before the load,
by the same rule and for the same reason serve refuses it.

Two consequences worth writing down. `unservable_reason()` is now the MODEL's half of
the reason only (what the state is and what carries it); each surface adds what it does
with a cache image, because "the server snapshots, rewinds and pages conversations out"
is false of batch and a shared reason that says it would be wrong on one of its two
callers. And neither refusal offers the other refused surface as the way out, which the
single message did — it sent an operator from serve to `xwen batch`. `XWEN_BATCH_NO_CACHE`
is deliberately not offered as a way through: it skips the shared prefix and leaves the
per-option snapshots, so it would be an escape hatch that works until the schema has an
enum in it.

**RETIRED 2026-08-30 by the entry below.** The refusal, the fallback, both surfaces'
messages, their shared `unservable_reason()` and `refuse_state_transfer` itself are all
deleted: the cache images carry the state, so `servable()` is true for every registry
checkpoint and `default_servable()` returns `default()`. `xwen serve` and `xwen batch`
run Flash-Next with no flags, `/v1/models` lists it and a request may select it — gated
now only by the download rule, so it is listed exactly when the file is really in the HF
cache and the 400 for an uncached one points at `xwen fetch`. The default flip itself
stands unchanged; what is retired is its serve-and-batch fallback clause. The entries
above stay as the record of what was refused and why, and of the batch correction that
had to land before the refusal could be lifted from one place for both surfaces.

**Two kinds of recurrent state, two routes into a cache image — and the PLE image rides
on its layer's entry rather than becoming a fourth layer kind.** qwen4exp carries two
things no image carried before, and they travel differently because they ARE different.
The QSA lightning-indexer raw keys are position-indexed exactly like a full-attention
layer's K/V: one row per token, one MQA key head, nothing recurrent about them. So a
snapshot stores NO data for them at all — a restore is `IndexerCache::truncate(pos)`,
exact — and only the page-out path moves bytes, as a `qsa` plane set plus a
`qsa_head_dim` inside `HostFullKv`, beside the trunk's K/V planes and travelling with
them through `range`, `concat` and `qsa_prefix` (one head, so a position range is a
slice of the buffer rather than the per-head gather the K/V planes need). The PLE conv
window and its rolling n-gram token history are the opposite: recurrent summaries with
no inverse, unreconstructible from any position, so they must travel as DATA — `PleImage`
/ `PleShape`, with `PleState::image/shape/accepts/restore`. Nothing about that is a
preference; it falls out of whether a position determines the state.

The layer alignment is the part that could have gone the other way. The snapshot's
`layers` vector stays ONE ENTRY PER TRUNK LAYER, and the PLE image rides on its layer's
own entry through a WRAPPER — `LayerSnapshot::Ple { inner, ple }`, host mirror
`HostLayerSnapshot::Ple`, disk tag `LAYER_PLE = 3` — rather than a fourth kind alongside
`Full`/`Swa`/`Linear`. Why a wrapper: the PLE layer is ALSO a DeltaNet layer, so a flat
`Ple` variant standing in for `Linear` would have silently dropped that layer's conv and
delta state, which is exactly the class of failure a snapshot cannot afford (it restores,
it runs, it generates different text). Nesting is one deep by construction and a `Ple`
inside a `Ple` is refused on both the assembly path and the read path. `LayerCache::
restore` and `check_restorable` unwrap to `inner` and never learn what a PLE is; the
state does not live in a `LayerCache` at all, and `XwenModel` is the one place that pairs
the layer entry with `Qwen4ExpParts`.

The container version went 3 → 4, and the interesting thing is that only ONE of the two
halves forced it. `disk_cache.rs`'s documented invariant is that the version discriminates
FRAMING and never content — content is discriminated by the checkpoint binding and by the
per-layer kind tags. The PLE state is a new per-layer tag inside unchanged framing, so an
old reader refuses that layer by its tag and no bump is needed, exactly as the DeltaNet
recurrent state landed. The QSA planes sit INSIDE the existing full-attention record,
after its K/V planes, where nothing tags them: a v3 reader would parse the K/V planes,
stop, and then fail on framed bytes it never consumed — a confusing corruption error over
a file that is not corrupt. The bump turns that into what it actually is, a `Binding`
rejection: the scan deletes the file and the conversation costs a re-prefill (2026-08-30).

**The FILE identifies the checkpoint on the one-shot path too, not just in serve.** With
`--model <gguf>` and no `--model-size`, `generate`/`chat`/`batch` used the CLI default
for the chat dialect, the drafter and the label, which after the default flip meant
someone's 35B conversion was rendered with Flash-Next's template and offered Flash-Next's
(nonexistent) drafter. Serve has read this off the file since it began identifying
checkpoints, and the two surfaces disagreeing about what a file IS is not a defensible
split. The rule now lives once, in `XwenConfig::identify`: the file first, `--model-size`
(or, on batch, the payload's `"model"`) as a cross-check that must agree, and a file that
identifies as nothing falling back to `Arch::model()` with a line. `serve::engine::
identify_checkpoint` keeps only the mapping onto `Target` and the startup log, which is
serve's own. The read is metadata-only and happens before the template knobs resolve, so
a contradicting flag still fails in milliseconds rather than after the load (2026-08-30).

**Space→hyphen folding applies to the exact-name comparison only.** The file calls
itself "Qwen3.8 Flash Next" where the official name is `Qwen3.8-Flash-Next`, so
identification folds the two spellings — but only on the exact-name pass, not on the
containment pass. The consequence is deliberate: a name like "Qwen3.8 Flash Next
(imatrix)" identifies as NOTHING rather than as this checkpoint, which is exactly the
existing rule that stops "Qwen3.6 27B MyFinetune" from claiming to be the official 27B.
A file that identifies as nothing still runs, under its own file name (2026-08-29).

**Every Flash-Next-only cost gets its own profiler stage, and profiled decode numbers are
not timings.** `XWEN_STACK_PROFILE` let the PLE layer, QSA selection and the token
readback fall into `inter_stage_host`, which made the three costs unique to this
checkpoint the three the profiler could not see; each now has its own bracket and the
qwen35 path emits none of them. The rule that came with it: **the profiler's per-stage
syncs inflate decode, so profiled decode figures rank stages and nothing more** — quote
only unprofiled tok/s. PLE decode measured 6.4 ms/token profiled against ~2.1 ms real,
which is a 3x error and was briefly believed. Prefill is not affected the same way (the
stages are large and already synchronised), and the prefill attribution it produced is
what P3 was planned from: ffn 51%, hyper-connection glue 34%, `mixer_delta` 8%, PLE 2%
(2026-08-29).

**Q5_1 in the vendored `mm_id`, copied verbatim from the pinned llama.cpp.** `block_q5_1`
and `dequantize_q5_1` are transcribed from `reference/llama.cpp` rather than re-derived,
the same rule the other vendored dtype arms follow — the tile loader was already 32-block
generic through Q8_0, so the arm is a dequant helper and an instantiation. It is
instantiated for the classic, `_hp` and `_t` families and **deliberately NOT for `_t_hp`**:
nothing routes a Q5_1 plane there, and an unused instantiation is compile time and a
kernel-cache entry for a path no file exercises. Q5_1 joins the oracle test dtypes so the
arm is graded, not assumed. This is D18's item (b); items (a) — a Q5_1 arm in the vendored
`mv_id` decode path — and (c) — per-stack `use_mm` — stay open, and the reason (a) did not
follow (b) automatically is that `mm` is now the prefill path for those 43 layers, while
decode still takes candle's baked `kernel_mul_mv_id_q5_1_f32` and did not move at all in
the A/B (2026-08-29).

**The hyper-connection glue is fused; the two Q8_0 gemms are not.** `low_rank` is 320, not
8, so the down and up projections are real gemms moving ~0.7 GB of weights per token —
library work, and they stay on `QLinear`/`QMatMul`. What gets fused is everything around
them: **K1** `hc_norm` (grouped statistics per `hidden`, full-width weight, and the
injection gemv `2·sigmoid((I·n)/hc_count)` folded into the same pass), **K2**
`hc_silu_quarter`, **K3** `hc_mix` (the mean over streams of `sigmoid(u)·n`), **K4**
`hc_write` (a single out-of-place FMA where the candle chain made three full-carrier
passes). 5+1 dispatches per layer-pair against 20+2, roughly 2128 → 600 hc dispatches per
forward. The precedent is the fused DeltaNet gated norm, and so is the rounding contract:
**K2 and K4 are bit-identical to the chains they replace and K1 and K3 are bounded**, each
partitioning across threads a reduction the reference runs in one order — the split is
recorded per kernel in `src/ops/hc.rs` because it decides which of them a bitwise test can
pin and which need the f32 oracle. One deliberate duplication: the injection head is kept
BOTH as a `QLinear` and as a dense f32 `[hc_count, hc_count·hidden]` copy, because the
fused kernel wants raw f32 rows and the classic path must keep working — 160 KiB, and a
replacement rather than a duplicate would have made `XWEN_HC_CLASSIC` a different
computation instead of the same one. A gate outside the kernels' geometry bounds (at most
`HC_MAX_STREAMS` streams, `hidden` a multiple of the simdgroup width that 256 divides)
keeps the candle chain rather than failing (2026-08-29).

**Below 32 tokens the hc norm splits across streams, and it is the same arithmetic.** The
fused `hc_norm` at `n == 1` ran ONE threadgroup per token, walking a 10240-wide row and the
injection head on that one threadgroup — measured as a 6% decode LOSS against the candle chain it
replaced, which is the fusion paying off at prefill and costing at decode. Below
`HC_SPLIT_MAX_N` (32, `XWEN_HC_SPLIT_MAX_N`) the norm runs one threadgroup per (token,
stream) and the injection dot becomes a second kernel over the same grid. **The two launch
shapes are bit-identical, not merely close**: same thread count and same partition for the
statistics, stream-major walk for the dot, pinned by `split_matches_single_bitwise` and
confirmed end to end as byte-identical generated text over 128 tokens. That is what makes
the threshold a free tuning knob rather than a second numeric path to grade: moving it
can never change a result, so `XWEN_HC_SPLIT_MAX_N` is an A/B knob and not a kill switch.
Why a threshold rather than always splitting — the single kernel reads the carrier once
and folds every reduction in one threadgroup, which is the cheaper shape as soon as the
token grid alone fills the machine; the split pair costs an extra read of the normed
carrier and buys `hc_count` times the parallelism, which only pays while the machine is
otherwise idle (2026-08-29).

**At decode the whole read gate is two launches, and it keeps the wide grid (2026-09-05).**
`kernel_hc_gate_down` folds the grouped norm, the injection head, the Q8_0 down projection
and the silu activation into one dispatch; `kernel_hc_gate_up_mix` folds the Q8_0 up
projection, the sigmoid and the stream mean into another; the write stays. 7 → 3
dispatches per gate, 672 → 288 per token, measured +9% plain decode (47.0 → 51.2 median,
+5-10% round by round; log.md "Fused hyper-connection decode gate"). The design rule it
obeys is the one the split arm established: never one threadgroup on a whole token. Kernel A
runs 41 threadgroups and each RECOMPUTES the per-stream scales from a cache-resident
re-read of the carrier and the norm weight rather than consuming a materialized `normed` —
redundant work bought the launch it saved, which is the trade the decode budget says to
make (a launch is ~4 µs of fixed cost, a cached 80 KiB re-read is not). Kernel B puts a
column's streams in adjacent lanes so the mean is a shuffle, not threadgroup memory. Both
are BOUNDED against the split path (the long dots are reassociated), graded at 1e-5
against `ref_hc` where they measure ~1.5e-7; `XWEN_HC_GATE_CLASSIC` is the kill switch
and the replay check's control arm, `XWEN_HC_GATE_FUSED_MAX_N` (8, inclusive) the
ceiling. **Deliberate numeric change in the 2..8-token window:** `without_mv_ext` froze
that window to candle's `QMatMul` matmul, whose half-precision activation tiles sit
1.7e-5 from the oracle at n = 3; the fused gate sits 1.6e-7 there. The fence's purpose —
that `XWEN_HC_CLASSIC` promises the candle chain — is intact, because the fused gate is
inside `read_fused` and both switches revert it; what changed is that the window's
default numerics are now the more accurate ones, and `XWEN_HC_GATE_FUSED_MAX_N=1`
restores the old ones for anyone who needs them verbatim. The tail mixer takes the same
kernels without the head threadgroup. This is the first lever pulled from the ceiling
diagnosis, and it landed on its prediction (−384 launches × ~4 µs ≈ +7.8%, measured +9%).
**And the 2..8-token window is where the gate wins biggest: +57-76% (2026-09-06).** Forcing
every forward to n tokens with `XWEN_PREFILL_CHUNK` and A/B-ing
`XWEN_HC_GATE_FUSED_MAX_N=1` against the default measures 149.7 vs 93.2 tok/s at n = 8,
108.9 vs 69.5 at 4, 68.1 vs 38.6 at 2 (log.md "The fused hc gate is +57-76%"). The size of
that is about the displaced path, not about the fusion: the hc planes are dense_mm-only, so
the split path in that window sends both bottleneck gemms through candle `QMatMul`'s tile
matmul, a gemm shape with half-precision activation tiles at a gemv-sized problem, and the
fused kernels replace it with two wide-grid gemv-style launches. The deliberate numeric
change recorded above therefore also buys throughput in the window it changed.


**PLE rows are prefetched, advisorily, and never gated on the gate.** The decode cost of
the PLE layer is page faults on the 16 IQ4_NL rows a token hashes to — flat per token,
~4.7% page-cache hits, i.e. essentially no reuse — not driver overhead, which is what the
sync-inflated profile had suggested. Since row addresses depend only on token ids, a
background thread per `PleTable` touches one byte per distinct page for the position about
to be forwarded: hinted at sample time for decode, before layer 0 for a prefill chunk. Three
rules the design turns on. It is **advisory** — a dropped or late hint costs a fault it
would have taken anyway, so the prefetcher never blocks and never owns correctness. It is
**never gated on the PLE gate value**, which is computed mid-forward; acting on it would
serialize the lookup behind the thing the lookup feeds, and unconditional retrieval is
cheap. And the row math is **single-sourced** (`PleTable::row_offset`, `PleLayer::gather_rows`
over the frozen hash), because a prefetcher computing addresses a second way would fault the
wrong pages and show up only as an absent speedup. `MADV_RANDOM` is applied to the TABLE's
byte range only — a 90-160 B row must not pull a readahead window — while the whole-file
`WillNeed` stays for the weights, which are read densely and want exactly the opposite
advice. `XWEN_PLE_NO_PREFETCH` and `XWEN_PLE_NO_RANDOM` exist so both halves stay
measurable. Measured effect: `measured 2026-08-29 with one cold prompt per arm (the same-prompt design is invalid — greedy decode hashes every arm to the same rows, so arm k warms arm k+1): median decode gather 0.002 ms with prefetch vs 0.97-1.02 ms without, PLE total 1.05 vs 2.03 ms per token, decode 45.0 vs 43.2 tok/s, pf_dropped 0; MADV_RANDOM is neutral either way (0.002 vs 0.002 with prefetch, 0.97 vs 1.02 without) and stays on only because it is harmless and switchable` (2026-08-29).

**Refuted: the ~15 GB of private memory is a leak.** Flash-Next dirties ~15 GB where
llama-server dirties 751 MB on the identical file, which read as a bug worth chasing. An
audit accounts for ~11.4 GB of it as deliberate design, all of it shared with the three
shipped checkpoints: attention and GDN projections dequantized to f16 planes for the
prefill gemm (~5.35 GB, ~6.14 after candle's power-of-two buffer rounding), `token_embd`
dequantized whole to f16 (1.27 → 2.15 GB plus a ~2.5 GB f32 transient), non-aliased Q8_0
copies for lm_head, the hc bottleneck projections, the shared expert and the PLE k/v
(~1.65 GB), transposed router weights at 8 MiB granularity across 48 layers (0.40),
indexer raw-key planes sized at `max_ctx` (0.81 at the 131072 default), delta state (0.15).
What should be aliased is aliased: the 77.5 GB expert stacks, the 28.8 GB PLE table and the
BF16 indexer projections never leave the mapping. So the gap against llama.cpp is a
different memory strategy — materialize for the gemm you want — and not a lifetime bug.
Three shrinks stay open and ledgered rather than taken now: alias the Q8_0 planes that only
ever feed `QMatMul`, grow the indexer planes on demand, and gather `token_embd` rows from
the quantized tensor instead of materializing the whole table (2026-08-29).

**Forced replay against llama.cpp is Flash-Next's math gate while the harness panics, and
free-run text is not a grade.** Every change to this checkpoint's math is graded by
teacher-forcing xwen along the oracle's own greedy trajectory and counting agreement —
186/192 with 0 hard mismatches and 0 non-finite after the Q5_1 arm and the fused hc, against
U7's pre-P3 189/192 on the same instrument. The difference is six rank-2 near-ties (margins
0.009-0.30 logit) changing sides, which is what a bounded kernel change looks like and what
the near-tie band exists for. **Free-run greedy output is explicitly NOT the instrument
here**: it forks at token 2 on a 0.0817-logit near-tie and everything after that is
incomparable, so "the text changed" carries no information about this checkpoint and must
not be reported as a regression — the same reasoning that makes the shipped checkpoints'
decode tier teacher-forced. This stands in until the `ReferenceExperts` panic on the
512-expert geometry is fixed and the real four-tier gate can run (2026-08-29).

**Phased, correctness before speed.** P0 scaffold (split-GGUF loader, config parse,
registry) → P1 CPU references and fixtures → P2 graph assembly, real file, greedy smoke,
oracle agreement → P3 Metal and perf → P4 serve, sampling defaults, harness extension.
P2 closed 2026-08-29 with the graph agreeing with the llama.cpp oracle at 189/192
forced-replay steps and zero hard mismatches; P3's first pass closed the prefill gap and
took decode past llama.cpp the same day (1.01x prefill, 1.04x decode at a 530-token
prompt, 186/192 forced replay), with the device-side PLE gate/conv and the QSA top-k
kernel still ledgered (2026-08-26, P3 pass 2026-08-29).

**Ceilings, as of 2026-09-05 (log.md "Ceiling diagnosis"), and how the ledger reads
them.** A decode token reads 6.33 GB of weights (GDN projections 2.25, routed experts
1.50, hc 0.69, lm_head 0.68, full attention 0.64, routers/shexp/indexer/PLE 0.58) plus
~0.3 GB of state and KV, which at the measured 537-565 GB/s is 11.7-12.3 ms of a
21.3 ms token: the bytes-only ceiling is 81-86 tok/s at the measured rate, and nothing
above it is reachable without reading fewer bytes per token. The rest of the token is ~1740 dispatches in a mostly dependent chain
(hc 672, MoE 576, GDN 252, attention ~200, QSA ~24 below budget / ~165 above) at ~4 µs
average fixed cost (a residual attribution between the 2.5 µs measured floor and the
8.4 µs gemv intercept), three host syncs (~0.9 ms) and the serial scan (~1.0 ms beyond
its bytes). So the decode lever
is dispatch COUNT — the budget prices a removed dispatch at ~4 µs and a removed sync
at ~0.3 ms — never per-kernel bandwidth, which the big planes already reach at 95-97%
of a pure read. The first lever pulled on that reading, the fused hc decode gate (−384
launches, "At decode the whole read gate is two launches" above), measured +9% against a
+7.8% prediction — the attribution holds.
[QUALIFIED 2026-09-06 by the fused shared expert, "The shared expert is ONE
dispatch at decode" above: that per-dispatch price is an average over launches of
very different byte weight, and it does NOT predict every fusion. The refinement is
at the end of this paragraph.]
Prefill at the 2048 chunk is 12.07 GFLOP per token; 3851 tokens run at
13.7 TFLOP/s end to end against 28-36 for the dense gemm in isolation, weight re-reads
are 9% of wall (every expert is touched per chunk — a lower bound inside the gemm time,
not additive to it), the dispatch floor is under 1%,
and the expert gemms cost 1.44 s of the 3.41 s by the amortized mm_id bench at their
real geometry (~12 TFLOP/s, dequant-bound by the 2026-08-30 code reading) — but two
in-situ A/Bs on prefill wall (classic tiles, and the pre-2026-08-30 full grid) transfer
those isolated rates at 0.82 and 0.32 respectively, so the share is **bracketed at
14-43%**. That makes the expert gemm the largest PRICED prefill candidate and CONTESTS
the 2026-08-30 "gemms are a minority of `ffn`" reading, which came off the
sync-inflated stage profiler; the ledger carries the bracket, not a point, until an
in-situ duplicate-dispatch probe replaces it. **SETTLED the same day by that probe
(log.md "Duplicate-dispatch probe"; `XWEN_DUP_STAGE`): the expert gemms are 0.96-1.09 s
of the 3.4 s wall at 3851 tokens (28-32%), the MoE glue 0.40 s, the hc gates 0.39 s (of
which the two gemms 0.14), the GDN kernels 0.23 s (scan 0.16), the shared expert ~0. The
"minority" reading is refuted — the gemms are 73% of the `ffn` stage — and the prefill
ledger is priced from these figures rather than from the bracket.** Prefill stages are
priced by the probe from now on: it is the only instrument on this machine that adds no
sync, and its one caveat is that a stage which leaves the GPU idle can overlap its own
copy and read low (the two-copy experts arm scaled linearly, 1.03 s per copy).
**In decode mode that caveat is the rule, not the corner case (2026-09-06,
`XWEN_DUP_DECODE`; log.md "probe learns decode mode").** A duplicated decode launch
has no buffer hazard against its original, writing a fresh output from the same inputs, so
candle's encoder inserts no barrier and the two run concurrently: a stage that leaves the
GPU idle prices at about zero. So a decode delta above zero is a FLOOR for that stage and a
delta of about zero means "it overlaps with itself", never "it is free", and the probe
cannot price a latency-bound decode stage at all. Measured at 596 tokens with `-n 128`:
the shared expert floors at 0.43 ms of a 19.65 ms token (2.2%), which is its five launches
per layer being a dependent chain, against the 0.77-0.96 ms the ~4 µs budget gives 240
launches (a chain, but a BYTE-bound one, which the refinement below is about); the MoE
glue and the newly wrapped `router_proj` stage both read zero, so both overlap fully and
neither is priced. The instrument itself checks out across lengths: the
`moe_glue` prefill delta is 0.058 s at 596 tokens where the 0.40 s at 3851 scales to 0.062.
What the zero on the router projection does say is that it runs at low occupancy, which is
the gemv hypothesis for a single-row mlx gemm over a 5.2 MB f32 plane, and the A/B that
replaces it with a wide-grid gemv is what would price it.

**And the launch budget itself is refined, 2026-09-06 (log.md "Fused MoE shared expert";
"The shared expert is ONE dispatch at decode" above).** The fused shared expert removed
192 decode launches, which the ~4 µs average priced at +3.5-4% on Flash-Next, and
measured +0.6% there and +1.6% on the 35B. The probe explains the gap on the fused
binary: the shared expert's cost is its BYTES, not its launches. Three Q8_0 planes of
1.74 MB are 5.2 MB per layer and 250 MB per token, 0.46 ms at the measured ~540 GB/s;
kernel A alone floors at 0.31 ms for its 3.5 MB per layer, which is ~535 GB/s and
therefore at rate, against 0.55 ms for the five classic launches. The removed launches
were bandwidth-bound and mostly hidden under traffic they had to move anyway, and,
hanging off the same input as the routed expert gathers with no hazard between them,
partly overlapped with those as well; fusing them recovered the gaps only, ~0.15 ms.
The hc gate is the contrast that makes the rule: its seven launches carried ~1 MB each on
a strictly dependent chain, so each launch's latency was exposed and the budget held.
**The rule, for ranking every remaining fusion candidate: (launches removed × ~4 µs) is
the right price ONLY for launches that carry less than ~4 µs of traffic at rate, roughly
2 MB, AND sit on the dependent chain. Byte-bound or overlapped launches yield their gaps
only, which is a small positive rather than the budget figure.** Re-ranked by it, the
decode candidates are the MoE glue kernels (router kernel and epilogue: tiny bytes, on the
chain), the token-id readback sync, the QSA tail and the GDN glue. The router projection
is not on that list: it is an occupancy item rather than a launch-count one (8
threadgroups for 5.2 MB), and its lever is the vendored wide-grid f32 gemv, being
implemented and unmeasured. **[MEASURED AND SHIPPED the same day; the paragraph below.]**

**And a third class of decode cost, named 2026-09-06 (log.md "Router projection on a
256-threadgroup gemv"): OCCUPANCY.** The router projection was invisible to both
instruments above. The probe read zero on it, because at decode its duplicate overlaps
itself, and the byte budget put it at 4% of a token's bytes, so neither ranked it. Yet
replacing candle's 8-threadgroup mlx gemv with a 256-threadgroup vendored one is worth
+10.3% decode on the 35B-A3B and +4.8% on Flash-Next: 0.8 and 0.9 ms per token recovered
against bytes floors of 0.16 and 0.45 ms, so the old kernel had been costing six and three
times its own bytes respectively. Bytes moved and launch gaps do not span the decode cost
space; a kernel that leaves the GPU mostly idle is its own class, and the narrower the
plane the worse the fixed 8-threadgroup shape gets, which is why the smaller router gained
the most. The instrument that would find the rest is an audit rather than a probe: list
every decode dispatch's threadgroup count against its bytes, and flag any plane over ~1 MB
running under ~32 threadgroups (TODO.md, decode ledger).
