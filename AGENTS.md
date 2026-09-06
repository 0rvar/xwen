# AGENTS.md

Read `README.md` first for what this is. This file is the context that is NOT obvious
from the code: ground-truth sources, hard-won gotchas, workflows. When picking up a
TODO.md item, read this whole file first — items acquire traps and the traps get
documented here.

Doc map. Every file here has exactly one job, and work that lands is recorded in the
one that owns it:

- **`README.md`** is getting started and the practical surface.
- **`AGENTS.md`** is the rules, the gotchas and this map.
- **`docs/decisions.md`** is an index over **`docs/decisions/<topic>.md`**, which hold the
  WHY by topic: every deliberate choice, default, policy and refuted direction, with its
  evidence. A new decision is a dated paragraph appended inside the topic it belongs to.
- **`docs/log.md`** is the timeline, newest first: short dated entries. Full arc
  write-ups, meaning protocol, tables and reviews, live in **`docs/records/<slug>.md`**
  behind a stub in the log.
- **`docs/parity.md`** is the verification runbook.
- **`docs/perf-state.md`** is the current figures, and it is their single source.
- **`docs/benching.md`** is how to measure anything on this machine.
- **`TODO.md`** is the backlog and the open ledger: a ranked **Front** of at most ten
  planned items, then every deferred scope grouped by area, each tagged and carrying a
  `From:` line. Closed text lives verbatim in
  [`docs/ledger-archive.md`](docs/ledger-archive.md) under the arc that deferred it,
  retired text under `Retired: <area>`.
- **`scripts/docs-check.ts`** asserts that links, anchors, unique titles and quoted
  references all resolve, and that the ledger keeps its shape (tags, `From:` lines,
  the front cap, item length).

Rules for keeping it that way:

- Finishing an arc means recording it. The shape is a judgement call per session: a new
  record for a substantial arc, an update to an existing record or decision paragraph
  when the work continues one, or a mix of both. Records are not mandatory.
- Log entries stay short, a paragraph with the headline figures and the links. The
  narrative and the tables belong in the record or in the topic file, not in the log.
- A `TODO.md` annotation is at most three lines plus a link. Do not duplicate results or
  narrative there.
- Headings are never renamed once written, because everything links by heading text.
- The ledger has two exits, shipped and retired, and the unit is the item or the
  lettered sub-item. Shipped:
  the text moves verbatim to [`docs/ledger-archive.md`](docs/ledger-archive.md) under
  the heading its `From:` line names, at the end of the arc that closes it. Retired: a
  dated line with the reason and a reopen condition, then the same move, under a
  `Retired: <area>` heading (the area is the TODO.md section the item sat in). Every
  moved block opens with one bracketed line saying what it is and where its open
  remainder lives. Retired means not planned, not forbidden: pick it up when
  its reopen condition holds, or for a reason the retirement did not foresee, and say
  why. Only `docs/decisions/` says refuted, and only with evidence; that is the one
  state not to relitigate without new evidence.
- The Front of `TODO.md` holds at most ten items, ranked, each with its expected gain
  against a ceiling in docs/perf-state.md, a user who is waiting, or, for an
  instrument, what it would price. Promoting one means
  demoting one. An item in an area section is not planned until it is promoted.
- Intake. A deferred scope or a review finding enters the ledger only when it carries a
  number (a measured gain or cost, against a ceiling in docs/perf-state.md) or a user who
  is waiting for it. A chore enters as `[small]` only when the next arc is expected to do
  it. Everything else goes in the arc's record as "not taken now" with the reason and a
  reopen condition, and is not a ledger item. "If this ever bites, the fix is X" is a
  record line, never an item: the record keeps the sketch, and the reopen condition is
  what makes it findable.
- Triage at the end of every arc, and for any item whose latest date is older than 30
  days: promote it, keep it with a fresh dated line saying why it is still worth it, or
  retire it. Archive text is never rewritten; the open ledger may be regrouped.
- Before handing off, make the next live TODO actionable: name the next experiment or
  implementation step, its entry point and prerequisites, and any unresolved risks or
  verification gaps. Distinguish a measured result from an unpriced candidate, and link
  to the evidence rather than making the next agent reconstruct the session.
- Run `bun scripts/docs-check.ts` before handing off.

## Non-negotiables

- Design target: maximum tok/s for Qwen3.6-27B, Qwen3.6-35B-A3B and Qwen3.8-27B GGUF on
  this one machine (M5 Max, Metal). Batch 1. No portability hedging. (3.8-27B runs the
  3.6-27B graph unchanged — it is a registry entry, not a port.)
- TODO.md is the deferred-work ledger. Scope is never silently dropped: it ships, it
  becomes a ledger item with context, or it is retired with a dated reason and a reopen
  condition (retired means not planned, not forbidden). Ledger text is never deleted:
  closed and retired items move verbatim to the archive.
- Every shipped arc updates the docs before it's done: dated log.md entry, README if
  the surface changed, decisions.md if a decision was made/changed/refuted, perf-state.md
  if a figure moved. A TODO.md update alone is not sufficient.
- The reference implementations (`ReferenceExperts`, and the DeltaNet reference once it
  lands) are frozen correctness oracles. Never "optimize" them.
- Any change touching model math re-runs the parity gate (docs/parity.md) before it
  ships. The harness is live: `bun scripts/parity-gate.ts` for the 35B,
  `--model-size 27b` for the dense file.

## Ground truth, in order of authority

1. llama.cpp master `src/models/qwen35.cpp`, `src/models/qwen35moe.cpp`, and
   `src/models/delta-net-base.cpp` (per-arch graphs moved out of llama-model.cpp) —
   the executable reference for both archs, including the delta-rule recurrence in
   recurrent and chunked (chunk=64) forms.
2. The GGUF metadata and tensor table of the blessed ggml-org files themselves.
3. HF transformers `modular_qwen3_5.py` / `modular_qwen3_5_moe.py` (thin shims over
   `modular_qwen3_next.py`, which holds the real math) — for intent, NOT for tensor
   layout: the GGUF has conversion-baked deltas (next section).
4. HF `config.json` of Qwen/Qwen3.6-* — last resort; its text params nest under
   `text_config` and its single `eos_token_id` is wrong for chat.

## Architecture cheat sheet (Qwen 3.6, ggml-org GGUF)

Shared by both models: head_dim 256; QK-RMSNorm over [256] per head (full-attention
layers only), applied before rope; partial NEoX rope over the first 64 dims, theta 1e7,
dims 64..255 unrotated (GGUF says IMROPE sections [11,11,10,0], which for text-only is
provably identical to NEoX over n_rot=64 — implement plain NEoX, still validate the
sections key); no biases anywhere except `ssm_dt.bias`-the-tensor (which is a
projection bias in name only — it's the dt offset vector); RMSNorm eps 1e-6; vocab
248320 (padded; real tokens end at 248076); untied embeddings, real `output.weight`
(Q6_K in Q4_K_M files); full attention at layer indices 3,7,11,… (`(i+1) % 4 == 0`),
gated DeltaNet everywhere else. Layer skeleton: `x + attn(norm(x))`, then
`h + ffn(post_attention_norm(h))` — post_attention_norm is the PRE-MLP norm; there is
no ffn_norm tensor.

Full-attention layer: `attn_q` is double-width (per-head interleaved `[q_h(256),
gate_h(256)] × n_head`); split q/gate by strided view, QK-norm on q AND k, rope q/k,
sdpa scale 1/√256, then `out *= sigmoid(gate)` BEFORE o_proj (`attn_output`).

Gated DeltaNet layer (all math per llama.cpp delta-net-base.cpp; state fp32):
`attn_qkv` → conv1d (depthwise, kernel 4, causal, NO bias, over the full fused width)
→ silu over the WHOLE fused stream (so q and k are silu'd before their L2 norm, not
just v) → split q [128×16H] / k [128×16H] / v [128×H_v] → L2-norm q,k in ggml's
clamp form `x / max(‖x‖, eps)` with eps = rms_norm_eps (ggml-cpu ops.cpp:4198-4204;
NOTE: HF uses `x·rsqrt(Σx²+eps)` — the two differ only for near-zero vectors, but
llama.cpp is the parity ground truth, so the clamp form is canonical here) →
repeat k-heads to H_v (GGUF V-order is TILED, so plain repeat, not interleave) →
delta rule with `beta = sigmoid(ssm_beta @ x)`, `g = ssm_a * softplus(ssm_alpha @ x +
ssm_dt.bias)` where `ssm_a` is pre-baked `-exp(A_log)`; recurrent step:
`S = S*exp(g); d = (v − (S·k)) * beta; S += k⊗d; o = S·q/√128`. Then gated RMSNorm
(norm, × ssm_norm.weight [128], THEN × silu(z)) where `z = attn_gate @ x`, then
`ssm_out`. Conv state: last 3 columns of the fused qkv stream; delta state:
[128,128,H_v] fp32 per layer per seq.

27B (`qwen35`): 64 layers (16 full-attn), hidden 5120, 24 Q / 4 KV heads, dense SwiGLU
FFN 17408; DeltaNet: 16 K-heads, 48 V-heads, head dims 128 (inner 6144). GGUF ssm keys
mislead: `time_step_rank`=48=V-heads, `group_count`=16=K-heads, `state_size`=128=head
dim, `inner_size`=6144.

35B-A3B (`qwen35moe`): 40 layers (10 full-attn), hidden 2048, 16 Q / 2 KV heads;
DeltaNet: 16 K-heads, 32 V-heads (inner 4096). Every layer MoE (no dense FFN, no
`feed_forward_length` key): router `ffn_gate_inp` [2048,256] F32, softmax over all 256
THEN top-8 THEN renormalize (clamp sum ≥ 6.103515625e-5), no expert weight scale;
experts `ffn_{gate,up,down}_exps` Q4_K, 512-wide; shared expert `ffn_*_shexp` Q8_0
512-wide, output scaled by `sigmoid(ffn_gate_inp_shexp @ x)` (a [2048] vector → one
scalar per token), added to routed output.

Tokenizer/chat: ChatML. Specials: `<|im_start|>` 248045, `<|im_end|>` 248046,
`<|endoftext|>` 248044, `<think>` 248068 / `</think>` 248069 (single tokens but
`special: false` — handle by id in the gen loop), `<tool_call>` 248058/248059,
`<tool_response>` 248066/248067. No BOS, ever. Stop on 248046 OR 248044. Sampling
defaults are MODE-KEYED per the official cards, identical across all three checkpoints:
thinking 1.0 / 0.95 / 20, non-thinking 0.7 / 0.80 / 20 (`SamplerOptions::recommended`;
explicit flags/config/request values always win). Generation prompt ends inside an open
`<think>\n` unless thinking is disabled (then a closed empty block is emitted). The chat
template is a per-checkpoint DIALECT (`Model::chat_dialect`): the 3.8 template renders a
reasoning_effort system preamble (xhigh default / low have sentences, medium renders
nothing; only with thinking on; synthesizes a system block when the conversation has
none), defaults preserve_thinking TRUE (3.6: false), emits no block for an empty system
message (3.6 does), and does not split inline `<think>` out of assistant content
(3.6 does — `split_reasoning` is Qwen36-gated).

## Checkpoint location

HF cache (`HF_HUB_CACHE` > `HF_HOME/hub`), cache-first via hf-hub, download on miss.
**The default checkpoint is Qwen3.8-Flash-Next as of 2026-08-30** (`unsloth/
Qwen3.8-Flash-Next-GGUF`, UD-Q4_K_XL, four shards, 111 GB, no drafter) — so a zero-flag
`generate`/`chat`/`fetch` run downloads 111 GB on a cold cache, after the usual size
notice. **Every surface runs it as of 2026-08-30 (P4)**, `xwen serve` and `xwen batch`
included: cache images carry the QSA indexer rows and the PLE state, so
`Model::servable()` is true for every registry checkpoint and `default_servable()` ==
`default()`. Both are kept as the seam the next half-ported arch says no through, and
the fallback branch is dead code on purpose. The two gates that DID stay closed:
`auto_fetch()` false — so on the wire Flash-Next is listed and selectable exactly when
the file is really in the HF cache, and an uncached one is a 400 pointing at
`xwen fetch` — and `supports_drafting()` false (no drafter exists for the graph; D6).
Other repos/files (hub.rs): `ggml-org/Qwen3.6-27B-GGUF`,
`ggml-org/Qwen3.6-35B-A3B-GGUF` and `ggml-org/Qwen3.8-27B-GGUF`, Q4_K_M. Sizes: 19.1 GB
(27B), 20.4 GB (35B), 19.0 GB (3.8-27B). Q8_0: 28.6 / 36.9 / 28.6 GB. Drafter sidecars,
one per checkpoint and TWO kinds: `dflash-*-BF16.gguf` on the 3.6 pair (3.5 GB / 0.8 GB)
and `mtp-Qwen3.8-27B-Q8_0.gguf` on the 3.8 (3.2 GB). Every drafter accessor on `Model`
is `Option`, and Flash-Next is the checkpoint that exercises it: it ships none, so the
default run decodes plain and says so. MTP sidecars also exist for both 3.6 checkpoints
and are unused (they have DFlash
heads, which are the better drafter there). `mmproj-*` files are the vision tower —
never load them.
Inherited hf-hub trap: refs/main is read verbatim; a trailing newline in a manually
edited ref costs a full re-download.

## GGUF facts that differ from the HF checkpoint (conversion-baked)

- Every norm weight arrives multiply-ready on the GGUF path — never add 1 to ANY of
  them. (Upstream detail: HF stores zero-centered Gemma-style `(1+w)` norms and the
  converter bakes the +1 in; `ssm_norm.weight` was never zero-centered upstream, so
  the converter skips it — the end state is identical: multiply directly.)
- `ssm_a` = `-exp(A_log)` pre-baked. Use as-is: `g = ssm_a * softplus(...)`.
- V-head ordering is tiled (converter permutes attn_qkv V-rows, attn_gate, ssm_alpha,
  ssm_beta, ssm_a, ssm_dt.bias, conv1d V-channels, ssm_out columns). Plain
  `repeat`-style K-head broadcast is correct against GGUF weights; HF-style
  `repeat_interleave` is WRONG here.
- DeltaNet projections ship under attention names: `attn_qkv` (fused q+k+v; the conv
  runs over this full width) and `attn_gate` (the z gate). There is no `ssm_in`.
- No `ffn_norm`; `post_attention_norm` is the pre-MLP norm.
- `general.file_type` 15 = Q4_K_M, 7 = Q8_0. The Q4_K_M mix is a custom override:
  attn/ssm/shexp Q8_0, ffn/experts Q4_K (including down_exps — the usual Q6_K bump is
  absent), lm_head Q6_K, norms/routers/conv/ssm_a/dt F32, token_embd Q4_K. **Qwen3.8's
  Q4_K_M differs in exactly one plane**: its 16 `blk.N.attn_output.weight` tensors are
  Q6_K, not Q8_0 (upstream's `output.weight=q6_k` rule substring-catches `attn_output`).
  Nothing asserts on that plane's quant and lm_head already exercises Q6_K.
- `general.name` is what identifies a checkpoint, NOT the architecture: two releases ship
  the dense `qwen35` graph with byte-identical configs. The blessed files carry their
  exact full name ("Qwen3.6-27B", "Qwen3.6-35B-A3B", "Qwen3.8-27B"). The FILE decides
  (`XwenConfig::checkpoint` / `Model::identify`): `general.name` first, then the file
  name, each matched as an exact full name or a whole full name found inside it (never a
  bare "3.6"/"3.8" — that would make someone's 14B finetune the official 27B); a name
  matching two checkpoints identifies as neither. `--model-size` is a CROSS-CHECK, not an
  override: it must agree with a file that identifies itself (disagreement is a startup
  error) and only settles a file that identifies as nothing. A file that still says
  nothing runs as `Arch::model()` with a logged warning, under its own file name.
  That rule is `XwenConfig::identify` (returning `Identity::Official`/`::Assumed`) and it
  applies on EVERY surface as of 2026-08-30, not just serve: `--model <gguf>` on
  `generate`/`chat`/`batch` reads the file too, because the checkpoint decides the chat
  dialect, the drafter and the label. On batch the payload's `"model"` is the
  cross-check, there being no size flag there. `serve::engine::identify_checkpoint` is
  now only the mapping onto `Target` plus the startup log.
  Qwen3.8's tokenizer.json is NOT
  byte-identical to 3.6's — it adds seven audio/TTS specials at 248070-248076 over an
  identical base vocab and merge table — but the embedded 3.6 tokenizer is what ships
  (TODO.md).
- Both models are single-file GGUFs; tokenizer + chat template are embedded in the GGUF
  metadata AND vendored under reference/ (embedded into the binary via include_bytes!).
- The GGUF advertises only `eos_token_id = 248046` and has no second-stop key; the
  full stop list [248046, 248044] exists only in the safetensors repo's
  generation_config.json and is therefore HARDCODED in xwen, never read from the GGUF.
- llama.cpp's `gguf-py/constants.py` does not match the shipped files (lists SSM_IN,
  omits ssm_beta and the shexp set). The shipped tensor tables are the spec; never
  read constants.py as one.

## The candle situation

Identical to laguna, unchanged and not relitigated: candle git rev 21cca0b (ships
kernel_mul_mv_id_*/kernel_mul_mm_id_* and the residency-set APIs), objc2 crates
`=`-pinned to what that rev resolves. See ../laguna/CLAUDE.md §"The candle situation"
for the full history; xwen inherits the conclusion, not the retelling.

## Verification workflow

docs/parity.md owns tiers, floors, taps, and runbook — don't restate them, re-read
them. The harness is live as of 2026-07-28 (P7): `bun scripts/parity-gate.ts`
(add `--model-size 27b` for the dense file) runs the whole Track-B cycle and exits
nonzero on any failure. It needs the oracle built once —
`bash scripts/build-llamacpp.sh` against the pinned clone in `reference/llama.cpp`.
`cargo test --release` (ops tests need a Metal device) still covers the
kernel-vs-reference invariants and is the fast pre-check. **Flash-Next cannot run that
harness**; its check is `bun scripts/flashnext-replay.ts --control <kill switch>=1`
(forced replay against llama-server over the committed fixtures, oracle cached under
/tmp; a mismatch is excused when the oracle OR the control arm held the decision by
less than the band, ≤8 excuses, everything else hard; docs/parity.md "Limitations").

## Operational hazards (each has already bitten laguna once; the machine is the same)

- One large model process at a time or GPU OOM. Two 20 GB processes fit RAM but not
  comfort; the 27B Q8_0 (28.6 GB) plus anything else is asking for it.
- Never pipe model output through a pager (`glance` exists; an EOF-spinning llama-cli
  once fed 88 GB into `less` on the laguna side). Scripted llama-cli needs
  `-st -no-cnv </dev/null`.
- Anonymous RSS lies under mmap — the weights are file-backed. Judge memory by
  footprint (`footprint <pid>`), not RSS.
- Never build with a nix Apple SDK in the env: flake.nix uses mkShellNoCC on purpose;
  a nixpkgs SDKROOT links pre-Metal-4 and every tensor-kernel compile fails at
  runtime. Diagnose with `otool -l target/release/xwen | grep -A4 LC_BUILD_VERSION`.
- Never report first-forward prefill as steady-state; state the power mode next to
  every number.
- Qwen-specific: the tokenizer has no BOS and chat stops on TWO eos ids — a gen loop
  that only checks 248044 runs through turn boundaries and looks like "the model won't
  stop", which is a config bug, not a sampling bug.
- Install with `just install` (`cargo install --path . --locked`). Plain
  `cargo install` IGNORES Cargo.lock: it silently re-resolves all deps, and a
  drifted metal/objc2 crate set has produced a binary whose Metal-4 kernels fail
  to compile at runtime (dense_mm.metal, mpp::tensor_ops identifiers
  undeclared). Same failure smell as the nix-SDK trap above, different cause.

## Perf state

[docs/perf-state.md](docs/perf-state.md) is the single source for current figures; these
are the headlines. Plain decode as of 2026-09-06, both at 24c4069: **35B-A3B 127.0 tok/s**
and **Flash-Next 52.9 at 596 tokens** (it ships no drafter, so it decodes plain). The
dense pair as last fitted: **27B 24.8-25.3 plain, 37.5-38.2 drafted on code**
(2026-08-08), **3.8-27B 23.7-24.8 plain, 34.4-35.7 drafted on code** (2026-08-15).
Prefill: **35B 3081-3090 at 3803 tokens** and **Flash-Next 1140 at 3851** (2026-08-30 and
2026-09-05), the dense **27B 702 at 925** (2026-07-29). Read a drafted or an A/B figure
only against the arm measured in its own session.

Three rules do not bend. [docs/benching.md](docs/benching.md) has the rest, and the
ceilings that rank the remaining levers are in perf-state.md.

- State the `pmset -g` line verbatim as of the session, and never claim high-power mode.
  Neither `lowpowermode` nor `powermode` can confirm it.
- One large model process at a time, and never a test suite alongside a bench.
- Bench a PINNED binary: a detached-worktree build under /tmp, `--bin` on the harness.
  A coding agent's `cargo build` in the main tree swaps the binary and its kernels under
  a running harness.

## Drafting (SHIPPED and ON BY DEFAULT; all three checkpoints as of 2026-08-15)

TWO drafter kinds, one verify machinery. `--no-draft` opts out; a zero-flag run fetches
and loads the checkpoint's own sidecar. Which kind a checkpoint ships is
`Model::drafter_kind()`, the file itself is the authority once opened
(`drafter::classify`), and `src/drafter.rs` is the seam. Everything downstream of the
proposal — checkpoint, batched `forward_all_logits`, `accept_drafts`, `kv_rollback`, the
retention cap, the auto-pause controller — is kind-agnostic, which is the whole reason a
second kind was affordable.

**DFlash block drafting (the 3.6 pair).** Adapted (P9), made a both-checkpoint win by the
K-snapshot fused verify (P9a) and flipped to opt-out the same day (3.5 GB 27B / 0.8 GB
35B). Sidecar facts (arch `dflash`; 27B: 5 layers, sliding_window 2048, taps
[2,17,32,47,62], mask 248070; 35B: 6 layers, sliding_window 4096, taps
[2,7,12,17,23,28,33,38], mask 248077; both block_size 16, fc.weight over concatenated
tapped layer outputs, own ffn_norm, q/k-norms [128]). It denoises a whole block in ONE
forward, so depth is nearly free and 15 is a cap, not a fitted value. Verify-walk
rollback uses the fused scan's K-snapshot planes (most-recent-first, llama.cpp's shape;
decisions.md "Model math").

**MTP chain drafting (Qwen3.8-27B).** `mtp-Qwen3.8-27B-Q8_0.gguf`, 3.2 GB, 18 tensors,
`src/mtp.rs`. The head is a 65th trunk-flavour full-attention layer with its own KV:
`eh_proj` over `[enorm(embed) ⊕ hnorm(hidden)]`, then the trunk's own `AttnBlock`/`Rope`
(so partial NEoX rope, QK-norm and the sigmoid output gate are the blessed ones, not a
re-derivation), then `shared_head_norm`. It REUSES the target's quantized `token_embd`
and `output` — the sidecar's BF16 duplicates of both are deliberately ignored, which is
3 GB of its 3.16 saved for nothing lost. It chains one forward per step and self-feeds,
so depth costs linearly and pays off geometrically less: llama.cpp's fitted `n_max` is 3
and so is ours. Ground truth is llama.cpp `graph_mtp` in `src/models/qwen35.cpp` plus the
chain semantics in `common/speculative.cpp`.

TWO silent-garbage traps in that head, both pinned by tests because neither fails loudly:
(1) **concat order is EMBEDDING FIRST** — `eh_proj` takes the embedding in the low half;
swapping the halves yields a graph that runs and drafts noise. (2) **both residuals
anchor on `eh_proj`'s OUTPUT** — `inpSA = eh_proj(cat)`, and attention and FFN both add
back to that; there is no outer residual re-adding the embedding or the incoming hidden.
A third, invisible in the tensor names: the `h` input is the target's hidden AFTER
`output_norm` (upstream commit 166fe294 chose that deliberately). The DFlash spec taps
are PRE-norm layer outputs and are the WRONG source; `XwenModel` has a separate accessor.
The sync rule is the other thing to get exactly right and it lives in one function: the
head's KV row for position `p` is built from `(token_p, hidden_{p-1})` — shifted right by
one, position 0 taking a zero hidden, mirroring llama.cpp's initial `pending_h` — and
`sync` takes tokens and hiddens at the SAME positions and owns the shift itself
(`the_sync_pairs_each_token_with_the_previous_positions_hidden`).

MTP limitations, both ledgered (TODO.md) rather than hidden: **a rewind resets the head**
— it keeps exactly one carry hidden, so `truncate` below what it holds drops it to zero
and that serve conversation stops speculating until a prefill from zero (the DFlash
drafter survives the same rewind, because each of its rows depends only on that
position's taps); and **a stored MTP image resumes only at the exact position it ends
at**, partial cover being refused by the kind-aware `drafter_planes_usable` predicate.
Both have the same root and the same fix.

Controller constants: `p_min` PER-CHECKPOINT via `Model::draft_p_min_default()` and depth
PER-KIND via `Model::draft_max_default()`, both in src/hub.rs; `pause_margin` stays a
shared 1.0. Values and the acceptance they buy are in
[docs/perf-state.md](docs/perf-state.md). `p_min` here is a FULL-VOCAB probability and deliberately NOT llama.cpp's top-10-renormalized one
(decisions.md), so any cross-check against llama.cpp must run both sides at `p_min` 0 or
it is comparing two different gates. The standing retune tool is `bun
scripts/retune-draft.ts` (two-stage, no cell reuse between stages, P9's qualification
criterion, print-only; `--depth-grid` crosses depth with p_min in stage 1) — if you
change `hub.rs`'s arms you must also update the script's `SHIPPED_P_MIN` and
`SHIPPED_DRAFT_MAX` tables, or the next sweep grades against a status quo that no longer
ships. `bun scripts/spec-equivalence.ts` covers all three checkpoints; its GREEDY mode is
the gate, and its sampled mode diverges on the shipped 3.6 checkpoints too (near ties,
not a regression — see [docs/perf-state.md](docs/perf-state.md)).

## serve (INHERITED, partially adapted)

The serve/ tree runs as forked. Its zero-flag default is `Model::default_servable()`,
which since P4 (2026-08-30) is just `Model::default()` — Flash-Next serves like any
other checkpoint, snapshots/rewind/page-out/disk tier all carrying its QSA indexer rows
and PLE state (decisions.md "Qwen3.8-Flash-Next"), and no surface falls back to
anything. Benchmarked on Flash-Next 2026-08-30 (log.md): decode at parity with
`generate` (42-47 tok/s through 32k), a 32k conversation resumes its next turn in
~0.5 s, and an edited prompt re-prefills from zero because prefix reuse quantizes to
snapshots (decisions.md "Serving"). The tool-call parser IS adapted
(2026-07-28, `src/serve/engine.rs`): Qwen's `<function=...>` XML-ish call format, string
arguments passed through raw and non-string arguments parsed as JSON. Thinking
semantics ARE adapted as of 2026-08-19: open-`<think>` seeding, per-dialect
preserve_thinking (a request field on the native and OpenAI dialects, the checkpoint
template's default otherwise — the normalizers pass ALL replayed reasoning through in
native tools mode; the renderer's dialect rule alone decides what renders), the 3.8
reasoning_effort preamble (the OpenAI `reasoning_effort` field drives the think budget
AND the template level, off-scale levels nearest-mapped — a deliberate divergence from
llama.cpp, which passes them raw and lets the template raise; `chat_template_kwargs`
{enable_thinking, preserve_thinking, reasoning_effort} is STRICTLY validated with
400s, the one exception to accept-and-drop, and a request-level template effort — the
kwarg or the native field — on a 3.6 target is itself a 400; `[thinking] effort` /
`serve --reasoning-effort` set a server-wide default, inert-but-legal on 3.6), and
mode-keyed sampling resolved per request after thinking is known (the fixed
DEFAULT_TEMPERATURE/TOP_K/TOP_P constants are gone; ServeSettings sampling keys are
Options, and a pinned value pins both modes). Still open on thinking: the Anthropic
dialect has no per-request effort knob (server-wide default applies) and penalties stay
accept-and-drop — both ledgered (TODO.md, From: the chat-dialect and sampling-defaults
arc).

API model names are FULL names only (`Qwen3.6-27B`, `Qwen3.6-35B-A3B`, `Qwen3.8-27B`,
`Qwen3.8-Flash-Next` — `Model::full_name`, matching `general.name` and the repo), plus
the served file's own id when that file is none of them. Flash-Next is listed and
selectable only while its shards are in the HF cache (`auto_fetch` false — an uncached
one is a 400 naming `xwen fetch`, not an in-request 111 GB download). The CLI's
`27b`/`35b`/`3.8-27b`/`flash-next` aliases are refused on the wire; an unknown `model`
is a 400 on every surface (both dialects, count_tokens and
the batch route), never a silent fall back to the default. `/v1/models` lists each id
exactly once and every listed id is selectable (2026-08-14).

A job names a `serve::types::Target` (checkpoint + "is this the served file"), not a bare
`Model`: on a custom-GGUF server the official checkpoint of the same architecture is a
DIFFERENT file, so an official name resolves the hub file while the file's own id
resolves the local one. Speculation is per checkpoint (`DraftMode::{Off,Official,
Custom}`), resolved at load, so a sidecar-less default checkpoint no longer disables
drafting for the others.
