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
- **`docs/qwen3-dense.md`** and **`docs/zimage.md`** are the dense Qwen3-4B architecture
  and the whole Z-Image-Turbo pipeline (the encoder role it came in for, then the
  diffusion transformer, the VAE, the scheduler and their traps), the way
  `docs/qwen4exp-port.md` is Flash-Next's port: per-architecture reference, not a rule
  and not a timeline.
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
  3.6-27B graph unchanged — it is a registry entry, not a port.) **Amended 2026-09-06:**
  dense Qwen3-4B (`model_type: qwen3`, HF BF16 safetensors) is in scope as a full
  checkpoint AND as the text-conditioning encoder for the diffusion image transformers
  to come, and it is NOT a tok/s target. It is held to correctness bars and to costing
  the checkpoints above nothing; no arc of it argues for a hot-path change on its behalf
  (decisions.md "Dense Qwen3-4B is a full checkpoint AND the conditioning encoder").
  **Amended again 2026-09-07:** the diffusion image transformers are in scope on the same
  terms, Z-Image-Turbo first. They are held to correctness bars against the reference
  pipeline, they get a secondary time-per-image figure in docs/perf-state.md so it can be
  known, and no image arc argues for a hot-path change in the language models on its
  behalf. A denoising step is prefill-shaped compute, so the decode levers here do not
  transfer either way (decisions.md "Diffusion image transformers are in scope, held to
  correctness bars first").
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

That order is for the GGUF archs. **For dense `qwen3` the order is different and HF is
not demoted**: llama.cpp `src/models/qwen3.cpp` and HF `modeling_qwen3.py` are joint
authority and they agree on every form that could have gone the other way, because there
is no converter between us and these weights: we read the HF safetensors directly, so
none of the conversion deltas below apply. The `config.json` of the checkpoint itself is
authoritative for its own parameters here (it does not nest, and both its stop ids are in
`generation_config.json`), and diffusers' `pipeline_z_image.py` is authority for the
encoder role. Details: [docs/qwen3-dense.md](docs/qwen3-dense.md),
[docs/zimage.md](docs/zimage.md).

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
defaults are MODE-KEYED per the official cards: thinking 1.0 / 0.95 / 20, non-thinking
0.7 / 0.80 / 20 on every checkpoint, plus `presence_penalty` 1.5 non-thinking on every
checkpoint and, thinking, 1.5 on the 35B-A3B alone (`SamplerOptions::recommended_for`,
`Model::recommended_presence_penalty`; explicit flags/config/request values always win).
The penalty is vLLM's presence penalty over the current reply's emitted ids, applied on
the device before the softmax so the fast path stays, and through the verify rows with a
round-end rollback truncation, so `--draft` and `--no-draft` stay equivalent (greedy
gate, 2026-09-06). `top_k` 0 means no top-k cut; 1 is greedy. Generation prompt ends inside an open
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

## Qwen3-4B dense (`qwen3`, HF BF16 safetensors)

A second weight format and a second vocabulary, not a variant of the graphs above. It is
in the repo for two roles at once: a full LM checkpoint, and the text-conditioning
encoder that the diffusion image transformers will call in-process (Z-Image-Turbo
first). [docs/qwen3-dense.md](docs/qwen3-dense.md) is the architecture and the
verification bars, [docs/zimage.md](docs/zimage.md) is the encoder role,
[docs/records/qwen3-dense.md](docs/records/qwen3-dense.md) is the arc.

Shape: 36 layers, ALL full attention, hidden 2560, 32 Q / 8 KV heads, head_dim 128,
dense SwiGLU 9728, vocab 151936, rms_norm_eps 1e-6, tied embeddings, no biases, no
sliding window, no output gate. Rope is FULL NEoX over all 128 head dims (theta 1e6, or
5e6 on Instruct-2507), not the partial 64-of-256 of 3.6. QK-RMSNorm over [128] on every
layer, before rope, v untouched. GQA broadcast is repeat_interleave (KV head j serves Q
heads 4j..4j+3), which is the form the 3.6 GGUF path calls wrong for ITSELF; there is no
converter here, so no tiled V-order and no pre-baked norm. Specials 151643
`<|endoftext|>`, 151644 `<|im_start|>`, 151645 `<|im_end|>`, 151667 `<think>` / 151668
`</think>` (`special: false`, same by-id trap). No BOS. Stops on 151645 AND 151643, and
unlike 3.6 both are in the upstream `generation_config.json`.

State as of 2026-09-07, after Arc 3: **every surface runs the two language models** -
`generate`, `chat`, `serve`, `batch` and `encode-text` - and the ENCODER runs
`encode-text` alone. `auto_fetch()` is still false on all three, so an uncached checkpoint
is a 400 or a CLI error naming `xwen fetch` and never an 8 GB download inside a request.
`Model::not_servable_reason()` is the single source of `servable()` and of the one
sentence the CLI and the HTTP 400 both print, and that gate runs on `generate` and `chat`
too: loading the encoder there SUCCEEDS, its weights parsing and its config being a
language model's, so without the gate those surfaces generate fluent garbage out of the
zero-filled layer 35 instead of failing. There is no drafter for this architecture and
none is planned.

Traps, each of which has already cost someone time:

- **Z-Image ships a corrupt copy of Qwen3-4B.** `text_encoder/` shard 3 has
  `model.layers.35.mlp.up_proj.weight` zero-filled for 14,772,816 contiguous elements
  from element 27,003 and `down_proj` for 3,938,425 from element 20,930,265; base
  Qwen3-4B has none. Harmless for the encoder (index 35 never evaluates layer 35) and
  disqualifying for anything else, which is why that entry is encode-only. Its alias is
  `zimage-turbo-encoder` as of 2026-09-07; `zimage-turbo` names the full diffusion
  pipeline now (see below). The loader
  refuses any zero run past 4096 elements unless the REGISTRY ENTRY allowlists that
  tensor by name, so a bare directory is refused and only the documented entry passes.
  Never widen the allowlist to make a load succeed, and never point an LM surface at
  that copy: `Qwen/Qwen3-4B` is the faithful one.
- **NFC.** Every Qwen tokenizer.json declares an NFC normalizer that the HF runtime
  applies and llama.cpp does not. `encode` is not injective over decomposed spellings
  and `decode(encode(x))` returns the NFC form. Pre-existing, affects 3.6 identically. A
  fixture whose ids come from `llama-tokenize` is a valid oracle only for NFC input.
- **Specials are per tokenizer instance now**, resolved by token text at load
  (`LagunaTokenizer::specials()`), not the 248k constants. The constants survive only in
  `config.rs`'s GGUF-only EOG fold and `ConstraintFactory::embedded`. A new call site
  that reaches for a constant will be a wrong number on this vocabulary, not a missing
  one. `TOKENIZATION_RULES_VERSION` stays at 3 on purpose.
- **Identity for a safetensors directory** is provenance first, and the rule is about the
  REPO, not the snapshot (tightened 2026-09-07): a directory is `Official` when it sits
  under the entry's repo directory in the hub cache, at the entry's subdirectory, with
  every registry file present, under ANY snapshot commit. The repo directory is what
  separates Z-Image from base, their configs being byte-identical. Canonicalize the
  DIRECTORY and never a file, hub cache files being symlinks into shared blobs. Failing
  that it is `Assumed`, with `rope_theta` picking the release (5e6 Instruct-2507, else
  base, which wins the tie with Z-Image). There is no name inside the set, so the
  `general.name` passes are unreachable here. `--model-size` stays a cross-check and a
  disagreement is a startup error.
- **Thinking on the Qwen3 dialect is MODEL-opened, not prompt-seeded.** A prompt's
  reasoning state is `chat::ThinkingEntry` (Answer / Seeded / ModelOpens), and
  `ChatDialect::model_opens_thinking()` is true for Qwen3 alone: the template writes no
  `<think>` after the assistant header, so the model writes its own. The decode loop
  enters thinking on a think opener only when it is the reply's FIRST tagged token, so a
  later `<think>` is text. Two things follow that a change here will break silently: the
  think budget must hold until the opener rather than arm at position 0, and `--min-think`
  is gated on being inside a block, or under ModelOpens it bans the stop tokens as a
  minimum answer length. Marker retention is an explicit `MarkerText` policy per call
  (Strip for the event consumers, Keep for callers that split on the literal marker), not
  a dialect property; a review round caught the raw-text loops having silently changed for
  the shipped checkpoints.
- **`CheckpointSource` is the one open seam.** Every consumer that used to call
  `gguf::open` itself now routes through it, so a new checkpoint consumer goes there and
  not beside it. It carries the caller's `Device` on the safetensors arm on purpose: a
  fresh `Device::new_metal(0)` inside the loader is a DIFFERENT candle device from the
  Generator's, and every op between them a DeviceMismatch. Note that `Qwen3Set::open`
  scans all 8 GB on every open (~1.4 s in dev), which every routed caller now pays,
  `encode-text` and `logits-dump` included; it is the first thing to fix when
  `servable()` flips.
- **The Python exception.** `scripts/zimage-ref-dump.py` is the only Python in the repo,
  run by hand under `uv` in a throwaway venv, never in CI. It exists because the encoder
  has no ONNX export and there is no bun path to torch. It is not a precedent.
- **A GGUF whose arch string is `qwen3` is refused**, pointing at the safetensors
  directory. Safetensors is the form this architecture ships in here.
- **Serve carries TWO vocabularies and they follow the request's target**, not the
  process (Arc 3). `src/serve/vocab.rs` holds one tokenizer plus grammar trie per
  `VocabFamily`, built together so they cannot disagree, and `constrain::shared()` is off
  every serve request path. Build a trie from the file its tokenizer was parsed from
  (`constrain::for_tokenizer` asks the tokenizer, which remembers its source); a trie and
  a tokenizer from different files agree about nothing and the symptom is wrong output,
  never an error. Lookup order for a family: the embedded copy for Qwen 3.6 with no search
  at all, then the served file's own, then any cached registry checkpoint of the family,
  then an ERROR naming the fetch. Never add a fallback to the embedded tokenizer: it would
  answer fluently in the wrong vocabulary. Mask width is `VocabFamily::logit_width()`,
  248320 against 151936, a registry constant because it is asked before any file is open.
- **Prompt admission uses the REQUEST TARGET's trained context, not the served
  checkpoint's** (fixed 2026-09-07, d48a3f4). `Model::trained_context()` is the registry
  constant, read off the cached files: 262144 everywhere except `Qwen/Qwen3-4B` and the
  Z-Image encoder at 40960. The handler caps it by `--context-length` and names the
  checkpoint in the refusal. Before the fix the check used `AppState.max_ctx`, the served
  checkpoint's, which on a base-default server clamped Instruct-2507's 262144 to 40960;
  do not reintroduce that by reaching for the served window in a handler. The engine
  re-derives its own limit at load and that stays authoritative for what actually runs.

## Z-Image-Turbo (diffusion, vendored candle module)

The first thing here that is not a language model. `xwen image --prompt <text>` renders a
PNG: the repo's own Qwen3-4B encoder at hidden index 35, the S3-DiT transformer, eight
flow-match Euler steps, the Flux VAE decoder. [docs/zimage.md](docs/zimage.md) is the
architecture and the full trap list,
[docs/decisions/zimage.md](docs/decisions/zimage.md) the decisions,
[docs/records/zimage-pipeline.md](docs/records/zimage-pipeline.md) the arc.

Shape: 6,154,908,736 parameters, dim 3840, 30 heads of 128 with no GQA, SwiGLU 10240,
34 blocks executed (30 `layers` plus 2 `noise_refiner`, all modulated, plus 2
`context_refiner` that are not), RMSNorm eps 1e-5 in the HF `+eps` form, one
affine-free LayerNorm at eps 1e-6 in the final layer. Latent 16 channels at 8x, patch 2,
so 1024x1024 is 4096 image tokens. Shipped F32 (24.6 GB, three shards) but bf16 values
in an F32 container, so the load-time cast to bf16 loses nothing (12.3 GB resident);
**activations are f32 and only the weights bf16** as of 2026-09-07, which is what the
tensor gemm takes and what took step-0 parity to cosine 0.999999. VAE is the Flux VAE,
bf16 on disk, 168 MB, run in f32 under
`force_upcast`, `shift_factor` 0.1159 and `scaling_factor` 0.3611. Scheduler is
`FlowMatchEulerDiscreteScheduler`, static `shift` 3.0, 8 steps, no CFG.

The seams, so a change lands in one place:

- **`src/zimage/`** is candle's `z_image` at rev 21cca0b (PR #3261), vendored and
  corrected in four places toward the reference (caption padding after the embedder with
  the learned pad token and no mask; the sigma grid; rope tables in f64 and the rotation
  in f32; QK-norm eps from the config). `src/zimage/pipeline.rs` is ours. Never
  "resync" it with upstream: the corrections are the point, and its scheduler, padding
  and step count were all wrong.
- **`src/zimage/linear.rs`** is every projection in the transformer, and it is ours, not
  vendored (2026-09-07): a bf16 `[out, in]` weight plane through `ops::matmul_bf16`, the
  Metal-4 cooperative-tensor kernel the language models prefill on, at 36.6-38.6 TFLOPS
  where candle's steel gemm does 14-15.6 whatever dtype it is handed. Two things there
  fail silently if changed. The activation stream is f32 because the kernel's contract is
  f32 in and f32 out, so a bf16 stream reintroduces a cast at every one of the seven
  token-width linears per block. And the kernel stages each weight tile to f16, so
  `ensure_weights_fit_f16` refuses at load, by tensor name, any projection with
  `|w| > 65504`; never widen that to make a load succeed, the shipped checkpoint's
  largest weight being 14.0. `XWEN_ZIMAGE_LINEAR=candle` is the bisect arm beside
  `XWEN_ZIMAGE_ATTN`, sharing no matmul code with the shipped path, and
  `tests/zimage_microbench.rs` is the ignored bench that priced the choice
  (decisions/zimage.md "The transformer's linears run on xwen's Metal-4 tensor gemm").
- **`src/zimage/profile.rs`** is the per-stage profiler, `XWEN_ZIMAGE_PROFILE=1`, printing
  transformer stages as a mean per step over steps 2..8 and the VAE decode's stages; off,
  it is one `Option` check per site. **Profiled numbers are not figures**: every mark syncs
  AND evicts candle's buffer pool, so a table runs 1.39x high at 1024x1024 and its small
  elementwise rows about 1.9x, while the gemm and sdpa rows hold up. Quote a step at its
  3.6 s steady state, never as an 8-step mean (docs/benching.md, and
  decisions/measurement-discipline.md "A Z-Image step is quoted at steady state").
- **`Model::text_encoder()`** is where the conditioning comes from. The pipeline entry
  holds no encoder spec of its own and `src/zimage/` has no text encoder: it takes a
  `[T, 2560]` caption tensor and `XwenModel::encode` produces it. candle's own
  `text_encoder.rs` was deliberately not vendored, being an ungraded second
  implementation of the one thing in this path that has a reference gate.
- **`Format::Diffusion`** is a third registry format beside GGUF and safetensors, and
  exactly one of `is_gguf()`, `is_safetensors()` and `is_diffusion()` is true per entry
  (tested). `is_safetensors()` means specifically "a Qwen3 set the Qwen3 loader opens",
  which is why the pipeline could not be one: `identify_cached_dir` iterates those and
  would have made the snapshot root identify as a language model.
- **`check_size`** owns the accepted resolutions, and there are three rules: both sides
  positive multiples of 16, `(w/16) * (h/16)` a multiple of 32, and each side at most
  8192 px, which is the 512 positions its RoPE table holds. The `x_pad_token` path that
  would lift the second rule is unimplemented, so a size needing it is refused with the
  token count in the message rather than silently padded. The third rule exists because
  candle's Metal `index_select` clamps an out-of-range position instead of failing, so
  past it the run returns a wrong image rather than an error; the same bound is re-checked
  inside `forward` against the loaded `axes_lens`, which also catches a caption long
  enough to push the image past axis 0's 1536.
- Aliases: `zimage-turbo` is the PIPELINE and `zimage-turbo-encoder` is the encode-only
  entry. Neither is auto-fetched and neither appears in `/v1/models`. The pipeline is
  served by the images route below and by nothing else; `servable()` is still false for
  both, so the chat routes, `generate`, `chat` and `batch` refuse them with the one
  sentence `not_servable_reason` owns.
- **`src/serve/images.rs`** is the whole serve surface for images (2026-09-07): `POST
  /v1/images/generations`, `/images/generations` and `/proxy/openai/images/generations`
  on one handler, and the `image-engine` thread beside the language engine with its own
  lazy load and its own idle unload on `--idle-unload`. The two engines do not coordinate
  residency, so both can be resident inside one idle window (fine at 20 GB plus 20 GB,
  thrashes with Flash-Next); not taken now, with the reopen condition in the record. The
  prompt rendering both `xwen image` and the route use is
  `zimage::conditioning::prompt_ids`, so a change to how the caption is rendered lands in
  one place. The `model` rule is split by path (full name or nothing on the first two,
  anything on the proxy path, logged), `negative_prompt` and `guidance_scale` are 400s
  because Turbo would ignore them silently, and the route never returns 401, 402, 409 or
  429: a missing key is a 403 and a full queue a 503, because the ComfyUI client rewrites
  those four into comfy.org prompts (decisions/zimage.md "CLI first, then serve as
  OpenAI", the Arc C paragraph).

Traps that are silent, the short list (all of them, with evidence, in docs/zimage.md):
rope is INTERLEAVED-pair, not the NEoX every Qwen graph here uses; the joint sequence is
IMAGE FIRST and the output is a prefix narrow; modulation is `1 + scale` with `tanh`
gates and no shift term, in the order scale_msa/gate_msa/scale_mlp/gate_mlp; the `t` fed
in is `1 - sigma` AND the model output is negated; the 32-multiple pad tokens are
learned, applied after the embedder, and NOT masked; 8 steps, not the 9 the model card
says; the static shift is 3.0 and `calculate_shift` is dead code; fp16 is disqualified,
not merely slower, because activations exceed 65504 and the image comes out black.

The transformer IS graded, as of 2026-09-07: `tests/zimage_parity.rs` (run with
`--ignored`, 13-14 s) gates the step-0 velocity against diffusers' fp32 run of the same
weights at cosine 0.998 and mean relative error 0.04, with two wrong-graph brackets run
every time, gates the VAE alone at 60 dB PSNR, and reports the final latent and the image
PSNR after eight steps. The fixture under `tests/fixtures/zimage-transformer/` holds BOTH
inputs, the noise and the caption features, so the gate grades the transformer and not the
encoder; `scripts/zimage-ref-dump.py --stage transformer` regenerates it in under two
minutes, and `xwen image --latents <file> --cap-feats <file> --dump <dir>` runs the same
comparison by hand (`--cap-feats` skips the encoder entirely and the prompt is ignored).
It reads cosine 0.999999 and mean relative error 0.0008 as of 2026-09-07, an order of
magnitude inside torch's own bf16 spread, on the f32 activation stream; on the bf16
stream it read 0.999302 / 0.0205, which was 1.6x noisier than torch and was the
activations rather than the sdpa kernel (docs/records/zimage-perf.md). One thing not to
expect: there
is no `CheckpointSource` arm for diffusion weights; `ZImagePipeline::load` goes through
candle's `VarBuilder` directly, casting fp32 to bf16 one tensor at a time, which is a
deliberate deferral until a second consumer exists.

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

**A green `cargo test` does not mean the cache-backed tests ran.** A great many read the
real checkpoints and self-skip when the HF cache lacks them. Every one of those skips goes
through `crate::test_support` (2026-09-07), which prints a SKIPPED line naming the file
and the fetch that would supply it, and `XWEN_REQUIRE_HF_CACHE=1` turns every skip into a
failure. Run `XWEN_REQUIRE_HF_CACHE=1 cargo test --release` when green has to mean those
paths really executed, and route any NEW self-skip through `test_support` rather than an
`eprintln!` and a `return`: an outside review found the two tests proving the Qwen3
vocabulary never crosses over were silently vacuous on a clean checkout.

**A bit-identical A/B is a result only when the two sides are known to be different
code.** `XWEN_QWEN3_ATTN=sdpa` was read as ruling out the flash kernel because it matched
to the bit; candle's Metal sdpa above one token dispatches the steel kernel that
`flash.metal` is a copy of, so it was one kernel compared with itself. The arm is a real
f32 chain since 30995b9 and its tiny-model test asserts a NONZERO difference under the
bar, which is the shape a reference arm's test should have.

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
2026-09-05), **Flash-Next 428-456 at 131424** (2026-09-06, sparse-tile attention; 231 that
morning), the dense **27B 702 at 880 tokens** (2026-07-29; recorded as "at 925" until
2026-09-06, which is the bench fixture's NAME and not its token count). Read a drafted or an A/B figure
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
and loads the checkpoint's own sidecar. **Except on the 35B-A3B as of 2026-09-06**, where
the default is off: its drafted arm now reads below plain at every length (-8% at 1k
tokens, -37% at 16k), the router gemv having lifted plain decode by 10.3% past the level
the drafting defaults were fitted against, so there `--draft official` is the opt-IN. The
heading above still holds for the other two. What silence means is
`Model::draft_default_on()`, on every surface; the fitted `p_min` and depth are
untouched, and the standing retune decides whether the 35B goes back on. Which kind a
checkpoint ships is
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
dialect has no per-request effort knob (server-wide default applies; retired 2026-09-06
with a reopen condition), and `frequency_penalty`/`repetition_penalty`/`min_p` stay
accept-and-drop while `presence_penalty` is consumed on every dialect that has a field
for it (2026-09-06).

API model names are FULL names only (`Qwen3.6-27B`, `Qwen3.6-35B-A3B`, `Qwen3.8-27B`,
`Qwen3.8-Flash-Next`, and since 2026-09-07 `Qwen3-4B` and `Qwen3-4B-Instruct-2507` —
`Model::full_name`, matching `general.name` and the repo), plus
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

The two dense Qwen3-4B language models joined serve and batch on 2026-09-07, which is
what forced the vocabulary to become per target (`src/serve/vocab.rs`, the trap list in
the Qwen3-4B section above, decisions.md "The tokenizer and the grammar trie follow the
request's target"). They are listed and selectable only while cached, `auto_fetch` being
false the way it is for Flash-Next. The Z-Image encoder is never listed and is refused on
every surface. Their 400s: `enable_thinking` on Instruct-2507 (its template has no
reasoning mode), a request-level `reasoning_effort` on either (the 3.6 rule, unchanged,
`supports_reasoning_effort()` being Qwen38-only), and tools on either dialect (the call
format is JSON and the serve parser reads `<function=`). A server-wide thinking default
stays INERT rather than refusing every request, which is the same rule
`reasoning_effort` already followed: the dialect drops the resolved value in chat.rs, so
an operator default can be silently ignored while an explicit request is an error.

The images route joined the same evening (`src/serve/images.rs`, the Z-Image section
above). Two things it changed on the shared surface: `/health` gained
`image_model_loaded`, a second flag from a second engine, beside the `model_loaded` and
`model` pair that still describe the language engine alone; and `require_api_key` answers
403 rather than 401 on the three images paths (`images::is_images_path`), because the
ComfyUI client turns a 401 into a comfy.org login prompt before reading the body. The
routes are registered inside the OpenAI-dialect block and ABOVE the body-limit layer, like
every other body-taking route.
