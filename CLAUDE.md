# CLAUDE.md

Read `README.md` first for what this is. This file is the context that is NOT obvious
from the code: ground-truth sources, hard-won gotchas, workflows. When picking up a
TODO.md item, read this whole file first — items acquire traps and the traps get
documented here.

Doc system: `README.md` is getting-started + the practical surface; `docs/decisions.md`
is the WHY, by topic — every deliberate choice, default, policy, and refuted direction,
with its evidence; `docs/log.md` is the chronological narrative both point into;
`docs/parity.md` is the verification runbook; `TODO.md` is the forward ledger.

## Non-negotiables

- Design target: maximum tok/s for Qwen3.6-27B and Qwen3.6-35B-A3B GGUF on this one
  machine (M5 Max, Metal). Batch 1. No portability hedging.
- TODO.md is the deferred-work ledger. Scope is never silently dropped: it ships, or it
  becomes a ledger item with context. Ledger items are never deleted, only annotated.
- Every shipped arc updates the docs before it's done: dated log.md entry, README if
  the surface changed, decisions.md if a decision was made/changed/refuted. A TODO.md
  update alone is not sufficient.
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
defaults 1.0 / 0.95 / 20. Generation prompt ends inside an open `<think>\n` unless
thinking is disabled (then a closed empty block is emitted).

## Checkpoint location

HF cache (`HF_HUB_CACHE` > `HF_HOME/hub`), cache-first via hf-hub, download on miss.
Default repos/files (hub.rs): `ggml-org/Qwen3.6-27B-GGUF` and
`ggml-org/Qwen3.6-35B-A3B-GGUF`, Q4_K_M. Sizes: 19.1 GB (27B), 20.4 GB (35B). Q8_0:
28.6 / 36.9 GB. DFlash drafter sidecars: `dflash-*-BF16.gguf` (3.5 GB / 0.8 GB). MTP
sidecars exist but are unused. `mmproj-*` files are the vision tower — never load them.
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
  absent), lm_head Q6_K, norms/routers/conv/ssm_a/dt F32, token_embd Q4_K.
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
kernel-vs-reference invariants and is the fast pre-check.

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

## Perf state

As of 2026-07-29 (Automatic power mode, interleaved protocol, --no-draft; full
history in log.md): 35B-A3B decode 103.1 tok/s, prefill ~2550@925 / ~2320@4k;
27B decode 19.6, prefill ~270@925 / ~236@4k (prefill DEGRADES with length — the
sequential DeltaNet reference scan, P8b owns it). Load 2.8-3.0s, 19.2 GB
resident at max_ctx 8192; cold first run adds ~9s of Metal pipeline compilation.
Head-to-head vs llama.cpp e9fa078 -fa 1, same GGUFs, same machine (log.md
2026-07-29): xwen wins decode on both models (1.05x / 1.02x), 35B prefill near
parity at steady state, 27B prefill 1.8-2.1x BEHIND — llama.cpp prefills the
DeltaNet layers chunked (chunk=64), xwen still runs the sequential scan, and
the 27B has 48 such layers at inner 6144 vs the 35B's 30 at 4096. Bandwidth
framing: 27B decode is at ~57% of the 614 GB/s peak (near the wall, ceiling
~23-30 tok/s); 35B decode at ~32% (still launch-bound, ceiling ~225-290);
prefill is compute-bound on both (~15-16 achieved TFLOP/s each). llama.cpp's
prefill thermal-boosts harder than xwen's (-17% vs -5% settling) — never read
a first-reps prefill ratio as steady state.

## DFlash (PLANNED — sidecars exist, adaptation not started)

laguna's dflash.rs is kept: ggml-org ships DFlash drafters for both models (arch
`dflash`; 27B: 5 layers, sliding_window 2048, taps [2,17,32,47,62], mask 248070; 35B:
6 layers, sliding_window 4096, taps [2,7,12,17,23,28,33,38], mask 248077; both
block_size 16, fc.weight over concatenated tapped layer outputs, own ffn_norm,
q/k-norms [128]). Adaptation = repoint decoder_arch check, tap indices, mask token,
and re-tune the pause controller. Recurrent-state rollback for the verify walk needs
delta/conv snapshot slots (llama.cpp keeps K=n_rs_seq+1 most-recent-first snapshots —
mirror that shape).

## serve (INHERITED, needs template adaptation)

The serve/ tree runs as forked. Not yet adapted: ChatML/tool-call parsing in the
dialect layers (Qwen's `<function=...>` XML-ish call format, string-args-raw rule),
thinking semantics (open-`<think>` seeding, preserve_thinking), and prefix-cache
snapshots carrying recurrent state (see decisions.md "Serving").
