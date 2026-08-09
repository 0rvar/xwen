# xwen

Pure-Rust, Metal-only inference engine for **Qwen3.6-27B** and **Qwen3.6-35B-A3B**
(GGUF), optimized for a single Apple Silicon machine (M5 Max). Manual fork of the
laguna/maxuna engine: candle-based, mmap no-copy weight loading, vendored Metal
kernels, speculative decoding, and an HTTP server speaking Anthropic Messages and
OpenAI Chat Completions.

**Status: runs, and matches upstream.** The 35B-A3B generates correctly end to end
(greedy, chat template, thinking split, clean stops) at ~59 tok/s decode, and both
checkpoints now pass the parity gate against upstream llama.cpp on identical GGUF
weights (`docs/parity.md`). No DeltaNet Metal kernels yet. See `TODO.md` for the
priority ledger and `docs/log.md` for the narrative.

## Docs

- `docs/decisions.md` — every deliberate choice and refuted direction, with evidence
- `docs/log.md` — dated engineering narrative
- `docs/parity.md` — verification runbook (vs upstream llama.cpp)
- `CLAUDE.md` — agent context: ground truth, architecture cheat sheet, hazards
- `TODO.md` — deferred-work ledger

## Build

System Rust toolchain, system Apple CLT SDK (the nix shell deliberately provides no
SDK — see flake.nix). `cargo build --release`. Ops tests need a Metal device.

## Models

Default checkpoints: `ggml-org/Qwen3.6-27B-GGUF` and `ggml-org/Qwen3.6-35B-A3B-GGUF`
(Q4_K_M), resolved through the HF cache, downloaded on first use.

## Speculative decoding

DFlash drafter sidecars ship alongside both checkpoints and are adapted. It is
**opt-out** as of 2026-07-29: a zero-flag run speculates with the checkpoint's official
sidecar, `--no-draft` decodes plain, and `--draft <gguf>` swaps in a custom drafter.
At the defaults fitted 2026-08-08, drafting measured +46 to +52% on the 27B (both prompt
kinds) and +26 to +28% on code / +15 to +17% on chat on the 35B-A3B, over plain decode
on the same machine state (greedy, 128 tokens, warm, medians of 3 reps, two independent
runs). Acceptance at those defaults is 78-86% on the 27B and 68-74% on the 35B-A3B; it
trades against draft length, so raising `--draft-p-min` buys acceptance and loses tok/s.
The default costs a sidecar load per run (3.5 GB on the 27B, 0.8 GB on the 35B-A3B).

**`--draft-p-min` has a per-checkpoint default: 0.5 on the 27B, 0.3 on the 35B-A3B**
(`Model::draft_p_min_default`). The 27B's target forward is expensive, so it wants short
confident drafts; the 35B-A3B's is cheap enough to profit from drafting deeper at lower
acceptance, and 0.5 costs it ~2.5%. Passing the flag, or `draft.p_min` in a serve
config, overrides it as before. `--draft-pause-margin` stays a single 1.0 for both.
`bun scripts/retune-draft.ts` re-fits both knobs and prints recommendations; it never
edits a default. See docs/decisions.md "Speculative decoding".

`--draft` should reproduce `--no-draft`; `bun scripts/spec-equivalence.ts` checks that on
both models in two modes — greedy, and sampled at a fixed seed (the only one that can
catch the spec loop drawing from the RNG a different number of times than plain decoding).
It prints the fork point when they differ; a near-tie landing differently is expected, a
first-line fork in sampled mode is not. See the script's header.

## Verifying a change

Any change to model math re-runs the parity gate. It compares our forward pass against
upstream llama.cpp on the identical GGUF, so it needs the oracle built once:

```bash
just init                                     # fetch the llama.cpp submodule (pinned)
bash scripts/build-llamacpp.sh
bun scripts/parity-gate.ts                    # 35B-A3B, all tiers
bun scripts/parity-gate.ts --model-size 27b   # 27b dense
```

`docs/parity.md` is the runbook: tiers, floors, tap mapping, and the pinned oracle
commit.
