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

DFlash drafter sidecars ship alongside both checkpoints and are adapted (`--draft
official`, or `--draft <gguf>` for a custom one). It is **opt-in**: measured a 1.5-7.4%
decode gain on the 27B and an 11.5-12.7% loss on the 35B-A3B, whose plain step is too
short to absorb the drafter's per-token cache sync. Acceptance is 85-95% on both. See
docs/decisions.md "Speculative decoding" for why, and TODO.md P9 for what would change it.

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
