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
