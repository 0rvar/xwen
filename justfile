init:
    git submodule update --init --recursive

# Build the pinned llama.cpp parity oracle (see docs/parity.md "The oracle").
# reference/llama.cpp is a git submodule; `just init` fetches it at the pinned sha.
oracle:
    bash scripts/build-llamacpp.sh

# The full parity cycle for one checkpoint. `just parity` gates the 35B-A3B,
# `just parity 27b` the dense 27B.
parity size="35b":
    bun scripts/parity-gate.ts --model-size {{size}}
