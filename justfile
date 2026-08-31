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

# Install the xwen binary. --locked is load-bearing: `cargo install` ignores
# Cargo.lock by default, and a re-resolved metal/objc2 crate set has produced
# a binary whose Metal-4 kernels fail to compile at runtime (dense_mm.metal,
# mpp::tensor_ops identifiers undeclared). See CLAUDE.md "Operational hazards".
install:
    cargo install --path . --locked
