#!/usr/bin/env bash
# Build the upstream llama.cpp parity oracle (llama-eval-callback, llama-cli,
# llama-server, llama-tokenize) in reference/llama.cpp.
#
# reference/llama.cpp is a shallow git SUBMODULE; `just init` fetches it at the
# pinned commit, which docs/parity.md also records in human-readable form. Only
# build/ is gitignored. Moving the pin means checking out the new sha inside the
# submodule and staging the resulting gitlink — a reviewable diff, deliberately,
# because a different oracle build moves the achieved cosines and invalidates the
# calibrated floors. Re-run the calibration in docs/parity.md "Floors" when you do.
#
# cmake comes from an ephemeral nix shell (Homebrew here is nix-managed); nix's
# cmake skips Apple SDK autodetection, so the sysroot and framework path are
# passed explicitly. It must be the SYSTEM CLT SDK — a nixpkgs Apple SDK links
# pre-Metal-4 and every tensor kernel fails at runtime. BLAS is off: Metal does
# the real work.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)/reference/llama.cpp"
cd "$REPO"

SDK="$(xcrun --show-sdk-path)"
nix shell nixpkgs#cmake --command bash -c "
  cmake -B build -DGGML_METAL=ON -DGGML_BLAS=OFF -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_BUILD_TESTS=OFF \
    -DCMAKE_OSX_SYSROOT='$SDK' \
    -DCMAKE_FRAMEWORK_PATH='$SDK/System/Library/Frameworks' &&
  cmake --build build -j
"

echo
echo "built binaries in $REPO/build/bin"
