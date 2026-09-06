#!/usr/bin/env bash
# Build the upstream llama.cpp parity oracle (llama-eval-callback, llama-cli,
# llama-server, llama-tokenize) in reference/llama.cpp, then xwen's own
# llama-logits-all on top of it.
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

# scripts/llama-logits-all.cpp is xwen's own oracle tool, not upstream's: it dumps
# llama.cpp's logits at every position of a prompt, which no shipped binary does.
# It links the libraries cmake just produced and lands beside them in build/bin, so
# @loader_path is all the rpath it needs. Keep the compile line here — it is the
# only record of how the binary was produced.
SRC="$(cd "$(dirname "$0")" && pwd)/llama-logits-all.cpp"
clang++ -std=c++17 -O2 -o "$REPO/build/bin/llama-logits-all" "$SRC" \
  -I "$REPO/include" -I "$REPO/ggml/include" \
  -L "$REPO/build/bin" -lllama -lggml -lggml-base \
  -Wl,-rpath,@loader_path

echo
echo "built binaries in $REPO/build/bin"
