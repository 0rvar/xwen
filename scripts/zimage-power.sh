#!/usr/bin/env bash
# Read the GPU power envelope under a Z-Image render.
#
# Runs `powermetrics` (needs sudo, so run this from a user shell) beside one
# `xwen image` run, timestamps every line xwen prints, and hands both logs to
# scripts/zimage-power-summary.ts, which prints GPU power, GPU frequency and the
# thermal pressure level per denoising step. The question it answers is whether
# the per-step ramp (docs/perf-state.md) is a power or thermal cap.
#
# Usage:
#   scripts/zimage-power.sh                      # 1024x1024, 8 steps, seed 7
#   scripts/zimage-power.sh --width 512 --height 512
#   scripts/zimage-power.sh --steps 24           # does the plateau hold or drift
#
# Everything after the script name is passed to `xwen image`. XWEN_BIN picks the
# binary (default target/release/xwen, copied to /tmp first so a cargo build in
# the tree cannot swap it mid-run). Logs land under /tmp/zimage-power-<stamp>/.
set -euo pipefail

cd "$(dirname "$0")/.."

BIN="${XWEN_BIN:-target/release/xwen}"
if [[ ! -x "$BIN" ]]; then
  echo "no binary at $BIN; cargo build --release first or set XWEN_BIN" >&2
  exit 1
fi

STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="/tmp/zimage-power-$STAMP"
mkdir -p "$OUT"
cp "$BIN" "$OUT/xwen"

PROMPT="a lighthouse on a rocky shore at dusk, oil painting"
ARGS=(--prompt "$PROMPT" --seed 7 -o "$OUT/out.png")

echo "pmset: $(pmset -g | grep -E 'lowpowermode|powermode' | tr -s ' ' | tr '\n' ';')"
echo "logs: $OUT"

# Sudo up front so the prompt does not land in the middle of the run.
sudo -v

# Same GPU lock the coding agents use, so a run never overlaps a bench or a test.
until mkdir /tmp/xwen-gpu.lock 2>/dev/null; do
  echo "waiting for /tmp/xwen-gpu.lock (held by $(cat /tmp/xwen-gpu.lock/owner 2>/dev/null || echo '?'))"
  sleep 5
done
echo "zimage-power.sh $$" > /tmp/xwen-gpu.lock/owner
release() {
  rm -f /tmp/xwen-gpu.lock/owner
  rmdir /tmp/xwen-gpu.lock 2>/dev/null || true
}
trap release EXIT

# 250 ms samples: a step is 1.3-2.1 s, so five to eight samples per step.
sudo powermetrics --samplers gpu_power,cpu_power,thermal -i 250 > "$OUT/powermetrics.log" 2>&1 &
PM_PID=$!
sleep 1

# Timestamp each xwen line with epoch seconds so the summary can bin power by step.
"$OUT/xwen" image "${ARGS[@]}" "$@" 2>&1 \
  | while IFS= read -r line; do printf '%s %s\n' "$(date +%s.%N)" "$line"; done \
  | tee "$OUT/xwen.log"

sleep 1
sudo kill -INT "$PM_PID" 2>/dev/null || true
wait "$PM_PID" 2>/dev/null || true

bun scripts/zimage-power-summary.ts "$OUT" | tee "$OUT/summary.txt"
