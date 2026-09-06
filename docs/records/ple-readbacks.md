# 2026-09-05 — PLE batches its three device-to-host readbacks

Moved verbatim from [docs/log.md](../log.md) on 2026-09-06; the log keeps this entry's
opening paragraph and links here.


The next Flash-Next perf candidate selected from TODO.md was the PLE readback
collapse. `PleLayer::forward` previously called `to_vec1` separately for key, value
and carrier, and candle's `MetalStorage::to_cpu` allocated, blitted and waited EACH
time. At single-token decode, `readback_inputs` now encodes all three copies into
one shared staging buffer and waits once. Multi-token prefill keeps its original
independent transfers. The key and carrier are 10240-wide, the value 2560-wide: 90 KiB/token
total. The host gate, conv, recurrent state, table gather and frozen references are
unchanged. `XWEN_PLE_READBACK_CLASSIC=1` restores the old transfers. The ownership,
offset and temporary-memory choices are in decisions.md "PLE decode readbacks".

**Verification:** 17 tests pass with
`cargo test --release --lib qwen4exp::ple::tests -- --test-threads=1`, including
the new bitwise readback comparison at one token and the full 2048-token chunk, unequal plane widths, offset/strided inputs,
F16 conversion, empty inputs and CPU fallback. The existing tests cover the fixture,
real geometry, chunk continuation, checkpoint/rollback and image state. The empty
F16 test initially hit candle's divide-by-zero dispatch for an empty conversion; the
helper now skips empty planes before conversion or blitting.

`bun scripts/parity-gate.ts`: **ALL PASS (6 graded)** on the 35B, with the previous
record's exact summary: strict cosine 1.000000, mm 0.999618; decode 63/64, 62/64,
61/64 with 1/2/3 excused and zero hard mismatches; ppl Δnll 0.001179. Its disposable
`/tmp` reference caches needed rebuilding (21 s full-logit, 74–83 s per greedy
fixture); the committed ppl reference was reused and no floor or reference code
changed. Flash-Next still cannot use that harness (parity.md's standing limitation).
Instead, the old readback arm free-ran 64 steps on each of code-short/text-mixed/
long-mixed and the new arm replayed those histories: **192/192 exact**, including
all top-five logit values, L2 norms and nonfinite counts. This is equivalence to the
pre-change engine, not a fresh llama.cpp comparison.

**Initial all-length experiment:** 18 full-model runs: four AB/BA rounds over
612 and 3677 tokens, plus start/end classic anchors at 612. Both the arm order and
length order reverse each round, with 60 s idle before each round and both anchors.
The initial candidate predates the final `n != 1` guard at the readback call;
omitting that guard reproduces the all-length experiment.
Every run uses `XWEN_BENCH=1 generate --model-size flash-next --no-draft --raw
--temp 0 --top-k 1 -n 128 --stats --max-ctx 8192`, with the committed long-mixed
fixture once or six copies joined by two newlines. No concurrent GPU work. Power
before/mid/after: `powermode 0`; no high-power claim. All outputs are byte-identical
between arms at each length. Metrics were routed to a separate file.

Rates below are tok/s; C = classic independent reads, B = all-length batching:

| prompt | round/order | prefill C | prefill B | decode C | decode B |
|---|---|---:|---:|---:|---:|
| 612 | 1 / CB | 1098.2 | 1098.4 | 45.36 | 46.49 |
| 612 | 2 / BC | 1098.7 | 1088.4 | 45.92 | 46.98 |
| 612 | 3 / CB | 1062.6 | 1080.5 | 45.81 | 45.16 |
| 612 | 4 / BC | 1104.0 | 1098.1 | 45.19 | 47.03 |
| 3677 | 1 / CB | 1011.5 | 967.2 | 44.88 | 45.56 |
| 3677 | 2 / BC | 974.0 | 988.6 | 44.80 | 46.35 |
| 3677 | 3 / CB | 997.2 | 967.5 | 44.69 | 45.06 |
| 3677 | 4 / BC | 1000.3 | 1003.6 | 45.13 | 46.12 |

Median decode rises 2.5% short and 2.2% long, but one short pair loses. Prefill
medians read −0.5% short / −2.1% long and the long-prompt pair ratios depend on
order. **Drift flag:** the decode anchor is 46.29 → 47.67
(+3.0007%, just over 3%); its prefill is
1106.4 → 1068.0 (-3.47%). Keep every cell, including
the loss; this is directional evidence for decode, not a clean absolute-throughput
headline. **Decision: enable batching only at seq == 1.** The wait cost matters
per token, while prefill has no established gain and its combined staging allocation
is larger. Multi-token prefill therefore keeps the existing path.

The ignored `ple_readback_bench` independently measures the complete transfer
transaction (including queued producers), with alternating arms and a wait per
transaction because a CPU readback requires completion. Two probes: decode
0.50–0.59 → 0.17–0.23 ms; 512-token reads approximately level; 2048-token reads
13–14 → 11–12 ms. These are isolated transfer costs, not per-step profiler charges
and not end-to-end throughput estimates.

**Final shipped configuration, decode only:** the 17 PLE tests and all six
35B parity checks pass again. A fresh 3677-token CB / BC check uses the same
128-token commands, a 60 s idle before each pair, and the first/last classic runs
as its start/end anchors. `powermode 0` before and after. Results:

| order | prefill classic | prefill decode-batched | decode classic | decode batched |
|---|---:|---:|---:|---:|
| CB | 992.1 | 988.7 | 44.99 | 46.23 |
| BC | 986.3 | 1005.9 | 44.57 | 45.89 |

Decode **+2.74% / +2.95%**, central rates **44.78 → 46.06 tok/s (+2.85%)**.
Classic decode anchor drift −0.94%, prefill −0.59%, both below the 3% flag. Prefill
varies −0.34% / +1.99% despite running the exact same transfer path, so no prefill
gain is claimed. Every run emits the same 128 tokens, byte-identical to the initial
sweep's long-prompt output. The decode-only route is the shipped default;
`XWEN_PLE_READBACK_CLASSIC=1` is its control. This measures `generate`; serve uses
the same PLE forward but was not independently benchmarked in this session.

**Candidate screened first, left unchanged:** hc's two Q8_0 decode projections on
the existing vendored `matmul_q8`. The new ignored `hc_q8_decode_bench` rotates 96
distinct shared QMatMul/kernel allocations (334 MB) at each actual shape, retains
outputs through the batch flush, and alternates arm order. Two exploratory probes
were too noisy to qualify the down projection; the up projection lost in both
probes (the vendored kernel launches 5120 two-row threadgroups at k=320). No route
changed and no end-to-end gain is claimed for that candidate. The bench remains available
for a future shape change, and the ledger annotation preserves the unresolved down
projection.

Artifacts for this session: `/tmp/xwen-ple-perf/` (replay JSONs, gate/build/test and
benchmark logs, runner scripts, raw per-run metrics); the exploratory Q8 logs are
`/tmp/xwen-hc-q8-bench{,2}.log`. These are disposable; the protocol and measured
results are recorded here so their later cleanup does not erase the result.
