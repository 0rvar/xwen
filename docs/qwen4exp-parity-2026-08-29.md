# U7 — Qwen3.8-Flash-Next parity vs the llama.cpp oracle (2026-08-29)

The full U7 record: method, artifacts and every number behind the "P2 parity result"
section of docs/qwen4exp-port.md, which carries the summary and the decisions.
Point-in-time — shas, file:line refs and the scratchpad artifact paths all go stale.


Date: 2026-08-29. Machine: M5 Max, `lowpowermode 0` (high-power mode NOT positively
confirmable on this machine — no `powermode` key — so it is not claimed).

Model under test (both sides took the FIRST shard; the 4-shard set is opened
transparently by both engines):

```
/Users/orvar/.cache/huggingface/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/\
c8b5954a88c2775c546b92593eda40ea041d3176/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
```

Oracle: `reference/llama.cpp/build/bin/`, pin `6fe749801` (has qwen4exp).
xwen: repo HEAD `76e678f` — see "Provenance caveat" at the end, the working tree was
being edited by another agent throughout.

All artifacts (prompts, id files, both sides' dumps, every log) live in
`/private/tmp/claude-501/-Users-orvar-develop-private-xwen/f0c6e8f5-1b14-49db-be40-8be7264e62c1/scratchpad/u7/`.

Nothing was staged, committed, or edited under `src/` by this agent.

---

## Bottom line

**xwen's flash-next graph agrees with the llama.cpp oracle on this file.** Three
independent instruments say so:

1. byte-identical tokenization over a 4218-token corpus,
2. **189/192 forced-replay greedy agreement, 0 hard mismatches** (all three
   divergences are near-ties far inside the repo's excusal band),
3. top-1 and top-5 exact agreement on full-vocab-normalized logprobs against the
   oracle's own `n_probs`.

**The parity HARNESS, on the other hand, cannot run on qwen4exp at all.** Every tier
is blocked by a single panic in the reference expert runner. That is the P4 item.

Two performance findings fall out: prefill is **3.5x slower** than llama.cpp
(reproducible, not warmup), and xwen dirties **15 GB** of memory where llama.cpp
dirties 0.75 GB.

---

## Step 1 — `scripts/parity-gate.ts` on this file: BLOCKED, all four tiers

| tier | ran? | result |
|---|---|---|
| decode | started, failed | reference-side dump panicked |
| ppl (`--regen-ppl-ref`) | started, failed | same panic, same site |
| strict / mm | not run | same code path — every tier's reference side is `--moe-impl reference` |

Exact commands and exit codes:

```bash
bun scripts/parity-gate.ts --model <file> --tiers decode --fixtures code-short   # exit 2
bun scripts/parity-gate.ts --model <file> --tiers ppl --regen-ppl-ref            # exit 2
```

Both die generating the Reference dump:

```
xwen: weights 76.9GB + KV 0.1GB + state 0.1GB = 77.1GB resident (KV grows to 0.1GB at max_ctx 4096)

thread 'main' panicked at src/moe.rs:198:21:
index out of bounds: the len is 512 but the index is 1073971200
```

`src/moe.rs:198` is the per-expert bucketing loop inside `ReferenceExperts::forward`
(`rows[e].push(t as u32)`, with `e = ids_v[t * top_k + k] as usize`). `len 512` is the
routed-expert count for this checkpoint (`n_expert 512`, `n_expert_used 10`,
`src/config.rs:1042-1043`).

### What `1073971200` is

`1073971200 = 0x40038000`, which is the **IEEE-754 f32 bit pattern of 2.0547**. An
expert id cannot be 2.0547 and cannot be a billion. So the u32-typed `ids` tensor is
handing back **float routing data — a router logit or a routing weight — read as u32
bits**, not selected indices. That is the shape of the bug: an f32 buffer reaching a
`to_vec1::<u32>()` read, rather than an off-by-one or an out-of-range selection.

### Isolation done

I re-ran the reference runner **with the fused router kernel active** (no
`XWEN_MOE_GLUE_CLASSIC=1`, which the gate's reference env sets):

```bash
./target/release/logits-dump --model <file> --moe-impl reference \
  --tokens 760,6511,314,9338,369,11751,11,321,279,6511,314,6124,369 \
  --output <dir>/refA-nofusedglue-off.json
# exit 101, byte-identical panic
```

Same panic, **same value**. So it is NOT the router branch: `MoeBlock::route()`
produces the same unusable ids through the fused `ops::moe_router` kernel and through
the candle `route_from_logits` chain (`softmax` → `arg_sort_last_dim` →
`narrow` → `contiguous`). Two independent selection paths yielding the same f32 bit
pattern points at something shared and downstream of the branch — how the `ids` tensor
is materialized/typed for this 512-expert, top-10 geometry — not at either kernel.
`moe_router_supported(512, 10)` is true (`MOE_ROUTER_MAX_EXPERTS = 512`,
`MOE_ROUTER_MAX_TOP_K = 32`), so the fused kernel is in bounds and was genuinely used.

The **fused** runner is unaffected — every measurement below ran on it.

### What the harness needs for qwen4exp (P4 ledger)

1. **`ReferenceExperts` must work on a 512-expert / top-10 / IQ4_NL geometry.** This
   one fix unblocks all four tiers; nothing else in the harness objected to this file.
2. `observed_delta_path()` in `src/bin/logits-dump.rs` hard-bails when no gated-DeltaNet
   layer forward ran (`(0,0) => bail`). It did NOT bite here — qwen4exp does run
   DeltaNet layers — but it is a latent gate on any layer-kind change.
3. No `tests/fixtures/reference-ppl-Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.json`
   exists. Same gap the 3.8-27B has (parity.md already ledgers that one).
4. **Split GGUFs are not a problem.** `gguf::open` → `open_split` handled the 4-shard
   set transparently, as did llama.cpp. The gate namespaced its dir as
   `/tmp/xwen-parity-Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004` — the basename
   carries the shard suffix, which is cosmetic but ugly, and would collide with nothing.
5. The gate's floors are global constants calibrated on the ggml-org Q4_K_M mix. This
   file is unsloth **UD-Q4_K_XL**, a different quant mix, so per docs/parity.md
   ("Floors are a property of the CHECKPOINT'S QUANT MIX") the floors would need
   re-deriving for this checkpoint even once the panic is fixed.

---

## Step 2 — greedy agreement vs llama.cpp

### Method (and why not `llama-cli`)

`llama-cli` has no raw mode: docs/parity.md §"Raw greedy oracle" already records that
`-st -no-cnv` applies the chat template, and `--help` confirms there is no
`--no-chat-template`. (`llama-completion -no-cnv` exists and is llama-cli's
raw-text twin, but it still owns the tokenizer.) I used the blessed oracle instead —
`llama-server /completion` with a **token-id array**, which bypasses template AND
tokenizer, so both engines provably see the identical input sequence:

```bash
# ids from llama.cpp's own tokenizer, ground truth for both sides
reference/llama.cpp/build/bin/llama-tokenize -m <file> -f promptN.txt --ids

reference/llama.cpp/build/bin/llama-server -m <file> -ngl 999 -c 4096 \
  --host 127.0.0.1 --port 8099
curl -s localhost:8099/completion -H 'Content-Type: application/json' -d \
  '{"prompt":[<ids>],"n_predict":64,"temperature":0,"top_k":1,
    "cache_prompt":false,"return_tokens":true}'

# xwen free-run greedy on the identical ids
./target/release/logits-dump --model <file> --moe-impl fused \
  --tokens <ids> --greedy 64 --output pN.xwen-greedy.json

# xwen teacher-forced along llama.cpp's exact trajectory
./target/release/logits-dump --model <file> --moe-impl fused \
  --replay pN.llama-greedy.json --output pN.xwen-replay.json
```

`pN.llama-greedy.json` is a synthesized `kind:"greedy"` dump carrying llama.cpp's
prompt ids and its 64 emitted tokens, so `--replay` teacher-forces xwen along the
**oracle's** sequence and records xwen's own argmax at each step BEFORE forcing. This
is the decode tier's exact methodology with llama.cpp standing in for the (blocked)
xwen reference runner — and it is strictly better evidence than free-run comparison,
which becomes meaningless the moment the two sequences diverge.

`top_k: 1` was forced per parity.md's warning that `temperature: 0` has historically
still dist-sampled on some builds.

### Prompts

| id | file | tokens | what it is |
|---|---|---|---|
| p1 | `p1-short.txt` | 13 | short factual ("The capital of France is Paris, and the capital of Japan is") |
| p2 | `p2-code.txt` | 290 | Python LRU cache, cut mid-signature at `def put(...)` |
| p3 | `p3-mixed.txt` | 530 | mixed technical prose on gated linear attention |

All three written without a trailing newline (parity.md's fixture rule).

### Results

| prompt | prompt tok | free-run prefix agree | forced replay | excused near-ties | HARD mismatch |
|---|---|---|---|---|---|
| p1 | 13 | 19/64 | **62/64** | 2 | **0** |
| p2 | 290 | **64/64** | **64/64** | 0 | **0** |
| p3 | 530 | 39/64 | **63/64** | 1 | **0** |

Total: **189/192 agreeing, 3 excused, 0 hard mismatches.**

Every divergence, with the margin below xwen's own top-1:

| prompt | step | xwen picked | llama.cpp picked | margin | llama's rank in xwen top-5 |
|---|---|---|---|---|---|
| p1 | 19 | 760 `"The"` | 1919 `"This"` | 0.2876 logit | 2 |
| p1 | 42 | 93868 `"**,"` | 159034 `"**."` | 0.0348 logit | 2 |
| p3 | 39 | 421 `" that"` | 15705 `" suite"` | 0.0097 logit | 2 |

All three are genuine near-ties, far inside the repo's `NEAR_TIE_MARGIN_Q8 = 1.0` band
(the widened band Qwen candidates fire, since `attn_decode == "q8"`), and in every case
llama.cpp's pick was xwen's rank-2. p3 step 39 at 0.0097 logit is a coin flip.

`nonfinite = 0` at every step. Per-step logit L2 ranged 490.8–1878.9 across the three
prompts — no scale anomaly.

Candidate provenance on every run:
`schema_version 8, moe_impl fused, mm_variant tensor, no_mm_id false, mm_min_seq 32,
attn_dtype f16, attn_mm tensor, attn_decode q8, combine fused, attn_glue fused,
sdpa f16, flash fused, act fused, delta fused, dense_mm fused, mv_ext fused` — i.e. the
shipped defaults, and `attn_decode q8` as the gate's `--expect-attn-decode` default
requires.

### Tokenizer parity — stronger than asked

llama.cpp's tokenization of the WHOLE 4218-token ppl corpus is **byte-identical** to
xwen's embedded 3.6 `reference/tokenizer.json`:

```
llama.cpp corpus tokens: 4218
xwen corpus tokens:      4218
first differing index:   -1
```

So flash-next reuses the Qwen 3.6 vocab and the CLAUDE.md concern about xwen shipping
the embedded 3.6 tokenizer does not apply to this file. (Spot-confirmed independently:
the oracle's ids for p1 map onto exactly the expected `reference/tokenizer.json`
strings — `"The" " capital" " of" " France" …`.)

### Distribution check — substitutes for the blocked strict/mm tiers

xwen full-vocab logits (from `logits-dump` with no `--greedy`/`--replay`, so the real
full 248320-wide last-position vector), log-softmaxed in f64 with a stable logsumexp,
against llama.cpp's `n_probs: 20` logprobs at the same position:

| prompt | top-1 | top-5 | max abs Δ logprob over llama's top-20 | mean abs Δ |
|---|---|---|---|---|
| p2 (last of 290) | **match** (198) | **5/5, same order** | 4.943e-1 | 1.412e-1 |
| p3 (last of 530) | **match** (271) | **5/5, same order** | 1.883e-1 | 8.103e-2 |

Ordering of llama's top-20 under xwen's logits first differs at rank 5 (p2) and rank 8
(p3) — i.e. only among tail entries at logprob −12 to −14, probabilities ~1e-6.

Sample rows (llama / xwen):

```
p2:  198 "\n"    -0.0005 / -0.0005     271 "\n\n"  -7.8049 / -7.6770
     695 " \n"  -10.8226 / -10.9019   1726 "        \n" -11.7928 / -11.8703
p3:  271 "\n\n"  -0.3938 / -0.4110     561 " The"  -2.4673 / -2.4432
    2844 " That" -3.1152 / -3.1300    1368 " If"   -3.3698 / -3.2942
```

This is the closest available stand-in for the strict/mm cosine tiers, which cannot run
until the reference runner is fixed.

---

## Step 3 — perplexity

**Not a parity number. The two protocols differ and the delta is dominated by that.**

| side | command | value | protocol |
|---|---|---|---|
| llama.cpp | `llama-perplexity -m <file> -f tests/fixtures/ppl-corpus.txt -c 512 --chunks 8 -ngl 999` | PPL **1.7182 ± 0.06321** → **0.5413 nats** | 8 INDEPENDENT 512-token windows, KV reset each; scores only positions 256–511 of each window (2048 scored) |
| xwen | `./target/release/logits-dump --model <file> --moe-impl fused --max-ctx 5120 --ppl tests/fixtures/ppl-corpus.txt --output xwen-ppl-fused.json` | mean_nll **0.369724**, 4217 scored, **0 nonfinite** | ONE continuous 4218-token context, KV never reset, every position scored |

**Δnll = 0.1716 nats** (xwen lower).

The delta is protocol, not fidelity, and the direction is what the protocols predict:
xwen scores with up to 4218 tokens of context where llama.cpp scores with at most 511
and resets every window. xwen's own per-chunk means show exactly that curve:

```
0.5262, 0.7622, 0.5230, 0.1695, 0.2397, 0.2903, 0.1945, 0.2642, 0.3201
```

llama.cpp's running per-chunk PPL for comparison:
`[1]1.8107 [2]2.8701 [3]2.3041 [4]1.9793 [5]1.8504 [6]1.7768 [7]1.7222 [8]1.7182`.

Corpus: `tests/fixtures/ppl-corpus.txt`, the committed WikiText-2 raw test-split head,
4218 tokens, identically tokenized by both engines (see above), so the scored token
streams are the same stream.

The real ppl TIER (xwen-reference vs xwen-fused on the identical corpus, which is what
`PPL_NLL_DELTA_MAX = 0.002` grades) is blocked by the same `ReferenceExperts` panic.
The llama-perplexity number above is therefore an **absolute sanity check only**.

### Flag for the ledger: this corpus may be contaminated for this checkpoint

0.37 nats is PPL **1.45** on WikiText-2 test. The 3.6 pair scores **1.69 nats** on this
identical corpus under the identical xwen protocol (docs/parity.md "Perplexity tier"),
so this is a 4.6x drop in NLL. llama.cpp independently reports PPL 1.72 at 256-token
context, so **both engines agree the model is this good on this text** — it is the
model, not an xwen bug. But a PPL of 1.45 on held-out prose reads like test-split
memorization, which makes this corpus a weak discriminator for this checkpoint. Worth
choosing a fresh held-out corpus for flash-next rather than reusing the frozen one, and
re-deriving `PPL_NLL_DELTA_MAX` against it.

---

## Throughput

| metric | xwen (`--no-draft --temp 0 --raw`) | llama.cpp |
|---|---|---|
| decode | **37.7 – 38.1 tok/s** | **40.9 – 41.5 tok/s** |
| prefill @ 530 tokens | **203.5 / 203.7 tok/s** | **713.4 tok/s** |
| prefill @ 290 tokens | not measured | 441.6 tok/s |
| prefill @ 13 tokens | not measured | 32.6 tok/s (warmup — ignore) |

xwen commands:

```bash
./target/release/xwen generate --model-size flash-next --no-draft --temp 0 --raw \
  --stats --max-tokens 400  -p "$(cat p3-mixed.txt)"
#   prefill: 530 tokens in 2.60s (203.7 tok/s)   decode: 400 tokens in 10.50s (38.1 tok/s)
./target/release/xwen generate --model-size flash-next --no-draft --temp 0 --raw \
  --stats --max-tokens 1500 -p "$(cat p3-mixed.txt)"
#   prefill: 530 tokens in 2.60s (203.5 tok/s)   decode: 926 tokens in 24.58s (37.7 tok/s)
```

llama.cpp figures are the `timings` block of the three `/completion` responses.

**Caveats.** `lowpowermode 0`; high-power mode is not positively confirmable on this
machine and is not claimed. llama.cpp thermal-boosts harder than xwen (parity.md:
−17% vs −5% settling), so its numbers flatter it slightly. The machine was shared with
other agents' cargo builds throughout, which depresses both arms; the decode ratio is
the trustworthy part, the absolutes less so. llama.cpp's 13-token prefill reading is
first-request warmup and must not be read as a rate.

**Decode is within ~8%** — unremarkable and roughly where the other checkpoints sit.

**Prefill is 3.5x slower on xwen and that is a real finding.** 2.60 s reproduced to the
centisecond across two independent runs, so it is not first-forward Metal pipeline
compilation. Worth its own P4 item; the dense-FFN prefill gemm was exactly this shape
of problem on the 27B (P8c) and took a vendored kernel to close.

---

## Step 4 — memory (footprint, mid-decode, NOT RSS)

Sampled at t=30 s into the 1500-token run, via `footprint <pid>`:

```
xwen [46921]: 64-bit    Footprint: 15 GB (16384 bytes per page)
    0 B      64 GB          0 B      12    mapped file
   15 GB     64 GB       24 MB    5002    TOTAL
    phys_footprint: 15 GB
    phys_footprint_peak: 17 GB
```

llama-server on the identical file, for contrast:

```
llama-server [42414]: 64-bit    Footprint: 751 MB (16384 bytes per page)
    0 B      76 GB          0 B      18    mapped file
  751 MB     76 GB       66 MB    4797    TOTAL
    phys_footprint: 751 MB
    phys_footprint_peak: 896 MB
```

xwen's own startup banner says "weights 76.9GB + KV 0.1GB + state 0.1GB = 77.1GB
resident" — that is the weight accounting, not the footprint, and the CLAUDE.md rule
about anonymous RSS lying under mmap cuts both ways here.

**The finding: xwen dirties ~15 GB where llama.cpp dirties 0.75 GB.** Both map the
weights (64 GB vs 76 GB clean, file-backed), but ~15 GB of xwen's weights are
materialized into private memory rather than aliased from the mapping. 64 + 15 = 79 GB
against llama.cpp's 76 GB + 0.75 GB. Both fit in 128 GiB, but 15 GB of real pressure is
worth understanding — especially with the one-large-process-at-a-time rule already
binding.

---

## Provenance caveat — read this before trusting the xwen numbers

The working tree changed under me mid-session. Another agent held modifications to
`src/{attention,gguf,hub,moe,serve/engine,serve/mod}.rs` and `src/qwen4exp/{indexer,
iq4nl,ple,stack}.rs` — 746 insertions across 9-10 files, none of it mine, nothing
staged.

`target/release/logits-dump` was rebuilt at **18:35** and `xwen` at **18:36**, both by
`parity-gate.ts`'s own build step at the start of my second gate invocation. Every
measurement artifact above is timestamped **18:25–18:34**, so all of them ran on the
EARLIER binary.

To check that this does not matter, I re-ran the p1 and p3 replays on the **new**
(18:35) binary:

```
p1: run1 vs run2 argmax identical at 64/64; run2 agreement vs llama = 62/64
p3: run1 vs run2 argmax identical at 64/64; run2 agreement vs llama = 63/64
```

Bit-stable across both tree states. The conclusions hold. The `src/moe.rs`
modification is a doc-comment only (one line, at line 55) and does not shift the
panic's line 198.

---

## Artifact index

Everything under
`/private/tmp/claude-501/-Users-orvar-develop-private-xwen/f0c6e8f5-1b14-49db-be40-8be7264e62c1/scratchpad/u7/`:

| file(s) | what |
|---|---|
| `p1-short.txt`, `p2-code.txt`, `p3-mixed.txt` | the three prompts, no trailing newline |
| `pN.ids.csv`, `pN.llama-tok.txt` | llama.cpp's ground-truth token ids |
| `pN.llama-completion.json`, `.tokens.csv` | oracle 64-token greedy output + timings |
| `pN.llama-greedy.json` | synthesized `kind:"greedy"` dumps driving `--replay` |
| `pN.xwen-greedy.json` | xwen free-run greedy |
| `pN.xwen-replay.json`, `pN.xwen-replay-v2.json` | forced replay, both binaries |
| `pN.xwen-full.json` | full 248320-wide logit dumps |
| `pN.llama-probs.json` | oracle `n_probs:20` logprobs at the same position |
| `xwen-ppl-fused.json`, `xwen-ppl.log` | xwen perplexity pass |
| `llama-ppl.log` | `llama-perplexity` output |
| `corpus.llama-ids.json` | oracle tokenization of the ppl corpus |
| `step1-decode.log`, `step1-ppl.log`, `refA.log` | the three panic reproductions |
| `xwen-gen.log`, `xwen-gen2.log` | throughput runs |
| `llama-server.log`, `llama-server2.log` | oracle server logs |
