# qwen4exp P1 golden fixtures

Golden input/output tensors for the qwen4exp port's new components, computed by
the HuggingFace transformers reference implementation
(`transformers.models.qwen4_exp`, the one executable ground truth per
docs/qwen4exp-port.md D5). The Rust CPU reference implementations are tested
against these files.

Provenance: transformers git main @ `598d8ba8baaec7fec5a22da0e2844c7bf4ea20e1`
(version 5.16.0.dev0), torch 2.13.0 CPU, generated 2026-08-26 by
`scripts/qwen4exp-fixtures/generate.py` (venv recipe in the README next to it).
The installed `modular_qwen4_exp.py` at that commit is what every "modular
line N" reference below points into.

Encoding: JSON; every float is the shortest-f64 repr of an exact f32 value
(round-trip verified at generation). Parse as f64, cast to f32 — bit-exact.
The PLE hash multipliers exceed 2^53 and are stored as decimal strings.

## Tiny config (structurally faithful, all files share it)

hidden 32, hc_count 4, hc_lowrank 8 (streams 4×32 = 128 wide); rms eps 1e-6;
vocab 64, eos 7 (analog of 248044); PLE: ngram_size 3 (orders {2,3}),
heads_per_ngram 2 → 4 heads × dim 8, prime base 97 → head vocabs
[97,101,103,107], table padded to [512,8], hash seed 1234, conv k=4 dilation 3
(state 9); QSA: 2 q-heads / 1 k-head × dim 8, budget 8, ratio 4 → block_topk 2,
rope theta 1e4 partial over first 4 of 8 dims (analog of 64 of 128).

## Norm-weight convention

HF's `Qwen4ExpTextRMSNorm` stores zero-centered weights and multiplies by
`(1 + w)`; the GGUF converter bakes the +1 in. Fixtures carry BOTH:
`*_weight_hf` (the HF parameter) and `*_weight_mult` (= 1 + hf, what a
GGUF-path implementation multiplies by directly). `Qwen4ExpTextRMSNormGated`'s
weight was never zero-centered — plain multiply, single `norm_weight` field.

## Files

- `gated_residual.json` — hyper-connection `GatedResidual`: grouped-norm read,
  `silu(down(n)/hc_count)` → `sigmoid(up(·))` mix weights, mean-over-streams
  (mean, not sum), `2·sigmoid(inject(n)/hc_count)` injection weights, and the
  decoder-layer write-back `stream + block_out ⊗ inject` onto the RAW un-normed
  stream (identity block; modular lines 825-826). Plus the tail
  `hyper_connection_mixer` (use_combine=False, no inject head, separate
  weights).
- `ple.json` — n-gram hash standalone (no-cache history padded `[eos,eos]`,
  eos mid-sequence exercising shift-right-ignore-eos; shift1/shift2 of the
  history and the final table row indices), the full PLE layer forward with
  all intermediates (embeddings, raw dot gate, signed-sqrt gate, gated value,
  its grouped norm, silu'd dilated-conv output, final output), and a scalar
  `gate_function_probe` pinning `sigmoid(sign(s)·√max(|s|,1e-6))` including
  the sub-1e-6 clamp region. NOTE: HF DOES clamp at this commit (modular line
  770) — docs/qwen4exp-port.md divergence item 3 ("HF doesn't") is stale.
- `qsa_indexer.json` — indexer weights, raw cached keys, per-query selected
  token index sets. `case_below_budget` (seq 8 = budget): selection asserted
  equal to dense causal. `case_above_budget` (seq 16): query 12 has a 1-token
  tail → 9 tokens selected (budget+1, NOT budget+ratio-1); query 14 has the
  full ratio-1 tail → 11; query 15 has NO tail → exactly 8, and its own token
  is NOT selected (its block lost the top-k — HF allows masking the query's
  own position when the tail is empty). Per-query block scores and the minimum
  top-k margin (0.47) are included; selection is discrete and must match
  exactly.
- `gated_norm.json` — `Qwen4ExpTextRMSNormGated` (GDN output norm z-gate):
  fp32 norm, plain weight multiply, `× act(z)`; sigmoid arm (what qwen4exp
  constructs) and silu arm (the ZGate enum's other case).
- `grouped_rmsnorm.json` — `Qwen4ExpTextRMSNorm` with group_size 4 over dim
  16, input with 4-decade per-group scale spread; ungrouped-norm contrast
  output included (max elementwise rel divergence ~1e2).

## The QSA tail question (docs/qwen4exp-port.md, divergence item 1) — ANSWERED

HF selects whole blocks + the raw tail, NOT a fixed budget+ratio-1 token count.
Modular lines 451-457: `topk(min(block_topk, num_complete_blocks))` blocks are
flattened to tokens and the tail `local_visible_indices[num_complete_blocks *
ratio:]` (length `visible mod ratio`, possibly 0) is concatenated; the
fixed-width `budget + ratio - 1` buffer (line 417) is only capacity — unused
slots stay -1 and are scattered to a dropped column (lines 465-467). A short
tail does NOT admit tokens of the next-ranked block. llama.cpp PR #27742
(always filling `top_k + ratio - 1` token slots) therefore diverges from HF
whenever `visible mod ratio != ratio - 1` and blocks exceed budget;
`case_above_budget` queries 12 (count 9) and 15 (count 8) pin HF's behavior.

## Tolerances

Floats: elementwise abs, suggested `max(1e-6, 10 × floor)` with the per-fixture
f64-vs-f32 floors below (a bit-faithful f32 reimplementation should land well
inside). Integers (hash rows, shifts, selected index sets) must match exactly
(index sets order-insensitive).

| fixture | tensor | f64 floor | suggested tol |
|---|---|---|---|
| gated_residual | mixed_output | 4.8e-8 | 1e-6 |
| gated_residual | injection_weights | 1.2e-7 | 1.2e-6 |
| gated_residual | stream_out | 2.0e-7 | 2.0e-6 |
| gated_residual | tail mixed | 8.1e-8 | 1e-6 |
| ple | output | 4.8e-7 | 4.8e-6 |
| ple | gate_signed_sqrt | 1.5e-6 | 1.5e-5 |
| ple | conv_out_silu | 3.0e-7 | 3.0e-6 |
| gated_norm | sigmoid / silu | 1.1e-7 / 7.1e-8 | 1e-6 |
| grouped_rmsnorm | output | 1.6e-7 | 1.6e-6 |
| qsa_indexer | raw_keys, scores | — | max(1e-6, 1e-5 rel) |

The PLE gate floor is the largest because √|s| amplifies noise for small |s|.
