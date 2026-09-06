# 2026-07-28 (late night) — Parity harness live (P7): both checkpoints match upstream llama.cpp, floors an order of magnitude tighter than laguna's

Moved verbatim from [docs/log.md](../log.md) on 2026-09-06; the log keeps this entry's
opening paragraph and links here.


**Context.** Everything before this rested on cited source lines, hand-computed unit
tests, and one greedy eyeball. The engine had never been compared against another
implementation on the same weights, and the 27B had never been run at all. P7 was the
item standing between "it produces fluent text" and "it computes the right thing".

**Change.** Upstream `ggml-org/llama.cpp` shallow-cloned into `reference/llama.cpp`
and PINNED at `e9fa0781f1c25fc4fe8c86be1edc6970661ad6f0` (recorded in docs/parity.md).
This session materialized it as a plain clone and gitignored the path, reasoning that
a submodule puts the reference tree in the index and the index is the human's review
surface; a `.gitmodules` entry declaring the same path as a submodule was staged
concurrently from elsewhere, and the owner settled it that night in favour of the
submodule — the gitlink makes the oracle sha reviewable in the diff, so moving the pin
becomes a staged change someone approves, which is the property the pin exists to
have. `scripts/build-llamacpp.sh` retargeted at the path.
`tests/fixtures/parity-prompts.json` regenerated with Qwen ids from the oracle's own
`llama-tokenize --no-bos`; the SWA-specific `long-swa` fixture replaced by
`long-mixed`, 612 tokens of prose that stresses the DeltaNet recurrence instead of a
sliding window that does not exist here. `hf.ts` repointed at the two ggml-org repos
with a size selector; `parity-gate.ts` gained `--model-size 27b|35b` and namespaces
every artifact by checkpoint basename (two official files, two architectures, two
sets of floors — they must never share a parity dir or a frozen ppl fixture).
`ref-dump.sh` retargeted. `scripts/parity.ts` learned the tap-name mapping
(`refTapNames`) rather than renaming engine taps. Laguna's `reference-ppl.json`
deleted; the committed-fixture test now validates every per-checkpoint fixture.

**Two latent parser bugs in the inherited `parity.ts`, both silent corrupters.**
First, the node-header regex captured names as `(\S+)`, so headers with spaces
(`cache_r_l0 (reshaped)`, `(view)`) were skipped and their value rows were attributed
to the previous node — which kept that node's `sum` but replaced its sampled row with
an unrelated tensor's. Symptom: `attn_norm-0` reporting `rowRelL2 = 2.29e+6` while its
values were in fact digit-identical to the oracle's. Second, `FLOAT_RE.test(line)` on a
shared `/g` regex advanced `lastIndex`, dropping every other value row. Neither would
ever produce a false PASS, but both produced convincing false divergences — the first
one cost a real detour before it was pinned down.

**Result.** Both checkpoints agree with upstream on identical GGUF weights. The final
four-tier run with the calibrated constants in place is `ALL PASS (6 graded)` on each —
42 s warm for the 35B, 2.0 min for the 27B, since the Reference dumps are cached and
only the candidates regenerate.

Track A (first-divergence bisection, 35B, code-short, 242 taps compared): no cliff.
The sampled-row rel-L2 profile is smooth and *flattens* rather than compounding —
`l_out` runs 1.8e-3 at layer 0, 1.2e-3 at 7, 2.4e-2 at 23, 1.4e-2 at 39 — and the
final-logits sampled cosine is 0.999995. Individual `sumRelErr` spikes up to 1.9e-1
are near-cancelling residual sums, not divergences; their own `rowRelL2` stays in the
neighbourhood's band.

Track B, `bun scripts/parity-gate.ts`, 35B-A3B Q4_K_M:

| tier | fixture | result |
|---|---|---|
| strict | code-short | PASS, cosine 0.999999861, top-1 = ref, top5 5/5 |
| mm | code-short | PASS, cosine 0.999539782, top-1 = ref, top5 5/5 |
| decode | code-short | PASS, 63/64 agree, 1 excused near-tie (0.0040 logit) |
| decode | text-mixed | PASS, 62/64 agree, 2 excused (0.5567, 0.2606) |
| decode | long-mixed | PASS, **64/64 agree**, 0 excused |
| ppl | — | PASS, Δmean_nll 0.000511 (fused 1.694170 vs reference 1.693659 over 4218 tokens) |

**The 27B's first forward was correct.** No bisection was needed: it loaded (18.2 GB
resident), prefilled 58 tokens in 9.6 s cold, and produced a top-5 that tracks the
35B's on the same prompt. Its parity numbers are *better* than the 35B's across the
board — strict bitwise 1.000000000, mm ≥ 0.999993294, decode **64/64 with zero excused
near-ties on all three fixtures**, ppl Δ 0.000221 (fused 1.748093 vs reference
1.747872) — with the caveat that on a dense model the
strict tier is near-vacuous, since `--moe-impl reference` and `fused` run the same
`DenseMlp` and the strict env pins everything else classic on both sides. The 27B's
real signal is the mm/decode tiers, which exercise the f16 attention path and the
fused glue.

**Floors, calibrated across both checkpoints and all three fixtures** (the constants
are global, so they are set under the WORST observed value): `COS_MIN_STRICT = 0.9998`
(worst achieved 0.999894, 35B long-mixed) and `COS_MIN_MM = 0.999` (worst achieved
0.999540, 35B code-short — ~1.7x the observed prompt-to-prompt spread). Both are an
order of magnitude tighter than laguna's 0.9955 / 0.985: these kernels track the
oracle much more closely on this architecture, largely because the Qwen Q4_K_M mix
keeps attention, ssm and shared-expert weights at q8_0.

**Verdict.** The engine is parity-validated. This is the checkpoint that makes every
later kernel change checkable instead of hopeful, and it is the gate P8's DeltaNet
kernels will be graded against. Four things are explicitly NOT proven and are in the
ledger: Track A cannot localize inside a layer (the tap set is still laguna's six —
no DeltaNet core, router logits, or shared-expert gate taps, which would need plumbing
in the model-math files); `provenance.flash` says `"fused"` while `flash.metal` is
compiled at head dim 128 and cannot serve Qwen's 256, so prefill is really candle sdpa
with a materialized mask; the dense strict tier as noted above; and the `_Q8` widened
bands were inherited, not recalibrated — measured worst per-step l2 deviation is
1.0211 against a 1.5 band (far too loose), while the near-tie window genuinely needed
its widened 1.0 (text-mixed step 15 excused at 0.5567).
