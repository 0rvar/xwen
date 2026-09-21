# Qwen-Image 2.1 text-to-image: encoder, transformer, VAE and the CLI

The record of the arc that took Qwen-Image 2.1 from a research doc to a graded render
from `xwen image`, 2026-09-21, commits ca71309 to 5ea0d43. The architecture is in
[docs/qwen-image.md](../qwen-image.md), the decisions in
[docs/decisions/qwen-image.md](../decisions/qwen-image.md), and the plan it executed in
[docs/qwen-image-2.1-plan.md](../qwen-image-2.1-plan.md). This file holds the protocol,
the tables, the review rounds, where the plan was wrong, and what was not taken.

## What shipped

| commit | what |
| --- | --- |
| ca71309 | the Qwen3-VL-8B text tower opens through the dense Qwen3 loader; the pre-norm tap; MRoPE tables; two registry entries; the raw-template renderer; `encode-text` |
| 7b69170 | the VAE, decoder and encoder, f32 on candle's conv path |
| 19e5c55 | the transformer and the dynamic-shift scheduler |
| 4dcca19 | the pipeline, `scripts/qwen-image-ref-dump.py`, the parity gate and its fixture |
| 9c1186d | the encoder gate's bars, set from the reference's own spread |
| 5ea0d43 | `xwen image --model qwen-image-2.1`, the RGBA rule, the decode drains, measured admission |

Plan phases 0, 1 and 2 are done. Phase 3 is half done: the VAE runs on the candle arm and
passes its gate, and the direct-conv arm is not built. Phase 4's memory half is done for
1 MP and its serve half is not. Phases 5 (editing) and 6 (the GUI) are untouched, though
the layout, the rope walk, the MRoPE tables, the VAE encoder and the prompt renderer are
shaped for Phase 5.

0585bcc, the same day, removed `--model-size` in favour of one `--model` flag. It is its
own arc with its own log entry and decision paragraph (decisions/serving.md), and it is
mentioned here only because the commands below use the new flag.

## Protocol

The work ran as parallel units, each briefed with the design decided, each ported from
the reference source with the plan as a trap checklist, each reviewed by two outside
models (Codex and DeepSeek) before it was committed, and each finding validated against
the code before it was acted on. The weights took about an hour and a half to download,
so the encoder, the VAE, the transformer and the scheduler were all written and
unit-tested before a single shard was open. Those tests are CPU tests on tiny random
models: the segmented attention against the dense mask, a cached step against a full
forward, each VAE shortcut stage against a literal transcription of the reference's
reshape arithmetic, the loader's name set against the shipped index and header.

Then the dumps ran, all five stages clean on the first real run, and the two gates. The
transformer gate passed on its first run against real weights. The encoder gate failed on
its bar, not its graph, and "The encoder gate" below is what was done about it.

Reference environment: torch 2.14.0, transformers 5.17.0, diffusers 0.41.0.dev0 at
`6256aa7`, snapshot `790c9263`. Every weight stage of the script asserts that its own
path equals the pipeline's method bit for bit on one prompt (`pipeline_check`:
`bitwise_equal` true, `drop_idx` 14), so the reference IS the pipeline and not a reading
of it.

## The encoder gate

12 prompts, the Z-Image Stage-2 set, against the fp32 CPU eager arm. String, ids and drop
index first: xwen's renderer reproduces the HF processor on all 12, with the one
deliberate exception of the prompt that spells a literal special token.

The first run, on the bar inherited from Z-Image (cosine 0.9999 and relative error 0.01
on every row): minimum cosine 0.99997 to 0.99998 on every prompt, the normed wrong graph
at minimum cosine 0.62 with all 73 of its rows outside, and 12 of 12 prompts FAILED, 1 to
4 rows each, always starting at kept row 0 with relative error 0.025397, the same figure
in every prompt. Every row that missed:

| prompt | kept row | token | norm / median | xwen cosine | xwen rel | torch bf16 cosine | torch bf16 rel |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all 12 | 0 | 151644 `<\|im_start\|>` | 0.84 to 1.05 | 0.99998306 | 0.02540 | 0.95385 | 1.57741 |
| 1 | 34 | 315 " of" | 1.45 | 0.99997219 | 0.01175 | 0.99814 | 0.09982 |
| 7 | 18 | 151667 `<think>` | 0.59 | 0.99999244 | 0.01874 | 0.99978 | 0.05239 |
| 10 | 341 | 279 " the" | 0.83 | 0.99997640 | 0.01071 | 0.99287 | 0.15873 |
| 11 | 209 | 279 | 0.92 | 0.99999177 | 0.01008 | 0.99966 | 0.03040 |
| 11 | 501 | 279 | 0.77 | 0.99999611 | 0.01329 | 0.99974 | 0.02936 |
| 11 | 552 | 279 | 0.76 | 0.99999142 | 0.01296 | 0.99952 | 0.06048 |

Every one passes cosine and misses only `rel`, the largest absolute difference over the
row's own largest magnitude, so a single channel decides it.

Kept row 0 is the user turn's `<|im_start|>` under the same context in every prompt,
which is why its figures repeat. Its OUTPUT norm is ordinary. Inside the stack it is not,
measured by a one-off torch bf16 probe on prompt 0 (residual norm per `hidden_states`
index; the probe is not in the script):

| index | position 0 (dropped) | kept row 0 | median of the rest |
| --- | --- | --- | --- |
| 8 | 13514 | 38.8 | 45 |
| 16 | 13514 | 63.6 | 76 |
| 24 | 13890 | 9391 | 172 |
| 32 | 13890 | 9391 | 530 |
| 34 | 13828 | 9390 | 688 |
| 35 | 9589 | 873 | 868 |

The last layers cancel a 9344-magnitude channel back to about 850, so the row is a small
difference of large numbers: the sibling gate's position-0 phenomenon, worse conditioned.
xwen's error there is 2.6 absolute, 2.7e-4 of the scale the row was computed at.

Over the 1805 rows past row 0:

| | rel p50 | rel p99 | rel max | cosine p50 | cosine min |
| --- | --- | --- | --- | --- | --- |
| xwen | 0.00232 | 0.00651 | 0.01874 | 0.99999825 | 0.99997219 |
| torch bf16 | 0.03406 | 0.15665 | 0.23329 | 0.99985956 | 0.99287018 |

xwen sits about 14 times inside the reference's own bf16 arm. The wrong graph's best row
past row 0 reads cosine 0.959 and rel 0.404, and on row 0 cosine 0.905 and rel 1.000. The
bars that followed are in the decisions file; the gate now reads 6 of 1805 rows over 0.01
(99.67% inside) and passes, with the bracket unchanged. `reference.json` carries the
spread block the bracket test reads, written by `--stage finalize`, which needs no
weights.

Other checks the dump made, from `reference.json`: `hidden_states_last_is_normed_without_
the_hook` true; normed against pre-norm minimum cosine 0.623, mean 0.813, rms ratio
0.265; text position ids equal on all three MRoPE axes and an `arange`, over the 87
tokens of prompt 1.

Load observation from the gate's log: 14.1 GB of weights plus 0.6 GB of KV at
`max_ctx` 4096, on the device in 1.3 s; 22,523 of 7,568,097,280 BF16 projection values
sit below f16's subnormal floor and none above its maximum.

## The transformer and VAE gate

`512x512-p1-s0`: prompt 1 ("portrait-golden-hour"), 73 kept caption rows, seed 0, 40
steps, `use_kv_cache` on, against diffusers fp32 on mps. The sigma grid agrees to
1.11e-16 and `mu` is 0.538710 on both sides.

| arm | cosine | mean rel | max rel | bar |
| --- | --- | --- | --- | --- |
| xwen against the fp32 reference | 1.000000 | 0.0006 | 0.0024 | inside |
| the reference's bf16 arm against its fp32 arm | 0.999948 | 0.0075 | 0.0330 | inside |
| bracket: fully bidirectional attention | 0.982474 | 0.1428 | 0.2650 | outside |
| bracket: text modulated from the real `t` | 0.987354 | 0.1225 | 0.2149 | outside |

| after 40 steps | xwen | the reference's bf16 arm |
| --- | --- | --- |
| final latent, cosine / mean rel | 0.999999 / 0.0008 | 0.999358 / 0.0192 |
| image PSNR against the fp32 PNG | 57.30 dB | 35.58 dB |
| VAE alone, the reference latent through this decoder | 91.07 dB, bar 60 | |

The largest projection weight is 3.4531, in `txt_in.out_layer.weight`, over
7,115,112,448 values, inside f16's range. The two images were looked at side by side and
are the same picture. The figures did not move after the RGB-plane change or after the
decode drains.

## The alpha plane

The diffusers fp32 reference for an ORDINARY prompt, 512x512, 262,144 pixels:

| alpha (u8) | pixels |
| --- | --- |
| 255 | 217,245 |
| 254 | 44,891 |
| 253 | 7 |
| 252 | 1 |

With the model card's transparency prompt ("This is an RGBA image with transparency. A
cute cartoon dragon sticker. ..."), 512x512, the two sides starting from different noise
so the pictures differ:

| alpha | diffusers bf16 on mps | xwen |
| --- | --- | --- |
| 0 | 1,619 | 3,895 |
| 1 to 8 | 125,502 (120,507 at exactly 1) | 159,807 (158,908 at exactly 1) |
| 9 to 127 | 3,911 | 1,366 |
| 128 to 247 | 27,154 | 2,597 |
| 248 to 255 | 103,958 | 94,479 |

Both are clean stickers on a transparent background, and xwen wrote RGBA with 163,702
clear pixels. Ordinary prompts on xwen read a minimum alpha of 252 at 512x512, 250 at
1024x1024 and 230 at 8 steps, with no clear pixel, and were written RGB.

## Memory

The kernel's `phys_footprint_peak`, sampled once a second from outside the process, one
render each:

| size | step phase | decode, no drains | decode, drained |
| --- | --- | --- | --- |
| 512x512 | 18 to 19 GiB | 29 GiB | 25 GiB |
| 1024x1024 | 27 GiB | 71 GiB (72.7 GB) | 55 to 56 GiB |

The encoder is gone before the transformer loads: the run prints the device at 0.0 GB
after the drop and 15.7 before, and the transformer plus VAE then hold 15.7 GB. The PNGs
after the first round of drains are byte-identical to the ones before at both sizes; the
final, finer placement was re-checked byte for byte at 512x512 only. In the agent sandbox
`memory telemetry unavailable: Operation not permitted` prints at startup, so the
coordinator's own event lines were not observed.

## Timing observations, which are not figures

One dev-tree release build, NOT a pinned binary, and `pmset -g` read
`lowpowermode         1` for every run below, so none of this goes in perf-state.md and
none of it tests the plan's estimate, which assumed automatic power mode.

| | 1024x1024, 40 steps | 512x512, 40 steps |
| --- | --- | --- |
| encoder load, warm | 2.3 s | |
| encode, 73 kept tokens | 92 ms | |
| transformer and VAE load | 3.0 s | |
| first step | 5.31 s | 0.70 s |
| later steps | 5.3 to 6.1 s | 0.93 to 1.15 s |
| VAE decode | 18.6 to 21.2 s | 3.6 to 4.7 s |
| total | 240 to 268 s | 53 to 62 s |

The parity gate's own 512x512 run, earlier the same day and also unpinned: steps 0.95 to
1.14 s, decode 3.46 s. The reference dump at 512x512 took 165 s in fp32 and 48 s in bf16
for 40 steps on mps. Decode seconds before and after the drains read 18.61 against 21.18
then 18.85 at 1024x1024, and 3.64 against 4.73 then 4.64 at 512x512, with the steps in the
same runs moving by 10% either way, so the cost of the drains is not separable from noise
and may be more than 5%.

## The review rounds

Two outside models per unit. What each found that was real:

- **VAE.** No math divergence from either. The zero-weight API test could not see a
  flipped posterior half, a missing or reversed denormalisation or a missing clamp; the
  norm constructor's `sqrt(C)` fold was covered only by an ignored test; the smoke test
  printed its correlation and asserted nothing. All fixed with non-zero deterministic
  weights. The dead `time_conv` count was reported as 14 and is 12; the code always said
  12.
- **Transformer and scheduler.** No math divergence from either; one reviewer reproduced
  the 40-step grid bitwise. Both found `max_abs_diff` folding with `f32::max`, which
  swallows NaN, behind all three equivalence tests. Also: the rope bound checked the
  counter after advancing it and refused a layout the reference accepts; the timestep
  test had no independent oracle; the Metal test passed silently without a device; the
  slot-mask contract of `Layout::from_slots` was unstated. All fixed. The Metal test now
  unwraps the device as `src/ops` tests do, `test_support` excluding no-Metal skips by
  its own rule.
- **Encoder.** No regression to any existing checkpoint and no way onto a language
  surface. A `text_encoder/` directory opened by path could not find `processor/
  tokenizer.json` (fixed by identifying provenance before picking the tokenizer); two
  call sites used bare `encode` and could never honour `final_norm` (fixed by
  `encode_spec`); the tokenizer byte-identity test skipped silently (now through
  `test_support`). One reviewer's first pass ran out of context at about 296K tokens and
  was rerun narrower.
- **Pipeline.** Three bare dict keys in the dump script would have raised `NameError` on
  the first step callback (fixed before the transformer stage reached them); `check_size`
  allocated before it bounded (fixed, closed form first); `use_kv_cache` in `meta.json`
  was a literal (now the value passed). The u8 tie rounding is "not taken" below.
- **CLI wiring.** No Z-Image regression. Four refusals fired only after a download or a
  load (`--cap-feats` fetching encoder shards, `XWEN_QWEN3_ATTN` unvalidated, the
  overlong prompt, an oversized injected caption); a flaky `clear_pixels == 0` assertion;
  a dispatch gate that could not tell the two pipelines apart; a serve refusal that said
  the pipeline was not implemented after it was. All fixed.

## Where the plan was wrong

The plan was written without access to the hub. Opening the files and reading the
reference line by line corrected it in nine places, and confirmed four things it could
only infer.

1. The VAE file is F32 and 1,350,989,512 bytes, not bf16 and about 0.25 GB.
2. There are nine shortcut stages and five are non-trivial, not four of eight.
3. With reference images the sequence is not `[text ; references ; target]`: each image
   slot expands four-fold inside the text stream.
4. The reference builds rope angles in f32, and an f64 table drifts from it at large
   positions.
5. The timestep embedder's layers are `linear_1` and `linear_2`, and the bf16 reference
   computes `timestep / 1000` in bf16; the fp32 reference the gate grades against sees
   sigma itself.
6. The pipeline's `latents=` takes the PACKED `[1, N, 64]` form and casts it without
   reshaping.
7. `conv2d_direct` is one entry point with a fusion struct, not a family of fused
   variants.
8. `ZImagePipeline::load_cancellable` takes no memory lease; the CLI and the serve engine
   do.
9. `Format::Diffusion` has one field. The plan caught this one itself, as drift in
   decisions/zimage.md, and a dated correction is appended there.

Confirmed: the tokenizer is the Qwen3 family file, byte-identical to `Qwen/Qwen3-4B`'s,
so there is no third vocabulary; the drop index is 14 on the shipped file and the model
card's neon prompt is 45 tokens with 31 kept; text-only MRoPE is plain NEoX, numerically,
on both sides; `hidden_states[-1]` comes back normed without the hook. And from
`preprocessor_config.json`, for Phase 5: image mean and std are 0.5, and the size bounds
are `shortest_edge` 65536 and `longest_edge` 16,777,216, so a 1024x1024 reference does
not shrink and the slot-count assertion holds. The repo total of about 33.1 GB and the
shard counts were right.

## Not taken now

- **The u8 rounding of exact ties** follows the shared Z-Image helper, half away from
  zero against NumPy's half to even. Reopen if an image PSNR figure is shown to be
  limited by ties.
- **A request field forcing RGBA** for an image that is only semi-transparent. Reopen
  when someone needs one; the rule has one owner, `keeps_alpha`, to add it beside.
- **The massive-activation probe in the dump script.** It needs the weights and was a
  one-off; the gate's module doc cites its figures. Reopen if the row-0 bar is ever
  questioned.
- **A per-row relative bar of 0.01 on every encoder row.** Six rows miss it that torch's
  own bf16 arm misses by 3 to 16 times. The 99% share check keeps the strict claim.
- **A LayerNorm fold kernel** for the block norms. `layer_norm_scaled` already passes
  `1 + scale` as the fused kernel's weight, so there is no full-tensor multiply to
  remove. Reopen if a profile ranks it.
- **A two-base-pointer K/V read** in `flash_t.metal` to stop copying `[prefix ; target]`
  per block per step. It is nothing for text-to-image (tens of text rows) and becomes
  2.15 GB per 1 MP reference with editing. Reopen with Phase 5 and a measured copy cost.
- **Per-text-segment masks are built on the CPU** each forward. Negligible for tens of
  tokens; unsized for long text runs after reference images. Reopen with Phase 5.
- **`--latents`, `--dump` and a snapshot root passed by path** were not run by hand on
  the new CLI path. The parity test exercises the same readers, and a gate covers a
  synthetic root by path.
- **A message cosmetic**: tensor names in `Qwen3Parts::new` errors still say
  `model.layers.N` for a VL set.
