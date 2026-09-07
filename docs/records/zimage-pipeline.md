# The Z-Image-Turbo pipeline

The arc write-up for text-to-image generation in xwen: `xwen image`, the vendored candle
`z_image` module, and the registry entry that names the whole pipeline. One section per
arc, appended as each lands. The architecture and its traps are in
[docs/zimage.md](../zimage.md), the decisions in
[docs/decisions/zimage.md](../decisions/zimage.md), the scope amendment that admitted any
of it in [docs/decisions/scope.md](../decisions/scope.md), and the encoder this pipeline
calls in [docs/records/qwen3-dense.md](qwen3-dense.md).

## Arc A, 2026-09-07: the first image

One commit, 493ae2e, on the `zimage` branch off master 8d2923c. `xwen image --prompt
<text>` renders a PNG through the Z-Image-Turbo pipeline: the repo's own verified
Qwen3-4B encoder at hidden index 35, then the S3-DiT transformer, then a flow-match Euler
loop of eight steps, then the Flux VAE decoder. The transformer, VAE, scheduler and
sampler are candle's `z_image` module at rev 21cca0b (PR #3261), vendored into
`src/zimage/` and corrected; `src/zimage/pipeline.rs` is new.

**It works, judged by eye.** Two prompts at 1024x1024. Seed 0, "A photo of a red bicycle
leaning against a white brick wall, golden hour", produced a photorealistic red city
bicycle with a black saddle, chain guard and rear rack, leaning on white-painted brick
with a long warm shadow and cracked pavement: every element of the prompt present,
geometry right, no artifacts. Seed 42, "A cozy bookshop storefront at night, rain-slick
cobblestones, warm light in the windows, a hand-painted sign that reads OPEN LATE",
produced the shopfront with lit book-filled windows, wet reflective cobbles, and a sign
that legibly reads OPEN LATE. Small book-cover text is gibberish, which is normal for
this model at this size. A seed reproduces exactly across processes: the seed-0 CLI run
and the seed-0 integration test wrote byte-identical PNGs, sha256 `82374724b17e...cb75`.

**Timings, and they are a first reading rather than a bench.** Dev-tree release build,
not a pinned binary, `pmset -g` reporting `lowpowermode 0` with no high-power claim.

| phase | run 1, cold cache | run 2, warm | test run |
| --- | --- | --- | --- |
| text encoder load | 3.3 s | 0.8 s | — |
| encode, 23 / 39 tokens | 166 ms | 6 ms | — |
| transformer + VAE load, fp32 to bf16 cast | 31.9 s | 3.3 s | — |
| transformer step, each of 8 | 5.0-5.4 s | 5.0-5.5 s | 5.0-5.6 s |
| VAE decode | 4.97 s | 4.86 s | 5.11 s |
| total wall | 82.3 s | 51.7 s | 52.5 s |

About 42 s of transformer per image, and no performance work was done at all, by
instruction. Steps creep from 5.0 to 5.5 s over a run, which is probably thermal. The
figures and their conditions are in [docs/perf-state.md](../perf-state.md); the ceiling
they should be read against is roughly 62 TFLOP per step, which at the ~19.9 TFLOP/s a
large fp16 matmul is reported to reach on this chip is about 3.1 s, so a step is running
at something like 60% of a rate that is itself only a secondary report.

### The registry shape, which deviated from the plan on purpose

The arc plan called for a `Format::SafeTensors`-shaped `ZImageTurbo` entry with
`model_index.json` first and the encoder spec hanging off it. That would have broken
invariants the registry already holds: `is_safetensors()` means "a Qwen3 set the Qwen3
loader opens", `identify_cached_dir` iterates exactly those entries and would have
identified the snapshot ROOT as the pipeline where an existing test pins the root as
identifying as nothing, and the safetensors tests assert `config == files[0]` over a
six-file set. So the entry got its own `Format::Diffusion { text_encoder }`, with
`is_gguf()`, `is_diffusion()` and `text_encoder()` beside it and a test that exactly one
of the three format predicates holds per entry. It carried the transformer, VAE and
scheduler config paths too until the review round after this arc, which found nothing
read them: `ZImagePipeline::load` states the diffusers layout itself, and it has to,
being called against an operator's own `--model <root>` that no entry describes. The
paths came out and `load` is the one source of that layout.

Five tests that meant "GGUF" and filtered on
`!is_safetensors()` now say `is_gguf()`. The argument is in
[docs/decisions/zimage.md](../decisions/zimage.md).

The entry lists all fifteen files, the encoder's five among them, so one fetch of it
leaves nothing to download, and `model_index.json` is first so the resolved path's parent
is the snapshot root. `encoder_spec()`, `safetensors_tokenizer()` and
`safetensors_rope_theta()` return None on the pipeline entry: it asks its
`text_encoder()` for those. `auto_fetch`, `supports_drafting` and `draft_default_on` are
all false, and `not_servable_reason()` is "it is a text-to-image diffusion pipeline, not
a language model; use `xwen image`", which is what `generate`, `chat`, `serve` and
`batch` print when pointed at it. The alias moved with the entry: `zimage-turbo` and
`z-image-turbo` now name the pipeline and the encoder is `zimage-turbo-encoder` /
`z-image-turbo-encoder`, updated in every string that named the old one.

### The CLI surface

```
xwen image --prompt <text> [--width 1024] [--height 1024] [--steps 8] [--seed N]
  [-o out.png] [-m <snapshot root>] [--model-size zimage-turbo]
  [--latents file.safetensors]
```

The size rule is `check_size`: both sides positive multiples of 16, the image token
count `(w/16) * (h/16)` a multiple of 32, and neither side past 8192 px, which is the
512 positions each spatial RoPE table holds. 1024x1024, 1024x768, 768x1024, 512x512 and
1536x1024 pass. 1000x1000 is refused for the multiple of 16, 528x528 for "1089 image
tokens ... not a multiple of 32", and 8208x1024 for the table bound. The 8192 px rule and
`forward`'s matching check against the loaded `axes_lens` (which also covers a caption
long enough to push the image past axis 0's 1536) came from the review round after the
arc: candle's Metal `index_select` clamps an out-of-range position instead of failing, so
without them the answer to an oversized request is a wrong image and not an error.
An omitted seed draws a random u64 and prints it; seeds
are xwen's own draw and mean nothing to torch. Per-phase timings go to stderr and one
summary line to stdout. `--latents` injects a latent from a safetensors file, which is
the hook Stage 3 and Stage 4 will use.

The encoder is opened as `Model::ZImageTurboEncoder` at `<root>/text_encoder` through
`CheckpointSource` and `XwenModel::load_encoder(.., 512)`, with the prompt rendered,
tokenized and truncated by `encoder_prompt_ids`, shared with `encode-text` whose
behaviour is unchanged. Encoder and transformer both stay resident for the run.

### Verified this arc

- `cargo test --release`, full suite: lib 1302 passed, 0 failed, 34 ignored; bin 14;
  cli_gates 4; parity 69 with 3 ignored; qwen3_encoder 17 with 1 ignored; qwen3_parity 13
  with 1 ignored. Exit 0. **After the review round** that followed the arc, the same run
  is lib 1308 (six new tests), cli_gates 6 (two new), everything else unchanged, 0 failed
  and no SKIPPED lines; and the ignored image test reproduces the figures below to the
  digit, which is the evidence the fixes touched no math.
- New unit tests, all passing: the 8-step sigma table and the dynamic-shift refusal; the
  five rope reference values at position (5,3,7) and the rope dtype; patchify ordering and
  the unpatchify roundtrip; caption padding length; seeded-noise reproducibility and
  moments; postprocess rounding; the size rule; the pipeline entry's shape; alias
  round-trips for every registry entry.
- The ignored end-to-end test `tests/zimage_image.rs`, run with `--ignored`: PASS in
  52.5 s, on non-degeneracy bars (channel means 203.0 / 180.1 / 159.9, stds 60.3 / 67.8 /
  63.8, mean neighbour absolute difference 6.10 on the middle row), and its PNG is
  byte-identical to the CLI's.

**What is NOT verified is the arithmetic.** No reference dump of the transformer exists
yet, so the block math, the VAE and the rope beyond its five pinned values are graded by
reading and by the images looking right. That is the whole content of Arc B, and until it
runs, "coherent image" is the only claim this arc makes.

### Not taken now

Each of these is a deliberate omission with a reopen condition, and the ones carrying a
number or a waiting user are ledger items in [TODO.md](../../TODO.md) instead.

- **A third `CheckpointSource` arm for diffusion checkpoints.** Every language-model
  consumer routes through that seam and a diffusion consumer should too, but there is
  exactly one of them and a seam shaped by one caller is a guess about the second.
  Reopen when a second consumer needs the transformer or the VAE opened, which the serve
  images route probably is.
- **The image pad-token path.** `x_pad_token` exists so a ragged image grid can pad up to
  a multiple of 32; the code path is refused by `check_size` instead, along with the
  batched attention-mask path beside it. Both are untested. This one is a ledger item,
  because the sizes it unlocks are sizes a user will ask for.
- **Footprint.** `footprint`, `ps` and `vmmap` were all refused with "operation not
  permitted" from the agent sandbox, so nothing was measured. Expected from the loads is
  7.6 GB of encoder (printed by the loader), 12.3 GB of bf16 transformer and 0.34 GB of
  f32 VAE, so about 20 GB plus activations, with the fp32-to-bf16 cast transiently
  holding one f32 tensor. Reopen from a user shell; it is a ledger item because 20 GB is
  a number the serve route has to plan around.
- **F32 activations against bf16 weights.** The encoder does exactly this and it bought
  about 10x accuracy over torch's all-bf16 path, but it doubles activation traffic on the
  heaviest graph in the repo. Reopen when the step-0 parity gap exists: if bf16 clears the
  bar there is nothing to buy, and if it does not, this is the first thing to try.
- **Quantization of any kind.** Refused on evidence rather than deferred on effort, and
  the evidence is that the step is compute-bound by 45x to 160x
  (decisions.md "The transformer runs bf16 end to end"). Reopen on a W8A8 kernel reaching
  the neural accelerators, or on footprint pressure.
- **Ties-to-even rounding in `postprocess_image`.** candle's `.round()` is
  ties-away-from-zero and numpy's `.round()`, which `VaeImageProcessor.numpy_to_pil`
  uses, is ties-to-even. They differ only for a value landing exactly on `.5` after the
  x255, which f32 VAE output effectively never produces, and the existing test's own case
  (127.5 to 128) agrees by luck because 128 is even. Reopen at the Stage 4 bit-exact
  comparison, if one is ever attempted: if the PSNR is being read against a byte-for-byte
  bar rather than a decibel one, this is a source of single-count differences and has to
  go first.
- **Threading `n_kv_heads` through the attention reshape.** `ZImageAttention::new` sizes
  `to_k` and `to_v` with `cfg.n_kv_heads` and `forward` reshapes k and v with
  `self.n_heads`; the shipped config has both at 30, so it is a conflation and not a bug,
  and if they ever differed the reshape would fail loudly rather than compute something
  wrong. diffusers has the same conflation (it uses `attn.heads` for all three) and the
  official repo threads the two separately. Reopen if a Z-Image release ships a config
  where the two differ, which would be a GQA variant of this transformer.
- **The HTTP encode route.**
 `encode-text` exposes the encoder on the CLI and the serve
  route for images will need the encoder in process; an HTTP endpoint that returns raw
  hidden states for an external pipeline is a different feature and nobody asked for it.
  Reopen if something outside this process wants the conditioning.

### Next

Arc B is the reference dump, and it is the prerequisite for everything else: no
performance work should touch this graph while its arithmetic is ungraded, because there
would be nothing to regress against. The step is to extend
`scripts/zimage-ref-dump.py` with a latent-injection stage that writes a fixed
`[1, 16, 128, 128]` fp32 latent plus the step-0 velocity field and the final image from
the official pipeline, then read xwen's own through `xwen image --latents`. Prerequisites:
the official `Tongyi-MAI/Z-Image` repo has an MPS branch in `inference.py`, so the oracle
runs on this machine; diffusers' `latents` argument bypasses the noise draw and the
official `generate()` does not expose it, so either patch it or run the reference through
diffusers. The bars get decided from the dump's own spread, after it exists.
