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
  digit, which is the evidence the fixes touched no math. **After the second round**
  (below) it is lib 1311 and cli_gates 9, still nothing else moved, and the image test
  still reproduces those figures.
- New unit tests, all passing: the 8-step sigma table and the dynamic-shift refusal; the
  five rope reference values at position (5,3,7) and the rope dtype; patchify ordering and
  the unpatchify roundtrip; caption padding length; seeded-noise reproducibility and
  moments; postprocess rounding; the size rule; the pipeline entry's shape; alias
  round-trips for every registry entry.
- The ignored end-to-end test `tests/zimage_image.rs`, run with `--ignored`: PASS in
  52.5 s, on non-degeneracy bars (channel means 203.0 / 180.1 / 159.9, stds 60.3 / 67.8 /
  63.8, mean neighbour absolute difference 6.10 on the middle row), and its PNG is
  byte-identical to the CLI's.

**What was NOT verified at this point was the arithmetic.** No reference dump of the
transformer existed when Arc A shipped, so the block math, the VAE and the rope beyond its
five pinned values were graded by reading and by the images looking right, and "coherent
image" was the only claim the arc made. Arc B, the same day, closed that: see "Arc B,
2026-09-07: Stage 3 and Stage 4, the transformer against its reference" below.

### The second review round, 2026-09-07

Four findings, and the interesting thing about them is that two were corrections of the
first round rather than of the arc.

**The scheduler attribution was wrong in the fix, not just in the original.** The first
round said the comments credited the official repo with diffusers' grid; the correction it
wrote said the two references genuinely differ by up to 5.0e-3. They do not. The official
pipeline assigns `scheduler.sigma_min = 0.0` on the line before `retrieve_timesteps`, and
with `use_dynamic_shifting: false` and shift 3.0 its interpolation branch then produces
`1.0, 0.9545455, 0.9, 0.8333333, 0.75, 0.6428571, 0.5, 0.3, 0` — diffusers' grid exactly.
The 5.0e-3 belongs to the scheduler's constructor default, which is what candle upstream
reproduces and what nothing ships. Recomputed both grids from the fetched sources before
touching a word; the decision paragraph now says so and the unit test's last assertion
reads "not the constructor-default grid".

**The pipeline entry fell through to the GGUF reader**, which is the arc's own bug and the
expensive one. `Model::ZImageTurbo`'s first file is `model_index.json`, so
`resolve_model` returned it and `CheckpointSource::open`, seeing neither a `config.json`
nor a `.safetensors`, tried it as a GGUF: `xwen inspect --model-size zimage-turbo`
downloaded 32.9 GB and then failed on a magic number. Fixed at the seam rather than at the
call site — `checkpoint::diffusion_snapshot_root` is now the one rule for what a snapshot
path is, `safetensors_dir` refuses it naming `xwen image`, and `inspect` gates on the
format ahead of the fetch. The gate there is deliberately the FORMAT and not
`servable()`: the text encoder is unservable and inspecting it is exactly what someone
wants. `encode-text`'s remap onto `text_encoder/` accepts both spellings of the snapshot
now, the directory and its `model_index.json`, which is the path `xwen fetch` prints.

**The PNG temporary name was per process, not per writer**, so two writers to one
destination inside one process would have shared it. Exclusively created with a counter
now, and removed on a failed rename as well as a failed encode.

**The attention A/B bar was 2e-3 against a signal of 1.2e-8.** The whole gap was
unmeasured, and the outside reviewer's claim that a broken arm passes it is correct: on
that fixture an arm with the `1 / sqrt(head_dim)` scale dropped differs by 9.1e-4 and one
with uniform probabilities by 8.9e-5. The reason is the fixture, not the bar. Driven
through `forward` with the test's random weights, the qk-norm weights are uniform on ±0.1,
the logits span ±1.3e-2 and the probabilities sit within 2e-4 of a flat 1/64 — a softmax
that is already uniform cannot report that its softmax broke. So the new test drives the
two arms directly with q and k at unit RMS, where the same two mutations move the output
by 2.0 and 0.96 against a real fused-versus-basic difference of 9.5e-7, and the bar is set
at 2e-5: bracketed from both sides, more than twenty times above every real difference and
more than four times below every wrong one.

## Arc B, 2026-09-07: Stage 3 and Stage 4, the transformer against its reference

The reference is diffusers 0.40.0 on torch 2.14.0, mps, running the shipped
`transformer/`, `vae/` and `scheduler/` from `prepare_latents` onwards, driven by
`scripts/zimage-ref-dump.py --stage transformer`. Two inputs are files both sides read:
`latents0`, a CPU `torch.randn` draw at seed 0, and `cap_feats`, the encoder's
`hidden_states[-2]` for prompt 1 of `prompts.json` ("portrait-golden-hour", 73 tokens),
computed fp32 on cpu and rounded once to bf16. Injecting the caption as well as the latent
is what makes this a transformer gate and not a pipeline gate: the encoder has its own
(Stage 2) and folding it in would have graded two things at once. Two arms: fp32 is the
reference and writes the inputs, bf16 rereads them so it differs in arithmetic alone and
its gap to fp32 is the bar's yardstick. The VAE decodes in f32 in both arms, as xwen does.
The dump, encoder load included, ran in under two minutes; the fp32 transformer arm was
1.10 s per step at 512x512 and the bf16 arm 0.35 s.

The fixture is `tests/fixtures/zimage-transformer/512x512-p1-s0/` (1.5 MB: the two
inputs, the fp32 velocity, latent and PNG, and `meta.json` with the bf16-vs-fp32 spread
and every sha256), and the gate is `tests/zimage_parity.rs`:

```
cargo test --release --test zimage_parity -- --ignored --nocapture
```

18.9 s, one pipeline load, and it prints this table before asserting. Rerunning the dump is
the two `--stage transformer` lines in the script's header; `--width`, `--height`,
`--prompt-idx` and `--seed` make another case, and the test grades every case directory
it finds.

| step-0 velocity, 512x512 | cosine | mean rel | max rel | bar |
| --- | --- | --- | --- | --- |
| xwen bf16 vs fp32 reference | 0.999302 | 0.0205 | 0.1006 | inside |
| reference bf16 vs fp32 (its own spread) | 0.999560 | 0.0175 | 0.0890 | inside |
| bracket: timestep one grid point off | 0.6045 | 0.6288 | 0.9541 | outside |
| bracket: caption tokens reversed | 0.8633 | 0.3259 | 0.6536 | outside |

| after 8 steps | xwen bf16 | reference bf16 arm |
| --- | --- | --- |
| final latent vs fp32, cosine / mean rel | 0.990845 / 0.0631 | 0.994750 / 0.0492 |
| image PSNR vs the fp32 PNG | 29.71 dB | 32.40 dB |
| reference latent through xwen's VAE vs the fp32 PNG | 92.62 dB | |

Gated: the step-0 velocity at cosine >= 0.998 and mean relative error <= 0.04, and the VAE
alone at PSNR >= 60 dB. Reported: the final latent and the image PSNR. The brackets run
inside the test on every execution, so the bar is re-proven to separate a wrong graph from
a rounding difference each time it passes (decisions.md "Verification is a torch dump with
an injected latent"). xwen against the reference's bf16 arm directly is cosine 0.99971,
closer than either is to fp32, which is what a shared bf16 rounding component looks like.

The code that made it possible: `ZImagePipeline::generate` returns `Rendered { image,
timings, velocity0, final_latents }`; `velocity(latents, cap_feats, t)` is one forward,
public so the test can run it with a wrong `t`; `decode(latents)` is the VAE tail alone;
`read_cap_feats(path)` mirrors `read_latents`; `encode_png(image) -> Vec<u8>` is the PNG
encoder split out of `write_png` for the serve route to come. On the CLI, `xwen image
--cap-feats <file>` skips the encoder entirely (the prompt is ignored and said so) and
`--dump <dir>` writes `velocity0.safetensors` (`velocity`) and `latents-final.safetensors`
(`latents`). The CLI run on the fixture inputs reproduces the test's numbers.

Timings from the same session, 512x512, power mode NOT read this session: xwen 1.23 s per
transformer step and 1.06 s VAE decode; torch mps fp32 1.10 s per step, bf16 0.35 s per
step, f32 VAE 0.7 s. torch's bf16 arm is about 3.5x faster per step than xwen at this size,
which is the first cross-implementation datum the step-time ledger item has.

### Not taken now, Arc B

- **xwen's bf16 arithmetic is a little noisier than torch's.** 1 - cosine at step 0 is
  7.0e-4 against the reference bf16 arm's 4.4e-4, about 1.6x, and it compounds to 29.7 dB
  against 32.4 dB after eight steps. The graph is right; the candidates are candle's Metal
  sdpa accumulation and the bf16 elementwise chains inside the block (norms, modulation,
  the residual adds). Reopen if the Stage 4 PSNR gap to the reference bf16 arm is judged
  visible on a real image, or if a perf change wants to spend precision and needs to know
  where the budget already goes.
- **A 1024x1024 fixture.** Not committed, at about 6 MB for the set; the script produces
  one in a minute with `--width 1024 --height 1024`. Reopen if a size-dependent bug is
  suspected (the rope tables past 32 positions per axis, the 4096-token attention).
- **F32 activations against bf16 weights.** The reopen condition in Arc A's list was "when
  the step-0 parity gap exists"; it exists now and it is small, so there is nothing to buy
  at the bar. It stays not taken, with the first item above as its new reopen condition.

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

## Arc C, 2026-09-07: the images route in `xwen serve`

The third arc of the day, and the one the user was waiting for: Z-Image-Turbo behind the
OpenAI images shape so ComfyUI on another machine can render through this one. What
shipped is `src/serve/images.rs`, one handler on three paths, and an `image-engine` thread
beside the language engine (decisions.md "CLI first, then serve as OpenAI", the Arc C
paragraph, for the six choices). The prompt rendering that `encode-text` and `image`
shared moved from the CLI into the library as `zimage::conditioning::prompt_ids`, so the
route renders the caption the way diffusers does without a copy of the code.

The engine is one OS thread over a bounded crossbeam queue of four. It opens everything
`xwen image` opens, from the cached snapshot, on the first request: the encoder through
`CheckpointSource` and `XwenModel::load_encoder`, then `ZImagePipeline::load`. It holds
both until `--idle-unload` elapses with nothing queued, then drops them and clears the
flag `/health` reports as `image_model_loaded`. `n` images render one after another with
seeds `seed, seed + 1, ...`; a drawn seed stays under 2^53 so a JavaScript client can echo
it back exactly. Every event is a `ServeLog::HostLine`: the load with its encoder and
transformer split, each render with its size, steps, seconds and seed, a proxy-path model
substitution, a truncated prompt, the idle unload with the measured and configured spans.
The two engines do not know about each other's residency (decisions.md, above).

The contract, as implemented. Paths `POST /v1/images/generations`, `POST
/images/generations` (where a client that `urljoin`s a base URL without its trailing
slash lands) and `POST /proxy/openai/images/generations` (ComfyUI's stock OpenAI image
node under `--comfy-api-base`). Unknown fields are dropped.

| Field | Rule |
| --- | --- |
| `prompt` | required, non-empty after trim; else 400, `param: "prompt"` |
| `model` | on `/v1` and `/images`: absent, `""` or `Z-Image-Turbo`, anything else 400 (the CLI alias included); on the proxy path: anything, logged |
| `size` | `WxH`, `auto` or absent (both 1024x1024); malformed is 400 `Invalid size format: '...'. Expected WIDTHxHEIGHT.`, then `check_size`'s own sentence as a 400, both `param: "size"` |
| `width`, `height` | both present override `size`; one alone is a 400 |
| `n` | 1 to 4, default 1 |
| `response_format` | absent or `b64_json`; `url` is a 400 saying the server hosts no URLs |
| `output_format` | absent or `png` |
| `stream` | `true` is a 400 |
| `negative_prompt` (non-empty), `guidance_scale` (non-zero) | 400: Turbo runs without guidance and would ignore them silently |
| `seed` / `rng_seed`, `steps` / `num_inference_steps` | either spelling; both present must agree; steps 1 to 50, default 8 |

Statuses: 400 for a request fault, including an uncached checkpoint (the message names
`xwen fetch --model-size zimage-turbo`); 503 with `retry-after: 5` when four requests are
already queued; 500 when a render fails or the engine is gone; 403 rather than 401 for a
missing API key on these paths. Never 401, 402, 409 or 429. The 200 body is `{"created",
"model": "Z-Image-Turbo", "size": "WxH", "steps", "output_format": "png", "data":
[{"b64_json", "seed"}]}`.

Smoke, `xwen serve --port 5252 --idle-unload 30s` on a dev-tree release build, the
language model never loaded, power mode not read:

| Request | Status | Wall |
| --- | --- | --- |
| `/v1/images/generations`, 1024x1024, seed 7, cold | 200 | 80.4 s: load 33.6 s (encoder 2.3, transformer and VAE 31.4) plus render 46.7 s; the PNG decodes as 1024x1024 RGB8 and shows the prompt |
| `/proxy/openai/images/generations`, the stock node's exact payload with `model: gpt-image-1`, warm | 200 | 48.6 s, substitution logged |
| `/images/generations`, 512x512, `n: 2`, seed 100 | 200 | 23.3 s, seeds 100 and 101, 11.6 s per render |
| `size: "1024x"` | 400 | the size message, `param: "size"` |
| `response_format: "url"` | 400 | the b64_json-only message |
| `model: gpt-image-1` on `/v1` | 400 | the unknown-model message naming Z-Image-Turbo |
| malformed JSON | 400 | the parse message |
| `/health` after the renders, then 40 s later | 200 | `image_model_loaded` true, then false; the log shows the unload after 30 s idle |

Ten unit tests cover the validation table and the three-path predicate without a
checkpoint, including that the stock node's exact payload parses. The full suite stayed
green (1321 library tests) and clippy reports nothing new in the touched files.

### Not taken now, Arc C

- **Cross-engine eviction.** The language engine and the image engine each unload on
  their own idle timer and neither asks the other to leave. Fine at 20 GB plus 20 GB;
  Flash-Next plus images thrashes, mmap-backed, until one of them idles out. Reopen when
  someone serves Flash-Next and images from one process and measures the stall; the
  mechanism would be a control message into `JobQueue` that the engine loop treats as an
  immediate idle unload.
- **Cancellation of a queued render.** A client that hangs up leaves its job in the queue
  and the engine renders it anyway; the reply goes nowhere. Bounded by four queued jobs of
  seconds to a minute each. Reopen if queues form in practice; the fix is a cancel flag
  on `ImageJob` checked before the render starts, as the language jobs already carry.
- **Listing Z-Image-Turbo in `/v1/models`.** Kept out, since the chat routes refuse it and
  the list is what chat clients pick from. Reopen if an OpenAI images client turns out to
  read `/v1/models` to populate its own dropdown.
- **The proof from the laptop.** The route is proven with the stock node's payload from
  curl, not yet from ComfyUI itself on another machine. That is the verification still
  owed, below.

### Next

Three arcs landed today and the graph is graded, so what remains on this ledger is a
proof, a candidate and a perf item. The proof is the one Arc C left owed: ComfyUI on the
laptop with `--comfy-api-base http://<this mac>:<port>` and the stock OpenAI image node,
no API key (the node sends none and would get a 403), rendering through this server; the
mechanism is confirmed in ComfyUI's source and unattested in the wild, and the first run
settles it. The candidate is LoRA support, which the user asked about and which is
server-side by design (merge at load, a `<lora:name:scale>` tag or an extension field, a
listing route); it enters the ledger as an area item behind the proof once it carries a
number or a named LoRA. The perf item is TODO.md "Z-Image step time", now with a measured
ceiling: torch bf16 on mps runs a 512x512 step in 0.35 s against xwen's 1.23, and the
first job there is to measure the matmul rate xwen achieves at 1024x1024, where the
parity gate above (unchanged by any of it) is what says a faster step is still the right
one. The diffusion `CheckpointSource` arm below now has its second consumer,
`images::Loaded::open`, which duplicates `run_image`'s open sequence; that is the reopen
condition met, and it is a chore for whichever arc touches the loader next.
