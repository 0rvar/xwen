# Flint kernels: sinks in flash, a decode kernel, and the sigmoid MoE glue

The record of the arc that took Flint's attention off the f32 chain the correctness arc
built as an oracle, and put its MoE layers on the fused glue the Qwen MoE already ran,
2026-10-26. Three commits:
[3063e0e](https://github.com/0rvar/xwen/commit/3063e0e) (the kernels),
[18e1294](https://github.com/0rvar/xwen/commit/18e1294) (the review fixes) and
[23fcd68](https://github.com/0rvar/xwen/commit/23fcd68) (memory admission refit). The
previous arc is [flint-correctness.md](flint-correctness.md), the decisions are in
[decisions/flint.md](../decisions/flint.md), and the parity track is
[docs/parity.md](../parity.md) "The Flint track".

## What changed and why

Before the arc both attention arms were the same f32 math. Every step widened the full
layers' f16 cache to f32 and transposed it, then built a score tile, appended a sink
column and ran a softmax. At 32k keys that is about 350 MiB of copies per full layer per
token, roughly 4.9 GB a token over the 14 NoPE layers, against ~2.3 GB of weights. Decode
halved between a short prompt and a 3.6k one for that reason. The MoE layers ran the
candle chain for routing (8 dispatches) and a separate shared expert, because the fused
router only did softmax and the fused epilogue assumed a gated shared expert.

Now:

- Prefill attention runs `flash.metal` with a per-head sink (`ops::flash_attn_sinks`).
- One-row attention runs a new kernel, `attn_decode.metal` (`ops::attn_decode`).
- `production_attention` takes the visible window as geometry, `(k_off, window)` from
  `flint::attention::visibility`, instead of a mask. Masks are built only when the
  reference arm is selected.
- The fused router has a sigmoid branch, and the epilogue and the fused shared-expert
  pair have ungated variants, so `MoeBlock::glue_fused` is true for Flint.
- `flint_scratch_bytes` is refitted to the new arm's footprints, and prompts past ~60k are
  no longer refused.

`reference_attention` is untouched. It is the oracle every kernel test below grades
against, and `XWEN_FLINT_ATTN=reference` is now a bisect arm that shares no attention
code with the shipped path.

## Prefill: the sink in flash.metal

`flash.metal` works in the log2 domain: Q is premultiplied by `scale * log2(e)` and the
softmax uses `exp2`. A sink is a raw logit, not scaled by 1/√128, so it enters as
`sink * log2(e)`. With `has_sinks` set, each row's running max starts at that value and
its running sum at 1, which is `exp2(sink - max)` at the seed. The kernel's existing
rescale then carries the sink term along as the max moves, and the final divide includes
it. Without sinks the seeds are `-inf` and 0 as before, chosen at runtime, and buffer 5 is
bound to q's buffer and never read. `FlashAttnArgs` grew two 4-byte fields (`has_sinks`,
`pad0`) so the i64 strides stay 8-aligned.

The window and causal edge were already computed in the kernel from `pos`, `k_off` and
`window`, so the sliding layers need no mask: `k_off = pos - min(pos, 513)` matches the
surviving ring entries `prepend_existing` puts in front of the chunk.

Measured against `reference_attention` with `attn_mask_for`'s mask, K/V f16-rounded on
both sides, as worst error relative to each output row's largest value:

| case | row-relative error |
| --- | --- |
| NoPE full chunk, 64 rows at pos 32,773, 32,837 keys, strided view, sinks | 1.77e-5 |
| same, sinks at -1e30 | 1.68e-5 |
| full layer, 37 unaligned rows at pos 100 | 2.2e-6 |
| sliding chunk across the wrap, 70 rows at pos 1200 | 2.8e-6 |
| sliding chunk wider than the window, 600 rows from 0 | 3.7e-6 |
| sinks at -1e30 against the sinkless kernel | bitwise |
| sinks, head-strided view against packed | bitwise |

The 32k row needed a 4e-5 bar. An f64 probe on four rows showed the kernel and the oracle
each about 1.1e-5 off the exact answer in independent directions, so the gap is two f32
errors added together, not a kernel fault. Every pre-existing flash test (the Qwen3 dense
path and Z-Image) is still bitwise against its reference.

## Decode: attn_decode.metal

Derived from candle's `sdpa_vector` and `sdpa_vector_2pass`, which xwen could not use
as they are: they take no sink, they assume K/V rows packed at the head dim (so no
head-strided cache views), and at Flint's GQA of 8 each query head re-reads its KV head,
8x the bytes.

- One threadgroup per (split, KV head) holds all 8 query heads of the group in registers,
  reads each K/V row once (half4 per lane, 256 B coalesced) and keeps 8 online-softmax
  states. Instantiated for group sizes 1, 2, 4 and 8.
- K and V are read in place through explicit head strides, so the full layer's capacity
  view and the 513-slot ring need no copy. Ring order does not matter at one row, since
  the row sees every key in the view.
- Four simdgroups take alternate keys of their split and merge through 16 KiB of
  threadgroup memory.
- Split-K over keys, `attn_decode_splits(keys, n_kv) = min(keys / 32, 128 / n_kv)`,
  clamped to 1..=128, so every split holds at least 32 keys. For Flint's 4 KV heads that
  is 16 splits at the 513-key ring and 32 at 4k and 32k. One split finishes in one
  dispatch with the sink folded into the threadgroup merge. More than one writes partial
  rows plus (max, sum) to the tail of the output allocation, and `attn_decode_combine`
  merges them and folds the sink in as one more term.
- Log2 domain throughout, fast math pinned like its sibling kernels, and a finite floor
  of -3e38 in place of `-inf`, so every exponent is finite and at most 0. Sinks at -1e30
  (the `NoSinks` bracket) are tested. Non-finite sinks are refused at load, because fast
  math assumes there are none.

Against `reference_attention` on Metal, same metric:

| case | row-relative error |
| --- | --- |
| 32,805-key NoPE layer through a 40,960 capacity view, splits 1 / 7 / 32 / 64 / 128 | 9.35e-6 worst (split 1), ~7e-6 at 32+ |
| ring views of 513, 200, 3 and 1 keys, splits 1 / 2 / 8 / default | 1.29e-6 |
| no sinks, split 1 and default | 2.7e-6 / 2.2e-6 |
| group sizes 1, 2, 4, 8 at split 1, default and 5 | under 2e-5 |
| head-strided against packed | bitwise |

The bar is 2e-5. The test also refuses the shapes the kernel cannot take: two query rows,
f16 q, f32 K, head dim 64, a group size with no instantiation, misaligned q/K/V views and a
split count whose key range would overflow i32.

The ignored microbench (`attn_decode_microbench`), Flint's 32 query and 4 KV heads, one
row, per call. `lowpowermode 1`, unpinned, and the GPU shared with other runs, so these
are ranges:

| keys | old f32 chain | read floor | kernel, default split |
| --- | --- | --- | --- |
| 513 | 300-480 µs | ~14 µs, dispatch bound | 12-17 µs |
| 4,096 | 420-490 µs | 11-16 µs | 31-36 µs |
| 32,768 | 3.7-5.6 ms | 129-135 µs | 188-250 µs |

The read floor is half the time of an f16 affine over K and V. Two variants were tried and
dropped: a butterfly reduction that leaves each lane owning one head was 10-25% slower at
4k and 32k, and four keys in flight per simdgroup was not distinguishable from the first
version.

The decode glue in `FlintAttn::forward` lost its copies too: at one row q, k, v and the
output are reshaped rather than transposed and made contiguous, and k's f16 cast is folded
into the rope on the sliding layers.

## The MoE glue

The router kernel's args gained a `gating` field. The sigmoid branch replaces only the
softmax step, computing `1 / (1 + exp(-logit))` per expert, which is candle's `usigmoid`.
The bitonic sort, gather, clamped sum and divide are shared. It is bitwise against
`route_sigmoid_from_logits` on ordinary logits at seq 1, 2, 7 and 64, on padded expert
counts, and on saturated ties: a plateau of near-equal high logits, every logit high,
every logit at -200 where all scores are +0 and the 6.1e-5 floor binds, and mixed rows.
The test asserts that the plateau's six winners really are one float on the device.

The epilogue became a template on `GATED`. The ungated variant computes
`value + shexp` and is bitwise against `ops::combine` plus candle's add, which is what the
ungated path ran before. The fused shared-expert pair (gate/up, then epilogue) got the
same treatment, but it cannot be bitwise against the candle chain because the gated pair
never was: it reassociates the dot products and is graded by tolerance. The ungated pair
is held to the same bars (rel L2 1e-5 against an f32 host oracle, 1e-6 at one row and
2e-6 above against the classic chain). Two bitwise identities against the gated kernels
at a saturated gate of 90, whose sigmoid is exactly 1.0, catch a wrong buffer slot or
branch.

At decode a Flint MoE layer went from 19 dispatches to 8, about 517 fewer per token over
47 layers. The f16 overflow guard on the routed down projection holds: the fused tail is
entered only when `FusedExperts::project` returns a result, and it returns none when
`needs_rescale` holds, so the default prefill path keeps the rescale chain.

## Parity

Against llama.cpp Metal at a37cc461, after all three commits, pooled over every position
(argmax / top-5 / mean |Δlogit|):

| run | before the arc | after |
| --- | --- | --- |
| prefill, chunk 512 | 0.970 / 0.949 / 0.210 | 0.9696 / 0.9461 / 0.2163 |
| prefill, chunk 2048 | 0.972 / 0.947 / 0.203 | 0.9711 / 0.9461 / 0.2140 |
| decode, 300 to 602 across the ring | 0.940 / 0.921 / 0.238 | 0.9485 / 0.9219 / 0.2366 |

Every bar holds with room. The brackets still fail every bar: sinks dropped reads argmax
0.0293 and rope on every layer 0.4208. The window-512 bracket reads 0.9715, inside the
noise as before. The ring test is what holds the window, and since 18e1294 it runs the
shipped `visibility()` helper, so an off-by-one in the path that ships fails it.
`tests/flint_chunks.rs` passes. The lib suite read 1594 passed and 1 failed, the failure
being `stack_profile::a_host_stage_does_not_swallow_the_gap_before_it`, a sleep-timing
assertion that passed three times out of three alone.

## Speed observations, which are not figures

`lowpowermode 1`, an unpinned binary at 3063e0e,
`xwen generate --model flint --raw --stats -n 256`:

| | before (correctness arc) | after |
| --- | --- | --- |
| decode, short prompt | 47.7 tok/s (24 tokens) | 59.4 tok/s (13 tokens) |
| decode at a 3654-token prompt | 23.7 | 50.7 |
| prefill at 3654 tokens | 292 | 1149 |

Decode no longer halves with context. None of this goes in perf-state.md; the pinned
bench in high power mode is the open item on the TODO.md Front.

## Memory

Peak footprint beyond the mapped weights, `/usr/bin/time -l` on
`xwen generate --model flint -n 8` with a pinned build of 3063e0e. The two longest rows
ran on a scratch build with admission switched off, since the old fit refused them.
Prefill rates are low power mode.

| prompt tokens | KV capacity | peak | old arm | new estimate | prefill |
| --- | --- | --- | --- | --- | --- |
| 544 | 8,192 | 7.26 GB | | 7.92 GB | 925 tok/s |
| 10,043 | 16,384 | 7.48 GB | 10.97 GB at 9,895 | 8.41 GB | 1077 tok/s |
| 28,392 | 32,768 | 7.92 GB | 16.28 GB at 28,717 | 9.36 GB | 752 tok/s |
| 56,028 | 65,536 | 7.93 GB | 31.89 GB at 56,883 | 10.78 GB | 499 tok/s |
| 98,617 | 131,072 | 10.43 GB | | 12.98 GB | 310 tok/s |
| 198,401 | 262,144 | 16.33 GB | | 18.13 GB | 167 tok/s |

The superlinear growth is gone. Below ~60k a fixed 7.3-7.9 GB transient from load and the
dequantized planes dominates. Above it the peak follows the KV capacity, about 5.1 GB plus
1.5 times the KV, because old and new planes coexist while the cache doubles. Decode at
198k read 22.4 tok/s over 8 tokens.

`flint_scratch_bytes` is now `(7 GiB + 48 KiB per KV slot) x 1.05`, 11-40% above every
measured point at its token count, and admission prices it at the KV capacity, which adds
more slack (14.7 GB at 131,072, 21.4 GB at 262,144; the old quadratic said 122.8 and
444.1). The test holds the six points as data. A full 262,144 context prices at about
58 GB with the weights, inside the 128 GiB budget, so `--max-ctx` is the limit now. That
last figure is arithmetic over the formula, not a run.

## Reviews

Codex, DeepSeek and a Claude semantic reviewer read 3063e0e. Qwen did not, the work being
on xwen itself. None found a correctness bug. What 18e1294 and 23fcd68 changed from their
findings:

- The ring test computed the window itself and bypassed the derivation that ships, which
  is the one error no logits bar can see. The rule is one helper, `visibility()`, that
  `forward` and the test both call.
- A position that disagrees with the cache length used to fail loudly as a mask shape
  error and would now have hidden keys silently. `forward` refuses it.
- The decode kernel refuses misaligned q/K/V views and a split range that overflows i32,
  and pins fast math like its siblings.
- The split count at 513 keys was 17 splits of 31 keys against a doc that said 16. The
  rule is now a floor of 32 keys per split, pinned by a test.
- Non-finite sinks are refused at load.
- The decode tests now grade the single-split no-sink branch and every group size at
  several split counts.
- Admission still priced the deleted f32 arm (23fcd68, above).

One claim was checked and was false: the production arm does not silently accept a mask,
it has no mask parameter.

## Not taken now

- **A simdgroup-MMA decode kernel.** At 32k keys the decode kernel runs ~1.5x above the
  read floor. Treating the 8 query heads of a group as the M=8 side of an 8x8 tile would
  cut per-key ALU and could approach the floor, worth at most ~14 x 60 µs, about 0.8 ms a
  token at 32k context, in low power. Reopen when the pinned bench shows long-context
  decode as Flint's largest gap to its ceiling.
- **An alignment check on the prefill flash kernel.** The decode kernel has one now; flash
  does not. Every cache view reaching it is row-aligned today. Reopen when a new caller
  can pass a view at an arbitrary offset.
- **The decode split rule is fitted on 4 KV heads only.** Other KV head counts get
  proportionally fewer splits. Reopen when a second architecture calls `attn_decode`.
- **The reference arm's memory is not priced.** It blocks at 256 query rows with no byte
  cap, ~1.9 GB per score tile at 57k keys, and admission prices the production arm. Reopen
  if the reference arm is needed past ~30k tokens.
- **The prefill precision gap is unchanged.** It sits in the projection matmuls, which
  this arc did not touch; the item stays in TODO.md "Prefill performance".
- **Recursive's Q4_1 GGUF** was a record line of the correctness arc. The owner asked
  about it on 2026-10-26, so it is a ledger item now, in TODO.md "Parity, provenance and
  tooling".

## Next

The pinned bench, on the TODO.md Front: plain decode short and at ~4k context and prefill
at ~3.6k on a pinned binary in high power mode, with the `pmset -g` line verbatim, so
perf-state.md gets Flint's first figures and the next Flint lever can be ranked against
the ~2.3 GB/token estimate.
