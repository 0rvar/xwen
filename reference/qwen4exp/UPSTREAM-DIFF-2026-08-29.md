# Upstream semantic diff: `bea3b12d` → `6fe749801` (read 2026-08-29)

A point-in-time reading of everything that changed in llama.cpp's qwen4exp
support between the snapshot originally vendored here (`bea3b12d`, the pre-merge
PR #27742 fork) and the current pin (`6fe749801`, upstream master). The merged
PR's base is `6fdd0ac8` and the PR itself landed as squash `6c84c7d5d`. This is a
READING, not an inventory: **PROVENANCE.md is the authority on what is vendored
and at which sha.** The findings are as of 2026-08-29 and go stale as upstream
moves; docs/qwen4exp-port.md carries whichever of them the port actually acts on.


## Files regenerated

All five under `/Users/orvar/develop/private/xwen/reference/qwen4exp/`, plus `PROVENANCE.md`.
Working tree only at the time of writing.

| what | sha |
|---|---|
| pre-PR base (merge-base with master) | `6fdd0ac8907fd973a42b876357823ad2124cd8ed` |
| PR #27742 squash merge | `6c84c7d5d8833c6e0df69628f75a0f599797934e` (2026-08-27) |
| follow-up #27880 "reduce number of graph splits" | `6fe74980162af0ed5e559870d5deccafaa034e7c` (2026-08-28) — the pin |

File checksums (all verified against the manifest after writing):

```
1fe800a11543d2af7aef898f75e6ee6ffb1f27f1486b4810cb926d94fc77e148  qwen4exp.cpp
12a0a5aea7877fbb8fe35af041a9c34f8b57b05278871b22c24c650b9760dfc3  qwen4exp.py
9ab1af23b4bb6244e153a82b4dbe81610060d9e9672fb97299806ba98c71b20a  gguf-py.diff
05e887b46d5dfdbab0f3870b315bf328ab62d8d85de162d715f22191be2a8af9  llama-cpp-core.diff
3e3ea6be268c65915f8f5b8419edc15c291f0adf374b97216f436191d23456b4  conversion-qwen-base.py
```

Merged PR is 28 files / +2881 (the pre-merge fork sha was 24 / +2191). Structural moves:

- `src/llama-memory-hybrid.cpp` is no longer patched — a new
  `src/llama-memory-hybrid-idx.{h,cpp}` replaces that approach.
- **`src/llama-graph.{h,cpp}` is no longer patched at all.** `build_attn`'s `top_k`
  parameter and `build_attn_mask_top_k` are gone; the sparse mask build is now
  arch-local in `qwen4exp.cpp::build_attn_qsa`.
- `gguf-py/gguf/lazy.py`, `conversion/base.py`, `src/llama-context.cpp` and
  `src/llama-model.h` are new to the diff.
- `conversion/qwen.py` is still untouched by the PR (its diff against the old vendored
  copy comes entirely from the unrelated `ca3d5a3e1` DSpark/Nemotron3.5 commit).

The two core sections are concatenated per-commit rather than taken as one
`6fdd0ac89..6fe749801` range diff, because 17 unrelated master commits land in that range
and four of them (`ca3d5a3e1` DSpark, `866322481` GDN/LID op gating, `18443257a`
ctx-per-slot, `4e97ac86e` test-save-load-state) touch the same files.

---

## Semantic diff, ranked by impact

### 1. QSA now applies the Hadamard rotation instead of refusing it

The old code routed QSA through `llm_graph_context::build_attn`, which carried:

```cpp
GGML_ASSERT(inp->self_k_rot == nullptr && inp->self_v_rot == nullptr);
```

QSA + quantized KV was a hard abort. The new arch-local `build_attn_qsa` does:

```cpp
if (inp->self_k_rot) {
    q_cur = llama_mul_mat_hadamard(ctx0, q_cur, inp->self_k_rot);
    k_cur = llama_mul_mat_hadamard(ctx0, k_cur, inp->self_k_rot);
}
if (inp->self_v_rot) {
    v_cur = llama_mul_mat_hadamard(ctx0, v_cur, inp->self_v_rot);
}
...
// the rotation is its own inverse, so undo it on the value side of the output
if (inp->self_v_rot) {
    cur = llama_mul_mat_hadamard(ctx0, cur, inp->self_v_rot);
}
```

explicitly after the indexer has scored with its own query, "so top_k is unaffected".

**Numerical effect on an f32/f16 run: none** (both rot tensors are null). It removes a
blocker for quantized KV.

### 2. QSA gained a second bias mode, `blk_bias` — equivalent, not a math change

Selected when the kq_mask shape matches and the run is plain causal:

```cpp
const bool blk_bias = kq_mask != nullptr &&
    kq_mask->ne[0] == n_kv && kq_mask->ne[1] == n_tps && kq_mask->ne[3] == n_stream &&
    cparams.causal_attn && !hparams.use_alibi;
```

It uploads `n_blocks` floats per query instead of `n_kv`, adds them to `score` *before*
the block→cell expansion, and then adds the attention `kq_mask` (cast f16→f32 for
flash-attn) in place of the old per-cell bias:

```cpp
if (blk_bias) { score = ggml_add(ctx0, score, inp->bias); }
... expanded = get_rows(permute(score), inp->cell_blk) ...
if (blk_bias) {
    ggml_tensor * mask = kq_mask->type == GGML_TYPE_F32 ? kq_mask : ggml_cast(ctx0, kq_mask, GGML_TYPE_F32);
    expanded = ggml_add(ctx0, expanded, ggml_reshape_3d(ctx0, mask, n_kv, n_tps, n_stream));
} else {
    expanded = ggml_add(ctx0, expanded, inp->bias);
}
```

Host side (`llama_memory_hybrid_idx_context::set_input_qsa`, moved there from
`llama_kv_cache`):

```cpp
if (blk_bias) {
    for (int64_t b = 0; b < n_blocks; ++b) {
        cur_blk_bias[b] = b*r >= tail_start ? 1e9f : (filled[b] < r ? -INFINITY : 0.0f);
    }
    continue;
}
// per-cell path, unchanged:
v = cells.pos_get(j) >= tail_start ? 1e9f : (blk_of[j] < 0 ? -INFINITY : 0.0f);
```

I checked equivalence rather than trusting the comments:

- `tail_start = (q+1)/r*r` is a multiple of `r`, so a block sits wholly inside or outside
  the tail. `b*r >= tail_start` per block == `pos >= tail_start` per cell.
- In `blk_bias` mode `blk_of[j]` is deliberately **not** reset to −1 for partial blocks
  (`filled[blk_of[j]] < r && !blk_bias`), so a partial block's own −INF reaches its cells
  through the expansion.
- Empty / foreign-sequence / future cells were masked by the host bias before; now
  `kq_mask` does it. Hence the causal-and-no-alibi guard.
- Adding a per-block constant before vs. after a `get_rows` expansion is identical.
- New `GGML_ASSERT((!blk_bias || !oor) && "qsa: cell position runs past the cell window")`
  covers the one case a per-block bias cannot express.
- f16→f32 mask cast is exact for the 0 / −INF values the mask holds.

The per-cell path is otherwise byte-identical. **`width = std::min<int64_t>(n_kv,
hparams.indexer_top_k + r - 1)` is unchanged**, whole-blocks-plus-tail is unchanged,
`tail_start` is unchanged, the rectified per-head score sum is unchanged.

**Numerical effect f32/f16: none.**

### 3. An unmerged QSA rework is in flight that WOULD change numerics

Branch `origin/tmp-q4`, head `f91123d2d02907c665cee42d6428eee5ee35d356` (2026-08-28,
Thiago Padilha, ~1450 lines, "Assisted-by: Codex"). Not on master, not vendored.
`git diff 6fe749801 f91123d2d -- src/` touches `llama-arch.cpp`, `llama-context.cpp`,
`llama-memory-hybrid-idx.{h,cpp}` (+445), `models/models.h`, `models/qwen4exp.cpp` (448
changed lines).

The QSA input signature is replaced wholesale:

```cpp
// merged (6fe749801)
set_input_qsa(cell_blk, blk_cells, blk_pos, bias, ubatch, ratio, blk_bias)
// tmp-q4
set_input_qsa(block_cells, block_pos, block_mask, selected, kq_mask, ubatch, ratio, block_topk)
```

What actually changes, semantically:

**Whole blocks vs token fill — this is the real change.** Merged code buckets cells by
absolute position: `b = p/r`, member slot `cur_blk_cells[b*r + (p%r)]`, and a block is
"partial" (`filled[b] < r`) if *any* position in its window is missing from the cache, in
which case the whole block is hard-masked. tmp-q4 instead walks a new per-sequence
`qsa_histories` list **in token order**, matches each history token to a cache cell by the
full key `(pos, ext.y, ext.x)` (with a `next_cell` index to disambiguate duplicates),
keeps it only if the real `kq_mask` says it is visible, and then **packs** the surviving
cells into blocks of `ratio` consecutive *visible* tokens:

```cpp
const size_t n_complete = visible.size()/ratio;
const size_t n_write    = std::min<size_t>(n_complete, n_blocks);
for (size_t ib = 0; ib < n_write; ++ib) {
    mask_data[iq*n_blocks + ib] = 0.0f;
    for (uint32_t ir = 0; ir < ratio; ++ir) {
        const uint32_t cell = visible[ib*ratio + ir].second;
        cell_data[(iq*n_blocks + ib)*ratio + ir] = cell;
        used_cells[cell] = 1;
    }
    ...
}
```

So a hole in the cache no longer voids a block — it shifts the packing. Blocks are formed
per query token, per sequence, from visible cells only.

**Tail.** Still unconditionally visible, but it is now the *packing* remainder
(`visible.size() % ratio` trailing tokens), not the positional remainder above
`(q+1)/r*r`:

```cpp
const size_t selected_start = n_complete <= block_topk ? 0 : n_complete*ratio;
for (size_t iv = selected_start; iv < visible.size(); ++iv) {
    selected_data[iq*n_kv + visible[iv].second] = 1.0f;
}
```

Note the budget is now expressed in **whole blocks** (`block_topk`), and when the number
of complete blocks is within budget the selection collapses to fully dense
(`selected_start = 0`). `selected` is an F32 `[n_kv, n_tokens]` indicator, replacing the
merged code's ±1e9 / −INF bias trick.

**Rope position of pooled keys.** Merged code writes a synthetic position, the block's
first *absolute* position, broadcast identically to all four M-RoPE sections:

```cpp
dst_blk_pos[sec*(n_blocks*n_ns) + s*n_blocks + b] = (int32_t) (b*r);
```

tmp-q4 writes the first member token's **actual full M-RoPE position vector**, section by
section:

```cpp
for (int64_t ip = 0; ip < n_pos; ++ip) {
    pos_data[(iq*n_pos + ip)*n_blocks + ib] = visible[ib*ratio].first->pos[ip];
}
```

That is exact for images too, where the merged comment admits its version is only
"approximate".

**Padding.** Unused block slots now get distinct unused fallback cells
(`used_cells` + `fallback_cell` scan) instead of reusing real history tokens — the commit
message's "avoids replacing padded tail entries with extra history tokens".

**Unified cache.** `qsa_histories` is keyed per sequence, which is the "prevents
unified-cache sequences from sharing pooled indexer keys" claim.

**PLE embedding widths.** The merged code silently bakes in
`ple_head_dim * ple_n_heads == n_embd`:

```cpp
layer.ple_key   = create_tensor(..., { n_embd, hc_dim }, 0);
layer.ple_value = create_tensor(..., { n_embd, n_embd }, 0);
```

tmp-q4 sizes both projections from the concatenated n-gram embedding instead:

```cpp
const int64_t ple_dim = (int64_t) hparams.ple_head_dim * hparams.ple_n_heads;
layer.ple_key   = create_tensor(..., { ple_dim, hc_dim }, 0);
layer.ple_value = create_tensor(..., { ple_dim, n_embd }, 0);
```

For the shipped checkpoint the two coincide by construction (the merged code could not
load the file otherwise), so this is a generality fix, not a numeric change — but the
latent shape assumption is worth documenting. tmp-q4 also derives `ple_rows` from
`max(offset + vocab_size)` rather than the tensor's `ne[1]` (making the table optional),
validates the PLE array lengths before copying into fixed storage, and validates the head
count as `uint64` before narrowing.

The rest of tmp-q4: metadata validation for GDN/HC/QSA/PLE dims at load rather than
graph-build abort; indexer-cache updates after sequence copies (raw keys copied without
RoPE shift); recurrent-state rollback; tensor-split disabled; a synthetic-arch QSA exact
mask check and a PLE save/load roundtrip in `test-llama-archs.cpp`.

**Bottom line: the vendored QSA selection should not be treated as settled.**

### 4. PLE conv state moved to its own recurrent row

`n_embd_r()` no longer includes `ple_conv_state()`; `llama_memory_recurrent` grows a third
tensor vector `p_l`, allocated only where `is_ple(i)`, named `cache_ple_r_l%d`, with its
own `size_p_bytes()` and its own state read/write rows (and a `pattern_ple_r_cache`
regex). `build_conv_state_at` lost its `row_offset` parameter and now asserts
`state_cols * channels == row_total`; `build_ple` reads `inp->mctx->get_p_l(il)` instead
of slicing an offset out of `get_r_l(il)`.

Output-identical, but **this refutes the port doc's "State-allocation note"** — upstream
now does what xwen planned.

### 5. PLE predecessor history left the model object

The `ple_hist` map and `ple_history` struct are gone from `llama_model_qwen4exp`.
Predecessors now come from the attention KV cells' `ext.tok` via
`llama_kv_cache::get_prev_tokens` (extended for M-RoPE gaps with a nearest-earlier-token
lookup and a `below[]` fallback, and for embd-only ubatches where predecessors resolve by
ubatch order rather than position). `apply_ubatch` stores `ext.tok` whenever
`ple_n_heads > 0`, padding embd batches with `ple_image_token_id` (or EOS).

```cpp
// old
auto & hist_map = pmodel.ple_hist;    // per-seq running list + next_pos contiguity check
// new
GGML_ASSERT(ubatch->n_seq_id[i] == 1 && "PLE n-gram embeddings do not support tokens shared by multiple sequences");
mctx->get_prev_tokens(*ubatch, n_prev, prev);
```

The cut logic is logically identical (`cut = cut || t < 0 || t == eos; ctx[s] = cut ? eos : t;`
versus the old assign-then-test). Identical output for a straight-line single-sequence
run; **differs after `seq_rm` / `seq_cp` / defrag**, where the old running-`next_pos` check
would reset the window to EOS padding and the new one recovers the real predecessors. An
oracle-vs-xwen *state* comparison will no longer line up.

### 6. GDN fused-QKV segmentation convention changed, value unchanged

```cpp
-    const int64_t head_v_dim   = d_inner / num_v_heads;
+    const int64_t head_v_dim   = hparams.ssm_d_state;
+    GGML_ASSERT(head_v_dim * num_v_heads == d_inner);
```

Both give 128 for this checkpoint, so no numeric change — but upstream now *requires*
`head_v_dim == head_k_dim == ssm_d_state`. `nb1_qkv` now derives from `conv_channels`
rather than a duplicate local `qkv_dim` (same expression, dedup only).

### 7. Graph-split follow-up (#27880) — output-equivalent, node ordering changes

- `build_inp_ple` split out of `build_ple` and hoisted above the layer loop with an
  explicit `ggml_build_forward_expand` "so ple_emb and build_inp_embd are in the same
  graph split" (safe because `n_ple == 1` is now asserted). `cb(emb, "ple_embd", -1)`
  where it used to be `il`.
- `inp_out_ids` row-dropping moved from after the final mixer into the **last layer**,
  applied to `cur`, `inject` and `res_hc` before the final `build_hc_combine`.
  Equivalent — everything below is per token — and cheaper.
- q/v/k expanded in a fixed order in `build_attn_qsa` ("expand k later to enable rope
  fusion which directly writes into k-v cache"), plus `ggml_build_forward_expand(gf, inpL)`.

Any node-level tap or profile taken against the old snapshot won't match.

### 8. QSA graph inputs are now shared and reusable

`qsa_inps` is a `map<ratio, llm_graph_input_qsa*>`, so layers sharing a compress ratio
share one input set (12 uploads → 1). Both `llm_graph_input_qsa` and `llm_graph_input_ple`
gained `can_reuse()`. `rs_rows` is keyed by cache tensor, not by layer.

### 9. Loader hardening, no math

Asserts on every SSM / HC / indexer dim; `GGML_ASSERT(n_ple == 1 && "qwen4exp supports
only one PLE layer")`; out-of-range PLE layer / n-gram size / head count are now
`runtime_error` not `GGML_ASSERT`; `is_ple_impl` switched from `std::fill` to
`.reset()`/`.set()` (a bitset); PLE head offsets and vocab sizes read as `uint64` then
narrowed with an `INT32_MAX` range check on offset, size and their sum (multipliers stay
64-bit); `require_weight` instead of `get_weight` + assert, plus a load-time check that
every head range fits the table's row count; the PLE table loads `TENSOR_READ_LAZY`.

**Unchanged math throughout**: `build_hc_mix` / `build_hc_combine` (grouped RMSNorm,
`1/hc` in both sigmoid args, `2·sigmoid` inject, mean collapse), `build_norm_gated`, the
signed-sqrt PLE gate clamp, the dilated depthwise conv (`dil = ple_ngram_size`,
`hist = (kern-1)*dil`, written as a sum of shifted per-channel-scaled copies), the MoE
path with its shared-expert sigmoid gate. Most of the 341-insert / 237-delete churn is
comment rewriting.

### 10. New memory type `llama_memory_hybrid_idx`

The indexer cache is its own `llama_kv_cache` (name-tagged `cache_idx_k_l%d`, MQA one head
of `indexer_head_size`), with cells forced to mirror the attention cache's slot infos
(`heads_idx = heads_attn`) and a `state_read_sinfo` so prompt-cache restore replays the
same slot layout into both caches. This is the multi-slot desync fix the port doc mentions
only as a PR-thread bullet.

### 11. MTP still entirely absent

No MTP in graph or converter (`no_mtp = True`, `supports_mtp_export = False`; the only
`NEXTN_*` hits in the diff are unrelated context lines). No change.

---

## Converter and GGUF keys

**Zero change to the GGUF surface.** Not one key literal, tensor name, `MODEL_TENSORS`
entry, hparam read, or tensor-math line differs between the fork sha and master.

**Metadata keys.** The only `constants.py` delta is a Python class rename,
`class PLE:` → `class PerLayerEmbedding:`. All ten key strings identical:
`"{arch}.hyper_connection.low_rank"` (`Keys.HyperConnection.LOW_RANK`, writer
`add_hyper_connection_low_rank`), and under `Keys.PerLayerEmbedding`: `"{arch}.ple.layers"`,
`"{arch}.ple.ngram_size"`, `"{arch}.ple.heads_per_ngram"`, `"{arch}.ple.conv_kernel"`,
`"{arch}.ple.layer_multipliers"`, `"{arch}.ple.head_offsets"`,
`"{arch}.ple.head_vocab_sizes"`, `"{arch}.ple.eos_token_id"`, `"{arch}.ple.image_token_id"`.
Writers `add_ple_*` unchanged, plus the private `_add_u64_array` helper — the three big
arrays are still written as `ARRAY of UINT64` deliberately (multipliers reach ~2.4e13 and
would truncate under INT32 inference).

**Enums.** `MODEL_ARCH.QWEN4EXP` → `"qwen4exp"`; new `MODEL_TENSOR` members
`HC_HEAD_NORM/DOWN/UP`, `HC_ATTN_NORM/DOWN/UP/INJECT`, `HC_FFN_NORM/DOWN/UP/INJECT`,
`PLE_KEY/VALUE/NORM_KEY/NORM_QUERY/NORM_CONV/CONV1D`. `INDEXER_Q_PROJ/K_PROJ/Q_NORM/K_NORM`
and `PER_LAYER_TOKEN_EMBD` were already in master pre-PR and are reused. The 48-entry
`MODEL_TENSORS[MODEL_ARCH.QWEN4EXP]` list is identical in both — notably no `OUTPUT_NORM`
/ `ATTN_NORM` / `ATTN_POST_NORM`, since hyper-connections replace every layer norm.

**Tensor names.** The `tensor_mapping.py` `QWEN4EXP` block is byte-identical, 19 entries.
No renames, additions or removals:

- `blk.{bid}.hc_attn_{norm,down,up,inject}` ← `model.layers.{bid}.attn_hyper_connection.{hc_norm,input_mix_weight_down,input_mix_weight_up,block_inject_weight}`
- `blk.{bid}.hc_ffn_*` ← the same four under `mlp_hyper_connection`
- `output_hc_{norm,down,up}` ← `model.hyper_connection_mixer.*`
- `blk.{bid}.indexer.{q,k}_norm` ← `model.layers.{bid}.self_attn.indexer.{q,k}_layernorm`
- `blk.{bid}.ple_{key,value,norm_key,norm_query,norm_conv,conv1d}` ← `model.layers.{bid}.ple.{key_proj,value_proj,norm_key,norm_query,norm_conv,conv1d}`
- `per_layer_token_embd.weight` emitted by hand from the PLE shards

**Converter math.** `modify_tensors` untouched: the `.indexer.index_qk_proj.weight` split
at `n_q = indexer_n_heads * indexer_head_dim` into `INDEXER_Q_PROJ`/`INDEXER_K_PROJ`, the
`data_torch + 1` Gemma-gamma folding for `.ple.norm_{key,query,conv}.weight` and
`.indexer.{q,k}_layernorm.weight`, and `.ple.conv1d.weight` → `.squeeze()` are all
identical. Inherited V-reorder and mrope behaviour unchanged (`conversion/qwen.py` is not
touched by the PR at all).

The one rewritten area is PLE table assembly, and it is a memory strategy swap, not a
layout change. Old: a `np.memmap` scratch file beside the output, shards written at
`idx * rows_per_shard`, tail truncated, temp file unlinked in a `write()` override. New:
shards are not materialised — only their names kept in `self._ple_shards: dict[int, str]`,
rows summed from `self.model_tensors[shard]().shape`, and a `gguf.LazyChunkedTensor` of
per-shard `load()` closures handed to `base.py`, quantized and written one row-chunk at a
time. Row order still ascending shard index, dtype still float32 — same bytes, bounded RSS.
`_write_ple_shard`, `_finish_ple_table`, `_ple_pending`, `_ple_rows_per_shard` and the
`write()` override are gone; `prepare_tensors` now checks `len(self._ple_shards) != n_parts`.
Everything else in that file's diff is cosmetic: dropped `json`/`MmprojModel` imports,
added `cast`, comment rewordings, and `@ModelBase.example` `unsloth/Qwen3.8-Flash-Next` →
`Qwen/Qwen3.8-Flash-Next`. Checked specifically; none touch output.

**Hparams.** Two behavioural deltas, neither changing what a well-formed checkpoint writes:

- `_image_token_id()` lost its `config.json` fallback and now reads
  `self.hparams.get("image_token_id")` only. Safe, and dead code before: `conversion/base.py`
  does `self.hparams = {**self.hparams, **self.hparams["text_config"]}` — a *merge*, so
  top-level keys survive narrowing — and that line is identical at the pre-PR sha.
- `_eos_token_id()` now raises
  `ValueError("eos_token_id is required: the PLE hash resets its n-grams on it")` instead
  of crashing on `int(None)`.

All other reads unchanged: `hc_count`, `hc_lowrank`,
`indexer_n_heads/head_dim/budget/compress_ratio`, `layer_types`, `ple_layer_ids` (1-based,
converted with `i - 1`), `ngram_size`, `heads_per_ngram`, `ple_conv_kernel_size`,
`split_ngram_parts`.

**`lazy.py` / `base.py` — infra only.** `LazyChunkedTensor` (61 lines) holds chunk-loading
callables plus shape/dtype/qtype/byteswap, exposes `nbytes`, `numpy()`, `quantize()`
(raising `QuantError` up front so base.py's F16 fallback still works), `byteswap()`, and a
`tofile()` that quantizes and writes chunk by chunk with a `written == self.nbytes` assert.
`base.py`'s 8 lines just route through it. Identical output bytes.

**Gap in the vendored diff.** `gguf-py.diff` is generated against `6c84c7d5d` vs its parent,
so it does **not** include `b19cbe925` "convert: prevent ndarray conversion in
LazyChunkedTensor (#27869)", which adds an `__array__` raising `TypeError` (numpy would
otherwise wrap the object and write 8 bytes) plus a short-write check in `gguf_writer.py`.
That is a real corruption fix that landed after the merge.

---

## Port-doc divergence check

Checked against `docs/qwen4exp-port.md` as it stood on 2026-08-29, before this
reading was folded into it; the doc has since been updated from it.

### Documented divergences: none GONE, one CHANGED

- **#2 "partially-filled non-tail blocks are hard-masked" — CHANGED (mechanism), still
  true.** There are now two bias paths. Partially-filled non-tail blocks are still fully
  hard-masked in both, so the divergence stands, but the doc's wording should be rebuilt
  around `blk_bias`, and xwen's rollback/defrag concern now has two upstream code paths to
  compare against.
- **#1 QSA top-k width — STILL STANDS**, `width = min(n_kv, indexer_top_k + r - 1)`
  verbatim identical. xwen's fixture-pinned whole-blocks-plus-tail choice still diverges
  from the oracle above budget.
- **#3 PLE gate clamp retraction — STILL ACCURATE.** `sqrt(clamp(abs(s), 1e-6, INF)) * sgn(s)`
  then sigmoid; matches HF; not a divergence.
- **#4 MoE renorm clamp — STILL STANDS.** `llama-graph.cpp` still clamps
  `weights_sum` at `6.103515625e-5` unconditionally inside `norm_w`, and qwen4exp still
  routes through `build_moe_ffn`.

### Two doc statements now flatly wrong about upstream

- The P0-pause **"State-allocation note"** — PLE conv state is no longer in `n_embd_r()`;
  it is a separate per-PLE-layer `p_l` row (finding 4).
- The implicit assumption from the vendored snapshot that **QSA refuses Hadamard-rotated
  KV** (finding 1).

### Traps checklist: all twelve survive, three need footnotes

1. GDN z-gate sigmoid — STILL STANDS (`build_norm_gated` unchanged).
2. PLE eos 248044 — STILL STANDS (`ple_eos_token_id`, still the reset token).
3. PLE hash over raw ids, constants from metadata — STILL STANDS. **Footnote:** offsets and
   vocab sizes are now read as `uint64` and narrowed with an `INT32_MAX` range check on
   offset, size and their sum; multipliers stay 64-bit.
4. `ple_layer_ids` one-indexed (config only; GGUF 0-based) — STILL STANDS. **Footnote:**
   `n_ple == 1` is now hard-asserted, and an out-of-range layer is a `runtime_error`.
5. `layer_types` "full_attention" means QSA — STILL STANDS (HF-side, unchanged).
6. HC write-back on un-normed stream, `/hc` in both sigmoid args, `2·sigmoid` inject, mean
   over streams — STILL STANDS, math byte-for-byte unchanged.
7. Signed sqrt + dilation 3 — STILL STANDS.
8. MoE no clamp / `[1,2560]` shexp gate — STILL STANDS.
9. QSA key/pool/rope/tail rules — STILL STANDS. **Footnote:** upstream still omits the
   `1/√128` score divisor the doc's formula includes. Monotone, so top-k is unaffected, but
   a numeric tap comparison will differ by that factor.
10. repeat×4 seed, final mixer, no output_norm — STILL STANDS.
11. `general.name` spaces — unaffected by upstream.
12. candle sdpa vector kernel ignores masks — xwen-internal, unaffected.

### Undocumented new upstream behaviour worth new entries

Findings 1, 2, 4, 5, 7, 8 and 10 above, plus:

- The converter's lost `image_token_id` fallback means a self-converted text-only file will
  likely carry no `ple.image_token_id` and silently fall back to EOS. Harmless text-only,
  but it looks like a regression worth reporting upstream.
- `ple_conv1d` is still **not** on the quantize skip list (only `ssm_conv1d`,
  `indexer.{q,k}_proj`, `per_layer_model_proj`), and the `qk_k <= 32` F16 fallback branch is
  still there — doc claim intact, still pin it with `--tensor-type` if self-converting.
- `--tensor-type` can now name `per_layer_token_embd.weight` past `--token-embedding-type`.
- The PLE table loads with `TENSOR_READ_LAZY` and `require_weight`, with a load-time check
  that every head range fits the table's row count — independent confirmation of D2.
- Finding 3's tmp-q4 branch, if it lands, invalidates most of the QSA entries.

### Sha- / line-pinned claims (stale by construction, listed not fixed)

- **D4 + D4 Update**: `bea3b12d` (the superseded snapshot), `6c84c7d5d`, `6fe749801`,
  PR head `eaf9376557`, master `17252c769`, `e9fa0781`.
- **All D4-Update numerics** are pre-merge PR-body figures: ppl 4.0068±0.0227 vs 4.0126,
  98.0% top-1, "bit-identical below 2048 on BF16/F32 (max delta 0.0 over 2051 rows)", ~3%
  divergence at 8192, 0.975 mean Jaccard vs a 0.991 floor, UD-IQ1_S max logit delta
  2.84e-3, and the DGX Spark GB10 perf (24-25 tok/s decode, 27.5 GB CPU + 78 GB CUDA).
  None re-measured post-merge, and findings 1/2/4/5/7 changed the graph.
- **Divergence #3**: "modular line 770" (HF `modular_qwen4_exp.py`).
- **Progress log 2026-08-26**: fixtures from "transformers main @ 598d8ba8"; commits
  `e99ffee`, `2914d7c`.
- **Reuse-seams map** header: "file:line refs from that audit"; the only live one is
  `model.rs:337` (`XwenModel::run_stack`), xwen-side, unverified here.
- **Conversion-baked deltas** section refers to "the sha in PROVENANCE.md", which now names
  `6fe749801` while the prose was written against `bea3b12d`.
