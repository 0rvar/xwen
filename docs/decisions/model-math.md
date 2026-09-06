# Model math: the forms that could have gone the other way

One topic of [docs/decisions.md](../decisions.md), the index of decisions by topic; moved
here verbatim on 2026-09-06. Dated paragraphs, newest additions appended within their topic.


Every entry here is a place where two defensible readings existed, the code had to pick
one, and picking wrong would have produced a model that runs, emits fluent text, and is
quietly incorrect. Each is pinned by a unit test, because "we checked once" does not
survive a refactor.

**silu runs over the WHOLE fused DeltaNet stream, before the q/k/v split — so q and k
are silu'd before their L2 normalization, not just v.** The natural misreading of the
recurrence is that silu is the value-path activation. It is not: `qwen35.cpp:397`
applies `ggml_silu` to the entire `[conv_dim, T]` conv output and the q/k/v views are
taken from the result afterwards (`:400-423`), matching HF's
`causal_conv1d_fn(..., activation="silu")`. Getting this wrong changes every q and k
that enters the delta rule and is invisible in any shape check (2026-07-28).

**The q/k L2 normalization uses ggml's clamp form `x / max(‖x‖, eps)`, NOT HF's
`x · rsqrt(Σx² + eps)`.** `ggml_compute_forward_l2_norm_f32` computes
`scale = 1.0f/fmaxf(sqrtf(sum), eps)` (`ggml/src/ggml-cpu/ops.cpp:4204`, read directly
from the vendored tree, not taken on report); HF/FLA computes the rsqrt form
(`modular_qwen3_next.py:222-224`). The two agree to rounding for any vector whose norm
clears eps, which a silu'd conv output always does — so this cannot move parity today.
It is still decided rather than left to chance: llama.cpp is the parity ground truth, a
strict tier may one day ask for bitwise agreement, and an "it cannot matter" difference
is exactly the kind that turns out to matter later. eps is `rms_norm_eps` read from the
checkpoint, not a hardcoded 1e-6 — llama.cpp passes `hparams.f_norm_rms_eps`, and only
the shipped checkpoints make those the same number (2026-07-28).

**The `1/√128` scale is applied to the readout only.** llama.cpp scales q once, before
the recurrence (`delta-net-base.cpp:319-321`), and q enters the recurrent form at
exactly one place — the `o = q·S` readout (`:365-366`). Scaling q up front and scaling
the output are therefore algebraically identical, and the chunked path applies the same
scale at the same point so the two forms agree. There is no second scale anywhere: not
on k, not folded into beta (2026-07-28).

**q and k are broadcast from K-heads up to V-heads by TILING — output head `j` reads
K-head `j % n_k_heads` — never by interleaving.** This was the single highest-risk
assumption in the DeltaNet port, because the usual way to write this broadcast in ggml
(reshape to `[d, 1, n_k, T]`, then repeat) yields interleave semantics, and both forms
type-check, run, and produce plausible output. `qwen35.cpp:442-443` repeats directly on
the natural `[head_k_dim, num_k_heads, T, S]` layout, and
`ggml_compute_forward_repeat_f32` (`ggml/src/ggml-cpu/ops.cpp:1723-1739`) writes
destination head `i1*ne01 + k1` from source head `k1` — tiled. This is deliberate, not
incidental: the converter pre-permutes every V-side weight from HF's grouped order into
tiled order precisely so ggml's repeat can replace an expensive interleaved one
(`conversion/qwen.py:355-378`). Reading GGUF, tiling is correct; reading HF safetensors
directly it would be `j / ratio`, and we do not read those (2026-07-28).

**The DeltaNet output norm is `rms_norm → × ssm_norm.weight → × silu(z)`, with the gate
LAST.** `build_norm_gated` (`qwen35.cpp:246-255`) normalizes, applies the weight, and
only then multiplies by `silu(z)`; current HF agrees and carries the literal comment
`# Norm before gate` (`modular_qwen3_next.py:76`). Older FLA `FusedRMSNormGated`
variants gate first, which is why this needs recording: a reader who reaches for the
wrong upstream finds a form that disagrees with both llama.cpp and current
transformers. Folding the gate in before the norm would change the statistic the norm
divides by, so it is not a reordering that washes out (2026-07-28).
