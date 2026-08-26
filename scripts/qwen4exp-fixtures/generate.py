#!/usr/bin/env python3
"""Generate golden fixtures for the qwen4exp port's new components (P1).

Runs the HuggingFace transformers reference implementation
(`transformers.models.qwen4_exp`, installed from git main) on tiny seeded
configs and dumps exact f32 inputs/outputs as JSON under
tests/fixtures/qwen4exp/. The Rust CPU reference implementations are tested
against these files.

See README.md next to this script for the venv recipe. Deterministic:
torch.manual_seed throughout; every weight used is dumped into the fixture, so
consumers never need to reproduce torch's RNG.

Precision protocol: all torch runs are f32 (that IS the reference — HF's
RMSNorm internally computes in f32 even under a f64 module, so a "f64 torch
run" would not be a cleaner oracle). For each continuous fixture a numpy f64
replica of the same math is evaluated and the max |f32 - f64| delta recorded
as `f64_delta_*` — the evaluation-order noise floor a bit-faithful f32
reimplementation should land well inside.
"""

import json
import math
import os
import sys
from datetime import date, datetime, timezone

import numpy as np
import torch
import torch.nn.functional as F

import transformers
from transformers.models.qwen4_exp.modular_qwen4_exp import (
    Qwen4ExpTextConfig,
    Qwen4ExpTextGatedResidual,
    Qwen4ExpTextPLELayer,
    Qwen4ExpTextQSAIndexer,
    Qwen4ExpTextRMSNorm,
    Qwen4ExpTextRMSNormGated,
    Qwen4ExpTextRotaryEmbedding,
    apply_rotary_pos_emb,
)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(REPO, "tests", "fixtures", "qwen4exp")

RMS_EPS = 1e-6


def transformers_commit() -> str:
    import pathlib

    site = pathlib.Path(transformers.__file__).parent.parent
    for d in site.glob("transformers-*.dist-info"):
        du = d / "direct_url.json"
        if du.exists():
            info = json.load(open(du))
            return info.get("vcs_info", {}).get("commit_id", "unknown")
    return "unknown"


META = {
    "transformers_version": transformers.__version__,
    "transformers_commit": transformers_commit(),
    "torch_version": torch.__version__,
    "generator": "scripts/qwen4exp-fixtures/generate.py",
    "generated": date.today().isoformat(),
    "dtype": "f32 (values are exact: JSON numbers are the shortest f64 repr of each f32)",
}


def tiny_config() -> Qwen4ExpTextConfig:
    return Qwen4ExpTextConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        num_experts=4,
        num_experts_per_tok=2,
        hc_count=4,
        hc_lowrank=8,
        ple_layer_ids=[2],  # one-indexed -> decoder layer 1
        ple_embed_dim=32,
        ple_conv_kernel_size=4,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=97,
        make_ngram_vocab_size_divisible_by=128,
        seed=1234,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=8,
        indexer_budget=8,
        indexer_compress_ratio=4,
        output_gate_type="sigmoid",
        eos_token_id=7,  # tiny-config analog of 248044
        pad_token_id=None,
        bos_token_id=None,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25,  # head_dim 16 -> rotary_dim 4
            "mrope_section": [1, 1, 0],
        },
        max_position_embeddings=64,
    )


def t(x: torch.Tensor):
    """Tensor -> nested lists of exact floats/ints."""
    return x.detach().cpu().tolist()


def seeded_uniform(shape, lo, hi, gen):
    return torch.empty(shape).uniform_(lo, hi, generator=gen)


def randomize(module: torch.nn.Module, gen, norm_range=(-0.25, 0.25), lin_range=(-0.5, 0.5)):
    """Fill every parameter deterministically. Norm-style params (1-D 'weight'
    of a norm module) get norm_range; everything else lin_range."""
    with torch.no_grad():
        for name, p in module.named_parameters():
            is_norm = name.endswith("weight") and p.ndim == 1 and "proj" not in name and "conv" not in name
            lo, hi = norm_range if is_norm else lin_range
            p.copy_(torch.empty_like(p).uniform_(lo, hi, generator=gen))


# ---------------------------------------------------------------- numpy f64 replicas


def np_rmsnorm_grouped(x, hf_weight, group_size, eps=RMS_EPS):
    """HF Qwen4ExpTextRMSNorm in f64: grouped rms stats, multiply by (1+w)."""
    x = x.astype(np.float64)
    orig = x.shape
    if group_size is not None:
        x = x.reshape(*orig[:-1], -1, group_size)
    out = x / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)
    out = out.reshape(orig)
    return out * (1.0 + hf_weight.astype(np.float64))


def np_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def np_silu(x):
    return x * np_sigmoid(x)


def np_gated_residual(stream, w, hc_count, hidden, use_combine=True):
    """f64 replica of Qwen4ExpTextGatedResidual.forward."""
    n = np_rmsnorm_grouped(stream, w["hc_norm"], hidden)
    mix = np_silu(n @ w["down"].astype(np.float64).T / hc_count)
    mix = np_sigmoid(mix @ w["up"].astype(np.float64).T)
    mix = mix.reshape(*mix.shape[:-1], hc_count, hidden)
    mixed = (mix * n.reshape(*n.shape[:-1], hc_count, hidden)).mean(axis=-2)
    if not use_combine:
        return mixed, None
    inject = 2.0 * np_sigmoid(n @ w["inject"].astype(np.float64).T / hc_count)
    return mixed, inject


# ---------------------------------------------------------------- fixture 1: hyper-connections


def gen_gated_residual(cfg):
    gen = torch.Generator().manual_seed(41)
    gr = Qwen4ExpTextGatedResidual(cfg)
    randomize(gr, gen)
    hc, hid = cfg.hc_count, cfg.hidden_size
    wide = hc * hid

    stream = seeded_uniform((1, 3, wide), -1.5, 1.5, gen)
    with torch.no_grad():
        mixed, hyper_input, inject = gr(stream)
        # Decoder-layer write-back with an identity block (block output == its
        # input `mixed`), mirroring modular_qwen4_exp.py lines 825-826 exactly.
        injection = mixed.unsqueeze(-2) * inject.unsqueeze(-1)
        stream_out = hyper_input + injection.flatten(-2)

    assert torch.equal(hyper_input, stream), "write-back must anchor on the raw un-normed stream"

    # Tail mixer (use_combine=False): separate instance, separate weights.
    tail = Qwen4ExpTextGatedResidual(cfg, use_combine=False)
    randomize(tail, gen)
    tail_in = seeded_uniform((1, 2, wide), -1.0, 1.0, gen)
    with torch.no_grad():
        tail_mixed = tail(tail_in)
    assert isinstance(tail_mixed, torch.Tensor) and tail_mixed.shape == (1, 2, hid)

    # f64 noise floor.
    wnp = {
        "hc_norm": gr.hc_norm.weight.detach().numpy(),
        "down": gr.input_mix_weight_down.weight.detach().numpy(),
        "up": gr.input_mix_weight_up.weight.detach().numpy(),
        "inject": gr.block_inject_weight.weight.detach().numpy(),
    }
    m64, i64 = np_gated_residual(stream.numpy(), wnp, hc, hid)
    inj64 = m64[..., None, :] * i64[..., :, None]
    s64 = stream.numpy().astype(np.float64) + inj64.reshape(*inj64.shape[:-2], -1)
    floor_mixed = float(np.abs(mixed.numpy() - m64).max())
    floor_inject = float(np.abs(inject.numpy() - i64).max())
    floor_stream = float(np.abs(stream_out.numpy() - s64).max())

    wnp_tail = {
        "hc_norm": tail.hc_norm.weight.detach().numpy(),
        "down": tail.input_mix_weight_down.weight.detach().numpy(),
        "up": tail.input_mix_weight_up.weight.detach().numpy(),
    }
    tm64, _ = np_gated_residual(tail_in.numpy(), wnp_tail, hc, hid, use_combine=False)
    floor_tail = float(np.abs(tail_mixed.numpy() - tm64).max())

    return {
        "meta": META,
        "what": "Qwen4ExpTextGatedResidual: HC read (grouped norm, silu(down/hc_count), "
        "sigmoid(up), mean over streams), 2*sigmoid(inject/hc_count), and the "
        "decoder-layer write-back onto the RAW un-normed stream with an identity block.",
        "config": {"hidden_size": hid, "hc_count": hc, "hc_lowrank": cfg.hc_lowrank, "rms_norm_eps": RMS_EPS},
        "weights": {
            "hc_norm_weight_hf": t(gr.hc_norm.weight),
            "hc_norm_weight_mult": t(gr.hc_norm.weight + 1.0),
            "input_mix_weight_down": t(gr.input_mix_weight_down.weight),
            "input_mix_weight_up": t(gr.input_mix_weight_up.weight),
            "block_inject_weight": t(gr.block_inject_weight.weight),
        },
        "input_stream": t(stream[0]),
        "mixed_output": t(mixed[0]),
        "injection_weights": t(inject[0]),
        "stream_out_identity_block": t(stream_out[0]),
        "tail_mixer": {
            "what": "hyper_connection_mixer (use_combine=False): read path only, no inject head.",
            "weights": {
                "hc_norm_weight_hf": t(tail.hc_norm.weight),
                "hc_norm_weight_mult": t(tail.hc_norm.weight + 1.0),
                "input_mix_weight_down": t(tail.input_mix_weight_down.weight),
                "input_mix_weight_up": t(tail.input_mix_weight_up.weight),
            },
            "input_stream": t(tail_in[0]),
            "mixed_output": t(tail_mixed[0]),
        },
        "f64_delta_mixed": floor_mixed,
        "f64_delta_injection": floor_inject,
        "f64_delta_stream_out": floor_stream,
        "f64_delta_tail_mixed": floor_tail,
        "tolerance_note": "bit-faithful f32 reimplementation: expect <= ~8x the f64_delta floors; "
        "suggested test tolerance max(1e-6, 10 * floor) per tensor, elementwise abs.",
    }


# ---------------------------------------------------------------- fixture 2: PLE


def gen_ple(cfg):
    gen = torch.Generator().manual_seed(42)
    ple = Qwen4ExpTextPLELayer(cfg, layer_idx=1, ple_layer_index=0)
    randomize(ple, gen)
    # conv weight was caught by the 'conv' exclusion in randomize -> got lin_range; fine.
    ng = ple.ple_embedding
    eos = ng.eos_token_id
    assert eos == 7

    hc, hid = cfg.hc_count, cfg.hidden_size
    wide = hc * hid

    # --- standalone hash fixture -------------------------------------------
    # eos mid-sequence (positions 2 and 7) exercises segment resets; with no
    # cache, previous_context is [eos, eos] which is the position-0 padding rule.
    hash_ids = torch.tensor([[3, 12, 7, 5, 9, 60, 2, 7, 1, 4]])
    history = torch.cat([torch.full((1, ng.context_len), eos), hash_ids], dim=-1)
    with torch.no_grad():
        shift1 = ng._shift_right_ignore_eos(history, 1)
        shift2 = ng._shift_right_ignore_eos(history, 2)
    captured = {}

    def grab(module, args):
        captured["ngram_ids"] = args[0].detach().clone()

    hook = ng.ngram_embedding.register_forward_pre_hook(grab)
    with torch.no_grad():
        emb_hash = ng(hash_ids, None)
    hash_rows = captured["ngram_ids"]  # [1, 10, 4] rows into the padded table
    assert hash_rows.shape == (1, hash_ids.shape[1], ng.ngram_heads)

    # --- full PLE layer forward --------------------------------------------
    ids = torch.tensor([[11, 3, 42, 8, 19, 7, 33, 2, 57, 5, 21, 40]])  # eos at position 5
    hidden = seeded_uniform((1, ids.shape[1], wide), -1.0, 1.0, gen)
    with torch.no_grad():
        out = ple(hidden, ids, None, conv_mask=None)
        layer_hash_rows = captured["ngram_ids"].clone()

        # Replica of forward (same modules, same op order) to expose the
        # intermediates; asserted bit-identical to the module output.
        embeddings = ng(ids, None)
        key_normed = ple.norm_key(ple.key_proj(embeddings)).unflatten(-1, (hc, hid))
        value = ple.value_proj(embeddings)
        query_normed = ple.norm_query(hidden).unflatten(-1, (hc, hid))
        gate_raw = (key_normed * query_normed).sum(dim=-1, keepdim=True) / math.sqrt(hid)
        gate = gate_raw.abs().clamp_min(1e-6).sqrt() * gate_raw.sign()
        gated_value = torch.sigmoid(gate) * value.unsqueeze(-2)
        gated_value_normed = ple.norm_conv(gated_value.flatten(-2))
        gated_value_flat = gated_value.flatten(-2)
        conv_out = ple._short_conv(gated_value_normed, None)
        replica = gated_value_flat + conv_out
    assert torch.equal(replica, out), "intermediate replica must be bit-identical to module forward"
    hook.remove()

    # Scalar probe for the exact gate function of modular line 770, including
    # the |s| >= 1e-6 clamp region (present in HF at this commit).
    probe_s = torch.tensor([-2.0, -0.5, -1e-6, -1e-8, 0.0, 1e-8, 1e-6, 1e-3, 0.5, 2.0])
    with torch.no_grad():
        probe_out = torch.sigmoid(probe_s.abs().clamp_min(1e-6).sqrt() * probe_s.sign())

    # f64 floor for the gate/injection path (embedding rows exact, so start there).
    emb64 = embeddings.numpy().astype(np.float64)
    kp = ple.key_proj.weight.detach().numpy().astype(np.float64)
    vp = ple.value_proj.weight.detach().numpy().astype(np.float64)
    key64 = np_rmsnorm_grouped(emb64 @ kp.T, ple.norm_key.weight.detach().numpy(), hid)
    key64 = key64.reshape(1, -1, hc, hid)
    val64 = emb64 @ vp.T
    q64 = np_rmsnorm_grouped(hidden.numpy(), ple.norm_query.weight.detach().numpy(), hid).reshape(1, -1, hc, hid)
    g64 = (key64 * q64).sum(-1, keepdims=True) / math.sqrt(hid)
    g64 = np.sqrt(np.maximum(np.abs(g64), 1e-6)) * np.sign(g64)
    gv64 = np_sigmoid(g64) * val64[:, :, None, :]
    gvn64 = np_rmsnorm_grouped(gv64.reshape(1, -1, wide), ple.norm_conv.weight.detach().numpy(), hid)
    # depthwise dilated causal conv, f64
    cw = ple.conv1d.weight.detach().numpy().astype(np.float64)  # [wide, 1, k]
    k, dil = cfg.ple_conv_kernel_size, cfg.ngram_size
    pad = (k - 1) * dil
    xpad = np.concatenate([np.zeros((1, pad, wide)), gvn64], axis=1)
    conv64 = np.zeros_like(gvn64)
    for tap in range(k):
        conv64 += xpad[:, tap * dil : tap * dil + gvn64.shape[1], :] * cw[None, None, :, 0, tap].reshape(1, 1, wide)
    out64 = gv64.reshape(1, -1, wide) + np_silu(conv64)
    floor_out = float(np.abs(out.numpy() - out64).max())
    floor_gate = float(np.abs(gate.numpy().squeeze(-1) - g64.squeeze(-1)).max())
    floor_conv = float(np.abs(conv_out.numpy() - np_silu(conv64)).max())

    return {
        "meta": META,
        "what": "PLE: n-gram hash rows (shift-right-ignore-eos over raw ids), 16-head "
        "analog table gather, signed-sqrt gate (with HF's |s|>=1e-6 clamp), per-stream "
        "sigmoid gating of the shared value, and the dilated depthwise conv residual.",
        "config": {
            "hidden_size": hid,
            "hc_count": hc,
            "ple_embed_dim": cfg.ple_embed_dim,
            "ngram_size": cfg.ngram_size,
            "heads_per_ngram": cfg.heads_per_ngram,
            "ngram_heads": ng.ngram_heads,
            "head_dim_per_ngram": cfg.ple_embed_dim // ng.ngram_heads,
            "ngram_vocab_size_base": cfg.ngram_vocab_size_base,
            "make_ngram_vocab_size_divisible_by": cfg.make_ngram_vocab_size_divisible_by,
            "seed": cfg.seed,
            "vocab_size": cfg.vocab_size,
            "eos_token_id": eos,
            "eos_note": "tiny analog of 248044; segments reset AT eos, shift never crosses it",
            "ple_conv_kernel_size": k,
            "conv_dilation": dil,
            "conv_state_len": ple.short_conv_state_len,
            "rms_norm_eps": RMS_EPS,
            "head_vocab_sizes": ng.head_vocab_sizes,
            "head_offsets": ng.head_offsets,
            "total_vocab_size": ng.total_vocab_size,
            "padded_vocab_size": ng.ngram_embedding.weight.shape[0],
            "layer_multipliers_i64_str": [str(v) for v in ng.layer_multipliers.tolist()],
        },
        "hash_case": {
            "what": "host-side hash standalone: no cache -> history padded with [eos, eos] "
            "(position-0 padding); eos at positions 2 and 7 exercises segment resets.",
            "input_ids": t(hash_ids[0]),
            "token_history": t(history[0]),
            "shift1_of_history": t(shift1[0]),
            "shift2_of_history": t(shift2[0]),
            "row_indices": t(hash_rows[0]),
            "row_indices_note": "[seq, 4 heads]: heads 0-1 are 2-gram, heads 2-3 are 3-gram; "
            "row = (t0*m0 ^ t1*m1 (^ t2*m2)) mod head_vocab + head_offset, over raw ids, i64",
        },
        "weights": {
            "ngram_embedding_table": t(ng.ngram_embedding.weight),
            "key_proj": t(ple.key_proj.weight),
            "value_proj": t(ple.value_proj.weight),
            "norm_key_weight_hf": t(ple.norm_key.weight),
            "norm_key_weight_mult": t(ple.norm_key.weight + 1.0),
            "norm_query_weight_hf": t(ple.norm_query.weight),
            "norm_query_weight_mult": t(ple.norm_query.weight + 1.0),
            "norm_conv_weight_hf": t(ple.norm_conv.weight),
            "norm_conv_weight_mult": t(ple.norm_conv.weight + 1.0),
            "conv1d_weight": t(ple.conv1d.weight.squeeze(1)),
        },
        "layer_case": {
            "what": "full PLE layer forward, no cache, eos at position 5",
            "input_ids": t(ids[0]),
            "hash_row_indices": t(layer_hash_rows[0]),
            "hidden_stream_in": t(hidden[0]),
            "ngram_embeddings": t(embeddings[0]),
            "gate_raw_dot": t(gate_raw[0].squeeze(-1)),
            "gate_signed_sqrt": t(gate[0].squeeze(-1)),
            "gated_value": t(gated_value_flat[0]),
            "gated_value_normed": t(gated_value_normed[0]),
            "conv_out_silu": t(conv_out[0]),
            "output": t(out[0]),
            "output_note": "caller adds this to the hyper-connection stream BEFORE the attn HC read",
        },
        "gate_function_probe": {
            "what": "exact scalar gate map s -> sigmoid(sign(s)*sqrt(max(|s|,1e-6))) "
            "(modular line 770; HF DOES clamp at this commit)",
            "s": t(probe_s),
            "sigmoid_gate": t(probe_out),
        },
        "f64_delta_output": floor_out,
        "f64_delta_gate": floor_gate,
        "f64_delta_conv": floor_conv,
        "tolerance_note": "hash rows and shifts are exact integers (must match exactly); "
        "float tensors: suggested tolerance max(1e-6, 10 * matching f64_delta floor), elementwise abs.",
    }


# ---------------------------------------------------------------- fixture 3: QSA indexer


def indexer_selected_sets(mask: torch.Tensor, seq: int):
    """Per-query sorted selected token indices from the returned bool mask."""
    return [mask[0, 0, q].nonzero().flatten().tolist() for q in range(seq)]


def _topk_margins(idx, rot, hidden, ratio):
    """Score-gap between last selected and best rejected block, per query with
    more blocks than block_topk. Same op order as the module's forward."""
    seq = hidden.shape[1]
    pos = torch.arange(seq).view(1, 1, -1).expand(3, 1, -1)
    cos, sin = rot(hidden, pos)
    margins = []
    with torch.no_grad():
        qk = idx.index_qk_proj(hidden)
        q_all = qk[..., : idx.index_n_heads * idx.index_head_dim].reshape(1, seq, -1, idx.index_head_dim)
        q_all = apply_rotary_pos_emb(idx.q_layernorm(q_all), cos=cos, sin=sin, unsqueeze_dim=2)
        keys = qk[..., idx.index_n_heads * idx.index_head_dim :]
        for qi in range(seq):
            nblocks = (qi + 1) // ratio
            if nblocks <= idx.block_topk:
                continue
            block_tok = torch.arange(nblocks * ratio).view(nblocks, ratio)
            pooled = idx.k_layernorm(keys[0, : nblocks * ratio].view(nblocks, ratio, -1).float().mean(dim=1))
            starts = block_tok[:, 0]
            bk = apply_rotary_pos_emb(
                pooled.unsqueeze(1), cos=cos[0].index_select(0, starts), sin=sin[0].index_select(0, starts)
            ).squeeze(1)
            sc = torch.matmul(q_all[0, qi].float(), bk.float().transpose(-1, -2)).transpose(-1, -2)
            sc = torch.relu(sc).sum(dim=-1) / math.sqrt(idx.index_head_dim)
            srt = sc.sort(descending=True).values
            margins.append(float(srt[idx.block_topk - 1] - srt[idx.block_topk]))
    return margins


def gen_qsa(cfg):
    gen = torch.Generator().manual_seed(43)
    idx = Qwen4ExpTextQSAIndexer(cfg, layer_idx=3)
    randomize(idx, gen)
    rot = Qwen4ExpTextRotaryEmbedding(cfg)
    ratio, budget = cfg.indexer_compress_ratio, cfg.indexer_budget

    def run(seq: int, hidden: torch.Tensor):
        pos = torch.arange(seq).view(1, 1, -1).expand(3, 1, -1)
        cos, sin = rot(hidden, pos)
        causal = torch.tril(torch.ones(seq, seq, dtype=torch.bool)).view(1, 1, seq, seq)
        with torch.no_grad():
            mask = idx(hidden, (cos, sin), causal, None)
            qk = idx.index_qk_proj(hidden)
            raw_keys = qk[..., idx.index_n_heads * idx.index_head_dim :]
        return hidden, raw_keys, causal, mask, (cos, sin)

    # Case A: seq 8 == budget -> every query below/at budget -> exactly dense.
    hidden_a, keys_a, causal_a, mask_a, _ = run(8, seeded_uniform((1, 8, cfg.hidden_size), -1.0, 1.0, gen))
    assert torch.equal(mask_a, causal_a), "below budget must equal dense attention"

    # Case B/C: seq 16. Query 14 sees 15 tokens (3 complete blocks > topk 2,
    # tail len 3 == ratio-1). Query 12 sees 13 (tail len 1, SHORT). Query 15
    # sees 16 (tail len 0 — no tail, exactly budget tokens selected).
    #
    # The hidden data's seed is SEARCHED so that every query's top-k sits a
    # clear margin away from a score tie (relu'd scores tie at 0.0 easily);
    # a tie would make the fixture depend on torch.topk's tie-breaking.
    hidden_b = None
    for cand in range(200):
        g2 = torch.Generator().manual_seed(4300 + cand)
        h = seeded_uniform((1, 16, cfg.hidden_size), -1.0, 1.0, g2)
        if min(m for m in _topk_margins(idx, rot, h, ratio)) > 0.05:
            hidden_b, data_seed = h, 4300 + cand
            break
    assert hidden_b is not None, "no tie-free seed found"
    _, keys_b, causal_b, mask_b, (cos_b, sin_b) = run(16, hidden_b)
    sel_b = indexer_selected_sets(mask_b, 16)

    # Replica of the selection math (f32, same ops) to dump per-query block
    # scores; must reproduce the module's selected sets.
    with torch.no_grad():
        qk = idx.index_qk_proj(hidden_b)
        q_all = qk[..., : idx.index_n_heads * idx.index_head_dim].reshape(1, 16, -1, idx.index_head_dim)
        q_all = idx.q_layernorm(q_all)
        q_all = apply_rotary_pos_emb(q_all, cos=cos_b, sin=sin_b, unsqueeze_dim=2)
        scores_per_query = []
        margins = []
        for qi in range(16):
            visible = qi + 1
            nblocks = visible // ratio
            if nblocks == 0:
                scores_per_query.append([])
                continue
            block_tok = torch.arange(nblocks * ratio).view(nblocks, ratio)
            kg = keys_b[0].index_select(0, block_tok.flatten()).view(nblocks, ratio, -1)
            pooled = kg.float().mean(dim=1).to(keys_b.dtype)
            pooled = idx.k_layernorm(pooled)
            starts = block_tok[:, 0]
            bk = apply_rotary_pos_emb(
                pooled.unsqueeze(1), cos=cos_b[0].index_select(0, starts), sin=sin_b[0].index_select(0, starts)
            ).squeeze(1)
            sc = torch.matmul(q_all[0, qi].float(), bk.float().transpose(-1, -2)).transpose(-1, -2)
            sc = torch.relu(sc).sum(dim=-1) / math.sqrt(idx.index_head_dim)
            scores_per_query.append(sc.tolist())
            ntop = min(idx.block_topk, nblocks)
            chosen = sc.topk(ntop, dim=0).indices
            expect = set(block_tok.index_select(0, chosen).flatten().tolist())
            tail = list(range(nblocks * ratio, visible))
            expect |= set(tail)
            assert expect == set(sel_b[qi]), f"replica selection mismatch at query {qi}"
            if nblocks > ntop:
                srt = sc.sort(descending=True).values
                margins.append(float(srt[ntop - 1] - srt[ntop]))
    tie_margin = min(margins) if margins else None

    # The open-question probes, stated as data:
    q12, q14, q15 = sel_b[12], sel_b[14], sel_b[15]
    assert len(q12) == budget + 1, "short tail (1 token): budget + 1 tokens, NOT budget + ratio - 1"
    assert len(q14) == budget + ratio - 1
    assert len(q15) == budget, "no tail: exactly budget tokens"
    assert 12 in q12 and 14 in q14  # tail (incl. the query token) always visible

    return {
        "meta": META,
        "what": "QSA indexer: raw cached keys, fp32 block mean -> k_layernorm -> rope at "
        "block-first position, relu-sum-over-heads scores, whole-block top-k + raw tail. "
        "Selection count = min(block_topk, nblocks)*ratio + tail_len — the tail is NOT "
        "padded out of the 513th block (divergence from llama.cpp PR #27742).",
        "config": {
            "hidden_size": cfg.hidden_size,
            "indexer_n_heads": cfg.indexer_n_heads,
            "indexer_kv_heads": cfg.indexer_kv_heads,
            "indexer_head_dim": cfg.indexer_head_dim,
            "indexer_budget": budget,
            "indexer_compress_ratio": ratio,
            "block_topk": idx.block_topk,
            "rms_norm_eps": RMS_EPS,
            "rope_theta": 10000.0,
            "rotary_dim": 4,
            "rotary_note": "partial NEoX over the first 4 of 8 indexer dims (analog of 64 of 128); "
            "inv_freq = [1.0, 0.01]; text-only mrope == plain NEoX",
            "case_above_budget_data_seed": data_seed,
        },
        "weights": {
            "index_qk_proj": t(idx.index_qk_proj.weight),
            "index_qk_proj_note": "[ (n_heads+1)*head_dim = 24, hidden 32 ]; rows 0..15 q (2 heads), rows 16..23 k",
            "q_layernorm_weight_hf": t(idx.q_layernorm.weight),
            "q_layernorm_weight_mult": t(idx.q_layernorm.weight + 1.0),
            "k_layernorm_weight_hf": t(idx.k_layernorm.weight),
            "k_layernorm_weight_mult": t(idx.k_layernorm.weight + 1.0),
        },
        "case_below_budget": {
            "what": "seq 8 == budget: selection equals dense causal attention (asserted)",
            "hidden_states": t(hidden_a[0]),
            "raw_keys": t(keys_a[0]),
            "selected_equals_causal": True,
            "selected_token_indices": indexer_selected_sets(mask_a, 8),
        },
        "case_above_budget": {
            "what": "seq 16: query 12 -> tail 1 (SHORT: 9 tokens selected), query 14 -> "
            "tail 3 = ratio-1 (11 tokens), query 15 -> tail 0 (8 tokens)",
            "hidden_states": t(hidden_b[0]),
            "raw_keys": t(keys_b[0]),
            "selected_token_indices": sel_b,
            "selected_counts": [len(s) for s in sel_b],
            "block_scores_per_query": scores_per_query,
            "min_topk_margin": tie_margin,
            "margin_note": "smallest score gap between the last selected and best rejected "
            "block across queries — selection is discrete and must match exactly; this margin "
            "shows how far from a tie the seeded data sits.",
        },
        "tolerance_note": "selected_token_indices are exact sets (order-insensitive). raw_keys/scores: "
        "max(1e-6, 1e-5 rel). The min_topk_margin >> f32 noise, so a faithful reimplementation "
        "cannot flip the selection.",
    }


# ---------------------------------------------------------------- fixture 4: gated norm


def gen_gated_norm(cfg):
    gen = torch.Generator().manual_seed(44)
    out = {"meta": META,
           "what": "Qwen4ExpTextRMSNormGated (GDN output norm): fp32 rms norm, multiply by "
           "weight (plain — NOT 1+w; never zero-centered), then multiply by act(z). "
           "qwen4exp constructs sigmoid; silu variant included for the ZGate enum's other arm.",
           "config": {"hidden_size": 8, "eps": RMS_EPS}}
    for act in ("sigmoid", "silu"):
        norm = Qwen4ExpTextRMSNormGated(8, eps=RMS_EPS, activation=act)
        with torch.no_grad():
            norm.weight.copy_(seeded_uniform((8,), 0.5, 1.5, gen))
        o = seeded_uniform((4, 8), -2.0, 2.0, gen)
        z = seeded_uniform((4, 8), -2.0, 2.0, gen)
        with torch.no_grad():
            y = norm(o, z)
        o64 = o.numpy().astype(np.float64)
        n64 = o64 / np.sqrt(np.mean(o64**2, axis=-1, keepdims=True) + RMS_EPS)
        n64 *= norm.weight.detach().numpy().astype(np.float64)
        z64 = z.numpy().astype(np.float64)
        y64 = n64 * (np_sigmoid(z64) if act == "sigmoid" else np_silu(z64))
        out[act] = {
            "norm_weight": t(norm.weight),
            "o": t(o),
            "z": t(z),
            "output": t(y),
            "f64_delta": float(np.abs(y.numpy() - y64).max()),
        }
    out["tolerance_note"] = "suggested tolerance max(1e-6, 10 * f64_delta), elementwise abs."
    return out


# ---------------------------------------------------------------- fixture 5: grouped RMSNorm


def gen_grouped_rmsnorm(cfg):
    gen = torch.Generator().manual_seed(45)
    dim, group = 16, 4
    norm = Qwen4ExpTextRMSNorm(dim, group_size=group, eps=RMS_EPS)
    flat = Qwen4ExpTextRMSNorm(dim, group_size=None, eps=RMS_EPS)
    with torch.no_grad():
        w = seeded_uniform((dim,), -0.25, 0.25, gen)
        norm.weight.copy_(w)
        flat.weight.copy_(w)
    # Per-group scales spread over 4 decades so grouped and ungrouped stats
    # differ visibly (max elementwise relative difference ~1e2).
    x = seeded_uniform((3, dim), -1.0, 1.0, gen)
    scales = torch.tensor([100.0, 1.0, 0.1, 0.01]).repeat_interleave(group)
    x = x * scales
    with torch.no_grad():
        y = norm(x)
        y_flat = flat(x)
    rel = (y - y_flat).abs() / y.abs().clamp_min(1e-12)
    y64 = np_rmsnorm_grouped(x.numpy(), w.numpy(), group)
    return {
        "meta": META,
        "what": "Qwen4ExpTextRMSNorm with group_size: rms stats per group of `group_size`, "
        "then elementwise (1+w) over the full dim. Groups carry 4-decade scale spread so "
        "an ungrouped implementation visibly diverges (contrast output included).",
        "config": {"dim": dim, "group_size": group, "eps": RMS_EPS},
        "weight_hf": t(w),
        "weight_mult": t(w + 1.0),
        "input": t(x),
        "output": t(y),
        "output_ungrouped_for_contrast": t(y_flat),
        "max_rel_grouped_vs_ungrouped": float(rel.max()),
        "f64_delta_output": float(np.abs(y.numpy() - y64).max()),
        "tolerance_note": "suggested tolerance max(1e-6, 10 * f64_delta_output), elementwise abs.",
    }


def main():
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float32)
    cfg = tiny_config()
    os.makedirs(OUT_DIR, exist_ok=True)
    fixtures = {
        "gated_residual.json": gen_gated_residual(cfg),
        "ple.json": gen_ple(cfg),
        "qsa_indexer.json": gen_qsa(cfg),
        "gated_norm.json": gen_gated_norm(cfg),
        "grouped_rmsnorm.json": gen_grouped_rmsnorm(cfg),
    }
    for name, data in fixtures.items():
        path = os.path.join(OUT_DIR, name)
        with open(path, "w") as f:
            json.dump(data, f, separators=(",", ":"))
            f.write("\n")
        print(f"{name}: {os.path.getsize(path)} bytes")
    print("floors:")
    for name, data in fixtures.items():
        for key, val in data.items():
            if key.startswith("f64_delta"):
                print(f"  {name} {key} = {val:.3e}")
        for sub in ("sigmoid", "silu"):
            if sub in data and "f64_delta" in data[sub]:
                print(f"  {name} [{sub}] f64_delta = {data[sub]['f64_delta']:.3e}")
    qsa = fixtures["qsa_indexer.json"]["case_above_budget"]
    print("qsa selected counts:", qsa["selected_counts"])
    print("qsa min top-k margin:", qsa["min_topk_margin"])


if __name__ == "__main__":
    main()
