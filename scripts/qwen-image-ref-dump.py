#!/usr/bin/env python3
# Reference dumps for xwen's Qwen-Image 2.1 path (docs/qwen-image-2.1-plan.md). Stage 1 is
# the text encoder: the rendered prompt, its ids, the system-turn drop, and the hidden
# state the pipeline conditions on.
#
#   uv venv /tmp/qwen-image-venv --python 3.12
#   uv pip install --python /tmp/qwen-image-venv/bin/python torch 'transformers>=5.17' \
#     safetensors numpy accelerate pillow torchvision \
#     'diffusers @ git+https://github.com/huggingface/diffusers@6256aa7666cedd47443adc8f82da9a10e110b09c'
#   /tmp/qwen-image-venv/bin/python scripts/qwen-image-ref-dump.py --stage tokens
#   /tmp/qwen-image-venv/bin/python scripts/qwen-image-ref-dump.py --stage fp32 \
#     && /tmp/qwen-image-venv/bin/python scripts/qwen-image-ref-dump.py --stage bf16 \
#     && /tmp/qwen-image-venv/bin/python scripts/qwen-image-ref-dump.py --stage finalize
#   /tmp/qwen-image-venv/bin/python scripts/qwen-image-ref-dump.py --stage transformer --dtype fp32 \
#     && /tmp/qwen-image-venv/bin/python scripts/qwen-image-ref-dump.py --stage transformer --dtype bf16
#
# The same standing as scripts/zimage-ref-dump.py, which is the only other Python in the
# repo: it exists because there is no bun path to torch, it runs by hand under `uv` in a
# throwaway venv to produce the references `tests/qwen_image_encoder.rs` grades xwen
# against, and it never runs in CI.
#
# What it reproduces is diffusers' `QwenImage21Pipeline._get_qwen_prompt_embeds` at
# 6256aa7 for text-to-image:
#
#   * the prompt is a RAW TEMPLATE STRING handed to the processor, not
#     `apply_chat_template` output: a fixed system turn, the user turn, an open assistant
#     turn; an empty prompt becomes one space;
#   * the hidden state is `hidden_states[-1]` with the final RMSNorm NEUTRALISED by a
#     forward hook that returns the norm's input. transformers 5 ties that entry to
#     `last_hidden_state`, the normed output, and the checkpoint was trained on the
#     un-normed residual after all 36 layers;
#   * the rows of the system turn are dropped, their count being the token length of the
#     chat-templated system message alone.
#
# The `tokens` stage needs no weights (the processor only) and writes the committed
# `tokens.json`: rendered strings, ids and the drop index. `fp32` (cpu, eager) is the
# acceptance reference, `bf16` (cpu, sdpa) measures the reference's own rounding spread,
# and `finalize` joins the two into the committed `reference.json`. Arrays are not
# committed; they live in --out-dir, which the Rust test reads through
# $XWEN_QWEN_IMAGE_REF_DIR.
#
# Three things are asserted rather than assumed, each on every run of a weight stage:
#
#   * the script's explicit path equals the pipeline's own method, bit for bit, on one
#     prompt (`pipeline_check`), so the reference IS the pipeline and not a reading of it;
#   * the text-only position ids `get_rope_index` builds are identical on all three MRoPE
#     axes (`mrope_text_ids`), which is what lets xwen run plain NEoX rope here;
#   * the normed state differs from the un-normed one by more than the bar
#     (`normed_vs_prenorm`), and the normed state of one prompt is dumped so the Rust gate
#     can show that the wrong tensor FAILS.
#
# The encoder stages are CPU only: the fp32 arm is the reference and mps matmuls are not
# bitwise reproducible across runs.
#
# The transformer stage writes `tests/fixtures/qwen-image-transformer/<WxH>-p<idx>-s<seed>/`,
# which `tests/qwen_image_parity.rs` grades against: the injected noise and caption
# features both sides start from, the reference step-0 velocity and final latent, and the
# fp32 arm's decoded PNG. It runs the pipeline's OWN `__call__` (`prompt_embeds=`,
# `latents=`, `use_kv_cache=True`, 40 steps, no guidance), so the scheduler grid, the
# prefix cache, the Euler step and the decode are the pipeline and not a reading of it.
# Run fp32 first: it writes the inputs, and the bf16 arm reuses them so the two arms
# differ in arithmetic alone. The caption features come from the pipeline's own
# `encode_prompt` in fp32 on cpu, rounded once to bf16; the encoder is freed before the
# transformer loads, the two not fitting comfortably at once in fp32. The transformer arms
# run on mps, the fp32 arm falling back to cpu if mps cannot hold it. The VAE is f32 in
# both arms, as it is in xwen.
#
# Three conventions of that fixture, each read off `pipeline_qwenimage21.py`:
#
#   * `latents0` is stored UNPACKED, `[1, 64, H/16, W/16]`. The pipeline `latents=`
#     takes the PACKED form `[1, N, 64]` and casts it without reshaping, packing being a
#     raster flatten (`view(B, C, H*W).transpose(1, 2)`), so the script packs it that way;
#   * the noise is standard normal and is NOT scaled;
#   * `latents-final` is the latent after the last Euler step, unpacked, in the
#     NORMALISED space the transformer works in, BEFORE the `z * std + mean` the pipeline
#     applies on its way into `vae.decode`.

import argparse
import gc
import hashlib
import json
import os
import time
import unicodedata
from pathlib import Path

import numpy as np
import torch

REPO_ID = "Qwen/Qwen-Image-2.1"
REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = REPO_ROOT / "tests/fixtures/qwen-image-encoder"
TRANSFORMER_FIXTURE_DIR = REPO_ROOT / "tests/fixtures/qwen-image-transformer"
NUM_STEPS = 40
LATENT_CHANNELS = 64
PIXELS_PER_TOKEN = 16

SYSTEM_PROMPT = "Comprehend and analyze the provided prompt."
TEMPLATE = (
    f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
N_LAYERS = 36
HIDDEN_SIZE = 4096
CHECK_PROMPT_IDX = 1  # the prompt the pipeline, normed and position-id checks use


def default_snapshot() -> Path:
    snapshots = (
        Path.home()
        / ".cache/huggingface/hub"
        / f"models--{REPO_ID.replace('/', '--')}/snapshots"
    )
    found = sorted(p for p in snapshots.iterdir() if p.is_dir()) if snapshots.is_dir() else []
    assert len(found) == 1, f"expected one snapshot under {snapshots}, found {len(found)}; pass --snapshot"
    return found[0]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_prompts() -> list[dict]:
    doc = json.loads((FIXTURE_DIR / "prompts.json").read_text(encoding="utf-8"))
    prompts = doc["prompts"]
    for p in prompts:
        text = p["text"]
        assert unicodedata.is_normalized("NFC", text), f"prompt {p['idx']} is not NFC"
        assert not text.endswith("\n"), f"prompt {p['idx']} ends with a newline"
    return prompts


def load_processor(snapshot: Path):
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(snapshot / "processor", local_files_only=True)


def render(text: str) -> str:
    return TEMPLATE.format(" " if not text else text)


def drop_index(processor) -> int:
    """The pipeline's `_drop_idx`: the token length of the chat-templated system turn."""
    message = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]
    tokens = processor.apply_chat_template(message, tokenize=True, return_dict=False)
    return len(tokens[0])


def tokenize(processor, rendered: str):
    """Exactly the pipeline's processor call, for a batch of one."""
    return processor(text=[rendered], padding=True, padding_side="left", return_tensors="pt")


def token_records(processor, prompts: list[dict]) -> tuple[list[dict], int]:
    drop = drop_index(processor)
    system_ids = processor.tokenizer(
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n", add_special_tokens=False
    )["input_ids"]
    assert len(system_ids) == drop, (
        f"the chat-templated system turn is {drop} tokens but the raw template's is "
        f"{len(system_ids)}: the drop would cut the wrong rows"
    )
    records = []
    for p in prompts:
        rendered = render(p["text"])
        inputs = tokenize(processor, rendered)
        assert bool(inputs.attention_mask.all()), f"prompt {p['idx']}: a batch of one was padded"
        ids = inputs.input_ids[0].tolist()
        assert ids[:drop] == system_ids, f"prompt {p['idx']}: ids do not open with the system turn"
        records.append(
            {
                "idx": p["idx"],
                "label": p["label"],
                "dir": f"{p['idx']:02d}",
                "T": len(ids),
                "T_kept": len(ids) - drop,
                "ids": ids,
                "rendered_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
                "rendered": rendered,
            }
        )
    return records, drop


def run_tokens(snapshot: Path) -> None:
    import transformers

    processor = load_processor(snapshot)
    records, drop = token_records(processor, load_prompts())
    tokenizer_json = snapshot / "processor" / "tokenizer.json"
    doc = {
        "note": (
            "Rendered prompts and ids for tests/qwen_image_encoder.rs, from "
            "scripts/qwen-image-ref-dump.py --stage tokens. No weights are involved."
        ),
        "repo_id": REPO_ID,
        "revision": snapshot.name,
        "system_prompt": SYSTEM_PROMPT,
        "drop": drop,
        "tokenizer_json_sha256": sha256_file(tokenizer_json),
        "versions": {"transformers": transformers.__version__},
        "prompts": records,
    }
    dest = FIXTURE_DIR / "tokens.json"
    dest.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for r in records:
        print(f"[tokens] {r['idx']:2d} {r['label']:22s} T={r['T']:4d} kept={r['T_kept']:4d}", flush=True)
    print(f"[tokens] drop={drop}, wrote {dest}", flush=True)


def build_model(snapshot: Path, dtype: torch.dtype, attn_impl: str):
    from transformers import Qwen3VLForConditionalGeneration

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        snapshot / "text_encoder",
        dtype=dtype,
        local_files_only=True,
        attn_implementation=attn_impl,
    )
    model.eval()
    model.to("cpu")
    return model


def text_model(model):
    return getattr(model.model, "language_model", model.model)


def forward(model, inputs, neutralise_norm: bool):
    """The pipeline's encoder call. With `neutralise_norm` the final RMSNorm returns its
    input, which is how the pipeline reads the un-normed residual out of
    `hidden_states[-1]` on transformers 5."""
    kwargs = {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
        "output_hidden_states": True,
    }
    if hasattr(inputs, "mm_token_type_ids"):
        kwargs["mm_token_type_ids"] = inputs.mm_token_type_ids
    captured = {}

    def capture(_module, args, output):
        captured["norm_in"] = args[0].detach()
        captured["norm_out"] = output.detach()
        return args[0] if neutralise_norm else None

    handle = text_model(model).norm.register_forward_hook(capture)
    try:
        with torch.inference_mode():
            out = model(**kwargs)
    finally:
        handle.remove()
    return out.hidden_states, captured


def mrope_text_ids(model, inputs) -> dict:
    """Text-only position ids: equal on the temporal, height and width axes."""
    # All zeros is "every token is text", which is what the processor reports for a prompt
    # with no image when it reports anything.
    mm = getattr(inputs, "mm_token_type_ids", None)
    if mm is None:
        mm = torch.zeros_like(inputs.input_ids, dtype=torch.int)
    position_ids, _deltas = model.model.get_rope_index(
        input_ids=inputs.input_ids, mm_token_type_ids=mm, attention_mask=inputs.attention_mask
    )
    assert position_ids.shape[0] == 3, f"expected three position axes, got {tuple(position_ids.shape)}"
    t, h, w = position_ids[0, 0], position_ids[1, 0], position_ids[2, 0]
    seq = inputs.input_ids.shape[1]
    equal = bool(torch.equal(t, h) and torch.equal(t, w))
    arange = bool(torch.equal(t.cpu(), torch.arange(seq)))
    assert equal, "text-only MRoPE ids differ across axes; plain NEoX rope is not equivalent"
    assert arange, "text-only MRoPE ids are not 0..T-1"
    return {"prompt_idx": CHECK_PROMPT_IDX, "tokens": seq, "axes_equal": equal, "is_arange": arange}


def pipeline_check(model, processor, text: str, ours: torch.Tensor) -> dict:
    """The pipeline's own `_get_qwen_prompt_embeds` against this script's path."""
    from diffusers import QwenImage21Pipeline

    pipe = QwenImage21Pipeline(
        scheduler=None, vae=None, text_encoder=model, processor=processor, transformer=None
    )
    embeds, _mask, _pad = pipe._get_qwen_prompt_embeds(text, None, torch.device("cpu"))
    theirs = embeds[0].float()
    equal = bool(torch.equal(theirs, ours))
    assert theirs.shape == ours.shape, f"pipeline {tuple(theirs.shape)} vs script {tuple(ours.shape)}"
    assert equal, (
        "the pipeline's prompt embeds differ from this script's: max abs "
        f"{float((theirs - ours).abs().max())}"
    )
    return {"prompt_idx": CHECK_PROMPT_IDX, "bitwise_equal": equal, "drop_idx": pipe._drop_idx}


def row_metrics(x: np.ndarray, y: np.ndarray) -> dict:
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    num = (x * y).sum(axis=1)
    den = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
    cos = num / np.maximum(den, 1e-30)
    rel = np.abs(y - x).max(axis=1) / np.maximum(np.abs(x).max(axis=1), 1e-6)
    return {
        "min_cosine": float(cos.min()),
        "mean_cosine": float(cos.mean()),
        "max_rel_error": float(rel.max()),
        "mean_rel_error": float(rel.mean()),
    }


def run_stage(stage: str, snapshot: Path, out_dir: Path, attn_impl: str) -> None:
    import transformers
    from safetensors.torch import save_file

    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[stage]
    processor = load_processor(snapshot)
    prompts = load_prompts()
    tokens, drop = token_records(processor, prompts)
    committed = json.loads((FIXTURE_DIR / "tokens.json").read_text(encoding="utf-8"))
    assert committed["drop"] == drop, "tokens.json records a different drop index; rerun --stage tokens"
    assert [r["ids"] for r in committed["prompts"]] == [r["ids"] for r in tokens], (
        "tokens.json records different ids; rerun --stage tokens"
    )

    print(f"[{stage}] loading model ({dtype}, attn={attn_impl})", flush=True)
    t0 = time.time()
    model = build_model(snapshot, dtype, attn_impl)
    load_s = time.time() - t0
    resolved_attn = getattr(model.config, "_attn_implementation", attn_impl)
    print(f"[{stage}] loaded in {load_s:.1f}s, attn={resolved_attn}", flush=True)

    records = []
    checks = {}
    for p, tok in zip(prompts, tokens):
        idx = p["idx"]
        rendered = tok["rendered"]
        inputs = tokenize(processor, rendered)
        d = out_dir / tok["dir"]
        d.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        hs, captured = forward(model, inputs, neutralise_norm=True)
        wall_s = time.time() - t0
        assert len(hs) == N_LAYERS + 1, f"prompt {idx}: {len(hs)} hidden states"
        # With the norm neutralised, the last entry IS the norm's input: the residual
        # after the last layer.
        assert torch.equal(hs[-1], captured["norm_in"]), f"prompt {idx}: the hook did not take"
        hidden = hs[-1][0, drop:].float().contiguous()
        assert hidden.shape == (tok["T_kept"], HIDDEN_SIZE), f"prompt {idx}: {tuple(hidden.shape)}"

        st = d / f"hidden_{stage}.safetensors"
        save_file({"hidden": hidden}, str(st))
        rec = dict(tok)
        rec.update(
            {
                f"hidden_{stage}": st.name,
                f"hidden_{stage}_sha256": sha256_file(st),
                f"wall_s_{stage}": round(wall_s, 3),
            }
        )

        if idx == CHECK_PROMPT_IDX:
            # The tensor a port gets by reading `hidden_states[-1]` without the hook.
            normed = captured["norm_out"][0, drop:].float().contiguous()
            hs_plain, _ = forward(model, inputs, neutralise_norm=False)
            checks["hidden_states_last_is_normed_without_the_hook"] = bool(
                torch.equal(hs_plain[-1][0, drop:].float(), normed)
            )
            nst = d / f"hidden_normed_{stage}.safetensors"
            save_file({"hidden": normed}, str(nst))
            rec[f"hidden_normed_{stage}"] = nst.name
            rec[f"hidden_normed_{stage}_sha256"] = sha256_file(nst)
            checks["normed_vs_prenorm"] = row_metrics(hidden.numpy(), normed.numpy())
            checks["normed_vs_prenorm"]["rms_ratio"] = float(
                normed.pow(2).mean().sqrt() / hidden.pow(2).mean().sqrt()
            )
            checks["mrope_text_ids"] = mrope_text_ids(model, inputs)
            checks["pipeline_check"] = pipeline_check(model, processor, p["text"], hidden)
            print(f"[{stage}] checks: {json.dumps(checks)}", flush=True)

        records.append(rec)
        print(
            f"[{stage}] {idx:2d} {p['label']:22s} T={tok['T']:4d} kept={tok['T_kept']:4d} {wall_s:7.2f}s",
            flush=True,
        )

    del model
    gc.collect()

    manifest = {
        "stage": stage,
        "dtype": str(dtype),
        "attn_implementation": resolved_attn,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "model_snapshot": str(snapshot),
        "repo_id": REPO_ID,
        "revision": snapshot.name,
        "drop": drop,
        "torch_num_threads": torch.get_num_threads(),
        "load_s": round(load_s, 1),
        "checks": checks,
        "records": records,
    }
    dest = out_dir / f"manifest-{stage}.json"
    dest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[{stage}] wrote {dest}", flush=True)


def row_vectors(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-row cosine and relative error, the Rust gate's two metrics."""
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    cos = (x * y).sum(axis=1) / np.maximum(
        np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1), 1e-30
    )
    rel = np.abs(y - x).max(axis=1) / np.maximum(np.abs(x).max(axis=1), 1e-6)
    return cos, rel


def finalize(out_dir: Path) -> None:
    from safetensors.torch import load_file

    # The spread the Rust gate's bars are set from, split the way the gate splits it. Kept
    # row 0 is the user turn's `<|im_start|>`, under the same context in every prompt: it
    # carries a massive activation (norm ~9400 against ~200-700) through layers 24-34 that
    # the last layers cancel back to an ordinary ~850, so its final value is a small
    # difference of large numbers and every arithmetic is at its worst there. Pooled with
    # the other rows it would be the only thing either side's figures said.
    row0 = {"bf16": [], "normed": []}
    rest = {"bf16_cos": [], "bf16_rel": [], "normed_cos": [], "normed_rel": []}

    fp32 = json.loads((out_dir / "manifest-fp32.json").read_text(encoding="utf-8"))
    bf16 = json.loads((out_dir / "manifest-bf16.json").read_text(encoding="utf-8"))
    by_idx = {r["idx"]: r for r in bf16["records"]}

    merged = []
    for r in fp32["records"]:
        b = by_idx[r["idx"]]
        assert b["ids"] == r["ids"], f"prompt {r['idx']}: ids differ between stages"
        x = load_file(str(out_dir / r["dir"] / r["hidden_fp32"]))["hidden"].numpy()
        y = load_file(str(out_dir / r["dir"] / b["hidden_bf16"]))["hidden"].numpy()
        assert x.shape == y.shape
        cos, rel = row_vectors(x, y)
        row0["bf16"].append((float(cos[0]), float(rel[0])))
        rest["bf16_cos"] += cos[1:].tolist()
        rest["bf16_rel"] += rel[1:].tolist()
        if "hidden_normed_fp32" in r:
            normed = load_file(str(out_dir / r["dir"] / r["hidden_normed_fp32"]))["hidden"].numpy()
            ncos, nrel = row_vectors(x, normed)
            row0["normed"].append((float(ncos[0]), float(nrel[0])))
            rest["normed_cos"] += ncos[1:].tolist()
            rest["normed_rel"] += nrel[1:].tolist()
        m = {k: v for k, v in r.items() if k not in {"ids", "rendered", "rendered_sha256"}}
        m.update(
            {
                "hidden_bf16": b["hidden_bf16"],
                "hidden_bf16_sha256": b["hidden_bf16_sha256"],
                "wall_s_bf16": b["wall_s_bf16"],
                "bf16_vs_fp32": row_metrics(x, y),
            }
        )
        merged.append(m)
        s = m["bf16_vs_fp32"]
        print(
            f"[final] {m['idx']:2d} {m['label']:22s} kept={m['T_kept']:4d} "
            f"min_cos={s['min_cosine']:.8f} max_rel={s['max_rel_error']:.5f}",
            flush=True,
        )

    assert row0["normed"], "no prompt carries the normed dump the bracket is set from"
    spread_doc = {
        "note": (
            "Per-row cosine and max-abs relative error against the fp32 arm, over every "
            "prompt. `row0` is kept row 0 alone, `rest` every other kept row pooled. "
            "`bf16` is the reference's own bf16 arm (a CORRECT graph at lower precision), "
            "`normed` the wrong graph: the final norm applied."
        ),
        "rows_rest": len(rest["bf16_rel"]),
        "row0": {
            "bf16": {
                "min_cosine": min(c for c, _ in row0["bf16"]),
                "max_rel_error": max(e for _, e in row0["bf16"]),
            },
            "normed": {
                "max_cosine": max(c for c, _ in row0["normed"]),
                "min_rel_error": min(e for _, e in row0["normed"]),
            },
        },
        "rest": {
            "bf16": {
                "min_cosine": float(np.min(rest["bf16_cos"])),
                "p50_rel_error": float(np.percentile(rest["bf16_rel"], 50)),
                "p99_rel_error": float(np.percentile(rest["bf16_rel"], 99)),
                "max_rel_error": float(np.max(rest["bf16_rel"])),
            },
            "normed": {
                "max_cosine": float(np.max(rest["normed_cos"])),
                "min_rel_error": float(np.min(rest["normed_rel"])),
            },
        },
    }
    print(f"[final] spread: {json.dumps(spread_doc['row0'])} {json.dumps(spread_doc['rest'])}", flush=True)

    reference = {
        "note": (
            "Reference for tests/qwen_image_encoder.rs. Dumps are NOT committed; point "
            "$XWEN_QWEN_IMAGE_REF_DIR at the output of scripts/qwen-image-ref-dump.py. Each "
            "prompt's arrays live in <XWEN_QWEN_IMAGE_REF_DIR>/<dir>/<file>. Rendered strings "
            "and ids are in tokens.json."
        ),
        "repo_id": REPO_ID,
        "revision": fp32["revision"],
        "hidden_index": N_LAYERS,
        "final_norm": False,
        "drop": fp32["drop"],
        "hidden_size": HIDDEN_SIZE,
        "check_prompt_idx": CHECK_PROMPT_IDX,
        "versions": {
            "torch": fp32["torch"],
            "transformers": fp32["transformers"],
            "numpy": fp32["numpy"],
        },
        "stages": {
            s["stage"]: {
                "dtype": s["dtype"],
                "attn_implementation": s["attn_implementation"],
                "load_s": s["load_s"],
            }
            for s in (fp32, bf16)
        },
        "checks": fp32["checks"],
        "spread": spread_doc,
        "prompts": merged,
    }
    dest = FIXTURE_DIR / "reference.json"
    dest.write_text(json.dumps(reference, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[final] wrote {dest}", flush=True)


def case_name(width: int, height: int, prompt_idx: int, seed: int) -> str:
    return f"{width}x{height}-p{prompt_idx}-s{seed}"


def save_st(path: Path, name: str, tensor: torch.Tensor) -> None:
    from safetensors.torch import save_file

    save_file({name: tensor.detach().cpu().contiguous()}, str(path))


def load_st(path: Path, name: str) -> torch.Tensor:
    from safetensors.torch import load_file

    tensors = load_file(str(path))
    assert list(tensors) == [name], f"{path} holds {list(tensors)}, expected [{name!r}]"
    return tensors[name]


def spread(a: torch.Tensor, b: torch.Tensor) -> dict:
    """cosine over the whole field, max |a-b| / max|b|, and the mean |a-b| / rms(b)."""
    a = a.double().flatten()
    b = b.double().flatten()
    cos = torch.dot(a, b) / (a.norm() * b.norm())
    diff = (a - b).abs()
    return {
        "cosine": cos.item(),
        "max_rel": (diff.max() / b.abs().max()).item(),
        "mean_rel": (diff.mean() / b.pow(2).mean().sqrt()).item(),
        "max_abs": diff.max().item(),
    }


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """PSNR in dB between two uint8 images of the same shape."""
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(255.0**2 / mse))


def cap_feats_for(snapshot: Path, prompt_idx: int) -> tuple[torch.Tensor, dict]:
    """The rows the pipeline feeds the transformer for one fixture prompt, from its own
    `encode_prompt` in fp32 on cpu (the pre-norm hook and the system-turn drop included),
    rounded once to bf16."""
    from diffusers import QwenImage21Pipeline

    processor = load_processor(snapshot)
    prompt = next(p for p in load_prompts() if p["idx"] == prompt_idx)
    assert not prompt.get("literal_specials"), (
        f"prompt {prompt_idx} holds literal special-token text, which xwen renders "
        "differently on purpose; pick another for the transformer fixture"
    )
    rendered = render(prompt["text"])
    ids = tokenize(processor, rendered).input_ids[0].tolist()
    print("[cap] loading the encoder (fp32, cpu, eager)", flush=True)
    model = build_model(snapshot, torch.float32, "eager")
    pipe = QwenImage21Pipeline(
        scheduler=None, vae=None, text_encoder=model, processor=processor, transformer=None
    )
    with torch.inference_mode():
        embeds, mask, pad_mask = pipe.encode_prompt(prompt=prompt["text"], device=torch.device("cpu"))
    assert mask is None, "a batch of one has no padding and the pipeline returns no mask"
    assert not bool(pad_mask.any()), "a text-only prompt has no image slots"
    hidden = embeds[0].float().contiguous()
    drop = pipe._drop_idx
    assert hidden.shape == (len(ids) - drop, HIDDEN_SIZE), (tuple(hidden.shape), len(ids), drop)
    del pipe, model, embeds
    gc.collect()
    info = {
        "prompt_idx": prompt_idx,
        "label": prompt["label"],
        "text": prompt["text"],
        "ids": ids,
        "drop_idx": drop,
        "T": hidden.shape[0],
        "encoder_dtype": "fp32 on cpu through the pipeline's encode_prompt, rounded once to bf16",
    }
    return hidden.to(torch.bfloat16), info


def run_transformer(
    snapshot: Path, out_root: Path, dtype_name: str, width: int, height: int,
    prompt_idx: int, seed: int, steps: int,
) -> None:
    import diffusers
    import PIL.Image
    from diffusers import (
        AutoencoderKLQwenImage21, FlowMatchEulerDiscreteScheduler, QwenImage21Pipeline,
        QwenImage21Transformer2DModel,
    )
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift

    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[dtype_name]
    assert width % 32 == 0 and height % 32 == 0, "sides must be multiples of 32"
    lat_h, lat_w = height // PIXELS_PER_TOKEN, width // PIXELS_PER_TOKEN
    tokens = lat_h * lat_w
    case = case_name(width, height, prompt_idx, seed)
    out = out_root / "transformer" / case
    out.mkdir(parents=True, exist_ok=True)
    tag = f"[{dtype_name}]"

    latents_path = out / "latents0.safetensors"
    cap_path = out / "cap_feats.safetensors"
    inputs_path = out / "inputs.json"
    if dtype_name == "fp32":
        cap_feats, cap_info = cap_feats_for(snapshot, prompt_idx)
        # torch's mps generator is not reproducible from Rust; the draw is on cpu and the
        # tensor is the fixture, the seed only labels it.
        gen = torch.Generator("cpu").manual_seed(seed)
        latents0 = torch.randn((1, LATENT_CHANNELS, lat_h, lat_w), generator=gen, dtype=torch.float32)
        save_st(latents_path, "latents", latents0)
        save_st(cap_path, "cap_feats", cap_feats)
        inputs_path.write_text(json.dumps(cap_info, indent=2, ensure_ascii=False) + "\n")
    else:
        assert latents_path.is_file() and cap_path.is_file(), "run --dtype fp32 first"
        latents0 = load_st(latents_path, "latents")
        cap_feats = load_st(cap_path, "cap_feats")
        cap_info = json.loads(inputs_path.read_text())
    assert latents0.shape == (1, LATENT_CHANNELS, lat_h, lat_w) and latents0.dtype == torch.float32
    assert cap_feats.dtype == torch.bfloat16 and cap_feats.shape[1] == HIDDEN_SIZE

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        snapshot / "scheduler", local_files_only=True
    )
    assert scheduler.config.use_dynamic_shifting is True
    assert scheduler.config.time_shift_type == "exponential"
    vae = AutoencoderKLQwenImage21.from_pretrained(
        snapshot / "vae", torch_dtype=torch.float32, local_files_only=True
    ).eval()

    def load_on(device: torch.device):
        print(f"{tag} loading the transformer ({dtype}) onto {device}", flush=True)
        t0 = time.time()
        model = QwenImage21Transformer2DModel.from_pretrained(
            snapshot / "transformer", torch_dtype=dtype, local_files_only=True
        )
        model.eval().to(device)
        print(f"{tag} loaded in {time.time() - t0:.1f}s", flush=True)
        return model

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    try:
        transformer = load_on(device)
    except RuntimeError as error:
        if device.type == "cpu" or dtype_name != "fp32":
            raise
        print(f"{tag} mps refused the fp32 transformer ({error}); falling back to cpu", flush=True)
        torch.mps.empty_cache()
        device = torch.device("cpu")
        transformer = load_on(device)
    vae.to(device)

    # The constructor reads the drop index off the processor, which needs no weights.
    pipe = QwenImage21Pipeline(
        scheduler=scheduler, vae=vae, text_encoder=None, processor=load_processor(snapshot),
        transformer=transformer,
    )
    pipe.set_progress_bar_config(disable=True)

    # The transformer's first output is the step-0 velocity over the joint sequence; the
    # pipeline keeps its last N rows, and so does this.
    captured = {"calls": 0, "step_s": [], "t0": time.time()}

    def on_forward(_module, _args, output):
        if captured["calls"] == 0:
            captured["velocity0"] = output[0][:, -tokens:].detach().float().cpu().clone()
        captured["calls"] += 1

    def on_step_end(_pipe, i, _t, kwargs):
        captured["final"] = kwargs["latents"]
        if device.type == "mps":
            torch.mps.synchronize()
        now = time.time()
        captured["step_s"].append(now - captured["t0"])
        captured["t0"] = now
        print(f"{tag} step {i} {captured['step_s'][-1]:.2f}s", flush=True)
        return {}

    handle = transformer.register_forward_hook(on_forward)
    packed = QwenImage21Pipeline._pack_latents(latents0, 1, LATENT_CHANNELS, lat_h, lat_w)
    # What the pipeline call below is handed, and what the arm file records of it.
    use_kv_cache = True
    try:
        with torch.inference_mode():
            result = pipe(
                prompt_embeds=cap_feats.unsqueeze(0).to(device).to(dtype),
                latents=packed.to(device),
                height=height,
                width=width,
                num_inference_steps=steps,
                true_cfg_scale=1.0,
                use_kv_cache=use_kv_cache,
                output_type="pil",
                callback_on_step_end=on_step_end,
                callback_on_step_end_tensor_inputs=["latents"],
            )
    finally:
        handle.remove()
    assert captured["calls"] == steps, f"{captured['calls']} forwards for {steps} steps"
    image = result.images[0]
    assert image.mode == "RGBA" and image.size == (width, height), (image.mode, image.size)
    image_u8 = np.asarray(image)

    sigmas = scheduler.sigmas.float().cpu().tolist()
    assert len(sigmas) == steps + 1 and sigmas[-1] == 0.0, sigmas
    mu = calculate_shift(
        tokens,
        scheduler.config.base_image_seq_len,
        scheduler.config.max_image_seq_len,
        scheduler.config.base_shift,
        scheduler.config.max_shift,
    )
    velocity0 = QwenImage21Pipeline._unpack_latents(
        captured["velocity0"], height, width, pipe.vae_scale_factor
    )[:, :, 0]
    final_latents = QwenImage21Pipeline._unpack_latents(
        captured["final"].detach().float().cpu(), height, width, pipe.vae_scale_factor
    )[:, :, 0]
    assert velocity0.shape == latents0.shape and final_latents.shape == latents0.shape
    del pipe, transformer, vae
    gc.collect()

    save_st(out / f"velocity0-{dtype_name}.safetensors", "velocity", velocity0)
    save_st(out / f"latents-final-{dtype_name}.safetensors", "latents", final_latents)
    image.save(out / f"image-{dtype_name}.png")
    arm = {
        "dtype": dtype_name,
        "use_kv_cache": use_kv_cache,
        "device": device.type,
        "torch": torch.__version__,
        "diffusers": diffusers.__version__,
        "sigmas": sigmas,
        "mu": float(mu),
        "step_s": captured["step_s"],
        "velocity0_abs_max": velocity0.abs().max().item(),
        "velocity0_rms": velocity0.pow(2).mean().sqrt().item(),
        "alpha_min": int(image_u8[..., 3].min()),
    }
    (out / f"arm-{dtype_name}.json").write_text(json.dumps(arm, indent=2) + "\n")
    print(f"{tag} steps {sum(captured['step_s']):.1f}s -> {out}", flush=True)

    # The bf16 arm closes the case: it prices its own gap to the fp32 arm and copies the
    # fixture set under tests/fixtures so the Rust gate runs from the checkout alone.
    if dtype_name == "bf16":
        import shutil

        ref_v = load_st(out / "velocity0-fp32.safetensors", "velocity")
        ref_l = load_st(out / "latents-final-fp32.safetensors", "latents")
        ref_img = np.asarray(PIL.Image.open(out / "image-fp32.png"))
        ref_arm = json.loads((out / "arm-fp32.json").read_text())
        assert ref_arm["use_kv_cache"] == use_kv_cache, "the two arms ran different step arms"
        fixture = TRANSFORMER_FIXTURE_DIR / case
        fixture.mkdir(parents=True, exist_ok=True)
        for name in (
            "latents0.safetensors", "cap_feats.safetensors",
            "velocity0-fp32.safetensors", "latents-final-fp32.safetensors", "image-fp32.png",
        ):
            shutil.copyfile(out / name, fixture / name)
        meta = {
            "note": (
                "Qwen-Image 2.1 transformer reference: diffusers QwenImage21Pipeline.__call__ "
                "with prompt_embeds and latents injected, use_kv_cache on, no guidance. Both "
                "sides start from latents0 (the noise, unpacked, unscaled) and cap_feats (the "
                "kept pre-norm encoder rows, bf16). latents-final is the normalised latent "
                "before the VAE's z * std + mean. The fp32 files are the reference; bf16_vs_fp32 "
                "is the reference's own gap when the transformer runs in bf16 on the same "
                "inputs, which is what the Rust bars are read against. The PNG is RGBA."
            ),
            "case": case,
            "width": width,
            "height": height,
            "seed": seed,
            "steps": steps,
            "use_kv_cache": ref_arm["use_kv_cache"],
            "sigmas": ref_arm["sigmas"],
            "mu": ref_arm["mu"],
            "prompt": cap_info,
            "snapshot_revision": snapshot.name,
            "reference": {k: ref_arm[k] for k in ("torch", "diffusers", "device", "dtype")},
            "reference_velocity0": {
                "abs_max": ref_arm["velocity0_abs_max"],
                "rms": ref_arm["velocity0_rms"],
            },
            "reference_alpha_min": ref_arm["alpha_min"],
            "bf16_vs_fp32": {
                "velocity0": spread(velocity0, ref_v),
                "final_latents": spread(final_latents, ref_l),
                "image_psnr_db": psnr(image_u8[..., :3], ref_img[..., :3]),
            },
            "sha256": {
                name: sha256_file(fixture / name) for name in sorted(os.listdir(fixture))
                if name != "meta.json"
            },
        }
        (fixture / "meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"[bf16] bf16 vs fp32: {json.dumps(meta['bf16_vs_fp32'], indent=2)}")
        print(f"[bf16] wrote {fixture}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True, choices=["tokens", "fp32", "bf16", "finalize", "transformer"])
    ap.add_argument("--snapshot", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("/tmp/qwen-image-ref"))
    ap.add_argument("--threads", type=int, default=8)
    # fp32 is the acceptance reference and runs eager for a transparent, deterministic
    # softmax; bf16 runs sdpa because that is what the diffusers pipeline executes.
    ap.add_argument("--attn-impl", default=None)
    # The transformer stage.
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default=None)
    ap.add_argument("--size", default="512x512", help="WIDTHxHEIGHT, multiples of 32")
    ap.add_argument("--prompt-idx", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=NUM_STEPS)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.stage == "finalize":
        finalize(args.out_dir)
        return
    snapshot = args.snapshot or default_snapshot()
    if args.stage == "tokens":
        run_tokens(snapshot)
        return
    if args.stage == "transformer":
        assert args.dtype, "--stage transformer needs --dtype fp32|bf16"
        width, height = (int(side) for side in args.size.lower().split("x"))
        run_transformer(
            snapshot, args.out_dir, args.dtype, width, height, args.prompt_idx, args.seed,
            args.steps,
        )
        return
    attn = args.attn_impl or ("eager" if args.stage == "fp32" else "sdpa")
    run_stage(args.stage, snapshot, args.out_dir, attn)


if __name__ == "__main__":
    main()
