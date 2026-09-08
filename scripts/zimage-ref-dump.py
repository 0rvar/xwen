#!/usr/bin/env python3
# Reference dumps for xwen's Z-Image path (docs/zimage.md). Stage 2 is the text encoder,
# Stages 3 and 4 are the diffusion transformer and the decoded image.
#
#   uv venv /tmp/zimage-venv --python 3.12
#   uv pip install --python /tmp/zimage-venv/bin/python torch transformers safetensors numpy \
#     diffusers accelerate
#   /tmp/zimage-venv/bin/python scripts/zimage-ref-dump.py --stage fp32 \
#     && /tmp/zimage-venv/bin/python scripts/zimage-ref-dump.py --stage bf16 \
#     && /tmp/zimage-venv/bin/python scripts/zimage-ref-dump.py --stage finalize
#   /tmp/zimage-venv/bin/python scripts/zimage-ref-dump.py --stage transformer --dtype fp32 \
#     && /tmp/zimage-venv/bin/python scripts/zimage-ref-dump.py --stage transformer --dtype bf16
# Control and LoRA reuse the checked-in 512x512 caption/noise fixture:
#   --stage control --author-source /tmp/xwen-videox-author --control-file <8steps.safetensors>
#     --control-image <map.png> --out-dir /tmp/xwen-image-control-ref
#   --stage lora --lora-file <adapter.safetensors> --lora-weight 0.8
#     --out-dir /tmp/xwen-image-control-ref
# Author sources are the three AUTHOR_FILES below from VideoX-Fun at AUTHOR_REVISION,
# under videox_fun/models/. They live outside the repo and their hashes are checked.
#
# This is the only Python in the repo. It exists because the Z-Image text encoder has no
# ONNX export and there is no bun path to torch; it runs once, by hand, to produce the
# references `tests/qwen3_encoder.rs` and `tests/zimage_parity.rs` grade xwen against. It
# never runs in CI.
#
# The transformer stage writes `tests/fixtures/zimage-transformer/<WxH>-p<idx>-s<seed>/`:
# the injected noise and caption features both sides start from, the reference's step-0
# velocity and final latent in the requested dtype, and the fp32 arm's decoded PNG. Run
# fp32 first: it writes the inputs, and the bf16 arm reuses them so the two arms differ in
# arithmetic alone. The Rust test grades xwen's bf16 run against the fp32 arm with a bar
# set from the reference's own fp32-to-bf16 spread. The transformer arms run on mps, the
# caption features come from the encoder in fp32 on cpu (see the note below).
#
# What it reproduces: diffusers `ZImagePipeline._encode_prompt` renders
# `[{"role": "user", "content": prompt}]` with add_generation_prompt=True and
# enable_thinking=True, tokenizes with padding="max_length", max_length=512,
# truncation=True, runs the encoder with output_hidden_states=True and takes
# `hidden_states[-2]` sliced to the real tokens. With right padding and causal attention a
# batch-1 unpadded forward is mathematically identical, so this script runs unpadded and
# verifies that claim once (--stage fp32 checks one prompt padded-to-512 against unpadded).
#
# `hidden_states[-2]` is index 35 for a 36-layer model: the output of layers[34], before
# `model.norm`. The script proves that index convention with forward hooks rather than
# assuming it.
#
# CPU ONLY, never mps. Layer 35's MLP in the Z-Image copy of the weights is partially
# zeroed (a known upstream corruption of shard 3: model.layers.35.mlp.up_proj has ~14.8M
# contiguous zeros, down_proj ~3.9M). Index 35 does not evaluate layer 35, so it does not
# affect this reference; it does mean the Z-Image copy is not a faithful full LM.

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

REPO_ID = "Tongyi-MAI/Z-Image-Turbo"
REVISION = "f332072aa78be7aecdf3ee76d5c247082da564a6"
DEFAULT_SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub"
    / f"models--{REPO_ID.replace('/', '--')}/snapshots"
    / REVISION
)
REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = REPO_ROOT / "tests/fixtures/zimage-encoder"

MAX_LENGTH = 512
HIDDEN_INDEX = 35  # hidden_states[-2] for a 36-layer model
PAD_ID = 151643  # <|endoftext|>
CHECK_PROMPT_IDX = 1  # the prompt the padded-vs-unpadded and sdpa-vs-eager checks use


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


def render(tok, text: str) -> str:
    return tok.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


def encode(tok, rendered: str) -> tuple[list[int], int, bool]:
    """Ids after the pipeline's truncation, plus the untruncated length."""
    full = tok(rendered, add_special_tokens=False)["input_ids"]
    # The pipeline calls the tokenizer without add_special_tokens; Qwen2Tokenizer adds
    # none, and this asserts that rather than trusting it.
    assert tok(rendered)["input_ids"] == full, "tokenizer added special tokens"
    truncated = len(full) > MAX_LENGTH
    return full[:MAX_LENGTH], len(full), truncated


def build_model(snapshot: Path, dtype: torch.dtype, attn_impl: str):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        snapshot / "text_encoder",
        dtype=dtype,
        local_files_only=True,
        attn_implementation=attn_impl,
    )
    model.eval()
    model.to("cpu")
    return model


def inner(model):
    """The Qwen3Model under the causal-LM head."""
    return model.model


def hidden_states_of(model, ids: list[int]) -> tuple[tuple, dict]:
    """Forward one unpadded sequence, returning hidden_states and the hook captures."""
    captured: dict[int, torch.Tensor] = {}

    def make_hook(i):
        def hook(_module, _args, output):
            captured[i] = (output[0] if isinstance(output, tuple) else output).detach()

        return hook

    layers = inner(model).layers
    handles = [
        layers[HIDDEN_INDEX - 1].register_forward_hook(make_hook(HIDDEN_INDEX - 1)),
        layers[HIDDEN_INDEX].register_forward_hook(make_hook(HIDDEN_INDEX)),
    ]
    try:
        with torch.inference_mode():
            out = model(
                input_ids=torch.tensor([ids], dtype=torch.long),
                output_hidden_states=True,
                use_cache=False,
            )
    finally:
        for h in handles:
            h.remove()
    return out.hidden_states, captured


def check_index_convention(model, hs, captured) -> dict:
    """Prove hidden_states[35] is the output of layers[34], pre-norm."""
    norm = inner(model).norm
    with torch.inference_mode():
        normed_35 = norm(hs[HIDDEN_INDEX])
        normed_layer35_out = norm(captured[HIDDEN_INDEX])
    return {
        "n_hidden_states": len(hs),
        # hidden_states[35] IS the layer-34 output tensor.
        "hs35_is_layer34_output": bool(
            torch.equal(hs[HIDDEN_INDEX].float(), captured[HIDDEN_INDEX - 1].float())
        ),
        "hs35_vs_layer34_max_abs": float(
            (hs[HIDDEN_INDEX].float() - captured[HIDDEN_INDEX - 1].float()).abs().max()
        ),
        # hidden_states[36] is norm(layer-35 output), NOT norm(hidden_states[35]).
        "hs36_vs_norm_layer35_max_abs": float(
            (hs[HIDDEN_INDEX + 1].float() - normed_layer35_out.float()).abs().max()
        ),
        "hs36_vs_norm_hs35_max_abs": float(
            (hs[HIDDEN_INDEX + 1].float() - normed_35.float()).abs().max()
        ),
    }


def padded_check(model, ids: list[int]) -> dict:
    """One prompt, padded to 512 with a mask, against the unpadded forward."""
    t = len(ids)
    padded = ids + [PAD_ID] * (MAX_LENGTH - t)
    mask = [1] * t + [0] * (MAX_LENGTH - t)
    with torch.inference_mode():
        out = model(
            input_ids=torch.tensor([padded], dtype=torch.long),
            attention_mask=torch.tensor([mask], dtype=torch.long),
            output_hidden_states=True,
            use_cache=False,
        )
        unpadded = model(
            input_ids=torch.tensor([ids], dtype=torch.long),
            output_hidden_states=True,
            use_cache=False,
        )
    a = out.hidden_states[HIDDEN_INDEX][0, :t].float()
    b = unpadded.hidden_states[HIDDEN_INDEX][0, :t].float()
    return {
        "prompt_idx": CHECK_PROMPT_IDX,
        "tokens": t,
        "max_abs_diff": float((a - b).abs().max()),
        "bitwise_equal": bool(torch.equal(a, b)),
    }


def run_stage(stage: str, snapshot: Path, out_dir: Path, attn_impl: str) -> None:
    from transformers import AutoTokenizer
    import transformers

    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[stage]
    tok = AutoTokenizer.from_pretrained(snapshot / "tokenizer", local_files_only=True)
    prompts = load_prompts()

    print(f"[{stage}] loading model ({dtype}, attn={attn_impl})", flush=True)
    t0 = time.time()
    model = build_model(snapshot, dtype, attn_impl)
    load_s = time.time() - t0
    resolved_attn = getattr(model.config, "_attn_implementation", attn_impl)
    print(f"[{stage}] loaded in {load_s:.1f}s, attn={resolved_attn}", flush=True)

    records = []
    checks = None
    for p in prompts:
        idx, text = p["idx"], p["text"]
        rendered = render(tok, text)
        assert rendered.endswith("<|im_start|>assistant\n"), f"prompt {idx}: bad tail"
        if "<think>" not in text:
            assert "<think>" not in rendered, f"prompt {idx}: template emitted <think>"
        ids, untruncated_len, truncated = encode(tok, rendered)

        d = out_dir / f"{idx:02d}"
        d.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        hs, captured = hidden_states_of(model, ids)
        wall_s = time.time() - t0

        assert len(hs) == 37, f"prompt {idx}: {len(hs)} hidden states, expected 37"
        conv = check_index_convention(model, hs, captured)
        assert conv["hs35_is_layer34_output"], f"prompt {idx}: index convention broken"
        if stage == "fp32":
            assert conv["hs36_vs_norm_layer35_max_abs"] < 1e-5, (
                f"prompt {idx}: hidden_states[36] != norm(layers[35] output)"
            )

        hidden = hs[HIDDEN_INDEX][0].float().contiguous()
        assert hidden.shape == (len(ids), 2560), f"prompt {idx}: {tuple(hidden.shape)}"

        npy = d / f"hidden_{stage}.npy"
        np.save(npy, hidden.numpy())
        rec = {
            "idx": idx,
            "label": p["label"],
            "dir": f"{idx:02d}",
            "T": len(ids),
            "truncated": truncated,
            "untruncated_len": untruncated_len,
            "ids": ids,
            "rendered_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
            "rendered": rendered,
            f"hidden_{stage}_npy": npy.name,
            f"hidden_{stage}_npy_sha256": sha256_file(npy),
            f"wall_s_{stage}": round(wall_s, 3),
            "index_convention": conv,
        }

        if stage == "fp32":
            (d / "rendered.txt").write_text(rendered, encoding="utf-8")
            (d / "ids.json").write_text(
                json.dumps(
                    {
                        "idx": idx,
                        "ids": ids,
                        "T": len(ids),
                        "truncated": truncated,
                        "untruncated_len": untruncated_len,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            try:
                from safetensors.torch import save_file

                st = d / "hidden_fp32.safetensors"
                save_file({"hidden": hidden}, str(st))
                rec["hidden_fp32_safetensors"] = st.name
                rec["hidden_fp32_safetensors_sha256"] = sha256_file(st)
            except ImportError:
                rec["hidden_fp32_safetensors"] = None

        records.append(rec)
        print(
            f"[{stage}] {idx:2d} {p['label']:22s} T={len(ids):4d} "
            f"trunc={truncated!s:5s} {wall_s:7.2f}s",
            flush=True,
        )

    if stage == "fp32":
        ids = records[CHECK_PROMPT_IDX]["ids"]
        checks = {"padded_vs_unpadded": padded_check(model, ids)}
        print(f"[fp32] padded-vs-unpadded: {checks['padded_vs_unpadded']}", flush=True)

    del model
    gc.collect()

    if stage == "fp32" and attn_impl == "eager":
        # Disentangle the attention kernel from dtype: the bf16 stage runs sdpa (what the
        # pipeline runs), so show that fp32 sdpa and fp32 eager agree first.
        m2 = build_model(snapshot, torch.float32, "sdpa")
        ids = records[CHECK_PROMPT_IDX]["ids"]
        with torch.inference_mode():
            o = m2(
                input_ids=torch.tensor([ids], dtype=torch.long),
                output_hidden_states=True,
                use_cache=False,
            )
        ref = np.load(out_dir / f"{CHECK_PROMPT_IDX:02d}" / "hidden_fp32.npy")
        d = np.abs(o.hidden_states[HIDDEN_INDEX][0].float().numpy() - ref)
        checks["fp32_sdpa_vs_eager"] = {
            "prompt_idx": CHECK_PROMPT_IDX,
            "max_abs_diff": float(d.max()),
        }
        print(f"[fp32] sdpa-vs-eager: {checks['fp32_sdpa_vs_eager']}", flush=True)
        del m2
        gc.collect()

    stage_manifest = {
        "stage": stage,
        "dtype": str(dtype),
        "attn_implementation": resolved_attn,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "model_snapshot": str(snapshot),
        "repo_id": REPO_ID,
        "revision": REVISION,
        "max_length": MAX_LENGTH,
        "hidden_index": HIDDEN_INDEX,
        "torch_num_threads": torch.get_num_threads(),
        "load_s": round(load_s, 1),
        "checks": checks,
        "records": records,
    }
    (out_dir / f"manifest-{stage}.json").write_text(
        json.dumps(stage_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[{stage}] wrote {out_dir / f'manifest-{stage}.json'}", flush=True)


def finalize(out_dir: Path) -> None:
    fp32 = json.loads((out_dir / "manifest-fp32.json").read_text(encoding="utf-8"))
    bf16 = json.loads((out_dir / "manifest-bf16.json").read_text(encoding="utf-8"))
    by_idx_bf16 = {r["idx"]: r for r in bf16["records"]}

    merged = []
    for r in fp32["records"]:
        b = by_idx_bf16[r["idx"]]
        assert b["ids"] == r["ids"], f"prompt {r['idx']}: ids differ between stages"
        x = np.load(out_dir / r["dir"] / "hidden_fp32.npy").astype(np.float64)
        y = np.load(out_dir / r["dir"] / "hidden_bf16.npy").astype(np.float64)
        assert x.shape == y.shape
        num = (x * y).sum(axis=1)
        den = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
        cos = num / np.maximum(den, 1e-30)
        rel = np.abs(y - x).max(axis=1) / np.maximum(np.abs(x).max(axis=1), 1e-6)
        m = dict(r)
        m.update(
            {
                "hidden_bf16_npy": b["hidden_bf16_npy"],
                "hidden_bf16_npy_sha256": b["hidden_bf16_npy_sha256"],
                "wall_s_bf16": b["wall_s_bf16"],
                "bf16_vs_fp32": {
                    "min_cosine": float(cos.min()),
                    "mean_cosine": float(cos.mean()),
                    "max_rel_error": float(rel.max()),
                    "mean_rel_error": float(rel.mean()),
                },
            }
        )
        merged.append(m)
        print(
            f"[final] {m['idx']:2d} {m['label']:22s} T={m['T']:4d} "
            f"min_cos={cos.min():.8f} max_rel={rel.max():.5f} "
            f"fp32={m['wall_s_fp32']:6.2f}s bf16={m['wall_s_bf16']:6.2f}s",
            flush=True,
        )

    manifest = {
        "generated_by": "scripts/zimage-ref-dump.py",
        "repo_id": REPO_ID,
        "revision": REVISION,
        "model_snapshot": fp32["model_snapshot"],
        "max_length": MAX_LENGTH,
        "hidden_index": HIDDEN_INDEX,
        "pad_id": PAD_ID,
        "versions": {
            "torch": fp32["torch"],
            "transformers": fp32["transformers"],
            "numpy": fp32["numpy"],
        },
        "stages": {
            "fp32": {
                "dtype": fp32["dtype"],
                "attn_implementation": fp32["attn_implementation"],
                "load_s": fp32["load_s"],
            },
            "bf16": {
                "dtype": bf16["dtype"],
                "attn_implementation": bf16["attn_implementation"],
                "load_s": bf16["load_s"],
            },
        },
        "torch_num_threads": fp32["torch_num_threads"],
        "checks": fp32["checks"],
        "prompts": merged,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[final] wrote {out_dir / 'manifest.json'}", flush=True)

    # The committed fixture: everything the Rust test needs without the dumps, plus the
    # names it uses to find them under $XWEN_ZIMAGE_REF_DIR.
    reference = {
        "note": (
            "Reference for tests/qwen3_encoder.rs. Dumps are NOT committed; point "
            "$XWEN_ZIMAGE_REF_DIR at the output of scripts/zimage-ref-dump.py. Each "
            "prompt's arrays live in <XWEN_ZIMAGE_REF_DIR>/<dir>/<file>."
        ),
        "repo_id": REPO_ID,
        "revision": REVISION,
        "max_length": MAX_LENGTH,
        "hidden_index": HIDDEN_INDEX,
        "pad_id": PAD_ID,
        "hidden_size": 2560,
        "versions": manifest["versions"],
        "stages": manifest["stages"],
        "checks": manifest["checks"],
        "prompts": [
            {
                k: v
                for k, v in m.items()
                if k
                in {
                    "idx",
                    "label",
                    "dir",
                    "T",
                    "truncated",
                    "untruncated_len",
                    "ids",
                    "rendered",
                    "rendered_sha256",
                    "hidden_fp32_npy",
                    "hidden_fp32_npy_sha256",
                    "hidden_fp32_safetensors",
                    "hidden_fp32_safetensors_sha256",
                    "hidden_bf16_npy",
                    "hidden_bf16_npy_sha256",
                    "bf16_vs_fp32",
                }
            }
            for m in merged
        ],
    }
    dest = FIXTURE_DIR / "reference.json"
    dest.write_text(
        json.dumps(reference, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[final] wrote {dest}", flush=True)


# --- Stages 3 and 4: the transformer and the decoded image -------------------------
#
# What this reproduces is diffusers' `ZImagePipeline.__call__` from `prepare_latents`
# onwards, with the noise and the caption features fixed by files so xwen can start from
# the same two tensors: the scheduler's sigma grid (`get_default_z_image_sigmas`, then the
# static shift, then a terminal zero), the `t = (1000 - timestep) / 1000` the transformer
# is fed, the negation of its output, `scheduler.step`, and the VAE decode after
# `latents / scaling_factor + shift_factor`. The caption features are the encoder's
# `hidden_states[-2]` for one prompt of prompts.json, computed the way Stage 2 does (fp32,
# cpu, unpadded) and rounded once to bf16, which is the dtype the pipeline hands the
# transformer; both sides read that rounded tensor, so the encoder is out of the picture
# here and Stage 2 stays its only gate.
#
# Two arms, both on mps. The fp32 arm is the reference and writes the inputs. The bf16 arm
# reads them back, so the two arms share every bit of input and their velocity gap is the
# reference's own bf16 rounding; that gap is what the Rust bar is set from, and it lands in
# the fixture's meta.json. The VAE is decoded in f32 in both arms (xwen does the same),
# so the image gap of the bf16 arm is the transformer's alone.

TRANSFORMER_FIXTURE_DIR = REPO_ROOT / "tests/fixtures/zimage-transformer"
NUM_STEPS = 8
VAE_SCALING_FACTOR = 0.3611
VAE_SHIFT_FACTOR = 0.1159


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


def cap_feats_for(snapshot: Path, prompt_idx: int) -> tuple[torch.Tensor, dict]:
    """The encoder's hidden_states[-2] for one fixture prompt, fp32 on cpu, then bf16."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(snapshot / "tokenizer", local_files_only=True)
    prompt = next(p for p in load_prompts() if p["idx"] == prompt_idx)
    rendered = render(tok, prompt["text"])
    ids, _untruncated, truncated = encode(tok, rendered)
    assert not truncated, f"prompt {prompt_idx} truncates; pick a shorter one"

    print(f"[cap] loading the encoder (fp32, cpu, eager)", flush=True)
    model = build_model(snapshot, torch.float32, "eager")
    hs, _captured = hidden_states_of(model, ids)
    hidden = hs[HIDDEN_INDEX][0].float().contiguous()
    assert hidden.shape == (len(ids), 2560), tuple(hidden.shape)
    del model, hs
    gc.collect()
    info = {
        "prompt_idx": prompt_idx,
        "label": prompt["label"],
        "text": prompt["text"],
        "ids": ids,
        "T": len(ids),
        "encoder_dtype": "fp32 on cpu, rounded once to bf16",
    }
    return hidden.to(torch.bfloat16), info


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


def run_transformer(
    snapshot: Path, out_root: Path, dtype_name: str, width: int, height: int,
    prompt_idx: int, seed: int,
) -> None:
    import PIL.Image
    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, ZImageTransformer2DModel
    from diffusers.pipelines.z_image.pipeline_z_image import get_default_z_image_sigmas
    import diffusers

    assert torch.backends.mps.is_available(), "this stage runs on mps"
    device = torch.device("mps")
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[dtype_name]
    assert width % 16 == 0 and height % 16 == 0, "sides must be multiples of 16"
    lat_h, lat_w = height // 8, width // 8
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
        latents0 = torch.randn((1, 16, lat_h, lat_w), generator=gen, dtype=torch.float32)
        save_st(latents_path, "latents", latents0)
        save_st(cap_path, "cap_feats", cap_feats)
        inputs_path.write_text(json.dumps(cap_info, indent=2, ensure_ascii=False) + "\n")
    else:
        assert latents_path.is_file() and cap_path.is_file(), "run --dtype fp32 first"
        latents0 = load_st(latents_path, "latents")
        cap_feats = load_st(cap_path, "cap_feats")
        cap_info = json.loads(inputs_path.read_text())
    assert latents0.shape == (1, 16, lat_h, lat_w) and latents0.dtype == torch.float32
    assert cap_feats.dtype == torch.bfloat16 and cap_feats.shape[1] == 2560

    print(f"{tag} loading the transformer ({dtype}) onto mps", flush=True)
    t0 = time.time()
    transformer = ZImageTransformer2DModel.from_pretrained(
        snapshot / "transformer", torch_dtype=dtype, local_files_only=True
    )
    transformer.eval().to(device)
    print(f"{tag} loaded in {time.time() - t0:.1f}s", flush=True)

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        snapshot / "scheduler", local_files_only=True
    )
    assert scheduler.config.use_dynamic_shifting is False and scheduler.config.shift == 3.0
    scheduler.set_timesteps(sigmas=get_default_z_image_sigmas(NUM_STEPS), device=device)
    scheduler.set_begin_index(0)
    sigmas = scheduler.sigmas.tolist()
    assert len(sigmas) == NUM_STEPS + 1 and sigmas[-1] == 0.0, sigmas

    latents = latents0.to(device)
    cap = [cap_feats.to(device).to(dtype)]
    step_s = []
    velocity0 = None
    with torch.inference_mode():
        for i, t in enumerate(scheduler.timesteps):
            t0 = time.time()
            timestep = ((1000 - t.expand(1)) / 1000).to(device)
            x = [latents.to(dtype).unsqueeze(2)[0]]
            model_out = transformer(x, timestep, cap, return_dict=False)[0]
            noise_pred = torch.stack([o.float() for o in model_out], dim=0).squeeze(2)
            noise_pred = -noise_pred
            if i == 0:
                velocity0 = noise_pred.detach().cpu().clone()
            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            assert latents.dtype == torch.float32
            torch.mps.synchronize()
            step_s.append(time.time() - t0)
            print(f"{tag} step {i} sigma={sigmas[i]:.7f} {step_s[-1]:.2f}s", flush=True)
    del transformer
    gc.collect()
    torch.mps.empty_cache()

    print(f"{tag} decoding through the VAE (f32)", flush=True)
    vae = AutoencoderKL.from_pretrained(
        snapshot / "vae", torch_dtype=torch.float32, local_files_only=True
    )
    vae.eval().to(device)
    assert vae.config.scaling_factor == VAE_SCALING_FACTOR
    assert vae.config.shift_factor == VAE_SHIFT_FACTOR
    t0 = time.time()
    with torch.inference_mode():
        scaled = latents / vae.config.scaling_factor + vae.config.shift_factor
        image = vae.decode(scaled, return_dict=False)[0]
        torch.mps.synchronize()
    vae_s = time.time() - t0
    # diffusers' VaeImageProcessor.postprocess: clamp to [-1, 1], map to [0, 1], then
    # `(images * 255).round().astype("uint8")`.
    image = ((image / 2 + 0.5).clamp(0, 1))[0].permute(1, 2, 0).cpu().numpy()
    image_u8 = (image * 255).round().astype(np.uint8)
    del vae

    final_latents = latents.detach().cpu()
    save_st(out / f"velocity0-{dtype_name}.safetensors", "velocity", velocity0)
    save_st(out / f"latents-final-{dtype_name}.safetensors", "latents", final_latents)
    PIL.Image.fromarray(image_u8).save(out / f"image-{dtype_name}.png")
    arm = {
        "dtype": dtype_name,
        "device": "mps",
        "torch": torch.__version__,
        "diffusers": diffusers.__version__,
        "sigmas": sigmas,
        "step_s": step_s,
        "vae_decode_s": vae_s,
        "velocity0_abs_max": velocity0.abs().max().item(),
        "velocity0_rms": velocity0.pow(2).mean().sqrt().item(),
    }
    (out / f"arm-{dtype_name}.json").write_text(json.dumps(arm, indent=2) + "\n")
    print(f"{tag} steps {sum(step_s):.1f}s, vae {vae_s:.1f}s -> {out}", flush=True)

    # The bf16 arm closes the case: it prices its own gap to the fp32 arm and copies the
    # fixture set (the two inputs, the fp32 velocity, latent and image, and the numbers)
    # under tests/fixtures so the Rust gate runs from the checkout alone.
    if dtype_name == "bf16":
        ref_v = load_st(out / "velocity0-fp32.safetensors", "velocity")
        ref_l = load_st(out / "latents-final-fp32.safetensors", "latents")
        ref_img = np.asarray(PIL.Image.open(out / "image-fp32.png"))
        ref_arm = json.loads((out / "arm-fp32.json").read_text())
        fixture = TRANSFORMER_FIXTURE_DIR / case
        fixture.mkdir(parents=True, exist_ok=True)
        import shutil

        for name in (
            "latents0.safetensors", "cap_feats.safetensors",
            "velocity0-fp32.safetensors", "latents-final-fp32.safetensors", "image-fp32.png",
        ):
            shutil.copyfile(out / name, fixture / name)
        meta = {
            "note": (
                "Z-Image transformer reference, diffusers on mps. Both sides start from "
                "latents0 (the noise) and cap_feats (the encoder output, bf16). The fp32 "
                "files are the reference; bf16_vs_fp32 is the reference's own gap when it "
                "runs the transformer in bf16 on the same inputs, which is what the Rust "
                "bars are set from."
            ),
            "case": case,
            "width": width,
            "height": height,
            "seed": seed,
            "steps": NUM_STEPS,
            "sigmas": sigmas,
            "prompt": cap_info,
            "snapshot_revision": REVISION,
            "reference": {k: ref_arm[k] for k in ("torch", "diffusers", "device", "dtype")},
            "reference_velocity0": {
                "abs_max": ref_arm["velocity0_abs_max"],
                "rms": ref_arm["velocity0_rms"],
            },
            "bf16_vs_fp32": {
                "velocity0": spread(velocity0, ref_v),
                "final_latents": spread(final_latents, ref_l),
                "image_psnr_db": psnr(image_u8, ref_img),
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


def run_image_edits(snapshot: Path, out_root: Path, seed: int, strength: float) -> None:
    """Run the actual diffusers edit pipelines with explicit posterior and diffusion noise."""
    import PIL.Image
    import diffusers
    from unittest.mock import patch
    from diffusers import (
        AutoencoderKL, FlowMatchEulerDiscreteScheduler, ZImageTransformer2DModel,
        ZImageImg2ImgPipeline, ZImageInpaintPipeline,
    )
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
    from diffusers.pipelines.z_image.pipeline_z_image import get_default_z_image_sigmas

    assert 0 < strength <= 1
    source_dir = TRANSFORMER_FIXTURE_DIR / "512x512-p1-s0"
    source = PIL.Image.open(source_dir / "image-fp32.png").convert("RGB")
    width, height = source.size
    cap = load_st(source_dir / "cap_feats.safetensors", "cap_feats")
    generator = torch.Generator("cpu").manual_seed(seed)
    shape = (1, 16, height // 8, width // 8)
    posterior_noise = torch.randn(shape, generator=generator)
    diffusion_noise = torch.randn(shape, generator=generator)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[height // 4:3 * height // 4, width // 4:3 * width // 4] = 255
    mask_image = PIL.Image.fromarray(mask)
    device = torch.device("mps")
    vae = AutoencoderKL.from_pretrained(
        snapshot / "vae", torch_dtype=torch.float32, local_files_only=True
    ).eval().to(device)
    transformer = ZImageTransformer2DModel.from_pretrained(
        snapshot / "transformer", torch_dtype=torch.float32, local_files_only=True
    ).eval().to(device)

    for mode, pipeline_cls in [("img2img", ZImageImg2ImgPipeline), ("inpaint", ZImageInpaintPipeline)]:
        out = out_root / "edits" / mode
        out.mkdir(parents=True, exist_ok=True)
        source.save(out / "source.png")
        mask_image.save(out / "mask.png")
        save_st(out / "cap_feats.safetensors", "cap_feats", cap)
        save_st(out / "posterior-noise.safetensors", "noise", posterior_noise)
        save_st(out / "noise.safetensors", "latents", diffusion_noise)
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            snapshot / "scheduler", local_files_only=True
        )
        pipe = pipeline_cls(
            scheduler=scheduler, vae=vae, transformer=transformer,
            text_encoder=None, tokenizer=None,
        )
        samples = []
        velocities = []
        noise_draws = []

        def posterior_sample(distribution, generator=None):
            assert tuple(distribution.mean.shape) == shape
            value = distribution.mean + distribution.std * posterior_noise.to(device)
            if not samples:
                save_st(out / "mean.safetensors", "mean", distribution.mean)
                save_st(out / "logvar.safetensors", "logvar", distribution.logvar)
                save_st(out / "source-latents.safetensors", "latents",
                        (value - vae.config.shift_factor) * vae.config.scaling_factor)
            samples.append(True)
            return value

        def draw_noise(requested_shape, generator=None, device=None, dtype=None, **kwargs):
            assert tuple(requested_shape) == shape
            noise_draws.append(True)
            return diffusion_noise.to(device=device, dtype=dtype)

        def capture_velocity(_module, _args, output):
            if not velocities:
                predictions = output[0]
                velocities.append(-torch.stack([o.float() for o in predictions]).squeeze(2).cpu())

        hook = transformer.register_forward_hook(capture_velocity)
        module = pipeline_cls.__module__
        kwargs = dict(
            prompt=None, image=source, strength=strength, num_inference_steps=NUM_STEPS,
            sigmas=get_default_z_image_sigmas(NUM_STEPS), guidance_scale=0,
            prompt_embeds=[cap.to(device).float()], output_type="latent",
        )
        if mode == "inpaint":
            kwargs["mask_image"] = mask_image
        started = time.time()
        try:
            with torch.inference_mode(), patch.object(DiagonalGaussianDistribution, "sample", posterior_sample), \
                    patch(f"{module}.randn_tensor", draw_noise):
                final = pipe(**kwargs).images
                if isinstance(final, list):
                    final = torch.stack(final)
                decoded = vae.decode(
                    final / vae.config.scaling_factor + vae.config.shift_factor, return_dict=False
                )[0]
        finally:
            hook.remove()
        assert samples and len(noise_draws) == 1 and velocities
        image = ((decoded / 2 + .5).clamp(0, 1))[0].permute(1, 2, 0).cpu().numpy()
        image = (image * 255).round().astype(np.uint8)
        PIL.Image.fromarray(image).save(out / "raw.png")
        composited = np.where(mask[:, :, None] != 0, image, np.asarray(source)) if mode == "inpaint" else image
        PIL.Image.fromarray(composited).save(out / "image.png")
        save_st(out / "velocity.safetensors", "velocity", velocities[0])
        save_st(out / "final.safetensors", "latents", final)
        meta = dict(
            mode=mode, width=width, height=height, strength=strength, steps=NUM_STEPS,
            start_step=int(max(NUM_STEPS - NUM_STEPS * strength, 0)), seed=seed,
            sigmas=scheduler.sigmas.tolist(), torch=torch.__version__, diffusers=diffusers.__version__,
            pipeline=f"{module}.{pipeline_cls.__name__}", revision=REVISION,
            seconds=time.time() - started,
            note="Actual reference pipeline; only its random draws are replaced by saved tensors. Pixel composite is xwen's extension.",
        )
        meta["sha256"] = {p.name: sha256_file(p) for p in sorted(out.iterdir()) if p.name != "meta.json"}
        (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(f"[{mode}] start {meta['start_step']}, {meta['seconds']:.1f}s -> {out}", flush=True)


AUTHOR_REVISION = "968f0e2192ba4c7a12868bf36d73260d135424ca"
AUTHOR_FILES = {
    "z_image_transformer2d.py": "d5a55c34a3f09721e0090106121f71f588da93492d606fd3d652ad4806261902",
    "z_image_transformer2d_control.py": "ae19699dbdb48a6698197dde698e00a918d3be1b2c46373b5715e2668ff57773",
    "attention_utils.py": "8cc9b14c299393423ce5bfb735fea0e4b14138851b041ad239792c3ffec062fc",
}


def author_control_class(source: Path):
    """Import unchanged pinned author files with unused multi-GPU/CUDA imports refused."""
    import importlib
    import sys
    import types

    for filename, digest in AUTHOR_FILES.items():
        assert sha256_file(source / filename) == digest, f"wrong author source: {filename}"
    package = "_xwen_videox_reference"
    root = types.ModuleType(package)
    root.__path__ = []
    models = types.ModuleType(package + ".models")
    models.__path__ = [str(source)]
    sys.modules[package] = root
    sys.modules[models.__name__] = models

    def unsupported(*args, **kwargs):
        raise AssertionError("the single-device SDPA reference cannot enter CUDA/distributed code")

    distributed = types.ModuleType(package + ".dist")
    for name in ("ZMultiGPUsSingleStreamAttnProcessor", "get_sequence_parallel_rank",
                 "get_sequence_parallel_world_size", "get_sp_group"):
        setattr(distributed, name, unsupported)
    sys.modules[distributed.__name__] = distributed
    sparse = types.ModuleType(package + ".models.attention_kernel")
    sparse._sparse_linear_attention = unsupported
    sparse.get_block_map = unsupported
    sys.modules[sparse.__name__] = sparse
    os.environ["VIDEOX_ATTENTION_TYPE"] = "SDPA"
    module = importlib.import_module(package + ".models.z_image_transformer2d_control")
    return module.ZImageControlTransformer2DModel


def controlled_reference(snapshot: Path, author_source: Path, control_file: Path):
    from accelerate import init_empty_weights
    from safetensors.torch import load_file
    cls = author_control_class(author_source)
    config = json.loads((snapshot / "transformer/config.json").read_text())
    import inspect
    config = {k: v for k, v in config.items() if k in inspect.signature(cls.__init__).parameters}
    count = 3 if "-lite-" in control_file.name else 15
    config.update(control_in_dim=33, control_layers_places=list(range(0, 30, 30 // count)),
                  control_refiner_layers_places=[0, 1], add_control_noise_refiner=True,
                  add_control_noise_refiner_correctly=True)
    with init_empty_weights():
        model = cls(**config)
    index = json.loads((snapshot / "transformer/diffusion_pytorch_model.safetensors.index.json").read_text())
    state = {}
    for shard in sorted(set(index["weight_map"].values())):
        state.update(load_file(snapshot / "transformer" / shard))
    state.update(load_file(control_file))
    model.load_state_dict(state, strict=True, assign=True)
    del state
    return model.float().eval().to("mps")


def dump_control_or_lora(args) -> None:
    """Replay saved inputs through author ControlNet or the actual diffusers LoRA loader."""
    import PIL.Image
    import diffusers
    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, ZImageTransformer2DModel, ZImagePipeline
    from diffusers.pipelines.z_image.pipeline_z_image import get_default_z_image_sigmas
    source_dir = TRANSFORMER_FIXTURE_DIR / "512x512-p1-s0"
    cap = load_st(source_dir / "cap_feats.safetensors", "cap_feats")
    noise = load_st(source_dir / "latents0.safetensors", "latents")
    device = torch.device("mps")
    root = args.out_dir / args.stage
    root.mkdir(parents=True, exist_ok=True)
    vae = AutoencoderKL.from_pretrained(args.snapshot / "vae", torch_dtype=torch.float32,
                                       local_files_only=True).eval().to(device)
    source = PIL.Image.open(source_dir / "image-fp32.png").convert("RGB")
    pixels = torch.from_numpy(np.asarray(source).copy()).permute(2, 0, 1).float().unsqueeze(0).to(device) / 127.5 - 1
    # Use the published canny control map so preprocessor differences stay out of the graph gate.
    if args.stage == "control":
        assert args.author_source and args.control_file and args.control_image
        control_image = PIL.Image.open(args.control_image).convert("RGB").resize(source.size, PIL.Image.Resampling.LANCZOS)
        control_pixels = torch.from_numpy(np.asarray(control_image).copy()).permute(2, 0, 1).float().unsqueeze(0).to(device) / 127.5 - 1
        with torch.inference_mode():
            control_mean = vae.encode(control_pixels).latent_dist.mode()
            control_latent = (control_mean - VAE_SHIFT_FACTOR) * VAE_SCALING_FACTOR
        model = controlled_reference(args.snapshot, args.author_source, args.control_file)
        cases = ["plain", "inpaint"]
        provenance = dict(author_revision=AUTHOR_REVISION, author_sha256=AUTHOR_FILES,
                          control_file=str(args.control_file), control_sha256=sha256_file(args.control_file))
    else:
        assert args.lora_file
        model = ZImageTransformer2DModel.from_pretrained(args.snapshot / "transformer", torch_dtype=torch.float32,
                                                        local_files_only=True).eval().to(device)
        pipe = ZImagePipeline(scheduler=FlowMatchEulerDiscreteScheduler(), transformer=model,
                              vae=vae, text_encoder=None, tokenizer=None)
        before = model.layers[0].attention.to_q.weight.detach().cpu().clone()
        pipe.load_lora_weights(str(args.lora_file.parent), weight_name=args.lora_file.name, adapter_name="parity")
        pipe.set_adapters("parity", adapter_weights=args.lora_weight)
        pipe.fuse_lora(components=["transformer"], lora_scale=1.0, safe_fusing=True)
        pipe.unload_lora_weights()
        after = model.layers[0].attention.to_q.weight.detach().cpu().clone()
        assert torch.count_nonzero(after - before) > 0, "split Q attention adapter did not merge"
        save_st(root / "attention-q-merged.safetensors", "weight", after)
        save_st(root / "attention-q-delta.safetensors", "weight", after - before)
        del before, after
        cases = ["merged"]
        provenance = dict(lora_file=str(args.lora_file), lora_sha256=sha256_file(args.lora_file),
                          lora_weight=args.lora_weight, merge="diffusers load_lora_weights/set_adapters/fuse_lora/unload_lora_weights in fp32")
    with torch.inference_mode():
        for case in cases:
            out = root / case
            out.mkdir(parents=True, exist_ok=True)
            save_st(out / "cap_feats.safetensors", "cap_feats", cap)
            save_st(out / "noise.safetensors", "latents", noise)
            if args.stage == "control":
                control_image.save(out / "control.png")
                source.save(out / "source.png")
                mask = torch.zeros((1, 1, 512, 512), device=device)
                mask[:, :, 128:384, 128:384] = 1
                PIL.Image.fromarray((mask[0,0].cpu().numpy()*255).astype(np.uint8)).save(out / "mask.png")
                if case == "inpaint":
                    masked_pixels = pixels * (mask < .5)
                    masked_mean = vae.encode(masked_pixels).latent_dist.mode()
                    source_latent = (masked_mean - VAE_SHIFT_FACTOR) * VAE_SCALING_FACTOR
                    keep = torch.nn.functional.interpolate(1-mask, size=(64,64), mode="nearest")
                    save_st(out / "masked-mean.safetensors", "mean", masked_mean)
                else:
                    source_latent = torch.zeros_like(control_latent)
                    keep = torch.zeros((1,1,64,64), device=device)
                context = torch.cat([control_latent, keep, source_latent], dim=1).unsqueeze(2)
                save_st(out / "control-mean.safetensors", "mean", control_mean)
                save_st(out / "context.safetensors", "context", context)
                print(f"[control/{case}] posterior.mode, context {tuple(context.shape)}, keep=[{keep.min().item()}, {keep.max().item()}]", flush=True)
            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.snapshot / "scheduler", local_files_only=True)
            scheduler.set_timesteps(sigmas=get_default_z_image_sigmas(NUM_STEPS), device=device)
            scheduler.set_begin_index(0)
            latents = noise.to(device)
            started = time.time()
            for i, t in enumerate(scheduler.timesteps):
                timestep = ((1000-t.expand(1))/1000).to(device)
                x = [latents.unsqueeze(2)[0]]
                if args.stage == "control":
                    output = model(x, timestep, [cap.to(device).float()],
                                   control_context=[context[0]], control_context_scale=.75)[0]
                    velocity = -output.float().squeeze(2)
                else:
                    output = model(x, timestep, [cap.to(device).float()], return_dict=False)[0]
                    velocity = -torch.stack([o.float() for o in output]).squeeze(2)
                if i == 0:
                    save_st(out / "velocity.safetensors", "velocity", velocity)
                latents = scheduler.step(velocity, t, latents, return_dict=False)[0]
                torch.mps.synchronize()
                print(f"[{args.stage}/{case}] step {i} elapsed {time.time()-started:.1f}s", flush=True)
            decoded = vae.decode(latents / VAE_SCALING_FACTOR + VAE_SHIFT_FACTOR, return_dict=False)[0]
            image = ((decoded/2+.5).clamp(0,1))[0].permute(1,2,0).cpu().numpy()
            PIL.Image.fromarray((image*255).round().astype(np.uint8)).save(out / "image.png")
            save_st(out / "final.safetensors", "latents", latents)
            meta = dict(width=512, height=512, steps=NUM_STEPS, scale=.75, case=case,
                        torch=torch.__version__, diffusers=diffusers.__version__, dtype="fp32",
                        sigmas=scheduler.sigmas.tolist(), seconds=time.time()-started, **provenance)
            meta["sha256"] = {p.name:sha256_file(p) for p in sorted(out.iterdir()) if p.name != "meta.json"}
            (out / "meta.json").write_text(json.dumps(meta, indent=2)+"\n")


def dump_preprocess_reference(args) -> None:
    """Author CPU models with shared input tensors for native preprocessing parity."""
    import sys
    import cv2
    import onnxruntime as ort
    from safetensors.torch import load_file
    from huggingface_hub import hf_hub_download
    import PIL.Image

    assert args.depth_author_source and args.dwpose_author_source and args.control_image
    sys.path.insert(0, str(args.depth_author_source))
    sys.path.insert(0, str(args.dwpose_author_source))
    from depth_anything_v2.dpt import DepthAnythingV2
    from dwpose.onnxdet import inference_detector
    from dwpose.onnxpose import inference_pose
    from dwpose.wholebody import Wholebody
    from dwpose import draw_pose
    out = args.out_dir / "preprocess"
    out.mkdir(parents=True, exist_ok=True)
    rgb = np.asarray(PIL.Image.open(args.control_image).convert("RGB"))
    depth_file = hf_hub_download("jeroenvlek/depth-anything-v2-safetensors", "depth_anything_v2_vits.safetensors", local_files_only=True)
    model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384]).eval()
    model.load_state_dict(load_file(depth_file), strict=True)
    if args.preprocess_input:
        depth_input = load_st(args.preprocess_input, "input")
    else:
        pixels = cv2.resize(rgb, (518, 518), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255
        pixels = (pixels - np.array([.485, .456, .406], dtype=np.float32)) / np.array([.229, .224, .225], dtype=np.float32)
        depth_input = torch.from_numpy(pixels.transpose(2, 0, 1).copy())[None]
    save_st(out / "depth-input.safetensors", "input", depth_input)
    depth = model(depth_input).unsqueeze(1)
    save_st(out / "depth-reference.safetensors", "depth", depth)
    del model
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    detector_file = hf_hub_download("yzd-v/DWPose", "yolox_l.onnx", local_files_only=True)
    estimator_file = hf_hub_download("yzd-v/DWPose", "dw-ll_ucoco_384.onnx", local_files_only=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    detector = ort.InferenceSession(detector_file, sess_options=options, providers=["CPUExecutionProvider"])
    estimator = ort.InferenceSession(estimator_file, sess_options=options, providers=["CPUExecutionProvider"])
    boxes = inference_detector(detector, bgr)
    author = Wholebody.__new__(Wholebody)
    author.session_det, author.session_pose = detector, estimator
    points, scores = author(bgr)
    (out / "pose-reference.json").write_text(json.dumps({"boxes": boxes.tolist(), "points": points.tolist(), "scores": scores.tolist()}, indent=2) + "\n")
    h, w = rgb.shape[:2]
    candidate = points.copy()
    candidate[..., 0] /= w
    candidate[..., 1] /= h
    body = candidate[:, :18].copy().reshape(-1, 2)
    subset = scores[:, :18].copy()
    for i in range(len(subset)):
        for j in range(18):
            subset[i, j] = 18 * i + j if subset[i, j] > .3 else -1
    candidate[scores < .3] = -1
    pose = {"bodies": {"candidate": body, "subset": subset}, "faces": candidate[:, 24:92], "hands": np.vstack([candidate[:, 92:113], candidate[:, 113:]])}
    PIL.Image.fromarray(draw_pose(pose, h, w)).save(out / "pose-reference.png")
    sources = list((args.depth_author_source / "depth_anything_v2").rglob("*.py")) + list((args.dwpose_author_source / "dwpose").glob("*.py"))
    manifest = {"torch": torch.__version__, "onnxruntime": ort.__version__, "opencv": cv2.__version__, "source_image": str(args.control_image), "source_sha256": sha256_file(args.control_image), "depth_policy": "518 square, shared RGB ImageNet-normalized f32 input, author model", "sha256": {str(p): sha256_file(p) for p in [Path(depth_file), Path(detector_file), Path(estimator_file)] + sources}}
    manifest["author_revisions"] = {name: (root / "revision").read_text().strip() if (root / "revision").is_file() else None for name, root in [("DepthAnything/Depth-Anything-V2", args.depth_author_source), ("IDEA-Research/DWPose", args.dwpose_author_source)]}
    manifest["fixture_sha256"] = {p.name: sha256_file(p) for p in sorted(out.iterdir()) if p.name != "meta.json"}
    (out / "meta.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[preprocess] wrote author references to {out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--stage", required=True, choices=["fp32", "bf16", "finalize", "transformer", "edits", "control", "lora", "preprocess"]
    )
    ap.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    ap.add_argument("--out-dir", type=Path, default=Path("/tmp/zimage-ref"))
    ap.add_argument("--threads", type=int, default=8)
    # fp32 is the acceptance reference and runs eager for a transparent, deterministic
    # softmax; bf16 runs sdpa because that is what the diffusers pipeline executes.
    ap.add_argument("--attn-impl", default=None)
    # The transformer stage.
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--prompt-idx", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--strength", type=float, default=0.6)
    ap.add_argument("--author-source", type=Path)
    ap.add_argument("--control-file", type=Path)
    ap.add_argument("--control-image", type=Path)
    ap.add_argument("--lora-file", type=Path)
    ap.add_argument("--lora-weight", type=float, default=0.8)
    ap.add_argument("--depth-author-source", type=Path)
    ap.add_argument("--dwpose-author-source", type=Path)
    ap.add_argument("--preprocess-input", type=Path)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.stage == "preprocess":
        dump_preprocess_reference(args)
        return
    if args.stage in ("control", "lora"):
        dump_control_or_lora(args)
        return
    if args.stage == "finalize":
        finalize(args.out_dir)
        return
    if args.stage == "edits":
        run_image_edits(args.snapshot, args.out_dir, args.seed, args.strength)
        return
    if args.stage == "transformer":
        run_transformer(
            args.snapshot, args.out_dir, args.dtype, args.width, args.height,
            args.prompt_idx, args.seed,
        )
        return
    attn = args.attn_impl or ("eager" if args.stage == "fp32" else "sdpa")
    run_stage(args.stage, args.snapshot, args.out_dir, attn)


if __name__ == "__main__":
    main()
