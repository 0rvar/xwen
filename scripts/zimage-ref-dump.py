#!/usr/bin/env python3
# Stage 2 reference dump for xwen's Z-Image text-encoder path (docs/zimage.md, plan D10).
#
#   uv venv /tmp/zimage-venv --python 3.12
#   uv pip install --python /tmp/zimage-venv/bin/python torch transformers safetensors numpy
#   /tmp/zimage-venv/bin/python scripts/zimage-ref-dump.py --stage fp32 \
#     && /tmp/zimage-venv/bin/python scripts/zimage-ref-dump.py --stage bf16 \
#     && /tmp/zimage-venv/bin/python scripts/zimage-ref-dump.py --stage finalize
#
# This is the only Python in the repo. It exists because the Z-Image text encoder has no
# ONNX export and there is no bun path to torch; it runs once, by hand, to produce the
# reference `tests/qwen3_encoder.rs` grades xwen against. It never runs in CI.
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True, choices=["fp32", "bf16", "finalize"])
    ap.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    ap.add_argument("--out-dir", type=Path, default=Path("/tmp/zimage-ref"))
    ap.add_argument("--threads", type=int, default=8)
    # fp32 is the acceptance reference and runs eager for a transparent, deterministic
    # softmax; bf16 runs sdpa because that is what the diffusers pipeline executes.
    ap.add_argument("--attn-impl", default=None)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.stage == "finalize":
        finalize(args.out_dir)
        return
    attn = args.attn_impl or ("eager" if args.stage == "fp32" else "sdpa")
    run_stage(args.stage, args.snapshot, args.out_dir, attn)


if __name__ == "__main__":
    main()
