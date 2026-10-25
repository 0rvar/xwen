#!/usr/bin/env bun
// The Flint parity oracle: llama.cpp's logits at every position of a few
// prompts, reduced to the committed fixture tests/fixtures/flint-parity.json
// that tests/flint_parity.rs grades xwen's forward pass against.
//
// The prompts and their ids come from tests/fixtures/flint-prompts.json (ids
// from `llama-tokenize` of a Flint-capable llama.cpp), so this needs no xwen
// tokenizer. One more prompt, `long-1400`, is the first 1400 ids of that
// fixture's `bench-prefill-4k`: long enough to wrap the 513-slot sliding
// window twice and to be prefilled in several chunks.
//
// The full logits ([n_tokens, 150144] f32, 600 MB per 1000 tokens) stay in
// --out-dir; the fixture keeps, per position, the oracle's top-10 ids and
// logits. Rerunning writes a byte-identical fixture for the same oracle.
//
// The oracle is the Flint checkout, reference/llama.cpp-flint at a37cc4610 (the
// Flint merge; the main pin predates it), built with
// `scripts/build-llamacpp.sh reference/llama.cpp-flint`. XWEN_LLAMACPP_BIN
// overrides the bin/ directory.
//
// The backend is Metal (all layers offloaded) unless --n-gpu-layers says
// otherwise, and the fixture records which. tests/flint_parity.rs refuses a
// fixture that was not produced on Metal: llama.cpp's CPU backend sits 0.61
// mean |d| from its Metal one on this model, which is the whole width of the
// gate's bars, so a CPU fixture would grade the oracle's backend and not xwen.
//
// Usage: bun scripts/flint-ref-logits.ts --out-dir <dir> [--n-gpu-layers N]
//          [--model <gguf>] [--fixture <json>]

import { spawnSync } from "bun";
import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const REPO = new URL("..", import.meta.url).pathname.replace(/\/$/, "");
const TOP = 10;
const PROMPTS = [
  "parity-code-short",
  "parity-text-mixed",
  "edge-marker-text",
  "edge-mixed-script-code",
  "corpus-tail",
  "parity-long-mixed",
];
const LONG = { id: "long-1400", from: "bench-prefill-4k", n: 1400 };

const args = process.argv.slice(2);
const argOf = (flag: string) => {
  const at = args.indexOf(flag);
  return at >= 0 ? args[at + 1] : undefined;
};

function oracleBin(name: string): string {
  const dir = process.env.XWEN_LLAMACPP_BIN ?? `${REPO}/reference/llama.cpp-flint/build/bin`;
  const path = join(dir, name);
  if (!existsSync(path)) {
    throw new Error(`no ${path}: run scripts/build-llamacpp.sh reference/llama.cpp-flint, or set XWEN_LLAMACPP_BIN`);
  }
  return path;
}

function findModel(): string | null {
  const hub =
    process.env.HF_HUB_CACHE ??
    (process.env.HF_HOME ? join(process.env.HF_HOME, "hub") : null) ??
    join(process.env.HOME ?? "", ".cache/huggingface/hub");
  const snapshots = join(hub, "models--ggml-org--Flint-GGUF", "snapshots");
  if (!existsSync(snapshots)) return null;
  for (const snap of readdirSync(snapshots)) {
    const file = join(snapshots, snap, "Flint-Q4_K_M.gguf");
    if (existsSync(file)) return file;
  }
  return null;
}

function sha256(path: string): string {
  const out = spawnSync(["shasum", "-a", "256", path]);
  if (!out.success) throw new Error(`shasum failed for ${path}`);
  return out.stdout.toString().trim().split(/\s+/)[0]!;
}

const outDir = argOf("--out-dir");
if (!outDir) {
  console.error("usage: bun scripts/flint-ref-logits.ts --out-dir <dir> [--n-gpu-layers N]");
  process.exit(1);
}
const model = argOf("--model") ?? findModel();
if (!model) {
  console.error("no Flint-Q4_K_M.gguf in the HF cache; `xwen fetch --model flint`");
  process.exit(1);
}
const nGpuLayers = argOf("--n-gpu-layers") ?? "99";
const fixturePath = argOf("--fixture") ?? `${REPO}/tests/fixtures/flint-parity.json`;
const TOOL = oracleBin("llama-logits-all");
const clone = TOOL.replace(/\/build\/bin\/[^/]+$/, "");
const commit = (() => {
  const out = spawnSync(["git", "-C", clone, "rev-parse", "HEAD"]);
  return out.success ? out.stdout.toString().trim() : "unknown";
})();

const source = JSON.parse(readFileSync(`${REPO}/tests/fixtures/flint-prompts.json`, "utf8")) as {
  vocab: number;
  prompts: { id: string; ids: number[] }[];
};
const byId = (id: string) => {
  const p = source.prompts.find((q) => q.id === id);
  if (!p) throw new Error(`flint-prompts.json has no prompt ${id}`);
  return p.ids;
};
const prompts = [
  ...PROMPTS.map((id) => ({ id, ids: byId(id) })),
  { id: LONG.id, ids: byId(LONG.from).slice(0, LONG.n) },
];

mkdirSync(outDir, { recursive: true });
console.log(`tool      ${TOOL}\nllama.cpp ${commit}\nmodel     ${model}`);
const modelSha = sha256(model);
const vocab = source.vocab;

const out: unknown[] = [];
for (const [i, prompt] of prompts.entries()) {
  const prefix = join(outDir, `prompt-${i}`);
  writeFileSync(`${prefix}.ids.txt`, prompt.ids.join("\n") + "\n");
  const t0 = Date.now();
  const run = spawnSync(
    [
      TOOL,
      "--model", model,
      "--ids", `${prefix}.ids.txt`,
      "--out", prefix,
      "--n-gpu-layers", nGpuLayers,
      "--batch", "512",
      "--kv-type", "f16",
      "--flash-attn", "auto",
      "--threads", "8",
      "--expect-vocab", String(vocab),
      "--model-sha256", modelSha,
      "--llamacpp-commit", commit,
    ],
    { stdout: "inherit", stderr: "pipe" },
  );
  if (!run.success) {
    process.stderr.write(run.stderr.toString());
    throw new Error(`prompt ${prompt.id} failed`);
  }
  const raw = new Float32Array(readFileSync(`${prefix}.f32`).buffer.slice(0));
  if (raw.length !== prompt.ids.length * vocab) {
    throw new Error(`${prefix}.f32 holds ${raw.length} floats, expected ${prompt.ids.length * vocab}`);
  }
  const top: [number, number][][] = [];
  for (let p = 0; p < prompt.ids.length; p++) {
    const row = raw.subarray(p * vocab, (p + 1) * vocab);
    const best: [number, number][] = [];
    for (let id = 0; id < vocab; id++) {
      const v = row[id]!;
      if (best.length === TOP && !(v > best[TOP - 1]![1])) continue;
      let at = best.length;
      while (at > 0 && v > best[at - 1]![1]) at--;
      best.splice(at, 0, [id, v]);
      if (best.length > TOP) best.pop();
    }
    // Five decimals is far below any bar the test applies.
    top.push(best.map(([id, v]) => [id, Math.round(v * 1e5) / 1e5]));
  }
  out.push({ id: prompt.id, ids: prompt.ids, top });
  console.log(`${prompt.id.padEnd(24)} ${String(prompt.ids.length).padStart(5)} tok  ${((Date.now() - t0) / 1000).toFixed(1)} s`);
}

writeFileSync(
  fixturePath,
  JSON.stringify(
    {
      note:
        "Flint parity oracle, written by scripts/flint-ref-logits.ts: per prompt and position, " +
        `llama.cpp's top-${TOP} (id, logit) over the next token, best first.`,
      model_file: "ggml-org/Flint-GGUF Flint-Q4_K_M.gguf",
      model_sha256: modelSha,
      llamacpp_commit: commit,
      backend: nGpuLayers === "0" ? "cpu" : `metal (n_gpu_layers ${nGpuLayers})`,
      vocab,
      top_k: TOP,
      prompts: out,
    },
    null,
    0,
  ) + "\n",
);
console.log(`wrote ${fixturePath}`);
