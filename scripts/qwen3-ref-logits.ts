#!/usr/bin/env bun
// Drives `llama-logits-all` over tests/fixtures/qwen3-prompts.json, producing the
// Stage-1 reference logits that xwen's dense Qwen3-4B forward pass is compared
// against, position by position.
//
// One tool invocation per prompt, one llama process at a time. The default is
// --n-gpu-layers 0, i.e. CPU: the oracle's arithmetic is part of what is being
// pinned (llama.cpp's CPU BF16 path and its Metal path do not agree bit for bit),
// so the backend is recorded per prompt rather than assumed.
//
// Outputs, per prompt index i: <dir>/prompt-<i>.f32, .json, .argmax.json and
// .argmax.txt, plus <dir>/manifest.json tying them to the fixture, the GGUF's
// sha256 and the llama.cpp commit that produced them.
//
// Usage: bun scripts/qwen3-ref-logits.ts --out-dir <dir> [--only 0,3,7]
//                                        [--n-gpu-layers N] [--model <gguf>]
//                                        [--batch N] [--kv-type f16|f32]
//                                        [--flash-attn on|off|auto] [--threads N]
//                                        [--force]

import { spawnSync } from "bun";
import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const REPO = new URL("..", import.meta.url).pathname.replace(/\/$/, "");

/** The built oracle. `reference/llama.cpp` is a submodule, so a worktree that
 *  has not initialized it borrows the main checkout's build; XWEN_LLAMACPP_BIN
 *  overrides both. */
function oracleBin(name: string): string {
  const candidates = [
    process.env.XWEN_LLAMACPP_BIN ? join(process.env.XWEN_LLAMACPP_BIN, name) : null,
    `${REPO}/reference/llama.cpp/build/bin/${name}`,
    join(process.env.HOME ?? "", `develop/private/xwen/reference/llama.cpp/build/bin/${name}`),
  ].filter((p): p is string => p !== null);
  for (const path of candidates) if (existsSync(path)) return path;
  throw new Error(
    `no built ${name}; run scripts/build-llamacpp.sh or set XWEN_LLAMACPP_BIN to a build's bin/`,
  );
}

/** The clone that produced the oracle binary, so the manifest can record its commit. */
function oracleRepo(binPath: string): string {
  return binPath.replace(/\/build\/bin\/[^/]+$/, "");
}

/** The one true BF16 Qwen3-4B GGUF in the HF cache, or null when absent. */
function findModel(): string | null {
  const hub =
    process.env.HF_HUB_CACHE ??
    (process.env.HF_HOME ? join(process.env.HF_HOME, "hub") : null) ??
    join(process.env.HOME ?? "", ".cache/huggingface/hub");
  const snapshots = join(hub, "models--unsloth--Qwen3-4B-GGUF", "snapshots");
  if (!existsSync(snapshots)) return null;
  for (const snap of readdirSync(snapshots)) {
    const file = join(snapshots, snap, "Qwen3-4B-BF16.gguf");
    if (existsSync(file)) return file;
  }
  return null;
}

function sha256(path: string): string {
  const out = spawnSync(["shasum", "-a", "256", path]);
  if (!out.success) throw new Error(`shasum failed for ${path}`);
  return out.stdout.toString().trim().split(/\s+/)[0]!;
}

const args = process.argv.slice(2);
const argOf = (flag: string) => {
  const at = args.indexOf(flag);
  return at >= 0 ? args[at + 1] : undefined;
};
const has = (flag: string) => args.includes(flag);

const outDir = argOf("--out-dir");
if (!outDir) {
  console.error("usage: bun scripts/qwen3-ref-logits.ts --out-dir <dir> [--only 0,3,7] [--n-gpu-layers N] [--force]");
  process.exit(1);
}
const model = argOf("--model") ?? findModel();
if (!model) {
  console.error(
    "no Qwen3-4B-BF16.gguf in the HF cache; pass --model <path> or fetch unsloth/Qwen3-4B-GGUF",
  );
  process.exit(1);
}

const nGpuLayers = argOf("--n-gpu-layers") ?? "0";
const batch = argOf("--batch") ?? "512";
const kvType = argOf("--kv-type") ?? "f16";
const flashAttn = argOf("--flash-attn") ?? "auto";
const threads = argOf("--threads") ?? "8";

const TOOL = oracleBin("llama-logits-all");
const CLONE = oracleRepo(TOOL);

const fixturePath = `${REPO}/tests/fixtures/qwen3-prompts.json`;
const fixture = JSON.parse(readFileSync(fixturePath, "utf8")) as {
  vocab: number;
  prompts: { id: string; source: string; text: string; ids: number[] }[];
};

const only = argOf("--only")
  ?.split(",")
  .map((s) => Number(s.trim()))
  .filter((n) => Number.isInteger(n));
if (only) {
  for (const idx of only) {
    if (idx < 0 || idx >= fixture.prompts.length) {
      console.error(`--only names prompt ${idx}, but the fixture has ${fixture.prompts.length}`);
      process.exit(1);
    }
  }
}
const indices = only ?? fixture.prompts.map((_, i) => i);

const manifestPath = join(outDir, "manifest.json");
if (existsSync(manifestPath) && !has("--force")) {
  console.error(`${manifestPath} already exists; pass --force to overwrite it`);
  process.exit(1);
}
mkdirSync(outDir, { recursive: true });

const commit = (() => {
  const out = spawnSync(["git", "-C", CLONE, "rev-parse", "HEAD"]);
  return out.success ? out.stdout.toString().trim() : "unknown";
})();

console.log(`tool     ${TOOL}`);
console.log(`llama.cpp ${commit}`);
console.log(`model    ${model}`);
process.stdout.write("hashing the GGUF... ");
const modelSha = sha256(model);
console.log(modelSha);

type Entry = {
  idx: number;
  id: string;
  n_tokens: number;
  files: { f32: string; json: string; argmax: string; argmax_txt: string };
  f32_sha256: string;
  f32_bytes: number;
  wall_seconds: number;
  tool_json: unknown;
};

const entries: Entry[] = [];

for (const idx of indices) {
  const prompt = fixture.prompts[idx]!;
  const prefix = join(outDir, `prompt-${idx}`);
  const idsPath = `${prefix}.ids.txt`;
  writeFileSync(idsPath, prompt.ids.join("\n") + "\n");

  const t0 = Date.now();
  const run = spawnSync(
    [
      TOOL,
      "--model", model,
      "--ids", idsPath,
      "--out", prefix,
      "--n-gpu-layers", nGpuLayers,
      "--batch", batch,
      "--kv-type", kvType,
      "--flash-attn", flashAttn,
      "--threads", threads,
      "--expect-vocab", String(fixture.vocab),
      "--model-sha256", modelSha,
      "--llamacpp-commit", commit,
    ],
    { stdout: "inherit", stderr: "pipe" },
  );
  const wall = (Date.now() - t0) / 1000;
  if (!run.success) {
    process.stderr.write(run.stderr.toString());
    console.error(`prompt ${idx} (${prompt.id}) failed`);
    process.exit(1);
  }

  const toolJson = JSON.parse(readFileSync(`${prefix}.json`, "utf8")) as {
    n_tokens: number;
    n_vocab: number;
  };
  const bytes = Bun.file(`${prefix}.f32`).size;
  const expected = toolJson.n_tokens * toolJson.n_vocab * 4;
  if (bytes !== expected) {
    console.error(
      `prompt ${idx}: ${prefix}.f32 is ${bytes} bytes, expected ${expected} ` +
        `(${toolJson.n_tokens} x ${toolJson.n_vocab} x 4)`,
    );
    process.exit(1);
  }
  if (toolJson.n_tokens !== prompt.ids.length) {
    console.error(
      `prompt ${idx}: tool saw ${toolJson.n_tokens} tokens, fixture has ${prompt.ids.length}`,
    );
    process.exit(1);
  }

  entries.push({
    idx,
    id: prompt.id,
    n_tokens: toolJson.n_tokens,
    files: {
      f32: `prompt-${idx}.f32`,
      json: `prompt-${idx}.json`,
      argmax: `prompt-${idx}.argmax.json`,
      argmax_txt: `prompt-${idx}.argmax.txt`,
    },
    f32_sha256: sha256(`${prefix}.f32`),
    f32_bytes: bytes,
    wall_seconds: wall,
    tool_json: JSON.parse(readFileSync(`${prefix}.json`, "utf8")),
  });

  console.log(
    `prompt ${String(idx).padStart(2)} ${prompt.id.padEnd(26)} ` +
      `${String(toolJson.n_tokens).padStart(5)} tok  ${wall.toFixed(1)} s  ` +
      `${(bytes / 1e9).toFixed(2)} GB`,
  );
}

writeFileSync(
  manifestPath,
  JSON.stringify(
    {
      note:
        "Stage-1 reference logits for the dense Qwen3-4B parity gate. Written by " +
        "scripts/qwen3-ref-logits.ts; each .f32 is [n_tokens, n_vocab] row-major f32.",
      generated: new Date().toISOString(),
      fixture: "tests/fixtures/qwen3-prompts.json",
      model,
      model_sha256: modelSha,
      llamacpp_commit: commit,
      tool: TOOL,
      settings: {
        n_gpu_layers: Number(nGpuLayers),
        batch: Number(batch),
        kv_type: kvType,
        flash_attn: flashAttn,
        threads: Number(threads),
      },
      prompts: entries,
    },
    null,
    2,
  ) + "\n",
);

const total = entries.reduce((a, e) => a + e.wall_seconds, 0);
console.log(`\nwrote ${entries.length} prompt(s) to ${outDir} in ${total.toFixed(1)} s total`);
console.log(`manifest ${manifestPath}`);
