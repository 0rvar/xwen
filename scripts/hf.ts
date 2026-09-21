// Shared Hugging Face hub-cache resolution for the repo's bun scripts.
// Mirrors src/hub.rs (the Rust resolver) — same repo constants, same cache
// precedence, same layout walk. Read-only: downloads are the binary's job
// (`xwen fetch`, or lazily on any default-model run).

import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";

/** The checkpoints, mirroring the `Checkpoint` consts in src/hub.rs.
 *  `flash-next` is the default everywhere no checkpoint is named, matching the
 *  CLI's `--model` default. `drafter` is null for a checkpoint whose release
 *  ships no sidecar at all; every current one ships one, in one of the two
 *  KINDS — a DFlash block drafter on the 3.6 pair, an MTP head on the 3.8 —
 *  which the scripts only care about where the kind changes what a run costs. */
export const CHECKPOINTS = {
  "35b": {
    repo: "ggml-org/Qwen3.6-35B-A3B-GGUF",
    model: "Qwen3.6-35B-A3B-Q4_K_M.gguf",
    drafter: "dflash-Qwen3.6-35B-A3B-BF16.gguf",
  },
  "27b": {
    repo: "ggml-org/Qwen3.6-27B-GGUF",
    model: "Qwen3.6-27B-Q4_K_M.gguf",
    drafter: "dflash-Qwen3.6-27B-BF16.gguf",
  },
  "3.8-27b": {
    repo: "ggml-org/Qwen3.8-27B-GGUF",
    model: "Qwen3.8-27B-Q4_K_M.gguf",
    drafter: "mtp-Qwen3.8-27B-Q8_0.gguf",
  },
  // The one SPLIT checkpoint: four shards, and `model` is shard 1 because that
  // is what every consumer passes on a command line — the loader (and
  // llama.cpp) walk to the siblings from there. `shards` lists all four and
  // `officialModel` insists on every one of them: an interrupted 111 GB fetch
  // leaves shard 1 cached and the rest not, which resolves as a hit against the
  // entry point alone and then fails deep in the load.
  // No drafter: the release ships an MTP head xwen does not load (hub.rs).
  "flash-next": {
    repo: "unsloth/Qwen3.8-Flash-Next-GGUF",
    model: "UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf",
    shards: [
      "UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf",
      "UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf",
      "UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf",
      "UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf",
    ],
    drafter: null,
  },
  // The three SAFETENSORS checkpoints. `model` is `config.json` rather than a
  // weight file, because that is what the Rust registry lists first and what
  // `CheckpointSource` resolves the checkpoint DIRECTORY from — a safetensors
  // checkpoint is opened as a directory, not as a file. `files` is every file
  // that has to be cached for it to load, tokenizer included.
  //
  // Not runnable yet: the qwen3 layer stack is not implemented, so these are
  // here so that a script can find and fetch them, not so that one can bench
  // them. None ships a drafter and none ever will — no drafter exists for this
  // graph.
  "qwen3-4b": {
    repo: "Qwen/Qwen3-4B",
    model: "config.json",
    files: [
      "config.json",
      "model.safetensors.index.json",
      "model-00001-of-00003.safetensors",
      "model-00002-of-00003.safetensors",
      "model-00003-of-00003.safetensors",
      "tokenizer.json",
    ],
    drafter: null,
  },
  "qwen3-4b-instruct-2507": {
    repo: "Qwen/Qwen3-4B-Instruct-2507",
    model: "config.json",
    files: [
      "config.json",
      "model.safetensors.index.json",
      "model-00001-of-00003.safetensors",
      "model-00002-of-00003.safetensors",
      "model-00003-of-00003.safetensors",
      "tokenizer.json",
    ],
    drafter: null,
  },
  // The encoder lives in a subdirectory of a diffusion repo, with its tokenizer
  // in a sibling one — which is why every path here is relative to the REPO and
  // not to the checkpoint directory.
  "zimage-turbo-encoder": {
    repo: "Tongyi-MAI/Z-Image-Turbo",
    model: "text_encoder/config.json",
    files: [
      "text_encoder/config.json",
      "text_encoder/model.safetensors.index.json",
      "text_encoder/model-00001-of-00003.safetensors",
      "text_encoder/model-00002-of-00003.safetensors",
      "text_encoder/model-00003-of-00003.safetensors",
      "tokenizer/tokenizer.json",
    ],
    drafter: null,
  },
  // The whole Z-Image-Turbo pipeline at the repo root: the encoder entry's files
  // again plus the transformer (fp32, three shards), the VAE and the scheduler.
  // `model` is model_index.json so that its parent is the snapshot root.
  "zimage-turbo": {
    repo: "Tongyi-MAI/Z-Image-Turbo",
    model: "model_index.json",
    files: [
      "model_index.json",
      "text_encoder/config.json",
      "text_encoder/model.safetensors.index.json",
      "text_encoder/model-00001-of-00003.safetensors",
      "text_encoder/model-00002-of-00003.safetensors",
      "text_encoder/model-00003-of-00003.safetensors",
      "tokenizer/tokenizer.json",
      "transformer/config.json",
      "transformer/diffusion_pytorch_model.safetensors.index.json",
      "transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
      "transformer/diffusion_pytorch_model-00002-of-00003.safetensors",
      "transformer/diffusion_pytorch_model-00003-of-00003.safetensors",
      "vae/config.json",
      "vae/diffusion_pytorch_model.safetensors",
      "scheduler/scheduler_config.json",
    ],
    drafter: null,
  },
} as const;

export type ModelSize = keyof typeof CHECKPOINTS;

/** Every spelling the binary's `--model` takes for a registry checkpoint,
 *  lowercase, mapped to its canonical alias: `ALIASES` in src/hub.rs plus each
 *  checkpoint's full name. A test there reads this table, so the two cannot
 *  drift. Not every checkpoint has a `CHECKPOINTS` entry, only the ones a script
 *  has had to resolve; naming another is refused by name, not mistaken for a
 *  path. */
export const SPELLINGS: Record<string, string> = {
  "27b": "27b",
  "27": "27b",
  "qwen3.6-27b": "27b",
  "35b": "35b",
  "35": "35b",
  "35b-a3b": "35b",
  "qwen3.6-35b-a3b": "35b",
  "35b-uncensored": "35b-uncensored",
  "qwen3.6-35b-a3b-uncensored": "35b-uncensored",
  "3.8-27b": "3.8-27b",
  "38": "3.8-27b",
  "3.8": "3.8-27b",
  "qwen3.8-27b": "3.8-27b",
  "flash-next": "flash-next",
  "3.8-flash-next": "flash-next",
  "qwen3.8-flash-next": "flash-next",
  "qwen3-4b": "qwen3-4b",
  "4b": "qwen3-4b",
  "qwen3-4b-instruct-2507": "qwen3-4b-instruct-2507",
  "4b-instruct": "qwen3-4b-instruct-2507",
  "zimage-turbo-encoder": "zimage-turbo-encoder",
  "z-image-turbo-encoder": "zimage-turbo-encoder",
  "z-image-turbo-text-encoder": "zimage-turbo-encoder",
  "zimage-turbo": "zimage-turbo",
  "z-image-turbo": "zimage-turbo",
  "qwen-image-2.1-encoder": "qwen-image-2.1-encoder",
  "qwen-image-2.1-text-encoder": "qwen-image-2.1-encoder",
  "qwen-image-2.1": "qwen-image-2.1",
};

/** What `--model` and `$XWEN_MODEL` name, read the way the binary reads its
 *  own `--model`: a checkpoint name first, else a path. */
export type ModelRef = { kind: "registry"; size: ModelSize } | { kind: "path"; path: string };

export function parseModelRef(value: string): ModelRef {
  const alias = SPELLINGS[value.trim().toLowerCase()];
  if (alias !== undefined) {
    if (alias in CHECKPOINTS) return { kind: "registry", size: alias as ModelSize };
    throw new Error(
      `${alias} is a checkpoint xwen knows, but scripts/hf.ts has no cache entry for it ` +
        `(it resolves ${Object.keys(CHECKPOINTS).join("|")}); pass its path instead`,
    );
  }
  if (existsSync(value)) return { kind: "path", path: value };
  throw new Error(
    `${JSON.stringify(value)} is not a checkpoint name (${Object.keys(CHECKPOINTS).join("|")}, ` +
      `or a full name) and no such file or directory exists`,
  );
}

/** `$XWEN_MODEL`, when it is set. */
export function envModelRef(): ModelRef | null {
  const value = process.env.XWEN_MODEL;
  return value ? parseModelRef(value) : null;
}

/** The model a script runs: its `--model` value, else `$XWEN_MODEL`, else
 *  `fallback`. One precedence for every script, so none of them ignores the
 *  environment and none falls back past a path it was given. */
export function resolveModelRef(flagValue: string | undefined, fallback: ModelSize): ModelRef {
  if (flagValue !== undefined && flagValue !== "") return parseModelRef(flagValue);
  return envModelRef() ?? { kind: "registry", size: fallback };
}

/** Refuse any `--flag` a script does not read. A flag that is silently ignored
 *  runs the default instead of what was asked for and reports it as if it were:
 *  a stale `--model-size 27b` would grade the default checkpoint and call it the
 *  27B. Everything after a bare `--` is another program's and is not checked. */
export function rejectUnknownFlags(script: string, args: string[], known: readonly string[]): void {
  const end = args.indexOf("--");
  for (const arg of end >= 0 ? args.slice(0, end) : args) {
    if (!arg.startsWith("--")) continue;
    const name = arg.slice(2).split("=")[0];
    if (known.includes(name)) continue;
    const hint =
      name === "model-size"
        ? "; it was removed, --model takes the same alias (or a full name, or a path)"
        : ` (known: ${known.map((k) => `--${k}`).join(" ")})`;
    console.error(`${script}: unknown flag --${name}${hint}`);
    process.exit(2);
  }
}

/** The path a `ModelRef` loads from: its own, or the cached official file. */
export function modelPath(ref: ModelRef): string {
  return ref.kind === "path" ? ref.path : officialModel(ref.size);
}

export const OFFICIAL_REPO = CHECKPOINTS["35b"].repo;
export const OFFICIAL_MODEL = CHECKPOINTS["35b"].model;
export const OFFICIAL_DRAFTER = CHECKPOINTS["35b"].drafter;

/** `$HF_HUB_CACHE` > `$HF_HOME/hub` > `~/.cache/huggingface/hub`. */
export function hubCacheRoot(): string {
  if (process.env.HF_HUB_CACHE) return process.env.HF_HUB_CACHE;
  if (process.env.HF_HOME) return join(process.env.HF_HOME, "hub");
  return join(process.env.HOME ?? "", ".cache/huggingface/hub");
}

export function repoDir(repo: string): string {
  return join(hubCacheRoot(), `models--${repo.replace("/", "--")}`);
}

/** Cached path of `file` in `repo` at the `main` ref, or null. Never
 *  downloads. refs/main is required, exactly as hf-hub resolves it — every
 *  writer (`hf download`, the hf-hub crate) creates it. */
export function cachedFile(repo: string, file: string): string | null {
  const dir = repoDir(repo);
  let commit: string;
  try {
    // Verbatim, no trim — hf-hub reads the ref the same way, and the two
    // resolvers must miss identically on a malformed (whitespace-bearing) ref
    // or the scripts would report a hit the binary can't see.
    commit = readFileSync(join(dir, "refs/main"), "utf8");
  } catch {
    return null;
  }
  const path = join(dir, "snapshots", commit, file);
  // existsSync follows symlinks, so a dangling blob link is a miss.
  return existsSync(path) ? path : null;
}

/** Every file that has to be cached for `size` to load: every file of a
 *  safetensors set, the shard set on a split GGUF, the single file otherwise. */
export function modelFiles(size: ModelSize): readonly string[] {
  const ck = CHECKPOINTS[size];
  if ("files" in ck) return ck.files;
  if ("shards" in ck) return ck.shards;
  return [ck.model];
}

/** The official model for `size`, or throw with the fix.
 *
 *  Returns the ENTRY POINT (shard 1 on a split checkpoint) but insists on the
 *  WHOLE set: the loader walks to the siblings from whatever file it is handed,
 *  so a cache holding only shard 1 resolves here and then fails deep inside the
 *  load. A half-finished 111 GB download is exactly the state this catches, and
 *  it names the missing shards rather than reporting a hit. */
export function officialModel(size: ModelSize): string {
  const ck = CHECKPOINTS[size];
  const files = modelFiles(size);
  const missing = files.filter((file) => !cachedFile(ck.repo, file));
  if (missing.length > 0) {
    const what =
      files.length === 1
        ? `${ck.repo}/${ck.model} is not in the Hugging Face cache`
        : `${missing.length} of ${ck.repo}'s ${files.length} shards are not in the Hugging ` +
          `Face cache (${missing.join(", ")})`;
    throw new Error(
      `${what}; run \`xwen fetch --model ${size}\` (or pass --model <path> / $XWEN_MODEL)`,
    );
  }
  return cachedFile(ck.repo, ck.model)!;
}

/** The official drafter for `size`, or null — which also covers a checkpoint
 *  whose release ships no sidecar at all. */
export function officialDrafter(size: ModelSize): string | null {
  const ck = CHECKPOINTS[size];
  return ck.drafter ? cachedFile(ck.repo, ck.drafter) : null;
}

/** The checkpoints that can speculate, for sweeps that only mean those. */
export function draftingSizes(): ModelSize[] {
  return (Object.keys(CHECKPOINTS) as ModelSize[]).filter((size) => CHECKPOINTS[size].drafter);
}

// CLI: `bun scripts/hf.ts [model|drafter] [27b|35b|3.8-27b|flash-next]` prints
// the resolved cache path, for shell commands that need an explicit path
// (ref-dump.sh, llama-server, …). The checkpoint is the second argument, else
// $XWEN_MODEL, else flash-next; a path names itself and has no drafter here.
if (import.meta.main) {
  const which = process.argv[2] ?? "model";
  let ref: ModelRef;
  try {
    ref = resolveModelRef(process.argv[3], "flash-next");
  } catch (e) {
    console.error(`hf.ts: ${(e as Error).message}`);
    process.exit(2);
  }
  if (which === "model") {
    console.log(modelPath(ref));
  } else if (which === "drafter") {
    if (ref.kind === "path") {
      console.error(`${ref.path} is a path; only a registry checkpoint has an official drafter`);
      process.exit(1);
    }
    const size = ref.size;
    const drafter = CHECKPOINTS[size].drafter;
    if (!drafter) {
      console.error(`${size} ships no drafter sidecar`);
      process.exit(1);
    }
    const path = officialDrafter(size);
    if (!path) {
      console.error(
        `${CHECKPOINTS[size].repo}/${drafter} is not in the Hugging Face cache; ` +
          `run \`xwen fetch --model ${size}\``,
      );
      process.exit(1);
    }
    console.log(path);
  } else {
    console.error(`usage: bun scripts/hf.ts [model|drafter] [${Object.keys(CHECKPOINTS).join("|")}]`);
    process.exit(2);
  }
}
