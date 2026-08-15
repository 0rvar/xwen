// Shared Hugging Face hub-cache resolution for the repo's bun scripts.
// Mirrors src/hub.rs (the Rust resolver) — same repo constants, same cache
// precedence, same layout walk. Read-only: downloads are the binary's job
// (`xwen fetch`, or lazily on any default-model run).

import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";

/** The checkpoints, mirroring the `Checkpoint` consts in src/hub.rs. `35b` is
 *  the default everywhere the size is not named, matching the CLI's
 *  `--model-size` default. `drafter` is null for a checkpoint whose release
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
} as const;

export type ModelSize = keyof typeof CHECKPOINTS;

/** `$XWEN_MODEL_SIZE`, else `35b`. Throws on an unknown size. */
export function defaultSize(): ModelSize {
  const s = (process.env.XWEN_MODEL_SIZE ?? "35b").toLowerCase();
  if (s in CHECKPOINTS) return s as ModelSize;
  throw new Error(`unknown model size ${s} (expected ${Object.keys(CHECKPOINTS).join("|")})`);
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

/** The official model for `size`, or throw with the fix. */
export function officialModel(size: ModelSize = defaultSize()): string {
  const ck = CHECKPOINTS[size];
  const path = cachedFile(ck.repo, ck.model);
  if (!path) {
    throw new Error(
      `${ck.repo}/${ck.model} is not in the Hugging Face cache; ` +
        `run \`xwen fetch --model-size ${size}\` (or pass --model / $XWEN_MODEL)`,
    );
  }
  return path;
}

/** The official drafter for `size`, or null — which also covers a checkpoint
 *  whose release ships no sidecar at all. */
export function officialDrafter(size: ModelSize = defaultSize()): string | null {
  const ck = CHECKPOINTS[size];
  return ck.drafter ? cachedFile(ck.repo, ck.drafter) : null;
}

/** The checkpoints that can speculate, for sweeps that only mean those. */
export function draftingSizes(): ModelSize[] {
  return (Object.keys(CHECKPOINTS) as ModelSize[]).filter((size) => CHECKPOINTS[size].drafter);
}

// CLI: `bun scripts/hf.ts [model|drafter] [27b|35b]` prints the resolved cache
// path, for shell commands that need an explicit path (ref-dump.sh,
// llama-server, …). The size defaults to $XWEN_MODEL_SIZE, else 35b.
if (import.meta.main) {
  const which = process.argv[2] ?? "model";
  const size = (process.argv[3] as ModelSize | undefined) ?? defaultSize();
  if (!(size in CHECKPOINTS)) {
    console.error(`unknown model size ${size} (expected ${Object.keys(CHECKPOINTS).join("|")})`);
    process.exit(2);
  }
  if (which === "model") {
    console.log(officialModel(size));
  } else if (which === "drafter") {
    const drafter = CHECKPOINTS[size].drafter;
    if (!drafter) {
      console.error(`${size} ships no drafter sidecar`);
      process.exit(1);
    }
    const path = officialDrafter(size);
    if (!path) {
      console.error(
        `${CHECKPOINTS[size].repo}/${drafter} is not in the Hugging Face cache; ` +
          `run \`xwen fetch --model-size ${size}\``,
      );
      process.exit(1);
    }
    console.log(path);
  } else {
    console.error(`usage: bun scripts/hf.ts [model|drafter] [${Object.keys(CHECKPOINTS).join("|")}]`);
    process.exit(2);
  }
}
