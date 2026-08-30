// Shared Hugging Face hub-cache resolution for the repo's bun scripts.
// Mirrors src/hub.rs (the Rust resolver) — same repo constants, same cache
// precedence, same layout walk. Read-only: downloads are the binary's job
// (`xwen fetch`, or lazily on any default-model run).

import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";

/** The checkpoints, mirroring the `Checkpoint` consts in src/hub.rs.
 *  `flash-next` is the default everywhere the size is not named, matching the
 *  CLI's `--model-size` default. `drafter` is null for a checkpoint whose release
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
} as const;

export type ModelSize = keyof typeof CHECKPOINTS;

/** `$XWEN_MODEL_SIZE`, else `flash-next` — the binary's own default. A script
 *  that CANNOT run the default checkpoint (parity-gate, which has no oracle for
 *  the qwen4exp graph) must name its size rather than lean on this. Throws on an
 *  unknown size. */
export function defaultSize(): ModelSize {
  const s = (process.env.XWEN_MODEL_SIZE ?? "flash-next").toLowerCase();
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

/** Every file that has to be cached for `size` to load: the shard set on a
 *  split checkpoint, the single file on the others. */
export function modelFiles(size: ModelSize): readonly string[] {
  const ck = CHECKPOINTS[size];
  return "shards" in ck ? ck.shards : [ck.model];
}

/** The official model for `size`, or throw with the fix.
 *
 *  Returns the ENTRY POINT (shard 1 on a split checkpoint) but insists on the
 *  WHOLE set: the loader walks to the siblings from whatever file it is handed,
 *  so a cache holding only shard 1 resolves here and then fails deep inside the
 *  load. A half-finished 111 GB download is exactly the state this catches, and
 *  it names the missing shards rather than reporting a hit. */
export function officialModel(size: ModelSize = defaultSize()): string {
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
      `${what}; run \`xwen fetch --model-size ${size}\` (or pass --model / $XWEN_MODEL)`,
    );
  }
  return cachedFile(ck.repo, ck.model)!;
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

// CLI: `bun scripts/hf.ts [model|drafter] [27b|35b|3.8-27b|flash-next]` prints
// the resolved cache path, for shell commands that need an explicit path
// (ref-dump.sh, llama-server, …). The size defaults to $XWEN_MODEL_SIZE, else
// flash-next.
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
