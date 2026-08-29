#!/usr/bin/env bun
/**
 * hf-fetch — populate the local Hugging Face hub cache in the exact layout the Rust
 * `hf-hub` crate expects, so a cache-first loader finds the files without touching
 * the network.
 *
 *   bun scripts/hf-fetch.ts <repo> <path...> [--jobs N] [--verify] [--dry-run]
 *
 *   bun scripts/hf-fetch.ts unsloth/Qwen3.8-Flash-Next-GGUF \
 *     UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf --jobs 2
 *
 * Layout produced under <cache>/models--<org>--<name>/:
 *   blobs/<etag>                            the file itself, named by its LFS oid
 *   snapshots/<commit-sha>/<path>           relative symlink into ../../blobs/
 *   refs/main                               the commit sha, NO trailing newline
 *
 * The missing newline is load-bearing: `CacheRepo::get` reads refs/main with
 * `read_to_string` and pushes the result onto the snapshot path WITHOUT trimming
 * (hf-hub 0.5.0 src/lib.rs), so a stray "\n" makes every lookup miss and
 * re-download the whole repo.
 *
 * The cache root follows `$HF_HUB_CACHE` > `$HF_HOME/hub` > ~/.cache/huggingface/hub,
 * matching xwen's own `hub::hub_cache_root` (hf-hub's `Cache::from_env` skips
 * HF_HUB_CACHE; xwen does not, so this script follows xwen).
 *
 * Downloads resume (`curl -C -`), are verified against the LFS oid with sha256
 * before they are moved into place, and are skipped when the blob is already there
 * at the right size (pass --verify to re-hash existing blobs anyway). Files stored
 * in git rather than LFS have a sha1 git-blob etag, not a sha256, so for those the
 * size is the only check available.
 */

import { spawn } from "node:child_process";
import { mkdirSync, statSync, lstatSync, symlinkSync, unlinkSync, rmSync, renameSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";

/** `oid` is the blob's name in the cache: an LFS sha256, or a git sha1 when `lfs` is false. */
type FileMeta = { path: string; size: number; oid: string; lfs: boolean };

function cacheRoot(): string {
  if (process.env.HF_HUB_CACHE) return process.env.HF_HUB_CACHE;
  if (process.env.HF_HOME) return join(process.env.HF_HOME, "hub");
  return join(homedir(), ".cache", "huggingface", "hub");
}

function repoDirName(repo: string): string {
  return "models--" + repo.replace(/\//g, "--");
}

function human(bytes: number): string {
  if (bytes >= 1e9) return (bytes / 1e9).toFixed(2) + " GB";
  if (bytes >= 1e6) return (bytes / 1e6).toFixed(1) + " MB";
  if (bytes >= 1e3) return (bytes / 1e3).toFixed(1) + " kB";
  return `${bytes} B`;
}

function ts(): string {
  return new Date().toISOString().replace("T", " ").slice(0, 19);
}

function log(msg: string) {
  console.log(`[${ts()}] ${msg}`);
}

function authHeaderMap(): Record<string, string> {
  const token = process.env.HF_TOKEN ?? process.env.HUGGING_FACE_HUB_TOKEN;
  return token ? { Authorization: `Bearer ${token}` } : {};
}

function authHeaders(): string[] {
  return Object.entries(authHeaderMap()).flatMap(([k, v]) => ["-H", `${k}: ${v}`]);
}

/** Percent-encode each segment; `encodeURI` alone leaves `#`, `?` and `&` intact. */
function encodePath(path: string): string {
  return path.split("/").map(encodeURIComponent).join("/");
}

function run(cmd: string, args: string[]): Promise<{ code: number; stdout: string }> {
  return new Promise((resolve) => {
    const p = spawn(cmd, args, { stdio: ["ignore", "pipe", "inherit"] });
    let out = "";
    p.stdout?.on("data", (d) => (out += d));
    p.on("close", (code) => resolve({ code: code ?? -1, stdout: out }));
  });
}

async function apiGet(url: string): Promise<Response> {
  const res = await fetch(url, { headers: authHeaderMap() });
  if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText} for ${url}`);
  return res;
}

/** Commit sha that `refs/main` must point at. */
async function resolveCommit(repo: string): Promise<string> {
  const info: any = await (await apiGet(`https://huggingface.co/api/models/${repo}`)).json();
  if (!info.sha) throw new Error(`no sha in model info for ${repo}`);
  return info.sha;
}

/**
 * The whole repo tree. `expand=true` caps the page size at 50 (and rejects a larger
 * `limit` outright with a 400), so the `Link: <...>; rel="next"` header must be
 * followed or every file past the first fifty entries looks like it does not exist.
 */
async function treeEntries(repo: string, commit: string): Promise<any[]> {
  let url: string | null = `https://huggingface.co/api/models/${repo}/tree/${commit}?recursive=true&expand=true`;
  const all: any[] = [];
  while (url) {
    const res: Response = await apiGet(url);
    all.push(...((await res.json()) as any[]));
    const link = res.headers.get("link");
    const next = link ? /<([^>]+)>;\s*rel="next"/.exec(link) : null;
    url = next ? next[1] : null;
  }
  return all;
}

/**
 * Per-file size and blob name. The tree API's `lfs.oid` is the sha256 and is also the
 * blob name hf-hub uses (its etag for LFS files); a file stored in git rather than LFS
 * has no `lfs` field, and its etag is the git blob sha1 — usable as a blob name but
 * NOT as a sha256 to verify against.
 */
async function resolveFiles(repo: string, commit: string, paths: string[]): Promise<FileMeta[]> {
  const tree = await treeEntries(repo, commit);
  const byPath = new Map<string, any>(tree.filter((e) => e.type === "file").map((e) => [e.path, e]));
  const out: FileMeta[] = [];
  for (const path of paths) {
    const entry = byPath.get(path);
    if (!entry) throw new Error(`${repo}: no such file in tree at ${commit}: ${path}`);
    if (entry.lfs?.oid) {
      out.push({ path, size: entry.lfs.size ?? entry.size, oid: entry.lfs.oid, lfs: true });
      continue;
    }
    const url = `https://huggingface.co/${repo}/resolve/${commit}/${encodePath(path)}`;
    const res = await fetch(url, { method: "HEAD", headers: authHeaderMap() });
    if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText} for HEAD ${url}`);
    const raw = res.headers.get("x-linked-etag") ?? res.headers.get("etag");
    const etag = raw ? /^(?:W\/)?"?([0-9a-f]+)"?$/.exec(raw.trim()) : null;
    if (!etag) throw new Error(`${path}: no usable etag in HEAD response (got ${raw ?? "none"})`);
    out.push({ path, size: entry.size, oid: etag[1], lfs: false });
  }
  return out;
}

function sizeOf(p: string): number {
  try {
    return statSync(p).size;
  } catch {
    return -1;
  }
}

async function sha256(path: string): Promise<string> {
  const { code, stdout } = await run("shasum", ["-a", "256", path]);
  if (code !== 0) throw new Error(`shasum failed for ${path}`);
  return stdout.trim().split(/\s+/)[0];
}

/** Log download progress by watching the partial file grow. */
function progressWatcher(label: string, partPath: string, total: number, everyMs = 60_000) {
  let last = Math.max(0, sizeOf(partPath));
  let lastAt = Date.now();
  return setInterval(() => {
    const now = Math.max(0, sizeOf(partPath));
    const dt = (Date.now() - lastAt) / 1000;
    const rate = dt > 0 ? (now - last) / dt / 1e6 : 0;
    const pct = ((now / total) * 100).toFixed(1);
    const etaS = rate > 0 ? (total - now) / (rate * 1e6) : Infinity;
    const eta = Number.isFinite(etaS) ? `${(etaS / 60).toFixed(0)}m` : "?";
    log(`${label}: ${human(now)} / ${human(total)} (${pct}%) at ${rate.toFixed(0)} MB/s, eta ${eta}`);
    last = now;
    lastAt = Date.now();
  }, everyMs);
}

type Result = { path: string; oid: string; check: string; seconds: number };

/** A short download is resumed; only a size-exact-but-wrong-hash file is thrown away. */
const MAX_ATTEMPTS = 5;
const MAX_CORRUPT = 3;

async function fetchOne(
  repo: string,
  commit: string,
  meta: FileMeta,
  opts: { verifyExisting: boolean; dryRun: boolean },
): Promise<Result> {
  const root = cacheRoot();
  const repoDir = join(root, repoDirName(repo));
  const blob = join(repoDir, "blobs", meta.oid);
  const link = join(repoDir, "snapshots", commit, meta.path);
  const started = Date.now();

  // One "../" per path segment, plus one to climb out of the commit directory:
  // "a.gguf" -> ../../blobs/<oid>, "q4/a.gguf" -> ../../../blobs/<oid>.
  const rel = "../".repeat(meta.path.split("/").length + 1) + "blobs/" + meta.oid;

  const linkBlob = () => {
    mkdirSync(dirname(link), { recursive: true });
    // lstat, not exists: a dangling symlink from an earlier run must be replaced too.
    try {
      lstatSync(link);
      unlinkSync(link);
    } catch {}
    symlinkSync(rel, link);
  };

  if (opts.dryRun) {
    log(`${meta.path}: would fetch ${human(meta.size)} -> ${blob}`);
    return { path: meta.path, oid: meta.oid, check: "n/a (dry run)", seconds: 0 };
  }

  mkdirSync(join(repoDir, "blobs"), { recursive: true });

  if (sizeOf(blob) === meta.size) {
    let ok = true;
    if (opts.verifyExisting && meta.lfs) {
      log(`${meta.path}: blob present, re-hashing`);
      ok = (await sha256(blob)) === meta.oid;
      if (!ok) {
        log(`${meta.path}: EXISTING BLOB FAILED sha256, re-downloading`);
        rmSync(blob, { force: true });
      }
    }
    if (ok) {
      linkBlob();
      log(`${meta.path}: already cached (${human(meta.size)}), linked`);
      return {
        path: meta.path,
        oid: meta.oid,
        check: opts.verifyExisting && meta.lfs ? "sha256 VERIFIED (already cached)" : "not re-checked (already cached)",
        seconds: (Date.now() - started) / 1000,
      };
    }
  }

  const part = blob + ".part";
  const url = `https://huggingface.co/${repo}/resolve/${commit}/${encodePath(meta.path)}`;

  let corrupt = 0;
  for (let attempt = 1; ; attempt++) {
    if (attempt > MAX_ATTEMPTS) throw new Error(`${meta.path}: giving up after ${MAX_ATTEMPTS} attempts`);

    // A partial LONGER than the remote file can never be resumed into shape: `curl -C -`
    // asks for a range past the end, the server answers 416, and curl still exits 0 — so
    // the partial would survive untouched and cost a full hash on every future run.
    if (sizeOf(part) > meta.size) {
      log(`${meta.path}: partial is larger than the remote file (${human(sizeOf(part))}), discarding`);
      rmSync(part, { force: true });
    }

    if (sizeOf(part) !== meta.size) {
      log(`${meta.path}: downloading ${human(meta.size)} (attempt ${attempt}, have ${human(Math.max(0, sizeOf(part)))})`);
      const watcher = progressWatcher(meta.path, part, meta.size);
      const { code } = await run("curl", [
        "-L",
        "-C",
        "-",
        "--retry",
        "20",
        "--retry-all-errors",
        "--retry-delay",
        "5",
        "--no-progress-meter",
        ...authHeaders(),
        "-o",
        part,
        url,
      ]);
      clearInterval(watcher);
      const have = Math.max(0, sizeOf(part));
      if (have !== meta.size) {
        log(`${meta.path}: curl exit ${code}, have ${human(have)} of ${human(meta.size)}, resuming`);
        continue;
      }
    }

    if (!meta.lfs) {
      log(`${meta.path}: size OK; stored in git, not LFS — its etag is a sha1, so no sha256 check`);
      break;
    }
    log(`${meta.path}: verifying sha256`);
    const got = await sha256(part);
    if (got === meta.oid) break;
    // Right size, wrong hash: corrupt beyond repair by resuming, so start over.
    log(`${meta.path}: sha256 MISMATCH (got ${got}, want ${meta.oid}), discarding partial`);
    rmSync(part, { force: true });
    if (++corrupt >= MAX_CORRUPT) throw new Error(`${meta.path}: sha256 mismatch after ${corrupt} full downloads`);
  }

  // renameSync, not `mv`: a failure here must abort rather than leave linkBlob() pointing
  // a snapshot symlink at a blob that was never put in place.
  renameSync(part, blob);
  linkBlob();
  const seconds = (Date.now() - started) / 1000;
  const check = meta.lfs ? "sha256 VERIFIED" : "size only (not an LFS file)";
  log(`${meta.path}: done in ${(seconds / 60).toFixed(1)}m, ${check}`);
  return { path: meta.path, oid: meta.oid, check, seconds };
}

async function main() {
  const argv = process.argv.slice(2);
  const usage = "usage: bun scripts/hf-fetch.ts <repo> <path...> [--jobs N] [--verify] [--dry-run]";
  let jobs = 1;
  let verifyExisting = false;
  let dryRun = false;
  const positional: string[] = [];
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--jobs") {
      const raw = argv[++i];
      jobs = Number(raw);
      // A NaN here used to spawn zero workers: nothing downloaded, refs/main written
      // anyway, and a "all done" report over an empty snapshot.
      if (!Number.isInteger(jobs) || jobs < 1) {
        console.error(`--jobs needs a positive integer, got ${raw === undefined ? "nothing" : `"${raw}"`}`);
        process.exit(2);
      }
    } else if (a === "--verify") verifyExisting = true;
    else if (a === "--dry-run") dryRun = true;
    else if (a === "-h" || a === "--help") {
      console.log(usage);
      process.exit(0);
    } else if (a.startsWith("--")) {
      console.error(`unknown flag ${a}\n${usage}`);
      process.exit(2);
    } else positional.push(a);
  }
  const [repo, ...rest] = positional;
  // Two workers on one path would race on the same .part and corrupt it.
  const paths = [...new Set(rest)];
  if (!repo || paths.length === 0) {
    console.error(usage);
    process.exit(2);
  }
  if (paths.length !== rest.length) log(`ignoring ${rest.length - paths.length} duplicate path(s)`);

  const root = cacheRoot();
  const repoDir = join(root, repoDirName(repo));
  log(`cache root: ${root}`);

  const commit = await resolveCommit(repo);
  log(`${repo} @ ${commit}`);
  const metas = await resolveFiles(repo, commit, paths);
  const total = metas.reduce((a, m) => a + m.size, 0);
  log(`${metas.length} file(s), ${human(total)} total, ${jobs} job(s)`);

  const results: Result[] = [];
  let next = 0;
  const worker = async () => {
    while (true) {
      const i = next++;
      if (i >= metas.length) return;
      results.push(await fetchOne(repo, commit, metas[i], { verifyExisting, dryRun }));
    }
  };
  const startedAll = Date.now();
  // Every file landed before refs/main moves: a rejection here exits nonzero and leaves
  // the ref pointing at whatever complete snapshot it pointed at before.
  await Promise.all(Array.from({ length: Math.min(jobs, metas.length) }, worker));

  if (!dryRun) {
    // refs/main verbatim: no trailing newline, or hf-hub misses the cache entirely.
    mkdirSync(join(repoDir, "refs"), { recursive: true });
    await Bun.write(join(repoDir, "refs", "main"), commit);
    log(`refs/main -> ${commit} (${commit.length} bytes, no newline)`);
  }

  const wall = (Date.now() - startedAll) / 1000;
  log(`all done in ${(wall / 60).toFixed(1)}m`);
  for (const r of results.sort((a, b) => a.path.localeCompare(b.path))) log(`  ${r.path}  ${r.check}`);
  log(`snapshot dir: ${join(repoDir, "snapshots", commit)}`);
}

main().catch((e) => {
  console.error(`[${ts()}] FATAL: ${e?.stack ?? e}`);
  process.exit(1);
});
