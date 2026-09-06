#!/usr/bin/env bun
// Long-context envelope harness: prefill, decode, acceptance and peak memory as
// a function of prompt length, up to the 131072 the CLI advertises.
//
// The standing bench rules (docs/benching.md) apply here and are enforced:
//   - ONE model process at a time. The script waits for `pgrep` to come back
//     empty AND for /tmp/xwen-gpu.lock to disappear, then takes the lock for the
//     duration of each run. Other agents on this machine take the same lock.
//   - A PINNED binary: `--bin` is how you point at a detached-worktree build.
//     Defaults to target/release/xwen only so an ad-hoc run works.
//   - Power mode is read from `pmset -g` and stamped on every row, verbatim.
//   - Model output goes to a file, never through a pager.
//   - Memory is `footprint`'s `phys_footprint_peak`, sampled once a second.
//     Anonymous RSS lies under mmap because the weights are file-backed.
//
// Prompts are synthesized from committed repo text (the bench fixtures plus the
// prose under docs/) and cut to a token target with `llama-tokenize` against the
// checkpoint's own GGUF vocab, which is a vocab-only load and costs ~0.4 s. They
// are cached under /tmp/longctx/ and are byte-identical across sessions, so two
// sessions measure the same prompt.
//
// Warm-up. XWEN_BENCH=1 makes the binary run a full throwaway prefill before the
// measured one, which is right at 4k and doubles the wall at 128k. So
// `--warmup auto` (the default) sets it below 65536 and above that discards the
// first repetition instead: same protection against a cold pipeline cache and a
// cold page cache, at one extra run rather than one extra prefill per run.
//
// Usage:
//   bun scripts/longctx.ts --bin /tmp/xwen-longctx/target/release/xwen \
//     --model-size flash-next --tokens 8192,32768,65536,131072 --reps 2
//   bun scripts/longctx.ts --model-size 35b --tokens 4096,8192 --draft-ctx 32768
//   bun scripts/longctx.ts ... -- --moe-impl fused      # extra generate args
//
// Flags:
//   --bin PATH         binary to measure (default target/release/xwen)
//   --model-size SIZE  27b|35b|3.8-27b|flash-next (default: flash-next)
//   --model PATH       an explicit GGUF, overriding --model-size
//   --tokens CSV       prompt lengths (default 8192,32768,65536,131072)
//   --reps N           repetitions per cell (default 2); the row prints the median
//   --n N              decode tokens per run (default 192)
//   --raw              feed the prompt untemplated (default: chat template + a
//                      thinking floor, which is what keeps decode long enough to
//                      be a rate at every prompt length)
//   --min-think N      thinking floor in decode tokens (default 160; ignored with --raw)
//   --no-draft         pass --no-draft (plain decode)
//   --draft-ctx N      pass --draft-ctx N
//   --warmup MODE      auto|on|off (default auto)
//   --timeout-min N    kill a run after this many minutes (default 20)
//   --label TEXT       stamped on every row, for grouping arms in the report
//   --out PATH         append the JSON rows here too (default /tmp/longctx/rows.jsonl)
//   --out-dir DIR      raw model output (default /tmp/longctx/out)
//   --                 everything after this is appended to the generate command

import { existsSync, mkdirSync, readFileSync, appendFileSync } from "node:fs";
import { join, dirname, basename } from "node:path";
import { officialModel, type ModelSize } from "./hf";

const repo = dirname(import.meta.dir);
const work = "/tmp/longctx";
const LOCK = "/tmp/xwen-gpu.lock";
const LOCK_OWNER = "128k";

const argv = process.argv.slice(2);
const sepAt = argv.indexOf("--");
const args = sepAt >= 0 ? argv.slice(0, sepAt) : argv;
const passthrough = sepAt >= 0 ? argv.slice(sepAt + 1) : [];

function flag(name: string): boolean {
  return args.includes(`--${name}`);
}
function opt(name: string, dflt: string): string {
  const i = args.indexOf(`--${name}`);
  return i >= 0 && args[i + 1] !== undefined ? args[i + 1]! : dflt;
}

// `XWEN_LONGCTX_BIN` is the same setting off the command line, and is the form
// to prefer: the repo's pgrep guard matches any command line containing
// `target/release/xwen`, so a `--bin` pointing at a worktree build makes THIS
// harness look like a running model process to every other bench on the machine.
const bin = opt("bin", process.env.XWEN_LONGCTX_BIN ?? join(repo, "target/release/xwen"));
const size = opt("model-size", "flash-next") as ModelSize;
const modelPath = opt("model", "") || officialModel(size);
const tokenTargets = opt("tokens", "8192,32768,65536,131072")
  .split(",")
  .map((t) => Number(t.trim()))
  .filter((t) => t > 0);
const reps = Number(opt("reps", "2"));
const nDecode = Number(opt("n", "192"));
// Decode has to be a RATE, and a raw continuation of a cut-off document emits an
// end-of-generation token almost immediately: 32768 raw tokens decoded 28 before
// stopping, and 0.67 s is not a decode measurement. So the default runs through
// the chat template with a thinking floor, which forces at least `--min-think`
// decode tokens at every length and is a realistic long-document turn besides.
// `--raw` restores the untemplated prompt for anyone measuring prefill alone.
const raw = flag("raw");
const minThink = Number(opt("min-think", "160"));
const noDraft = flag("no-draft");
const draftCtx = opt("draft-ctx", "");
const warmupMode = opt("warmup", "auto");
const timeoutMs = Number(opt("timeout-min", "20")) * 60_000;
const label = opt("label", "");
const rowsFile = opt("out", join(work, "rows.jsonl"));
const outDir = opt("out-dir", join(work, "out"));

mkdirSync(work, { recursive: true });
mkdirSync(outDir, { recursive: true });

if (!existsSync(bin)) {
  console.error(`ABORT: no binary at ${bin} (build a detached worktree and pass --bin)`);
  process.exit(2);
}

// ---------------------------------------------------------------- power mode

/** The `pmset -g` line verbatim, because docs/benching.md requires it printed
 *  next to every figure and the key is spelled two ways on this OS. */
function powerModeLine(): string {
  const out = Bun.spawnSync(["pmset", "-g"]).stdout.toString();
  const line = out.split("\n").find((l) => /\b(?:low)?powermode\b/.test(l));
  if (!line) throw new Error("pmset -g reported neither lowpowermode nor powermode");
  return line.trim();
}
const POWER = powerModeLine();

// --------------------------------------------------------------- the GPU lock

function modelProcesses(): string[] {
  const proc = Bun.spawnSync([
    "pgrep",
    "-fl",
    "target/release/(xwen|logits-dump|deps/parity)|llama-(cli|server|bench|eval-callback)",
  ]);
  return proc.stdout
    .toString()
    .split("\n")
    .filter((l) => l.trim() && !l.includes("longctx.ts"));
}

async function acquireGpu(what: string) {
  let waited = 0;
  for (;;) {
    const busy = modelProcesses();
    const locked = existsSync(LOCK);
    if (busy.length === 0 && !locked) break;
    if (waited % 30 === 0) {
      const why = locked ? `lock held by ${readFileSync(LOCK, "utf8").trim()}` : busy.join("; ");
      console.error(`  waiting for the GPU (${why})`);
    }
    await Bun.sleep(1000);
    waited += 1;
    if (waited > 3600) throw new Error("gave up waiting an hour for the GPU");
  }
  Bun.write(LOCK, `${LOCK_OWNER}\n${what}\n`);
}

function releaseGpu() {
  try {
    if (existsSync(LOCK) && readFileSync(LOCK, "utf8").startsWith(LOCK_OWNER)) {
      Bun.spawnSync(["rm", "-f", LOCK]);
    }
  } catch {
    /* releasing a lock is best-effort; a stale one is visible in the log */
  }
}
process.on("exit", releaseGpu);
process.on("SIGINT", () => {
  releaseGpu();
  process.exit(130);
});

// ------------------------------------------------------------------- prompts

const tokenizeBin = join(repo, "reference/llama.cpp/build/bin/llama-tokenize");

/** Token count of `text` under the GGUF's own vocab. A vocab-only load, ~0.4 s,
 *  so a few of these per prompt is cheaper than one wrong prefill. */
function countTokens(file: string): number {
  const proc = Bun.spawnSync([
    tokenizeBin,
    "-m",
    modelPath,
    "-f",
    file,
    "--show-count",
    "--no-bos",
    "--no-parse-special",
  ]);
  const m = (proc.stdout.toString() + proc.stderr.toString()).match(
    /Total number of tokens:\s*(\d+)/,
  );
  if (!m) throw new Error(`llama-tokenize did not report a count for ${file}`);
  return Number(m[1]);
}

function corpus(): string {
  const path = join(work, "corpus.txt");
  if (!existsSync(path)) {
    const parts: string[] = [];
    for (const f of ["prefill-4k.txt", "prefill-925.txt", "decode-630.txt"]) {
      parts.push(readFileSync(join(repo, "tests/fixtures/bench-prompts", f), "utf8"));
    }
    const docs = Bun.spawnSync(["find", join(repo, "docs"), "-name", "*.md"])
      .stdout.toString()
      .split("\n")
      .filter(Boolean)
      .sort();
    for (const d of docs) parts.push(readFileSync(d, "utf8"));
    Bun.write(path, parts.join("\n\n"));
  }
  return readFileSync(path, "utf8");
}

/** The tail every synthesized prompt ends on. Without it the prompt stops in the
 *  middle of whatever document the cut landed in, and the model's most likely
 *  next token there is an end-of-generation one: a 32768-token run decoded zero
 *  tokens and reported no decode rate at all. In raw mode the cue also ends
 *  mid-sentence, which is the only lever there for keeping decode alive. */
const CUE =
  "\n\n---\n\nSummarize, in detail, the operational rules and the measured " +
  "figures described above.";
const RAW_TAIL = "\n\nAnswer: The";

/** A prompt of `target` tokens (within 0.5%), cached. Cut from the corpus by
 *  secant iteration on the chars-per-token ratio: the corpus is real prose, so
 *  the ratio is stable and this converges in two or three probes. */
function promptFor(target: number): { file: string; tokens: number } {
  const key = `${basename(modelPath).replace(/\.gguf$/, "")}-${target}${raw ? "-raw" : ""}`;
  const file = join(work, `prompt-${key}.txt`);
  const meta = join(work, `prompt-${key}.count`);
  if (existsSync(file) && existsSync(meta)) {
    return { file, tokens: Number(readFileSync(meta, "utf8").trim()) };
  }
  const text = corpus();
  let ratio = 3.4; // chars per token, refined from the first probe
  let chars = Math.round(target * ratio);
  let tokens = 0;
  for (let i = 0; i < 10; i += 1) {
    chars = Math.min(chars, text.length);
    let cut = text.slice(0, chars);
    // Never end on half a surrogate pair; the file has to be valid UTF-8.
    if (cut.length && /[\uD800-\uDBFF]/.test(cut[cut.length - 1]!)) cut = cut.slice(0, -1);
    cut += raw ? CUE + RAW_TAIL : CUE;
    Bun.write(file, cut);
    tokens = countTokens(file);
    if (Math.abs(tokens - target) / target <= 0.005) break;
    if (chars >= text.length && tokens < target) {
      throw new Error(
        `the corpus is only ${tokens} tokens, short of ${target}; add fixtures or prose`,
      );
    }
    ratio = cut.length / tokens;
    chars = Math.round(target * ratio);
  }
  Bun.write(meta, String(tokens));
  return { file, tokens };
}

// ------------------------------------------------------------------ the runs

interface Row {
  label: string;
  checkpoint: string;
  arm: string;
  target_tokens: number;
  prompt_tokens: number;
  prefill_tokens: number | null;
  prefill_secs: number | null;
  prefill_tps: number | null;
  decode_tokens: number | null;
  decode_secs: number | null;
  decode_tps: number | null;
  wall_secs: number;
  peak_footprint_gb: number | null;
  spec: string | null;
  drafting: string | null;
  status: string;
  power: string;
  rep: number;
}

/** Peak physical footprint in bytes, from `footprint`'s auxiliary block. The
 *  detailed table needs privileges this process does not have; phys_footprint_peak
 *  does not, and being a peak it survives a one-second sampling interval. */
function footprintPeak(pid: number): number | null {
  const proc = Bun.spawnSync(["footprint", String(pid)]);
  const m = proc.stdout.toString().match(/phys_footprint_peak:\s*([\d.]+)\s*(\w+)/);
  if (!m) return null;
  const scale: Record<string, number> = {
    B: 1,
    KB: 1e3,
    MB: 1e6,
    GB: 1e9,
    TB: 1e12,
  };
  return Number(m[1]) * (scale[m[2]!] ?? 1);
}

function parseStats(text: string) {
  const pre = text.match(/^prefill:\s*(\d+) tokens in ([\d.]+)s \(([\d.]+) tok\/s\)/m);
  const dec = text.match(/^decode:\s*(\d+) tokens in ([\d.]+)s \(([\d.]+) tok\/s\)/m);
  const spec = text.match(/^\s*spec:.*$/m);
  const drafting = text.match(/^\s*drafting:.*$/m);
  return {
    prefill_tokens: pre ? Number(pre[1]) : null,
    prefill_secs: pre ? Number(pre[2]) : null,
    prefill_tps: pre ? Number(pre[3]) : null,
    decode_tokens: dec ? Number(dec[1]) : null,
    decode_secs: dec ? Number(dec[2]) : null,
    decode_tps: dec ? Number(dec[3]) : null,
    spec: spec ? spec[0].trim() : null,
    drafting: drafting ? drafting[0].trim() : null,
  };
}

const arm = [noDraft ? "plain" : "drafted", draftCtx ? `draft-ctx=${draftCtx}` : ""]
  .filter(Boolean)
  .join(" ");

async function runOne(target: number, prompt: string, rep: number, warmup: boolean): Promise<Row> {
  const cmd = [
    bin,
    "generate",
    "--model",
    modelPath,
    "--stats",
    ...(raw ? ["--raw"] : ["--min-think", String(minThink)]),
    "-n",
    String(nDecode),
    // Headroom over the prompt for the chat template's own tokens and the
    // decode run, so `--max-ctx` never becomes the thing being measured.
    "--max-ctx",
    String(target + 1024),
    // `--draft official` is passed rather than relying on the zero-flag
    // default, because that default is per checkpoint: it is off on the
    // 35B-A3B since 2026-09-06, and a bare run there would be a plain arm
    // labelled "drafted".
    ...(noDraft ? ["--no-draft"] : ["--draft", "official"]),
    ...(draftCtx ? ["--draft-ctx", draftCtx] : []),
    ...passthrough,
    "--prompt",
    readFileSync(prompt, "utf8"),
  ];

  await acquireGpu(`longctx ${size} ${target} rep${rep}`);
  const outFile = join(outDir, `${size}-${target}-${arm.replace(/\W+/g, "_")}-r${rep}.txt`);
  const started = Date.now();
  const env = { ...process.env } as Record<string, string>;
  // Marks every record this sweep writes as harness-driven, so a few dozen
  // synthetic long-context runs stay in the metrics history but out of the
  // default `xwen stats` table, which answers what real use cost.
  env.XWEN_METRICS_TAG = "bench";
  if (warmup) env.XWEN_BENCH = "1";
  else delete env.XWEN_BENCH;

  const proc = Bun.spawn(cmd, {
    cwd: repo,
    env,
    stdout: "pipe",
    stderr: "pipe",
  });

  let peak: number | null = null;
  let timedOut = false;
  const sampler = setInterval(() => {
    const f = footprintPeak(proc.pid);
    if (f !== null && (peak === null || f > peak)) peak = f;
  }, 1000);
  const killer = setTimeout(() => {
    timedOut = true;
    proc.kill("SIGKILL");
  }, timeoutMs);

  const [out, err] = await Promise.all([
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
  ]);
  const code = await proc.exited;
  clearInterval(sampler);
  clearTimeout(killer);
  releaseGpu();

  const wall = (Date.now() - started) / 1000;
  Bun.write(outFile, out + "\n===== stderr =====\n" + err);
  const stats = parseStats(out + "\n" + err);
  const status = timedOut ? `timeout after ${timeoutMs / 60000} min` : code === 0 ? "ok" : `exit ${code}`;

  return {
    label,
    checkpoint: size,
    arm,
    target_tokens: target,
    prompt_tokens: 0, // filled by the caller, which knows the tokenizer count
    ...stats,
    wall_secs: Number(wall.toFixed(2)),
    peak_footprint_gb: peak === null ? null : Number((peak / 1e9).toFixed(2)),
    status,
    power: POWER,
    rep,
  } as Row;
}

function median(xs: number[]): number | null {
  const v = xs.filter((x) => Number.isFinite(x)).sort((a, b) => a - b);
  if (v.length === 0) return null;
  const mid = v.length >> 1;
  return v.length % 2 ? v[mid]! : (v[mid - 1]! + v[mid]!) / 2;
}

// ---------------------------------------------------------------------- main

console.log(`# longctx: ${size} (${basename(modelPath)}), arm "${arm || "drafted"}"`);
console.log(`# bin: ${bin}`);
console.log(`# power: ${POWER}`);

// Prompts first, before anything touches the GPU: the tokenizer probes are a
// vocab-only load and have no business inside a measured stretch.
const cells = tokenTargets.map((target) => {
  const { file, tokens } = promptFor(target);
  const warmup = warmupMode === "on" ? true : warmupMode === "off" ? false : target < 65536;
  // With no in-process warm-up the first pass pays pipeline compilation and a
  // cold page cache, so it is run and thrown away rather than averaged in.
  return { target, file, tokens, warmup, passes: warmup ? reps : reps + 1, rows: [] as Row[], done: false };
});
for (const c of cells) {
  console.log(
    `# ${c.target} -> ${c.tokens} tokens by the model's vocab, ${c.passes} run(s), ` +
      `warm-up ${c.warmup ? "in-process" : "by discarded first run"}`,
  );
}

// `--prompts-only` stops here: synthesizing and caching the prompts is pure
// tokenizer work, and doing it ahead of time keeps it out of a measured window
// and off a GPU another agent may be holding.
if (flag("prompts-only")) {
  console.log("\n# prompts cached; nothing measured (--prompts-only)");
  process.exit(0);
}

// INTERLEAVED, A B A B: all cells at pass 1, then all cells at pass 2. Running
// every repetition of one length before starting the next makes a thermal or a
// contention drift look like a property of the length (docs/benching.md).
const maxPass = Math.max(...cells.map((c) => c.passes));
let first = true;
for (let pass = 1; pass <= maxPass; pass += 1) {
  for (const cell of cells) {
    if (pass > cell.passes || cell.done) continue;
    // Duty cycle shows up directly in these numbers; idle between rounds.
    if (!first) await Bun.sleep(60_000);
    first = false;
    const row = await runOne(cell.target, cell.file, pass, cell.warmup);
    row.prompt_tokens = cell.tokens;
    const kept = cell.warmup || pass > 1;
    if (kept) cell.rows.push(row);
    appendFileSync(rowsFile, JSON.stringify({ ...row, kept, pass }) + "\n");
    console.log(JSON.stringify({ ...row, kept, pass }));
    if (row.status !== "ok") {
      console.error(`  ${cell.target}: did not finish cleanly (${row.status}); dropping this cell`);
      cell.done = true;
    }
  }
}

console.log("\n# summaries");
for (const cell of cells) {
  const rows = cell.rows;
  const summary = {
    summary: true,
    label,
    checkpoint: size,
    arm,
    target_tokens: cell.target,
    prompt_tokens: cell.tokens,
    n: rows.length,
    prefill_tokens: rows[0]?.prefill_tokens ?? null,
    prefill_tps: median(rows.map((r) => r.prefill_tps!)),
    prefill_secs: median(rows.map((r) => r.prefill_secs!)),
    decode_tps: median(rows.map((r) => r.decode_tps!)),
    decode_tokens: median(rows.map((r) => r.decode_tokens!)),
    peak_footprint_gb: median(rows.map((r) => r.peak_footprint_gb!)),
    wall_secs: median(rows.map((r) => r.wall_secs)),
    spec: rows[rows.length - 1]?.spec ?? null,
    drafting: rows[rows.length - 1]?.drafting ?? null,
    power: POWER,
  };
  appendFileSync(rowsFile, JSON.stringify(summary) + "\n");
  console.log(JSON.stringify(summary));
}

console.log(`\n# rows appended to ${rowsFile}; raw output under ${outDir}`);
