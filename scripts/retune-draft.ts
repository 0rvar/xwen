#!/usr/bin/env bun
/**
 * Retune the speculative-decode controller constants: `draft_p_min` and
 * `pause_margin`.
 *
 * Why this exists as a script rather than a one-off driver: both constants were
 * fitted (2026-07-29, P9) against a verify cost curve that no longer exists —
 * the K-snapshot fused verify replaced the per-token reference scan the fit
 * assumed, and the curve has moved again since. Every future change to the
 * verify round invalidates the fit, so the retune has to be one command rather
 * than a scratchpad driver that has to be reconstructed from a log entry.
 *
 * What it measures. Two stages per checkpoint, each a set of interleaved arms:
 *
 *   stage 1  p_min over the grid at the current default pause_margin, against
 *            a `--no-draft` plain arm. Picks the p_min — and, when a depth grid
 *            is given, the chain depth with it: the arms are then the CROSS
 *            PRODUCT of the two, because the pair interacts. A confidence floor
 *            is a rule about when to stop drafting and a depth is a cap on the
 *            same thing, so the best floor at depth 2 need not be the best at
 *            depth 4, and sweeping them separately would fit each against the
 *            other's shipped value rather than against its own best partner.
 *   stage 2  pause_margin over the grid at stage 1's winning p_min, again
 *            against plain, and always including both `margin 0` (never-pause)
 *            and the shipped margin. The never-pause arm is the diagnostic that
 *            decides whether the auto-pause controller is earning its keep at
 *            all — it is the comparison decisions.md records the controller
 *            winning, so a retune that stopped measuring it could keep a
 *            controller that had since become dead weight. The shipped margin is
 *            always present so "keep the current value" is a graded option
 *            rather than an untested assumption.
 *
 * Stage 2 re-runs EVERY arm it scores, including plain and the shipped margin,
 * even though stage 1 already measured that exact configuration. Reusing those
 * cells would be cheaper by a third and it is wrong: this machine drifts 10-15%
 * over the timescale separating the two stages, so a stage-2 arm graded against
 * a stage-1 plain median is graded against a different machine. Stage 1's only
 * output is the p_min carried into stage 2; none of its measurements are ever
 * stage-2 data.
 *
 * The status quo it measures against is PER-CHECKPOINT for p_min (27B 0.5, 35B
 * 0.3, 3.8-27B 0.7) and for depth (15 on the two DFlash block drafters, 4 on
 * the 3.8's MTP head), and shared for pause_margin (1.0). `SHIPPED_P_MIN` and
 * `SHIPPED_DRAFT_MAX` below mirror `Model::draft_p_min_default()` and
 * `Model::draft_max_default()` in src/hub.rs and must be updated with them.
 *
 * The criterion is the one the shipped default was chosen by, not a bare mean:
 * an arm qualifies only if it is ahead of plain decode on BOTH prompt kinds in
 * EVERY individual rep (each rep's tok/s over that prompt's plain median). A
 * setting that wins on average but loses to plain on chat in one run out of
 * three is not a default. The winner among qualifiers is the highest mean of
 * the two per-prompt medians. When nothing qualifies, that is the finding — the
 * script says so and labels the best-by-mean arm non-robust rather than
 * dressing it up as a winner.
 *
 * Protocol (the rules this machine has already enforced the hard way — CLAUDE.md):
 *   - ONE model process at a time. Every run waits on a pgrep guard first.
 *   - Arms are INTERLEAVED (rep 1 of every arm, then rep 2, ...). Thermal drift
 *     and machine contention move all arms together that way instead of
 *     penalising whichever arm ran last.
 *   - Greedy (`--temp 0`), 128 tokens, XWEN_BENCH=1 (warm-up forward, so the
 *     numbers are steady-state rather than first-forward). A run that decodes
 *     fewer than 128 tokens is FAILED, not recorded: a short run reports
 *     boost-clock fiction and is not comparable to a full one.
 *   - The child env is scrubbed of every XWEN_* var (a stray kernel toggle
 *     would apply to all arms and silently retune against a path nobody asked
 *     for) — only XWEN_BENCH, PATH/HOME/TMPDIR/LANG and one resolved
 *     HF_HUB_CACHE survive. No token is passed: the checkpoints are verified
 *     present in the cache before the first run, so nothing downloads.
 *   - `pmset -g` is captured at start AND end and printed verbatim. NOTE: this
 *     machine emits no `powermode` key, so a run is never positive evidence of
 *     high-power mode; the value is recorded, never interpreted.
 *
 * It NEVER edits source. The recommendation block prints the three places a
 * changed default has to land, and stops there.
 *
 * Usage:
 *   bun scripts/retune-draft.ts                          # every drafting model, both stages, 3 reps
 *   bun scripts/retune-draft.ts --dry-run                # print the run matrix, run nothing
 *   bun scripts/retune-draft.ts --model-size 27b --stage 1
 *   bun scripts/retune-draft.ts --reps 5 --p-min-grid 0.2,0.3,0.4
 *   bun scripts/retune-draft.ts --margin-grid 0.8,1.0,1.2 --timeout 600
 *   bun scripts/retune-draft.ts --model-size 3.8-27b --p-min-grid 0.3,0.5,0.7 --depth-grid 2,3,4
 *
 * Wall time: one run is ~128 greedy tokens plus a model load. Without a depth
 * grid one model's two stages are 5 arms x 2 prompts x 3 reps x 2 stages = 60
 * runs; a 3x3 p_min-by-depth stage 1 is 60 runs on its own. Budget in tens of
 * minutes per checkpoint and keep the machine otherwise idle.
 */

import { closeSync, existsSync, fsyncSync, mkdirSync, openSync, realpathSync, renameSync, writeFileSync } from "node:fs";
import { basename, dirname, join, resolve } from "node:path";
import { DRAFT_PROMPTS } from "./lib/draft-prompts";
import { CHECKPOINTS, draftingSizes, officialDrafter, officialModel, type ModelSize } from "./hf";

const ROOT = resolve(import.meta.dir, "..");

/** Tokens decoded per run. 128 greedy tokens is the P9a fixture length; the
 *  numbers in decisions.md are directly comparable only at this length. */
const N_TOKENS = 128;
/**
 * The shipped defaults this sweep measures against.
 *
 * `SHIPPED_P_MIN` MUST mirror `Model::draft_p_min_default()` in src/hub.rs — the
 * p_min default is PER-CHECKPOINT (the 27B pays far more per target forward and
 * wants shorter, more confident drafts), so there is no single shipped value.
 * A retune that installs a new default has to update BOTH: the Rust function is
 * what ships, this table is what the next sweep grades against, and a stale
 * table silently measures the wrong status quo — it would report "keep current"
 * for a value that is no longer current, and break the tie-toward-status-quo
 * rule and the cross-checkpoint conflict check with it.
 *
 * `SHIPPED_MARGIN` stays a single shared constant because pause_margin is one:
 * src/serve/config.rs DEFAULT_DRAFT_PAUSE_MARGIN and the `--draft-pause-margin`
 * clap default are both 1.0 for both checkpoints.
 *
 * Used four ways: stage 1 holds the margin fixed here while it sweeps p_min;
 * stage 2 always grades this margin so "keep the current value" is a measured
 * option; a tie resolves toward these; and a checkpoint with no winner is
 * reported as keeping them, which is what the conflict check compares.
 *
 * Only the checkpoints that ship a sidecar have an arm here, because only they
 * have a drafted arm to grade: `both` means those (`draftingSizes`), and naming
 * a sidecar-less checkpoint outright dies at the drafter check before any run.
 * Every current checkpoint ships one; the table stays `Partial` because a
 * future release need not.
 */
const SHIPPED_P_MIN: Partial<Record<ModelSize, number>> = {
  "27b": 0.5,
  "35b": 0.3,
  "3.8-27b": 0.7,
};

/**
 * The shipped chain depth, mirroring `Model::draft_max_default()` in src/hub.rs
 * under the same must-be-updated-together rule as `SHIPPED_P_MIN`.
 *
 * It is per-checkpoint because it is per-DRAFTER-KIND, and the split is large.
 * A DFlash block drafter emits its whole block in one forward, so depth is very
 * nearly free and 15 is a cap rather than a fitted value. An MTP head pays one
 * forward per step and compounds its own guesses, so depth costs linearly and
 * each further step is likelier to be wrong: 4 was fitted here on 2026-08-15
 * (llama.cpp's own default for this head is 3), and it is a real trade-off
 * rather than a ceiling — depths 5, 6 and 8 all measured worse.
 *
 * Which is why only a sweep that NAMES a depth grid varies it. The default is
 * each checkpoint's shipped depth and nothing else, so the ordinary p_min
 * retune keeps measuring exactly what it always measured.
 */
const SHIPPED_DRAFT_MAX: Partial<Record<ModelSize, number>> = {
  "27b": 15,
  "35b": 15,
  "3.8-27b": 4,
};

/** The shipped floor for a size this sweep is allowed to grade. Every caller is
 *  already past the drafter guard in `main`, which refuses a sidecar-less
 *  checkpoint before any arm is planned; this turns "then the table has it" from
 *  an assumption into a checked one, rather than printing `undefined` into a
 *  command line. */
function shippedPMin(size: ModelSize): number {
  const value = SHIPPED_P_MIN[size];
  if (value === undefined) {
    die(`no shipped draft_p_min for ${size} — it ships no drafter, so there is nothing to grade.`);
  }
  return value;
}

/** The shipped chain depth for a size this sweep is allowed to grade, checked
 *  the same way and for the same reason as `shippedPMin`. */
function shippedDraftMax(size: ModelSize): number {
  const value = SHIPPED_DRAFT_MAX[size];
  if (value === undefined) {
    die(`no shipped draft_max for ${size} — it ships no drafter, so there is nothing to grade.`);
  }
  return value;
}
const SHIPPED_MARGIN = 1.0;
/** The never-pause diagnostic arm, always present in stage 2. */
const NEVER_PAUSE_MARGIN = 0;

const DEFAULT_P_MIN_GRID = [0.2, 0.3, 0.5, 0.7];
const DEFAULT_MARGIN_GRID = [0.8, 1.0, 1.2];

const PROMPT_KINDS = Object.keys(DRAFT_PROMPTS); // ["code", "chat"]

/**
 * Processes that must not be running when a model run starts. These are `pgrep
 * -x` patterns: an extended regex matched against the whole executable NAME, so
 * a shell that merely quotes one of these in its argv does not match (the
 * false-abort that bit parity-gate.ts). The selected `--binary`'s own basename
 * is appended at runtime — a `--binary /tmp/xwen-candidate` runs as
 * `xwen-candidate` and would otherwise be invisible to this guard.
 *
 * Accepted limitation: there is a check-to-spawn race here. Two harnesses can
 * both pass the guard before either spawns. This is a single-operator machine
 * and a real lock is out of scope; the guard exists to catch the common case of
 * a sweep started while another agent's model run is already up.
 */
const MODEL_PROCS = [
  "xwen",
  "logits-dump",
  "spec-verify-bench",
  "llama-(cli|server|bench|eval-callback)",
  // cargo's parity test binaries, named parity-<hash>.
  "parity-[0-9a-f]+",
];
const CONTENTION_WAIT_MS = 5 * 60 * 1000;

// -------------------------------------------------------------------- args

interface Opts {
  sizes: ModelSize[];
  reps: number;
  stages: (1 | 2)[];
  pMinGrid: number[];
  marginGrid: number[];
  /** Chain depths to cross with the p_min grid, or null for "each checkpoint's
   *  shipped depth only". Null rather than a default array because the default
   *  is PER-SIZE, so it cannot be resolved until a size is in hand. */
  depthGrid: number[] | null;
  dryRun: boolean;
  binary: string;
  timeoutMs: number;
}

function die(msg: string): never {
  console.error(`retune-draft: ${msg}`);
  process.exit(2);
}

function parseArgs(argv: string[]): Opts {
  const flags: Record<string, string | boolean> = {};
  for (let i = 0; i < argv.length; i++) {
    const t = argv[i];
    if (!t.startsWith("--")) die(`unexpected argument ${JSON.stringify(t)}`);
    const key = t.slice(2);
    const next = argv[i + 1];
    if (next === undefined || next.startsWith("--")) flags[key] = true;
    else {
      flags[key] = next;
      i++;
    }
  }

  const known = new Set([
    "model-size", "reps", "stage", "p-min-grid", "margin-grid", "depth-grid", "dry-run", "binary",
    "timeout",
  ]);
  for (const k of Object.keys(flags)) {
    if (!known.has(k)) die(`unknown flag --${k} (valid: ${[...known].map((f) => `--${f}`).join(", ")})`);
  }

  // `max` is the difference that matters between the two grids: p_min is a
  // PROBABILITY, and the server rejects anything outside [0,1] outright
  // (src/serve/config.rs), so a sweep that recommended 1.2 would recommend a
  // value that cannot be installed as the shared default. pause_margin is a
  // cost ratio with no upper bound.
  const grid = (v: unknown, dflt: number[], label: string, max?: number): number[] => {
    if (v === undefined) return dflt;
    const items = String(v).split(",").map((s) => s.trim()).filter(Boolean).map((s) => {
      const n = Number(s);
      if (!Number.isFinite(n) || n < 0 || (max !== undefined && n > max)) {
        die(`${label}: ${JSON.stringify(s)} is not a finite number in [0, ${max ?? "inf"}]`);
      }
      return n;
    });
    if (items.length === 0) die(`${label}: empty grid`);
    return items;
  };

  // Depth is a token COUNT, not a rate: a fractional or zero depth is not a
  // slower setting, it is a command line the binary would reject or silently
  // round. Kept separate from `grid` for that reason alone.
  const depthGrid = ((v: unknown): number[] | null => {
    if (v === undefined) return null;
    const items = String(v).split(",").map((s) => s.trim()).filter(Boolean).map((s) => {
      const n = Number(s);
      if (!Number.isInteger(n) || n < 1) die(`--depth-grid: ${JSON.stringify(s)} is not an integer >= 1`);
      return n;
    });
    if (items.length === 0) die("--depth-grid: empty grid");
    return items;
  })(flags["depth-grid"]);

  const sizeArg = flags["model-size"] === undefined ? "both" : String(flags["model-size"]).toLowerCase();
  // `both` means the checkpoints that can speculate at all — a release with no
  // DFlash sidecar has no floor to fit and no drafted arm to fit it against.
  const sizes: ModelSize[] =
    sizeArg === "both"
      ? draftingSizes()
      : sizeArg in CHECKPOINTS
        ? [sizeArg as ModelSize]
        : die(`--model-size must be ${Object.keys(CHECKPOINTS).join("|")}|both, got ${JSON.stringify(sizeArg)}`);
  // 27b first when both are requested: it is the slower checkpoint, so a sweep
  // that has to be interrupted has produced the expensive half already.
  sizes.sort((a, b) => (a === "27b" ? -1 : b === "27b" ? 1 : 0));

  const stageArg = flags.stage === undefined ? "both" : String(flags.stage);
  const stages: (1 | 2)[] =
    stageArg === "both" ? [1, 2] : stageArg === "1" ? [1] : stageArg === "2" ? [2] : die(`--stage must be 1|2|both, got ${JSON.stringify(stageArg)}`);

  const reps = flags.reps === undefined ? 3 : Number(flags.reps);
  if (!Number.isInteger(reps) || reps < 1) die(`--reps must be a positive integer, got ${JSON.stringify(String(flags.reps))}`);

  const timeoutS = flags.timeout === undefined ? 300 : Number(flags.timeout);
  if (!Number.isFinite(timeoutS) || timeoutS <= 0) die(`--timeout must be a positive number of seconds`);

  return {
    sizes,
    reps,
    stages,
    pMinGrid: grid(flags["p-min-grid"], DEFAULT_P_MIN_GRID, "--p-min-grid", 1),
    marginGrid: grid(flags["margin-grid"], DEFAULT_MARGIN_GRID, "--margin-grid"),
    depthGrid,
    dryRun: Boolean(flags["dry-run"]),
    binary: typeof flags.binary === "string" ? resolve(String(flags.binary)) : join(ROOT, "target/release/xwen"),
    timeoutMs: Math.round(timeoutS * 1000),
  };
}

// -------------------------------------------------------------------- arms

interface Arm {
  /** Table label, unique within a stage. */
  label: string;
  plain: boolean;
  pMin: number | null;
  margin: number | null;
  /** Chain depth (`--draft-max`). Null only on the plain arm and in the dry
   *  run's stage 2, where stage 1's winner is not known yet. */
  draftMax: number | null;
  /** Set on the stage-2 never-pause diagnostic, so the table can mark it. */
  diagnostic?: boolean;
}

/**
 * DISPLAY formatting only — never identity. Two decimals so the default grids
 * line up in the tables, but a value that two decimals would round away keeps
 * its exact form: 0.2 prints "0.20", 0.301 prints "0.301". Rounding a value
 * into a label that another value also produces would silently merge two arms.
 */
function fmtVal(v: number): string {
  const rounded = v.toFixed(2);
  return Number(rounded) === v ? rounded : String(v);
}
/** IDENTITY — exact numeric serialization, never rounded. `String(x)` on a
 *  double is the shortest representation that round-trips, so two distinct
 *  values can never produce the same key. */
const exact = (v: number | null) => (v === null ? "none" : String(v));

const PLAIN: Arm = { label: "plain", plain: true, pMin: null, margin: null, draftMax: null };

/** Exact-duplicate grid values collapse to one arm. Set uses SameValueZero, so
 *  this dedupes on the exact double — 0.301 and 0.304 stay two arms, as they
 *  must, while a grid that names 0.3 twice yields one. */
const dedupe = (xs: number[]) => [...new Set(xs)];

/** Distinct configurations must have distinct labels: the scoring and table
 *  code looks arms up BY LABEL, so a collision would mix two settings' runs. */
function assertUniqueLabels(arms: Arm[]): Arm[] {
  const seen = new Set<string>();
  for (const a of arms) {
    if (seen.has(a.label)) die(`internal: duplicate arm label ${JSON.stringify(a.label)}`);
    seen.add(a.label);
  }
  return arms;
}

/**
 * Stage 1's arms: the p_min grid crossed with the depth grid.
 *
 * `depthGrid` is this SIZE's already-resolved list — a single-element one when
 * no `--depth-grid` was given, which is the case that keeps the labels (and so
 * the tables, and so the comparability with previous sweeps) exactly as they
 * were. The depth only enters the label when there is more than one of them,
 * because a label that never varies is column noise; the CELL KEY carries it
 * unconditionally, since that is identity rather than display.
 */
function stage1Arms(pMinGrid: number[], depthGrid: number[]): Arm[] {
  const depths = dedupe(depthGrid);
  const showDepth = depths.length > 1;
  const arms: Arm[] = [PLAIN];
  for (const p of dedupe(pMinGrid)) {
    for (const d of depths) {
      arms.push({
        label: showDepth ? `p=${fmtVal(p)} d=${d}` : `p=${fmtVal(p)}`,
        plain: false,
        pMin: p,
        margin: SHIPPED_MARGIN,
        draftMax: d,
      });
    }
  }
  return assertUniqueLabels(arms);
}

function stage2Arms(pMin: number | null, draftMax: number | null, marginGrid: number[]): Arm[] {
  // Two margins are always graded whatever the grid says: 0, the never-pause
  // control that decides whether auto-pause earns its keep, and the shipped
  // margin, so "keep the current value" is a measured option rather than an
  // untested assumption.
  const margins = dedupe([NEVER_PAUSE_MARGIN, SHIPPED_MARGIN, ...marginGrid]).sort((a, b) => a - b);
  return assertUniqueLabels([
    PLAIN,
    ...margins.map((m) => ({
      label: `m=${fmtVal(m)}`,
      plain: false,
      pMin,
      margin: m,
      draftMax,
      diagnostic: m === NEVER_PAUSE_MARGIN,
    })),
  ]);
}

/** Identity of one measured cell, for the raw records. Stage-scoped on purpose:
 *  stage 1 and stage 2 measure the shipped-margin configuration separately and
 *  those measurements must never be conflated (they are thermally unrelated). */
function cellKey(stage: 1 | 2, size: ModelSize, prompt: string, arm: Arm): string {
  const cfg = arm.plain ? "plain" : `p${exact(arm.pMin)}_m${exact(arm.margin)}_d${exact(arm.draftMax)}`;
  return `s${stage}|${size}|${prompt}|${cfg}`;
}

function argsFor(size: ModelSize, prompt: string, arm: Arm): string[] {
  return [
    "generate",
    "--model-size", size,
    "--prompt", DRAFT_PROMPTS[prompt],
    "-n", String(N_TOKENS),
    "--temp", "0",
    "--stats",
    ...(arm.plain
      // Speculation is opt-OUT, so the plain arm is the one that needs a flag.
      ? ["--no-draft"]
      // `--draft official` is passed explicitly rather than relying on the
      // default, so the drafted arms stay pinned if the default ever flips.
      // `--draft-max` likewise: its resolved default is per-checkpoint, so an
      // arm that omitted it would be measuring whatever hub.rs currently says
      // rather than the depth this cell is named for.
      : [
          "--draft", "official",
          "--draft-p-min", String(arm.pMin),
          "--draft-pause-margin", String(arm.margin),
          "--draft-max", String(arm.draftMax),
        ]),
  ];
}

/** The same argv with the (long) prompt elided, for printing. A null p_min only
 *  happens in --dry-run, where stage 1's winner is not known yet. */
function displayArgs(size: ModelSize, prompt: string, arm: Arm): string {
  return argsFor(size, prompt, arm)
    .map((x) => (x === DRAFT_PROMPTS[prompt] ? `<${prompt.toUpperCase()}_PROMPT>` : x === "null" ? "<stage1-winner>" : x))
    .join(" ");
}

// ---------------------------------------------------------------- environment

/** The hub cache root every child is pinned to, resolved once in preflight. */
let HF_CACHE_ROOT = "";

/**
 * Resolve the Hugging Face cache root ONCE, to an absolute canonical path.
 *
 * Two mismatches this closes, both of which turn a cache hit into a multi-GB
 * download mid-sweep. First, a RELATIVE `HF_HUB_CACHE`/`HF_HOME` resolves
 * against this script's cwd during preflight but against the repo root in the
 * child (the children run with `cwd: ROOT`), so preflight can verify a file the
 * child then cannot find. Second, `scripts/hf.ts` treats `HF_HUB_CACHE=""` as
 * absent while the Rust resolver treats the variable as present with an empty
 * path — the two would pick different roots. An empty override is a broken
 * config either way, so it is a hard error rather than a guess.
 *
 * The resolved root is written back into THIS process's env before any lookup,
 * so scripts/hf.ts resolves against exactly the root the children are given.
 */
function resolveHfCacheRoot(): string {
  const named = (k: string): string | null => {
    const v = process.env[k];
    if (v === undefined) return null;
    if (v.trim() === "") {
      die(
        `${k} is set but empty. That resolves to different cache roots here and in the child ` +
          `(the Rust side reads it as an empty path, this script as unset), which would restart a ` +
          `multi-GB download. Unset it, or point it at a real directory.`,
      );
    }
    return v;
  };
  const hubCache = named("HF_HUB_CACHE");
  const hfHome = named("HF_HOME");
  const raw = hubCache ?? (hfHome ? join(hfHome, "hub") : join(process.env.HOME ?? "", ".cache/huggingface/hub"));
  const abs = resolve(raw);
  // Canonicalize through symlinks when the root exists; a missing root is not
  // fatal here (the per-checkpoint preflight reports it far more usefully).
  try {
    return realpathSync(abs);
  } catch {
    return abs;
  }
}

/**
 * A clean child environment.
 *
 * Every XWEN_* var is dropped — the sweep must grade the shipped kernel paths,
 * and an inherited toggle would apply to every arm at once, silently retuning
 * against a path nobody asked for — then XWEN_BENCH is set back on deliberately.
 *
 * Only ONE HF variable is passed, the resolved absolute cache root. Notably no
 * `HF_TOKEN`: every checkpoint is verified present in the cache before the first
 * run, so a child never has cause to talk to the network, and a credential that
 * is never passed cannot leak into the raw records.
 */
function childEnv(): Record<string, string> {
  const e: Record<string, string> = {};
  for (const k of ["PATH", "HOME", "TMPDIR", "LANG", "LC_ALL"]) {
    const v = process.env[k];
    if (v !== undefined) e[k] = v;
  }
  if (HF_CACHE_ROOT) e.HF_HUB_CACHE = HF_CACHE_ROOT;
  e.XWEN_BENCH = "1";
  return e;
}

// ------------------------------------------------------------------- machine

/** The raw `(low)powermode` line from `pmset -g`, or null. Recorded, never
 *  interpreted: this machine emits no `powermode` key, so a run can never be
 *  positive evidence of high-power mode. */
async function powerModeLine(): Promise<string | null> {
  const proc = Bun.spawn({ cmd: ["pmset", "-g"], stdout: "pipe", stderr: "ignore" });
  const out = await new Response(proc.stdout).text();
  await proc.exited;
  const line = out.split("\n").find((l) => /\b(?:low)?powermode\b/.test(l));
  return line ? line.trim() : null;
}

/** MODEL_PROCS plus the selected binary's own basename (regex-escaped — these
 *  are `pgrep -x` patterns, so a `.` in a filename would otherwise match any
 *  character). Deduped so the common `--binary .../xwen` adds nothing. */
function contentionPatterns(): string[] {
  const own = basename(OPTS?.binary ?? "").replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return [...new Set([...MODEL_PROCS, ...(own ? [own] : [])])];
}

function modelProcRunning(): string | null {
  for (const name of contentionPatterns()) {
    const p = Bun.spawnSync(["pgrep", "-x", name]);
    const pids = p.stdout.toString().trim();
    if (pids) return `${name}: ${pids.split("\n").join(",")}`;
  }
  return null;
}

/** Block until no other model process is running. Two ~20 GB loads fit RAM but
 *  not comfort, and a contended run reads low in EVERY arm (measured: three
 *  runs 3x low across the board). */
async function waitForQuietMachine(what: string): Promise<void> {
  const deadline = Date.now() + CONTENTION_WAIT_MS;
  for (;;) {
    const hit = modelProcRunning();
    if (!hit) return;
    if (Date.now() > deadline) {
      die(`contention persisted 5 min before ${what} — ${hit}`);
    }
    console.error(`  CONTENTION before ${what}: ${hit} — waiting 10s`);
    await Bun.sleep(10_000);
  }
}

// -------------------------------------------------------------------- stats

interface Stats {
  decodeTps: number | null;
  decodeTokens: number | null;
  prefillTps: number | null;
  rounds: number | null;
  paused: number | null;
  drafted: number | null;
  accepted: number | null;
  acceptPct: number | null;
  verifyMsPerRound: number | null;
  /** Drafter cost per round that actually ran the drafter (the stats line's
   * `ms/draft` figure — NOT averaged over all rounds like verifyMsPerRound). */
  draftMsPerDraft: number | null;
}

/** Parse the `--stats` block (all of it on stderr; see src/bin/xwen/main.rs). */
function parseStats(err: string): Stats {
  const num = (re: RegExp): number | null => {
    const m = err.match(re);
    return m ? Number(m[1]) : null;
  };
  const hasSpec = /^spec:/m.test(err);
  return {
    decodeTps: num(/^decode:\s+\d+ tokens in [\d.]+s \(([\d.]+) tok\/s\)/m),
    decodeTokens: num(/^decode:\s+(\d+) tokens/m),
    prefillTps: num(/^prefill:\s+\d+ tokens in [\d.]+s \(([\d.]+) tok\/s\)/m),
    rounds: num(/^spec:\s+(\d+) rounds/m),
    // The "(N paused)" clause is omitted entirely when nothing paused, so an
    // absent clause on a run that DID speculate means zero, not unknown.
    paused: num(/^spec:\s+\d+ rounds \((\d+) paused\)/m) ?? (hasSpec ? 0 : null),
    drafted: num(/(\d+) drafted/),
    accepted: num(/(\d+) accepted \(/),
    acceptPct: num(/accepted \(([\d.]+)%\)/),
    verifyMsPerRound: num(/verify [\d.]+s \((\d+)ms\/round\)/),
    draftMsPerDraft: num(/draft [\d.]+s \((\d+)ms\/draft\)/),
  };
}

// --------------------------------------------------------------------- runs

interface RunRecord {
  stage: 1 | 2;
  size: ModelSize;
  prompt: string;
  arm: string;
  cell: string;
  rep: number;
  cmd: string[];
  /** Which variables the child was given. NAMES only — a value here would be
   *  written to disk, and an inherited credential (HF_TOKEN was the one that
   *  prompted this) must never be persisted into a benchmark artifact. */
  envNames: string[];
  /** The XWEN_* values, which ARE the scientifically load-bearing ones: they
   *  select kernel paths, so a reader has to be able to see exactly which
   *  toggles a number was produced under. Never credentials. */
  envXwen: Record<string, string>;
  startedAt: string;
  wallSecs: number;
  exitCode: number | null;
  timedOut: boolean;
  stats: Stats;
  stdout: string;
  stderr: string;
}

interface Cell {
  stage: 1 | 2;
  size: ModelSize;
  prompt: string;
  arm: Arm;
  runs: RunRecord[];
}

const RECORDS: RunRecord[] = [];
const FAILURES: string[] = [];

let RAW_PATH = "";
let RAW_META: Record<string, unknown> = {};

/**
 * Rewrite the raw dump. Atomic and 0600.
 *
 * Atomic because this is called after EVERY run and the file is the crash
 * guarantee: an in-place rewrite truncates the previous good dump first, so a
 * kill landing inside that window destroys exactly the data the frequent
 * flushing exists to protect. Write a sibling, fsync it, rename over.
 *
 * 0600 because a benchmark artifact in a world-readable /tmp should not be one
 * either — cheap, and it keeps the file's audience the operator who made it.
 */
function flushRaw(): void {
  if (!RAW_PATH) return;
  const json = JSON.stringify({ ...RAW_META, runs: RECORDS }, null, 2);
  const tmp = `${RAW_PATH}.tmp`;
  const fd = openSync(tmp, "w", 0o600);
  try {
    writeFileSync(fd, json);
    fsyncSync(fd);
  } finally {
    closeSync(fd);
  }
  renameSync(tmp, RAW_PATH);
}

async function spawnWithTimeout(
  cmd: string[],
  env: Record<string, string>,
  timeoutMs: number,
): Promise<{ stdout: string; stderr: string; exitCode: number | null; timedOut: boolean }> {
  const proc = Bun.spawn({ cmd, cwd: ROOT, env, stdin: "ignore", stdout: "pipe", stderr: "pipe" });
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    proc.kill(9);
  }, timeoutMs);
  try {
    const [stdout, stderr] = await Promise.all([
      new Response(proc.stdout).text(),
      new Response(proc.stderr).text(),
    ]);
    const exitCode = await proc.exited;
    return { stdout, stderr, exitCode, timedOut };
  } finally {
    clearTimeout(timer);
  }
}

let OPTS: Opts;

/** Test seam: the contention patterns depend on the selected binary, and that
 *  is the part of the guard worth asserting offline (pgrep's own visibility
 *  varies by sandbox, so a live-process test proves nothing portable). */
function setOptsForTest(o: Opts): void {
  OPTS = o;
}

/**
 * Is this run a valid observation? The single predicate both the failure
 * reporting and the scoring use, so a run can never be announced as FAILED and
 * then still feed a median (or the reverse).
 *
 * The short-run clause is the subtle one. `-n 128` is a MAXIMUM: a run that
 * hits EOG at four tokens exits 0 and reports a perfectly well-formed tok/s
 * that is boost-clock fiction, not a steady-state rate. Worse, it is
 * arm-dependent — speculative near-tie divergence can make one arm terminate
 * early where plain does not — so accepting it would compare a sprint against
 * a marathon and call the sprinter faster.
 */
function usableRun(r: RunRecord): boolean {
  return !r.timedOut && r.exitCode === 0 && r.stats.decodeTps !== null && r.stats.decodeTokens === N_TOKENS;
}

async function runOne(stage: 1 | 2, size: ModelSize, prompt: string, arm: Arm, rep: number): Promise<RunRecord> {
  const cmd = [OPTS.binary, ...argsFor(size, prompt, arm)];
  const env = childEnv();
  const what = `${size}/${prompt}/${arm.label} rep${rep}`;
  await waitForQuietMachine(what);

  const startedAt = new Date().toISOString();
  const t0 = performance.now();
  const r = await spawnWithTimeout(cmd, env, OPTS.timeoutMs);
  const wallSecs = (performance.now() - t0) / 1000;
  const stats = parseStats(r.stderr);

  const rec: RunRecord = {
    stage,
    size,
    prompt,
    arm: arm.label,
    cell: cellKey(stage, size, prompt, arm),
    rep,
    cmd,
    envNames: Object.keys(env).sort(),
    envXwen: Object.fromEntries(Object.entries(env).filter(([k]) => k.startsWith("XWEN_"))),
    startedAt,
    wallSecs,
    exitCode: r.exitCode,
    timedOut: r.timedOut,
    stats,
    stdout: r.stdout,
    stderr: r.stderr,
  };
  RECORDS.push(rec);
  flushRaw();

  if (!usableRun(rec)) {
    const why = r.timedOut
      ? `TIMEOUT after ${(OPTS.timeoutMs / 1000).toFixed(0)}s`
      : r.exitCode !== 0
        ? `exit ${r.exitCode}`
        : stats.decodeTps === null
          ? "no decode rate in --stats output"
          : `decoded ${stats.decodeTokens ?? "?"} of ${N_TOKENS} tokens`;
    FAILURES.push(`stage ${stage} ${what}: ${why}`);
    console.log(`  FAILED  ${what}: ${why} (raw record #${RECORDS.length} in ${RAW_PATH})`);
    console.log(`          ${r.stderr.trimEnd().split("\n").slice(-6).join("\n          ")}`);
    return rec;
  }

  const specNote =
    stats.rounds !== null
      ? `, ${stats.rounds} rounds (${stats.paused} paused), ${stats.acceptPct ?? "?"}% accepted`
      : "";
  console.log(`  ${what}: ${stats.decodeTps!.toFixed(1)} tok/s${specNote} [${wallSecs.toFixed(1)}s wall]`);
  return rec;
}

/**
 * Measure every cell of one stage, interleaving arms across reps.
 *
 * EVERY cell is fresh, including configurations an earlier stage already
 * measured. Nothing is carried between stages: a stage is one self-contained
 * experiment whose arms all saw the same machine, and cross-stage reuse would
 * grade a stage-2 arm against a baseline from a different thermal epoch.
 */
async function runStage(stage: 1 | 2, size: ModelSize, arms: Arm[]): Promise<Cell[]> {
  const cells: Cell[] = [];
  for (const prompt of PROMPT_KINDS) {
    for (const arm of arms) cells.push({ stage, size, prompt, arm, runs: [] });
  }

  console.log(
    `\n-- stage ${stage} ${size}: ${arms.length} arms x ${PROMPT_KINDS.length} prompts x ${OPTS.reps} reps` +
      ` = ${cells.length * OPTS.reps} runs --`,
  );

  for (let rep = 1; rep <= OPTS.reps; rep++) {
    for (const cell of cells) {
      cell.runs.push(await runOne(stage, size, cell.prompt, cell.arm, rep));
    }
  }
  return cells;
}

// ------------------------------------------------------------------ scoring

function median(xs: number[]): number | null {
  if (xs.length === 0) return null;
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

/** Decode rates of the USABLE runs in a cell, in run order. */
function rates(cell: Cell): number[] {
  return cell.runs.filter(usableRun).map((r) => r.stats.decodeTps!);
}

interface ArmScore {
  arm: Arm;
  /** Per-prompt medians, null when the cell produced no usable rep. */
  medians: Record<string, number | null>;
  /** Mean of the per-prompt medians; null if any is missing. */
  mean: number | null;
  /** Ahead of plain's median on both prompts in EVERY rep. */
  qualifies: boolean;
  /** Why not, when it does not qualify. */
  note: string;
}

interface StageScores {
  scores: ArmScore[];
  plainMedians: Record<string, number | null>;
  /** Per prompt: did the plain arm produce a COMPLETE set of reps? */
  baselineValid: Record<string, boolean>;
  /** Prompts whose baseline is unusable; non-empty means no verdict is possible. */
  invalidBaselines: string[];
}

/**
 * Score one stage's arms against its plain arm.
 *
 * Plain is the ruler, so it is held to a stricter standard than the arms it
 * measures: its median must come from a COMPLETE set of reps. A baseline built
 * from one surviving run of three is not a median, it is a single sample, and
 * an arm compared against a low survivor would be certified robust on the
 * strength of the baseline's bad luck. When any prompt's baseline is
 * incomplete, nothing qualifies for that stage and the recommendation says the
 * checkpoint is invalid rather than reporting a winner.
 */
function scoreArms(cells: Cell[], arms: Arm[], reps: number): StageScores {
  const byArm = (arm: Arm, prompt: string) => cells.find((c) => c.arm.label === arm.label && c.prompt === prompt);

  const plainMedians: Record<string, number | null> = {};
  const baselineValid: Record<string, boolean> = {};
  const invalidBaselines: string[] = [];
  for (const prompt of PROMPT_KINDS) {
    const c = byArm(PLAIN, prompt);
    const rs = c ? rates(c) : [];
    plainMedians[prompt] = median(rs);
    baselineValid[prompt] = rs.length === reps;
    if (!baselineValid[prompt]) invalidBaselines.push(prompt);
  }

  const scores: ArmScore[] = [];
  for (const arm of arms) {
    if (arm.plain) continue;
    const medians: Record<string, number | null> = {};
    let qualifies = true;
    const notes: string[] = [];
    for (const prompt of PROMPT_KINDS) {
      const cell = byArm(arm, prompt);
      const rs = cell ? rates(cell) : [];
      medians[prompt] = median(rs);
      const base = plainMedians[prompt];
      if (!baselineValid[prompt] || base === null) {
        qualifies = false;
        const got = (byArm(PLAIN, prompt)?.runs ?? []).filter(usableRun).length;
        notes.push(`plain baseline on ${prompt} is only ${got}/${reps} reps`);
        continue;
      }
      // Every rep has to clear plain, so a rep that failed is a missing
      // observation the criterion cannot be evaluated over.
      if (rs.length < reps) {
        qualifies = false;
        notes.push(`${reps - rs.length}/${reps} reps failed on ${prompt}`);
        continue;
      }
      const losers = rs.filter((v) => v <= base).length;
      if (losers > 0) {
        qualifies = false;
        notes.push(`${losers}/${reps} reps at or below plain on ${prompt}`);
      }
    }
    const ms = PROMPT_KINDS.map((p) => medians[p]);
    const mean = ms.every((v) => v !== null && v !== undefined)
      ? (ms as number[]).reduce((a, b) => a + b, 0) / ms.length
      : null;
    scores.push({ arm, medians, mean, qualifies, note: notes.join("; ") });
  }
  return { scores, plainMedians, baselineValid, invalidBaselines };
}

// ------------------------------------------------------------------- output

function fmtTps(v: number | null | undefined): string {
  return v === null || v === undefined ? "  --  " : v.toFixed(1).padStart(6);
}

function printCellTable(stage: 1 | 2, size: ModelSize, cells: Cell[], arms: Arm[], plainMedians: Record<string, number | null>): void {
  console.log(`\n  ${size} stage ${stage} — median tok/s, then every rep in run order`);
  console.log(
    `    ${"prompt".padEnd(6)} ${"arm".padEnd(13)} ${"median".padStart(6)} ${"vs plain".padStart(9)}  ${"reps".padEnd(26)} spec (rounds/paused/accept%)`,
  );
  for (const prompt of PROMPT_KINDS) {
    for (const arm of arms) {
      const cell = cells.find((c) => c.arm.label === arm.label && c.prompt === prompt);
      if (!cell) continue;
      const rs = rates(cell);
      const med = median(rs);
      const base = plainMedians[prompt];
      const delta =
        med !== null && base !== null && base !== undefined && !arm.plain
          ? `${(((med - base) / base) * 100 >= 0 ? "+" : "")}${(((med - base) / base) * 100).toFixed(1)}%`
          : "";
      // A run that is not usable prints FAIL even when it reported a rate — a
      // short or crashed run's number is not an observation, and showing it
      // next to real ones invites reading it as one.
      const repStr = cell.runs
        .map((r) => (usableRun(r) ? r.stats.decodeTps!.toFixed(1) : "FAIL"))
        .join(" ");
      const withStats = cell.runs.filter((r) => r.stats.rounds !== null);
      const spec = withStats.length
        ? `${median(withStats.map((r) => r.stats.rounds!))!.toFixed(0)}/` +
          `${median(withStats.map((r) => r.stats.paused ?? 0))!.toFixed(0)}/` +
          `${(median(withStats.map((r) => r.stats.acceptPct ?? 0)) ?? 0).toFixed(1)}%`
        : "-";
      const marks = arm.diagnostic ? " (never-pause diagnostic)" : "";
      console.log(
        `    ${prompt.padEnd(6)} ${arm.label.padEnd(13)} ${fmtTps(med)} ${delta.padStart(9)}  ${repStr.padEnd(26)} ${spec}${marks}`,
      );
    }
  }
}

/**
 * Is this arm the configuration the checkpoint already ships, for the knobs the
 * stage is choosing? Ties resolve toward it.
 *
 * Per-size, because the status quo is: the 27B ships p_min 0.5 and the 35B 0.3,
 * so a shared value would break the tie toward a setting the checkpoint does
 * not actually have. Stage 1 tests BOTH its knobs — an arm at the shipped p_min
 * but a different depth is a change, and a tie gives no more reason to install
 * it than a tie on p_min would.
 */
function isStatusQuo(stage: 1 | 2, size: ModelSize, a: Arm): boolean {
  return stage === 1
    ? a.pMin === shippedPMin(size) && a.draftMax === shippedDraftMax(size)
    : a.margin === SHIPPED_MARGIN;
}

/**
 * Everything at the maximum mean, compared at FULL float precision.
 *
 * Ties are not hypothetical: the Rust side prints tok/s to one decimal, so two
 * arms landing on identical medians is ordinary. Taking `sort()[0]` would then
 * pick by grid order, which means reversing `--p-min-grid` could change the
 * recommendation without changing a single measurement.
 */
function maxima(pool: ArmScore[]): ArmScore[] {
  const withMean = pool.filter((s) => s.mean !== null);
  if (withMean.length === 0) return [];
  const top = Math.max(...withMean.map((s) => s.mean!));
  return withMean.filter((s) => s.mean === top);
}

function printStageVerdict(stage: 1 | 2, size: ModelSize, stageScores: StageScores): ArmScore | null {
  const { scores, invalidBaselines } = stageScores;

  console.log(`\n  ${size} stage ${stage} verdict (criterion: ahead of plain's median on BOTH prompts in EVERY rep)`);
  for (const s of scores) {
    const tag = s.qualifies ? "QUALIFIES" : "no       ";
    const meanStr = s.mean === null ? " --" : s.mean.toFixed(1);
    console.log(`    ${tag} ${s.arm.label.padEnd(13)} mean-of-medians ${meanStr.padStart(6)} tok/s${s.note ? `  (${s.note})` : ""}`);
  }

  // Plain is the ruler. A broken ruler invalidates every comparison made with
  // it, so this is reported as no result at all rather than as a verdict.
  if (invalidBaselines.length) {
    console.log(
      `  -> stage ${stage}: INVALID — the plain baseline is incomplete on ${invalidBaselines.join(" and ")}. ` +
        `Every arm was measured against a ruler built from missing runs, so no verdict is possible. Re-run this stage.`,
    );
    return null;
  }

  const decide = (pool: ArmScore[], kind: "qualifying" | "best-by-mean"): ArmScore | null => {
    const top = maxima(pool);
    if (top.length <= 1) return top[0] ?? null;
    // Prefer the status quo when it is among the tied: changing a shipped
    // constant needs evidence that it is BETTER, and a tie is not that.
    const shipped = top.find((s) => isStatusQuo(stage, size, s.arm));
    if (shipped) {
      console.log(
        `     tie at ${top[0].mean!.toFixed(3)} tok/s between ${top.map((s) => s.arm.label).join(", ")} ` +
          `(${kind}) — resolved to the shipped value, which a tie gives no reason to change`,
      );
      return shipped;
    }
    console.log(
      `     TIE at ${top[0].mean!.toFixed(3)} tok/s between ${top.map((s) => s.arm.label).join(", ")} (${kind}), ` +
        `and the shipped value is not among them — this sweep cannot separate them`,
    );
    return null;
  };

  const qualified = scores.filter((s) => s.qualifies && s.mean !== null);
  if (qualified.length) {
    const w = decide(qualified, "qualifying");
    if (w) {
      console.log(`  -> stage ${stage} winner: ${w.arm.label} (${w.mean!.toFixed(1)} tok/s mean-of-medians)`);
      return w;
    }
    console.log(`  -> stage ${stage}: no winner — the qualifying arms are tied. Re-run with more reps to separate them.`);
    return null;
  }

  const fallback = decide(scores, "best-by-mean");
  if (!fallback) {
    const anyUsable = scores.some((s) => s.mean !== null);
    console.log(
      anyUsable
        ? `  -> stage ${stage}: NO arm beat plain on both prompts in every rep, and the best are tied — nothing to recommend`
        : `  -> stage ${stage}: NO usable arm (every cell failed) — nothing to recommend`,
    );
    return null;
  }
  console.log(
    `  -> stage ${stage}: NO arm beat plain on both prompts in every rep. Best by mean is ` +
      `${fallback.arm.label} (${fallback.mean!.toFixed(1)} tok/s), but it is NON-ROBUST: ${fallback.note}`,
  );
  return null;
}

// --------------------------------------------------------------------- plan

interface PlannedRun {
  stage: 1 | 2;
  size: ModelSize;
  prompt: string;
  arm: Arm;
  rep: number;
}

/**
 * The run matrix as it would execute, for --dry-run.
 *
 * The count here is exact, not a lower bound: nothing is reused between stages,
 * so no cell's existence depends on which p_min stage 1 happens to pick. Only
 * stage 2's p_min VALUE is unknown before stage 1 runs, and that is shown
 * symbolically.
 */
/** The depths to sweep for one size: the named grid, else that checkpoint's
 *  shipped depth alone. */
function depthsFor(opts: Opts, size: ModelSize): number[] {
  return opts.depthGrid ?? [shippedDraftMax(size)];
}

function planRuns(opts: Opts): PlannedRun[] {
  const plan: PlannedRun[] = [];
  for (const size of opts.sizes) {
    const stage1Runs = opts.stages.includes(1);
    for (const stage of opts.stages) {
      const arms =
        stage === 1
          ? stage1Arms(opts.pMinGrid, depthsFor(opts, size))
          : stage2Arms(
              stage1Runs ? null : shippedPMin(size),
              stage1Runs ? null : shippedDraftMax(size),
              opts.marginGrid,
            );
      const cells: { prompt: string; arm: Arm }[] = [];
      for (const prompt of PROMPT_KINDS) for (const arm of arms) cells.push({ prompt, arm });
      for (let rep = 1; rep <= opts.reps; rep++) {
        for (const c of cells) plan.push({ stage, size, prompt: c.prompt, arm: c.arm, rep });
      }
    }
  }
  return plan;
}

function printDryRun(opts: Opts): void {
  console.log("=== planned run matrix (--dry-run: nothing is executed) ===");
  console.log(
    `models=[${opts.sizes.join(",")}] stages=[${opts.stages.join(",")}] reps=${opts.reps} ` +
      `p-min-grid=[${opts.pMinGrid.join(",")}] margin-grid=[${opts.marginGrid.join(",")}] ` +
      `depth-grid=[${
        opts.depthGrid
          ? opts.depthGrid.join(",")
          : opts.sizes.map((s) => `${s} ${shippedDraftMax(s)}`).join(", ") + " (shipped)"
      }] ` +
      `(stage 1 holds pause_margin at ${fmtVal(SHIPPED_MARGIN)}; stage 2 always adds margins ` +
      `${fmtVal(NEVER_PAUSE_MARGIN)} and ${fmtVal(SHIPPED_MARGIN)})`,
  );
  console.log(`binary=${opts.binary}${existsSync(opts.binary) ? "" : "  [MISSING — a real run would abort]"}`);
  console.log(`per-run timeout=${(opts.timeoutMs / 1000).toFixed(0)}s, env=${Object.keys(childEnv()).sort().join(",")}`);
  console.log("no cell is reused between stages — every stage-2 arm, plain included, is re-measured");
  if (opts.stages.includes(2)) {
    console.log(
      opts.stages.includes(1)
        ? "stage 2 p_min prints as <stage1-winner> — it is stage 1's winner, unknown until stage 1 runs"
        : `stage 2 p_min is each checkpoint's shipped default (${opts.sizes
            .map((s) => `${s} ${shippedPMin(s)}`)
            .join(", ")}) — --stage 2 skips the p_min sweep`,
    );
  }

  const plan = planRuns(opts);
  let i = 0;
  let header = "";
  for (const p of plan) {
    const h = `${p.size} stage ${p.stage}`;
    if (h !== header) {
      header = h;
      const stageRuns = plan.filter((q) => q.size === p.size && q.stage === p.stage).length;
      console.log(`\n-- ${h}: ${stageRuns} runs --`);
    }
    i++;
    const shown = displayArgs(p.size, p.prompt, p.arm);
    console.log(`  ${String(i).padStart(3)}. rep${p.rep} ${p.prompt.padEnd(4)} ${p.arm.label.padEnd(13)} XWEN_BENCH=1 ${opts.binary} ${shown}`);
  }
  console.log(`\ntotal: ${plan.length} model runs (${N_TOKENS} greedy tokens each, plus a model load)`);
}

// ------------------------------------------------------------- recommendation

/** Where each default lives in the tree. The two knobs are shaped differently:
 *  `draft_p_min` became per-checkpoint on 2026-08-08 and now has a single home
 *  that the CLI and the serve merge both resolve against, while `pause_margin`
 *  is still one shared value spelled out at three sites. */
const P_MIN_LOCATION =
  "src/hub.rs             Model::draft_p_min_default() — one arm per checkpoint";
const DRAFT_MAX_LOCATION =
  "src/hub.rs             Model::draft_max_default() — one arm per drafter KIND";
const MARGIN_LOCATIONS = [
  "src/generate.rs        SpecParams::default() — pause_margin",
  "src/serve/config.rs    DEFAULT_DRAFT_PAUSE_MARGIN",
  "src/bin/xwen/main.rs   --draft-pause-margin clap default",
];

interface Recommendation {
  /** Stage 1's winning p_min, or null — never the fallback stage 2 ran at. */
  pMin: number | null;
  /** Stage 1's winning depth, or null. Null also when the sweep never varied
   *  depth, which is not a result and must not print as one. */
  draftMax: number | null;
  /** Whether depth was swept at all. */
  depthSwept: boolean;
  margin: number | null;
  stage1Ran: boolean;
  stage2Ran: boolean;
  /** False when either stage failed to produce a qualifying arm. */
  robust: boolean;
  /** Prompts whose plain baseline was incomplete, per stage. Non-empty means
   *  this checkpoint's numbers cannot support any recommendation at all. */
  invalidBaselines: string[];
}

function printRecommendation(rec: Map<ModelSize, Recommendation>): void {
  console.log("\n==================== recommendation ====================");
  for (const [size, r] of rec) {
    if (r.invalidBaselines.length) {
      console.log(
        `  ${size}: INVALID — the plain baseline was incomplete (${[...new Set(r.invalidBaselines)].join(", ")}). ` +
          `No recommendation; re-run this checkpoint.`,
      );
      continue;
    }
    const p =
      r.pMin !== null ? String(r.pMin) : r.stage1Ran ? "KEEP CURRENT (no stage-1 winner)" : "not measured (stage 1 skipped)";
    const m =
      r.margin !== null ? String(r.margin) : r.stage2Ran ? "KEEP CURRENT (no stage-2 winner)" : "not measured (stage 2 skipped)";
    // "not swept" and "swept and unchanged" are different findings, and only the
    // second is evidence about the depth.
    const d = !r.depthSwept
      ? `not swept (held at the shipped ${shippedDraftMax(size)})`
      : r.draftMax !== null
        ? String(r.draftMax)
        : r.stage1Ran
          ? "KEEP CURRENT (no stage-1 winner)"
          : "not measured (stage 1 skipped)";
    console.log(
      `  ${size}: draft_p_min = ${p}, draft_max = ${d}, pause_margin = ${m}` +
        `${r.robust ? "" : "   [NON-ROBUST: see the stage verdicts above]"}`,
    );
  }

  // This compares the value each checkpoint would END UP with, not just the
  // measured winners: a checkpoint with no winner keeps the shipped one. What
  // divergence means differs per knob — draft_p_min has a per-model home, so
  // two winners can both be installed, while pause_margin is one shared value
  // and a split there forces a choice. Checkpoints with an invalid baseline are
  // excluded: they have no outcome to compare, and the INVALID line above
  // already says the sweep cannot conclude anything for them.
  const usable = [...rec.entries()].filter(([, r]) => r.invalidBaselines.length === 0);
  // The unchanged p_min is that CHECKPOINT's shipped default, not one global
  // value: a 35B with no winner keeps 0.3 while a 27B with no winner keeps 0.5,
  // so comparing both against a single constant would invent a divergence
  // between two checkpoints that each kept their own default (or hide a real
  // one).
  const effective = (size: ModelSize, r: Recommendation) => ({
    p: r.pMin ?? shippedPMin(size),
    m: r.margin ?? SHIPPED_MARGIN,
  });
  const pMins = new Set(usable.map(([size, r]) => effective(size, r).p));
  const margins = new Set(usable.map(([, r]) => r.margin ?? SHIPPED_MARGIN));
  if (usable.length > 1 && (pMins.size > 1 || margins.size > 1)) {
    console.log("\n  The checkpoints would end up with DIFFERENT constants:");
    for (const [size, r] of usable) {
      const e = effective(size, r);
      const pSrc = r.pMin === null ? " (unchanged)" : "";
      const mSrc = r.margin === null ? " (unchanged)" : "";
      console.log(`    ${size}: draft_p_min ${e.p}${pSrc}, pause_margin ${e.m}${mSrc}`);
    }
    if (pMins.size > 1) {
      console.log(
        "  draft_p_min is per-checkpoint, so two different winners are not a conflict: install\n" +
          "  each one at its own arm of Model::draft_p_min_default().",
      );
    }
    if (margins.size > 1) {
      console.log(
        "  pause_margin is still ONE shared value, so differing margins do force a choice. Note\n" +
          "  that installing one checkpoint's winner also changes the other checkpoint to a value\n" +
          "  this sweep did not measure as a winner for it.",
      );
    }
  }
  console.log("\n  These are the only places a default lives. This script never edits them:");
  console.log("    draft_p_min — per checkpoint, one site:");
  console.log(`      ${P_MIN_LOCATION}`);
  console.log("    draft_max — per drafter kind, one site:");
  console.log(`      ${DRAFT_MAX_LOCATION}`);
  console.log("    pause_margin — one shared value, three sites:");
  for (const loc of MARGIN_LOCATIONS) console.log(`      ${loc}`);
  console.log(
    "\n  MIRROR BOTH PLACES: installing a new draft_p_min or draft_max means editing src/hub.rs\n" +
      // The WHOLE table, not just the swept checkpoints: whoever edits the Rust
      // match needs to see every arm to keep them in step.
      `    AND this script's tables (currently SHIPPED_P_MIN ${Object.entries(SHIPPED_P_MIN)
        .map(([s, v]) => `${s} ${v}`)
        .join(", ")}; SHIPPED_DRAFT_MAX ${Object.entries(SHIPPED_DRAFT_MAX)
        .map(([s, v]) => `${s} ${v}`)
        .join(", ")}).\n` +
      "    The Rust functions are what ship; the tables are what the NEXT sweep grades \"keep\n" +
      "    current\" against, breaks ties toward, and compares across checkpoints. A stale table\n" +
      "    measures the wrong status quo in silence. Note draft_max is keyed by KIND in Rust and\n" +
      "    by SIZE here, so changing it for one checkpoint changes it for every checkpoint that\n" +
      "    shares that drafter kind — check the other arms before installing one.",
  );
  console.log(
    "\n  A changed default is a shipped decision: it needs a dated docs/log.md entry and a\n" +
      "  docs/decisions.md update recording what the new cost curve made true, per CLAUDE.md.",
  );
}

// --------------------------------------------------------------------- main

async function main(): Promise<void> {
  OPTS = parseArgs(process.argv.slice(2));

  // Before the dry run too, not just before a real one: a checkpoint with no
  // sidecar has no drafted arm and no fitted floor, so its "planned matrix"
  // would print arms at `--draft-p-min undefined` — a matrix nothing could run,
  // which is exactly what a dry run exists to rule out.
  for (const size of OPTS.sizes) {
    if (!CHECKPOINTS[size].drafter) {
      die(`${size} ships no drafter sidecar, so there is no drafted arm to sweep.`);
    }
  }

  if (OPTS.dryRun) {
    printDryRun(OPTS);
    return;
  }

  if (!existsSync(OPTS.binary)) {
    die(
      `binary not found: ${OPTS.binary}\n` +
        `  Build it first (building is deliberately not this script's job — a sweep must\n` +
        `  grade a binary you chose):\n` +
        `    cargo build --release --bin xwen\n` +
        `  or point at another one with --binary <path>.`,
    );
  }
  // Pin the cache root before ANY lookup, so this process and every child
  // resolve against the same absolute path (see resolveHfCacheRoot).
  HF_CACHE_ROOT = resolveHfCacheRoot();
  process.env.HF_HUB_CACHE = HF_CACHE_ROOT;

  // Fail before the first 20 GB load rather than after it: a missing checkpoint
  // would otherwise trigger a multi-GB download mid-sweep, and a missing
  // drafter would make every drafted arm silently fall back to plain decode.
  for (const size of OPTS.sizes) {
    try {
      officialModel(size);
    } catch (e) {
      die(String((e as Error).message));
    }
    // Refused above, before the dry run; named again here for the message.
    const drafter = CHECKPOINTS[size].drafter!;
    if (!officialDrafter(size)) {
      die(
        `the drafter sidecar for ${size} is not in the Hugging Face cache ` +
          `(${drafter}); run \`xwen fetch --model-size ${size}\`. ` +
          `Every drafted arm needs it.`,
      );
    }
  }

  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  RAW_PATH = `/tmp/retune-draft-${stamp}.json`;
  mkdirSync(dirname(RAW_PATH), { recursive: true });

  const startPower = await powerModeLine();
  RAW_META = {
    startedAt: new Date().toISOString(),
    opts: { ...OPTS },
    nTokens: N_TOKENS,
    shippedPMin: SHIPPED_P_MIN,
    shippedDraftMax: SHIPPED_DRAFT_MAX,
    shippedMargin: SHIPPED_MARGIN,
    neverPauseMargin: NEVER_PAUSE_MARGIN,
    hfCacheRoot: HF_CACHE_ROOT,
    powerModeStart: startPower,
    prompts: DRAFT_PROMPTS,
  };
  flushRaw();

  console.log("=== retune-draft ===");
  console.log(`binary   ${OPTS.binary}`);
  console.log(`models   ${OPTS.sizes.join(", ")}   stages ${OPTS.stages.join(",")}   reps ${OPTS.reps}`);
  console.log(`p_min    ${OPTS.pMinGrid.join(", ")} (at pause_margin ${fmtVal(SHIPPED_MARGIN)})`);
  console.log(
    `depth    ${
      OPTS.depthGrid
        ? `${OPTS.depthGrid.join(", ")} (crossed with p_min in stage 1)`
        : `${OPTS.sizes.map((s) => `${s} ${shippedDraftMax(s)}`).join(", ")} (shipped; not swept)`
    }`,
  );
  console.log(
    `margin   ${OPTS.marginGrid.join(", ")} (+ ${fmtVal(NEVER_PAUSE_MARGIN)} never-pause diagnostic, ` +
      `+ ${fmtVal(SHIPPED_MARGIN)} shipped)`,
  );
  // The status quo every verdict is measured against, printed so a stale mirror
  // of src/hub.rs is visible at the top of the run rather than inferred later.
  console.log(
    `shipped  draft_p_min ${OPTS.sizes.map((s) => `${s} ${shippedPMin(s)}`).join(", ")}, ` +
      `draft_max ${OPTS.sizes.map((s) => `${s} ${shippedDraftMax(s)}`).join(", ")}, ` +
      `pause_margin ${fmtVal(SHIPPED_MARGIN)} ` +
      `(mirrors Model::draft_p_min_default / draft_max_default, src/hub.rs)`,
  );
  console.log(`hf cache ${HF_CACHE_ROOT}`);
  // Recorded, not interpreted: no `powermode` key on this machine means high
  // power mode is never positively confirmable, so it is never claimed.
  console.log(`pmset    ${startPower ?? "(no (low)powermode key reported)"}`);
  console.log(`raw      ${RAW_PATH}`);

  const recommendation = new Map<ModelSize, Recommendation>();

  for (const size of OPTS.sizes) {
    console.log(`\n================ ${size} ================`);
    let winnerPMin: number | null = null;
    let winnerDraftMax: number | null = null;
    let robust = true;
    const invalidBaselines: string[] = [];

    /** Measure, tabulate and grade one stage; returns its winning arm or null. */
    const doStage = async (stage: 1 | 2, arms: Arm[]): Promise<ArmScore | null> => {
      const cells = await runStage(stage, size, arms);
      const scored = scoreArms(cells, arms, OPTS.reps);
      printCellTable(stage, size, cells, arms, scored.plainMedians);
      for (const p of scored.invalidBaselines) invalidBaselines.push(`stage ${stage} ${p}`);
      const w = printStageVerdict(stage, size, scored);
      if (!w) robust = false;
      return w;
    };

    if (OPTS.stages.includes(1)) {
      const w = await doStage(1, stage1Arms(OPTS.pMinGrid, depthsFor(OPTS, size)));
      if (w) {
        winnerPMin = w.arm.pMin;
        winnerDraftMax = w.arm.draftMax;
      }
    }

    // Stage 2 needs a p_min. Without a stage-1 winner it sweeps the margin at
    // THIS checkpoint's shipped value, which still answers "is the controller
    // earning its keep" even though the p_min question came back unresolved.
    // Per-size matters here: running the 27B's margin sweep at the 35B's 0.3
    // would tune the margin around a drafting depth the 27B does not ship.
    let marginPMin = winnerPMin;
    let marginDraftMax = winnerDraftMax;
    if (OPTS.stages.includes(2) && marginPMin === null) {
      marginPMin = shippedPMin(size);
      marginDraftMax = shippedDraftMax(size);
      console.log(
        `\n  stage 2 runs at ${size}'s shipped p_min ${shippedPMin(size)} and depth ` +
          `${shippedDraftMax(size)} (no stage-1 winner to carry).`,
      );
    }

    let winnerMargin: number | null = null;
    if (OPTS.stages.includes(2)) {
      const w = await doStage(2, stage2Arms(marginPMin, marginDraftMax, OPTS.marginGrid));
      if (w) winnerMargin = w.arm.margin;
    }

    // `winnerPMin`, not the value stage 2 actually ran at: a fallback to the
    // shipped p_min is not a measurement result and must not print as one.
    recommendation.set(size, {
      pMin: winnerPMin,
      draftMax: winnerDraftMax,
      depthSwept: depthsFor(OPTS, size).length > 1,
      margin: winnerMargin,
      stage1Ran: OPTS.stages.includes(1),
      stage2Ran: OPTS.stages.includes(2),
      robust,
      invalidBaselines,
    });
  }

  const endPower = await powerModeLine();
  RAW_META = { ...RAW_META, finishedAt: new Date().toISOString(), powerModeEnd: endPower, failures: FAILURES };
  flushRaw();

  printRecommendation(recommendation);

  console.log("\n  power mode (recorded, not interpreted — this machine has no `powermode` key):");
  console.log(`    start ${startPower ?? "(none)"}`);
  console.log(`    end   ${endPower ?? "(none)"}`);
  if (startPower !== endPower) {
    console.log("    THE POWER MODE CHANGED MID-SWEEP — the numbers above are not comparable.");
  }
  console.log(`\n  raw records: ${RAW_PATH}`);

  if (FAILURES.length) {
    console.log(`\n${FAILURES.length} run(s) FAILED — the sweep is incomplete and its verdicts are suspect:`);
    for (const f of FAILURES) console.log(`  ${f}`);
    process.exit(1);
  }
  if (startPower !== endPower) process.exit(1);
}

// Guarded so the scoring and table helpers above can be imported and exercised
// offline. Without it, `import { scoreArms } from "./retune-draft.ts"` would run
// the entire sweep as a side effect of the import — hours of model time and two
// 20 GB loads nobody asked for. The same trap parity-gate.ts documents.
if (import.meta.main) {
  main().catch((e) => {
    console.error("retune-draft: unexpected error:", e);
    flushRaw();
    process.exit(1);
  });
}

export {
  PLAIN, SHIPPED_DRAFT_MAX, SHIPPED_MARGIN, SHIPPED_P_MIN, cellKey, contentionPatterns, dedupe,
  fmtVal, maxima, median, parseStats, printCellTable, printStageVerdict, rates, scoreArms,
  setOptsForTest, stage1Arms, stage2Arms, usableRun,
};
export type { Arm, ArmScore, Cell, RunRecord, StageScores };
