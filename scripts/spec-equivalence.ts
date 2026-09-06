#!/usr/bin/env bun
// Speculative decoding must be lossless: `--draft` has to emit the same text as
// `--no-draft`, because every drafted token is only committed after the target's
// own forward agrees with it.
//
// Two modes, because they catch different bugs:
//
//   greedy    temperature 0. The sampler takes the argmax and never draws from
//             the RNG, so this checks the verify walk's token selection alone.
//   sampled   temperature > 0 at a fixed seed, with `--draft-p-min 0` and
//             `--draft-pause-margin 0` so every round drafts a full block and
//             nothing pauses into plain decoding. This is the only mode that can
//             catch the spec loop advancing the SAMPLER STREAM a different number
//             of times than the plain loop: one extra or missing draw reroutes
//             every token after it, so a stream bug diverges at the first
//             sampled token rather than somewhere in the middle. Auto-pause is
//             off because with it on, which rounds batch-verify depends on
//             wall-clock timing and a temperature>0 run is not reproducible from
//             a seed at all (see serve/config.rs, `pause_margin`).
//
// Both modes assert the spec path actually RAN. A run that paused into plain
// decoding, or one whose drafter never cleared `p_min`, would otherwise report a
// cheerful OK having exercised no verify or rollback at all.
//
// The known-benign divergence: the verify walk's batched forward over a drafted
// span reassociates its f32 sums differently from the single-token forward the
// plain loop runs, so at a NEAR-TIE the two argmaxes can pick different tokens
// and the texts fork from there. That is the same reassociation the decode
// parity tier's near-tie rule exists to grade (docs/parity.md), not a
// verify-walk bug, and it is why this is a hand-run diagnostic rather than a
// `cargo test` gate. A fork at the very first line in `sampled` mode is the
// opposite: that is the sampler stream, and it is a real bug.
//
// Both drafter kinds are covered, and they are not the same test. A DFlash
// block drafter proposes a whole block out of one forward, so its verify walk
// rolls back a block at a time; an MTP head chains one forward per step and
// feeds itself, so a rollback has to unwind the head's own carried hidden as
// well as the target's KV. The 3.8-27b arm is the only one that exercises the
// second. It was hand-run for the MTP arc's stage B and wired up here in stage
// C (docs/log.md 2026-08-15).
//
// Measured 2026-07-29 on the P9 adaptation: see docs/log.md.
//
// Usage:
//   bun scripts/spec-equivalence.ts                     # every model, both modes
//   bun scripts/spec-equivalence.ts --models 27b        # one model
//   bun scripts/spec-equivalence.ts --models 3.8-27b    # the MTP arm alone
//   bun scripts/spec-equivalence.ts --modes sampled --n 256
//   bun scripts/spec-equivalence.ts --presence-penalty 0   # pin the penalty on both arms

import { mkdirSync, writeFileSync, statSync, readdirSync } from "node:fs";
import { join, dirname } from "node:path";
import { DRAFT_PROMPTS } from "./lib/draft-prompts";

const repo = dirname(import.meta.dir);
const args = process.argv.slice(2);
const opt = (n: string, d: string) => {
  const i = args.indexOf(`--${n}`);
  return i >= 0 && args[i + 1] ? args[i + 1] : d;
};

const models = opt("models", "27b,35b,3.8-27b").split(",");
const modes = opt("modes", "greedy,sampled").split(",");
const nTokens = opt("n", "128");
// One value for every model on purpose, and deliberately below every shipped
// floor (0.5 / 0.3 / 0.5): this is a COVERAGE knob, not a fitted default. A
// lower floor drafts longer chains, which is more verify and more rollback per
// run — the opposite of what a sweep wants and exactly what a lossless-ness
// check wants.
const greedyPMin = opt("p-min", "0.3");
const seed = opt("seed", "42");
const temp = opt("temp", "0.8");
// Passthrough for the sampler's presence penalty, mirroring --temp. Unset, both
// arms take the checkpoint's own mode default, which is what ships — and since
// this harness sends a RAW prompt, that is the THINKING default (1.5 on the
// 35B-A3B, 0 elsewhere). Both arms always get the same value: the property under
// test is that drafting is lossless, so the penalty has to be live on both sides
// or the comparison is between two different samplers. Pin it to 0 to reproduce
// the pre-penalty behaviour.
const presencePenalty = args.includes("--presence-penalty")
  ? opt("presence-penalty", "0")
  : null;
const penaltyArgs = presencePenalty === null ? [] : ["--presence-penalty", presencePenalty];
const outDir = opt("out-dir", "/tmp/xwen-spec-equivalence");
mkdirSync(outDir, { recursive: true });

// Shared with scripts/retune-draft.ts: both harnesses have to exercise the same
// two prompt kinds, and the tuning one compares against numbers recorded from
// these exact strings.
const prompts = DRAFT_PROMPTS;

function guardNoModelProcess() {
  const proc = Bun.spawnSync([
    "pgrep",
    "-fl",
    "target/release/(xwen|logits-dump|deps/parity)|llama-(cli|server|bench|eval-callback)",
  ]);
  const hits = proc.stdout.toString().split("\n").filter((l) => l.trim());
  if (hits.length > 0) {
    console.error("ABORT: model process already running:\n" + hits.join("\n"));
    process.exit(1);
  }
}

/// Rebuild before comparing. Without this the script blesses whatever binary
/// happens to be on disk, which after an edit is the PREVIOUS build — an
/// equivalence result for code that is no longer the code.
function buildRelease() {
  console.log("building (release)...");
  const proc = Bun.spawnSync(["cargo", "build", "--release"], {
    cwd: repo,
    stdout: "pipe",
    stderr: "pipe",
  });
  if (proc.exitCode !== 0) {
    console.error("ABORT: cargo build --release failed:\n" + proc.stderr.toString());
    process.exit(1);
  }
  // Belt and braces, since a build can report success without producing a newer
  // binary: verify it is not older than the newest source file.
  const bin = join(repo, "target/release/xwen");
  const binAge = statSync(bin).mtimeMs;
  let newest = 0;
  const walk = (dir: string) => {
    for (const e of readdirSync(dir, { withFileTypes: true })) {
      const full = join(dir, e.name);
      if (e.isDirectory()) walk(full);
      else if (/\.(rs|metal)$/.test(e.name)) newest = Math.max(newest, statSync(full).mtimeMs);
    }
  };
  walk(join(repo, "src"));
  if (newest > binAge) {
    console.error(
      "ABORT: target/release/xwen is older than the newest source file " +
        `(${new Date(binAge).toISOString()} vs ${new Date(newest).toISOString()})`,
    );
    process.exit(1);
  }
}

interface Run {
  /// Generated text: STDOUT only. The stats all go to stderr, so nothing has to
  /// be filtered out of the content — an earlier version stripped lines matching
  /// the stat patterns from both streams, which would silently discard a real
  /// divergence in any generated line that happened to look like one.
  text: string;
  verifyPositions: number | null;
  drafted: number | null;
  paused: number | null;
}

function generate(model: string, prompt: string, extra: string[], file: string): Run {
  const proc = Bun.spawnSync(
    [
      join(repo, "target/release/xwen"),
      "generate",
      "--model-size", model,
      "--prompt", prompt,
      "-n", nTokens,
      "--seed", seed,
      "--stats",
      ...extra,
    ],
    {
      cwd: repo,
      stdin: "ignore",
      // Harness-driven, so the runs stay out of `xwen stats`' default table.
      env: { ...process.env, XWEN_METRICS_TAG: "bench" },
      maxBuffer: 64 * 1024 * 1024,
    },
  );
  const out = proc.stdout.toString();
  const err = proc.stderr.toString();
  writeFileSync(file, `=== stdout\n${out}\n=== stderr\n${err}`);
  if (proc.exitCode !== 0) {
    console.error(`ABORT: ${model} exited ${proc.exitCode}; see ${file}`);
    process.exit(1);
  }
  const num = (re: RegExp) => {
    const m = err.match(re);
    return m ? Number(m[1]) : null;
  };
  return {
    text: out,
    verifyPositions: num(/(\d+) verified positions/),
    drafted: num(/spec:.*?, (\d+) drafted/),
    paused: num(/spec:\s+\d+ rounds \((\d+) paused\)/),
  };
}

buildRelease();
guardNoModelProcess();

let failures = 0;
for (const model of models) {
  for (const mode of modes) {
    const specArgs = [
      ...(mode === "greedy"
        ? ["--temp", "0", "--draft", "official", "--draft-p-min", greedyPMin]
        : [
            "--temp", temp,
            "--draft", "official",
            "--draft-p-min", "0",
            "--draft-pause-margin", "0",
          ]),
      ...penaltyArgs,
    ];
    const plainArgs = [
      ...(mode === "greedy" ? ["--temp", "0", "--no-draft"] : ["--temp", temp, "--no-draft"]),
      ...penaltyArgs,
    ];

    for (const [name, prompt] of Object.entries(prompts)) {
      const tag = `${model}-${mode}-${name}`;
      const plain = generate(model, prompt, plainArgs, join(outDir, `${tag}-plain.txt`));
      const spec = generate(model, prompt, specArgs, join(outDir, `${tag}-draft.txt`));

      // A comparison that never drafted proves nothing about the verify walk.
      if (!spec.verifyPositions || !spec.drafted) {
        console.log(
          `NO COVERAGE ${tag}: the drafter proposed ${spec.drafted ?? 0} tokens over ` +
            `${spec.verifyPositions ?? 0} verified positions — this run exercised no verify ` +
            "or rollback, so its agreement means nothing",
        );
        failures++;
        continue;
      }
      if (mode === "sampled" && spec.paused) {
        console.log(
          `NO COVERAGE ${tag}: ${spec.paused} rounds paused although auto-pause is disabled`,
        );
        failures++;
        continue;
      }

      if (plain.text === spec.text) {
        console.log(
          `OK        ${tag}: identical (${spec.drafted} drafted, ` +
            `${spec.verifyPositions} verified positions)`,
        );
        continue;
      }
      failures++;
      const p = plain.text.split("\n");
      const s = spec.text.split("\n");
      const at = p.findIndex((line, i) => line !== s[i]);
      console.log(`DIVERGED  ${tag}: first difference at line ${at + 1}`);
      console.log(`  plain: ${JSON.stringify(p[at] ?? "")}`);
      console.log(`  draft: ${JSON.stringify(s[at] ?? "")}`);
      if (mode === "sampled" && at <= 0) {
        console.log(
          "  NOTE: diverging on the very first line under a fixed seed points at the sampler " +
            "stream, not a near tie — the spec loop drew a different number of times than plain",
        );
      }
      console.log(`  full outputs under ${outDir}/${tag}-{plain,draft}.txt`);
    }
  }
}

console.log(
  failures === 0
    ? "\nall comparisons identical, and every one of them drafted"
    : `\n${failures} comparison(s) diverged or had no coverage — for a divergence, check ` +
        "whether the fork point is a near tie (expected) or a real bug (not)",
);
process.exit(failures === 0 ? 0 : 1);
