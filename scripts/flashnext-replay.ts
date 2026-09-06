// Flash-Next correctness check: forced replay against the pinned llama.cpp oracle.
//
// The four-tier parity harness (parity-gate.ts) cannot run on Qwen3.8-Flash-Next
// (ReferenceExperts panics on its 512-expert geometry; docs/parity.md
// "Limitations"), so this checkpoint is graded the decode-tier way with
// llama.cpp standing in for the reference runner: the oracle free-runs greedy
// N steps on each committed fixture, the candidate is teacher-forced along that
// exact trajectory, and its own argmax at every step is compared with the
// oracle's. Method and first results: docs/qwen4exp-parity-2026-08-29.md.
//
// Grading, per fixture (N = 64 steps):
//   - a step AGREES when the candidate's top-1 is the oracle's token;
//   - otherwise it is EXCUSED as a near-tie when EITHER
//       (a) the oracle's own margin over the candidate's pick is below the band
//           (1.0 logit at q8 attention decode, 0.5 otherwise; the decode tier's
//           NEAR_TIE_MARGIN), or
//       (b) `--control` was given and the control arm (the same binary with the
//           change under test switched OFF) holds the oracle's token over the
//           candidate's pick by less than the band. The pre-change engine did
//           not confidently make that decision either, so a flip there is
//           reassociation crossing a boundary, not new math. The 2026-09-05
//           PLE device tail is the recorded case: reversing the summation order
//           of one host f32 dot product reproduced the identical "hard"
//           mismatch (docs/log.md 2026-09-05, gate and conv);
//   - anything else is a HARD mismatch and fails the fixture;
//   - more than 8 excused steps fails the fixture (a flood of near-ties is drift);
//   - a non-finite logit anywhere fails the fixture.
//
// Usage:
//   bun scripts/flashnext-replay.ts [--control XWEN_PLE_TAIL_CLASSIC=1[,K=V...]]
//                                   [--model <gguf>] [--dir /tmp/xwen-flashnext-replay]
//                                   [--rebuild-oracle] [--port 18099] [--steps 64]
//
// The oracle dumps are cached under --dir and reused while the llama.cpp pin,
// the fixture tokens and the step count match; --rebuild-oracle forces a fresh
// run. Inherited XWEN_* variables are stripped from every arm so a stray switch
// in the caller's shell cannot grade a path the report does not name; the
// candidate arm is the binary's defaults, the control arm adds --control.
// Needs the oracle built once (`bash scripts/build-llamacpp.sh`) and exclusive
// use of the GPU: no bench, no serve, no test suite alongside.

import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readdirSync } from "node:fs";
import { join } from "node:path";

const root = new URL("..", import.meta.url).pathname.replace(/\/$/, "");
const args = process.argv.slice(2);
const flag = (name: string) => args.includes(`--${name}`);
const opt = (name: string, dflt: string) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 && args[i + 1] ? args[i + 1] : dflt;
};
const fail = (msg: string): never => {
  console.error(`flashnext-replay: ${msg}`);
  process.exit(2);
};

const dir = opt("dir", "/tmp/xwen-flashnext-replay");
const steps = Number(opt("steps", "64"));
const port = Number(opt("port", "18099"));
const controlSpec = opt("control", "");
const control: Record<string, string> = {};
for (const kv of controlSpec ? controlSpec.split(",") : []) {
  const eq = kv.indexOf("=");
  if (eq <= 0) fail(`--control expects KEY=VALUE, got "${kv}"`);
  control[kv.slice(0, eq)] = kv.slice(eq + 1);
}
const bin = join(root, "target/release/logits-dump");
const oracleBin = join(root, "reference/llama.cpp/build/bin/llama-server");
if (!existsSync(bin)) fail(`${bin} missing; cargo build --release --bin logits-dump`);
if (!existsSync(oracleBin)) fail(`${oracleBin} missing; bash scripts/build-llamacpp.sh`);

function flashNextModel(): string {
  const hub =
    process.env.HF_HUB_CACHE ??
    (process.env.HF_HOME ? join(process.env.HF_HOME, "hub") : join(process.env.HOME!, ".cache/huggingface/hub"));
  const snaps = join(hub, "models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots");
  if (!existsSync(snaps)) fail(`Flash-Next is not in the HF cache (${snaps}); run xwen fetch --model-size flash-next`);
  for (const snap of readdirSync(snaps)) {
    const d = join(snaps, snap, "UD-Q4_K_XL");
    if (!existsSync(d)) continue;
    const shard = readdirSync(d).find((f) => f.endsWith("00001-of-00004.gguf"));
    if (shard) return join(d, shard);
  }
  return fail("no UD-Q4_K_XL first shard under the Flash-Next snapshot");
}
const model = opt("model", "") || flashNextModel();

// Every arm runs on a clean environment: no inherited kernel switch, no profiler.
const baseEnv: Record<string, string> = {};
for (const [k, v] of Object.entries(process.env)) {
  if (v !== undefined && !k.startsWith("XWEN_")) baseEnv[k] = v;
}
// Set after the XWEN_ strip, which is what every other var here is subject to.
// logits-dump writes no metrics record today, so this changes nothing now; it
// is here so that a replay run can never reach the history as real use.
baseEnv.XWEN_METRICS_TAG = "bench";

async function run(cmd: string[], env = baseEnv): Promise<string> {
  const p = Bun.spawn(cmd, { cwd: root, env, stdout: "pipe", stderr: "pipe" });
  const [out, err, code] = await Promise.all([new Response(p.stdout).text(), new Response(p.stderr).text(), p.exited]);
  if (code !== 0) fail(`${cmd.join(" ")} exited ${code}\n${err.slice(-2000)}`);
  return (out + err).trim();
}
const sha256 = async (path: string) =>
  createHash("sha256").update(new Uint8Array(await Bun.file(path).arrayBuffer())).digest("hex");

const pin = await run(["git", "-C", join(root, "reference/llama.cpp"), "rev-parse", "HEAD"]);
const oracleVersion = await run([oracleBin, "--version"]);
if (!oracleVersion.includes(pin.slice(0, 7))) fail(`llama-server version "${oracleVersion}" is not the pinned ${pin.slice(0, 7)}; rebuild the oracle`);
const fixturesPath = join(root, "tests/fixtures/parity-prompts.json");
const fixtures: { id: string; tokens: number[] }[] = (await Bun.file(fixturesPath).json()).prompts;
mkdirSync(dir, { recursive: true });

// ------------------------------------------------------------------ oracle

type Step = { token: number; top1: [number, number]; top2: [number, number]; top5: [number, number][] };
type OracleDump = { kind: "greedy"; model: string; tokens: number[]; steps: Step[]; oracleProvenance: Record<string, unknown> };

async function cachedOracle(id: string, tokens: number[]): Promise<OracleDump | null> {
  const path = join(dir, `${id}.oracle.json`);
  if (flag("rebuild-oracle") || !existsSync(path)) return null;
  try {
    const j = (await Bun.file(path).json()) as OracleDump;
    const ok =
      j.kind === "greedy" &&
      j.steps?.length === steps &&
      JSON.stringify(j.tokens) === JSON.stringify(tokens) &&
      j.oracleProvenance?.oraclePin === pin;
    return ok ? j : null;
  } catch {
    return null;
  }
}

async function buildOracles(missing: { id: string; tokens: number[] }[]) {
  try {
    await fetch(`http://127.0.0.1:${port}/health`, { signal: AbortSignal.timeout(500) });
    fail(`port ${port} already answers; another server is up`);
  } catch (e) {
    if (String(e).includes("already answers")) throw e;
  }
  const server = Bun.spawn([oracleBin, "-m", model, "-ngl", "999", "-c", "4096", "--parallel", "1", "--host", "127.0.0.1", "--port", String(port)], {
    cwd: root,
    env: baseEnv,
    stdout: Bun.file(join(dir, "oracle.stdout")),
    stderr: Bun.file(join(dir, "oracle.stderr")),
  });
  const stop = async () => {
    server.kill("SIGTERM");
    await server.exited;
  };
  process.on("SIGINT", () => void stop().then(() => process.exit(130)));
  try {
    let ready = false;
    for (let i = 0; i < 300 && !ready; i++) {
      if (server.exitCode !== null) fail("llama-server exited during startup; see oracle.stderr");
      try {
        ready = (await fetch(`http://127.0.0.1:${port}/health`, { signal: AbortSignal.timeout(1000) })).ok;
      } catch {}
      if (!ready) await Bun.sleep(1000);
    }
    if (!ready) fail("llama-server did not become healthy in 300 s");
    for (const f of missing) {
      const tokens = [...f.tokens];
      const out: Step[] = [];
      for (let i = 0; i < steps; i++) {
        const body = {
          prompt: [...tokens],
          n_predict: 1,
          temperature: 0,
          top_k: 1,
          seed: 0,
          cache_prompt: i > 0,
          return_tokens: true,
          n_probs: 20,
          post_sampling_probs: false,
          ignore_eos: false,
          repeat_penalty: 1,
        };
        const r = await fetch(`http://127.0.0.1:${port}/completion`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
          signal: AbortSignal.timeout(300_000),
        });
        if (!r.ok) fail(`oracle ${f.id} step ${i}: ${await r.text()}`);
        const probs = (await r.json()).completion_probabilities;
        if (probs?.length !== 1) fail(`oracle ${f.id} step ${i}: expected one probability row`);
        const top: [number, number][] = probs[0].top_logprobs.map((p: any) => [p.id, p.logprob]);
        top.sort((a, b) => b[1] - a[1]);
        if (top.length < 5 || !top.every((p) => Number.isInteger(p[0]) && Number.isFinite(p[1]))) fail(`oracle ${f.id} step ${i}: bad probabilities`);
        out.push({ token: top[0][0], top1: top[0], top2: top[1], top5: top.slice(0, 5) });
        tokens.push(top[0][0]);
      }
      const dump: OracleDump = {
        kind: "greedy",
        model,
        tokens: f.tokens,
        steps: out,
        oracleProvenance: { oraclePin: pin, oracleVersion, oracleSha256: await sha256(oracleBin), date: new Date().toISOString(), steps },
      };
      await Bun.write(join(dir, `${f.id}.oracle.json`), JSON.stringify(dump));
      console.log(JSON.stringify({ fixture: f.id, oracle: "built", steps }));
    }
  } finally {
    await stop();
  }
}

const missing: { id: string; tokens: number[] }[] = [];
for (const f of fixtures) if (!(await cachedOracle(f.id, f.tokens))) missing.push(f);
if (missing.length) await buildOracles(missing);
else console.log(JSON.stringify({ oracle: "reused", pin: pin.slice(0, 7), fixtures: fixtures.map((f) => f.id) }));

// ---------------------------------------------------------------- replays

type ReplayStep = { forced_token: number; top1: [number, number]; top5: [number, number][]; l2: number; nonfinite: number };
type ReplayDump = { kind: "replay"; tokens: number[]; steps: ReplayStep[]; provenance: Record<string, unknown> };

async function replay(id: string, arm: string, env: Record<string, string>): Promise<ReplayDump> {
  const out = join(dir, `${id}.${arm}.json`);
  await run([bin, "--model", model, "--moe-impl", "fused", "--replay", join(dir, `${id}.oracle.json`), "--output", out], env);
  const d = (await Bun.file(out).json()) as ReplayDump;
  if (d.kind !== "replay" || d.steps.length !== steps) fail(`${id}/${arm}: bad replay dump`);
  return d;
}

const candidateEnv = { ...baseEnv };
const controlEnv = { ...baseEnv, ...control };
const hasControl = Object.keys(control).length > 0;
const binarySha = await sha256(bin);
const grades: any[] = [];

for (const f of fixtures) {
  const oracle = (await cachedOracle(f.id, f.tokens))!;
  const cand = await replay(f.id, "candidate", candidateEnv);
  const ctrl = hasControl ? await replay(f.id, "control", controlEnv) : null;
  if (JSON.stringify(cand.tokens) !== JSON.stringify(oracle.tokens)) fail(`${f.id}: candidate prompt differs from the oracle's`);
  const band = cand.provenance.attn_decode === "q8" ? 1.0 : 0.5;
  let agree = 0;
  const excused: any[] = [];
  const hard: any[] = [];
  let winnerChangesVsControl = 0;
  let maxCommonLogitDeltaVsControl = 0;
  for (let i = 0; i < steps; i++) {
    const o = oracle.steps[i];
    const c = cand.steps[i];
    if (c.forced_token !== o.token) fail(`${f.id} step ${i}: forced trajectory diverged`);
    if (c.nonfinite !== 0 || !Number.isFinite(c.l2) || !c.top5.every((x) => Number.isFinite(x[1]))) hard.push({ step: i, reason: "non-finite" });
    if (ctrl) {
      const k = ctrl.steps[i];
      if (k.top1[0] !== c.top1[0]) winnerChangesVsControl++;
      for (const [tok, v] of c.top5) {
        const kv = k.top5.find((x) => x[0] === tok);
        if (kv) maxCommonLogitDeltaVsControl = Math.max(maxCommonLogitDeltaVsControl, Math.abs(v - kv[1]));
      }
    }
    if (c.top1[0] === o.token) {
      agree++;
      continue;
    }
    const pick = c.top1[0];
    const inOracle = o.top5.find((x) => x[0] === pick);
    const oracleGap = inOracle ? o.top1[1] - inOracle[1] : null;
    let controlGap: number | null = null;
    if (ctrl) {
      const k = ctrl.steps[i];
      const kOracle = k.top5.find((x) => x[0] === o.token);
      const kPick = k.top5.find((x) => x[0] === pick);
      if (kOracle && kPick) controlGap = kOracle[1] - kPick[1];
    }
    const row = { step: i, oracle: o.token, candidate: pick, oracleGap, controlGap };
    if (oracleGap !== null && oracleGap < band) excused.push({ ...row, by: "oracle near-tie" });
    else if (controlGap !== null && controlGap < band) excused.push({ ...row, by: "engine near-tie" });
    else hard.push({ ...row, reason: "hard mismatch" });
  }
  if (excused.length > 8) hard.push({ reason: "near-tie cap exceeded", excused: excused.length });
  const grade = {
    fixture: f.id,
    promptTokens: f.tokens.length,
    steps,
    band,
    agree,
    excused,
    hard,
    passed: hard.length === 0,
    vsControl: ctrl ? { winnerChanges: winnerChangesVsControl, maxCommonLogitDelta: +maxCommonLogitDeltaVsControl.toFixed(4) } : null,
    candidateProvenance: cand.provenance,
  };
  grades.push(grade);
  console.log(
    JSON.stringify({
      fixture: f.id,
      agree: `${agree}/${steps}`,
      excused: excused.map((e) => `${e.step}:${e.by === "oracle near-tie" ? "o" : "e"}${(e.by === "oracle near-tie" ? e.oracleGap : e.controlGap).toFixed(3)}`),
      hard: hard.map((h) => h.step ?? h.reason),
      passed: grade.passed,
    }),
  );
}

const report = {
  date: new Date().toISOString(),
  repoHead: await run(["git", "rev-parse", "HEAD"]),
  repoDirty: (await run(["git", "status", "--porcelain"])) !== "",
  model,
  oraclePin: pin,
  candidate: { binary: bin, sha256: binarySha, env: {} },
  control: hasControl ? control : null,
  grades,
};
await Bun.write(join(dir, "grades.json"), JSON.stringify(report, null, 2));
const passed = grades.every((g) => g.passed);
console.log(`${passed ? "PASS" : "FAIL"}: ${grades.map((g) => `${g.fixture} ${g.agree}/${steps} (${g.excused.length} excused, ${g.hard.length} hard)`).join("; ")} -> ${join(dir, "grades.json")}`);
process.exit(passed ? 0 : 1);
