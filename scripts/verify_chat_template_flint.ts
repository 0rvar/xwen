#!/usr/bin/env bun
// Byte-exact validation of tests/fixtures/flint-chat.json against llama.cpp's
// own jinja renderer, via `llama-server`'s /apply-template.
//
// src/chat.rs asserts `build_prompt` under `ChatDialect::Flint` reproduces each
// fixture's `expected`; a match here proves those literals are what the
// reference renderer writes for the same conversation. Two arms: the template
// embedded in the Flint GGUF, and the vendored copy
// (reference/flint/chat_template.jinja) through `--chat-template-file`.
//
// Tool-call arguments go over the wire as the JSON STRING the OpenAI shape
// carries; llama-server parses them into the object the template iterates.
//
// Every fixture tool carries a `description`: llama-server fills in an empty
// one for a tool that has none before the template sees it, which HF's
// `apply_chat_template` does not, so a description-less schema would compare
// the server's normalization rather than the template.
//
// A fixture marked `prefill` ends in an assistant turn, which llama-server
// renders as an open assistant PREFILL without the generation prompt; its
// expected string is compared with the trailing `<|assistant|>` cut off.
//
// A fixture marked `template_only` holds a branch of the template llama-server
// never renders (it says why: the server rewrites a `developer` message to
// `system` before the template runs). It is reported as SKIP with the reason
// and graded by the Rust tests alone.
//
// The oracle has to know Flint, which the main reference/llama.cpp pin predates,
// so it is the Flint checkout reference/llama.cpp-flint (a37cc4610), or the bin/
// XWEN_LLAMACPP_BIN names. The server
// runs with `-ngl 0` (rendering needs no compute) and is started and stopped
// here.
//
// Usage: bun scripts/verify_chat_template_flint.ts [--model <gguf>] [--port N]

import { spawn } from "bun";
import { readdirSync, existsSync, readFileSync } from "node:fs";
import { join } from "node:path";

const REPO = new URL("..", import.meta.url).pathname.replace(/\/$/, "");
const BIN = process.env.XWEN_LLAMACPP_BIN ?? `${REPO}/reference/llama.cpp-flint/build/bin`;
if (!existsSync(join(BIN, "llama-server"))) {
  console.error(`no Flint-capable llama.cpp at ${BIN}: run scripts/build-llamacpp.sh reference/llama.cpp-flint, or set XWEN_LLAMACPP_BIN`);
  process.exit(1);
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

const args = process.argv.slice(2);
const argOf = (flag: string) => {
  const at = args.indexOf(flag);
  return at >= 0 ? args[at + 1] : undefined;
};
const model = argOf("--model") ?? findModel();
const port = Number(argOf("--port") ?? 8089);
const BASE = `http://127.0.0.1:${port}`;
if (!model) {
  console.error("no Flint-Q4_K_M.gguf in the HF cache; pass --model <path>");
  process.exit(1);
}

type Case = {
  name: string;
  messages: Record<string, unknown>[];
  tools?: unknown[];
  expected: string;
  prefill?: boolean;
  template_only?: string;
};
const cases: Case[] = JSON.parse(readFileSync(`${REPO}/tests/fixtures/flint-chat.json`, "utf8")).cases;
const GENERATION_PROMPT = "<|assistant|>";

/** The fixture's messages in the OpenAI wire shape: call arguments as a string. */
function wire(messages: Record<string, unknown>[]): unknown[] {
  return messages.map((m) => {
    const calls = m.tool_calls as { type: string; function: { name: string; arguments: unknown } }[] | undefined;
    if (!calls) return m;
    return {
      ...m,
      tool_calls: calls.map((c, i) => ({
        id: `call_${i}`,
        type: c.type,
        function: { name: c.function.name, arguments: JSON.stringify(c.function.arguments) },
      })),
    };
  });
}

async function render(c: Case): Promise<string> {
  const body: Record<string, unknown> = { messages: wire(c.messages), add_generation_prompt: true };
  if (c.tools) body.tools = c.tools;
  const res = await fetch(`${BASE}/apply-template`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`/apply-template ${res.status}: ${await res.text()}`);
  return (await res.json()).prompt;
}

async function waitForServer(timeoutMs = 300_000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      if ((await fetch(`${BASE}/health`)).ok) return;
    } catch {
      // not up yet
    }
    await new Promise((r) => setTimeout(r, 500));
  }
  throw new Error(`llama-server did not come up on ${BASE}`);
}

async function withServer(label: string, extra: string[], body: () => Promise<number>) {
  const cmd = [
    join(BIN!, "llama-server"),
    "-m", model!,
    "--jinja",
    "-ngl", "0",
    "--no-warmup",
    "-c", "512",
    "--port", String(port),
    "--host", "127.0.0.1",
    ...extra,
  ];
  console.log(`\n=== ${label} ===\n$ ${cmd.join(" ")}`);
  const proc = spawn({ cmd, stdout: "ignore", stderr: "ignore" });
  try {
    await waitForServer();
    return await body();
  } finally {
    proc.kill();
    await proc.exited;
  }
}

function firstDiff(a: string, b: string): number {
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) if (a[i] !== b[i]) return i;
  return n;
}

async function runCases(): Promise<number> {
  let fail = 0;
  let skipped = 0;
  for (const c of cases) {
    if (c.template_only) {
      console.log(`SKIP ${c.name}\n  ${c.template_only}`);
      skipped++;
      continue;
    }
    const got = await render(c);
    let expected = c.expected;
    if (c.prefill) {
      if (!expected.endsWith(GENERATION_PROMPT)) throw new Error(`${c.name}: a prefill fixture must end in the generation prompt`);
      expected = expected.slice(0, -GENERATION_PROMPT.length);
    }
    if (got === expected) {
      console.log(`PASS ${c.name}${c.prefill ? " (context only: assistant prefill)" : ""}`);
      continue;
    }
    fail++;
    const i = firstDiff(got, expected);
    console.log(`FAIL ${c.name}\n  divergence at ${i} (got ${got.length}, expected ${expected.length})`);
    console.log(`  expected around: ${JSON.stringify(expected.slice(Math.max(0, i - 40), i + 40))}`);
    console.log(`  got around:      ${JSON.stringify(got.slice(Math.max(0, i - 40), i + 40))}`);
  }
  console.log(`${cases.length - skipped - fail}/${cases.length - skipped} byte-exact, ${skipped} skipped`);
  return fail;
}

let failures = 0;
failures += await withServer("the GGUF's embedded template", [], runCases);
failures += await withServer(
  "the vendored template via --chat-template-file",
  ["--chat-template-file", `${REPO}/reference/flint/chat_template.jinja`],
  runCases,
);
console.log(`\n${failures === 0 ? "ALL BYTE-EXACT" : `${failures} MISMATCH(ES)`}`);
process.exit(failures === 0 ? 0 : 1);
