#!/usr/bin/env bun
// Byte-exact validation of the Qwen3 chat fixtures in src/chat.rs against
// llama.cpp's own jinja renderer, via `llama-server`'s /apply-template.
//
// The `expected` strings below are the exact literals asserted in the
// src/chat.rs tests. Those tests pass against `build_prompt`, so a match here
// proves `build_prompt` == the reference renderer for these conversations.
//
// Two templates are checked. The BF16 Qwen3-4B GGUF embeds the BASE (hybrid
// thinking) template, which is what `ChatDialect::Qwen3` renders; the
// Instruct-2507 template ships only in that repo's tokenizer_config.json, so
// its arm re-runs the same server against the vendored copy through
// `--chat-template-file`.
//
// The server runs with `-ngl 0`: template rendering needs no compute, and the
// GPU on this machine is reserved for benchmarks. The script starts and stops
// the server itself.
//
// Two things this endpoint cannot reach, both because llama-server preprocesses
// the conversation before the template sees it:
//
//   1. A conversation ending in an assistant message is an assistant PREFILL:
//      the turn is rendered open, with neither its `<|im_end|>` nor the
//      generation prompt, whatever `add_generation_prompt` says. Such cases
//      carry `prefill: true` and are compared with that fixed tail removed.
//   2. Assistant content holding an inline `<think>` block is parsed by the
//      server's own reasoning parser first, which sets `reasoning_content` to
//      the empty string and passes the content through whole — so the
//      template's own split never runs and the render comes back with an empty
//      block followed by the raw text. The renderer implements the TEMPLATE's
//      split (what `apply_chat_template` does), so that branch is asserted in
//      src/chat.rs against the vendored jinja source instead of here.
//
// Usage: bun scripts/verify_chat_template_qwen3.ts [--model <gguf>] [--port N]

import { spawn } from "bun";
import { readdirSync, existsSync } from "node:fs";
import { join } from "node:path";

const REPO = new URL("..", import.meta.url).pathname.replace(/\/$/, "");

function oracleBin(name: string): string {
  const candidates = [
    process.env.XWEN_LLAMACPP_BIN ? join(process.env.XWEN_LLAMACPP_BIN, name) : null,
    `${REPO}/reference/llama.cpp/build/bin/${name}`,
    join(process.env.HOME ?? "", `develop/private/xwen/reference/llama.cpp/build/bin/${name}`),
  ].filter((p): p is string => p !== null);
  for (const path of candidates) if (existsSync(path)) return path;
  throw new Error(`no built ${name}; run scripts/build-llamacpp.sh or set XWEN_LLAMACPP_BIN`);
}

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

const args = process.argv.slice(2);
const argOf = (flag: string) => {
  const at = args.indexOf(flag);
  return at >= 0 ? args[at + 1] : undefined;
};
const model = argOf("--model") ?? findModel();
const port = Number(argOf("--port") ?? 8087);
const BASE = `http://127.0.0.1:${port}`;
if (!model) {
  console.error("no Qwen3-4B-BF16.gguf in the HF cache; pass --model <path>");
  process.exit(1);
}

/** The generation prompt both templates write, and the turn terminator before
 *  it. A conversation ending in an assistant message is an ASSISTANT PREFILL to
 *  llama-server: it renders the turn open, dropping both of these, whatever
 *  `add_generation_prompt` says. Such a case is marked `prefill` and its
 *  fixture is compared with that tail removed — the tail itself is fixed text
 *  every other case already pins. */
const TURN_END_AND_GENERATION_PROMPT = "<|im_end|>\n<|im_start|>assistant\n";

type Case = {
  name: string;
  messages: unknown[];
  /** Omitted entirely when the template's `is defined` test is the point. */
  enable_thinking?: boolean;
  expected: string;
  /** The conversation ends in an assistant turn; see the constant above. */
  prefill?: boolean;
};

// ---------------------------------------------------------------- base ---
// reference/chat_template-qwen3.jinja, the hybrid-thinking template.
const baseCases: Case[] = [
  {
    // The Z-Image text encoder's prompt, exactly: the diffusion pipeline
    // renders one user turn with add_generation_prompt and thinking ON, and
    // this template writes no <think> opener there.
    name: "(a) single user turn, thinking on — the Z-Image encoder prompt",
    messages: [{ role: "user", content: "a red fox in the snow" }],
    enable_thinking: true,
    expected: "<|im_start|>user\na red fox in the snow<|im_end|>\n<|im_start|>assistant\n",
  },
  {
    name: "(b) single user turn, thinking off",
    messages: [{ role: "user", content: "Hi" }],
    enable_thinking: false,
    expected: "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
  },
  {
    name: "(c) system and user, thinking on",
    messages: [
      { role: "system", content: "You are a pirate." },
      { role: "user", content: "Hi" },
    ],
    enable_thinking: true,
    expected:
      "<|im_start|>system\nYou are a pirate.<|im_end|>\n" +
      "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n",
  },
  {
    name: "(d) an empty system message still writes its block",
    messages: [
      { role: "system", content: "" },
      { role: "user", content: "Hi" },
    ],
    enable_thinking: true,
    expected:
      "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n",
  },
  {
    name: "(e) a superseded assistant turn drops its reasoning",
    messages: [
      { role: "user", content: "Q1" },
      { role: "assistant", content: "A1", reasoning_content: "reasoning one" },
      { role: "user", content: "Q2" },
    ],
    enable_thinking: true,
    expected:
      "<|im_start|>user\nQ1<|im_end|>\n" +
      "<|im_start|>assistant\nA1<|im_end|>\n" +
      "<|im_start|>user\nQ2<|im_end|>\n" +
      "<|im_start|>assistant\n",
  },
  {
    // A wrapped tool result is not a new query, so the assistant turn before it
    // is still the current one and keeps its reasoning. This is the reachable
    // half of the template's `loop.last or reasoning_content` disjunction.
    name: "(f) a turn past the last query keeps its reasoning",
    messages: [
      { role: "user", content: "Q1" },
      { role: "assistant", content: "A1", reasoning_content: "reasoning one" },
      { role: "user", content: "<tool_response>\nsunny\n</tool_response>" },
    ],
    enable_thinking: true,
    expected:
      "<|im_start|>user\nQ1<|im_end|>\n" +
      "<|im_start|>assistant\n<think>\nreasoning one\n</think>\n\nA1<|im_end|>\n" +
      "<|im_start|>user\n<tool_response>\nsunny\n</tool_response><|im_end|>\n" +
      "<|im_start|>assistant\n",
  },
  {
    // The same conversation without reasoning: past the last query, not the
    // last message, nothing to write — so the block is skipped entirely. This
    // is where the Qwen3 template diverges from 3.6, which would write an empty
    // block here.
    name: "(g) an intermediate turn with no reasoning is written bare",
    messages: [
      { role: "user", content: "Q1" },
      { role: "assistant", content: "A1" },
      { role: "user", content: "<tool_response>\nsunny\n</tool_response>" },
    ],
    enable_thinking: true,
    expected:
      "<|im_start|>user\nQ1<|im_end|>\n" +
      "<|im_start|>assistant\nA1<|im_end|>\n" +
      "<|im_start|>user\n<tool_response>\nsunny\n</tool_response><|im_end|>\n" +
      "<|im_start|>assistant\n",
  },
  {
    // The other half of the disjunction: the LAST message writes the block even
    // with nothing to put in it.
    name: "(h) the last message writes the block even when empty",
    messages: [
      { role: "user", content: "Q1" },
      { role: "assistant", content: "A1" },
    ],
    enable_thinking: true,
    prefill: true,
    expected:
      "<|im_start|>user\nQ1<|im_end|>\n" +
      "<|im_start|>assistant\n<think>\n\n</think>\n\nA1<|im_end|>\n" +
      "<|im_start|>assistant\n",
  },
  {
    name: "(i) bodies are interpolated raw, not stripped",
    messages: [{ role: "user", content: "  padded  \n" }],
    enable_thinking: true,
    expected: "<|im_start|>user\n  padded  \n<|im_end|>\n<|im_start|>assistant\n",
  },
  {
    name: "(j) a system message past the first position is an ordinary turn",
    messages: [
      { role: "user", content: "Q1" },
      { role: "system", content: "be terse" },
      { role: "user", content: "Q2" },
    ],
    enable_thinking: true,
    expected:
      "<|im_start|>user\nQ1<|im_end|>\n" +
      "<|im_start|>system\nbe terse<|im_end|>\n" +
      "<|im_start|>user\nQ2<|im_end|>\n" +
      "<|im_start|>assistant\n",
  },
  {
    // A tool result that OPENS the conversation: the Qwen3 template writes the
    // user turn for it, where the 3.6 one leaves it unopened.
    name: "(k) a leading tool result opens its own user turn",
    messages: [
      { role: "tool", content: "sunny" },
      { role: "user", content: "Q" },
    ],
    enable_thinking: true,
    expected:
      "<|im_start|>user\n<tool_response>\nsunny\n</tool_response><|im_end|>\n" +
      "<|im_start|>user\nQ<|im_end|>\n" +
      "<|im_start|>assistant\n",
  },
  {
    // No user query at all: the 3.6 template raises, this one leaves its scan
    // at the last message index and renders.
    name: "(l) a conversation with no user query renders",
    messages: [{ role: "system", content: "be terse" }],
    enable_thinking: true,
    expected: "<|im_start|>system\nbe terse<|im_end|>\n<|im_start|>assistant\n",
  },
];

// ----------------------------------------------------- instruct-2507 ---
// reference/chat_template-qwen3-instruct-2507.jinja: no thinking anywhere.
const instructCases: Case[] = [
  {
    name: "(a) single user turn — the tail is always bare",
    messages: [{ role: "user", content: "Hi" }],
    enable_thinking: true,
    expected: "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n",
  },
  {
    name: "(b) enable_thinking false changes nothing",
    messages: [{ role: "user", content: "Hi" }],
    enable_thinking: false,
    expected: "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n",
  },
  {
    name: "(c) a reasoning field is ignored",
    messages: [
      { role: "system", content: "You are a pirate." },
      { role: "user", content: "Q1" },
      { role: "assistant", content: "A1", reasoning_content: "reasoning one" },
      { role: "user", content: "Q2" },
    ],
    expected:
      "<|im_start|>system\nYou are a pirate.<|im_end|>\n" +
      "<|im_start|>user\nQ1<|im_end|>\n" +
      "<|im_start|>assistant\nA1<|im_end|>\n" +
      "<|im_start|>user\nQ2<|im_end|>\n" +
      "<|im_start|>assistant\n",
  },
  {
    name: "(d) bodies are interpolated raw here too",
    messages: [{ role: "user", content: "  padded  \n" }],
    expected: "<|im_start|>user\n  padded  \n<|im_end|>\n<|im_start|>assistant\n",
  },
];

function hexView(s: string): string {
  return [...s].map((c) => c.codePointAt(0)!.toString(16).padStart(4, "0")).join(" ");
}

function firstDiff(a: string, b: string): number {
  const n = Math.min(a.length, b.length);
  let i = 0;
  for (; i < n && a[i] === b[i]; i++);
  return i;
}

async function render(c: Case): Promise<string> {
  const body: Record<string, unknown> = {
    messages: c.messages,
    add_generation_prompt: true,
  };
  if (c.enable_thinking !== undefined) {
    body.chat_template_kwargs = { enable_thinking: c.enable_thinking };
  }
  const res = await fetch(`${BASE}/apply-template`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`/apply-template ${res.status}: ${await res.text()}`);
  return (await res.json()).prompt;
}

async function waitForServer(timeoutMs = 180_000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const res = await fetch(`${BASE}/health`);
      if (res.ok) return;
    } catch {
      // not up yet
    }
    await new Promise((r) => setTimeout(r, 500));
  }
  throw new Error(`llama-server did not come up on ${BASE}`);
}

/** Run `body` against a server started with `extra` flags, then stop it. */
async function withServer(label: string, extra: string[], body: () => Promise<number>) {
  const cmd = [
    oracleBin("llama-server"),
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

async function runCases(cases: Case[]): Promise<number> {
  let fail = 0;
  for (const c of cases) {
    const got = await render(c);
    let expected = c.expected;
    if (c.prefill) {
      if (!expected.endsWith(TURN_END_AND_GENERATION_PROMPT)) {
        throw new Error(`${c.name}: a prefill fixture must end in the generation prompt`);
      }
      expected = expected.slice(0, -TURN_END_AND_GENERATION_PROMPT.length);
    }
    if (got === expected) {
      console.log(`PASS ${c.name}${c.prefill ? " (context only; assistant prefill)" : ""}`);
      continue;
    }
    fail++;
    console.log(`FAIL ${c.name}`);
    const i = firstDiff(got, expected);
    console.log(`  divergence at index ${i} (got ${got.length}, expected ${expected.length})`);
    console.log(`  expected around: ${JSON.stringify(expected.slice(Math.max(0, i - 40), i + 40))}`);
    console.log(`  got around:      ${JSON.stringify(got.slice(Math.max(0, i - 40), i + 40))}`);
    console.log(`  expected hex: ${hexView(expected.slice(Math.max(0, i - 8), i + 8))}`);
    console.log(`  got hex:      ${hexView(got.slice(Math.max(0, i - 8), i + 8))}`);
    console.log(`  --- full expected ---\n${JSON.stringify(expected)}`);
    console.log(`  --- full got ---\n${JSON.stringify(got)}`);
  }
  console.log(`${cases.length - fail}/${cases.length} byte-exact`);
  return fail;
}

let failures = 0;
failures += await withServer(
  "ChatDialect::Qwen3 (the GGUF's own embedded template)",
  [],
  () => runCases(baseCases),
);
failures += await withServer(
  "ChatDialect::Qwen3Instruct (vendored template via --chat-template-file)",
  ["--chat-template-file", `${REPO}/reference/chat_template-qwen3-instruct-2507.jinja`],
  () => runCases(instructCases),
);

console.log(`\n${failures === 0 ? "ALL BYTE-EXACT" : `${failures} MISMATCH(ES)`}`);
process.exit(failures === 0 ? 0 : 1);
