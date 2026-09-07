#!/usr/bin/env bun
// Regenerates tests/fixtures/qwen3-prompts.json: twenty prompts with the token
// ids the ORACLE assigns them, which is what src/tokenizer.rs asserts xwen's
// own encoder reproduces.
//
// The ids come from llama.cpp's `llama-tokenize` against the BF16 Qwen3-4B
// GGUF, run once per prompt over stdin so no shell quoting touches the text:
//
//   llama-tokenize -m <gguf> --ids --no-bos --stdin
//
// Special-token parsing is ON (llama-tokenize's default; `--no-parse-special`
// would turn it off), which is the mode `LagunaTokenizer::encode` runs in: a
// literal `<|im_start|>` in the text maps to its single added-vocabulary id.
// The prompts here are drawn from the repo's own fixtures, so the corpus a
// tokenizer test runs on is the corpus the parity and bench harnesses use.
//
// Vocabulary-only work: llama-tokenize never builds a compute graph, so this
// runs on the CPU and never touches the GPU.
//
// Usage: bun scripts/qwen3-fixtures.ts [--model <gguf>] [--out <json>]

import { spawnSync } from "bun";
import { readFileSync, readdirSync, writeFileSync, existsSync } from "node:fs";
import { join } from "node:path";

const REPO = new URL("..", import.meta.url).pathname.replace(/\/$/, "");

/** The built oracle. `reference/llama.cpp` is a submodule, so a worktree that
 *  has not initialized it borrows the main checkout's build; XWEN_LLAMACPP_BIN
 *  overrides both. */
function oracleBin(name: string): string {
  const candidates = [
    process.env.XWEN_LLAMACPP_BIN ? join(process.env.XWEN_LLAMACPP_BIN, name) : null,
    `${REPO}/reference/llama.cpp/build/bin/${name}`,
    join(process.env.HOME ?? "", `develop/private/xwen/reference/llama.cpp/build/bin/${name}`),
  ].filter((p): p is string => p !== null);
  for (const path of candidates) if (existsSync(path)) return path;
  throw new Error(
    `no built ${name}; run scripts/build-llamacpp.sh or set XWEN_LLAMACPP_BIN to a build's bin/`,
  );
}

const ORACLE = oracleBin("llama-tokenize");

/** The one true BF16 Qwen3-4B GGUF in the HF cache, or null when absent. */
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
const out = argOf("--out") ?? `${REPO}/tests/fixtures/qwen3-prompts.json`;
if (!model) {
  console.error(
    "no Qwen3-4B-BF16.gguf in the HF cache; pass --model <path> or fetch unsloth/Qwen3-4B-GGUF",
  );
  process.exit(1);
}

const read = (path: string) => readFileSync(`${REPO}/${path}`, "utf8");
const parity = JSON.parse(read("tests/fixtures/parity-prompts.json"));
const parityText = (id: string) => {
  const found = parity.prompts.find((p: { id: string }) => p.id === id);
  if (!found) throw new Error(`parity-prompts.json has no prompt ${id}`);
  return found.text as string;
};
const corpus = read("tests/fixtures/ppl-corpus.txt");
const decode630 = read("tests/fixtures/bench-prompts/decode-630.txt");
const prefill4k = read("tests/fixtures/bench-prompts/prefill-4k.txt");

/** Character slice, taken on a whitespace boundary so a slice never cuts a word. */
function slice(text: string, from: number, to: number): string {
  let start = from;
  while (start > 0 && !/\s/.test(text[start - 1]!)) start -= 1;
  let end = Math.min(to, text.length);
  while (end < text.length && !/\s/.test(text[end]!)) end += 1;
  return text.slice(start, end).trim();
}

type Prompt = { id: string; source: string; text: string };

const prompts: Prompt[] = [
  // Three from the parity corpus, so the tokenizer test and the logits parity
  // harness disagree about nothing.
  { id: "parity-code-short", source: "tests/fixtures/parity-prompts.json#code-short", text: parityText("code-short") },
  { id: "parity-text-mixed", source: "tests/fixtures/parity-prompts.json#text-mixed", text: parityText("text-mixed") },
  { id: "parity-long-mixed", source: "tests/fixtures/parity-prompts.json#long-mixed", text: parityText("long-mixed") },

  // Prose slices of the perplexity corpus, at four lengths.
  { id: "corpus-head", source: "tests/fixtures/ppl-corpus.txt[0..400]", text: slice(corpus, 0, 400) },
  { id: "corpus-middle", source: "tests/fixtures/ppl-corpus.txt[6000..6900]", text: slice(corpus, 6000, 6900) },
  { id: "corpus-tail", source: "tests/fixtures/ppl-corpus.txt[16000..17600]", text: slice(corpus, 16000, 17600) },
  { id: "bench-decode-630", source: "tests/fixtures/bench-prompts/decode-630.txt", text: decode630.trim() },

  // The one prompt past a thousand tokens: the whole 4k prefill fixture, which
  // is what exercises a tokenizer's long-input path.
  { id: "bench-prefill-4k", source: "tests/fixtures/bench-prompts/prefill-4k.txt", text: prefill4k.trim() },

  // Edge cases. Each isolates one thing a byte-level BPE tokenizer can get
  // wrong, and each is small enough that a mismatch names its own cause.
  { id: "edge-empty-ish", source: "hand-written: a single space", text: " " },
  { id: "edge-ascii-punctuation", source: "hand-written: dense ASCII punctuation", text: "!@#$%^&*()_+-=[]{}|;':\",./<>?`~\\" },
  { id: "edge-leading-whitespace", source: "hand-written: leading and repeated spaces", text: "   leading spaces and   doubled   ones, then a tab\tand a return\r\n" },
  { id: "edge-newline-runs", source: "hand-written: runs of newlines", text: "one\n\ntwo\n\n\nthree\n\n\n\nfour\n" },
  { id: "edge-cjk", source: "hand-written: CJK", text: "通义千问是阿里云研发的大规模语言模型。今天天气很好，我们去公园散步吧。" },
  { id: "edge-emoji", source: "hand-written: emoji with zero-width joiners", text: "🙂 family: 👨‍👩‍👧‍👦, flag: 🇸🇪, skin tone: 👍🏽, accented: éàü" },
  { id: "edge-rtl-and-diacritics", source: "hand-written: Arabic, Hebrew, Nordic", text: "مرحبا بالعالم — שלום עולם — Ærø, Ångström, œuvre, ß, Ǆ" },
  { id: "edge-marker-text", source: "hand-written: chat markers as literal text", text: "The template writes <|im_start|>assistant and closes with <|im_end|>; a <think> block ends at </think>." },
  { id: "edge-json", source: "hand-written: compact JSON", text: '{"name":"laguna","count":3,"nested":{"a":[1,2.5,-7,null,true]},"unicode":"café ☕"}' },
  { id: "edge-code-rust", source: "hand-written: Rust with generics and lifetimes", text: "pub fn encode<'a, T: AsRef<str> + ?Sized>(&'a self, text: &'a T) -> Result<Vec<u32>> {\n    let ids = self.inner.encode(text.as_ref(), false)?;\n    Ok(ids.get_ids().to_vec())\n}" },
  { id: "edge-numbers", source: "hand-written: numeric formats", text: "0 1 42 -17 3.14159 6.022e23 0xFF 0b1010 1_000_000 1/3 99.9% $1,234.56 2026-09-06T12:34:56Z" },
  { id: "edge-mixed-script-code", source: "hand-written: comments in several scripts", text: "# 计算斐波那契数列\ndef fib(n):  # النمو الأسي\n    return n if n < 2 else fib(n-1) + fib(n-2)\n# Тест: русский текст, ελληνικά, 한국어\n" },
];

if (prompts.length !== 20) {
  console.error(`expected 20 prompts, built ${prompts.length}`);
  process.exit(1);
}
const ids = new Set(prompts.map((p) => p.id));
if (ids.size !== prompts.length) {
  console.error("prompt ids are not unique");
  process.exit(1);
}
// Every prompt must already be in NFC. Both tokenizer.json files declare an
// NFC normalizer, which the HF `tokenizers` runtime applies and llama.cpp's
// GGUF tokenizer does not implement — so a decomposed prompt would pin THAT
// divergence rather than the vocabulary, and would not survive
// `decode(encode(text)) == text` either. The divergence itself is pinned by
// `tokenizer::tests::the_hf_pipeline_normalizes_where_llama_cpp_does_not`.
for (const p of prompts) {
  if (p.text.normalize("NFC") !== p.text) {
    console.error(`prompt ${p.id} is not in NFC; normalize it at the source`);
    process.exit(1);
  }
}

/** One `llama-tokenize` run, prompt on stdin. */
function tokenize(text: string): number[] {
  const run = spawnSync({
    cmd: [ORACLE, "-m", model!, "--ids", "--no-bos", "--stdin"],
    stdin: Buffer.from(text, "utf8"),
    stdout: "pipe",
    stderr: "pipe",
  });
  if (run.exitCode !== 0) {
    throw new Error(`llama-tokenize failed: ${new TextDecoder().decode(run.stderr)}`);
  }
  const stdout = new TextDecoder().decode(run.stdout);
  // The `--ids` form prints a single Python-style list; the loader banner goes
  // to stderr, but take the last bracketed run regardless.
  const match = stdout.match(/\[[^\]]*\]\s*$/);
  if (!match) throw new Error(`no id list in llama-tokenize output: ${stdout.slice(-400)}`);
  return JSON.parse(match[0]);
}

const entries = prompts.map((p) => {
  const tokenIds = tokenize(p.text);
  console.error(`${p.id.padEnd(26)} ${String(tokenIds.length).padStart(5)} tokens`);
  return { id: p.id, source: p.source, text: p.text, ids: tokenIds };
});

const longest = Math.max(...entries.map((e) => e.ids.length));
if (longest <= 1000) {
  console.error(`no prompt exceeds 1000 tokens (longest ${longest})`);
  process.exit(1);
}

writeFileSync(
  out,
  JSON.stringify(
    {
      note:
        "Tokenizer round-trip fixtures for the dense Qwen3-4B vocabulary (base, " +
        "Instruct-2507 and the Z-Image text encoder ship a sha256-identical " +
        "tokenizer.json). `ids` came from the oracle, one run per prompt: " +
        "`llama-tokenize -m Qwen3-4B-BF16.gguf --ids --no-bos --stdin`. " +
        "Special-token parsing is on, which is the mode LagunaTokenizer::encode " +
        "runs in. The Qwen vocabulary has no BOS: nothing is prepended. " +
        "Regenerate with `bun scripts/qwen3-fixtures.ts`.",
      add_bos: false,
      vocab: 151936,
      oracle: "llama-tokenize -m Qwen3-4B-BF16.gguf --ids --no-bos --stdin",
      prompts: entries,
    },
    null,
    2,
  ) + "\n",
);
console.error(`\nwrote ${out}: ${entries.length} prompts, longest ${longest} tokens`);
