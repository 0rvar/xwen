#!/usr/bin/env bun
// Regenerates tests/fixtures/flint-prompts.json: prompts with the token ids the
// ORACLE assigns them, which src/tokenizer.rs asserts the embedded Flint
// tokenizer reproduces.
//
// The ids come from llama.cpp's `llama-tokenize` against the Flint GGUF, one
// run per prompt over stdin so no shell quoting touches the text:
//
//   llama-tokenize -m Flint-Q4_K_M.gguf --ids --no-bos --stdin
//
// Special-token parsing is on (llama-tokenize's default), the mode
// `LagunaTokenizer::encode` runs in. Vocabulary-only work: no compute graph is
// built and the GPU is not touched.
//
// The oracle has to know the `flint` pre-tokenizer, which the main
// reference/llama.cpp pin predates, so it is the Flint checkout,
// reference/llama.cpp-flint at a37cc4610 ("models : add Flint model"), built with
// `scripts/build-llamacpp.sh reference/llama.cpp-flint`. XWEN_LLAMACPP_BIN
// overrides the bin/ directory. Without a build the script refuses rather than
// falling back to one that cannot tokenize Flint.
//
// Usage: bun scripts/flint-fixtures.ts [--model <gguf>] [--out <json>]

import { spawnSync } from "bun";
import { readFileSync, readdirSync, writeFileSync, existsSync } from "node:fs";
import { join } from "node:path";

const REPO = new URL("..", import.meta.url).pathname.replace(/\/$/, "");

const BIN = process.env.XWEN_LLAMACPP_BIN ?? `${REPO}/reference/llama.cpp-flint/build/bin`;
if (!existsSync(join(BIN, "llama-tokenize"))) {
  console.error(
    `no Flint-capable llama.cpp at ${BIN}: run scripts/build-llamacpp.sh reference/llama.cpp-flint, or set XWEN_LLAMACPP_BIN`,
  );
  process.exit(1);
}
const ORACLE = join(BIN, "llama-tokenize");

/** The Flint Q4_K_M GGUF in the HF cache, or null when absent. */
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
const out = argOf("--out") ?? `${REPO}/tests/fixtures/flint-prompts.json`;
if (!model) {
  console.error("no Flint-Q4_K_M.gguf in the HF cache; pass --model <path> or fetch ggml-org/Flint-GGUF");
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
  { id: "parity-code-short", source: "tests/fixtures/parity-prompts.json#code-short", text: parityText("code-short") },
  { id: "parity-text-mixed", source: "tests/fixtures/parity-prompts.json#text-mixed", text: parityText("text-mixed") },
  { id: "parity-long-mixed", source: "tests/fixtures/parity-prompts.json#long-mixed", text: parityText("long-mixed") },
  { id: "corpus-head", source: "tests/fixtures/ppl-corpus.txt[0..400]", text: slice(corpus, 0, 400) },
  { id: "corpus-tail", source: "tests/fixtures/ppl-corpus.txt[16000..17600]", text: slice(corpus, 16000, 17600) },
  { id: "bench-prefill-4k", source: "tests/fixtures/bench-prompts/prefill-4k.txt", text: prefill4k.trim() },

  // Each edge case isolates one stage of Flint's pre-tokenizer sequence:
  // digit runs split three at a time, CJK and kana runs split out whole, the
  // o200k-style word/contraction/punctuation regex, then byte-level BPE.
  { id: "edge-single-space", source: "hand-written: a single space", text: " " },
  { id: "edge-digit-runs", source: "hand-written: digit runs", text: "1 12 123 1234 12345 1234567890 3.14159 0x7FFF 2026-10-25 +46701234567" },
  { id: "edge-contractions", source: "hand-written: contractions in several cases", text: "I'm sure we'll go; they'd've known. DON'T SHOUT, It's fine, you're right, we've seen O'Neil's" },
  { id: "edge-case-runs", source: "hand-written: mixed-case words", text: "HTTPServer parseJSONResponse iPhone McDonald XMLHttpRequest camelCase snake_case SCREAMING_CASE" },
  { id: "edge-whitespace-runs", source: "hand-written: spaces, tabs and newlines", text: "a  b   c\t\td\n\ne   \n  f\r\n\r\ng      \n\n\n    h" },
  { id: "edge-trailing-whitespace", source: "hand-written: trailing whitespace", text: "end with spaces   " },
  { id: "edge-punctuation", source: "hand-written: dense ASCII punctuation", text: "!@#$%^&*()_+-=[]{}|;':\",./<>?`~\\ ... !!! ??? --> <== ::" },
  { id: "edge-cjk-han", source: "hand-written: Chinese", text: "通义千问是阿里云研发的大规模语言模型。今天天气很好，我们去公园散步吧。" },
  { id: "edge-japanese", source: "hand-written: Japanese with kana and kanji", text: "東京は日本の首都です。ひらがなとカタカナ、そして漢字を混ぜて書きます。コーヒーを飲みたい。" },
  { id: "edge-korean", source: "hand-written: Hangul", text: "안녕하세요, 세계! 한국어 텍스트입니다." },
  { id: "edge-emoji", source: "hand-written: emoji with zero-width joiners", text: "🙂 family: 👨‍👩‍👧‍👦, flag: 🇸🇪, skin tone: 👍🏽, accented: éàü" },
  { id: "edge-decomposed", source: "hand-written: NFD spellings (no normalizer in this vocabulary)", text: "Café Amélie ñ Ångström" },
  { id: "edge-rtl-and-diacritics", source: "hand-written: Arabic, Hebrew, Nordic", text: "مرحبا بالعالم — שלום עולם — Ærø, Ångström, œuvre, ß, Ǆ" },
  { id: "edge-marker-text", source: "hand-written: Flint markers as literal text", text: "<|system|>Be brief.<|user|>Hi<|assistant|><think>greet</think>Hello<tool_call>f<arg_key>k</arg_key><arg_value>v</arg_value></tool_call><|observation|><tool_response>ok</tool_response><|end_of_text|>" },
  { id: "edge-json", source: "hand-written: compact JSON", text: '{"name":"flint","count":3,"nested":{"a":[1,2.5,-7,null,true]},"unicode":"café ☕"}' },
  { id: "edge-code-rust", source: "hand-written: Rust with generics and lifetimes", text: "pub fn encode<'a, T: AsRef<str> + ?Sized>(&'a self, text: &'a T) -> Result<Vec<u32>> {\n    let ids = self.inner.encode(text.as_ref(), false)?;\n    Ok(ids.get_ids().to_vec())\n}" },
  { id: "edge-code-python", source: "hand-written: Python with indentation", text: "class Node:\n    def __init__(self, value: int) -> None:\n        self.value = value\n        self.children: list['Node'] = []\n\n    def walk(self):\n        yield self\n        for child in self.children:\n            yield from child.walk()\n" },
  { id: "edge-diff", source: "hand-written: a unified diff", text: "@@ -1,3 +1,4 @@\n-let x = 1;\n+let x = 2;\n+let y = x * 3;\n  fn main() {}\n" },
  { id: "edge-mixed-script-code", source: "hand-written: comments in several scripts", text: "# 计算斐波那契数列\ndef fib(n):  # النمو الأسي\n    return n if n < 2 else fib(n-1) + fib(n-2)\n# Тест: русский текст, ελληνικά, 한국어\n" },
];

if (new Set(prompts.map((p) => p.id)).size !== prompts.length) {
  console.error("prompt ids are not unique");
  process.exit(1);
}

/** One `llama-tokenize` run, prompt on stdin. */
function tokenize(text: string): number[] {
  const run = spawnSync({
    cmd: [ORACLE, "-m", model!, "--ids", "--no-bos", "--stdin", "--log-disable"],
    stdin: Buffer.from(text, "utf8"),
    stdout: "pipe",
    stderr: "pipe",
  });
  if (run.exitCode !== 0) {
    throw new Error(`llama-tokenize failed: ${new TextDecoder().decode(run.stderr)}`);
  }
  const stdout = new TextDecoder().decode(run.stdout);
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
        "Tokenizer round-trip fixtures for the Flint vocabulary (reference/flint/tokenizer.json, " +
        "embedded in the binary). `ids` came from the oracle, one run per prompt: " +
        "`llama-tokenize -m Flint-Q4_K_M.gguf --ids --no-bos --stdin` on a llama.cpp build at or past " +
        "a37cc4610. Special-token parsing is on, the mode LagunaTokenizer::encode runs in. " +
        "Nothing is prepended. Regenerate with `bun scripts/flint-fixtures.ts`.",
      add_bos: false,
      vocab: 150144,
      oracle: "llama-tokenize -m Flint-Q4_K_M.gguf --ids --no-bos --stdin",
      prompts: entries,
    },
    null,
    2,
  ) + "\n",
);
console.error(`\nwrote ${out}: ${entries.length} prompts, longest ${longest} tokens`);
