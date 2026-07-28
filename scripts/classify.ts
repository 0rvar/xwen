#!/usr/bin/env bun
// Example client for the native endpoint: POST /xwen/v1/generate.
//
// A classification/tagging task that exercises the three native-only knobs
// together: an injected current-turn thinking scaffold, `close_thinking` to
// skip free-form reasoning (fast, cheap classification), and a JSON-schema
// output format. Greedy sampling (temperature 0) so runs are comparable.
//
// Usage (server must be running: `xwen serve`):
//   bun scripts/classify.ts                 # run the built-in eval set, print accuracy
//   bun scripts/classify.ts --reason        # let the model think freely instead of the injected scaffold
//   bun scripts/classify.ts -v              # also print each item's model output + thinking
//   bun scripts/classify.ts --one "text"    # classify one ad-hoc input
//   bun scripts/classify.ts --url http://127.0.0.1:5241

const args = process.argv.slice(2);
function flag(name: string): boolean {
  return args.includes(`--${name}`) || (name.length === 1 && args.includes(`-${name}`));
}
function opt(name: string, dflt: string): string {
  const i = args.indexOf(`--${name}`);
  return i >= 0 && args[i + 1] ? args[i + 1] : dflt;
}

const url = opt("url", "http://127.0.0.1:5241");
const reason = flag("reason");
const verbose = flag("v") || flag("verbose");

// ---------------------------------------------------------------------------
// The task: triage inbound messages for a developer-tools product.

const CATEGORIES = [
  "bug_report",
  "feature_request",
  "question",
  "billing",
  "praise",
  "complaint",
] as const;

const TAGS = [
  "crash",
  "performance",
  "ui",
  "docs",
  "api",
  "auth",
  "data-loss",
  "pricing",
  "install",
  "integrations",
] as const;

const SYSTEM = `You triage inbound messages for Fjordline, a developer-tools company. \
Classify each message so it can be routed and prioritized automatically.

Rules:
- category: walk this ladder top to bottom and take the FIRST match.
  1. billing — money or plan terms are the subject (charges, invoices, \
refunds, seats, pricing), even when phrased as a question or a bug.
  2. bug_report — a specific defect or incident we could act on: something \
concrete behaves wrongly.
  3. feature_request — asks for a capability we don't have ("please add", \
"would love", "any chance of").
  4. question — asks how to use what exists, or whether something is \
possible today.
  5. praise — appreciation with nothing asked.
  6. complaint — dissatisfaction or venting that fits none of the above \
(no specific defect, nothing asked, or the author gave up).
- tags describe the technical area; choose only tags plainly supported by the \
text, each tag AT MOST ONCE, never more than three, and none is fine.
- sentiment reflects the author's tone alone.
- priority: urgent = data loss, security exposure, or a production outage; \
high = a defect or missing capability blocking the author's work or rollout; \
medium = actionable but not blocking; low = everything else (praise, musings, \
minor papercuts).`;

// The injected scaffold: this text becomes the model's ENTIRE reasoning for
// the turn (close_thinking cuts thinking off right after it), steering the
// answer without paying for free-form reasoning tokens.
const SCAFFOLD = `I walk the category ladder top to bottom and take the first match. \
Then I pick at most three tags plainly supported by the text, never repeating \
a tag. Then sentiment from the author's tone alone, then the priority ladder \
applied literally.`;

const SCHEMA = {
  type: "object",
  properties: {
    category: { type: "string", enum: [...CATEGORIES] },
    tags: { type: "array", items: { type: "string", enum: [...TAGS] }, maxItems: 3 },
    sentiment: { type: "string", enum: ["positive", "neutral", "negative"] },
    priority: { type: "string", enum: ["low", "medium", "high", "urgent"] },
  },
  required: ["category", "tags", "sentiment", "priority"],
  additionalProperties: false,
};

interface Expected {
  category: (typeof CATEGORIES)[number];
  sentiment?: "positive" | "neutral" | "negative";
}

// Eval set: clear-cut on category (the graded axis); sentiment expectations
// only where tone is unambiguous.
const EVAL: { text: string; expect: Expected }[] = [
  {
    text: "The CLI segfaults every time I run `fjord deploy` with a config file over 1MB. Stack trace attached.",
    expect: { category: "bug_report", sentiment: "neutral" },
  },
  {
    text: "Would love a dark mode for the dashboard. My eyes at 2am would thank you.",
    expect: { category: "feature_request" },
  },
  {
    text: "How do I rotate an API key without downtime? Couldn't find it in the docs.",
    expect: { category: "question" },
  },
  {
    text: "I was charged twice this month. Invoice #4821 and #4822 are for the same plan and period.",
    expect: { category: "billing" },
  },
  {
    text: "Just migrated our whole pipeline to Fjordline and it cut our build times in half. Fantastic work.",
    expect: { category: "praise", sentiment: "positive" },
  },
  {
    text: "Third outage this quarter. We pay for the enterprise tier and honestly it feels like a beta product.",
    expect: { category: "complaint", sentiment: "negative" },
  },
  {
    text: "Is it possible to trigger a sync from a GitHub Action? We want to automate our staging refresh.",
    expect: { category: "question" },
  },
  {
    text: "After yesterday's update the sync job silently drops rows with unicode keys. We lost about 40k records before noticing.",
    expect: { category: "bug_report", sentiment: "negative" },
  },
  {
    text: "Please add SSO with Okta. We can't roll this out org-wide without it.",
    expect: { category: "feature_request" },
  },
  {
    text: "Your pricing page says 'unlimited seats' but sales quoted us per-seat. Which is it?",
    expect: { category: "billing" },
  },
  {
    text: "The new editor keeps freezing for 5-10 seconds whenever I paste more than a few lines. Makes it unusable for real work.",
    expect: { category: "bug_report", sentiment: "negative" },
  },
  {
    text: "Honestly the docs are a maze. I spent an hour looking for webhook retry semantics and gave up.",
    expect: { category: "complaint", sentiment: "negative" },
  },
];

// ---------------------------------------------------------------------------

interface GenerateResponse {
  id: string;
  model: string;
  thinking: string | null;
  content: string;
  stop_reason: string;
  stop_sequence: string | null;
  usage: {
    input_tokens: number;
    cached_tokens: number;
    output_tokens: number;
    thinking_tokens: number;
  };
}

async function classify(text: string): Promise<GenerateResponse> {
  const body = {
    system: SYSTEM,
    messages: [{ role: "user", content: text }],
    // Default mode: the scaffold IS the reasoning. --reason drops the
    // continuation so the model thinks freely before answering.
    ...(reason ? {} : { continue: { thinking: SCAFFOLD, close_thinking: true } }),
    format: { type: "json_schema", schema: SCHEMA },
    max_tokens: reason ? 2048 : 256,
    temperature: 0,
  };
  const res = await fetch(`${url}/xwen/v1/generate`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    throw new Error(`HTTP ${res.status}: ${await res.text()}`);
  }
  return (await res.json()) as GenerateResponse;
}

const one = opt("one", "");
if (one) {
  const r = await classify(one);
  if (verbose && r.thinking) console.log(`thinking: ${r.thinking}\n`);
  console.log(JSON.stringify(JSON.parse(r.content), null, 2));
  console.log(
    `\n[${r.stop_reason}; in ${r.usage.input_tokens} (cached ${r.usage.cached_tokens}) / out ${r.usage.output_tokens} (thinking ${r.usage.thinking_tokens})]`,
  );
  process.exit(0);
}

let catHits = 0;
let sentHits = 0;
let sentTotal = 0;
let outTokens = 0;
const t0 = performance.now();

for (const [i, item] of EVAL.entries()) {
  const r = await classify(item.text);
  const parsed = JSON.parse(r.content);
  // Constrained greedy decoding may repeat an enum item (llguidance has no
  // uniqueItems), so a client dedupes; -v shows the raw output.
  const tags: string[] = [...new Set(parsed.tags as string[])];
  const catOk = parsed.category === item.expect.category;
  let sentMark = "";
  if (item.expect.sentiment) {
    sentTotal++;
    const ok = parsed.sentiment === item.expect.sentiment;
    if (ok) sentHits++;
    sentMark = ok ? "" : ` sentiment ${parsed.sentiment}≠${item.expect.sentiment}`;
  }
  if (catOk) catHits++;
  outTokens += r.usage.output_tokens;
  const mark = catOk ? "ok " : `MISS ${parsed.category}≠${item.expect.category}`;
  console.log(
    `${String(i + 1).padStart(2)}. ${mark}${sentMark}  [${parsed.priority}] ${tags.join(",") || "-"}  ${item.text.slice(0, 60)}`,
  );
  if (verbose) {
    if (r.thinking) console.log(`    thinking: ${r.thinking}`);
    console.log(`    ${r.content}`);
  }
}

const secs = (performance.now() - t0) / 1000;
console.log(
  `\ncategory ${catHits}/${EVAL.length}` +
    (sentTotal ? `, sentiment ${sentHits}/${sentTotal}` : "") +
    `  (${reason ? "free reasoning" : "injected scaffold"}, ${secs.toFixed(1)}s, ${outTokens} output tokens)`,
);
