#!/usr/bin/env bun
// Demo of DECOMPOSED classification against `xwen batch`: instead of asking one
// prompt for every label at once, each taxonomy gets its own item with its own
// schema, and all of them go out as a single batch request.
//
// The shape is what makes it cheap. Every item repeats the same system prompt
// and the same source text and differs only in its trailing instruction, so the
// batch command finds one long shared token prefix, prefills it once, and each
// item resumes from that snapshot. Per-item schemas also keep each answer to a
// short constrained value drawn from a known option set, which is what lets
// `include_score` report a distribution over the allowed values rather than
// over tokens.
//
// `--compare-thinking` adds a second arm: every item goes out twice in the SAME
// batch, once thinking-off and once with a closed think block injected ahead of
// the answer. The scaffold restates the rubric and the discipline and never any
// document-specific evidence, so what it measures is whether re-reading the
// rubric changes the answer — not whether a hint was smuggled in. Injected
// thinking is prompt rather than completion, so the token budgets are unchanged
// and both arms still share the one prefix, snapshot and load.
//
// What the shapes in here are worth, stated as properties rather than as a
// sales pitch:
//
// Decomposition. One taxonomy per item beats a single multi-taxonomy JSON for
// the same reason per-tag scored booleans beat one joint array: each label gets
// the model's whole attention and carries its own confidence. A tag set emitted
// as one array drops labels that the same set, asked as one boolean per tag,
// recovers — and the per-tag P(true) shows WHERE the decision boundary sits,
// not merely which side of it a tag landed on.
//
// Injected thinking. A closed scaffold that restates the rubric and the
// evidence discipline — never a fact from the document — moves near-tie fields
// toward the considered reading and leaves already-confident fields alone. It
// costs no completion tokens, being prompt, and it narrows the spread between
// checkpoints: the small fast model converges on the large one's judgments. A
// scaffold belongs to its taxonomy and has to hold up on held-out documents;
// one tuned against a single message is fitted to that message.
//
// Ground truth. An underspecified rubric announces itself twice over: the label
// moves with the asking protocol, and its probability strands between the
// confident mass and the noise floor. The score distribution is a diagnostic on
// the LABEL, not only on the model.
//
// The two models run STRICTLY SEQUENTIALLY — one large model process at a time
// on this machine (CLAUDE.md, operational hazards).
//
// Usage:
//   bun scripts/classify-demo.ts                     # 35b then 27b, print the report
//   bun scripts/classify-demo.ts --model 35b         # just one model
//   bun scripts/classify-demo.ts --no-draft          # passed through to `xwen batch`
//   bun scripts/classify-demo.ts --json              # dump raw responses instead
//   bun scripts/classify-demo.ts --compare-thinking  # both arms, side by side

import { dirname, join } from "node:path";

const repo = dirname(import.meta.dir);
const args = process.argv.slice(2);
const flag = (n: string) => args.includes(`--${n}`);
const opt = (n: string, d: string) => {
  const i = args.indexOf(`--${n}`);
  return i >= 0 && args[i + 1] ? args[i + 1] : d;
};

if (flag("help") || args.includes("-h")) {
  console.log(
    [
      "Usage: bun scripts/classify-demo.ts [--model 35b|27b] [--no-draft] [--json]",
      "                                    [--compare-thinking]",
      "",
      "  --model 35b|27b     run only this checkpoint (default: 35b then 27b, in that order)",
      "  --no-draft          pass --no-draft through to `xwen batch` (plain decode)",
      "  --json              print the raw batch responses instead of the report",
      "  --compare-thinking  run every item twice in one batch, thinking-off vs an",
      "                      injected closed think block, and compare the two arms",
      "  --help              this text",
    ].join("\n"),
  );
  process.exit(0);
}

const only = opt("model", "");
if (only && only !== "35b" && only !== "27b") {
  console.error(`unknown --model ${only}; expected 35b or 27b`);
  process.exit(2);
}
const models = only ? [only] : ["35b", "27b"];
const noDraft = flag("no-draft");
const rawJson = flag("json");
const compare = flag("compare-thinking");
/// Id suffixes, which are also the arm names. Without the flag there is one
/// unsuffixed arm, so the request is exactly what it has always been.
const ARMS = compare ? [":plain", ":think"] : [""];

// ---------------------------------------------------------------------------
// The task: one support email, classified along nine taxonomies.

const SOURCE_TEXT = `Subject: Third replacement request — order #EU-88412 — please just refund me

Hi,

I'm writing about my Brewmaster Pro espresso machine, order #EU-88412, placed on July 14th. The first unit arrived with a cracked water tank — water everywhere the first time I filled it. Your support team was friendly and sent a replacement quickly, which I appreciated. Unfortunately the replacement has now developed the same fault: a hairline crack along the tank seam, and this morning I found my kitchen counter soaked again.

I've been patient through two rounds of this, but I no longer trust this product line. I'd like a full refund to my original payment method rather than another replacement. My daughter's birthday is next Saturday and I had planned to finally make proper cappuccinos for the family — that's clearly not happening with this machine, so I'd appreciate the refund being processed before then if at all possible.

You can reach me at maria.lindqvist@example.com if you need anything else from me. I want to be clear that your support staff have been perfectly pleasant — it's the product that has let me down twice.

Regards,
Maria Lindqvist`;

const SYSTEM_PROMPT =
  "You are a precise support-ticket classification engine. You will be shown a " +
  "customer message and then asked to classify it along one specific taxonomy. " +
  "Answer with JSON only, following the schema exactly. Base your answer only on " +
  "the message content.";

interface Field {
  name: string;
  values: readonly string[] | null; // null = boolean
  expected: string | boolean;
}

interface Taxonomy {
  id: string;
  rules?: string;
  fields: Field[];
  /// Absent for the scalar taxonomies. The two tag items select a SET out of one
  /// shared tag vocabulary, and differ only in how they are asked for it: `tags`
  /// takes the grammar path (one array, no scores available), `tags_scored`
  /// asks the same question as one boolean per tag so every tag carries its own
  /// probability. The pair exists to be compared: the two arms are the same
  /// question, and a tag the array form omits is often one the boolean form
  /// answers true with room to spare.
  kind?: "tags" | "tags_scored";
  maxTokens?: number;
}

const TAG_SET = [
  "product_defect",
  "shipping_damage",
  "refund_request",
  "replacement",
  "positive_support_experience",
  "deadline_mentioned",
  "repeat_issue",
  "billing_issue",
  "safety_concern",
  "cancellation",
] as const;

const EXPECTED_TAGS = [
  "product_defect",
  "refund_request",
  "replacement",
  "positive_support_experience",
  "deadline_mentioned",
  "repeat_issue",
];

const TAG_RULES =
  "Select every tag that applies to the message; leave out tags that do not. " +
  "product_defect = the product itself is faulty; shipping_damage = damage " +
  "caused in transit; repeat_issue = the same problem occurred more than once; " +
  "deadline_mentioned = the customer names a date or timeframe; " +
  "positive_support_experience = the customer praises support staff; " +
  "replacement = a replacement unit was sent, offered, or requested at any " +
  "point, even if the customer now declines further replacements.";

// The last entry is COMPOUND: two related taxonomies share one schema and are
// emitted together, which is the cheaper shape whenever the labels inform each
// other. The rest are one field each.
const TAXONOMIES: Taxonomy[] = [
  {
    id: "sentiment",
    rules:
      "Overall sentiment toward the company/product. 'mixed' only when praise " +
      "and criticism are roughly balanced.",
    fields: [
      {
        name: "sentiment",
        values: ["positive", "neutral", "negative", "mixed"],
        expected: "negative",
      },
    ],
  },
  {
    id: "urgency",
    rules:
      "'critical' is reserved for safety issues or total service outages; " +
      "explicit deadlines raise urgency.",
    fields: [
      { name: "urgency", values: ["low", "medium", "high", "critical"], expected: "high" },
    ],
  },
  {
    id: "intent",
    rules: "The customer's PRIMARY ask, not secondary remarks.",
    fields: [
      {
        name: "intent",
        values: ["complaint", "question", "feature_request", "refund_request", "praise", "other"],
        expected: "refund_request",
      },
    ],
  },
  {
    id: "product_area",
    rules: "The area the underlying problem belongs to.",
    fields: [
      {
        name: "product_area",
        values: ["hardware", "software", "billing", "shipping", "documentation", "other"],
        expected: "hardware",
      },
    ],
  },
  {
    id: "tone",
    rules: "The dominant emotional register of the writing itself.",
    fields: [
      {
        name: "tone",
        values: ["angry", "frustrated", "polite", "sarcastic", "enthusiastic", "neutral"],
        expected: "frustrated",
      },
    ],
  },
  {
    id: "language",
    rules: "The language the message is written in.",
    fields: [{ name: "language", values: ["en", "sv", "de", "fr", "other"], expected: "en" }],
  },
  {
    id: "contains_pii",
    rules:
      "True if the message contains personally identifying information (names, " +
      "emails, order numbers).",
    fields: [{ name: "contains_pii", values: null, expected: true }],
  },
  {
    id: "resolution",
    rules: "needs_human when money movement or judgment is required.",
    fields: [
      {
        name: "resolution",
        values: ["needs_human", "auto_resolvable", "no_action_needed"],
        expected: "needs_human",
      },
    ],
  },
  {
    id: "emotion",
    rules:
      "emotion is the customer's dominant feeling; intensity is how strongly it " +
      "is expressed.",
    fields: [
      {
        name: "emotion",
        values: ["frustration", "anger", "joy", "disappointment", "confusion", "neutral"],
        expected: "disappointment",
      },
      { name: "emotion_intensity", values: ["low", "medium", "high"], expected: "medium" },
    ],
  },
  { id: "tags", kind: "tags", rules: TAG_RULES, fields: [], maxTokens: 96 },
  { id: "tags_scored", kind: "tags_scored", rules: TAG_RULES, fields: [], maxTokens: 192 },
];

// One row per scalar field, plus one row per tag item — the tag set is graded
// as a whole, since a per-tag tally would drown the scalar labels.
const TOTAL_ROWS =
  TAXONOMIES.flatMap((t) => t.fields).length +
  TAXONOMIES.filter((t) => t.kind !== undefined).length;

// ---------------------------------------------------------------------------
// Request construction.

function schemaFor(t: Taxonomy) {
  if (t.kind === "tags") {
    return {
      type: "object",
      properties: { tags: { type: "array", items: { enum: [...TAG_SET] } } },
      required: ["tags"],
      additionalProperties: false,
    };
  }
  if (t.kind === "tags_scored") {
    const properties: Record<string, unknown> = {};
    for (const tag of TAG_SET) properties[tag] = { type: "boolean", include_score: "all" };
    return {
      type: "object",
      properties,
      required: [...TAG_SET],
      additionalProperties: false,
    };
  }
  const properties: Record<string, unknown> = {};
  for (const f of t.fields) {
    properties[f.name] =
      f.values === null
        ? { type: "boolean", include_score: "all" }
        : { type: "string", enum: [...f.values], include_score: "all" };
  }
  return {
    type: "object",
    properties,
    required: t.fields.map((f) => f.name),
    additionalProperties: false,
  };
}

// Shared content first, task-specific text last: the shared prefix is only as
// long as the identical HEAD of every item's prompt, so ordering is the whole
// trick. Moving the instruction to the top would cut the prefix to nothing.
function userMessage(t: Taxonomy): string {
  return `${SOURCE_TEXT}\n\n---\n\nClassify along this taxonomy.\n${t.rules ?? ""}`;
}

/// The reasoning injected into the `:think` arm. It restates the rubric, the
/// option set and the discipline to apply, and deliberately contains no
/// evidence from the message — a scaffold that quoted the message would be
/// answering the question rather than framing it.
function scaffoldFor(t: Taxonomy): string {
  if (t.kind) {
    return (
      "I am selecting every applicable tag from a fixed set. The rules: " +
      `${TAG_RULES} For each tag I will ask only: does the message contain ` +
      "explicit evidence for this tag? I will include exactly the tags with " +
      "evidence and no others, in the required JSON shape."
    );
  }
  // A compound item names both of its fields and both option lists; a single
  // one needs no such labelling.
  const names = t.fields.map((f) => f.name).join(" and ");
  const options = t.fields
    .map((f) => {
      const values = f.values === null ? ["true", "false"] : [...f.values];
      return t.fields.length > 1 ? `${f.name}: ${values.join(", ")}` : values.join(", ");
    })
    .join("; ");
  return (
    `I am classifying along one taxonomy: ${names}. The rubric: ${t.rules ?? ""} ` +
    `The options are: ${options}. I will weigh each option against explicit ` +
    "evidence in the message, prefer the dominant signal over isolated phrases, " +
    "and not read in anything the message does not say. Then I will answer with " +
    "the single best option in the required JSON shape."
  );
}

function buildRequest(model: string) {
  return {
    model,
    defaults: {
      max_tokens: 64,
      sampling: { temperature: 0 },
      thinking: false,
    },
    items: TAXONOMIES.flatMap((t) => {
      const body = {
        messages: [
          { role: "system", content: SYSTEM_PROMPT },
          { role: "user", content: userMessage(t) },
        ],
        schema: schemaFor(t),
        // The tag items need a bigger budget than a one-value answer: the
        // scored one is checked against the worst case for its whole skeleton,
        // not just the values it will actually emit. The `:think` arm keeps the
        // same budget, since injected reasoning is prompt, not completion.
        max_tokens: t.maxTokens ?? 64,
      };
      if (!compare) return [{ id: t.id, ...body }];
      // The plain arm carries no `thinking` key at all, so it inherits the
      // default `false` and is the same item the unflagged run sends.
      return [
        { id: `${t.id}:plain`, ...body },
        { id: `${t.id}:think`, ...body, thinking: scaffoldFor(t) },
      ];
    }),
  };
}

// ---------------------------------------------------------------------------
// `xwen batch`: JSON on stdin, JSON on stdout, stats and logs on stderr.

interface BatchItem {
  id: string;
  content: string;
  text: string;
  json: Record<string, unknown> | null;
  finish_reason: string;
  usage: {
    prompt_tokens: number;
    cached_prefix_tokens: number;
    completion_tokens: number;
  };
  error?: string;
}

interface BatchResponse {
  model: string;
  items: BatchItem[];
  stats: {
    shared_prefix_tokens: number;
    snapshot_ms: number;
    load_ms: number;
    total_ms: number;
  };
}

async function runBatch(model: string): Promise<BatchResponse | null> {
  const bin = join(repo, "target/release/xwen");
  const proc = Bun.spawn([bin, "batch", ...(noDraft ? ["--no-draft"] : [])], {
    cwd: repo,
    stdin: "pipe",
    stdout: "pipe",
    stderr: "inherit",
  });
  proc.stdin.write(JSON.stringify(buildRequest(model)));
  proc.stdin.end();
  const out = await new Response(proc.stdout).text();
  const code = await proc.exited;

  if (code !== 0) {
    console.error(`\n${model}: xwen batch exited ${code}`);
    if (out.trim()) console.error(out.trim());
    return null;
  }
  try {
    return JSON.parse(out) as BatchResponse;
  } catch (e) {
    console.error(`\n${model}: response was not JSON (${String(e)})`);
    if (out.trim()) console.error(out.slice(0, 2000));
    return null;
  }
}

// ---------------------------------------------------------------------------
// Report.

interface Scored {
  value: unknown;
  score: number | null;
  scores: Record<string, number> | null;
  escape: number | null;
}

/// A field with `include_score` comes back wrapped as
/// {value, score, scores, escape}; without it, as a bare value. Accept both so
/// the report still prints if scoring is unavailable.
function unwrap(raw: unknown): Scored {
  if (raw && typeof raw === "object" && !Array.isArray(raw) && "value" in raw) {
    const o = raw as Record<string, unknown>;
    return {
      value: o.value,
      score: typeof o.score === "number" ? o.score : null,
      scores:
        o.scores && typeof o.scores === "object"
          ? (o.scores as Record<string, number>)
          : null,
      escape: typeof o.escape === "number" ? o.escape : null,
    };
  }
  return { value: raw, score: null, scores: null, escape: null };
}

interface Row {
  taxonomy: string;
  predicted: string;
  score: number | null;
  expected: string;
  ok: boolean;
  /// Lines printed indented under the row. Scalar rows use this for the score
  /// distribution, tag rows for the set diff and the per-tag probabilities.
  detail: string[];
}

const pad = (s: string, n: number) => s + " ".repeat(Math.max(0, n - s.length));
const num = (v: number | null, d = 3) => (v === null ? "-" : v.toFixed(d));
// Escape mass is normally tiny; fixed decimals would round every interesting
// value to 0.000. Read it only for the quoted enum fields, where it is the mass
// leaving the option set; on a bare boolean it is confounded by formatting and
// carries less than its magnitude suggests.
const tiny = (v: number | null) =>
  v === null ? "-" : v === 0 ? "0" : v < 1e-3 ? v.toExponential(1) : v.toFixed(3);

/// P(true) for a scored boolean. The wrapper reports the confidence in the
/// value it chose, so a `false` answer has to be inverted to be comparable
/// across tags.
function pTrue(s: Scored): number | null {
  if (s.scores && "true" in s.scores) return s.scores.true;
  if (s.score === null) return null;
  return s.value === true ? s.score : 1 - s.score;
}

const inTagOrder = (tags: Iterable<string>) => {
  const set = new Set(tags);
  const known = TAG_SET.filter((t) => set.has(t));
  const unknown = [...set].filter((t) => !TAG_SET.includes(t as (typeof TAG_SET)[number]));
  return [...known, ...unknown];
};

const sameSet = (a: string[], b: string[]) =>
  a.length === b.length && a.every((x) => b.includes(x));

function tagDiff(predicted: string[]): string[] {
  if (sameSet(predicted, EXPECTED_TAGS)) return [];
  const missing = EXPECTED_TAGS.filter((t) => !predicted.includes(t));
  const extra = predicted.filter((t) => !EXPECTED_TAGS.includes(t));
  return [`missing: ${missing.join(",") || "none"}   extra: ${extra.join(",") || "none"}`];
}

function rowsFor(resp: BatchResponse, arm = ""): Row[] {
  const byId = new Map(resp.items.map((it) => [it.id, it]));
  const rows: Row[] = [];

  const miss = (name: string, expected: string, note: string): Row => ({
    taxonomy: name,
    predicted: "-",
    score: null,
    expected,
    ok: false,
    detail: [note],
  });

  for (const t of TAXONOMIES) {
    const itemId = t.id + arm;
    const item = byId.get(itemId);
    // The predicted cell already carries a whole set; repeating the expected
    // set verbatim would double the row width to say nothing the diff line
    // below it does not say better.
    const expectedTags = `${EXPECTED_TAGS.length} tags`;

    if (t.kind) {
      if (!item) {
        rows.push(miss(t.id, expectedTags, `no item '${itemId}' in the response`));
        continue;
      }
      if (item.error) {
        rows.push(miss(t.id, expectedTags, `error: ${item.error}`));
        continue;
      }
      if (!item.json) {
        rows.push(
          miss(t.id, expectedTags, `no json (finish_reason ${item.finish_reason})`),
        );
        continue;
      }

      if (t.kind === "tags") {
        const raw = item.json.tags;
        if (!Array.isArray(raw)) {
          rows.push(miss(t.id, expectedTags, "json has no 'tags' array"));
          continue;
        }
        const predicted = inTagOrder(raw.map(String));
        // Constrained decoding has no uniqueItems, so a repeated tag is the
        // model's, not a bug — dedupe for grading and say that it happened.
        const dupes = raw.length !== predicted.length;
        rows.push({
          taxonomy: t.id,
          predicted: (predicted.join(",") || "(none)") + (dupes ? " (dupes)" : ""),
          score: null,
          expected: expectedTags,
          ok: sameSet(predicted, EXPECTED_TAGS),
          detail: tagDiff(predicted),
        });
        continue;
      }

      const scored = TAG_SET.map((tag) => [tag, unwrap(item.json![tag])] as const);
      const predicted = scored.filter(([, s]) => s.value === true).map(([tag]) => tag);
      const picked = scored.filter(([, s]) => s.value === true).map(([, s]) => s.score);
      const mean =
        picked.length && picked.every((v) => v !== null)
          ? (picked as number[]).reduce((a, b) => a + b, 0) / picked.length
          : null;
      // Always shown: per-tag probabilities are the whole point of asking for
      // the set this way, whether or not the answer came out right.
      const line = scored
        .map(([tag, s]) => [tag, pTrue(s)] as const)
        .sort((a, b) => (b[1] ?? -1) - (a[1] ?? -1))
        .map(([tag, p]) => `${tag} ${num(p)}`)
        .join("  ");
      rows.push({
        taxonomy: t.id,
        predicted: predicted.join(",") || "(none)",
        score: mean,
        expected: expectedTags,
        ok: sameSet(predicted, EXPECTED_TAGS),
        detail: [...tagDiff(predicted), `P(true): ${line}`],
      });
      continue;
    }

    for (const f of t.fields) {
      const expected = String(f.expected);
      if (!item) {
        rows.push(miss(f.name, expected, `no item '${itemId}' in the response`));
        continue;
      }
      if (item.error) {
        rows.push(miss(f.name, expected, `error: ${item.error}`));
        continue;
      }
      if (!item.json || !(f.name in item.json)) {
        rows.push(
          miss(
            f.name,
            expected,
            `field missing from json (finish_reason ${item.finish_reason})`,
          ),
        );
        continue;
      }
      const s = unwrap(item.json[f.name]);
      const ok = String(s.value) === expected;
      // Show the distribution wherever the answer is wrong or the model was not
      // confident — those are the rows a decomposed setup exists to expose.
      const shaky = !ok || s.score === null || s.score < 0.9;
      const dist = s.scores
        ? Object.entries(s.scores)
            .sort((a, b) => b[1] - a[1])
            .map(([k, v]) => `${k} ${num(v)}`)
            .join("  ")
        : "scores unavailable";
      rows.push({
        taxonomy: f.name,
        predicted: String(s.value),
        score: s.score,
        expected,
        ok,
        detail: shaky ? [`${dist}   escape ${tiny(s.escape)}`] : [],
      });
    }
  }
  return rows;
}

function printReport(resp: BatchResponse, rows: Row[]) {
  // Widths are capped so one long cell — the tag rows carry a whole set —
  // cannot push the ten scalar rows' columns off to the right.
  const cap = (n: number) => Math.min(n, 30);
  const cols = [
    Math.max(8, ...rows.map((r) => r.taxonomy.length)),
    cap(Math.max(9, ...rows.map((r) => r.predicted.length))),
    5,
    cap(Math.max(8, ...rows.map((r) => r.expected.length))),
  ];
  console.log(
    `${pad("taxonomy", cols[0])}  ${pad("predicted", cols[1])}  ${pad("score", cols[2])}  ` +
      `${pad("expected", cols[3])}  ok`,
  );
  for (const r of rows) {
    console.log(
      `${pad(r.taxonomy, cols[0])}  ${pad(r.predicted, cols[1])}  ` +
        `${pad(num(r.score, 2), cols[2])}  ${pad(r.expected, cols[3])}  ${r.ok ? "ok" : "MISS"}`,
    );
    for (const line of r.detail) console.log(`    ${line}`);
  }

  const hits = rows.filter((r) => r.ok).length;
  // The shared prefix is prefilled once BEFORE any item runs and every item
  // resumes from that one snapshot, so all items report the same figure —
  // including the first. A disagreement means some item did not resume from the
  // snapshot at all (a failed item reports zero), which is worth seeing.
  const cached = [
    ...new Set(
      TAXONOMIES.map((t) => resp.items.find((it) => it.id === t.id)).map(
        (it) => it?.usage.cached_prefix_tokens ?? -1,
      ),
    ),
  ];
  const cachedText =
    cached.length === 1 ? `${cached[0]}` : `${cached.join("/")} (items disagree)`;
  console.log(
    `\naccuracy ${hits}/${TOTAL_ROWS}   shared_prefix ${resp.stats.shared_prefix_tokens} tok   ` +
      `cached_prefix ${cachedText} tok/item   ` +
      `load ${resp.stats.load_ms} ms   total ${resp.stats.total_ms} ms`,
  );
}

/// Side-by-side arms. The two rows for a taxonomy answered the same question
/// off the same prefix and differ only in the injected reasoning, so the
/// interesting column is the one saying whether that changed anything.
function printCompare(resp: BatchResponse, plain: Row[], think: Row[]) {
  const cell = (r: Row) => `${r.predicted} (${num(r.score, 2)})`;
  const pairs = plain.map((p, i) => [p, think[i]] as const);
  const cap = (n: number) => Math.min(n, 34);
  const cols = [
    Math.max(8, ...plain.map((r) => r.taxonomy.length)),
    cap(Math.max(11, ...plain.map((r) => cell(r).length))),
    cap(Math.max(11, ...think.map((r) => cell(r).length))),
    cap(Math.max(8, ...plain.map((r) => r.expected.length))),
  ];
  console.log(
    `${pad("taxonomy", cols[0])}  ${pad("plain", cols[1])}  ${pad("think", cols[2])}  ` +
      `${pad("expected", cols[3])}  note`,
  );
  for (const [p, k] of pairs) {
    const changed = p.predicted !== k.predicted;
    const delta = p.score !== null && k.score !== null ? k.score - p.score : null;
    const notes: string[] = [];
    if (changed) notes.push("CHANGED");
    // Scores compare with tolerance, never bit-for-bit: the two arms reach the
    // same field over different prompt lengths, and a genuine near-tie can land
    // either way. Only a move worth acting on gets reported.
    if (delta !== null && Math.abs(delta) >= 0.05) {
      notes.push(`${delta >= 0 ? "+" : ""}${delta.toFixed(2)}`);
    }
    // Which arm was wrong is what makes CHANGED readable as better or worse.
    if (!p.ok && !k.ok) notes.push("both MISS");
    else if (!p.ok) notes.push("plain MISS");
    else if (!k.ok) notes.push("think MISS");
    console.log(
      `${pad(p.taxonomy, cols[0])}  ${pad(cell(p), cols[1])}  ${pad(cell(k), cols[2])}  ` +
        `${pad(p.expected, cols[3])}  ${notes.join("  ")}`,
    );
    if (!p.ok || !k.ok || changed) {
      for (const line of p.detail) console.log(`    plain  ${line}`);
      for (const line of k.detail) console.log(`    think  ${line}`);
    }
  }

  const completion = (arm: string) =>
    resp.items
      .filter((it) => it.id.endsWith(arm))
      .reduce((sum, it) => sum + it.usage.completion_tokens, 0);
  console.log(
    `\naccuracy plain ${plain.filter((r) => r.ok).length}/${TOTAL_ROWS}   ` +
      `think ${think.filter((r) => r.ok).length}/${TOTAL_ROWS}   ` +
      `completion tokens plain ${completion(":plain")}  think ${completion(":think")}   ` +
      `shared_prefix ${resp.stats.shared_prefix_tokens} tok   total ${resp.stats.total_ms} ms`,
  );
}

// ---------------------------------------------------------------------------

const results = new Map<string, Map<string, Row[]>>();
let ran = 0;

for (const model of models) {
  console.log(`\n=== ${model}${noDraft ? " (--no-draft)" : ""} ===`);
  const resp = await runBatch(model);
  if (!resp) continue;
  ran++;
  if (rawJson) {
    console.log(JSON.stringify(resp, null, 2));
    continue;
  }
  const arms = new Map(ARMS.map((arm) => [arm, rowsFor(resp, arm)]));
  results.set(model, arms);
  if (compare) printCompare(resp, arms.get(":plain")!, arms.get(":think")!);
  else printReport(resp, arms.get("")!);
}

if (!rawJson && results.size === 2) {
  const [a, b] = models;
  for (const arm of ARMS) {
    const ra = results.get(a)!.get(arm)!;
    const rb = results.get(b)!.get(arm)!;
    const disagree = ra
      .map((r, i) => [r, rb[i]] as const)
      .filter(([r, o]) => r.predicted !== o.predicted);
    console.log(`\n=== ${a} vs ${b}${arm ? ` (${arm.slice(1)})` : ""} ===`);
    if (disagree.length === 0) {
      console.log("identical predictions on all fields");
      continue;
    }
    const w = Math.max(...disagree.map(([r]) => r.taxonomy.length));
    for (const [r, o] of disagree) {
      console.log(
        `${pad(r.taxonomy, w)}  ${a} ${r.predicted} (${num(r.score, 2)})  ` +
          `${b} ${o.predicted} (${num(o.score, 2)})  expected ${r.expected}`,
      );
    }
  }
}

process.exit(ran === 0 ? 1 : 0);
