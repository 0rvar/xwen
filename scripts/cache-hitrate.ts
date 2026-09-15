// Prefix-cache hit rate by how many OTHER sessions the server served in between two
// turns of one session, read off the metrics log `xwen serve` writes.
//   bun scripts/cache-hitrate.ts [--model <substring>] [--file <metrics.jsonl>]
//
// A row is scored when its session's previous request succeeded and was at least 5000
// tokens (prompt plus reply): the expected reuse is that previous request's whole
// length, a hit is a cached count of at least half of it, and the bucket is the number
// of distinct other sessions served between the two. The 0 and 1 rows are what a
// two-slot cache can serve; the 2+ rows are what more slots, or a byte budget, buy.
import { readFileSync } from "node:fs";
import { homedir } from "node:os";

type Row = {
  model: string;
  session?: string;
  ok: boolean;
  prompt_tokens: number;
  cached_tokens: number;
  decode_tokens?: number;
};

const args = process.argv.slice(2);
const flag = (name: string): string | undefined => {
  const at = args.indexOf(name);
  return at >= 0 ? args[at + 1] : undefined;
};
const model = flag("--model") ?? "";
const file =
  flag("--file") ??
  `${process.env.XDG_STATE_HOME ?? `${homedir()}/.local/state`}/xwen/metrics.jsonl`;

const rows: Row[] = readFileSync(file, "utf8")
  .trim()
  .split("\n")
  .filter((line) => line.length > 0)
  .map((line) => JSON.parse(line) as Row)
  .filter((row) => row.session && row.model.includes(model));

const MIN_PREVIOUS = 5000;
const lastIndex: Record<string, number> = {};
const buckets = new Map<number, { requests: number; hits: number; fraction: number }>();
let scored = 0;
for (let i = 0; i < rows.length; i++) {
  const row = rows[i];
  const session = row.session!;
  const previousAt = lastIndex[session];
  if (previousAt !== undefined) {
    const previous = rows[previousAt];
    const expected = previous.prompt_tokens + (previous.decode_tokens ?? 0);
    if (previous.ok && expected >= MIN_PREVIOUS) {
      const between = new Set(rows.slice(previousAt + 1, i).map((r) => r.session)).size;
      const key = Math.min(between, 4);
      const bucket = buckets.get(key) ?? { requests: 0, hits: 0, fraction: 0 };
      bucket.requests += 1;
      if (row.cached_tokens >= expected / 2) bucket.hits += 1;
      bucket.fraction += row.cached_tokens / expected;
      buckets.set(key, bucket);
      scored += 1;
    }
  }
  lastIndex[session] = i;
}

console.log("sessions_between  requests  hits(>=50%)  mean_cached_fraction");
for (const key of [...buckets.keys()].sort((a, b) => a - b)) {
  const bucket = buckets.get(key)!;
  const label = key === 4 ? "4+" : String(key);
  const rate = ((100 * bucket.hits) / bucket.requests).toFixed(0);
  console.log(
    `${label.padEnd(16)}  ${String(bucket.requests).padStart(8)}  ${String(bucket.hits).padStart(5)} (${rate}%)  ${(bucket.fraction / bucket.requests).toFixed(2)}`,
  );
}
console.log(
  `scored ${scored} of ${rows.length} rows${model ? ` matching "${model}"` : ""} in ${file}`,
);
