// Structural check of the docs tree: headings, links, quoted references.
//   bun scripts/docs-check.ts
// Exits nonzero on any failure. Run it after any doc move or rename.
//
// What it checks:
//   1. Every markdown link to a local .md file resolves, and when it carries a
//      #fragment, a heading in that file slugs to it (GitHub's rule: lowercase,
//      strip punctuation except - and space, spaces to -).
//   2. No two files claim the same top-level title, except the sanctioned pairs:
//      a record's `#` title equals its log stub's `##` heading, and a decisions
//      topic file's `#` title equals the index entry that links to it.
//   3. Every prose reference of the form `log.md "X"` / `decisions.md "X"` finds
//      X somewhere under the file it names (log.md + docs/records, decisions.md +
//      docs/decisions), as a substring — the repo quotes names, not slugs.
import { readdirSync, readFileSync, statSync, existsSync } from "fs";
import { join, dirname, resolve, relative } from "path";
const root = resolve(import.meta.dir, "..");
const mdFiles: string[] = [];
const walk = (d: string) => { for (const e of readdirSync(d)) { if (e.startsWith(".") || e === "node_modules" || e === "target" || e === "reference") continue; const p = join(d, e); const s = statSync(p); if (s.isDirectory()) walk(p); else if (e.endsWith(".md")) mdFiles.push(p); } };
walk(root);
const srcFiles: string[] = []; // this checker excludes itself: its regexes quote the reference forms
const walkSrc = (d: string) => { for (const e of readdirSync(d)) { const p = join(d, e); const s = statSync(p); if (s.isDirectory()) walkSrc(p); else if (/\.(rs|ts|metal)$/.test(e) && e !== "docs-check.ts") srcFiles.push(p); } };
for (const d of ["src", "scripts", "tests"]) if (existsSync(join(root, d))) walkSrc(join(root, d));
const slug = (h: string) => h.trim().toLowerCase().replace(/[^\p{L}\p{N} \-_]/gu, "").replace(/ /g, "-");
const headings = new Map<string, string[]>(); // file -> heading texts
for (const f of mdFiles) {
  const hs: string[] = [];
  let inFence = false;
  for (const line of readFileSync(f, "utf8").split("\n")) {
    if (line.startsWith("```")) inFence = !inFence;
    if (!inFence && /^#{1,6} /.test(line)) hs.push(line.replace(/^#+ /, ""));
  }
  headings.set(f, hs);
}
const failures: string[] = [];
// 1. links
for (const f of mdFiles) {
  const text = readFileSync(f, "utf8");
  for (const m of text.matchAll(/\]\(([^)\s]+?\.md)(#[^)]*)?\)/g)) {
    const target = m[1].startsWith("/") ? join(root, m[1]) : resolve(dirname(f), m[1]);
    if (!existsSync(target)) { failures.push(`${relative(root, f)}: link to missing file ${m[1]}`); continue; }
    if (m[2]) {
      const frag = decodeURIComponent(m[2].slice(1));
      const hs = headings.get(resolve(target)) ?? [];
      if (!hs.some((h) => slug(h) === frag)) failures.push(`${relative(root, f)}: anchor ${m[1]}${m[2]} matches no heading`);
    }
  }
}
// 2. titles
const titleOwners = new Map<string, string[]>();
for (const [f, hs] of headings) { const t = hs[0]; if (!t) continue; titleOwners.set(t, [...(titleOwners.get(t) ?? []), relative(root, f)]); }
const logHeadings = new Set(headings.get(join(root, "docs/log.md")) ?? []);
for (const [t, owners] of titleOwners) {
  if (owners.length < 2) continue;
  const allRecords = owners.every((o) => o.startsWith("docs/records/"));
  if (allRecords && logHeadings.has(t)) continue;
  failures.push(`title "${t.slice(0, 60)}" owned by ${owners.join(", ")}`);
}
for (const [f, hs] of headings) {
  if (f.endsWith("docs/log.md")) { const seen = new Set<string>(); for (const h of hs) { if (seen.has(h)) failures.push(`log.md: duplicate heading "${h.slice(0, 60)}"`); seen.add(h); } }
}
// 3. quoted references
const corpus = (files: string[]) => files.map((f) => readFileSync(f, "utf8")).join("\n");
const logCorpus = corpus(mdFiles.filter((f) => f.endsWith("docs/log.md") || f.includes("/docs/records/")));
const decCorpus = corpus(mdFiles.filter((f) => f.endsWith("docs/decisions.md") || f.includes("/docs/decisions/")));
for (const f of [...mdFiles, ...srcFiles]) {
  const text = readFileSync(f, "utf8");
  for (const m of text.matchAll(/(log|decisions)\.md[ ,]*(?:§ ?)?"([^"\n]{3,80})"/g)) {
    const [, which, name] = m;
    const c = which === "log" ? logCorpus : decCorpus;
    if (!c.includes(name)) failures.push(`${relative(root, f)}: ${which}.md "${name}" not found`);
  }
}
if (failures.length) { console.error(`docs-check: ${failures.length} failure(s)`); for (const x of failures) console.error("  " + x); process.exit(1); }
console.log(`docs-check: ok (${mdFiles.length} markdown files, ${srcFiles.length} source files scanned)`);
