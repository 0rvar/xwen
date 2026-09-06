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
//   4. TODO.md keeps its shape: the Front has at most ten entries and each names an
//      item; every area item opens with one of the four state tags and carries a
//      `From:` line whose heading exists in docs/ledger-archive.md; no item runs past
//      40 lines. Also prints the age histogram (latest date in each item), which is
//      informational: the 30-day triage rule is a judgement, not a failure.
import { readdirSync, readFileSync, statSync, existsSync } from "fs";
import { join, dirname, resolve, relative } from "path";
const root = resolve(import.meta.dir, "..");
const mdFiles: string[] = [];
const walk = (d: string) => { for (const e of readdirSync(d)) { if (e.startsWith(".") || e === "node_modules" || e === "target" || e === "reference") continue; const p = join(d, e); const s = statSync(p); if (s.isDirectory()) walk(p); else if (e.endsWith(".md")) mdFiles.push(p); } };
walk(root);
const srcFiles: string[] = []; // this checker excludes itself: its regexes quote the reference forms
const walkSrc = (d: string) => { for (const e of readdirSync(d)) { if (e.startsWith(".")) continue; const p = join(d, e); const s = statSync(p); if (s.isDirectory()) walkSrc(p); else if (/\.(rs|ts|metal)$/.test(e) && e !== "docs-check.ts") srcFiles.push(p); } };
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
// 4. ledger shape
{
  // Fenced blocks are blanked (not removed) so line numbers stay true; CRLF is normalized.
  const raw = readFileSync(join(root, "TODO.md"), "utf8").replace(/\r\n?/g, "\n").split("\n");
  let fence = false; const todo = raw.map((l) => { if (l.startsWith("```")) { fence = !fence; return ""; } return fence ? "" : l; });
  const archiveHeadings = new Set(headings.get(join(root, "docs/ledger-archive.md")) ?? []);
  const TAGS = ["measured", "unpriced", "blocked", "small"];
  const MAX_ITEM_LINES = 40; const FRONT_CAP = 10;
  type Item = { start: number; lines: string[]; area: string };
  const items: Item[] = []; const frontTitles: string[] = []; let frontEntries = 0; let frontSeen = false;
  let section = ""; let cur: Item | null = null;
  const finish = () => { if (cur) { while (cur.lines.length && cur.lines[cur.lines.length - 1].trim() === "") cur.lines.pop(); items.push(cur); } cur = null; };
  todo.forEach((l, i) => {
    if (l.startsWith("## ")) { finish(); section = l.slice(3); if (section.startsWith("Front:")) frontSeen = true; if (section.startsWith("Retired")) failures.push(`TODO.md:${i + 1}: retired items live in docs/ledger-archive.md, not here`); return; }
    if (section.startsWith("Front:")) {
      if (/^\d+\. /.test(l)) { frontEntries++; const m = l.match(/^\d+\. \*\*(.+?)\*\*/); if (m) frontTitles.push(m[1]); else failures.push(`TODO.md:${i + 1}: Front entry does not open with a bold item title`); }
      return;
    }
    if (!section) return;
    if (/^- /.test(l)) { finish(); cur = { start: i + 1, lines: [l], area: section }; return; }
    if (/^(\* |\d+\. )/.test(l)) { finish(); failures.push(`TODO.md:${i + 1}: item must be written as "- [ ] [tag] **Title.**", not "${l.slice(0, 4)}…"`); return; }
    if (cur) cur.lines.push(l);
  });
  finish();
  if (!frontSeen) failures.push("TODO.md: no \"## Front\" section");
  if (frontEntries > FRONT_CAP) failures.push(`TODO.md: Front has ${frontEntries} entries, cap is ${FRONT_CAP}`);
  for (const t of frontTitles) { const hits = items.filter((it) => it.lines.slice(0, 2).map((l) => l.trim()).join(" ").includes(`**${t}`)); if (hits.length !== 1) failures.push(`TODO.md: Front entry "${t.slice(0, 60)}" matches ${hits.length} items`); }
  const ages: number[] = []; let undated = 0;
  const now = new Date(); const today = Date.UTC(now.getFullYear(), now.getMonth(), now.getDate());
  for (const it of items) {
    const head = it.lines[0];
    const m = head.match(/^- \[ \] \[(\w+)\] /);
    if (!m || !TAGS.includes(m[1])) failures.push(`TODO.md:${it.start}: item lacks a state tag [${TAGS.join("|")}]`);
    // An open item is titled by its open scope: a closed headline or a ticked box means the closed part never moved.
    const title = head.replace(/^- \[ \] \[\w+\] /, "");
    if (/^\[x\]/i.test(title) || /^\*\*(DONE|SHIPPED|CLOSED|ADAPTED|RESOLVED|LANDED|MOSTLY DONE|SETTLED)\b/i.test(title)) failures.push(`TODO.md:${it.start}: open item headlined as finished ("${title.slice(0, 40)}…")`);
    const from = it.lines.map((l) => l.match(/^\s*From: (.+)\.$/)).find(Boolean);
    if (!from) failures.push(it.lines.some((l) => /^\s*From:/.test(l)) ? `TODO.md:${it.start}: From: line must read "From: <archive heading>." on one line` : `TODO.md:${it.start}: item lacks a From: line`);
    // Retirement is a move, not an annotation: a dated retired line inside an open item means the move was skipped.
    if (it.lines.some((l) => /^\s*\[?Retired \d{4}-\d{2}-\d{2}/i.test(l))) failures.push(`TODO.md:${it.start}: retired item still in the open ledger; move it to docs/ledger-archive.md under "Retired: <area>"`);
    else if (!archiveHeadings.has(from[1])) failures.push(`TODO.md:${it.start}: From: "${from[1].slice(0, 60)}" is no heading in docs/ledger-archive.md`);
    if (it.lines.length > MAX_ITEM_LINES) failures.push(`TODO.md:${it.start}: item runs ${it.lines.length} lines, limit ${MAX_ITEM_LINES}`);
    const dates = (it.lines.join(" ").match(/\b20\d\d-\d\d-\d\d\b/g) ?? []).map((d) => Date.UTC(+d.slice(0, 4), +d.slice(5, 7) - 1, +d.slice(8, 10))).filter((t) => !Number.isNaN(t));
    if (!dates.length) { undated++; continue; }
    ages.push(Math.max(0, Math.floor((today - Math.max(...dates)) / 86400000)));
  }
  const bucket = (lo: number, hi: number) => ages.filter((a) => a >= lo && a < hi).length;
  console.log(`ledger: ${items.length} open items, front ${frontEntries}/${FRONT_CAP}; age by latest date: <7d ${bucket(0, 7)}, 7-30d ${bucket(7, 30)}, 30-60d ${bucket(30, 60)}, 60d+ ${bucket(60, Infinity)}, undated ${undated}`);
}
if (failures.length) { console.error(`docs-check: ${failures.length} failure(s)`); for (const x of failures) console.error("  " + x); process.exit(1); }
console.log(`docs-check: ok (${mdFiles.length} markdown files, ${srcFiles.length} source files scanned)`);
