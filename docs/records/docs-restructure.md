# 2026-09-06 — Docs restructured: decisions by topic, records behind log stubs, perf-state and benching split out, and the ledger regrouped by area with a ten-item Front

Two passes on one day, both driven by scripts with a line-accounting assertion, both
reviewed. The log stub of this heading carries the headline; this is the arc.

## Why

Six weeks of shipping had produced four files nobody could read top to bottom:
AGENTS.md at 524 lines with 140 of them perf history, TODO.md at 3194 with 25-line DONE
annotations inside open items, log.md at 5600, decisions.md at 3206. The rules that grew
them were good rules: every arc records itself, scope is never silently dropped, nothing
is deleted. What was missing was a home per kind of fact, and an exit from the ledger
other than shipping. The owner's brief for the structure was that instructions must not
mandate a new record every session: sometimes an arc updates an existing record or a
decision paragraph, sometimes a mix.

## Pass one: a home per kind of fact

- `docs/decisions.md` became a 236-line index over `docs/decisions/<topic>.md`, 17 topic
  files, split by the existing topic headings (`/tmp/ceil/docs-migrate/split-decisions.ts`).
  Every bold lead is listed in the index, so the old grep still works.
- `docs/log.md` went from 5600 to 1796 lines: 32 entries of 80 lines or more moved
  verbatim to `docs/records/<slug>.md` behind a stub, slugs hand-assigned from a rules
  table after an auto-slugging attempt produced unreadable names. The stubs were then
  rewritten as 4-12 line summaries with the headings byte-identical, because everything
  links by heading text.
- `docs/perf-state.md` (136 lines) became the single source for the current figures: a
  23-row table with value, date, commit and the power line, plus the ceilings. Several
  figures turned out to have no recorded commit. `docs/benching.md` (110 lines) took the
  measurement rules out of AGENTS.md.
- `docs/ledger-archive.md` received 44 closed TODO items verbatim, adjudicated item by
  item by an agent reading each one ("unsure means open"), under the ledger's own section
  headings; ~50 annotations were trimmed to three lines and a link, each figure checked
  present at the link target first.
- AGENTS.md went from 524 to 388 lines: a doc map plus a rules block replaced the
  doc-system paragraphs, and the Perf state section shrank to headlines and links.
- `scripts/docs-check.ts` asserts links, anchors, unique titles and quoted
  `log.md "X"` / `decisions.md "X"` references. Its first run found three references that
  had silently drifted from their headings.

Commits: 7ad7d8b, 27c4b27, 40fed1a, c413d8a, 9d2e72c, b091577.

## Pass two: the ledger as memory and backlog

The owner's question after pass one was whether the ledger had become the classic
infinite backlog. It had, by construction: with "never dropped" and "never deleted" the
only exit was shipping, about 134 items had entered in six weeks against 44 closed, and
the biggest item was a 527-line journal wearing a checkbox. The redesign is recorded in
decisions.md "The ledger is a memory and a backlog"; the mechanics were:

- `extract.ts` split TODO.md into 90 items and 219 lines of section prose, each item
  written out with relative line numbers.
- Two classification agents produced one JSON decision per item: area (eight fixed
  names), state tag, closed line ranges, promoted sub-item ranges with their own area,
  tag and title, a Front proposal and a retire proposal. One agent took the 89 ordinary
  items, the other the port item alone, which it tiled into 24 archive ranges (330
  lines) and 15 promoted sub-items (184 lines), with no gap and no overlap.
- `apply.ts` rebuilt TODO.md as a preamble, a Front of ten ranked entries with the
  Position paragraph, and eight area sections; each open item opens
  `- [ ] [tag] **Title.**` and closes with a `From:` line naming its origin heading.
  Closed text went to the archive under that heading with a one-line note, links
  re-based for `docs/`, and every non-blank original line was asserted to land exactly
  once by index. Ten mixed items were split at the sub-item; one closed outright.
- A tidy agent then made the promoted items that opened mid-sentence read as whole
  items, folded three duplicates into their survivors (the duplicate text moved to the
  archive), and rewrote the "below"/"in the P3 ledger" cross-references the move broke.
- docs-check gained the ledger-shape checks: tag, `From:` heading present in the
  archive, Front cap of ten with every entry naming an item, 40-line item limit, and an
  age histogram from each item's latest date.

- Three reviewers (Codex, an in-model reviewer, Qwen through the local server) read the
  result. Their findings drove a second tidy batch: five items still headlined as
  finished were retitled to their open scope, sub-items relettered, seven duplicated
  bold leads dropped, eleven archive note lines corrected, tags regraded on seven items,
  the rule text made consistent on the number of exits and where retired text goes,
  and the checker hardened (Front required, malformed items and finished headlines
  fail, in-place retirement fails, fences and CRLF handled, ages counted correctly).
- The first ten retirements happened the same day, on the owner's call: the chunked
  DeltaNet scan, YaRN long context, two span-48 anomalies and the span-2 window floor,
  the in-place decode scan, mid-message snapshots, the MLX comparison arm, and two PLE
  items promoted from the port. Each sits in the archive under `Retired: <area>`
  behind its dated reason and reopen condition.

Result: TODO.md 2116 → 1345 lines, 98 open items, Front 10/10; archive 858 → 2118
(the size moved from the open file to the archive, as intended). Twenty-one items
carry a latest date older than 30 days; the triage rule was written this arc and its
first pass over those items is the next ledger chore, not done here. That pass ran the
same day (below): `bun scripts/docs-check.ts` now prints `49 open items, front 10/10`,
and TODO.md is 844 lines against an archive of 2764.

## Pass three: the first triage and the intake rule

The regroup left 98 open items, and reading them by kind is what explained the number.
Ten were the Front's levers. Behind them sat roughly fourteen priced levers waiting for
a Front slot, seven instruments and open questions, fourteen chores, three decisions
nobody had made, nine features nobody had asked for, one item blocked upstream, and ten
outright duplicates of another item. The largest group was none of those: about
twenty-six items were "if this ever bites, the fix is X" notes, each a real observation
with a sketch of a fix, no number and no user.

Two causes, and only one of them is about writing. The first is the intake path: a
ledger item was the only exit a review finding had that was not silence, so every
finding became one, and a note about an edge nobody had hit ranked in the same list as a
measured 10% decode lever. The second is mechanical: the unit that closed was the item,
so an item that shipped in part but was never lettered stayed open in whole, and
finished text kept its place in the open file and went stale there.

The moves. 34 items and one lettered sub-item retired, each behind a dated reason and
the condition that reopens it; 10 duplicates folded into the partner item that already
owned the question, with a pointer line wherever the fold carried something the partner
did not say; 5 closed, two of them shipped long ago (the `<function=...>` tool-call
parser, the padded-vocabulary grammar trie), two settled by decision in this pass
(plane-less slots decode plain, temperature stays before the cut) and one answered by a
line in docs/benching.md. Every moved block went to
[docs/ledger-archive.md](../ledger-archive.md) verbatim, and the script asserted that
each non-blank line of the old TODO.md lands in exactly one of the two files.

Three of the nine unrequested features were kept on the owner's say, and they are being
built next rather than sitting as ledger text: the Claude Code agent-id field, the
cards' recommended penalties for the 35B-A3B, and the 128k context envelope. The chores
that survived need the GPU free rather than a decision, and they wait for a session that
has it.

The rule that came out of the pass is in AGENTS.md and restated in the TODO.md preamble:
a deferred scope or a review finding enters the ledger only with a number or a user
waiting for it, a chore only when the next arc is expected to do it, and everything else
is a "not taken now" line in the arc's record with a reopen condition. The record keeps
the sketch; the reopen condition is what makes it findable again.

## What is still weak

- The Front's ranking was set from the 2026-09-05 re-ranked ledger and today's position
  paragraph; it has not been re-priced since the router gemv landed.
- Areas were assigned by one reading. A misfiled item costs a move, not text.
- The 35B-A3B bytes-only ceiling in the Position paragraph is still a back-of-envelope
  ~200 tok/s; measuring it is an open item.
- The age histogram counts from the latest date in an item's text, so an item annotated
  today reads as fresh even if its work is old. That is the intended trade-off: the
  triage rule asks for a dated line saying why an item is still worth keeping.
