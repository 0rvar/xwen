# Process

One topic of [docs/decisions.md](../decisions.md), the index of decisions by topic; moved
here verbatim on 2026-09-06. Dated paragraphs, newest additions appended within their topic.


Inherited unchanged: multi-reviewer review with external model families on evidence
(reviewers recorded as wrong with disproofs, not just as right); a reviewer reads the
path you wrote, a live check walks the path you forgot; let the existing suite arbitrate
a proposed fix; docs drift is tracked work. Every shipped arc updates log.md (dated
entry) + README if the surface changed + this file if a decision was made, changed, or
refuted — a TODO.md update alone is not sufficient (2026-07-28).

**The ledger is a memory and a backlog, and the two have different rules (2026-09-06,
log.md "Docs restructured").** Six weeks of "scope is never silently dropped" plus
"items are never deleted" had grown TODO.md to 3200 lines with one exit, shipping, and an
open set growing by about two items a day. The fix separates the parts. The Front is the
backlog: at most ten ranked items, each priced against a ceiling in perf-state.md or
named by a waiting user, and promoting one demotes one. The area sections are the
ledger: complete, grouped by the part of the system an item touches rather than the arc
that deferred it, with a `From:` line keeping the provenance and a one-word state tag
(`measured`, `unpriced`, `blocked`, `small`) making the measured-versus-unpriced
distinction visible. Two things changed in the rules. The unit that closes is the item
OR the lettered sub-item, so a mixed item no longer keeps its shipped halves in the open
file. And there is a second exit beside shipping: retired, a dated reason plus a reopen
condition, moved to the archive like a closed item. Retired is deliberately weaker than
refuted: refuted lives only in these topic files, with evidence, and is the one state not to
relitigate without new evidence; retired means not planned and not forbidden, and the mandatory reopen
line is what stops an agent reading it as a verdict. Review findings not worth an item
are recorded in the record as "not taken now" with the same reopen shape, which cuts the
intake that grew the file. Triage is a forced choice, not an automatic deletion: at the
end of an arc and past 30 days, promote, re-date with a reason, or retire. The archive
stays verbatim, shipped text under the pre-regroup headings and retired text under
`Retired: <area>`; the open ledger may be regrouped, which
amends the earlier "never rewritten" rule for the open file only. The regroup itself ran
as a script over a per-item classification, asserting every original line landed exactly
once (record: [docs restructure](../records/docs-restructure.md)).
