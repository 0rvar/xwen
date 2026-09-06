# 2026-08-14 — Review round: a job now names a FILE, not just a checkpoint; drafting is resolved per checkpoint; a contradicting `--model-size` fails at startup

Moved verbatim from [docs/log.md](../log.md) on 2026-09-06; the log keeps this entry's
opening paragraph and links here.


Two reviews (Claude and Codex/gpt-5.6-sol) of the entry below found one shared root
behind most of their findings: the arc had introduced a second kind of model identity
(the served file's own id) without giving the engine a way to represent it. Everything
here follows from fixing that.

**A job now names a `Target` — a checkpoint plus "is this the served file" — instead of
a bare `Model`.** On a server started with a GGUF that identifies as none of the official
checkpoints, the official checkpoint of the same architecture is a DIFFERENT FILE with
the same name for sizing, and `Model` alone could not say which was meant. The
consequences the reviewers found, all now fixed: `checkpoint_paths` short-circuited to
the served file whenever `size == default_size`, so an official name was answered by
unchecked weights; the id `/v1/models` advertised for a custom GGUF was refused by the
resolver that was supposed to accept it (400 on the one id it published); and the batch
handler labeled such a run with the arch-fallback's full name, so the response document
claimed official weights had run. Equality on `Target` is file identity, which is also
what the engine's swap check and the disk tier's binding actually wanted — the tier is
bound to `settings.model`, not to a checkpoint id, and now compares as such.

**`--model-size` is a tie-break, not an override.** It settled a file that identifies as
nothing; against a file that identifies itself it silently won and then made
`EngineState::load`'s own checks fail on every request — a server that starts clean and
500s forever. It is now a startup error naming both sides, and must also agree with the
architecture. The load-time checks stay as a backstop for a file that changed under a
running server.

**Drafting is per checkpoint (promoted from a ledger item).** `ServeSettings.draft` was
one resolved `Option<PathBuf>`, so a server whose DEFAULT checkpoint ships no sidecar ran
every other checkpoint plain as well — invisibly, and worth -46 to -52% on the 27B. It is
now a `DraftMode` (`Off` / `Official` / `Custom`), resolved when each checkpoint loads.
The TUI's drafting cell follows the loaded checkpoint rather than the setting, for the
same reason. `xwen serve --draft official` on a sidecar-less checkpoint now errors like
the one-shot commands do instead of degrading quietly.

**`identify`'s file-name branch was dead code, and its live half was too loose.** It
matched `Path::file_name` (extension included) against full names, so nothing ever
matched exactly and only a bare `3.6`/`3.8` substring test fired — which would have
identified `My-Qwen3.6-14B-finetune.gguf` as the official 27B. File names are now matched
on the full name as a case-insensitive substring of the STEM, the loose release-substring
pass is kept only for `general.name` (which the converter wrote about the model it
holds), and a name matching more than one checkpoint identifies as none rather than as
whichever the table lists first.

**Also fixed:** `/v1/messages/count_tokens` ignored its `model` field, which made the
"every surface refuses an unknown model" claim false — it validates now (the count itself
is checkpoint-independent, since every checkpoint shares a tokenizer);
`retune-draft.ts --dry-run` printed `--draft-p-min undefined` for a sidecar-less
checkpoint because the drafter guard sat after the dry-run early return, and
`SHIPPED_P_MIN` is now `Partial` with a checked accessor rather than a `Record` missing a
key; two tests named `..._echoed_verbatim` asserted strings that can no longer reach
`prepare`, and now pin the real contract; the fallback drafting floor's comments stop
calling the 35B's fitted 0.3 "the shared base" and say what it is.

**New tests.** The dialect handlers are now driven directly (`#[tokio::test]` over
`probe_state`): an SDK id and both CLI aliases 400 in each dialect's own envelope with
every valid name in the message, nothing reaches the queue, a case-insensitive full name
does reach it as the right `Target`, and `count_tokens` behaves the same. `identify` gains
the blessed stems, an ambiguous name, and `My-Qwen3.6-14B-finetune.gguf`. `/v1/models`'s
test now validates EVERY listed id through the resolver, first entry included — the entry
it previously skipped is exactly where the custom-GGUF bug lived. One trap worth
recording: a non-streaming handler test hangs forever against a probe queue (the handler
waits for engine events that never come), so the servable-model case asks for a stream,
which returns as soon as the job is queued.

**Second pass over the same round.** A verification pass confirmed the ten findings above
closed and found four more, two of them introduced by the fixes themselves — worth
recording, because both were the same mistake in different clothes: a rule stated in a
comment and not checked anywhere.

**The dashboard's drafting cell was pinned OFF on every drafting server.** The cell was
made to follow the loaded checkpoint by clearing on `ModelLoaded` and setting on
`DrafterLoaded` — but `DrafterLoaded` is emitted from `attach_drafter` INSIDE
`EngineState::load` while `ModelLoaded` is emitted after that load returns, so the
clearing arm always ran last. Confirmed in a live log: `drafter loaded in 0.5s` then
`model loaded in 4.0s`. The fix does not reorder the engine (that timing is
`ModelLoaded`'s meaning): the cell is now decided only by the two drafter events and RESET
by the events that end a residency (`CheckpointSwappingOut`, `IdleUnloaded`), falling back
to a new `draft_configured` when nothing is loaded. Ordering-independent by construction.
The old test fed the events in an order the engine never produces and asserted nothing
about the cell; the new one feeds the real order through a full swap cycle and asserts the
rendered header at each step.

**The served checkpoint's official sidecar lost its startup preflight** when `validate_model`
was narrowed to custom drafters — reintroducing exactly the 500-per-request class this
round set out to kill, because "official" does not mean "fits": a custom GGUF served as
its architecture's checkpoint gets that checkpoint's sidecar. Startup now judges the
served target's official sidecar too (offline, from what the CLI's prefetch left in the
cache); other checkpoints keep attach-time checking, since their sidecars may not be
downloaded yet. Verified by manufacturing the case — an APFS clone of the 27B with
`general.name` blanked to `mymodel-27x` and `embedding_length` patched 5120 → 4096 — which
now refuses at startup with `the drafter has a hidden size of 5120 but the target has
4096` instead of starting and failing every request. The old test claimed to pin this and
only exercised `check_against_target` as a pure function; it now drives `validate_model`
itself, alongside a test of which drafter startup selects in each mode.

**`general.name`'s loose matching went too.** It kept a bare release-substring pass
("3.6"/"3.8") that the file-name pass had already been tightened away from — and since
`Arch::Moe` has exactly one candidate, no ambiguity check could save `MyMoE-3.6` from
becoming `Qwen3.6-35B-A3B` and answering official-name requests with unchecked weights.
Both sources now use one rule: exact full name, or a whole full name found inside the
name. The blessed files are unaffected (their `general.name` IS the exact full name,
verified on all three).

**Docs corrected where they had gone stale:** CLAUDE.md's cheat-sheet line and a
decisions.md paragraph still described the old "explicit `--model-size` first" precedence;
the flag is a cross-check that must agree with a file that identifies itself. The
decisions.md paragraph carries a SUPERSEDED note per that file's convention. The batch
label round-trip comment no longer claims a resubmit always works — it does not for a
custom server's own id, which names no checkpoint.

**Verification.** `cargo build --release`, `cargo fmt --check`, `cargo test --release`:
873 passed, 0 failed. Live: the full swap cycle above (35B drafting → 3.8 plain → 35B
drafting) with the event order captured; the manufactured preflight failure; and the serve
checks in the entry below, re-run after these fixes. One limitation stated plainly: the
TUI cell itself was verified through its rendering test (a real frame, asserting
`draft ON`/`draft off`) plus the live event order, not by reading a live dashboard — a
headless pty gives ratatui a zero-size terminal, so no frame text can be captured here.
