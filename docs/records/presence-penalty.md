# 2026-09-06 — Presence penalty: the cards' recipe, through the speculative verify, on by default

The 2026-08-19 refusal (decisions.md "Thinking budget and sampling controls") said the
penalty was sampler + verify + gate work as one unit. This is that unit.

## What shipped

vLLM's presence penalty: p subtracted from the logit of every distinct id the current
reply has emitted, prompt and earlier turns excluded, reasoning tokens included, before
temperature and before any cut, in greedy mode too. `SamplerOptions::presence_penalty`,
`PresenceHistory` in src/sampler.rs (emitted ids in order, per-id counts, an eager
unique-id list, a cached device id tensor rebuilt only when the unique set changes),
`SampleControl::presence`, and `SamplerOptions::recommended_for(model, thinking)` as the
one resolution helper. Defaults per checkpoint and mode in `Model::recommended_presence_penalty`:
1.5 non-thinking on all four checkpoints, thinking 1.5 on the 35B-A3B alone. Surfaces:
`--presence-penalty` on generate/chat/serve, `presence_penalty` in serve.toml (pins both
modes), the OpenAI field consumed, a native field, a batch payload field; the Anthropic
dialect has no field and takes the server default. `top_k` 0 is no top-k cut, 1 is
greedy, in the sampler and in batch's `select_option`.

Three draw shapes, one bus crossing each: no control, device softmax as before;
penalty-only on Metal, `Tensor::index_add` scatter then the same device softmax (candle's
`ia_u32_f32` kernel at rev 21cca0b, pinned bitwise against the CPU loop by a Metal unit
test); anything else, raw readback and the CPU chain with the penalty applied first.
Ten sampler sites in src/generate.rs feed the history; each speculative loop truncates it
to the emitted count at round end (generate.rs 2528, 2949, 3590), which is what keeps
rollback honest. The drafters propose unpenalized; only the target's verify draw is
penalized, llama.cpp's shape.

## Verification

`pmset -g` for the session: `lowpowermode 0`. Every run on a pinned build under
/tmp/xwen-pen, one model process at a time behind the session's GPU lock.

`cargo test --release`: lib 1141, parity 12, integration 69, all passing.

Greedy equivalence, 35B-A3B (the gate), every comparison identical and every one drafted:

| run | code prompt | chat prompt |
|---|---|---|
| shipped default (1.5, thinking) | identical, 131 drafted / 180 verified | identical, 119 drafted / 155 verified |
| `--presence-penalty 0` | identical, 159 drafted / 179 verified | identical, 108 drafted / 151 verified |

Speed and acceptance, 35B-A3B, code prompt, 256 tokens, three interleaved reps, medians:

| arm | decode tok/s | acceptance |
|---|---|---|
| plain, penalty 0 | 126.5 | |
| plain, penalty 1.5 | 126.9 | |
| drafted, penalty 0 | 121.1 | 63.0% |
| drafted, penalty 1.5 | 119.6 | 59.4% |

Plain is unchanged, which is the device scatter working. Drafted loses ~1.2% because the
proposals are unpenalized. Incidental and not a penalty effect: drafting reads below plain
on this prompt at penalty 0 too, which is a post-router-gemv fact and is ledgered.

Sampled mode, 35B-A3B (advisory on this checkpoint, AGENTS.md): identical at penalty 0
(360/384 and 420/448 drafted/verified), DIVERGED at 1.5 at lines 7 and 12. Greedy holding
at every verify position rules out a history bug. The measured mechanism: the plain arm
softmaxes on candle's Metal kernel and the controlled verify arm on the hand-rolled CPU
`softmax_in_place`, which differ at the ULP on nearly every entry (3772/4096 Metal vs
candle CPU; 4096/4096 against `softmax_in_place` on a synthetic row), and a constant
subtracted from many ids manufactures near-ties, where a ULP decides a draw.

Two non-thinking 35B replies at 1.5 and at 0 read as normal prose (the report has them).

## Not taken now

- **One softmax for both arms.** Routing the controlled branch through
  `candle_nn::ops::softmax_last_dim` would put plain and verify on the same numerics,
  at ~0.4% of a verify round, and would change every other controlled draw (grammar
  masks, the think-budget bias). Reopen if sampled-mode divergence on a shipped
  checkpoint is ever seed-independent at line 1, or if a consumer needs sampled
  draft/no-draft equivalence.
- **`frequency_penalty`, `repetition_penalty`, `min_p`** stay accept-and-drop. Reopen when
  a card recommends one or a client needs it.
- **The card values were not re-read from the HF cards in this arc**; the 2026-08-19
  ledger item cites the lines, and the table is in one place under one test.

Full agent report: /tmp/agent-report-penalties.md (session artifact, not in the repo).
