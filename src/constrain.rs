//! Schema-constrained decoding: JSON Schema -> per-step token masks, via
//! llguidance.
//!
//! The shape mirrors the sampler seam it feeds: a [`ConstraintFactory`] is
//! built once per process (token trie + parser factory over the embedded
//! vocabulary, ~150 ms), [`ConstraintFactory::compile`] turns one request's
//! schema into a [`Grammar`] (~0.5 ms, so it runs on the serve HTTP thread
//! where a bad schema is still a 400), and [`GrammarState`] rides the job into
//! the decode loop, yielding an allow-bitmask before each draw and advancing on
//! each committed token.
//!
//! Two llguidance facts shape the API:
//!
//! - **Any `Err` from the matcher poisons it permanently** — every later call
//!   returns the same cached error. There is no in-place recovery, so errors
//!   here are request-fatal by design; the decode loop propagates them and the
//!   job fails.
//! - **EOS is offered only in an accepting state.** The trie is built with the
//!   EOG ids as its EOS set, so the mask itself holds the model inside the
//!   schema until the value is complete, and offers the stop ids exactly when
//!   stopping is legal. Control tokens are `0xFF`-prefixed in the trie
//!   (`toktrie`'s special-token marker), which no grammar byte can match: a
//!   masked draw can never produce `<think>`, `<tool_call>`, or any other
//!   added-vocabulary marker.
//!
//!   That last property does NOT come from the `special` flag in the
//!   vocabulary file — the thinking and tool markers carry `special: false`.
//!   `toktrie_hf_tokenizers` marks an added token special if that flag is set
//!   OR its text is `<…>`-shaped, and every marker in this vocabulary is
//!   angle-bracketed, so all 26 are covered. A future marker spelled without
//!   the brackets would be an ordinary text token that a grammar could emit
//!   inside a string; `tests::no_control_token_is_ever_offered` is what would
//!   catch that, and it is a compile-time property because the vocabulary is
//!   embedded rather than downloaded.
//!
//! The grammar constrains the ANSWER section only. While the model is inside
//! its `<think>` block the state is dormant (no mask, nothing consumed); it
//! arms itself when it sees `</think>` commit — the same activation edge
//! `ThinkBudget` uses, and correct under DFlash because activation rides
//! `on_committed`, which the verify walk calls per row in commit order.

use std::sync::{Arc, OnceLock};

use anyhow::{Result, anyhow, bail, ensure};
use llguidance::api::TopLevelGrammar;
use llguidance::{Matcher, ParserFactory};
use toktrie::{SimpleVob, TokEnv};
use toktrie_hf_tokenizers::{ByteTokenizer, ByteTokenizerEnv};

use crate::tokenizer::{EMBEDDED_TOKENIZER_JSON, LagunaTokenizer};

/// Token trie + parser factory over the embedded vocabulary. Build once and
/// share (`Arc`-clone is cheap; the factory itself is `Send + Sync`).
pub struct ConstraintFactory {
    factory: Arc<ParserFactory>,
}

/// The process-wide factory over the embedded vocabulary, built on first use
/// (~150 ms) and kept for the life of the process. The server always runs the
/// embedded tokenizer (`serve/engine.rs::load_tokenizer`), so this is always
/// the right trie for a served request.
pub fn shared() -> Result<&'static ConstraintFactory> {
    static SHARED: OnceLock<std::result::Result<ConstraintFactory, String>> = OnceLock::new();
    match SHARED.get_or_init(|| ConstraintFactory::embedded().map_err(|e| format!("{e:#}"))) {
        Ok(factory) => Ok(factory),
        Err(e) => Err(anyhow!(
            "constrain: the shared factory failed to build: {e}"
        )),
    }
}

impl ConstraintFactory {
    /// [`ConstraintFactory::new`] over the embedded snapshot, sized to the
    /// checkpoint's LOGIT width rather than the tokenizer's id space.
    ///
    /// The two differ (248320 vs 248070), and the mask is indexed by logit: the
    /// sampler refuses one narrower than the vocabulary, because a short mask
    /// silently bans every id past its end. Sizing to the wider number pads the
    /// trie's tail with placeholder specials, which no grammar byte can match —
    /// so the padded ids are unreachable by construction, which is what they
    /// should be.
    pub fn embedded() -> Result<Self> {
        Self::new(LagunaTokenizer::PADDED_VOCAB)
    }

    /// Builds the trie from the same embedded `tokenizer.json` bytes the real
    /// tokenizer parses, so the two views cannot drift. `expected_vocab` is the
    /// model's logit width; the mask must cover exactly that many ids.
    pub fn new(expected_vocab: usize) -> Result<Self> {
        Self::new_from(Some(expected_vocab))
    }

    fn new_from(expected_vocab: Option<usize>) -> Result<Self> {
        let mut bt = ByteTokenizer::from_json_bytes(EMBEDDED_TOKENIZER_JSON)
            .map_err(|e| anyhow!("constrain: building the grammar token trie failed: {e}"))?;
        // Chat ends on either EOG id, and llguidance's auto-detection settles
        // on one stop token. Naming both keeps the grammar offering a stop the
        // decode loop will actually act on; leaving it to the scan lets a
        // constrained run reach max_tokens with the value long since complete.
        bt.set_eos_tokens(&LagunaTokenizer::EOG);
        let env: TokEnv = ByteTokenizerEnv::new(bt, expected_vocab)
            .map_err(|e| anyhow!("constrain: sizing the token trie failed: {e}"))?
            .to_env();
        if let Some(expected) = expected_vocab {
            ensure!(
                env.tok_trie().vocab_size() == expected,
                "constrain: trie holds {} tokens but the model's vocabulary is {expected}",
                env.tok_trie().vocab_size(),
            );
        }
        let mut factory = ParserFactory::new_simple(&env)
            .map_err(|e| anyhow!("constrain: building the parser factory failed: {e}"))?;
        // llguidance logs straight to stderr by default. Under `serve` stderr
        // is the dashboard's frame stream; a raw write there is corruption, so
        // the library is silenced and errors travel through return values.
        factory.quiet();
        Ok(Self {
            factory: Arc::new(factory),
        })
    }

    /// Compiles a JSON schema into a ready matcher. Errors name the schema
    /// problem (unsupported keyword, unsatisfiable constraint) and are safe to
    /// echo into a 400 response.
    pub fn compile(&self, schema: &serde_json::Value) -> Result<Grammar> {
        let grammar = TopLevelGrammar::from_json_schema(schema.clone());
        let mut matcher = Matcher::new(self.factory.create_parser(grammar));
        if let Some(err) = matcher.get_error() {
            bail!("json_schema rejected: {err}");
        }
        let warnings = matcher.grammar_warnings();
        Ok(Grammar { matcher, warnings })
    }

    /// The unconstrained-shape variant: any JSON object (OpenAI's
    /// `response_format: {"type": "json_object"}`).
    pub fn compile_any_object(&self) -> Result<Grammar> {
        self.compile(&serde_json::json!({ "type": "object" }))
    }
}

/// A compiled, not-yet-started grammar. Produced on the HTTP thread, carried
/// by the job (`Matcher` is `Send`), armed in the decode loop.
pub struct Grammar {
    matcher: Matcher,
    warnings: Vec<String>,
}

impl Grammar {
    /// Schema-compilation warnings (soft schema problems llguidance noted while
    /// compiling). Worth logging; never fatal.
    pub fn warnings(&self) -> &[String] {
        &self.warnings
    }

    /// Turns the compiled grammar into decode-loop state. `in_thinking` says
    /// whether generation starts inside a `<think>` block; if so the state
    /// stays dormant until `</think>` commits.
    pub fn into_state(self, in_thinking: bool) -> GrammarState {
        GrammarState {
            matcher: self.matcher,
            active: !in_thinking,
            done: false,
            mask: None,
        }
    }
}

/// Per-request grammar cursor. The decode loop calls [`mask_words`] before a
/// draw and [`on_committed`] after one; [`is_done`] reports that the value is
/// complete and generation should stop (the mask never offers a token past the
/// end, so continuing would only sample unconstrained trailing text).
///
/// [`mask_words`]: GrammarState::mask_words
/// [`on_committed`]: GrammarState::on_committed
/// [`is_done`]: GrammarState::is_done
pub struct GrammarState {
    matcher: Matcher,
    /// False while the model is still thinking; nothing is masked or consumed
    /// until `</think>` commits.
    active: bool,
    /// The grammar reached a state it cannot extend: the value is complete.
    done: bool,
    /// The most recent mask, held so the sampler can borrow its words.
    mask: Option<SimpleVob>,
}

impl GrammarState {
    /// The allow-bitmask for the next draw (bit `t` set = token `t` legal), or
    /// `None` while dormant. Packed 32-bit words, index `t / 32`, LSB-first —
    /// the layout `SampleControl::allowed` applies.
    pub fn mask_words(&mut self) -> Result<Option<&[u32]>> {
        if !self.active || self.done {
            return Ok(None);
        }
        let vob = self
            .matcher
            .compute_mask()
            .map_err(|e| anyhow!("constrain: grammar mask failed: {e}"))?;
        self.mask = Some(vob);
        Ok(self.mask.as_ref().map(|m| m.as_slice()))
    }

    /// Feed an already-rendered response prefix into the matcher before decoding
    /// starts, so the first mask continues that document instead of opening a
    /// second one at the root. The tokens come from our own tokenizer, so the
    /// ids are in-vocab.
    ///
    /// `try_consume_tokens` is what makes a bad prefix reportable: it stops at
    /// the first token the grammar refuses and leaves the matcher healthy
    /// holding exactly the accepted ones, where `consume_token`'s `Err` would
    /// poison it for good.
    pub fn consume_prefix(&mut self, tokens: &[u32]) -> Result<()> {
        ensure!(
            self.active && !self.done,
            "constrain: a response prefix can only be fed to a live grammar"
        );
        let consumed = self
            .matcher
            .try_consume_tokens(tokens)
            .map_err(|e| anyhow!("constrain: feeding the response prefix failed: {e}"))?;
        ensure!(
            consumed == tokens.len(),
            "the prefix leaves the schema at token {consumed} of {}",
            tokens.len()
        );
        // A grammar that is already complete would decode UNCONSTRAINED — the
        // mask is `None` once done — so a prefix that finishes the document has
        // nothing left to constrain and is refused rather than served that way.
        ensure!(
            !self.matcher.is_stopped(),
            "the prefix already completes the document, leaving nothing to generate"
        );
        Ok(())
    }

    /// Observe a committed token, in commit order. Dormant: watches for
    /// `</think>` and arms itself. Active: advances the grammar; when the
    /// grammar can no longer be extended the state flips to done. EOG ids are
    /// never fed to the matcher — the loops break on them, and the mask only
    /// offers them when stopping is already legal.
    pub fn on_committed(&mut self, token: u32) -> Result<()> {
        if self.done || LagunaTokenizer::EOG.contains(&token) {
            return Ok(());
        }
        if !self.active {
            if token == LagunaTokenizer::THINK_CLOSE {
                self.active = true;
            }
            return Ok(());
        }
        self.matcher
            .consume_token(token)
            .map_err(|e| anyhow!("constrain: grammar rejected committed token {token}: {e}"))?;
        if self.matcher.is_stopped() {
            self.done = true;
        }
        Ok(())
    }

    /// True once the constrained value is complete. The caller stops decoding:
    /// there is nothing legal left to draw except an EOG, and emitting one is
    /// the loop's job, not the grammar's.
    pub fn is_done(&self) -> bool {
        self.done
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::OnceLock;

    /// One factory for the whole test module: the trie + factory build costs
    /// ~150 ms and is stateless across compiles. Built exactly as the served
    /// one is, so the masks under test are the masks the sampler would see.
    fn factory() -> &'static ConstraintFactory {
        static FACTORY: OnceLock<ConstraintFactory> = OnceLock::new();
        FACTORY.get_or_init(|| ConstraintFactory::embedded().unwrap())
    }

    fn schema() -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" },
                "count": { "type": "integer" }
            },
            "required": ["name", "count"],
            "additionalProperties": false
        })
    }

    fn bit(words: &[u32], id: u32) -> bool {
        words
            .get((id / 32) as usize)
            .is_some_and(|w| w & (1 << (id % 32)) != 0)
    }

    // A canonical tokenization of a schema-valid document is accepted token by
    // token: every id is in the mask before its draw, and the grammar reports
    // completion exactly at the closing brace.
    #[test]
    fn walks_a_valid_document_and_completes() {
        let tok = LagunaTokenizer::embedded().unwrap();
        let ids = tok.encode("{\"name\": \"laguna\", \"count\": 3}").unwrap();
        let mut state = factory().compile(&schema()).unwrap().into_state(false);
        for (i, &id) in ids.iter().enumerate() {
            let words = state
                .mask_words()
                .unwrap()
                .expect("active grammar yields a mask");
            assert!(bit(words, id), "token {id} (step {i}) missing from mask");
            // Mid-document the stop ids stay masked out: the model cannot end
            // the turn on an incomplete value.
            if i + 1 < ids.len() {
                for eog in LagunaTokenizer::EOG {
                    assert!(!bit(words, eog), "EOG {eog} offered mid-document");
                }
            }
            // Control tokens are structurally unmaskable.
            assert!(
                !bit(words, LagunaTokenizer::THINK_OPEN),
                "<think> offered by the mask"
            );
            assert!(
                !bit(words, LagunaTokenizer::TOOL_CALL_OPEN),
                "<tool_call> offered by the mask"
            );
            state.on_committed(id).unwrap();
        }
        assert!(state.is_done(), "grammar not complete after the full value");
        assert!(
            state.mask_words().unwrap().is_none(),
            "done state still masks"
        );
    }

    // The mask is indexed by LOGIT, so it has to span the model's output layer
    // and not merely the tokenizer's id space. A mask short of the vocabulary is
    // refused by the sampler, because the ids past its end would be banned by
    // omission rather than by the grammar.
    #[test]
    fn the_mask_spans_the_models_logit_width() {
        let tok = LagunaTokenizer::embedded().unwrap();
        assert!(LagunaTokenizer::PADDED_VOCAB > tok.vocab_size());
        let mut state = factory().compile(&schema()).unwrap().into_state(false);
        let words = state
            .mask_words()
            .unwrap()
            .expect("an active grammar masks");
        assert!(
            words.len() * 32 >= LagunaTokenizer::PADDED_VOCAB,
            "mask covers {} ids but the model's logits hold {}",
            words.len() * 32,
            LagunaTokenizer::PADDED_VOCAB
        );
    }

    // No id past the last ordinary text token may ever be drawn under a
    // grammar: not a chat marker, not a thinking or tool marker, and not one of
    // the padded ids that have no text behind them at all. The interesting
    // position is inside a JSON string, where the grammar accepts almost the
    // whole vocabulary and the literal bytes `<think>` would be valid content —
    // the markers are excluded there because the trie holds them as specials,
    // not because their bytes are illegal.
    #[test]
    fn no_control_token_is_ever_offered() {
        let tok = LagunaTokenizer::embedded().unwrap();
        let mut state = factory().compile(&schema()).unwrap().into_state(false);
        let ids = tok.encode("{\"name\": \"laguna\", \"count\": 3}").unwrap();
        let mut widest = 0;
        for (step, &id) in ids.iter().enumerate() {
            let words = state
                .mask_words()
                .unwrap()
                .expect("an active grammar masks");
            widest = widest.max(words.iter().map(|w| w.count_ones() as usize).sum());
            for control in LagunaTokenizer::ENDOFTEXT as usize..LagunaTokenizer::PADDED_VOCAB {
                assert!(
                    !bit(words, control as u32),
                    "id {control} ({:?}) offered at step {step}",
                    tok.id_to_token(control as u32)
                );
            }
            state.on_committed(id).unwrap();
        }
        // The sweep is only meaningful if it ran against a permissive mask: a
        // string body accepts most of the vocabulary, so if this collapses the
        // test above has stopped proving anything.
        assert!(
            widest > tok.vocab_size() / 2,
            "the widest mask offered only {widest} ids"
        );
    }

    // A grammar-invalid continuation is absent from the mask, and the matcher
    // (fed only mask-approved ids) never sees it.
    #[test]
    fn mask_excludes_invalid_continuations() {
        let tok = LagunaTokenizer::embedded().unwrap();
        let open = tok.encode("{").unwrap();
        let close_bracket = tok.encode("]").unwrap();
        assert_eq!(open.len(), 1);
        assert_eq!(close_bracket.len(), 1);
        let mut state = factory().compile(&schema()).unwrap().into_state(false);
        let words = state.mask_words().unwrap().unwrap();
        assert!(bit(words, open[0]), "opening brace must be legal");
        assert!(!bit(words, close_bracket[0]), "']' cannot open an object");
    }

    // Dormant until `</think>` commits: no mask during thinking, and thinking
    // tokens are not fed to the matcher.
    #[test]
    fn stays_dormant_until_think_close() {
        let tok = LagunaTokenizer::embedded().unwrap();
        let mut state = factory().compile(&schema()).unwrap().into_state(true);
        assert!(state.mask_words().unwrap().is_none());
        // Arbitrary thinking-text tokens pass through without touching the
        // grammar.
        for id in tok.encode("let me reason about ] this } first").unwrap() {
            state.on_committed(id).unwrap();
        }
        assert!(state.mask_words().unwrap().is_none());
        state.on_committed(LagunaTokenizer::THINK_CLOSE).unwrap();
        let words = state.mask_words().unwrap().expect("armed after </think>");
        assert!(bit(words, tok.encode("{").unwrap()[0]));
    }

    // A prefix the caller already put in the prompt is consumed before the first
    // draw, and the grammar carries on from there: the mask continues the
    // document mid-string rather than offering a fresh one, and the rest of the
    // value walks to completion.
    #[test]
    fn consume_prefix_continues_the_document() {
        let tok = LagunaTokenizer::embedded().unwrap();
        let mut state = factory().compile(&schema()).unwrap().into_state(false);
        state
            .consume_prefix(&tok.encode("{\"name\": \"laguna\",").unwrap())
            .unwrap();

        let words = state.mask_words().unwrap().expect("live after the prefix");
        assert!(
            !bit(words, tok.encode("]").unwrap()[0]),
            "']' cannot continue an object"
        );
        for eog in LagunaTokenizer::EOG {
            assert!(!bit(words, eog), "EOG {eog} offered on an incomplete value");
        }
        for id in tok.encode("\"count\": 3}").unwrap() {
            let words = state.mask_words().unwrap().expect("live mid-document");
            assert!(bit(words, id), "token {id} missing from the mask");
            state.on_committed(id).unwrap();
        }
        assert!(state.is_done(), "the value never completed");
    }

    // A prefix that leaves the schema names the token it diverged at, and says
    // so as an error rather than by masking the whole vocabulary out later.
    #[test]
    fn an_invalid_prefix_names_where_it_diverged() {
        let tok = LagunaTokenizer::embedded().unwrap();
        let mut state = factory().compile(&schema()).unwrap().into_state(false);
        let error = state
            .consume_prefix(&tok.encode("{\"name\": [1").unwrap())
            .expect_err("an array is not a string");
        let msg = error.to_string();
        assert!(msg.contains("leaves the schema at token"), "{msg}");
    }

    // A prefix holding the whole value is refused: a grammar that is already
    // done masks nothing, so decoding on would be unconstrained.
    #[test]
    fn a_prefix_that_completes_the_document_is_refused() {
        let tok = LagunaTokenizer::embedded().unwrap();
        let mut state = factory().compile(&schema()).unwrap().into_state(false);
        let error = state
            .consume_prefix(&tok.encode("{\"name\": \"laguna\", \"count\": 3}").unwrap())
            .expect_err("a complete value leaves nothing to generate");
        assert!(error.to_string().contains("already completes"), "{error}");
    }

    // Feeding a prefix to a state still waiting on `</think>` is a caller bug:
    // the text would be consumed as if the answer had started.
    #[test]
    fn consume_prefix_on_a_dormant_state_errors() {
        let tok = LagunaTokenizer::embedded().unwrap();
        let mut state = factory().compile(&schema()).unwrap().into_state(true);
        let error = state
            .consume_prefix(&tok.encode("{").unwrap())
            .expect_err("a dormant grammar has nothing to consume with");
        assert!(error.to_string().contains("live grammar"), "{error}");
    }

    // Unsupported schema keywords fail at compile time with a message safe to
    // put in a 400, not at decode time.
    #[test]
    fn unsupported_schema_errors_at_compile() {
        let err = factory()
            .compile(&serde_json::json!({ "not": { "type": "string" } }))
            .err()
            .expect("`not` is unsupported and must be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("json_schema rejected"),
            "unexpected error: {msg}"
        );
    }

    // The json_object variant accepts any object shape.
    #[test]
    fn any_object_grammar_walks_arbitrary_objects() {
        let tok = LagunaTokenizer::embedded().unwrap();
        let ids = tok
            .encode("{\"anything\": [1, {\"nested\": null}]}")
            .unwrap();
        let mut state = factory().compile_any_object().unwrap().into_state(false);
        for &id in &ids {
            let words = state.mask_words().unwrap().unwrap();
            assert!(bit(words, id), "token {id} missing from json_object mask");
            state.on_committed(id).unwrap();
        }
        assert!(state.is_done());
    }
}
