//! One tokenizer and one grammar trie per VOCABULARY FAMILY, resolved from the
//! request's target rather than from a process global.
//!
//! Until Qwen3-4B arrived, a server had one vocabulary and could hold it in one
//! field: every checkpoint in the registry spoke Qwen 3.6's 248320 ids, the
//! tokenizer was compiled into the binary, and `constrain::shared()` built one
//! trie over it for the life of the process. A server can now hold two, and
//! nothing about one transfers to the other — not the ids, not the specials,
//! not the width. Encoding a prompt with the wrong one produces a request that
//! runs to completion and returns nonsense; masking with the wrong one produces
//! a grammar whose mask does not line up with the model's logits at all.
//!
//! So both follow the [`Target`], and they are cached per
//! [`VocabFamily`] rather than per checkpoint, because a family is exactly the
//! set of checkpoints for which they are the same object. Building a pair costs
//! ~150 ms (the trie dominates), which is why it is cached; it is built lazily,
//! because a server that never sees a request for the other family should never
//! pay for it.
//!
//! What this module does NOT do is decide which checkpoint answers a request.
//! That is [`super::resolve_requested_model`], and it has already run by the
//! time anything here is asked anything.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::{Context, Result};

use crate::constrain::ConstraintFactory;
use crate::hub::{self, Model, VocabFamily};
use crate::serve::types::Target;
use crate::tokenizer::LagunaTokenizer;

/// The two views of one vocabulary a served request needs: the tokenizer that
/// turns a conversation into ids, and the trie a schema is compiled against.
///
/// They are built together and handed out together because they must agree.
/// The trie is built over the SAME bytes the tokenizer parsed, so a token's id
/// means one thing to both; its stop set is that tokenizer's own `eog()`, so a
/// constrained reply is offered a stop the decode loop will act on; and its
/// width is the family's logit width, so the mask indexes the logits it is
/// applied to.
pub struct Vocabulary {
    family: VocabFamily,
    tokenizer: Arc<LagunaTokenizer>,
    grammars: Arc<ConstraintFactory>,
    /// Where the tokenizer came from, for the startup log and for errors.
    /// `None` for the family whose vocabulary is compiled into the binary.
    source: Option<PathBuf>,
}

impl Vocabulary {
    pub fn family(&self) -> VocabFamily {
        self.family
    }

    pub fn tokenizer(&self) -> &Arc<LagunaTokenizer> {
        &self.tokenizer
    }

    pub fn grammars(&self) -> &ConstraintFactory {
        &self.grammars
    }

    /// The file this vocabulary was read from, or `None` for the embedded one.
    pub fn source(&self) -> Option<&std::path::Path> {
        self.source.as_deref()
    }
}

impl std::fmt::Debug for Vocabulary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Vocabulary")
            .field("family", &self.family)
            .field("source", &self.source)
            .field("ids", &self.tokenizer.vocab_size())
            .finish()
    }
}

/// Every vocabulary this server may need, built on first use and kept.
///
/// Cheap to clone into `AppState` (it is held behind an `Arc` there) and safe
/// to share, and each family is built EXACTLY ONCE however many requests want
/// it at once.
///
/// Two locks doing two jobs. The outer mutex guards the map and is held only
/// long enough to hand back a family's cell — never across a build, so a
/// request for one family never waits on a build for the other. The cell is a
/// `OnceLock`, so the first caller into an empty one builds and the rest block
/// on it and take its result: no duplicated ~150 ms trie, and no window in
/// which two `Arc<Vocabulary>` exist for one family.
///
/// The build is synchronous on whatever thread asks, which on the HTTP path is
/// a Tokio worker. That is deliberate rather than `spawn_blocking`: every
/// family this server can reach is built at startup by [`Vocabularies::warm`],
/// so a request reaching a cold cell means a `tokenizer.json` appeared on disk
/// after the server started. Making the request path async to move a build
/// that should never happen off the worker would put an await point in
/// `submit`, which is sync all the way down.
pub struct Vocabularies {
    /// The tokenizer to prefer for the SERVED file's own family, when that file
    /// carries one. A server started on a safetensors directory is running that
    /// directory's tokenizer, not whatever copy of the same family happens to
    /// be in the hub cache — and possibly when no copy is.
    served: Option<(VocabFamily, PathBuf)>,
    /// One cell per family, created empty on first ask and filled once. The
    /// `String` is a rendered error: a `OnceLock` has to hold a `Clone`-able
    /// value, and a failed build is worth caching too — a family with no
    /// tokenizer on disk will not grow one mid-request, and rebuilding to fail
    /// again on every request would be the cost without the information.
    built: Mutex<HashMap<VocabFamily, Arc<OnceLock<Result<Arc<Vocabulary>, String>>>>>,
}

impl Vocabularies {
    /// The set of vocabularies a server serving `model_path` as `served` can
    /// reach.
    ///
    /// `model_path` is consulted, not opened: [`crate::checkpoint::tokenizer_beside`]
    /// looks for the file next to the checkpoint, where opening the checkpoint
    /// to ask would read eight gigabytes.
    pub fn new(model_path: &std::path::Path, served: Target) -> Self {
        let family = served.model.vocab_family();
        let served = crate::checkpoint::tokenizer_beside(model_path).map(|path| (family, path));
        Self {
            served,
            built: Mutex::new(HashMap::new()),
        }
    }

    /// A set with no served file to prefer — the embedded family only, plus
    /// whatever the hub cache holds for the others. For tests and for a server
    /// whose own file carries no tokenizer (every GGUF).
    pub fn hub_only() -> Self {
        Self {
            served: None,
            built: Mutex::new(HashMap::new()),
        }
    }

    /// The vocabulary a request running on `target` is encoded and masked with.
    pub fn for_target(&self, target: Target) -> Result<Arc<Vocabulary>> {
        self.for_family(target.model.vocab_family())
    }

    /// The vocabulary for one family, built on first use and once.
    pub fn for_family(&self, family: VocabFamily) -> Result<Arc<Vocabulary>> {
        // The map lock is released before the build: it only hands back the
        // cell. Two callers for two families never meet; two for one family
        // meet inside the cell, where exactly one of them builds.
        let cell = Arc::clone(self.lock().entry(family).or_default());
        match cell.get_or_init(|| {
            self.build(family)
                .map(Arc::new)
                .map_err(|error| format!("{error:#}"))
        }) {
            Ok(vocabulary) => Ok(Arc::clone(vocabulary)),
            Err(error) => Err(anyhow::anyhow!("{error}")),
        }
    }

    /// Build every family this server can reach, now rather than on the request
    /// that needs it.
    ///
    /// The SERVED checkpoint's is required: a missing or unreadable
    /// `tokenizer.json` there is a configuration mistake, and the first request
    /// is far too late to learn it — it would fail that request, and every retry
    /// after it, forever.
    ///
    /// Every OTHER family is best-effort, and warming them is what keeps the
    /// ~150 ms build off the request path entirely. A family whose tokenizer is
    /// not on this machine simply has none to build, which is not a reason to
    /// refuse to start: nothing on that family is selectable either
    /// (`checkpoint_selectable` wants the weights cached too), so the operator
    /// has lost nothing they had.
    pub fn warm(&self, served: Target) -> Result<()> {
        let family = served.model.vocab_family();
        self.for_family(family).with_context(|| {
            format!("loading the vocabulary {} speaks", served.model.full_name())
        })?;
        for other in [VocabFamily::Qwen36, VocabFamily::Qwen3] {
            if other != family {
                let _ = self.for_family(other);
            }
        }
        Ok(())
    }

    /// Whether this family has already been built — for the tests, and for a
    /// caller that wants to know whether asking would be free.
    pub fn is_built(&self, family: VocabFamily) -> bool {
        self.lock()
            .get(&family)
            .is_some_and(|cell| cell.get().is_some_and(Result::is_ok))
    }

    fn lock(
        &self,
    ) -> std::sync::MutexGuard<
        '_,
        HashMap<VocabFamily, Arc<OnceLock<Result<Arc<Vocabulary>, String>>>>,
    > {
        // A poisoned mutex here means a previous caller panicked while holding
        // a map of `Arc`s, which cannot leave a torn value: take it back rather
        // than making every later request fail for something already survived.
        self.built
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn build(&self, family: VocabFamily) -> Result<Vocabulary> {
        // The embedded copy IS the Qwen 3.6 family's vocabulary — the same
        // bytes its GGUFs carry — so that family needs no file, no search and
        // no second 12 MB parse. Every other family is read from disk, and a
        // family with nothing to read from is an error rather than a silent
        // fall back to the embedded one, which is the single worst outcome
        // available here: it would run and return nonsense.
        if family == VocabFamily::Qwen36 {
            let tokenizer = LagunaTokenizer::embedded()?;
            let grammars = crate::constrain::for_tokenizer(&tokenizer, family.logit_width())?;
            return Ok(Vocabulary {
                family,
                tokenizer: Arc::new(tokenizer),
                grammars,
                source: None,
            });
        }
        let path = self
            .tokenizer_source(family)
            .ok_or_else(|| missing_vocabulary(family))?;
        let tokenizer = LagunaTokenizer::from_file(&path)?;
        // Through `for_tokenizer` rather than by reading the file a second time
        // here: the trie is built over the very bytes this tokenizer was parsed
        // from, and that is a property of the tokenizer rather than of the
        // caller's bookkeeping.
        let grammars = crate::constrain::for_tokenizer(&tokenizer, family.logit_width())?;
        Ok(Vocabulary {
            family,
            tokenizer: Arc::new(tokenizer),
            grammars,
            source: Some(path),
        })
    }

    /// Where this family's `tokenizer.json` is, or `None` when this machine
    /// holds no copy of it. Only asked for a family that has a file at all —
    /// [`Vocabularies::build`] answers Qwen 3.6 from the embedded bytes before
    /// it gets here.
    ///
    /// Two places, in order. The SERVED file's own, when the server was
    /// started on a checkpoint of this family — that file is the one running,
    /// and it may be a directory the hub cache has never heard of. Then any
    /// cached registry checkpoint of the family, because the releases in a
    /// family ship a byte-identical `tokenizer.json` (verified for the three
    /// `qwen3` entries: the same sha256 in `Qwen/Qwen3-4B`,
    /// `Qwen/Qwen3-4B-Instruct-2507` and `Tongyi-MAI/Z-Image-Turbo`) and any of
    /// them is therefore the family's.
    fn tokenizer_source(&self, family: VocabFamily) -> Option<PathBuf> {
        if let Some((served_family, path)) = &self.served
            && *served_family == family
        {
            return Some(path.clone());
        }
        hub::MODELS
            .into_iter()
            .filter(|model| model.vocab_family() == family)
            .find_map(cached_tokenizer)
    }
}

impl std::fmt::Debug for Vocabularies {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Vocabularies")
            .field("served", &self.served)
            .field(
                "built",
                &[VocabFamily::Qwen36, VocabFamily::Qwen3]
                    .into_iter()
                    .filter(|family| self.is_built(*family))
                    .collect::<Vec<VocabFamily>>(),
            )
            .finish()
    }
}

/// This entry's `tokenizer.json` in the hub cache, or `None`. Offline — a
/// vocabulary is never downloaded to answer a request.
fn cached_tokenizer(model: Model) -> Option<PathBuf> {
    hub::cached_file(model.repo(), model.safetensors_tokenizer()?)
}

/// The error a family with no tokenizer on this machine produces, naming the
/// one command that puts one there.
pub(super) fn missing_vocabulary(family: VocabFamily) -> anyhow::Error {
    let names: Vec<String> = hub::MODELS
        .into_iter()
        .filter(|model| model.vocab_family() == family)
        .map(|model| format!("xwen fetch --model-size {model}"))
        .collect();
    anyhow::anyhow!(
        "no tokenizer for the {family:?} vocabulary is on this machine, and one is not \
         downloaded to answer a request; fetch a checkpoint that carries it ({})",
        names.join(", ")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The Qwen3 family needs a `tokenizer.json` on this machine; the tests that
    /// exercise two families skip themselves without one, because what they are
    /// about is the SWITCH and a machine with one checkpoint cannot show it.
    ///
    /// Absence is the ONLY thing this reports. A file that is present and does
    /// not build is the breakage these tests exist to catch, and it has to fail
    /// them rather than skip them.
    fn qwen3_available() -> bool {
        Vocabularies::hub_only()
            .tokenizer_source(VocabFamily::Qwen3)
            .is_some()
    }

    /// The two families are two different vocabularies, and the object that
    /// answers for one must never answer for the other.
    ///
    /// Asserted on the three things a request actually depends on — the ids a
    /// prompt encodes to, the stop id the decode loop watches for, and the width
    /// of the mask a schema produces — rather than on which file was opened,
    /// because opening the right file and using it wrongly looks identical from
    /// the outside.
    #[test]
    fn each_family_gets_its_own_ids_stops_and_mask_width() {
        if !qwen3_available() {
            eprintln!("skipping: no Qwen3 tokenizer in the Hugging Face cache");
            return;
        }
        let vocabs = Vocabularies::hub_only();
        let qwen36 = vocabs.for_family(VocabFamily::Qwen36).unwrap();
        let qwen3 = vocabs.for_family(VocabFamily::Qwen3).unwrap();

        // The same text, two different id sequences. Not merely "different
        // numbers": each has to round-trip through its OWN tokenizer, which is
        // what rules out one of them being nonsense.
        let text = "The quick brown fox jumps over the lazy dog.";
        let a = qwen36.tokenizer().encode(text).unwrap();
        let b = qwen3.tokenizer().encode(text).unwrap();
        assert_ne!(
            a, b,
            "two vocabularies cannot encode one string identically"
        );
        assert_eq!(qwen36.tokenizer().decode(&a).unwrap(), text);
        assert_eq!(qwen3.tokenizer().decode(&b).unwrap(), text);

        // The stop ids, which are what a decode loop watches for. A run that
        // took the other family's would run through every turn boundary and
        // read as a model that will not stop.
        assert_eq!(
            qwen36.tokenizer().specials().eog(),
            crate::tokenizer::LagunaTokenizer::EOG
        );
        assert_eq!(qwen3.tokenizer().specials().eog(), crate::qwen3::QWEN3_EOG);
        assert_ne!(
            qwen36.tokenizer().specials().im_end,
            qwen3.tokenizer().specials().im_end
        );

        // And the mask, which is the one that fails silently: a mask is indexed
        // by logit, so one built at another family's width lines up with nothing
        // the model produced.
        //
        // Two properties, not one number. It has to COVER its family's logits,
        // because the sampler treats everything past the mask's end as banned —
        // a short mask silently deletes the tail of the vocabulary. And it must
        // NOT stretch to the other family's width, which is what a borrowed trie
        // would do. The exact figure is the logit width rounded up to a whole
        // bitset word plus llguidance's own padding word, which is its business
        // and not a contract worth pinning.
        for (vocab, family) in [(&qwen36, VocabFamily::Qwen36), (&qwen3, VocabFamily::Qwen3)] {
            let width = mask_width(vocab);
            assert!(
                width >= family.logit_width(),
                "{family:?}: a {width}-wide mask does not cover {} logits",
                family.logit_width()
            );
            assert!(
                width < family.logit_width() + 64,
                "{family:?}: a {width}-wide mask is wider than {} logits plus padding",
                family.logit_width()
            );
        }
        assert!(
            mask_width(&qwen3) < VocabFamily::Qwen36.logit_width(),
            "the Qwen3 mask must not be wide enough to be the other family's"
        );
    }

    /// The width of the allow-mask this vocabulary's grammars produce, measured
    /// through the whole path a request takes: compile a schema, arm the state
    /// as an answering reply, ask for the mask.
    fn mask_width(vocab: &Vocabulary) -> usize {
        let schema = serde_json::json!({"type": "object", "properties": {}});
        let grammar = vocab
            .grammars()
            .compile(&schema)
            .expect("the schema compiles");
        let mut state = grammar.into_state(
            crate::chat::ThinkingEntry::Answer,
            *vocab.tokenizer().specials(),
        );
        let words = state
            .mask_words()
            .expect("an armed state masks")
            .expect("an armed state masks");
        words.len() * 32
    }

    /// One object per family, however many callers ask and in whatever order —
    /// the point of the cache, since building one costs ~150 ms.
    #[test]
    fn a_family_is_built_once_and_shared() {
        let vocabs = Vocabularies::hub_only();
        let first = vocabs.for_family(VocabFamily::Qwen36).unwrap();
        let second = vocabs.for_family(VocabFamily::Qwen36).unwrap();
        assert!(
            Arc::ptr_eq(&first, &second),
            "the second caller must get the first one's vocabulary"
        );
        // And the target-keyed entry point lands on the same object, since
        // every Qwen 3.6 checkpoint is one family.
        for model in [Model::Qwen27B, Model::Qwen35BA3B, Model::Qwen3827B] {
            let by_target = vocabs.for_target(Target::official(model)).unwrap();
            assert!(Arc::ptr_eq(&first, &by_target), "{model:?}");
        }
    }

    /// The served file's own tokenizer wins over any copy in the hub cache.
    ///
    /// A server started on a safetensors directory is running THAT directory's
    /// vocabulary. It may be a checkpoint the cache has never held, and it is
    /// the only file whose ids are certainly the ones the loaded weights were
    /// trained against.
    #[test]
    fn the_served_files_own_tokenizer_wins() {
        if !qwen3_available() {
            eprintln!("skipping: no Qwen3 tokenizer in the Hugging Face cache");
            return;
        }
        let cached = Vocabularies::hub_only()
            .tokenizer_source(VocabFamily::Qwen3)
            .unwrap();

        // A directory that is not in the cache, carrying the same tokenizer.
        let dir = std::env::temp_dir().join(format!("xwen_vocab_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), b"{}").unwrap();
        std::fs::copy(&cached, dir.join("tokenizer.json")).unwrap();

        let served = Target::served(Model::Qwen34B);
        let vocabs = Vocabularies::new(&dir, served);
        assert_eq!(
            vocabs.tokenizer_source(VocabFamily::Qwen3).as_deref(),
            Some(dir.join("tokenizer.json").as_path())
        );
        // The other family is unaffected: it is still the embedded one.
        assert_eq!(vocabs.tokenizer_source(VocabFamily::Qwen36), None);
        assert!(vocabs.for_target(served).is_ok());

        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A family with no tokenizer anywhere is an error naming the fetch, not a
    /// silent fall back to the embedded one.
    ///
    /// That fallback is the single worst outcome available here: it would build,
    /// run, and answer every request with tokens from someone else's vocabulary.
    #[test]
    fn a_family_with_no_tokenizer_says_so() {
        let error = missing_vocabulary(VocabFamily::Qwen3).to_string();
        assert!(error.contains("no tokenizer"), "{error}");
        assert!(
            error.contains("xwen fetch --model-size qwen3-4b"),
            "{error}"
        );
        // Never the embedded one under another family's name.
        let vocabs = Vocabularies::hub_only();
        let embedded = vocabs.for_family(VocabFamily::Qwen36).unwrap();
        assert_eq!(embedded.source(), None);
        assert_eq!(embedded.family(), VocabFamily::Qwen36);
    }
}
