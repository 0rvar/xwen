//! Wrapper around the HF `tokenizers` runtime for the Qwen BPE vocab.
//!
//! There is no BOS token and nothing is ever prepended to a prompt: the
//! conversation begins with the chat template's first `<|im_start|>`. A turn
//! ends on either end-of-generation id — `<|im_end|>` closes a chat turn and
//! `<|endoftext|>` ends the document — and a loop that watches only one of them
//! decodes straight through the turn boundary.
//!
//! The chat template (see `chat.rs`) emits every structural marker as literal
//! text: `<|im_start|>`, `<|im_end|>`, `<think>`/`</think>`, `<tool_call>` and
//! `<tool_response>` with their closers. All of them are entries in the
//! tokenizer's added vocabulary, so `encode` maps each to its single id — the
//! ChatML control ids the model was trained on. Note that the thinking and tool
//! markers carry `special: false` in the vocabulary file; that flag governs
//! rendering, not matching, so they resolve to their ids exactly like the
//! `<|…|>` specials do.
//!
//! That mapping is a capability the model's own template deserves but client
//! content must not have: a user message containing the literal text `<think>`
//! would otherwise inject a real control token. `encode_prompt` takes the byte
//! ranges of client content (reported by `chat::build_prompt_parts_with_spans`)
//! and encodes any added-token string found there as plain byte-level BPE
//! instead.

use std::ops::Range;
use std::path::Path;
use std::sync::OnceLock;

use anyhow::{Result, anyhow};

/// The checkpoint `tokenizer.json`, embedded verbatim. Shared with
/// `constrain.rs`, which parses it a second time through llguidance's own
/// (older) `tokenizers` copy to build the grammar token trie — sharing the
/// bytes keeps the two views pinned to the same snapshot and embeds the file
/// once.
pub(crate) static EMBEDDED_TOKENIZER_JSON: &[u8] = include_bytes!("../reference/tokenizer.json");

/// The byte-level BPE alphabet, inverted: each of the 256 printable stand-in
/// characters a BPE vocab entry spells a byte with, mapped back to that byte.
/// This is GPT-2's `bytes_to_unicode` reversed — the three printable Latin-1
/// ranges (`!`..=`~`, `¡`..=`¬`, `®`..=`ÿ`) stand for themselves, and every
/// other byte takes the next code point from U+0100 up, in ascending byte
/// order (`Ġ` is space, `Ċ` is newline). Qwen's tokenizer is byte-level BPE,
/// so every non-added vocab entry is a string over exactly this alphabet.
fn byte_level_inverse() -> std::collections::HashMap<char, u8> {
    let mut inverse = std::collections::HashMap::with_capacity(256);
    let mut fallback = 0x100u32;
    for byte in 0u32..256 {
        let kept = matches!(byte, 0x21..=0x7E | 0xA1..=0xAC | 0xAE..=0xFF);
        let ch = if kept {
            char::from_u32(byte).expect("Latin-1 range is valid chars")
        } else {
            let ch = char::from_u32(fallback).expect("U+0100.. is valid chars");
            fallback += 1;
            ch
        };
        inverse.insert(ch, byte as u8);
    }
    inverse
}

pub struct LagunaTokenizer {
    inner: tokenizers::Tokenizer,
    /// Added-token strings with their ids, sorted longest-first so a scan that
    /// takes the first hit at a position implements the same leftmost-longest
    /// discipline as the added-vocabulary matcher inside `inner`.
    markers: Vec<(String, u32)>,
    /// Bytes any marker starts with, so the scan skips most positions cheaply.
    marker_first_bytes: [bool; 256],
    /// Lazily built copy of `inner` with an empty added vocabulary: pure
    /// byte-level BPE, under which no added-token string is reachable (no merge
    /// path builds one — `plain_bpe_cannot_reach_any_added_token_id` pins it).
    plain: OnceLock<tokenizers::Tokenizer>,
    /// Lazily built raw BYTES of every encodable id, for sweeps that classify
    /// the whole vocabulary by token content (the scored batch path's escape
    /// measure). Bytes rather than decoded strings deliberately: byte-level
    /// BPE tokens are free to hold part of a multi-byte UTF-8 character, and
    /// `decode` on such an id is lossy (U+FFFD), which would misclassify the
    /// canonical opener of any non-ASCII option value. Built once per
    /// tokenizer on first use.
    decoded: OnceLock<Vec<Vec<u8>>>,
}

impl LagunaTokenizer {
    /// `<|endoftext|>` — the document terminator, and the id used wherever a
    /// slot must be filled with something inert (padding, an out-of-range
    /// fallback). It is one of the two end-of-generation ids.
    pub const ENDOFTEXT: u32 = 248_044;
    /// Padding / fallback id. The vocabulary has no dedicated pad entry, so
    /// `<|endoftext|>` serves.
    pub const PAD: u32 = Self::ENDOFTEXT;
    /// `<|im_start|>` — opens a ChatML turn header.
    pub const IM_START: u32 = 248_045;
    /// `<|im_end|>` — closes a ChatML turn.
    pub const IM_END: u32 = 248_046;
    /// End-of-generation tokens: `<|im_end|>` ends the assistant's turn,
    /// `<|endoftext|>` ends the document. Generation stops on EITHER; watching
    /// only one lets decoding run past the turn boundary and continue the
    /// conversation on its own.
    pub const EOG: [u32; 2] = [Self::IM_END, Self::ENDOFTEXT];
    /// `<think>` — opens the reasoning block. The generation prompt already
    /// contains it, so the model does not emit it; it is here for code that
    /// recognizes a replayed reasoning span by id.
    pub const THINK_OPEN: u32 = 248_068;
    /// `</think>` — closes the reasoning block the generation prompt opens with
    /// `<|im_start|>assistant\n<think>\n`. A single added token, so generation
    /// code can gate on it by id (e.g. to force a minimum reasoning length).
    pub const THINK_CLOSE: u32 = 248_069;
    /// `<tool_call>` / `</tool_call>` — wrap one function call the assistant
    /// writes. The call's own body (`<function=…>`, `<parameter=…>`) is
    /// ordinary text, not vocabulary entries.
    pub const TOOL_CALL_OPEN: u32 = 248_058;
    pub const TOOL_CALL_CLOSE: u32 = 248_059;
    /// `<tool_response>` / `</tool_response>` — wrap one tool result inside the
    /// user turn that carries results back to the model.
    pub const TOOL_RESPONSE_OPEN: u32 = 248_066;
    pub const TOOL_RESPONSE_CLOSE: u32 = 248_067;

    /// The checkpoint tokenizer, compiled into the binary from
    /// `reference/tokenizer.json` at build time. This is the default vocabulary
    /// everywhere (the binary runs from any working directory, offline), and it
    /// pins the exact bytes the parity fixtures were validated against. An
    /// upstream repo can revise its tokenizer file in place — a flipped
    /// `special` flag alone changes how a prompt tokenizes — so the vocabulary
    /// is a compile-time input, not a runtime download. Changing the embedded
    /// file changes the token stream: bump `chat::TOKENIZATION_RULES_VERSION`.
    pub fn embedded() -> Result<Self> {
        let inner = tokenizers::Tokenizer::from_bytes(EMBEDDED_TOKENIZER_JSON)
            .map_err(|e| anyhow!("failed to parse the embedded tokenizer: {e}"))?;
        Ok(Self::from_inner(inner))
    }

    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let inner = tokenizers::Tokenizer::from_file(path.as_ref())
            .map_err(|e| anyhow!("failed to load tokenizer from {:?}: {e}", path.as_ref()))?;
        Ok(Self::from_inner(inner))
    }

    fn from_inner(inner: tokenizers::Tokenizer) -> Self {
        let mut markers: Vec<(String, u32)> = inner
            .get_added_vocabulary()
            .get_vocab()
            .iter()
            .map(|(text, &id)| (text.clone(), id))
            .collect();
        // Longest first (ties broken lexically for determinism): the scanner
        // takes the first match at a position, which must be the longest one.
        markers.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then_with(|| a.0.cmp(&b.0)));
        let mut marker_first_bytes = [false; 256];
        for (text, _) in &markers {
            if let Some(&byte) = text.as_bytes().first() {
                marker_first_bytes[byte as usize] = true;
            }
        }
        Self {
            inner,
            markers,
            marker_first_bytes,
            plain: OnceLock::new(),
            decoded: OnceLock::new(),
        }
    }

    /// Load the vocab embedded in the GGUF metadata (no tokenizer.json needed).
    pub fn from_gguf(content: &candle_core::quantized::gguf_file::Content) -> Result<Self> {
        let _ = content;
        // Reconstructing the byte-level BPE model (merges, byte-level pre-tokenizer
        // and decoder, and the 26-entry added vocabulary) from the flat
        // `tokenizer.ggml.*` arrays would duplicate a large slice of the
        // `tokenizers` builders for no parity benefit while `tokenizer.json` ships
        // alongside every checkpoint. Use `--tokenizer <tokenizer.json>` instead.
        Err(anyhow!(
            "building the tokenizer from GGUF metadata is not supported; \
             pass the tokenizer.json path via --tokenizer"
        ))
    }

    /// Encode prompt text into token ids.
    ///
    /// `add_special_tokens` is `false`: nothing is prepended or appended to the
    /// caller's text. The vocabulary carries no BOS and its post-processor is a
    /// plain byte-level one, so the flag has nothing to add here — it is passed
    /// explicitly so a future vocabulary that does define a template processor
    /// cannot start injecting tokens the chat renderer did not ask for.
    ///
    /// The added-vocabulary matcher is a separate mechanism and stays on: any
    /// literal added-token string in the text maps to its single id (gated by
    /// `encode_special_tokens`, left at its default `false`), so `<|im_start|>`,
    /// `</think>`, `<tool_call>`, … resolve to 248045/248069/248058/….
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| anyhow!("encode failed: {e}"))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Encode a rendered prompt without letting client-supplied content mint
    /// control tokens.
    ///
    /// `content_ranges` are the byte ranges of `text` that hold client content
    /// (as reported by `chat::build_prompt_parts_with_spans`). Added-token
    /// occurrences are located with the same leftmost-longest, non-overlapping
    /// discipline the tokenizer's own matcher uses; an occurrence that BEGINS
    /// inside a content range encodes as plain byte-level BPE — never as its
    /// added-token id — while everything else encodes exactly as `encode`
    /// would. The cuts this introduces fall precisely where the added-token
    /// matcher already cuts (it BPE-encodes the fragments between occurrences
    /// independently), so the token stream outside the demoted markers is
    /// unchanged; with no ranges, or with marker-free content, the result is
    /// bit-identical to `encode`.
    ///
    /// Seam rule: demotion keys on where an occurrence begins. A marker whose
    /// first byte is client content stays inert even if it runs past the
    /// range's end (client bytes must not become a control token by borrowing
    /// template bytes); a marker beginning outside every range is structural.
    /// The template never emits a proper prefix of an added token immediately
    /// before client content — content follows a newline, `<function=`,
    /// `<parameter=` or `<tools>`, and no vocabulary entry begins with any of
    /// those — so the structural case cannot absorb client bytes either.
    pub fn encode_prompt(&self, text: &str, content_ranges: &[Range<usize>]) -> Result<Vec<u32>> {
        if content_ranges.is_empty() {
            return self.encode(text);
        }
        let mut ids = Vec::new();
        // Start of the pending fragment that will go through the normal
        // encoder (structural markers inside it still map to their ids there).
        let mut fragment_start = 0;
        let mut scan = 0;
        while let Some((start, end)) = self.find_marker(text, scan) {
            let in_content = content_ranges
                .iter()
                .any(|r| r.start <= start && start < r.end);
            if in_content {
                if start > fragment_start {
                    ids.extend(self.encode(&text[fragment_start..start])?);
                }
                ids.extend(self.encode_plain(&text[start..end])?);
                fragment_start = end;
            }
            scan = end;
        }
        if fragment_start < text.len() {
            ids.extend(self.encode(&text[fragment_start..])?);
        }
        Ok(ids)
    }

    /// The next added-token occurrence at or after byte `from`, as
    /// `(start, end)`. Leftmost position wins, longest marker at that position
    /// wins, and callers resume past `end` — the exact semantics of the
    /// leftmost-longest, non-overlapping AhoCorasick matcher `encode` uses, so
    /// the occurrences seen here are the ones `encode` would turn into ids.
    fn find_marker(&self, text: &str, from: usize) -> Option<(usize, usize)> {
        let bytes = text.as_bytes();
        for start in from..bytes.len() {
            if !self.marker_first_bytes[bytes[start] as usize] {
                continue;
            }
            // `markers` is sorted longest-first: take the first hit.
            for (marker, _) in &self.markers {
                if bytes[start..].starts_with(marker.as_bytes()) {
                    return Some((start, start + marker.len()));
                }
            }
        }
        None
    }

    /// Encode through the added-vocabulary-free tokenizer: the same model,
    /// pre-tokenizer and byte-level vocab, but no string maps to an added
    /// token, so marker text ends up as ordinary byte-BPE pieces that decode
    /// back to the literal characters.
    fn encode_plain(&self, text: &str) -> Result<Vec<u32>> {
        if self.plain.get().is_none() {
            let spec = self
                .inner
                .to_string(false)
                .map_err(|e| anyhow!("serialize tokenizer: {e}"))?;
            let mut spec: serde_json::Value = serde_json::from_str(&spec)?;
            spec["added_tokens"] = serde_json::Value::Array(Vec::new());
            let plain = tokenizers::Tokenizer::from_bytes(serde_json::to_vec(&spec)?)
                .map_err(|e| anyhow!("build added-vocabulary-free tokenizer: {e}"))?;
            let _ = self.plain.set(plain);
        }
        let encoding = self
            .plain
            .get()
            .expect("initialized above")
            .encode(text, false)
            .map_err(|e| anyhow!("plain encode failed: {e}"))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decode ids back to text. Special tokens are rendered verbatim (lossless):
    /// the generation loop stops on an EOG before it would decode one, and the
    /// structural markers callers care about (`<think>` etc.) are non-special
    /// added tokens that render as their literal text regardless.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner
            .decode(ids, false)
            .map_err(|e| anyhow!("decode failed: {e}"))
    }

    /// The checkpoint's logit width — the output layer's row count, which is
    /// padded well past the last encodable token. Both checkpoints share it.
    ///
    /// This is the OTHER vocabulary size, and the two are not interchangeable
    /// (see [`LagunaTokenizer::vocab_size`]). Anything indexed by logit — an
    /// allow-mask handed to the sampler, a bias vector — must span this many
    /// ids or it silently bans the tail it fails to cover. Anything that reads
    /// a token's text must stop at `vocab_size()` instead, since the ids
    /// between the two have no text at all.
    ///
    /// It is named rather than derived because the consumers that need it
    /// (the grammar trie in `constrain.rs`) are built before any model is
    /// loaded. `config.rs` reads the same number out of the GGUF for the model
    /// it actually loaded, which is the copy to trust if the two ever disagree.
    pub const PADDED_VOCAB: usize = 248_320;

    /// Number of ids this tokenizer can encode to or decode from, added tokens
    /// included. Ids `0..n` are the scannable range for sweeps that need a
    /// token's TEXT (decoded-substring blacklists and the like).
    ///
    /// This is deliberately not the model's logit width — see
    /// [`LagunaTokenizer::PADDED_VOCAB`] for the distinction and which side
    /// each kind of caller belongs on.
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// Ids of the added-vocabulary entries, sorted. This is where every
    /// structural marker lives — `<|im_start|>`, `<|im_end|>`, `<think>`,
    /// `<tool_call>`, … — so it is the set a logit mask must never touch, and
    /// reading it from the tokenizer beats hand-listing ids that the next
    /// checkpoint may renumber.
    pub fn added_token_ids(&self) -> Vec<u32> {
        let mut ids: Vec<u32> = self
            .inner
            .get_added_vocabulary()
            .get_vocab()
            .values()
            .copied()
            .collect();
        ids.sort_unstable();
        ids
    }

    /// The literal string for a token id, or `None` if out of range.
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id)
    }

    /// Raw bytes of every encodable id, indexed by id, materialized once per
    /// tokenizer on first use and cached (the walk over the ~248k-entry
    /// vocabulary is too slow to repeat per call site).
    ///
    /// A BPE entry's stored string is in the byte-level ALPHABET (`Ġ` for
    /// space, `Ċ` for newline, one printable char per byte), so it is mapped
    /// back through the alphabet's inverse — never through `decode`, which is
    /// lossy (U+FFFD) for a token holding part of a multi-byte UTF-8 character
    /// and would misclassify exactly the ids a byte-precise sweep exists to
    /// classify. An added token has no byte-level form and contributes its
    /// literal text's bytes; an id with no entry at all contributes nothing.
    /// `token_bytes_reverse_the_byte_level_alphabet` pins the mapping against
    /// `encode` on ASCII, whitespace-led and multi-byte-UTF-8 text alike.
    pub fn decoded_vocab(&self) -> &[Vec<u8>] {
        self.decoded.get_or_init(|| {
            let inverse = byte_level_inverse();
            let added: std::collections::HashMap<u32, &str> = self
                .markers
                .iter()
                .map(|(text, id)| (*id, text.as_str()))
                .collect();
            (0..self.vocab_size() as u32)
                .map(|id| {
                    if let Some(text) = added.get(&id) {
                        return text.as_bytes().to_vec();
                    }
                    let Some(token) = self.inner.id_to_token(id) else {
                        return Vec::new();
                    };
                    token
                        .chars()
                        .filter_map(|c| inverse.get(&c).copied())
                        .collect()
                })
                .collect()
        })
    }

    /// Incremental decoder for streaming (handles multi-token UTF-8).
    pub fn decode_stream(&self) -> DecodeStream<'_> {
        DecodeStream {
            tokenizer: self,
            ids: Vec::new(),
            prefix: String::new(),
            prefix_index: 0,
        }
    }
}

/// Streaming, UTF-8-safe decoder.
///
/// Mirrors the prefix-diff algorithm of `tokenizers::DecodeStream`: it keeps a
/// rolling suffix of ids around the last emitted `prefix` so a decode of the buffer
/// always reproduces `prefix` as a leading substring; the freshly finalized text is
/// whatever the current decode adds beyond `prefix`. Bytes that would land mid
/// UTF-8 sequence decode to the replacement char `U+FFFD` and are withheld (return
/// `None`) until a later token completes them.
pub struct DecodeStream<'a> {
    tokenizer: &'a LagunaTokenizer,
    ids: Vec<u32>,
    prefix: String,
    prefix_index: usize,
}

impl DecodeStream<'_> {
    /// Feed one token; returns text newly finalized by it, if any.
    pub fn step(&mut self, id: u32) -> Result<Option<String>> {
        if self.prefix.is_empty() && !self.ids.is_empty() {
            let new_prefix = self.tokenizer.decode(&self.ids)?;
            if !new_prefix.ends_with('\u{fffd}') {
                self.prefix = new_prefix;
                self.prefix_index = self.ids.len();
            }
        }

        self.ids.push(id);
        let string = self.tokenizer.decode(&self.ids)?;
        if string.len() > self.prefix.len() && !string.ends_with('\u{fffd}') {
            if !string.starts_with(&self.prefix) {
                return Err(anyhow!(
                    "streaming decode produced {string:?}, which does not extend prefix {:?}",
                    self.prefix
                ));
            }
            let new_text = string[self.prefix.len()..].to_string();
            let new_prefix_index = self.ids.len() - self.prefix_index;
            self.ids = self.ids.split_off(self.prefix_index);
            self.prefix = self.tokenizer.decode(&self.ids)?;
            self.prefix_index = new_prefix_index;
            Ok(Some(new_text))
        } else {
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokenizer() -> LagunaTokenizer {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/reference/tokenizer.json");
        LagunaTokenizer::from_file(path).expect("load reference tokenizer")
    }

    /// `decoded_vocab` must reverse the byte-level alphabet exactly: for any
    /// text, the concatenated bytes of its encoded ids are the text's own
    /// UTF-8 bytes. Checked across the three regimes that differ — plain
    /// ASCII, whitespace-led spellings (`Ġ`-class stand-ins), and a character
    /// whose UTF-8 spans multiple tokens, where `decode(&[id])` per id is
    /// lossy (U+FFFD) and this table must not be.
    #[test]
    fn token_bytes_reverse_the_byte_level_alphabet() {
        let t = tokenizer();
        let table = t.decoded_vocab();
        assert_eq!(table.len(), t.vocab_size());
        for text in [
            "true",
            " true",
            "{\"urgent\":",
            "h\u{00e9}llo \u{1F9FF}!",
            "\u{1F9FF}",
        ] {
            let ids = t.encode(text).unwrap();
            let bytes: Vec<u8> = ids
                .iter()
                .flat_map(|&id| table[id as usize].iter().copied())
                .collect();
            assert_eq!(bytes, text.as_bytes(), "{text:?} via {ids:?}");
        }
        // An added token contributes its literal text, not an alphabet form.
        assert_eq!(
            table[LagunaTokenizer::IM_END as usize],
            b"<|im_end|>".to_vec()
        );
    }

    /// The default vocabulary is the bytes compiled into the binary; loading
    /// `reference/tokenizer.json` from disk must be the same tokenizer, or the
    /// `--tokenizer reference/tokenizer.json` override would not be a no-op.
    #[test]
    fn embedded_tokenizer_matches_the_reference_file() {
        let embedded = LagunaTokenizer::embedded().expect("parse embedded tokenizer");
        let from_disk = tokenizer();
        assert_eq!(embedded.vocab_size(), from_disk.vocab_size());
        assert_eq!(embedded.markers, from_disk.markers);
        let sample = "<|im_start|>user\nfn main() { println!(\"héllo\"); }<|im_end|>\n<|im_start|>assistant\n<think>\n";
        assert_eq!(
            embedded.encode(sample).unwrap(),
            from_disk.encode(sample).unwrap()
        );
    }

    #[test]
    fn special_ids_map_to_expected_strings() {
        let t = tokenizer();
        assert_eq!(
            t.id_to_token(LagunaTokenizer::ENDOFTEXT).as_deref(),
            Some("<|endoftext|>")
        );
        assert_eq!(
            t.id_to_token(LagunaTokenizer::IM_START).as_deref(),
            Some("<|im_start|>")
        );
        assert_eq!(
            t.id_to_token(LagunaTokenizer::IM_END).as_deref(),
            Some("<|im_end|>")
        );
        assert_eq!(
            t.id_to_token(LagunaTokenizer::THINK_OPEN).as_deref(),
            Some("<think>")
        );
        assert_eq!(
            t.id_to_token(LagunaTokenizer::THINK_CLOSE).as_deref(),
            Some("</think>")
        );
        assert_eq!(
            t.id_to_token(LagunaTokenizer::TOOL_CALL_OPEN).as_deref(),
            Some("<tool_call>")
        );
        assert_eq!(
            t.id_to_token(LagunaTokenizer::TOOL_CALL_CLOSE).as_deref(),
            Some("</tool_call>")
        );
        assert_eq!(
            t.id_to_token(LagunaTokenizer::TOOL_RESPONSE_OPEN)
                .as_deref(),
            Some("<tool_response>")
        );
        assert_eq!(
            t.id_to_token(LagunaTokenizer::TOOL_RESPONSE_CLOSE)
                .as_deref(),
            Some("</tool_response>")
        );
    }

    /// Both end-of-generation ids are real vocabulary entries, and they are the
    /// two the generation loop must stop on: `<|im_end|>` for a chat turn,
    /// `<|endoftext|>` for the document.
    #[test]
    fn eog_holds_both_turn_terminators() {
        let t = tokenizer();
        assert_eq!(
            LagunaTokenizer::EOG,
            [LagunaTokenizer::IM_END, LagunaTokenizer::ENDOFTEXT]
        );
        let added = added_ids(&t);
        for eog in LagunaTokenizer::EOG {
            assert!(added.contains(&eog), "{eog} is not a vocabulary entry");
        }
        assert_eq!(LagunaTokenizer::PAD, LagunaTokenizer::ENDOFTEXT);
    }

    /// Nothing is prepended to an encoded prompt: the conversation starts at the
    /// chat template's first `<|im_start|>`, and empty text encodes to no tokens
    /// at all.
    #[test]
    fn encoding_prepends_no_token() {
        let t = tokenizer();
        let added = added_ids(&t);
        assert!(t.encode("").unwrap().is_empty());
        let ids = t.encode("Hi").unwrap();
        assert!(
            ids.iter().all(|id| !added.contains(id)),
            "plain text encoded to a control token: {ids:?}"
        );
        assert_eq!(t.decode(&ids).unwrap(), "Hi");
        // A prompt's first id is the template's own marker, not a prepended one.
        assert_eq!(
            t.encode("<|im_start|>user\n").unwrap()[0],
            LagunaTokenizer::IM_START
        );
    }

    /// The tokenizer's id space is smaller than the model's logits width: the
    /// checkpoint pads its output layer past the last encodable token. Ids at or
    /// past this bound have no text, which is why a decoding sweep must use this
    /// number and a logit mask must not.
    #[test]
    fn vocab_size_is_the_encodable_id_space() {
        let t = tokenizer();
        assert_eq!(t.vocab_size(), 248_070);
        assert!(t.id_to_token(t.vocab_size() as u32 - 1).is_some());
        assert!(t.id_to_token(t.vocab_size() as u32).is_none());
        // The logit width is the larger of the two, and the ids between them
        // are the padded tail: real logit positions with no text behind them.
        assert!(LagunaTokenizer::PADDED_VOCAB > t.vocab_size());
        assert!(
            t.id_to_token(LagunaTokenizer::PADDED_VOCAB as u32 - 1)
                .is_none()
        );
        // Every id this module names is inside it.
        for id in [
            LagunaTokenizer::ENDOFTEXT,
            LagunaTokenizer::IM_START,
            LagunaTokenizer::IM_END,
            LagunaTokenizer::THINK_OPEN,
            LagunaTokenizer::THINK_CLOSE,
            LagunaTokenizer::TOOL_CALL_OPEN,
            LagunaTokenizer::TOOL_CALL_CLOSE,
            LagunaTokenizer::TOOL_RESPONSE_OPEN,
            LagunaTokenizer::TOOL_RESPONSE_CLOSE,
        ] {
            assert!((id as usize) < t.vocab_size());
        }
    }

    #[test]
    fn structural_markers_map_to_added_token_ids() {
        let t = tokenizer();
        // The thinking and tool markers carry `special: false` in the vocabulary
        // file, which governs rendering rather than matching: they resolve to
        // their single ids exactly as the `<|…|>` specials do.
        let ids = t
            .encode("<|im_start|><think></think><tool_call></tool_call><tool_response></tool_response><|im_end|><|endoftext|>")
            .unwrap();
        assert_eq!(
            ids,
            vec![
                LagunaTokenizer::IM_START,
                LagunaTokenizer::THINK_OPEN,
                LagunaTokenizer::THINK_CLOSE,
                LagunaTokenizer::TOOL_CALL_OPEN,
                LagunaTokenizer::TOOL_CALL_CLOSE,
                LagunaTokenizer::TOOL_RESPONSE_OPEN,
                LagunaTokenizer::TOOL_RESPONSE_CLOSE,
                LagunaTokenizer::IM_END,
                LagunaTokenizer::ENDOFTEXT,
            ]
        );
    }

    /// The markers the chat template emits as content-adjacent structure — the
    /// strings whose injection from client content `encode_prompt` exists to
    /// prevent.
    const DANGEROUS_MARKERS: [&str; 9] = [
        "<think>",
        "</think>",
        "<|im_start|>",
        "<|im_end|>",
        "<|endoftext|>",
        "<tool_call>",
        "</tool_call>",
        "<tool_response>",
        "</tool_response>",
    ];

    fn added_ids(t: &LagunaTokenizer) -> std::collections::HashSet<u32> {
        t.added_token_ids().into_iter().collect()
    }

    #[test]
    fn content_ranges_demote_markers_to_plain_text() {
        let t = tokenizer();
        let added = added_ids(&t);
        for marker in DANGEROUS_MARKERS {
            let text = format!("try: {marker} please\n");
            let start = "try: ".len();
            let range = start..start + marker.len();
            // Without ranges the marker becomes a real control token (the
            // capability the template's own markers need).
            let vulnerable = t.encode(&text).unwrap();
            assert!(
                vulnerable.iter().any(|id| added.contains(id)),
                "{marker:?} should map to an added id outside content ranges"
            );
            // As content it must not: no added id anywhere in the stream
            // (nothing else in this text is an added token), and the ids decode
            // back to the exact literal bytes.
            let ids = t.encode_prompt(&text, &[range]).unwrap();
            assert!(
                ids.iter().all(|id| !added.contains(id)),
                "content {marker:?} injected an added id: {ids:?}"
            );
            assert_eq!(t.decode(&ids).unwrap(), text);
        }
    }

    // The safety argument for `encode_plain`: byte-level BPE cannot build any
    // added-token string — no merge path reaches one — so demoted marker text
    // can never round-trip into a control id. Sweeps the whole added
    // vocabulary, ids derived from the tokenizer rather than hardcoded.
    #[test]
    fn plain_bpe_cannot_reach_any_added_token_id() {
        let t = tokenizer();
        let added = added_ids(&t);
        assert!(!added.is_empty());
        for &id in &added {
            let text = t.id_to_token(id).expect("added id has text");
            let ids = t.encode_prompt(&text, &[0..text.len()]).unwrap();
            assert!(!ids.is_empty());
            assert!(
                ids.iter().all(|i| !added.contains(i)),
                "plain BPE of {text:?} reached an added id: {ids:?}"
            );
            assert_eq!(t.decode(&ids).unwrap(), text, "round-trip of {text:?}");
        }
    }

    #[test]
    fn encode_prompt_is_bit_identical_to_encode_without_demotions() {
        let t = tokenizer();
        let text = "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nWhat is 2+2? 🎉<|im_end|>\n<|im_start|>assistant\n<think>\n";
        // No ranges at all.
        assert_eq!(t.encode_prompt(text, &[]).unwrap(), t.encode(text).unwrap());
        // Ranges over marker-free content change nothing either.
        let start = text.find("What").unwrap();
        let range = start..start + "What is 2+2? 🎉".len();
        assert_eq!(
            t.encode_prompt(text, &[range]).unwrap(),
            t.encode(text).unwrap()
        );
    }

    // Seam rule: an occurrence is content iff its FIRST byte lies inside a
    // content range — a marker straddling the range's end is still demoted
    // (client bytes must not become a control token by borrowing template
    // bytes), while a marker starting exactly at the range's end is structural.
    #[test]
    fn demotion_keys_on_where_the_marker_begins() {
        let t = tokenizer();
        let added = added_ids(&t);
        let text = "ab<think>cd";
        // Range covers "ab<": the marker begins inside it and runs past it.
        let ids = t.encode_prompt(text, &[0..3]).unwrap();
        assert!(ids.iter().all(|id| !added.contains(id)));
        assert_eq!(t.decode(&ids).unwrap(), text);
        // Range covers only "ab": the marker begins at the boundary, outside.
        let ids = t.encode_prompt(text, &[0..2]).unwrap();
        assert!(ids.contains(&LagunaTokenizer::THINK_OPEN));
    }

    // One string, both roles at once: the structural copies keep their ids
    // while the copy inside the content range is spelled out as text.
    #[test]
    fn the_same_marker_is_structural_outside_and_text_inside_a_content_range() {
        let t = tokenizer();
        let text =
            "<|im_start|>user\n<think>hi</think><|im_end|>\n<|im_start|>assistant\n<think>\n";
        let body = "<|im_start|>user\n";
        let content = body.len()..body.len() + "<think>hi</think>".len();
        let ids = t.encode_prompt(text, &[content]).unwrap();
        // Only the generation header's <think> survives as a control id; the
        // user's copy (and their </think>) are plain text. The turn markers are
        // structural throughout.
        let count = |id| ids.iter().filter(|&&i| i == id).count();
        assert_eq!(count(LagunaTokenizer::THINK_OPEN), 1);
        assert_eq!(count(LagunaTokenizer::THINK_CLOSE), 0);
        assert_eq!(count(LagunaTokenizer::IM_START), 2);
        assert_eq!(count(LagunaTokenizer::IM_END), 1);
        assert_eq!(t.decode(&ids).unwrap(), text);
    }

    #[test]
    fn decode_roundtrips_plain_text() {
        let t = tokenizer();
        let ids = t.encode("Hello, world!").unwrap();
        assert_eq!(t.decode(&ids).unwrap(), "Hello, world!");
    }

    #[test]
    fn decode_stream_matches_whole_decode() {
        let t = tokenizer();
        let ids = t
            .encode("The quick brown fox jumps over the lazy dog.")
            .unwrap();
        let mut stream = t.decode_stream();
        let mut streamed = String::new();
        for &id in &ids {
            if let Some(chunk) = stream.step(id).unwrap() {
                streamed.push_str(&chunk);
            }
        }
        assert_eq!(streamed, t.decode(&ids).unwrap());
    }

    #[test]
    fn decode_stream_withholds_partial_utf8() {
        let t = tokenizer();
        // A multi-byte grapheme ("界", U+754C) generally spans several byte-level
        // tokens; the stream must never emit a lone replacement char and, summed,
        // must reproduce the full decode.
        let ids = t.encode("世界").unwrap();
        let mut stream = t.decode_stream();
        let mut streamed = String::new();
        for &id in &ids {
            if let Some(chunk) = stream.step(id).unwrap() {
                assert!(
                    !chunk.contains('\u{fffd}'),
                    "emitted a partial-UTF8 chunk: {chunk:?}"
                );
                streamed.push_str(&chunk);
            }
        }
        assert_eq!(streamed, "世界");
    }
}
