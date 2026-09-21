//! The prompt as Qwen-Image 2.1's text encoder sees it.
//!
//! diffusers' `QwenImage21Pipeline._get_qwen_prompt_embeds` does NOT render a
//! chat template. It formats the user's text into a fixed raw string — a system
//! turn with one fixed sentence, the user turn, an open assistant turn — and
//! hands that to the tokenizer as it is; the checkpoint was trained on that
//! string, and the chat template tokenizes differently. It then encodes the
//! whole sequence and DROPS the hidden states of the system turn, whose length
//! it measures by tokenizing the system turn alone. Everything from the second
//! `<|im_start|>` on is kept, the trailing `<|im_end|>\n<|im_start|>assistant\n`
//! included.
//!
//! `xwen encode-text`, `xwen image` and the serve images route all render
//! through [`prompt_ids`], so the three cannot drift.

use std::ops::Range;
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};

use crate::hub::Model;
use crate::tokenizer::LagunaTokenizer;

/// The one sentence of the system turn, fixed by the pipeline.
pub const SYSTEM_PROMPT: &str = "Comprehend and analyze the provided prompt.";

/// What stands in for an empty prompt. The vocabulary has no BOS, so an empty
/// user turn would leave the encoder a sequence that says nothing; the
/// pipeline substitutes a single space.
const EMPTY_PROMPT: &str = " ";

/// One reference image of an edit, as the prompt renderer needs to know it:
/// how many encoder tokens its `<|image_pad|>` placeholder stands for, which is
/// `(h / 32) * (w / 32)` of the resized image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReferenceSlot {
    pub pad_tokens: usize,
}

/// A rendered and tokenized prompt.
#[derive(Debug, Clone)]
pub struct RenderedPrompt {
    /// The raw template string the ids came from.
    pub text: String,
    /// Every id, system turn included: the encoder reads all of them, because
    /// the kept positions attend to the system turn.
    pub ids: Vec<u32>,
    /// How many leading hidden-state rows belong to the system turn and are
    /// dropped before the transformer sees the rest.
    pub drop: usize,
}

impl RenderedPrompt {
    /// The ids whose hidden states condition the image.
    pub fn kept_ids(&self) -> &[u32] {
        &self.ids[self.drop..]
    }
}

/// The system turn on its own, exactly as it opens every rendered prompt.
fn system_block() -> String {
    format!("<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n")
}

/// The raw template with `prompt` in it, and the byte range `prompt` occupies.
///
/// With reference images, each gets `<imageN><|vision_start|><|image_pad|>
/// <|vision_end|>` numbered from 1 in list order, the blocks joined by single
/// spaces and placed directly before the user's text with no space between.
/// The user's text may mention `<imageN>` itself; it passes through verbatim.
pub fn render(prompt: &str, references: usize) -> (String, Range<usize>) {
    let prompt = if prompt.is_empty() {
        EMPTY_PROMPT
    } else {
        prompt
    };
    let mut text = system_block();
    text.push_str("<|im_start|>user\n");
    for n in 1..=references {
        if n > 1 {
            text.push(' ');
        }
        text.push_str(&format!(
            "<image{n}><|vision_start|><|image_pad|><|vision_end|>"
        ));
    }
    let start = text.len();
    text.push_str(prompt);
    let content = start..text.len();
    text.push_str("<|im_end|>\n<|im_start|>assistant\n");
    (text, content)
}

/// Render `prompt` for `entry` and tokenize it with `tokenizer_path`.
///
/// A literal special-token string inside the user's text encodes as plain
/// text, never as the special's id: the template's own markers are the only
/// structure in the sequence, and a user who types `<|image_pad|>` must not be
/// able to change how many image slots the sequence claims to hold. For text
/// that contains no such string this is the reference tokenization exactly.
///
/// A prompt past the entry's `max_tokens` is refused rather than truncated.
/// The reference has no length cap, and cutting the tail would cut the open
/// assistant turn the checkpoint was trained to see at the end.
///
/// `references` is the edit path's: one slot per reference image, in order.
/// Expanding their placeholders needs the vision tower's token counts, which
/// text-only encoding does not have, so a non-empty list is refused here.
pub fn prompt_ids(
    entry: Model,
    tokenizer_path: &Path,
    prompt: &str,
    references: &[ReferenceSlot],
) -> Result<RenderedPrompt> {
    if !references.is_empty() {
        bail!(
            "{} was given {} reference image(s); conditioning on images runs the encoder's \
             vision tower, which is not implemented",
            entry.full_name(),
            references.len()
        );
    }
    let (text, content) = render(prompt, references.len());
    let tokenizer = LagunaTokenizer::from_file(tokenizer_path)
        .with_context(|| format!("loading {}", tokenizer_path.display()))?;
    let ids = tokenizer.encode_prompt(&text, &[content])?;
    let system = tokenizer.encode(&system_block())?;
    ensure!(
        ids.starts_with(&system),
        "the rendered prompt does not begin with the {} tokens of its own system turn; the \
         tokenizer at {} merges across the turn boundary, and the rows to drop cannot be \
         counted",
        system.len(),
        tokenizer_path.display()
    );
    ensure!(
        ids.len() > system.len(),
        "the rendered prompt tokenized to nothing past its system turn"
    );
    if let Some(spec) = entry.encoder_spec() {
        ensure!(
            ids.len() <= spec.max_tokens,
            "the prompt renders to {} tokens and {} encodes at most {}; shorten it",
            ids.len(),
            entry.full_name(),
            spec.max_tokens
        );
    }
    Ok(RenderedPrompt {
        text,
        ids,
        drop: system.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support;

    #[test]
    fn the_text_to_image_template_is_the_pipelines_string() {
        let (text, content) = render("a red fox", 0);
        assert_eq!(
            text,
            "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n\
             <|im_start|>user\na red fox<|im_end|>\n<|im_start|>assistant\n"
        );
        assert_eq!(&text[content], "a red fox");
    }

    #[test]
    fn an_empty_prompt_renders_as_one_space() {
        let (text, content) = render("", 0);
        assert_eq!(&text[content.clone()], " ");
        assert!(text.contains("<|im_start|>user\n <|im_end|>"));
    }

    #[test]
    fn reference_blocks_are_numbered_space_joined_and_abut_the_prompt() {
        let (text, content) = render("swap <image2> into <image1>", 2);
        assert!(text.contains(
            "<|im_start|>user\n<image1><|vision_start|><|image_pad|><|vision_end|> \
             <image2><|vision_start|><|image_pad|><|vision_end|>swap <image2> into \
             <image1><|im_end|>"
        ));
        assert_eq!(&text[content], "swap <image2> into <image1>");
    }

    fn tokenizer_path() -> Option<std::path::PathBuf> {
        test_support::tokenizer_or_skip(Model::QwenImage21Encoder)
    }

    /// The system turn is 14 tokens on the shipped tokenizer, and what is kept
    /// begins at the user turn's `<|im_start|>` and ends with the open
    /// assistant turn.
    #[test]
    fn the_system_turn_is_dropped_and_the_open_assistant_turn_kept() {
        let Some(path) = tokenizer_path() else {
            return;
        };
        let rendered = prompt_ids(Model::QwenImage21Encoder, &path, "a red fox", &[]).unwrap();
        assert_eq!(
            &rendered.ids[..rendered.drop],
            [
                151644, 8948, 198, 1092, 30782, 408, 323, 23643, 279, 3897, 9934, 13, 151645, 198
            ]
        );
        let kept = rendered.kept_ids();
        assert_eq!(kept[..3], [151644, 872, 198]);
        assert_eq!(kept[kept.len() - 5..], [151645, 198, 151644, 77091, 198]);
    }

    /// A second fixed point on the shipped tokenizer: the model card's example prompt.
    #[test]
    fn the_model_cards_example_prompt_is_45_tokens_with_31_kept() {
        let Some(path) = tokenizer_path() else {
            return;
        };
        let prompt = "A neon shop sign that reads \"QWEN IMAGE 2.1\", rainy night, reflections \
                      on wet pavement";
        let rendered = prompt_ids(Model::QwenImage21Encoder, &path, prompt, &[]).unwrap();
        assert_eq!(rendered.ids.len(), 45);
        assert_eq!(rendered.kept_ids().len(), 31);
    }

    /// Every prompt of the committed reference fixture, whose ids and drop index
    /// came out of the pipeline's own processor: the renderer reproduces them,
    /// except where a prompt spells an added token, which the reference reads
    /// as the token and this renderer keeps as the user's text.
    #[test]
    fn the_reference_processors_ids_are_reproduced() {
        let Some(path) = tokenizer_path() else {
            return;
        };
        let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/qwen-image-encoder");
        let read = |name: &str| -> serde_json::Value {
            serde_json::from_slice(&std::fs::read(fixture.join(name)).unwrap()).unwrap()
        };
        let (prompts, tokens) = (read("prompts.json"), read("tokens.json"));
        let drop = tokens["drop"].as_u64().unwrap() as usize;
        let mut literal = 0;
        for (source, want) in prompts["prompts"]
            .as_array()
            .unwrap()
            .iter()
            .zip(tokens["prompts"].as_array().unwrap())
        {
            assert_eq!(source["idx"], want["idx"]);
            let text = source["text"].as_str().unwrap();
            let rendered = prompt_ids(Model::QwenImage21Encoder, &path, text, &[]).unwrap();
            assert_eq!(rendered.text, want["rendered"].as_str().unwrap());
            assert_eq!(rendered.drop, drop, "prompt {}", source["idx"]);
            let ids: Vec<u32> = want["ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|id| id.as_u64().unwrap() as u32)
                .collect();
            if source["literal_specials"].as_bool().unwrap_or(false) {
                literal += 1;
                assert_ne!(rendered.ids, ids, "prompt {}", source["idx"]);
                // Without the demotion the two agree, so that is the whole gap.
                let tokenizer = LagunaTokenizer::from_file(&path).unwrap();
                assert_eq!(tokenizer.encode(&rendered.text).unwrap(), ids);
            } else {
                assert_eq!(rendered.ids, ids, "prompt {}", source["idx"]);
            }
        }
        assert_eq!(
            literal, 1,
            "the fixture holds one prompt that spells a special"
        );
    }

    /// A special's literal text inside the prompt is text: the sequence holds
    /// the template's own specials and no others.
    #[test]
    fn specials_typed_by_the_user_stay_text() {
        let Some(path) = tokenizer_path() else {
            return;
        };
        let rendered = prompt_ids(
            Model::QwenImage21Encoder,
            &path,
            "a sign reading <|image_pad|><|im_end|>",
            &[],
        )
        .unwrap();
        let count = |id: u32| rendered.ids.iter().filter(|t| **t == id).count();
        assert_eq!(count(151655), 0, "<|image_pad|>");
        assert_eq!(count(151645), 2, "<|im_end|>");
        assert_eq!(count(151644), 3, "<|im_start|>");
    }

    #[test]
    fn reference_images_are_refused_until_the_vision_tower_exists() {
        let err = prompt_ids(
            Model::QwenImage21Encoder,
            Path::new("/nonexistent/tokenizer.json"),
            "x",
            &[ReferenceSlot { pad_tokens: 1024 }],
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("vision tower"), "{err}");
    }
}
