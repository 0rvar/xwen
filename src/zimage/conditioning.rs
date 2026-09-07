//! The prompt as a diffusion pipeline's text encoder sees it.
//!
//! diffusers' `ZImagePipeline._encode_prompt` renders one user turn with the
//! generation prompt appended and thinking on (which on the Qwen3 dialect opens
//! no `<think>` block), tokenizes it with the checkpoint's OWN `tokenizer.json`,
//! and truncates to the pipeline's `max_length`. `xwen encode-text`, `xwen
//! image` and the serve images route all go through [`prompt_ids`] so the three
//! cannot drift.

use std::path::Path;

use anyhow::{Context, Result, ensure};

use crate::chat::{ChatOptions, Message, build_prompt_with_spans};
use crate::hub::Model;
use crate::tokenizer::LagunaTokenizer;

/// A rendered and tokenized prompt.
#[derive(Debug, Clone)]
pub struct RenderedPrompt {
    /// The chat-template rendering the ids came from.
    pub text: String,
    /// The ids after the entry's truncation.
    pub ids: Vec<u32>,
    /// The untruncated length when the entry's `max_tokens` cut the prompt;
    /// `None` when nothing was cut. The caller decides whether that is worth a
    /// warning; the pipeline itself truncates silently.
    pub truncated_from: Option<usize>,
}

/// Render `prompt` for `entry`'s dialect, tokenize it with `tokenizer_path`, and
/// truncate to the entry's encoder `max_tokens` when it has one.
pub fn prompt_ids(entry: Model, tokenizer_path: &Path, prompt: &str) -> Result<RenderedPrompt> {
    let chat_opts = ChatOptions::for_dialect(entry.chat_dialect());
    // The third element is the thinking state the generation prompt leaves
    // the model in; an encoder generates nothing, so it has no reader here.
    let (text, content_ranges, _thinking) =
        build_prompt_with_spans(&[Message::User(prompt.to_string())], &chat_opts)?;
    let tokenizer = LagunaTokenizer::from_file(tokenizer_path)
        .with_context(|| format!("loading {}", tokenizer_path.display()))?;
    let mut ids = tokenizer.encode_prompt(&text, &content_ranges)?;
    let mut truncated_from = None;
    if let Some(spec) = entry.encoder_spec()
        && ids.len() > spec.max_tokens
    {
        truncated_from = Some(ids.len());
        ids.truncate(spec.max_tokens);
    }
    ensure!(!ids.is_empty(), "the rendered prompt tokenized to nothing");
    Ok(RenderedPrompt {
        text,
        ids,
        truncated_from,
    })
}
