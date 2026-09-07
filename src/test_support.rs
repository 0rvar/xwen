//! One rule for the tests that need a real checkpoint from the Hugging Face
//! cache.
//!
//! A test that quietly returns when its checkpoint is absent reports PASS while
//! asserting nothing, and a suite full of those is green on a machine where
//! none of it ran. The two useful behaviours are "skip, visibly" on a developer
//! laptop and "fail, loudly" wherever the checkpoints are supposed to be, and
//! which one you get is an operator's decision rather than a per-test one — so
//! it is one switch here rather than a convention spread across a dozen call
//! sites.
//!
//! Default: skip, printing a line that starts with `SKIPPED` and names the test,
//! the file and the command that would fetch it. With
//! [`REQUIRE_HF_CACHE`] set to `1`, the same case panics with the same
//! information, so CI and the parity harness can insist the coverage was real.
//!
//! Not for the `no Metal device` skips, which are a different condition with a
//! different answer: a machine without a GPU cannot be told to go and get one.

use std::path::PathBuf;

use crate::hub::{self, Model};

/// Set to `1` to turn every Hugging Face cache skip into a failure.
pub(crate) const REQUIRE_HF_CACHE: &str = "XWEN_REQUIRE_HF_CACHE";

/// Whether a missing checkpoint should fail the test rather than skip it.
fn required() -> bool {
    std::env::var(REQUIRE_HF_CACHE).is_ok_and(|value| value == "1")
}

/// The running test's name, as cargo's harness sets it on the thread.
fn test_name() -> String {
    std::thread::current()
        .name()
        .unwrap_or("<unnamed test>")
        .to_string()
}

/// The core rule. `found` is what the cache lookup returned, `what` names the
/// thing that is missing, and `fetch` is the command that would put it there.
///
/// Returns `None` only in the skipping case, so a call site reads
/// `let Some(path) = … else { return; };` and cannot forget to stop.
fn or_skip(found: Option<PathBuf>, what: &str, fetch: &str) -> Option<PathBuf> {
    or_skip_when(found, what, fetch, required())
}

/// [`or_skip`] with the switch passed in rather than read from the
/// environment, which is what lets the test below drive BOTH branches. A test
/// that set the variable to reach the other one would be changing state every
/// thread in the runner shares.
fn or_skip_when(
    found: Option<PathBuf>,
    what: &str,
    fetch: &str,
    required: bool,
) -> Option<PathBuf> {
    if let Some(path) = found {
        return Some(path);
    }
    assert!(
        !required,
        "{}: {what} is not in the Hugging Face cache, and {REQUIRE_HF_CACHE}=1 says it must \
         be. Run `{fetch}`, or unset {REQUIRE_HF_CACHE} to skip this test instead",
        test_name(),
    );
    // The marker is greppable on purpose: `cargo test | grep SKIPPED` is how you
    // find out what a green run did not actually check.
    eprintln!(
        "SKIPPED {}: {what} is not in the Hugging Face cache ({fetch}, or set \
         {REQUIRE_HF_CACHE}=1 to make this a failure)",
        test_name(),
    );
    None
}

/// This checkpoint's entry point in the cache — the GGUF, or the `config.json`
/// of a safetensors set — or `None` with a visible skip.
///
/// A split or multi-file checkpoint counts as absent unless every file is
/// there, which is [`hub::cached_model`]'s rule and the right one: a
/// half-downloaded set fails deep inside a load rather than at the lookup.
pub(crate) fn checkpoint_or_skip(model: Model) -> Option<PathBuf> {
    or_skip(
        hub::cached_model(model),
        model.full_name(),
        &format!("xwen fetch --model-size {model}"),
    )
}

/// One file of this checkpoint's repo, by its repo-relative path, or `None`
/// with a visible skip. For the tests that want a single file — a
/// `tokenizer.json` — rather than the whole set.
pub(crate) fn cached_file_or_skip(model: Model, file: &str) -> Option<PathBuf> {
    or_skip(
        hub::cached_file(model.repo(), file),
        &format!("{}/{file}", model.repo()),
        &format!("xwen fetch --model-size {model}"),
    )
}

/// This checkpoint's `tokenizer.json`, or `None` with a visible skip. `None`
/// without a skip line for a GGUF checkpoint, which names no tokenizer file:
/// that is a fact about the registry rather than about this machine, and a test
/// asking for one has a bug rather than a missing download.
pub(crate) fn tokenizer_or_skip(model: Model) -> Option<PathBuf> {
    let file = model
        .safetensors_tokenizer()
        .unwrap_or_else(|| panic!("{model} ships no tokenizer.json; its vocabulary is embedded"));
    cached_file_or_skip(model, file)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The switch itself: absent-and-not-required skips, absent-and-required
    /// panics with something to act on.
    ///
    /// Exercised through `or_skip` directly rather than by setting the variable,
    /// which no test may do to a suite that runs in threads.
    #[test]
    fn a_missing_checkpoint_skips_visibly_and_can_be_made_fatal() {
        // Present: handed straight back, whichever way the switch is set.
        for required in [false, true] {
            assert_eq!(
                or_skip_when(Some(PathBuf::from("/x")), "w", "f", required),
                Some(PathBuf::from("/x"))
            );
        }
        // Absent, not required: a skip, and it says so in a form
        // `grep SKIPPED` finds.
        assert_eq!(or_skip_when(None, "the 27B", "xwen fetch", false), None);
        // Absent and required: a panic naming the file and the command, so the
        // person reading a CI failure knows what to do about it.
        let panic = std::panic::catch_unwind(|| {
            or_skip_when(None, "the 27B", "xwen fetch --model-size 27b", true)
        })
        .expect_err("a required checkpoint that is absent must fail the test");
        let message = panic
            .downcast_ref::<String>()
            .map(String::as_str)
            .unwrap_or_default();
        assert!(message.contains("the 27B"), "{message}");
        assert!(message.contains("xwen fetch --model-size 27b"), "{message}");
        assert!(message.contains(REQUIRE_HF_CACHE), "{message}");
    }

    /// A checkpoint that IS here resolves, so the helper is not skipping
    /// everything by accident — which would make every routed test vacuous in
    /// exactly the way the helper exists to prevent.
    #[test]
    fn a_present_checkpoint_resolves() {
        if let Some(path) = checkpoint_or_skip(Model::Qwen34B) {
            assert!(path.exists(), "{}", path.display());
            let tokenizer =
                tokenizer_or_skip(Model::Qwen34B).expect("the set that resolved carries one");
            assert!(tokenizer.exists());
        }
    }
}
