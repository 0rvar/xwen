//! Startup refusals, exercised through the real binary.
//!
//! These are ORDERING tests, and ordering is the one thing a unit test on a
//! predicate cannot see. `xwen serve --model-size qwen3-4b` used to identify the
//! checkpoint, download eight gigabytes, start the server, list the model on
//! `/v1/models` and only then die on the first request — with every individual
//! predicate answering correctly the whole way down. What was wrong was where
//! the question got asked, so the test has to run the thing that asks it.
//!
//! Cheap by construction: every case here fails before any hub access, so
//! nothing is fetched, no port is bound and no model is loaded. If one of them
//! ever starts taking seconds, the gate it covers has moved behind a download.

use std::io::Write;
use std::process::{Command, Stdio};

/// A `serve` config that exists and says nothing.
///
/// Not "no config": with no `--config`, serve reads the operator's own
/// `~/.config/xwen/serve.toml`, and one with a `model` key in it sends the run
/// down a different branch. An empty file pins the branch under test on any
/// machine.
///
/// One per test: the cases here run in parallel in one process, and a file
/// shared between them is one test deleting another test's config mid-run.
fn empty_config(label: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!(
        "xwen_cli_gates_{}_{label}.toml",
        std::process::id()
    ));
    std::fs::write(&path, b"").unwrap();
    path
}

fn xwen() -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_xwen"));
    // The runs below are refusals and record nothing, but a future one that got
    // further must not write into whoever is running the suite.
    cmd.env("XWEN_METRICS_FILE", "off");
    cmd
}

/// `xwen serve --model-size <a checkpoint this build cannot run>` fails at
/// startup, names the checkpoint and says why.
#[test]
fn serve_refuses_an_unrunnable_checkpoint_before_it_fetches_anything() {
    let config = empty_config("serve_refuses");
    for (alias, expected) in [
        ("qwen3-4b", "layer stack is not implemented"),
        ("qwen3-4b-instruct-2507", "layer stack is not implemented"),
        ("zimage-turbo", "encode-only"),
    ] {
        let out = xwen()
            .args(["serve", "--config"])
            .arg(&config)
            .args(["--model-size", alias])
            .output()
            .expect("running xwen serve");
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            !out.status.success(),
            "serve --model-size {alias} started; it must refuse\n{stderr}"
        );
        assert!(
            stderr.contains("cannot be served or batched"),
            "{alias}: {stderr}"
        );
        assert!(stderr.contains(expected), "{alias}: {stderr}");
        // The refusal has to happen before the download, and a message about
        // fetching would mean it did not.
        assert!(
            !stderr.contains("downloading"),
            "{alias} fetched before refusing: {stderr}"
        );
    }
    std::fs::remove_file(&config).unwrap();
}

/// The same gate on `xwen batch`, which names its checkpoint in the payload
/// rather than in a flag and moves cache state for the same reason serve does.
///
/// Batch reports a whole-request failure as a JSON document on stdout and exits
/// 1, so that is where the message is.
#[test]
fn batch_refuses_an_unrunnable_checkpoint_named_in_its_payload() {
    for (name, expected) in [
        ("Qwen3-4B", "layer stack is not implemented"),
        ("Z-Image-Turbo-text-encoder", "encode-only"),
    ] {
        let mut child = xwen()
            .arg("batch")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("running xwen batch");
        let payload = format!(
            r#"{{"model":"{name}","items":[{{"id":"a","messages":[{{"role":"user","content":"hi"}}]}}]}}"#
        );
        child
            .stdin
            .take()
            .unwrap()
            .write_all(payload.as_bytes())
            .unwrap();
        let out = child.wait_with_output().expect("waiting for xwen batch");
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(!out.status.success(), "batch on {name} succeeded\n{stdout}");
        assert!(
            stdout.contains("cannot be served or batched"),
            "{name}: {stdout}"
        );
        assert!(stdout.contains(expected), "{name}: {stdout}");
        assert!(
            !String::from_utf8_lossy(&out.stderr).contains("downloading"),
            "{name} fetched before refusing"
        );
    }
}

/// A checkpoint this build CAN run gets past the gate — so the tests above are
/// measuring the gate and not some earlier failure common to every invocation.
///
/// Stops at the first thing after the gate that needs the machine, which is the
/// checkpoint file itself: the message names the cache or the file, never the
/// refusal.
#[test]
fn a_runnable_checkpoint_gets_past_the_gate() {
    let config = empty_config("runnable");
    let out = xwen()
        .args(["serve", "--config"])
        .arg(&config)
        .args(["--model-size", "35b", "--port", "0"])
        // A directory with no cache in it, so the run cannot find a checkpoint
        // and cannot download one either: it gets past the gate and stops at
        // the fetch, which is the ordering this asserts.
        .env(
            "HF_HUB_CACHE",
            std::env::temp_dir().join("xwen-gates-empty-cache"),
        )
        .env("HF_ENDPOINT", "http://127.0.0.1:1")
        .output()
        .expect("running xwen serve");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !stderr.contains("cannot be served or batched"),
        "a servable checkpoint must not be refused by the gate: {stderr}"
    );
    std::fs::remove_file(&config).unwrap();
}
