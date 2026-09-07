//! Startup refusals, exercised through the real binary.
//!
//! These are ORDERING tests, and ordering is the one thing a unit test on a
//! predicate cannot see. `xwen serve --model-size <an unrunnable checkpoint>`
//! used to identify it, download eight gigabytes, start the server, list the
//! model on `/v1/models` and only then die on the first request — with every
//! individual predicate answering correctly the whole way down. What was wrong
//! was where the question got asked, so the test has to run the thing that asks
//! it.
//!
//! Two entries are refused today and for different reasons: the Z-Image text
//! encoder, an encode-only checkpoint over weights with a zero-filled layer,
//! and the Z-Image-Turbo pipeline, which is a text-to-image model and not a
//! language model at all. Each refusal names the command that does work, and
//! the cases below assert which one, because sending someone from the pipeline
//! to `xwen encode-text` would be a wrong answer that still passes a test for
//! "was refused". The two Qwen3-4B language models were refused here until
//! their layer stack landed, and are covered below by the case that says they
//! are NOT.
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
    for (alias, expected, command) in [
        ("zimage-turbo-encoder", "encode-only", "xwen encode-text"),
        ("zimage-turbo", "text-to-image", "xwen image"),
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
        assert!(stderr.contains("cannot be run"), "{alias}: {stderr}");
        assert!(stderr.contains(expected), "{alias}: {stderr}");
        // Which command to reach for instead, since that is the whole reason
        // these two entries have separate sentences.
        assert!(stderr.contains(command), "{alias}: {stderr}");
        // The refusal has to happen before the download, and a message about
        // fetching would mean it did not.
        assert!(
            !stderr.contains("downloading"),
            "{alias} fetched before refusing: {stderr}"
        );
    }
    std::fs::remove_file(&config).unwrap();
}

/// The same two refusals on the one-shot CLI surfaces, which apply the gate at
/// a different call site from serve's and could stop applying it on their own.
///
/// `generate` and `chat` do not move cache state the way serve and batch do,
/// but they run the graph, and the pipeline entry has no graph to run: without
/// the gate `generate --model-size zimage-turbo` would try to open a
/// `model_index.json` as a checkpoint after a 32.9 GB download.
#[test]
fn the_one_shot_surfaces_refuse_both_z_image_entries() {
    for (alias, expected, command) in [
        ("zimage-turbo-encoder", "encode-only", "xwen encode-text"),
        ("zimage-turbo", "text-to-image", "xwen image"),
    ] {
        // `chat` takes no `--prompt`; it reads a REPL it never gets to.
        for (subcommand, extra) in [("generate", vec!["--prompt", "hi"]), ("chat", Vec::new())] {
            let out = xwen()
                .args([subcommand, "--model-size", alias])
                .args(&extra)
                .output()
                .unwrap_or_else(|e| panic!("running xwen {subcommand}: {e}"));
            let stderr = String::from_utf8_lossy(&out.stderr);
            assert!(
                !out.status.success(),
                "{subcommand} --model-size {alias} ran; it must refuse\n{stderr}"
            );
            assert!(
                stderr.contains("cannot be run"),
                "{subcommand} {alias}: {stderr}"
            );
            assert!(stderr.contains(expected), "{subcommand} {alias}: {stderr}");
            assert!(stderr.contains(command), "{subcommand} {alias}: {stderr}");
            assert!(
                !stderr.contains("downloading"),
                "{subcommand} {alias} fetched before refusing: {stderr}"
            );
        }
    }
}

/// The same gate on `xwen batch`, which names its checkpoint in the payload
/// rather than in a flag and moves cache state for the same reason serve does.
///
/// Batch reports a whole-request failure as a JSON document on stdout and exits
/// 1, so that is where the message is.
#[test]
fn batch_refuses_an_unrunnable_checkpoint_named_in_its_payload() {
    for (name, expected, command) in [
        (
            "Z-Image-Turbo-text-encoder",
            "encode-only",
            "xwen encode-text",
        ),
        ("Z-Image-Turbo", "text-to-image", "xwen image"),
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
        assert!(stdout.contains("cannot be run"), "{name}: {stdout}");
        assert!(stdout.contains(expected), "{name}: {stdout}");
        assert!(stdout.contains(command), "{name}: {stdout}");
        assert!(
            !String::from_utf8_lossy(&out.stderr).contains("downloading"),
            "{name} fetched before refusing"
        );
    }
}

/// Run a command against an EMPTY hub cache and an endpoint that refuses
/// connections, and return its output.
///
/// Everything past the servable gate then stops at the same place: the run
/// resolves its checkpoint, finds nothing cached, announces the download and
/// fails to reach the hub. That announcement is the positive evidence the gate
/// was passed — an assertion that the refusal is merely ABSENT would be
/// satisfied by a run that died of a typo in its arguments.
fn past_the_gate(args: &[&str], stdin: Option<&str>) -> (String, String, bool) {
    let mut cmd = xwen();
    cmd.args(args)
        .env(
            "HF_HUB_CACHE",
            std::env::temp_dir().join("xwen-gates-empty-cache"),
        )
        .env("HF_ENDPOINT", "http://127.0.0.1:1");
    let out = match stdin {
        None => cmd.output().expect("running xwen"),
        Some(payload) => {
            let mut child = cmd
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .expect("running xwen");
            child
                .stdin
                .take()
                .unwrap()
                .write_all(payload.as_bytes())
                .unwrap();
            child.wait_with_output().expect("waiting for xwen")
        }
    };
    (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
        out.status.success(),
    )
}

/// What a run that got past the gate and stopped at the empty cache looks like.
fn assert_reached_the_fetch(what: &str, repo: &str, stdout: &str, stderr: &str, ok: bool) {
    let both = format!("{stderr}{stdout}");
    assert!(
        !ok,
        "{what}: an empty cache cannot produce a successful run"
    );
    assert!(
        !both.contains("cannot be run"),
        "{what} must not be refused by the gate: {both}"
    );
    assert!(
        both.contains("is not in the Hugging Face cache"),
        "{what} must reach the download notice: {both}"
    );
    assert!(
        both.contains(repo),
        "{what} must name the repo it tried to fetch: {both}"
    );
    assert!(
        both.contains("fetching"),
        "{what} must fail at the fetch, which is the next thing after the gate: {both}"
    );
}
/// A checkpoint this build CAN run gets past the gate — so the tests above are
/// measuring the gate and not some earlier failure common to every invocation.
///
/// Stops at the first thing after the gate that needs the machine — the
/// checkpoint file itself — and SAYS so: the download notice naming the repo
/// is what proves the gate was reached and passed rather than that the run
/// fell over somewhere earlier for a reason of its own.
///
/// Both cache-moving surfaces, because both apply the gate and the two resolve
/// their checkpoint by different routes: serve from `--model-size`, batch from
/// the payload.
#[test]
fn a_runnable_checkpoint_gets_past_the_gate() {
    let config = empty_config("runnable");
    let (stdout, stderr, ok) = past_the_gate(
        &[
            "serve",
            "--config",
            config.to_str().unwrap(),
            "--model-size",
            "35b",
            "--port",
            "0",
        ],
        None,
    );
    assert_reached_the_fetch(
        "serve --model-size 35b",
        "ggml-org/Qwen3.6-35B-A3B-GGUF",
        &stdout,
        &stderr,
        ok,
    );

    let (stdout, stderr, ok) = past_the_gate(
        &["batch"],
        Some(
            r#"{"model":"Qwen3.6-35B-A3B","items":[{"id":"a","messages":[{"role":"user","content":"hi"}]}]}"#,
        ),
    );
    assert_reached_the_fetch(
        "batch on Qwen3.6-35B-A3B",
        "ggml-org/Qwen3.6-35B-A3B-GGUF",
        &stdout,
        &stderr,
        ok,
    );
    std::fs::remove_file(&config).unwrap();
}

/// `encode-text` accepts BOTH Z-Image aliases and encodes with the same
/// weights either way.
///
/// The pipeline alias used to mean the encoder and is what the README, the
/// docs and any operator script had been spelling; when it moved to the
/// pipeline entry, the same command line started failing on a sentence about
/// GGUF checkpoints, which a diffusion pipeline is not. Both spellings must
/// reach the encoder's own repo, and the way to see that they did is the
/// download notice naming it.
#[test]
fn encode_text_takes_either_z_image_alias() {
    for alias in ["zimage-turbo-encoder", "zimage-turbo"] {
        let out = std::env::temp_dir().join(format!("xwen-encode-{alias}.safetensors"));
        let (stdout, stderr, ok) = past_the_gate(
            &[
                "encode-text",
                "--model-size",
                alias,
                "--prompt",
                "hi",
                "-o",
                out.to_str().unwrap(),
            ],
            None,
        );
        let both = format!("{stderr}{stdout}");
        assert!(
            !both.contains("GGUF checkpoint"),
            "{alias} was refused as a GGUF: {both}"
        );
        assert!(
            !both.contains("text-to-image pipeline rather than"),
            "{alias} was refused as a pipeline: {both}"
        );
        assert_reached_the_fetch(
            &format!("encode-text --model-size {alias}"),
            "Tongyi-MAI/Z-Image-Turbo",
            &stdout,
            &stderr,
            ok,
        );
    }
}

/// The two Qwen3-4B language models are NOT refused: their layer stack landed,
/// so serve and batch run them like any other checkpoint.
///
/// Here rather than only as a predicate test because the refusal was a startup
/// ordering decision, and the thing that has to stop happening is the startup
/// bail. Both runs go on to fail at the checkpoint itself, against a cache
/// directory with nothing in it and an endpoint that refuses connections, which
/// is the next thing after the gate and proves the gate is what was passed.
#[test]
fn the_qwen3_language_models_are_no_longer_refused_at_startup() {
    let config = empty_config("qwen3_runs");
    for (alias, repo, name) in [
        ("qwen3-4b", "Qwen/Qwen3-4B", "Qwen3-4B"),
        (
            "qwen3-4b-instruct-2507",
            "Qwen/Qwen3-4B-Instruct-2507",
            "Qwen3-4B-Instruct-2507",
        ),
    ] {
        let (stdout, stderr, ok) = past_the_gate(
            &[
                "serve",
                "--config",
                config.to_str().unwrap(),
                "--model-size",
                alias,
                "--port",
                "0",
            ],
            None,
        );
        assert_reached_the_fetch(alias, repo, &stdout, &stderr, ok);

        let payload = format!(
            r#"{{"model":"{name}","items":[{{"id":"a","messages":[{{"role":"user","content":"hi"}}]}}]}}"#
        );
        let (stdout, stderr, ok) = past_the_gate(&["batch"], Some(&payload));
        assert_reached_the_fetch(name, repo, &stdout, &stderr, ok);
    }
    std::fs::remove_file(&config).unwrap();
}
