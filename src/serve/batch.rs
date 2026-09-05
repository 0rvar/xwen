//! The native batch endpoint: POST /xwen/v1/batch.
//!
//! The body is exactly the JSON document `xwen batch` reads on stdin
//! ([`crate::batch::BatchRequest`]) and the response is exactly the document it
//! prints ([`crate::batch::BatchResponse`]) — one surface, two transports. The
//! request's `model` field picks the checkpoint per request by full name
//! ("Qwen3.6-27B"), defaulting to the server's default checkpoint rather than
//! the CLI's compile-time default: a server was started around one model, and a
//! request that names none means that one. The engine lazy-loads whichever
//! checkpoint the job names, swapping the resident one out when they differ.
//!
//! Strict like the rest of the native surface: the request type is
//! `deny_unknown_fields`, and an unknown model name is a 400 — the CLI errors
//! on it too, and the queue must never carry a job the engine cannot run.

use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};

use super::types::{
    BatchJob, Cancel, CancelGuard, ClientId, Dialect, EngineEvent, RequestOrigin, Target,
};
use super::{ApiError, AppState, EVENT_CHANNEL_CAPACITY, SubmitError, native};
use crate::batch::{BatchRequest, BatchResponse, DEFAULT_MAX_TOKENS};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Instant;
use tokio::sync::mpsc;

/// This endpoint shares the native dialect's error envelope: it lives under
/// `/xwen`, and a client that can parse one can parse the other.
fn bad_request(message: impl Into<String>) -> ApiError {
    native::error(StatusCode::BAD_REQUEST, "invalid_request_error", message)
}

/// The target a batch request runs on, and the name its response document is
/// labeled with — the same rule the compat dialects use
/// ([`super::resolve_requested_model`]), so one document means the same thing on
/// every surface. Full names and this server's own model id; the CLI's short
/// aliases stay a CLI spelling.
fn resolve_model(
    request: &BatchRequest,
    served: Target,
    served_id: &str,
) -> Result<(Target, String), ApiError> {
    super::resolve_requested_model(request.model.as_deref(), served, served_id)
        .map_err(|message| bad_request(format!("model: {message}")))
}

/// A conservative token estimate for text nobody has rendered yet, for queue
/// scheduling and the watchdog deadline. Three bytes per token overestimates
/// prose (≈4 bytes/token), which errs the right way twice: the scheduler lets
/// small chat requests go first, and the deadline errs loose.
fn estimated_tokens(bytes: usize) -> usize {
    bytes / 3
}

/// The two size estimates a batch job is queued with: prompt-side tokens
/// (every item's messages, plus a declared `shared_prefix` once PER ITEM) and
/// the summed output budgets.
///
/// The prefix is counted per item even though the runner usually prefills it
/// once: the token-level dedup is conditional (two or more live items, a
/// common prefix past `MIN_SHARED_PREFIX`, `XWEN_BATCH_NO_CACHE` unset), and
/// when it does not fire every item repays the prefix in full. Counting it per
/// item matches what an inline spelling of the same document would have
/// weighed, and errs the way this whole estimate errs: the scheduler lets
/// small chats go first, and the watchdog deadline errs loose — an
/// undercounted estimate would instead arm a deadline shorter than the real
/// prefill. The runner reports measured truth later.
fn size_estimates(request: &BatchRequest) -> (usize, usize) {
    // Saturating throughout: these are client-supplied numbers feeding a
    // scheduling estimate, and a request built to overflow them should get a
    // pinned-at-max estimate, not an overflow.
    let prefix_bytes = request
        .shared_prefix
        .as_ref()
        .map_or(0, String::len)
        .saturating_mul(request.items.len());
    let prompt_bytes = request
        .items
        .iter()
        .flat_map(|item| &item.messages)
        .map(|m| m.content.len() + m.thinking.as_ref().map_or(0, String::len))
        .fold(prefix_bytes, usize::saturating_add);
    let max_tokens = request
        .items
        .iter()
        .map(|item| {
            item.max_tokens
                .or(request.defaults.max_tokens)
                .unwrap_or(DEFAULT_MAX_TOKENS)
        })
        .fold(0usize, usize::saturating_add);
    (estimated_tokens(prompt_bytes), max_tokens)
}

/// Queue one batch job, returning the channel its single terminal event will
/// arrive on and the guard that cancels it when the response is dropped.
fn submit_batch(
    state: &AppState,
    request: BatchRequest,
    model: Target,
    who: ClientId,
) -> Result<(mpsc::Receiver<EngineEvent>, CancelGuard), SubmitError> {
    let (prompt_tokens, max_tokens) = size_estimates(&request);
    let (events, receiver) = mpsc::channel(EVENT_CHANNEL_CAPACITY);
    let cancel = Arc::new(Cancel::default());
    let job = BatchJob {
        origin: RequestOrigin {
            id: state.next_request_id.fetch_add(1, Ordering::Relaxed),
            dialect: Dialect::Native,
            streaming: false,
            // The payload carries no client id of its own; the session header
            // is read here exactly as on the two chat routes.
            client: who.client,
            session: who.session,
        },
        request,
        model,
        max_tokens,
        cancel: Arc::clone(&cancel),
        deadline: None,
        events,
    };
    state
        .jobs
        .push(super::queue::Queued {
            job: super::types::Job::Batch(Box::new(job)),
            submitted: Instant::now(),
            prompt_tokens,
        })
        .map(|()| (receiver, CancelGuard::new(cancel)))
}

/// Drain the job's channel to its single terminal event.
async fn collect_batch(
    events: &mut mpsc::Receiver<EngineEvent>,
) -> Result<Box<BatchResponse>, super::EngineFailure> {
    while let Some(event) = events.recv().await {
        match event {
            EngineEvent::BatchDone(response) => return Ok(response),
            EngineEvent::Error {
                message,
                request_fault,
            } => {
                return Err(super::EngineFailure::Reported {
                    message,
                    request_fault,
                });
            }
            // A batch job produces no other events; anything else is ignored
            // rather than trusted.
            _ => {}
        }
    }
    Err(super::EngineFailure::Hangup)
}

pub(crate) async fn batch(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let mut request: BatchRequest = match serde_json::from_slice(&body) {
        Ok(request) => request,
        Err(e) => {
            return bad_request(format!("could not parse the request body: {e}")).into_response();
        }
    };
    // Judged before the queue, like everything a handler can judge alone: the
    // engine must never pick up a batch it cannot run.
    if request.items.is_empty() {
        return bad_request("batch: the request holds no items").into_response();
    }
    let (model, label) = match resolve_model(&request, state.default_target, &state.model_id) {
        Ok(resolved) => resolved,
        Err(e) => return e.into_response(),
    };
    // Written back so the runner and the response document agree with the
    // resolution above: an absent field means the SERVER's default here, while
    // the runner's own `BatchRequest::model()` would read it as the CLI's
    // compile-time default — on a 27B server those diverge, and the response
    // would label a 27B run as the 35B-A3B. The label is the id this server
    // answers under, which for a GGUF that is none of the official checkpoints
    // is its own file name and NOT the checkpoint it runs as: the document must
    // not claim official weights ran.
    request.model = Some(label);

    let who = ClientId::new(None, super::session_header(&headers));
    let (mut events, guard) = match submit_batch(&state, request, model, who) {
        Ok(submitted) => submitted,
        Err(SubmitError::Invalid(message)) => return bad_request(message).into_response(),
        Err(SubmitError::Overloaded) => {
            return native::error(
                StatusCode::TOO_MANY_REQUESTS,
                "overloaded_error",
                "the model is busy with other requests; retry shortly",
            )
            .with_header("retry-after", "1")
            .into_response();
        }
        Err(SubmitError::EngineGone) => {
            return native::error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "api_error",
                "the inference engine is not running",
            )
            .into_response();
        }
    };

    // Held across the drain: hyper dropping this future — the client hung up —
    // drops the guard, which cancels the batch between items (and per decoded
    // token) with its reason stated.
    let _guard = guard;
    match collect_batch(&mut events).await {
        Ok(response) => axum::Json(*response).into_response(),
        Err(super::EngineFailure::Reported {
            message,
            request_fault,
        }) => {
            let status = if request_fault {
                StatusCode::BAD_REQUEST
            } else {
                StatusCode::INTERNAL_SERVER_ERROR
            };
            let kind = if request_fault {
                "invalid_request_error"
            } else {
                "api_error"
            };
            native::error(status, kind, message).into_response()
        }
        Err(super::EngineFailure::Hangup) => native::error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "api_error",
            "the inference engine stopped before finishing the response",
        )
        .into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hub::Model;

    fn request(json: &str) -> BatchRequest {
        serde_json::from_str(json).expect("request parses")
    }

    /// The wire shape is the CLI's: the same document parses on both surfaces,
    /// and unknown fields are refused, not dropped — this is our own API.
    #[test]
    fn the_request_shape_is_the_clis() {
        let parsed = request(
            r#"{"model":"Qwen3.6-27B","defaults":{"max_tokens":64},
                "items":[{"id":"a","messages":[{"role":"user","content":"hi"}]}]}"#,
        );
        assert_eq!(parsed.model.as_deref(), Some("Qwen3.6-27B"));
        assert_eq!(parsed.items.len(), 1);
        assert!(
            serde_json::from_str::<BatchRequest>(r#"{"itmes":[{"id":"a","messages":[]}]}"#)
                .is_err(),
            "a misspelled field is a parse error"
        );
    }

    /// An absent field means the SERVER's default, which is not the runner's
    /// own compile-time default — so the handler writes the resolved name back
    /// into the request, and that written-back name is what labels the response
    /// document. This pins both halves.
    #[test]
    fn an_absent_model_is_normalized_to_the_servers_default() {
        let served = Target::official(Model::Qwen27B);
        let mut absent = request(r#"{"items":[{"id":"a","messages":[]}]}"#);
        let (target, label) = resolve_model(&absent, served, "Qwen3.6-27B").unwrap();
        absent.model = Some(label);
        assert_eq!(target, served);
        assert_eq!(
            absent.model().unwrap(),
            Model::Qwen27B,
            "the runner must read the checkpoint the handler resolved, not its own default"
        );
    }

    /// The model field selects per request: a checkpoint named by its full name
    /// wins, absence means the server's default, and anything else is a 400 —
    /// never a silent fallback, this field exists to select. The CLI's short
    /// aliases are not the wire's vocabulary and are refused here with the rest.
    #[test]
    fn the_model_field_selects_per_request() {
        let served = Target::official(Model::Qwen35BA3B);
        let named = request(r#"{"model":"Qwen3.6-27B","items":[{"id":"a","messages":[]}]}"#);
        assert_eq!(
            resolve_model(&named, served, "Qwen3.6-35B-A3B").unwrap(),
            (Target::official(Model::Qwen27B), "Qwen3.6-27B".to_string())
        );
        let absent = request(r#"{"items":[{"id":"a","messages":[]}]}"#);
        assert_eq!(
            resolve_model(&absent, served, "Qwen3.6-35B-A3B").unwrap().0,
            served,
            "absent means the server default, not the CLI's compile-time default"
        );
        for name in ["13b", "27b", "3.8"] {
            let unknown = request(&format!(
                r#"{{"model":"{name}","items":[{{"id":"a","messages":[]}}]}}"#
            ));
            let error = resolve_model(&unknown, served, "Qwen3.6-35B-A3B")
                .expect_err("only full names select");
            assert_eq!(error.status, StatusCode::BAD_REQUEST);
        }
    }

    /// A server started with a GGUF that is none of the official checkpoints
    /// answers under its own file name — the id `/v1/models` advertises — and a
    /// request naming an official checkpoint gets that checkpoint's own file,
    /// never these weights. The response label follows the same rule, so the
    /// document never claims official weights ran.
    #[test]
    fn a_custom_gguf_answers_under_its_own_id_and_no_other() {
        let served = Target::served(Model::Qwen35BA3B);
        let stem = "my-finetune-Q4_K_M";

        let by_stem = request(&format!(
            r#"{{"model":"{stem}","items":[{{"id":"a","messages":[]}}]}}"#
        ));
        assert_eq!(
            resolve_model(&by_stem, served, stem).unwrap(),
            (served, stem.to_string())
        );

        // The official checkpoint of the SAME architecture is a different file.
        let official = request(r#"{"model":"Qwen3.6-35B-A3B","items":[{"id":"a","messages":[]}]}"#);
        let (target, label) = resolve_model(&official, served, stem).unwrap();
        assert_eq!(target, Target::official(Model::Qwen35BA3B));
        assert_ne!(target, served, "an official name is not the served file");
        assert_eq!(label, "Qwen3.6-35B-A3B");

        let absent = request(r#"{"items":[{"id":"a","messages":[]}]}"#);
        assert_eq!(
            resolve_model(&absent, served, stem).unwrap(),
            (served, stem.to_string()),
            "an absent field labels the document with the served file's own id"
        );
    }

    /// The scheduling estimates: prompt bytes over three (overestimating, so
    /// batches yield to small chats), output budgets summed with the runner's
    /// own default where an item names none.
    #[test]
    fn size_estimates_cover_every_item() {
        let parsed = request(
            r#"{"defaults":{"max_tokens":100},"items":[
                {"id":"a","messages":[{"role":"user","content":"aaaaaaaaa"}]},
                {"id":"b","max_tokens":7,"messages":[{"role":"user","content":"bbb","thinking":null}]},
                {"id":"c","messages":[{"role":"assistant","content":"","thinking":"ccc"}]}]}"#,
        );
        let (prompt, output) = size_estimates(&parsed);
        assert_eq!(prompt, (9 + 3 + 3) / 3);
        assert_eq!(output, 100 + 7 + 100);

        let unbudgeted = request(r#"{"items":[{"id":"a","messages":[]}]}"#);
        assert_eq!(size_estimates(&unbudgeted).1, DEFAULT_MAX_TOKENS);
    }

    /// A declared shared_prefix weighs once PER ITEM, exactly as the same
    /// document spelled inline would — the runner's one-prefill dedup is
    /// conditional, and an estimate that assumed it would arm a watchdog
    /// deadline shorter than the worst-case prefill.
    #[test]
    fn a_shared_prefix_is_estimated_per_item() {
        let parsed = request(
            r#"{"shared_prefix":"aaaaaa","items":[
                {"id":"a","messages":[{"role":"user","content":"bbb"}]},
                {"id":"b","messages":[{"role":"user","content":"ccc"}]}]}"#,
        );
        let (prompt, _) = size_estimates(&parsed);
        assert_eq!(prompt, (6 * 2 + 3 + 3) / 3);
    }
}
