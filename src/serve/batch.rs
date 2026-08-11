//! The native batch endpoint: POST /xwen/v1/batch.
//!
//! The body is exactly the JSON document `xwen batch` reads on stdin
//! ([`crate::batch::BatchRequest`]) and the response is exactly the document it
//! prints ([`crate::batch::BatchResponse`]) — one surface, two transports. The
//! request's `model` field picks the checkpoint per request, `27b` or `35b`,
//! defaulting to the server's default checkpoint rather than the CLI's
//! compile-time default: a server was started around one model, and a request
//! that names none means that one. The engine lazy-loads whichever checkpoint
//! the job names, swapping the resident one out when they differ.
//!
//! Strict like the rest of the native surface: the request type is
//! `deny_unknown_fields`, and an unknown model name is a 400 — the CLI errors
//! on it too, and the queue must never carry a job the engine cannot run.

use axum::body::Bytes;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

use super::types::{BatchJob, Cancel, CancelGuard, Dialect, EngineEvent, RequestOrigin};
use super::{ApiError, AppState, EVENT_CHANNEL_CAPACITY, SubmitError, native};
use crate::batch::{BatchRequest, BatchResponse, DEFAULT_MAX_TOKENS};
use crate::hub::Model;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Instant;
use tokio::sync::mpsc;

/// This endpoint shares the native dialect's error envelope: it lives under
/// `/xwen`, and a client that can parse one can parse the other.
fn bad_request(message: impl Into<String>) -> ApiError {
    native::error(StatusCode::BAD_REQUEST, "invalid_request_error", message)
}

/// The checkpoint a batch request runs on: the one it names, or the server's
/// default when it names none. Unlike the compat dialects this is strict — the
/// field exists to select, so a name that selects nothing is an error, not an
/// echo.
fn resolve_model(request: &BatchRequest, default: Model) -> Result<Model, ApiError> {
    match &request.model {
        Some(name) => name
            .parse()
            .map_err(|e: String| bad_request(format!("model: {e}"))),
        None => Ok(default),
    }
}

/// A conservative token estimate for text nobody has rendered yet, for queue
/// scheduling and the watchdog deadline. Three bytes per token overestimates
/// prose (≈4 bytes/token), which errs the right way twice: the scheduler lets
/// small chat requests go first, and the deadline errs loose.
fn estimated_tokens(bytes: usize) -> usize {
    bytes / 3
}

/// The two size estimates a batch job is queued with: prompt-side tokens
/// (every item's messages, shared prefix counted once per item — the runner
/// reports measured truth later) and the summed output budgets.
fn size_estimates(request: &BatchRequest) -> (usize, usize) {
    // Saturating throughout: these are client-supplied numbers feeding a
    // scheduling estimate, and a request built to overflow them should get a
    // pinned-at-max estimate, not an overflow.
    let prompt_bytes = request
        .items
        .iter()
        .flat_map(|item| &item.messages)
        .map(|m| m.content.len() + m.thinking.as_ref().map_or(0, String::len))
        .fold(0usize, usize::saturating_add);
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
    model: Model,
) -> Result<(mpsc::Receiver<EngineEvent>, CancelGuard), SubmitError> {
    let (prompt_tokens, max_tokens) = size_estimates(&request);
    let (events, receiver) = mpsc::channel(EVENT_CHANNEL_CAPACITY);
    let cancel = Arc::new(Cancel::default());
    let job = BatchJob {
        origin: RequestOrigin {
            id: state.next_request_id.fetch_add(1, Ordering::Relaxed),
            dialect: Dialect::Native,
            streaming: false,
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

pub(crate) async fn batch(State(state): State<AppState>, body: Bytes) -> Response {
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
    let model = match resolve_model(&request, state.default_size) {
        Ok(model) => model,
        Err(e) => return e.into_response(),
    };
    // Written back so the runner and the response document agree with the
    // resolution above: an absent field means the SERVER's default here, while
    // the runner's own `BatchRequest::model()` would read it as the CLI's
    // compile-time default — on a 27B server those diverge, and the response
    // would label a 27B run "35b".
    request.model = Some(model.to_string());

    let (mut events, guard) = match submit_batch(&state, request, model) {
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

    fn request(json: &str) -> BatchRequest {
        serde_json::from_str(json).expect("request parses")
    }

    /// The wire shape is the CLI's: the same document parses on both surfaces,
    /// and unknown fields are refused, not dropped — this is our own API.
    #[test]
    fn the_request_shape_is_the_clis() {
        let parsed = request(
            r#"{"model":"27b","defaults":{"max_tokens":64},
                "items":[{"id":"a","messages":[{"role":"user","content":"hi"}]}]}"#,
        );
        assert_eq!(parsed.model.as_deref(), Some("27b"));
        assert_eq!(parsed.items.len(), 1);
        assert!(
            serde_json::from_str::<BatchRequest>(r#"{"itmes":[{"id":"a","messages":[]}]}"#)
                .is_err(),
            "a misspelled field is a parse error"
        );
    }

    /// The handler's resolution and the runner's own `BatchRequest::model()`
    /// must agree, and for an absent field they naturally do not (server
    /// default vs the CLI's compile-time default) — which is why the handler
    /// writes the resolved name back into the request before queueing it. This
    /// pins the write-back's effect: after it, the runner reads the same
    /// checkpoint the job runs on, so the response document is labeled with
    /// the model that actually ran.
    #[test]
    fn an_absent_model_is_normalized_to_the_servers_default() {
        let mut absent = request(r#"{"items":[{"id":"a","messages":[]}]}"#);
        let resolved = resolve_model(&absent, Model::Qwen27B).unwrap();
        absent.model = Some(resolved.to_string());
        assert_eq!(
            absent.model().unwrap(),
            Model::Qwen27B,
            "the runner must read the checkpoint the handler resolved, not its own default"
        );
    }

    /// The model field selects per request: a named checkpoint wins, absence
    /// means the server's default, and an unknown name is a 400 — never a
    /// silent fallback, this field exists to select.
    #[test]
    fn the_model_field_selects_per_request() {
        let named = request(r#"{"model":"27b","items":[{"id":"a","messages":[]}]}"#);
        assert_eq!(
            resolve_model(&named, Model::Qwen35BA3B).unwrap(),
            Model::Qwen27B
        );
        let absent = request(r#"{"items":[{"id":"a","messages":[]}]}"#);
        assert_eq!(
            resolve_model(&absent, Model::Qwen27B).unwrap(),
            Model::Qwen27B,
            "absent means the server default, not the CLI's compile-time default"
        );
        let unknown = request(r#"{"model":"13b","items":[{"id":"a","messages":[]}]}"#);
        let error = resolve_model(&unknown, Model::Qwen27B).expect_err("unknown model is refused");
        assert_eq!(error.status, StatusCode::BAD_REQUEST);
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
}
