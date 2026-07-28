//! The job queue between the HTTP handlers and the engine thread: an
//! inspectable replacement for the bounded channel it grew out of.
//!
//! A channel is strict FIFO with no visibility, which is exactly wrong for a
//! batch=1 server: a 4-token title request submitted behind a 100k-token cold
//! prefill waits all of it, and a job whose client hung up cannot be removed
//! from the middle. Here dead and stale entries are swept at every push and
//! every dequeue, and the engine's dequeue ([`JobQueue::take`]) picks the job
//! with the least prefill actually required — prompt tokens minus what the KV
//! cache already holds — with an age limit so nothing starves.
//! `schedule = "fifo"` restores arrival order.

use std::sync::{Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use super::SubmitError;
use super::config::Schedule;
use super::log::{QueueEntry, ServeLog, ServeLogger};
use super::types::{EngineEvent, GenerationJob};

/// How [`choose`] orders the queue: the schedule plus the two time bounds the
/// sweep and the starvation guard run on. Carried by the queue, passed to the
/// free function so selection unit-tests without threads.
#[derive(Debug, Clone, Copy)]
pub struct SchedulePolicy {
    pub schedule: Schedule,
    /// An entry that has waited this long is dropped with an error, not served.
    pub queue_timeout: Duration,
    /// An entry that has waited this long wins over any cheaper newcomer.
    pub age_limit: Duration,
}

/// One queued job, with the two facts selection runs on.
pub struct Queued {
    pub job: GenerationJob,
    pub submitted: Instant,
    /// The encoded prompt's length, the gross cost estimate the cache discount
    /// is subtracted from.
    pub prompt_tokens: usize,
}

struct Inner {
    /// Arrival order; selection removes from the middle without disturbing it.
    jobs: Vec<Queued>,
    /// Closed at shutdown: pushes fail, and `take` drains what is left and then
    /// returns `None` instead of blocking.
    closed: bool,
}

/// The queue the handlers push into and the engine takes from. `Mutex` +
/// `Condvar` rather than a channel so the engine can inspect, sweep and reorder
/// the waiting jobs under one lock; the mutex is `Sync`, which is what axum's
/// `State` requires of everything in `AppState`.
pub struct JobQueue {
    inner: Mutex<Inner>,
    wake: Condvar,
    capacity: usize,
    policy: SchedulePolicy,
    /// Both queue log sites fire under the lock, from the engine thread and from
    /// handler threads alike, so the send behind this handle must never block.
    logger: ServeLogger,
}

impl JobQueue {
    pub fn new(capacity: usize, policy: SchedulePolicy, logger: ServeLogger) -> Self {
        Self {
            inner: Mutex::new(Inner {
                jobs: Vec::new(),
                closed: false,
            }),
            wake: Condvar::new(),
            capacity: capacity.max(1),
            policy,
            logger,
        }
    }

    /// The queue's data — a vec and a flag — is valid at every point a panic
    /// could poison the lock, so poisoning is recovered from rather than
    /// spread: one panicked engine sweep must not turn every later submit into
    /// a handler panic.
    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Non-blocking append, the shape `submit` needs: a full queue is reported
    /// as [`SubmitError::Overloaded`] for the handler to answer with its
    /// dialect's retryable status, never waited on. Crate-visible because the
    /// error type is the crate-internal submit protocol.
    pub(crate) fn push(&self, queued: Queued) -> Result<(), SubmitError> {
        let mut inner = self.lock();
        if inner.closed {
            return Err(SubmitError::EngineGone);
        }
        // Swept before the capacity check: while a long generation keeps the
        // engine away from `take`, entries whose clients cancelled or that
        // outlived the queue timeout must not pin capacity and turn every new
        // request into a spurious overload answer.
        let now = Instant::now();
        sweep(&mut inner.jobs, now, &self.policy, &self.logger);
        if inner.jobs.len() >= self.capacity {
            // The sweep above may well have changed the queue even though this
            // push did not join it.
            report(&inner.jobs, &self.logger);
            return Err(SubmitError::Overloaded);
        }
        inner.jobs.push(queued);
        report(&inner.jobs, &self.logger);
        self.wake.notify_one();
        Ok(())
    }

    /// Blocking pop: the next job the schedule picks, `None` on timeout (the
    /// idle-unload path — `None` there means "keep the countdown's promise")
    /// or once the queue is closed and drained. `hot` scores a candidate's
    /// cached prefix in tokens; the engine passes a closure over its slots, and
    /// with no model loaded everything scores 0.
    ///
    /// The timeout is measured from this call, so every taken job restarts the
    /// caller's idle countdown in full — the same semantics `recv_timeout` gave
    /// the engine loop.
    pub fn take(&self, timeout: Option<Duration>, hot: &dyn Fn(&[u32]) -> usize) -> Option<Queued> {
        // A timeout too large for an `Instant` to hold — `parse_idle_unload`
        // accepts spans up to u64::MAX seconds — is the same promise as no
        // timeout at all: block until a job arrives.
        let deadline = timeout.and_then(|timeout| Instant::now().checked_add(timeout));
        let mut inner = self.lock();
        loop {
            if let Some(queued) = choose(
                &mut inner.jobs,
                hot,
                Instant::now(),
                &self.policy,
                &self.logger,
            ) {
                return Some(queued);
            }
            if inner.closed {
                return None;
            }
            inner = match deadline {
                Some(deadline) => {
                    let now = Instant::now();
                    if now >= deadline {
                        return None;
                    }
                    self.wake
                        .wait_timeout(inner, deadline - now)
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .0
                }
                None => self
                    .wake
                    .wait(inner)
                    .unwrap_or_else(|poisoned| poisoned.into_inner()),
            };
        }
    }

    /// Stop accepting jobs and unblock the engine. Jobs already queued still
    /// come out of `take` — the engine's own dequeue check is what drops them
    /// at shutdown — so nothing is destroyed while a destructor could run.
    pub fn close(&self) {
        self.lock().closed = true;
        self.wake.notify_all();
    }

    /// Whether [`close`](Self::close) has run: how the engine tells "the
    /// server is gone" from "the idle timeout elapsed" when `take` says `None`.
    pub fn is_closed(&self) -> bool {
        self.lock().closed
    }
}

/// Say what is waiting, after something changed it. Reported from inside the
/// lock — the queue is capped at a handful of entries, so building the summary
/// costs less than the lock round trip a consumer would otherwise have to take,
/// and a consumer that took this lock could stall the engine's dequeue.
///
/// Each entry carries when it arrived rather than how long it had waited: a
/// snapshot is only produced when the queue changes, and a consumer drawing an
/// age has to be able to keep counting between two of them.
fn report(jobs: &[Queued], logger: &ServeLogger) {
    logger.log(ServeLog::QueueSnapshot(
        jobs.iter()
            .map(|queued| QueueEntry {
                id: queued.job.origin.id,
                prompt_tokens: queued.prompt_tokens,
                queued_at: queued.submitted,
            })
            .collect(),
    ));
}

/// Drop the queue entries nobody is waiting on. Runs under the queue lock from
/// both `push` (so dead entries never pin capacity) and [`choose`] (so the
/// engine never picks one). Survivors keep their arrival order.
///
/// Entries whose cancel fired or whose event channel already closed go
/// silently, and entries older than the queue timeout with a best-effort error
/// event first: their 200 already went out, so an SSE error frame (or the
/// buffered handler's 500) is the only way left to say anything. The send is
/// `try_send` — never blocking, whichever caller is holding the lock.
fn sweep(jobs: &mut Vec<Queued>, now: Instant, policy: &SchedulePolicy, logger: &ServeLogger) {
    jobs.retain(|queued| {
        if queued.job.cancel.is_cancelled() || queued.job.events.is_closed() {
            return false;
        }
        if now.duration_since(queued.submitted) >= policy.queue_timeout {
            let waited = now.duration_since(queued.submitted);
            // Best-effort: a client that stopped reading its channel gets
            // nothing, which is fine — the message is for one that is still
            // there.
            let _ = queued.job.events.try_send(EngineEvent::Error {
                message: format!(
                    "the request waited {:.0}s in the queue without reaching the model \
                     (queue_timeout is {}s); the server is saturated, retry later",
                    waited.as_secs_f64(),
                    policy.queue_timeout.as_secs()
                ),
                request_fault: false,
            });
            logger.log(ServeLog::QueuedRequestDropped {
                waited,
                queue_timeout: policy.queue_timeout,
            });
            return false;
        }
        true
    });
}

/// Sweep the dead entries out of `jobs` and pick the one to run, or `None` if
/// nothing runnable is waiting. Survivors keep their arrival order.
///
/// Selection: any survivor past the age limit wins, oldest first — and since
/// every entry older than an aged one is itself aged, that pick is the global
/// oldest, so aging never reorders. Otherwise the lowest cost wins, where cost
/// is the prefill actually required (`prompt_tokens - hot(prompt)`), ties to
/// the earlier arrival. `Schedule::Fifo` skips the cost model entirely.
pub fn choose(
    jobs: &mut Vec<Queued>,
    hot: &dyn Fn(&[u32]) -> usize,
    now: Instant,
    policy: &SchedulePolicy,
    logger: &ServeLogger,
) -> Option<Queued> {
    sweep(jobs, now, policy, logger);
    if jobs.is_empty() {
        // Still worth reporting: the sweep may have just emptied the queue.
        report(jobs, logger);
        return None;
    }

    let oldest = jobs
        .iter()
        .enumerate()
        .min_by_key(|(_, queued)| queued.submitted)
        .map(|(index, _)| index)
        .expect("jobs is non-empty");

    let picked = match policy.schedule {
        Schedule::Fifo => oldest,
        Schedule::ShortestPrefill => {
            if now.duration_since(jobs[oldest].submitted) >= policy.age_limit {
                // The starvation guard: the oldest entry has waited long
                // enough to outrank every cost estimate.
                oldest
            } else {
                jobs.iter()
                    .enumerate()
                    .min_by_key(|(_, queued)| {
                        (
                            queued.prompt_tokens.saturating_sub(hot(&queued.job.prompt)),
                            queued.submitted,
                        )
                    })
                    .map(|(index, _)| index)
                    .expect("jobs is non-empty")
            }
        }
    };

    // The one observable act of scheduling is picking a job ahead of an older
    // one; said out loud so the policy is falsifiable from the log.
    if picked != oldest {
        let cost = jobs[picked]
            .prompt_tokens
            .saturating_sub(hot(&jobs[picked].job.prompt));
        let older = jobs
            .iter()
            .filter(|queued| queued.submitted < jobs[picked].submitted)
            .count();
        logger.log(ServeLog::OutOfOrderPick {
            prompt_tokens: jobs[picked].prompt_tokens,
            cost,
            older,
            oldest_waited: now.duration_since(jobs[oldest].submitted),
        });
    }
    let taken = jobs.remove(picked);
    report(jobs, logger);
    Some(taken)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

    use crate::sampler::SamplerOptions;
    use crate::serve::log::EventLog;
    use crate::serve::types::{Cancel, CancelReason, Dialect, RequestOrigin};

    fn policy() -> SchedulePolicy {
        SchedulePolicy {
            schedule: Schedule::ShortestPrefill,
            queue_timeout: Duration::from_secs(300),
            age_limit: Duration::from_secs(20),
        }
    }

    /// Ids for the entries a test builds, distinct and increasing as `submit`
    /// hands them out.
    static NEXT_ID: AtomicU64 = AtomicU64::new(1);

    /// A queued job whose prompt is `prompt` and that arrived `waited` ago,
    /// with its event receiver kept so the entry survives the sweep.
    fn entry(
        prompt: Vec<u32>,
        waited: Duration,
        now: Instant,
    ) -> (Queued, tokio::sync::mpsc::Receiver<EngineEvent>) {
        let (events, receiver) = tokio::sync::mpsc::channel(8);
        let prompt_tokens = prompt.len();
        let queued = Queued {
            job: GenerationJob {
                origin: RequestOrigin {
                    id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
                    dialect: Dialect::Anthropic,
                    streaming: true,
                },
                prompt,
                boundary: 0,
                anchor: None,
                starts_in_thinking: false,
                max_think: None,
                max_tokens: 16,
                sampling: SamplerOptions::default(),
                stop_sequences: Vec::new(),
                tools: Vec::new(),
                grammar: None,
                cancel: Arc::new(Cancel::default()),
                deadline: None,
                events,
            },
            submitted: now - waited,
            prompt_tokens,
        };
        (queued, receiver)
    }

    fn no_cache(_: &[u32]) -> usize {
        0
    }

    /// Selection is what these tests are about; the lines it would emit are
    /// pinned in `log.rs`.
    fn logger() -> ServeLogger {
        ServeLogger::discarding()
    }

    /// Build a queue vec, keeping the receivers alive for the call.
    fn jobs(
        entries: Vec<(Vec<u32>, Duration)>,
        now: Instant,
    ) -> (Vec<Queued>, Vec<tokio::sync::mpsc::Receiver<EngineEvent>>) {
        let mut queued = Vec::new();
        let mut receivers = Vec::new();
        for (prompt, waited) in entries {
            let (q, r) = entry(prompt, waited, now);
            queued.push(q);
            receivers.push(r);
        }
        (queued, receivers)
    }

    #[test]
    fn the_shortest_prompt_wins_when_nothing_is_cached() {
        let now = Instant::now();
        let (mut queue, _rx) = jobs(
            vec![
                (vec![1; 50], Duration::from_secs(2)),
                (vec![2; 3], Duration::from_secs(1)),
                (vec![3; 10], Duration::from_secs(1)),
            ],
            now,
        );
        let picked =
            choose(&mut queue, &no_cache, now, &policy(), &logger()).expect("a job is picked");
        assert_eq!(picked.prompt_tokens, 3);
        // Survivors keep arrival order.
        let remaining: Vec<usize> = queue.iter().map(|q| q.prompt_tokens).collect();
        assert_eq!(remaining, vec![50, 10]);
    }

    #[test]
    fn equal_costs_tie_break_to_the_older_entry() {
        let now = Instant::now();
        let (mut queue, _rx) = jobs(
            vec![
                (vec![1; 5], Duration::from_secs(1)),
                (vec![2; 5], Duration::from_secs(4)),
            ],
            now,
        );
        let picked =
            choose(&mut queue, &no_cache, now, &policy(), &logger()).expect("a job is picked");
        assert_eq!(
            picked.job.prompt[0], 2,
            "the older of two equals runs first"
        );
    }

    #[test]
    fn an_aged_entry_beats_a_cheaper_younger_one() {
        let now = Instant::now();
        let (mut queue, _rx) = jobs(
            vec![
                (vec![1; 1000], Duration::from_secs(25)),
                (vec![2; 3], Duration::from_secs(1)),
            ],
            now,
        );
        let picked =
            choose(&mut queue, &no_cache, now, &policy(), &logger()).expect("a job is picked");
        assert_eq!(picked.prompt_tokens, 1000, "age outranks cost");
    }

    #[test]
    fn the_oldest_aged_entry_wins_among_several() {
        let now = Instant::now();
        let (mut queue, _rx) = jobs(
            vec![
                (vec![1; 10], Duration::from_secs(30)),
                (vec![2; 5], Duration::from_secs(40)),
                (vec![3; 1], Duration::from_secs(25)),
            ],
            now,
        );
        let picked =
            choose(&mut queue, &no_cache, now, &policy(), &logger()).expect("a job is picked");
        assert_eq!(picked.job.prompt[0], 2, "FIFO among the aged");
    }

    /// A long prompt whose prefix is already in the KV cache costs less than a
    /// short cold one — the discount is the whole point of asking the slots.
    #[test]
    fn a_hot_prefix_discount_flips_the_order() {
        let now = Instant::now();
        let (mut queue, _rx) = jobs(
            vec![
                (vec![1; 100], Duration::from_secs(1)),
                (vec![2; 40], Duration::from_secs(1)),
            ],
            now,
        );
        // The 100-token prompt has 95 tokens cached: cost 5 beats cost 40.
        let hot = |prompt: &[u32]| if prompt[0] == 1 { 95 } else { 0 };
        let picked = choose(&mut queue, &hot, now, &policy(), &logger()).expect("a job is picked");
        assert_eq!(picked.prompt_tokens, 100);

        // A discount past the prompt length saturates rather than wrapping.
        let (mut queue, _rx) = jobs(vec![(vec![1; 10], Duration::from_secs(1))], now);
        let overshoot = |_: &[u32]| 1000usize;
        assert!(choose(&mut queue, &overshoot, now, &policy(), &logger()).is_some());
    }

    #[test]
    fn fifo_reproduces_arrival_order_regardless_of_cost() {
        let now = Instant::now();
        let (mut queue, _rx) = jobs(
            vec![
                (vec![1; 1000], Duration::from_secs(3)),
                (vec![2; 1], Duration::from_secs(2)),
                (vec![3; 1], Duration::from_secs(1)),
            ],
            now,
        );
        let fifo = SchedulePolicy {
            schedule: Schedule::Fifo,
            ..policy()
        };
        let first = choose(&mut queue, &no_cache, now, &fifo, &logger()).expect("a job is picked");
        let second = choose(&mut queue, &no_cache, now, &fifo, &logger()).expect("a job is picked");
        let third = choose(&mut queue, &no_cache, now, &fifo, &logger()).expect("a job is picked");
        assert_eq!(
            [
                first.job.prompt[0],
                second.job.prompt[0],
                third.job.prompt[0]
            ],
            [1, 2, 3]
        );
    }

    /// A cancelled entry in the middle is swept without a word — its client is
    /// gone, so its channel just closes — and the survivors keep their order.
    #[test]
    fn the_sweep_drops_a_cancelled_middle_entry_silently() {
        let now = Instant::now();
        let (mut queue, mut rx) = jobs(
            vec![
                (vec![1; 10], Duration::from_secs(3)),
                (vec![2; 10], Duration::from_secs(2)),
                (vec![3; 10], Duration::from_secs(1)),
            ],
            now,
        );
        queue[1].job.cancel.cancel(CancelReason::ClientGone);
        let picked =
            choose(&mut queue, &no_cache, now, &policy(), &logger()).expect("a job is picked");
        assert_eq!(picked.job.prompt[0], 1);
        assert_eq!(queue.len(), 1);
        assert_eq!(queue[0].job.prompt[0], 3);
        // The cancelled entry's channel closed with nothing in it.
        match rx[1].try_recv() {
            Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => {}
            other => panic!("a swept cancelled entry says nothing: {other:?}"),
        }
    }

    /// An entry past the queue timeout is dropped, but its client — who got a
    /// 200 long ago — is told why over the event channel first.
    #[test]
    fn a_stale_entry_gets_an_error_event_before_it_is_dropped() {
        let now = Instant::now();
        let (mut queue, mut rx) = jobs(
            vec![
                (vec![1; 10], Duration::from_secs(400)),
                (vec![2; 10], Duration::from_secs(1)),
            ],
            now,
        );
        let picked =
            choose(&mut queue, &no_cache, now, &policy(), &logger()).expect("a job is picked");
        assert_eq!(picked.job.prompt[0], 2);
        assert!(queue.is_empty());
        match rx[0].try_recv() {
            Ok(EngineEvent::Error {
                message,
                request_fault,
            }) => {
                assert!(message.contains("queue"), "{message}");
                assert!(
                    !request_fault,
                    "a saturated server is not the request's fault"
                );
            }
            other => panic!("a stale entry owes its client an error: {other:?}"),
        }
    }

    #[test]
    fn choosing_from_an_empty_or_fully_swept_queue_yields_nothing() {
        let now = Instant::now();
        let mut empty: Vec<Queued> = Vec::new();
        assert!(choose(&mut empty, &no_cache, now, &policy(), &logger()).is_none());

        let (mut queue, _rx) = jobs(vec![(vec![1; 10], Duration::from_secs(1))], now);
        queue[0].job.cancel.cancel(CancelReason::ClientGone);
        assert!(choose(&mut queue, &no_cache, now, &policy(), &logger()).is_none());
        assert!(queue.is_empty());
    }

    #[test]
    fn push_at_capacity_reports_overloaded() {
        let queue = JobQueue::new(2, policy(), logger());
        let now = Instant::now();
        let mut receivers = Vec::new();
        for id in 0..2 {
            let (q, r) = entry(vec![id; 4], Duration::ZERO, now);
            receivers.push(r);
            queue.push(q).expect("under capacity");
        }
        let (q, _r) = entry(vec![9; 4], Duration::ZERO, now);
        assert!(matches!(queue.push(q), Err(SubmitError::Overloaded)));
    }

    /// Capacity counts jobs somebody is still waiting on: a queue full of
    /// entries whose clients hung up sweeps them at push time instead of
    /// answering a live request with a spurious overload.
    #[test]
    fn a_push_sweeps_dead_entries_before_judging_capacity() {
        let queue = JobQueue::new(2, policy(), logger());
        let now = Instant::now();
        for id in 0..2 {
            let (q, r) = entry(vec![id; 4], Duration::ZERO, now);
            // The dropped receiver is the hung-up client.
            drop(r);
            queue.push(q).expect("under capacity");
        }
        let (q, _r) = entry(vec![9; 4], Duration::ZERO, now);
        queue.push(q).expect("the dead entries make room");
    }

    /// A push also sweeps entries past the queue timeout, and their clients —
    /// still reading — get the same error event the dequeue sweep sends.
    #[test]
    fn a_push_sweeps_stale_entries_and_tells_their_clients() {
        let queue = JobQueue::new(1, policy(), logger());
        let now = Instant::now();
        let (stale, mut rx) = entry(vec![1; 4], Duration::from_secs(400), now);
        queue.push(stale).expect("under capacity");
        let (q, _r) = entry(vec![2; 4], Duration::ZERO, now);
        queue.push(q).expect("the stale entry makes room");
        match rx.try_recv() {
            Ok(EngineEvent::Error { message, .. }) => {
                assert!(message.contains("queue"), "{message}")
            }
            other => panic!("a swept stale entry owes its client an error: {other:?}"),
        }
    }

    /// A timeout too large for an `Instant` to hold means "block until a job
    /// arrives" — it must never panic the engine thread doing deadline
    /// arithmetic. `parse_idle_unload` accepts spans up to u64::MAX seconds.
    #[test]
    fn a_timeout_past_what_instants_measure_is_no_timeout_at_all() {
        let queue = JobQueue::new(4, policy(), logger());
        let (q, _r) = entry(vec![3; 4], Duration::ZERO, Instant::now());
        queue.push(q).expect("the queue has room");
        let taken = queue
            .take(Some(Duration::from_secs(u64::MAX)), &no_cache)
            .expect("the queued job comes out");
        assert_eq!(taken.job.prompt[0], 3);
    }

    #[test]
    fn take_blocks_until_a_push_wakes_it() {
        let queue = Arc::new(JobQueue::new(4, policy(), logger()));
        let pusher = {
            let queue = Arc::clone(&queue);
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(50));
                let (q, r) = entry(vec![7; 4], Duration::ZERO, Instant::now());
                queue.push(q).expect("the queue has room");
                r
            })
        };
        let taken = queue
            .take(None, &no_cache)
            .expect("the push wakes the blocked take");
        assert_eq!(taken.job.prompt[0], 7);
        drop(pusher.join().expect("the pusher finishes"));
    }

    #[test]
    fn take_honors_its_timeout_when_nothing_arrives() {
        let queue = JobQueue::new(4, policy(), logger());
        let started = Instant::now();
        assert!(
            queue
                .take(Some(Duration::from_millis(50)), &no_cache)
                .is_none()
        );
        assert!(started.elapsed() >= Duration::from_millis(50));
        assert!(!queue.is_closed(), "a timeout is not a shutdown");
    }

    #[test]
    fn close_unblocks_take_and_fails_later_pushes() {
        let queue = Arc::new(JobQueue::new(4, policy(), logger()));
        let closer = {
            let queue = Arc::clone(&queue);
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(50));
                queue.close();
            })
        };
        assert!(queue.take(None, &no_cache).is_none());
        assert!(queue.is_closed());
        closer.join().expect("the closer finishes");

        let (q, _r) = entry(vec![1; 4], Duration::ZERO, Instant::now());
        assert!(matches!(queue.push(q), Err(SubmitError::EngineGone)));
    }

    /// Jobs already accepted still come out after `close`, so shutdown drains
    /// through the engine's own dequeue check instead of destroying them here.
    #[test]
    fn a_closed_queue_drains_before_reporting_none() {
        let queue = JobQueue::new(4, policy(), logger());
        let (q, _r) = entry(vec![5; 4], Duration::ZERO, Instant::now());
        queue.push(q).expect("the queue has room");
        queue.close();
        assert_eq!(
            queue
                .take(None, &no_cache)
                .expect("the accepted job drains")
                .job
                .prompt[0],
            5
        );
        assert!(queue.take(None, &no_cache).is_none());
    }

    /// Every change to what is waiting is reported, so a consumer never has to
    /// take the queue lock to find out — pushes and dequeues alike, and with the
    /// ids submit handed out, so an entry can be followed from here to the job
    /// record it ends in.
    #[test]
    fn the_queue_reports_what_is_waiting_after_every_change() {
        let (logger, events) = crate::serve::log::collecting();
        let queue = JobQueue::new(4, policy(), logger);
        let now = Instant::now();
        let (first, _r1) = entry(vec![1; 40], Duration::from_secs(2), now);
        let (second, _r2) = entry(vec![2; 10], Duration::ZERO, now);
        let (first_id, second_id) = (first.job.origin.id, second.job.origin.id);
        queue.push(first).expect("room");
        queue.push(second).expect("room");

        assert_eq!(
            snapshots(&events),
            vec![vec![(first_id, 40)], vec![(first_id, 40), (second_id, 10)],],
            "each push reports the queue it joined"
        );

        let taken = queue
            .take(Some(Duration::ZERO), &no_cache)
            .expect("a job is ready");
        assert_eq!(taken.job.origin.id, second_id, "the cheaper job runs first");
        assert_eq!(
            snapshots(&events),
            vec![vec![(first_id, 40)]],
            "the dequeue reports what it left behind"
        );

        let _ = queue.take(Some(Duration::ZERO), &no_cache);
        assert_eq!(
            snapshots(&events),
            vec![Vec::new()],
            "an emptied queue is reported empty"
        );
    }

    /// An entry is reported by when it arrived, not by how long it had waited
    /// when the snapshot was taken: snapshots only happen when the queue
    /// changes, and a consumer drawing an age has to keep counting between two
    /// of them.
    #[test]
    fn a_snapshot_carries_arrival_instants_rather_than_ages() {
        let (logger, events) = crate::serve::log::collecting();
        let queue = JobQueue::new(4, policy(), logger);
        let now = Instant::now();
        let (waiting, _rx) = entry(vec![1; 40], Duration::from_secs(9), now);
        let submitted = waiting.submitted;
        queue.push(waiting).expect("room");

        let reported = events
            .drain()
            .into_iter()
            .find_map(|event| match event {
                ServeLog::QueueSnapshot(entries) => Some(entries),
                _ => None,
            })
            .expect("the push reported the queue");
        assert_eq!(reported[0].queued_at, submitted);
    }

    /// The queue snapshots so far, as `(id, prompt tokens)` per entry.
    fn snapshots(events: &EventLog) -> Vec<Vec<(u64, usize)>> {
        events
            .drain()
            .into_iter()
            .filter_map(|event| match event {
                ServeLog::QueueSnapshot(entries) => Some(
                    entries
                        .into_iter()
                        .map(|entry| (entry.id, entry.prompt_tokens))
                        .collect(),
                ),
                _ => None,
            })
            .collect()
    }

    /// The cost closure runs while the caller holds the queue lock, so `take`
    /// must feed it every candidate exactly as `choose` does — this pins the
    /// wiring, not the policy.
    #[test]
    fn take_scores_candidates_through_the_hot_closure() {
        let queue = JobQueue::new(4, policy(), logger());
        let now = Instant::now();
        let (a, _ra) = entry(vec![1; 100], Duration::ZERO, now);
        let (b, _rb) = entry(vec![2; 40], Duration::ZERO, now);
        queue.push(a).expect("room");
        queue.push(b).expect("room");
        let calls = AtomicUsize::new(0);
        let hot = |prompt: &[u32]| {
            calls.fetch_add(1, Ordering::Relaxed);
            if prompt[0] == 1 { 95 } else { 0 }
        };
        let taken = queue
            .take(Some(Duration::ZERO), &hot)
            .expect("a job is ready");
        assert_eq!(taken.prompt_tokens, 100, "the discount decided it");
        assert!(calls.load(Ordering::Relaxed) >= 2);
    }
}
