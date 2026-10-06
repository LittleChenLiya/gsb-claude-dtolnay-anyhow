//! Explicit one-shot handoff of a worker task's error to its supervisor.
//!
//! This module provides a small container for the moment at which a worker
//! task fails and wants its supervisor to observe the failure. The worker
//! deposits the original [`anyhow::Error`] together with an explicit task
//! identifier and the lifecycle stage at which the failure occurred; the
//! supervisor waits for, takes, or cancels the handoff.
//!
//! # Why not just send a string?
//!
//! The handoff stores the *original* `anyhow::Error`, not a clone of its
//! `to_string()` output. The supervisor therefore keeps the full cause
//! chain and can downcast to the worker's concrete error type. A string
//! would throw both of those away.
//!
//! # One-shot semantics
//!
//! The error can be taken exactly once. The first call to
//! [`HandoffConsumer::take`] returns the error; every later call returns
//! [`TakeError::AlreadyTaken`].
//!
//! # Cancellation
//!
//! [`HandoffConsumer::cancel`] wakes every thread blocked in
//! [`HandoffConsumer::wait`]. It does not destroy an error that has already
//! been deposited and still has an owner: a later `take` still succeeds.
//!
//! # Task identifiers
//!
//! The [`TaskId`] is supplied explicitly when the handoff is created. It is
//! never inferred from the name of the thread that happens to run the task.
//!
//! # Example
//!
//! ```
//! use anyhow::handoff::{self, Stage, TaskId, WaitOutcome};
//! use std::fmt;
//! use std::sync::{Arc, Barrier};
//! use std::thread;
//!
//! #[derive(Debug)]
//! struct WorkerError {
//!     code: i32,
//! }
//!
//! impl fmt::Display for WorkerError {
//!     fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
//!         write!(formatter, "worker error (code {})", self.code)
//!     }
//! }
//!
//! impl std::error::Error for WorkerError {}
//!
//! let (producer, consumer) = handoff::new(TaskId(7));
//! let start = Arc::new(Barrier::new(2));
//! let start2 = Arc::clone(&start);
//! let worker = thread::spawn(move || {
//!     start2.wait();
//!     producer.fail(Stage::Running, anyhow::anyhow!(WorkerError { code: 3 }));
//! });
//! start.wait();
//! assert_eq!(consumer.wait(), WaitOutcome::Ready);
//! let handed = consumer.take().unwrap();
//! assert_eq!(handed.task_id(), TaskId(7));
//! assert_eq!(handed.stage(), Stage::Running);
//! let worker_error = handed.into_error().downcast::<WorkerError>().unwrap();
//! assert_eq!(worker_error.code, 3);
//! worker.join().unwrap();
//! ```

use crate::Error;
use core::fmt;
use std::sync::{Arc, Condvar, Mutex};

/// Identifier of the task that produced a handed-off error.
///
/// The identifier is chosen by the application and passed explicitly to
/// [`new`]; it is never derived from a thread name.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TaskId(pub u64);

/// The lifecycle stage at which a task's error was handed off.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Stage {
    /// The task had not started running yet.
    Created,
    /// The task was running.
    Running,
    /// The task was shutting down.
    ShuttingDown,
}

/// The outcome of waiting for a handoff.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum WaitOutcome {
    /// An error is ready to be taken.
    Ready,
    /// The handoff was cancelled before an error could be taken.
    Cancelled,
    /// The producer finished without depositing an error.
    Finished,
}

/// An error handed off from a worker task to its supervisor.
///
/// The original [`anyhow::Error`] is preserved: its cause chain and
/// downcasting behavior are intact. The [`Backtrace`] captured by the error
/// is never printed by this type's `Debug` or `Display` implementations.
///
/// [`Backtrace`]: std::backtrace::Backtrace
pub struct HandedOff {
    task_id: TaskId,
    stage: Stage,
    error: Error,
}

impl HandedOff {
    /// The identifier of the task that produced the error.
    pub fn task_id(&self) -> TaskId {
        self.task_id
    }

    /// The lifecycle stage at which the error was produced.
    pub fn stage(&self) -> Stage {
        self.stage
    }

    /// A shared reference to the original error.
    pub fn error(&self) -> &Error {
        &self.error
    }

    /// The original error.
    pub fn into_error(self) -> Error {
        self.error
    }
}

impl fmt::Debug for HandedOff {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter
            .debug_struct("HandedOff")
            .field("task_id", &self.task_id)
            .field("stage", &self.stage)
            .field("error", &RedactedError(&self.error))
            .finish()
    }
}

impl fmt::Display for HandedOff {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(
            formatter,
            "task {:?} failed at stage {:?}: {:#}",
            self.task_id.0, self.stage, self.error
        )
    }
}

/// Prints an error's cause chain using `Display`, never its backtrace.
struct RedactedError<'a>(&'a Error);

impl fmt::Debug for RedactedError<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        // `{:#}` prints the outermost error and its causes, never the
        // backtrace.
        write!(formatter, "{:#}", self.0)
    }
}

/// The error could not be taken from the handoff.
#[derive(Debug)]
#[non_exhaustive]
pub enum TakeError {
    /// The handoff was cancelled before an error was deposited.
    Cancelled,
    /// The error was already taken by an earlier call.
    AlreadyTaken,
    /// No error has been deposited yet.
    NotReady,
}

impl fmt::Display for TakeError {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match self {
            TakeError::Cancelled => formatter.write_str("handoff was cancelled"),
            TakeError::AlreadyTaken => formatter.write_str("error was already taken"),
            TakeError::NotReady => formatter.write_str("no error has been deposited yet"),
        }
    }
}

impl std::error::Error for TakeError {}

/// Type-state marker for a [`HandoffProducer`] that may still fail.
pub struct Live {
    _priv: (),
}

/// Type-state marker for a [`HandoffProducer`] that has deposited its error.
pub struct Delivered {
    _priv: (),
}

/// The producer (worker) side of an error handoff.
///
/// The producer is held by the worker task. It can fail exactly once:
/// [`HandoffProducer::fail`] consumes the producer and transitions it from
/// the [`Live`] state to the [`Delivered`] state. Dropping a [`Live`]
/// producer without failing signals clean completion to the supervisor.
pub struct HandoffProducer<State = Live> {
    inner: Arc<HandoffInner>,
    _state: State,
}

impl HandoffProducer<Live> {
    /// Deposit the worker's error at the given lifecycle stage.
    ///
    /// Consumes the producer: a task can only hand off its error once. The
    /// original error is stored unchanged, so the supervisor can downcast
    /// to the concrete type and walk the cause chain.
    pub fn fail(self, stage: Stage, error: Error) -> HandoffProducer<Delivered> {
        // Wrap self so its Drop impl does not run when we move the Arc out.
        let this = core::mem::ManuallyDrop::new(self);
        {
            let mut state = this.inner.state.lock().unwrap();
            match &*state {
                HandoffState::Pending | HandoffState::Cancelled => {
                    *state = HandoffState::Ready { stage, error };
                }
                HandoffState::Ready { .. } | HandoffState::Taken | HandoffState::Finished => {
                    unreachable!("a Live producer can only fail once");
                }
            }
        }
        // SAFETY: `this` is wrapped in ManuallyDrop and is never dropped, so
        // the Arc read here is not double-freed. The read Arc is moved into
        // the returned Delivered producer.
        let inner = unsafe { core::ptr::read(&this.inner) };
        inner.condvar.notify_all();
        HandoffProducer {
            inner,
            _state: Delivered { _priv: () },
        }
    }
}

impl<State> Drop for HandoffProducer<State> {
    fn drop(&mut self) {
        // A Live producer that never failed leaves the state Pending; signal
        // clean completion. A Delivered producer leaves the state Ready (the
        // error still has an owner), so nothing is destroyed here.
        let mut state = self.inner.state.lock().unwrap();
        if matches!(*state, HandoffState::Pending) {
            *state = HandoffState::Finished;
            self.inner.condvar.notify_all();
        }
    }
}

impl<State> fmt::Debug for HandoffProducer<State> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter
            .debug_struct("HandoffProducer")
            .field("task_id", &self.inner.task_id)
            .finish_non_exhaustive()
    }
}

/// The consumer (supervisor) side of an error handoff.
///
/// The consumer can wait for, take, or cancel the handoff. It is cheaply
/// cloneable so that multiple supervisor tasks can observe the same handoff.
#[derive(Clone)]
pub struct HandoffConsumer {
    inner: Arc<HandoffInner>,
}

impl HandoffConsumer {
    /// Block until an error is deposited, the producer finishes, or the
    /// handoff is cancelled.
    pub fn wait(&self) -> WaitOutcome {
        let mut state = self.inner.state.lock().unwrap();
        loop {
            match &*state {
                HandoffState::Ready { .. } => return WaitOutcome::Ready,
                HandoffState::Cancelled => return WaitOutcome::Cancelled,
                HandoffState::Finished => return WaitOutcome::Finished,
                HandoffState::Taken => return WaitOutcome::Cancelled,
                HandoffState::Pending => {
                    state = self.inner.condvar.wait(state).unwrap();
                }
            }
        }
    }

    /// Take the handed-off error.
    ///
    /// The first call returns the error; every later call returns
    /// [`TakeError::AlreadyTaken`]. If the handoff was cancelled before an
    /// error was deposited, returns [`TakeError::Cancelled`].
    pub fn take(&self) -> Result<HandedOff, TakeError> {
        let mut state = self.inner.state.lock().unwrap();
        match core::mem::replace(&mut *state, HandoffState::Taken) {
            HandoffState::Ready { stage, error } => Ok(HandedOff {
                task_id: self.inner.task_id,
                stage,
                error,
            }),
            HandoffState::Taken => {
                *state = HandoffState::Taken;
                Err(TakeError::AlreadyTaken)
            }
            HandoffState::Cancelled => {
                *state = HandoffState::Cancelled;
                Err(TakeError::Cancelled)
            }
            HandoffState::Pending => {
                *state = HandoffState::Pending;
                Err(TakeError::NotReady)
            }
            HandoffState::Finished => {
                *state = HandoffState::Finished;
                Err(TakeError::Cancelled)
            }
        }
    }

    /// Cancel the handoff.
    ///
    /// Wakes every thread blocked in [`wait`](Self::wait). Does not destroy
    /// an error that has already been deposited and still has an owner: a
    /// later [`take`](Self::take) still succeeds.
    pub fn cancel(&self) {
        let mut state = self.inner.state.lock().unwrap();
        match &*state {
            HandoffState::Pending => *state = HandoffState::Cancelled,
            HandoffState::Ready { .. } => {
                // Keep the error: it still has an owner. Wake waiters so
                // they can observe the ready state and take it.
            }
            HandoffState::Taken | HandoffState::Cancelled | HandoffState::Finished => {}
        }
        self.inner.condvar.notify_all();
    }
}

impl fmt::Debug for HandoffConsumer {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter
            .debug_struct("HandoffConsumer")
            .field("task_id", &self.inner.task_id)
            .finish_non_exhaustive()
    }
}

struct HandoffInner {
    task_id: TaskId,
    state: Mutex<HandoffState>,
    condvar: Condvar,
}

enum HandoffState {
    /// No error deposited yet; the producer is still live.
    Pending,
    /// An error is available for taking.
    Ready { stage: Stage, error: Error },
    /// The error was taken.
    Taken,
    /// The handoff was cancelled by the consumer.
    Cancelled,
    /// The producer dropped without depositing an error.
    Finished,
}

/// Create an explicit error handoff for the given task.
///
/// Returns the producer (worker) side and the consumer (supervisor) side.
/// The `task_id` is stored verbatim and is never inferred from a thread
/// name.
pub fn new(task_id: TaskId) -> (HandoffProducer<Live>, HandoffConsumer) {
    let inner = Arc::new(HandoffInner {
        task_id,
        state: Mutex::new(HandoffState::Pending),
        condvar: Condvar::new(),
    });
    (
        HandoffProducer {
            inner: Arc::clone(&inner),
            _state: Live { _priv: () },
        },
        HandoffConsumer { inner },
    )
}
