//! Explicit error handoff between a worker and its waiters.
//!
//! The ordinary way to move an [`Error`] out of a worker thread is to return
//! it through a `Result` from the thread closure, or to stash it in a
//! `std::sync::mpsc` / oneshot channel. Both approaches treat the error as an
//! opaque payload: ownership is expressed only by convention, a second
//! consumer cannot tell *why* it did not receive anything, and code that
//! merely abandons its wait can be indistinguishable from the error being
//! destroyed.
//!
//! This module provides a small, explicitly typed handoff channel that keeps
//! the original [`Error`] value intact:
//!
//! - [`Handoff::new`] receives the **task id as an explicit argument**. It is
//!   never inferred from a thread name or any other ambient property.
//! - [`Reporter::handoff`] consumes the reporter and stores the worker's
//!   error together with the **phase** the worker had reached. The error is
//!   stored by value, exactly as produced; it is never stringified, cloned,
//!   or wrapped in ad-hoc context, so the underlying type can still be
//!   recovered with [`Error::downcast`] and the source chain is untouched.
//! - [`Awaiter::take`] is the *only* way to obtain the error. The first
//!   successful call moves the [`Parcel`] out; every later call from any
//!   awaiter returns [`TakeError::AlreadyTaken`] rather than `None`, so the
//!   distinction between "not handed off yet" ([`TakeError::Pending`]) and
//!   "gone forever" is explicit.
//! - Dropping a pending [`Awaiter::wait`] future only unregisters that one
//!   waiter. The error stays alive in the slot as long as either the
//!   reporter or an awaiter owns a handle, even if every waiter that ever
//!   polled `wait` cancels before the handoff.
//!
//! Ownership is expressed by the types, not by documentation:
//!
//! ```text
//! Handoff<I, P>        owns the slot exclusively
//!   +-- split() --> Reporter<I, P>   sole producer; handoff() consumes it
//!                   Awaiter<I, P>    consumer handle; Clone, cannot produce
//!        handoff()  --> state Filled { phase, Error }  (Error owned by slot)
//!        take()     --> Parcel<I, P> (Error moved out) --> state Taken
//! ```
//!
//! # Example
//!
//! ```
//! use anyhow::handoff::{Handoff, TakeError};
//! use std::thread;
//!
//! #[derive(Clone, Copy, Debug, PartialEq, Eq)]
//! enum Phase {
//!     Connect,
//!     Flush,
//! }
//!
//! let handoff = Handoff::new(42_u64); // task id is an explicit argument
//! let (reporter, mut awaiter) = handoff.split();
//!
//! thread::spawn(move || {
//!     // reporter.handoff(Phase::Flush, err) at the failure site
//!     let err = anyhow::anyhow!("disk full");
//!     reporter.handoff(Phase::Flush, err);
//! });
//!
//! // Block until the worker reports its error.
//! let parcel = awaiter.wait_take().unwrap();
//! assert_eq!(parcel.task_id(), &42);
//! assert_eq!(parcel.phase(), &Phase::Flush);
//! assert_eq!(parcel.error().to_string(), "disk full");
//!
//! // A second take is an explicit refusal, not a silent `None`.
//! assert!(matches!(awaiter.take(), Err(TakeError::AlreadyTaken)));
//! ```
//!
//! The handoff machinery itself never renders a backtrace. A [`Parcel`]'s
//! [`Debug`] representation is deliberately the error's [`Display`] text;
//! use [`Parcel::error`] together with [`Error::chain`] if you want the full
//! anyhow rendering.

use crate::Error;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::fmt::{self, Debug, Display};
use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll, Waker};
use std::error::Error as StdError;
use std::sync::{Condvar, Mutex};

enum SlotState<P> {
    /// No error yet. The wakers registered here are the *waiters*, never the
    /// error; dropping them only unregisters a waiter.
    Empty { waiters: Vec<Waker> },
    /// The worker has handed off its error. The error is owned by the slot.
    Filled { phase: P, error: Error },
    /// The error has been moved out by the first successful `take`.
    Taken,
    /// The reporter was dropped without ever handing an error off.
    Closed,
}

struct Slot<I, P> {
    task_id: I,
    state: Mutex<SlotState<P>>,
    /// Signalled for blocking waiters ([`Awaiter::wait_take`]).
    condvar: Condvar,
}

/// The handoff slot for one task.
///
/// A `Handoff` owns the slot exclusively until it is split with
/// [`Handoff::split`] (or the [`Handoff::reporter`] / [`Handoff::awaiter`]
/// convenience constructors). Constructing the handoff requires the task id
/// to be passed explicitly; it is never derived from a thread name.
///
/// Dropping a `Handoff` that was never split simply discards the empty slot,
/// exactly like dropping an unused channel.
pub struct Handoff<I, P = &'static str> {
    slot: Arc<Slot<I, P>>,
}

/// The sole producer handle for a handoff slot.
///
/// `Reporter` is not `Clone`: at most one place in the program may hand an
/// error off for a given task. [`Reporter::handoff`] consumes the reporter,
/// making a second handoff unrepresentable. Dropping a reporter without
/// handing off transitions the slot to the [`TakeError::Closed`] state.
pub struct Reporter<I, P = &'static str> {
    slot: Arc<Slot<I, P>>,
}

/// A consumer handle for a handoff slot.
///
/// Unlike [`Reporter`], an `Awaiter` is freely `Clone` so that several
/// supervisors can race to observe the worker failure; only the first
/// successful [`Awaiter::take`] across *all* clones receives the [`Parcel`].
pub struct Awaiter<I, P = &'static str> {
    slot: Arc<Slot<I, P>>,
}

/// The error once it has been handed off, together with its provenance.
///
/// A `Parcel` owns the *original* [`Error`] value the worker produced. The
/// handoff layer does not attach context, does not format the error into a
/// string, and does not copy it, so:
///
/// - [`Parcel::error`] / [`Deref`](core::ops::Deref) reach the real error and
///   its original [`source`](StdError::source) chain;
/// - the concrete cause type is still recoverable with
///   [`Error::downcast`] (see also [`Parcel::downcast`]);
/// - the [`Debug`] output is the error's [`Display`] text only -- backtrace
///   contents are never printed by this type.
pub struct Parcel<I, P = &'static str> {
    task_id: I,
    phase: P,
    error: Error,
}

/// Why an [`Awaiter::take`] did not return a [`Parcel`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TakeError {
    /// The reporter has not handed an error off yet. The awaiter may retry,
    /// block with [`Awaiter::wait_take`], or register interest with
    /// [`Awaiter::wait`].
    Pending,
    /// An error was handed off, but it has already been removed by an earlier
    /// successful [`Awaiter::take`] (possibly through a cloned awaiter). It
    /// cannot be produced a second time.
    AlreadyTaken,
    /// The reporter was dropped without ever handing an error off. No error
    /// will ever arrive on this slot.
    Closed,
}

impl<I, P> Handoff<I, P> {
    /// Create a new empty handoff slot for the given task.
    ///
    /// The `task_id` is supplied explicitly at the call site. The handoff
    /// implementation never inspects thread names or any other ambient
    /// identification.
    pub fn new(task_id: I) -> Self {
        Handoff {
            slot: Arc::new(Slot {
                task_id,
                state: Mutex::new(SlotState::Empty {
                    waiters: Vec::new(),
                }),
                condvar: Condvar::new(),
            }),
        }
    }

    /// The task id this slot was created with.
    pub fn task_id(&self) -> &I {
        &self.slot.task_id
    }

    /// Split the slot into its producer and consumer handles, consuming it.
    ///
    /// After this call the slot is shared between the returned [`Reporter`]
    /// (the one owner of the right to produce) and [`Awaiter`] (any number of
    /// consumers once cloned).
    pub fn split(self) -> (Reporter<I, P>, Awaiter<I, P>) {
        let reporter = Reporter {
            slot: Arc::clone(&self.slot),
        };
        let awaiter = Awaiter { slot: self.slot };
        (reporter, awaiter)
    }

    /// Convenience for keeping only the producer handle. Use
    /// [`Handoff::split`] when both handles are needed.
    pub fn reporter(self) -> Reporter<I, P> {
        Reporter { slot: self.slot }
    }

    /// Convenience for keeping only a consumer handle. Use
    /// [`Handoff::split`] when both handles are needed.
    pub fn awaiter(self) -> Awaiter<I, P> {
        Awaiter { slot: self.slot }
    }
}

impl<I, P> Reporter<I, P> {
    /// Hand the worker's error off, recording the `phase` it had reached.
    ///
    /// This consumes the reporter: a task can hand its terminal failure off
    /// exactly once. The `error` is stored by value as-is -- it is not
    /// formatted into a message and no context is added by the handoff, so
    /// downcasting and the source chain remain fully intact.
    ///
    /// All currently registered waiters are woken.
    pub fn handoff(self, phase: P, error: Error) {
        let mut guard = self.slot.state.lock().unwrap();
        let waiters = match &mut *guard {
            SlotState::Empty { waiters } => core::mem::take(waiters),
            // Every other state is unrepresentable while a reporter exists.
            SlotState::Filled { .. } | SlotState::Taken | SlotState::Closed => unreachable!(),
        };
        *guard = SlotState::Filled { phase, error };
        drop(guard);
        self.slot.condvar.notify_all();
        for waker in waiters {
            waker.wake();
        }
    }
}

impl<I, P> Drop for Reporter<I, P> {
    fn drop(&mut self) {
        let mut guard = self.slot.state.lock().unwrap();
        if let SlotState::Empty { waiters } = &mut *guard {
            let waiters = core::mem::take(waiters);
            *guard = SlotState::Closed;
            drop(guard);
            self.slot.condvar.notify_all();
            for waker in waiters {
                waker.wake();
            }
        }
    }
}

/// Attempt to move the parcel out of the slot under an already-held lock.
fn locked_take<I, P>(state: &mut SlotState<P>, task_id: &I) -> Result<Parcel<I, P>, TakeError>
where
    I: Clone,
{
    match state {
        SlotState::Empty { .. } => Err(TakeError::Pending),
        SlotState::Closed => Err(TakeError::Closed),
        SlotState::Taken => Err(TakeError::AlreadyTaken),
        SlotState::Filled { .. } => {
            let SlotState::Filled { phase, error } =
                core::mem::replace::<SlotState<P>>(state, SlotState::Taken)
            else {
                unreachable!()
            };
            Ok(Parcel {
                task_id: task_id.clone(),
                phase,
                error,
            })
        }
    }
}

impl<I, P> Awaiter<I, P> {
    /// The task id this awaiter's slot was created with.
    pub fn task_id(&self) -> &I {
        &self.slot.task_id
    }

    /// Try to take the handed-off error without blocking.
    ///
    /// The first successful call (across this awaiter and all of its clones)
    /// moves the [`Parcel`] out of the slot; subsequent calls return
    /// [`TakeError::AlreadyTaken`]. Before the handoff the result is
    /// [`TakeError::Pending`]; if the reporter was dropped without handing
    /// anything off it is [`TakeError::Closed`].
    pub fn take(&mut self) -> Result<Parcel<I, P>, TakeError>
    where
        I: Clone,
    {
        let mut guard = self.slot.state.lock().unwrap();
        locked_take(&mut guard, &self.slot.task_id)
    }

    /// Block the current thread until the worker hands an error off (or the
    /// reporter is dropped), then take it.
    ///
    /// Returns [`TakeError::AlreadyTaken`] or [`TakeError::Closed`] the same
    /// way [`Awaiter::take`] does; [`TakeError::Pending`] is only observed
    /// transiently inside the wait loop and is retried internally.
    pub fn wait_take(&mut self) -> Result<Parcel<I, P>, TakeError>
    where
        I: Clone,
    {
        let mut guard = self.slot.state.lock().unwrap();
        loop {
            match locked_take(&mut guard, &self.slot.task_id) {
                Err(TakeError::Pending) => {
                    guard = self.slot.condvar.wait(guard).unwrap();
                }
                result => return result,
            }
        }
    }

    /// Wait asynchronously for the handoff and take it once it arrives.
    ///
    /// Dropping the returned future before it is ready only cancels *this
    /// waiter's* wait: its waker is unregistered. The error itself is owned
    /// by the slot and survives the cancellation of any number of waiters,
    /// as long as a reporter or awaiter handle still exists.
    pub fn wait(&mut self) -> Wait<'_, I, P> {
        Wait {
            awaiter: self,
            waker: None,
        }
    }
}

impl<I, P> Clone for Awaiter<I, P> {
    fn clone(&self) -> Self {
        Awaiter {
            slot: Arc::clone(&self.slot),
        }
    }
}

/// Future returned by [`Awaiter::wait`].
#[must_use = "futures do nothing unless you `.await` or poll them"]
pub struct Wait<'a, I, P> {
    awaiter: &'a mut Awaiter<I, P>,
    waker: Option<Waker>,
}

impl<I, P> Future for Wait<'_, I, P>
where
    I: Clone,
{
    type Output = Result<Parcel<I, P>, TakeError>;

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        let slot = &this.awaiter.slot;
        let mut guard = slot.state.lock().unwrap();
        match locked_take(&mut guard, &slot.task_id) {
            Ok(parcel) => Poll::Ready(Ok(parcel)),
            Err(TakeError::Pending) => {
                if let SlotState::Empty { waiters } = &mut *guard {
                    let new_waker = context.waker();
                    let already_registered = match this.waker.as_ref() {
                        Some(waker) => waker.will_wake(new_waker),
                        None => false,
                    };
                    if !already_registered {
                        if let Some(old_waker) = this.waker.replace(new_waker.clone()) {
                            // Replacing a stale registration: remove the old
                            // waker before registering the new one.
                            waiters.retain(|w| !w.will_wake(&old_waker));
                        }
                        waiters.push(new_waker.clone());
                    }
                }
                Poll::Pending
            }
            // Terminal non-pending states: this waiter is done either way.
            Err(other) => Poll::Ready(Err(other)),
        }
    }
}

impl<I, P> Drop for Wait<'_, I, P> {
    fn drop(&mut self) {
        // Cancellation unregisters only this waiter. The error in the slot
        // is not touched.
        if let Some(waker) = self.waker.take() {
            if let Ok(mut guard) = self.awaiter.slot.state.lock() {
                if let SlotState::Empty { waiters } = &mut *guard {
                    waiters.retain(|w| !w.will_wake(&waker));
                }
            }
        }
    }
}

impl<I, P> Parcel<I, P> {
    /// The explicit task id the handoff slot was created with.
    pub fn task_id(&self) -> &I {
        &self.task_id
    }

    /// The phase the worker recorded when it handed the error off.
    pub fn phase(&self) -> &P {
        &self.phase
    }

    /// The original error, by reference.
    ///
    /// This is the exact [`Error`] the worker passed to
    /// [`Reporter::handoff`], not a reformatted copy.
    pub fn error(&self) -> &Error {
        &self.error
    }

    /// Consume the parcel, returning the task id, phase, and the original
    /// error by value.
    pub fn into_parts(self) -> (I, P, Error) {
        (self.task_id, self.phase, self.error)
    }

    /// Move the original error out, discarding the provenance.
    pub fn into_error(self) -> Error {
        self.error
    }

    /// Attempt to downcast the original error to a concrete type by value.
    ///
    /// This is a shorthand for `self.into_error().downcast::<E>()`; on
    /// failure the original parcel is returned unchanged inside the `Err`.
    pub fn downcast<E>(self) -> Result<E, Parcel<I, P>>
    where
        E: StdError + Send + Sync + 'static,
    {
        let Parcel {
            task_id,
            phase,
            error,
        } = self;
        match error.downcast::<E>() {
            Ok(value) => Ok(value),
            Err(error) => Err(Parcel {
                task_id,
                phase,
                error,
            }),
        }
    }
}

impl<I, P> core::ops::Deref for Parcel<I, P> {
    type Target = Error;

    fn deref(&self) -> &Self::Target {
        &self.error
    }
}

impl<I, P> Display for Parcel<I, P> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.error, formatter)
    }
}

/// The handoff layer renders only the error's own message, never the
/// backtrace captured by anyhow. Use [`Parcel::error`] /
/// [`Parcel::into_parts`] together with [`Error::chain`] for the full
/// rendering.
impl<I, P> Debug for Parcel<I, P> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.error)
    }
}

impl<I, P> StdError for Parcel<I, P>
where
    I: Debug,
    P: Debug,
{
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        self.error.source()
    }
}

impl Display for TakeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TakeError::Pending => formatter.write_str("worker error has not been handed off yet"),
            TakeError::AlreadyTaken => formatter.write_str("worker error has already been taken"),
            TakeError::Closed => {
                formatter.write_str("worker finished without handing off an error")
            }
        }
    }
}

impl StdError for TakeError {}
