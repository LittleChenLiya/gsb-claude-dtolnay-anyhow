#![allow(clippy::assertions_on_result_states)]

//! Acceptance tests for `anyhow::handoff`: explicit error handoff container.
//!
//! Coverage:
//! - the original error handed off by a worker is moved by value and its
//!   concrete type stays downcast-able, with the source chain intact;
//! - the first `take` succeeds and every later take is explicitly refused
//!   with `AlreadyTaken`;
//! - cancelling an async wait unregisters only the waiter and never destroys
//!   an error that still has an owner;
//! - the task id is the value passed at construction (never a thread name);
//! - a `std::sync::Barrier`-synchronized cross-thread scenario.

use anyhow::handoff::{Handoff, TakeError};
use anyhow::Error;
use std::error::Error as StdError;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::task::{Context, Poll, Waker};
use std::thread;
use thiserror::Error as ThisError;

#[derive(Debug, ThisError)]
#[error("worker failed in phase {phase}")]
struct WorkerFailure {
    phase: u32,
    #[source]
    cause: std::io::Error,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Prepare,
    Commit,
}

fn worker_error() -> Error {
    Error::new(WorkerFailure {
        phase: 7,
        cause: std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied"),
    })
}

fn parcel_for() -> (
    anyhow::handoff::Reporter<u64, Phase>,
    impl FnOnce() -> anyhow::handoff::Parcel<u64, Phase>,
) {
    let (reporter, mut awaiter) = Handoff::<u64, Phase>::new(1).split();
    (reporter, move || {
        awaiter.wait_take().expect("handoff must arrive")
    })
}

// The concrete underlying type survives the handoff and can be downcast by
// value; provenance and the source chain are preserved.
#[test]
fn handed_off_error_is_downcastable_with_source_chain() {
    let (reporter, get_parcel) = parcel_for();
    reporter.handoff(Phase::Commit, worker_error());
    let parcel = get_parcel();

    assert_eq!(parcel.task_id(), &1_u64);
    assert_eq!(parcel.phase(), &Phase::Commit);

    // Parcel behaves as the real error in the std::error chain.
    let source = <dyn StdError>::source(&parcel).expect("source must be preserved");
    let io = source
        .downcast_ref::<std::io::Error>()
        .expect("source must still be the original io::Error");
    assert_eq!(io.kind(), std::io::ErrorKind::PermissionDenied);

    // The value moved is the real error, not a stringified clone: downcasting
    // to the concrete worker type succeeds and carries its original fields.
    let failure = parcel
        .downcast::<WorkerFailure>()
        .ok()
        .expect("underlying worker type must be downcast-able after handoff");
    assert_eq!(failure.phase, 7);
    assert_eq!(failure.cause.kind(), std::io::ErrorKind::PermissionDenied);
}

// The first take succeeds; every later take, even through a cloned awaiter,
// is refused with an explicit `AlreadyTaken`.
#[test]
fn first_take_succeeds_second_take_refused() {
    let (reporter, mut awaiter) = Handoff::<u64, Phase>::new(9).split();
    let mut clone = awaiter.clone();

    assert_eq!(awaiter.take().unwrap_err(), TakeError::Pending);

    reporter.handoff(Phase::Prepare, worker_error());

    let parcel = awaiter.take().expect("first take must succeed");
    assert_eq!(parcel.phase(), &Phase::Prepare);

    // Same handle, cloned handle: both are now explicitly refused.
    assert_eq!(awaiter.take().unwrap_err(), TakeError::AlreadyTaken);
    assert_eq!(clone.take().unwrap_err(), TakeError::AlreadyTaken);

    // The parcel already holds the only copy; no second error is fabricated.
    drop(parcel);
}

// Cancelling a pending async wait unregisters only that waiter; the error
// that arrives afterwards is still owned by the slot and can be taken.
#[test]
fn cancel_wait_does_not_destroy_owned_error() {
    let (reporter, mut awaiter) = Handoff::<u64, Phase>::new(3).split();

    let woken = Arc::new(AtomicBool::new(false));
    let waker = counting_waker(Arc::clone(&woken));
    let mut context = Context::from_waker(&waker);

    // Poll once while empty: the waiter is registered.
    let mut wait = awaiter.wait();
    assert!(Pin::new(&mut wait).poll(&mut context).is_pending());
    assert!(!woken.load(Ordering::SeqCst));

    // Cancel the wait before any handoff: it only unregisters this waiter.
    drop(wait);

    // The reporter still owns the right to produce; the error it now hands
    // off must not have been destroyed by the cancellation.
    reporter.handoff(Phase::Commit, worker_error());

    let parcel = awaiter
        .take()
        .expect("error must survive cancellation of a waiter");
    assert_eq!(parcel.phase(), &Phase::Commit);
    let _ = parcel
        .downcast::<WorkerFailure>()
        .expect("cancelled waiter must leave the original error intact");

    // The cancelled waiter was not woken by the handoff it abandoned.
    assert!(!woken.load(Ordering::SeqCst));
}

// When several waiters are registered and only one of them cancels, the
// error is neither destroyed nor stolen: another waiter still receives it.
#[test]
fn cancelling_one_waiter_leaves_error_for_others() {
    let (reporter, mut awaiter) = Handoff::<u64, Phase>::new(6).split();
    let mut clone = awaiter.clone();

    let waker_one = counting_waker(Arc::new(AtomicBool::new(false)));
    let woken_two = Arc::new(AtomicBool::new(false));
    let waker_two = counting_waker(Arc::clone(&woken_two));
    let mut cx_one = Context::from_waker(&waker_one);
    let mut cx_two = Context::from_waker(&waker_two);

    let mut wait_one = awaiter.wait();
    let mut wait_two = clone.wait();
    assert!(Pin::new(&mut wait_one).poll(&mut cx_one).is_pending());
    assert!(Pin::new(&mut wait_two).poll(&mut cx_two).is_pending());

    // Cancel waiter one only.
    drop(wait_one);

    reporter.handoff(Phase::Prepare, worker_error());
    assert!(woken_two.load(Ordering::SeqCst));

    let parcel = Pin::new(&mut wait_two)
        .poll(&mut cx_two)
        .ready()
        .expect("surviving waiter must still receive the error");
    assert_eq!(parcel.task_id(), &6_u64);
}

// The handed-off error is dropped only after the parcel is dropped and no
// handle can produce it again; cancellation of waiters does not drop it.
#[test]
fn error_not_dropped_while_owned_after_cancelled_wait() {
    use std::sync::atomic::AtomicBool as DropFlag;

    let flag = Arc::new(DropFlag::new(false));

    #[derive(Debug)]
    struct Tracked(#[allow(dead_code)] Arc<DropFlag>);
    impl StdError for Tracked {}
    impl std::fmt::Display for Tracked {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("tracked")
        }
    }
    impl Drop for Tracked {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    let (reporter, mut awaiter) = Handoff::<u64, Phase>::new(7).split();
    let waker = counting_waker(Arc::new(AtomicBool::new(false)));
    let mut context = Context::from_waker(&waker);

    let mut wait = awaiter.wait();
    assert!(Pin::new(&mut wait).poll(&mut context).is_pending());
    drop(wait); // cancel while the error does not even exist yet

    assert!(!flag.load(Ordering::SeqCst));
    reporter.handoff(Phase::Prepare, Error::new(Tracked(Arc::clone(&flag))));
    assert!(!flag.load(Ordering::SeqCst), "handoff must own the error");

    let parcel = awaiter.take().unwrap();
    assert!(!flag.load(Ordering::SeqCst), "take moves, does not drop");
    drop(parcel);
    assert!(flag.load(Ordering::SeqCst), "error drops with its parcel");
}

// An async waiter that does not cancel is woken by the handoff and receives
// the parcel from its next poll.
#[test]
fn async_wait_is_woken_and_receives_parcel() {
    let (reporter, mut awaiter) = Handoff::<u64, Phase>::new(4).split();

    let woken = Arc::new(AtomicBool::new(false));
    let waker = counting_waker(Arc::clone(&woken));
    let mut context = Context::from_waker(&waker);

    let mut wait = awaiter.wait();
    assert!(Pin::new(&mut wait).poll(&mut context).is_pending());

    reporter.handoff(Phase::Prepare, worker_error());
    assert!(woken.load(Ordering::SeqCst), "handoff must wake the waiter");

    let parcel = Pin::new(&mut wait)
        .poll(&mut context)
        .ready()
        .expect("second poll must yield the parcel");
    assert_eq!(parcel.task_id(), &4_u64);
    drop(parcel);
    drop(wait);
    assert_eq!(
        Pin::new(&mut awaiter.wait()).poll(&mut context).ready_err(),
        TakeError::AlreadyTaken,
    );
}

// If the reporter goes away without a handoff, waiters are woken and observe
// an explicit `Closed`, rather than hanging or receiving a fabricated error.
#[test]
fn reporter_drop_closes_slot_without_fabricating_error() {
    let (reporter, mut awaiter) = Handoff::<u64, Phase>::new(5).split();
    drop(reporter);
    assert_eq!(awaiter.take().unwrap_err(), TakeError::Closed);
    assert_eq!(awaiter.take().unwrap_err(), TakeError::Closed);
}

// The task id is whatever was passed at construction, even when the worker
// thread carries a deliberately different name; the implementation never
// guesses it from the thread name.
#[test]
fn task_id_is_explicit_not_guessed_from_thread_name() {
    let handoff = Handoff::new(77_u32);
    assert_eq!(*handoff.task_id(), 77);
    let (reporter, mut awaiter) = handoff.split();
    assert_eq!(*awaiter.task_id(), 77);

    let barrier = Arc::new(Barrier::new(2));
    let worker_barrier = Arc::clone(&barrier);
    let worker = thread::Builder::new()
        .name("totally-different-name".to_owned())
        .spawn(move || {
            worker_barrier.wait(); // synchronized with the assertion below
            reporter.handoff(Phase::Prepare, worker_error());
        })
        .unwrap();

    barrier.wait();
    let parcel = awaiter.wait_take().unwrap();
    assert_eq!(parcel.task_id(), &77_u32);
    worker.join().unwrap();
}

// Barrier-synchronized end-to-end handoff from a real worker thread.
#[test]
fn barrier_worker_handoff_roundtrip() {
    let (reporter, mut awaiter) = Handoff::new(0xAB_CD_u64).split();
    let barrier = Arc::new(Barrier::new(2));
    let worker_barrier = Arc::clone(&barrier);

    let worker = thread::spawn(move || {
        // Both threads reach this point before the failure is produced, so
        // the ordering relative to the consumer is deterministic with no
        // sleeps.
        worker_barrier.wait();
        reporter.handoff(Phase::Commit, worker_error());
    });

    barrier.wait();
    let parcel = awaiter.wait_take().expect("worker must hand off");
    assert_eq!(parcel.task_id(), &0xAB_CD_u64);
    assert_eq!(parcel.phase(), &Phase::Commit);

    let failure = parcel
        .downcast_ref::<WorkerFailure>()
        .expect("by-ref downcast through Parcel's Deref to anyhow::Error");
    assert_eq!(failure.phase, 7);

    worker.join().unwrap();
    assert_eq!(awaiter.take().unwrap_err(), TakeError::AlreadyTaken);
}

// The handoff layer renders no backtrace: Debug is the error's Display text.
#[test]
fn debug_never_renders_backtrace() {
    let (reporter, get_parcel) = parcel_for();
    reporter.handoff(Phase::Prepare, worker_error());
    let parcel = get_parcel();

    let rendered = format!("{parcel:?}");
    assert_eq!(rendered, "worker failed in phase 7");
    assert!(!rendered.to_ascii_lowercase().contains("backtrace"));
    assert_eq!(rendered, parcel.error().to_string());
}

// `into_parts` hands back ownership of everything.
#[test]
fn into_parts_returns_original_error_by_value() {
    let (reporter, mut awaiter) = Handoff::<u64, Phase>::new(11).split();
    reporter.handoff(Phase::Prepare, worker_error());
    let parcel = awaiter.take().unwrap();

    let (task_id, phase, error) = parcel.into_parts();
    assert_eq!(task_id, 11);
    assert_eq!(phase, Phase::Prepare);
    error
        .downcast::<WorkerFailure>()
        .expect("original error recovered by value from into_parts");
}

// --- minimal test waker machinery (no executor needed) ---

struct CountingWaker {
    wakes: AtomicUsize,
    woken: Arc<AtomicBool>,
}

impl counting_waker_mod::Wake for CountingWaker {
    fn wake(self: Arc<Self>) {
        self.woken.store(true, Ordering::SeqCst);
        self.wakes.fetch_add(1, Ordering::SeqCst);
    }
}

mod counting_waker_mod {
    use super::*;
    use std::task::RawWaker;
    use std::task::RawWakerVTable;

    pub trait Wake: Send + Sync + 'static {
        fn wake(self: Arc<Self>);
    }

    pub fn waker<W: Wake>(waker: Arc<W>) -> Waker {
        let ptr = Arc::into_raw(waker) as *const ();
        // SAFETY: vtable functions honor the raw waker contract.
        unsafe { Waker::from_raw(RawWaker::new(ptr, vtable::<W>())) }
    }

    fn vtable<W: Wake>() -> &'static RawWakerVTable {
        &RawWakerVTable::new(clone::<W>, wake::<W>, wake_by_ref::<W>, drop_waker::<W>)
    }

    unsafe fn clone<W: Wake>(ptr: *const ()) -> RawWaker {
        let arc = unsafe { Arc::from_raw(ptr as *const W) };
        let cloned = Arc::clone(&arc);
        std::mem::forget(arc);
        RawWaker::new(Arc::into_raw(cloned) as *const (), vtable::<W>())
    }

    unsafe fn wake<W: Wake>(ptr: *const ()) {
        let arc = unsafe { Arc::from_raw(ptr as *const W) };
        arc.wake();
    }

    unsafe fn wake_by_ref<W: Wake>(ptr: *const ()) {
        let arc = unsafe { Arc::from_raw(ptr as *const W) };
        let w = Arc::clone(&arc);
        std::mem::forget(arc);
        w.wake();
    }

    unsafe fn drop_waker<W: Wake>(ptr: *const ()) {
        drop(unsafe { Arc::from_raw(ptr as *const W) });
    }
}

fn counting_waker(woken: Arc<AtomicBool>) -> Waker {
    counting_waker_mod::waker(Arc::new(CountingWaker {
        wakes: AtomicUsize::new(0),
        woken,
    }))
}

trait PollExt<T> {
    fn ready(self) -> T;
    fn ready_err<E>(self) -> E
    where
        T: ResultLike<E>;
}

trait ResultLike<E> {
    fn into_err(self) -> E;
}

impl<T, E> ResultLike<E> for Result<T, E> {
    fn into_err(self) -> E {
        self.err().expect("expected Err")
    }
}

impl<T> PollExt<T> for Poll<T> {
    fn ready(self) -> T {
        match self {
            Poll::Ready(value) => value,
            Poll::Pending => panic!("expected Poll::Ready"),
        }
    }
    fn ready_err<E>(self) -> E
    where
        T: ResultLike<E>,
    {
        self.ready().into_err()
    }
}
