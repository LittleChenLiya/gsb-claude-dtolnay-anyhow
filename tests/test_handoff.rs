use anyhow::handoff::{self, Stage, TakeError, TaskId, WaitOutcome};
use anyhow::anyhow;
use std::error::Error as StdError;
use std::fmt;
use std::sync::{Arc, Barrier};
use std::thread;

#[derive(Debug)]
struct WorkerError {
    code: i32,
}

impl fmt::Display for WorkerError {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "worker error (code {})", self.code)
    }
}

impl StdError for WorkerError {}

#[test]
fn worker_handoff_preserves_downcast() {
    let (producer, consumer) = handoff::new(TaskId(42));
    let start = Arc::new(Barrier::new(2));
    let start2 = Arc::clone(&start);
    let worker = thread::spawn(move || {
        start2.wait();
        producer.fail(Stage::Running, anyhow!(WorkerError { code: 7 }));
    });
    start.wait();
    assert_eq!(consumer.wait(), WaitOutcome::Ready);
    let handed = consumer.take().unwrap();
    assert_eq!(handed.task_id(), TaskId(42));
    assert_eq!(handed.stage(), Stage::Running);
    let error = handed.into_error();
    let worker_error = error.downcast_ref::<WorkerError>().unwrap();
    assert_eq!(worker_error.code, 7);
    worker.join().unwrap();
}

#[test]
fn take_succeeds_once_then_rejects() {
    let (producer, consumer) = handoff::new(TaskId(1));
    producer.fail(Stage::Created, anyhow!(WorkerError { code: 1 }));
    let first = consumer.take().unwrap();
    assert_eq!(first.task_id(), TaskId(1));
    assert!(first.error().downcast_ref::<WorkerError>().is_some());
    assert!(matches!(consumer.take(), Err(TakeError::AlreadyTaken)));
}

#[test]
fn cancel_wait_keeps_error() {
    let (producer, consumer) = handoff::new(TaskId(9));
    let registered = Arc::new(Barrier::new(2));
    let registered2 = Arc::clone(&registered);
    let waiter_consumer = consumer.clone();
    let waiter = thread::spawn(move || {
        registered2.wait();
        waiter_consumer.wait()
    });
    registered.wait();
    consumer.cancel();
    assert_eq!(waiter.join().unwrap(), WaitOutcome::Cancelled);

    // The worker still owns the producer; its error must not be destroyed.
    let release = Arc::new(Barrier::new(2));
    let release2 = Arc::clone(&release);
    let worker = thread::spawn(move || {
        release2.wait();
        producer.fail(Stage::Running, anyhow!(WorkerError { code: 3 }));
    });
    release.wait();
    worker.join().unwrap();

    let handed = consumer.take().unwrap();
    assert_eq!(handed.task_id(), TaskId(9));
    assert!(handed.error().downcast_ref::<WorkerError>().is_some());
}

#[test]
fn cancel_without_waiter_then_wait_is_cancelled() {
    let (producer, consumer) = handoff::new(TaskId(4));
    consumer.cancel();
    assert_eq!(consumer.wait(), WaitOutcome::Cancelled);
    // A late deposit is still kept because the producer still has an owner.
    producer.fail(Stage::Running, anyhow!(WorkerError { code: 5 }));
    let handed = consumer.take().unwrap();
    assert!(handed.error().downcast_ref::<WorkerError>().is_some());
}

#[test]
fn producer_drop_without_fail_finishes() {
    let (producer, consumer) = handoff::new(TaskId(2));
    drop(producer);
    assert_eq!(consumer.wait(), WaitOutcome::Finished);
    assert!(matches!(consumer.take(), Err(TakeError::Cancelled)));
}

#[test]
fn source_chain_and_message_preserved() {
    let inner = std::io::Error::new(std::io::ErrorKind::Other, "inner boom");
    let (producer, consumer) = handoff::new(TaskId(3));
    producer.fail(Stage::Running, anyhow::Error::new(inner));
    let handed = consumer.take().unwrap();
    let error = handed.error();
    assert_eq!(error.to_string(), "inner boom");
    let mut chain = error.chain();
    assert_eq!(chain.next().unwrap().to_string(), "inner boom");
    assert!(chain.next().is_none());
}

#[test]
fn debug_and_display_do_not_print_backtrace() {
    let (producer, consumer) = handoff::new(TaskId(5));
    producer.fail(Stage::Running, anyhow!(WorkerError { code: 1 }));
    let handed = consumer.take().unwrap();
    let debug = format!("{:?}", handed);
    assert!(debug.contains("worker error (code 1)"));
    assert!(!debug.to_ascii_lowercase().contains("backtrace"));
    let display = format!("{}", handed);
    assert!(display.contains("worker error (code 1)"));
    assert!(!display.to_ascii_lowercase().contains("backtrace"));
}

#[test]
fn task_id_is_explicit_not_thread_name() {
    let (producer, consumer) = handoff::new(TaskId(123));
    let worker = thread::Builder::new()
        .name("a-name-that-is-not-the-task-id".to_owned())
        .spawn(move || {
            producer.fail(Stage::Running, anyhow!(WorkerError { code: 0 }));
        })
        .unwrap();
    worker.join().unwrap();
    let handed = consumer.take().unwrap();
    assert_eq!(handed.task_id(), TaskId(123));
}

#[test]
fn fail_consumes_producer_once() {
    let (producer, consumer) = handoff::new(TaskId(11));
    let delivered = producer.fail(Stage::ShuttingDown, anyhow!(WorkerError { code: 8 }));
    // The Delivered producer can no longer fail; dropping it is harmless.
    drop(delivered);
    let handed = consumer.take().unwrap();
    assert_eq!(handed.stage(), Stage::ShuttingDown);
    assert!(handed.error().downcast_ref::<WorkerError>().is_some());
}

#[test]
fn cancel_after_deposit_keeps_error() {
    let (producer, consumer) = handoff::new(TaskId(13));
    producer.fail(Stage::Running, anyhow!(WorkerError { code: 4 }));
    // The error is deposited and still has an owner; cancel must not destroy
    // it.
    consumer.cancel();
    assert_eq!(consumer.wait(), WaitOutcome::Ready);
    let handed = consumer.take().unwrap();
    assert_eq!(handed.task_id(), TaskId(13));
    assert!(handed.error().downcast_ref::<WorkerError>().is_some());
}
