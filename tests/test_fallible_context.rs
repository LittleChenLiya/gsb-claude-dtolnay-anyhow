// Acceptance tests for the non-default `fallible-context` feature:
// `Context::with_fallible_context`.
//
// Covers: business error and factory error are two distinct custom types, both
// causes are inspectable, the original downcast survives, the factory runs
// exactly once on Err and zero times on Ok, plus a small temp-dir fixture.
#![cfg(feature = "fallible-context")]

use anyhow::{Context, Error};
use std::cell::Cell;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use thiserror::Error;

// The business root cause.
#[derive(Debug, Error)]
#[error("business exploded: {0}")]
struct BusinessError(String);

// A *different* custom type: the reason why building context failed.
#[derive(Debug, Error, PartialEq, Eq)]
#[error("context factory blew up: {0}")]
struct FactoryError(&'static str);

// Human-readable context used when the factory succeeds.
#[derive(Debug, Error)]
#[error("attached context: {0}")]
struct GoodContext(&'static str);

fn business_result() -> Result<(), BusinessError> {
    Err(BusinessError("disk on fire".to_owned()))
}

// 1. Factory fails: business error stays root/downcastable, factory error is
//    stashed separately, factory invoked exactly once, business chain intact.
#[test]
fn test_factory_failure_preserves_business_root_cause() {
    let calls = Arc::new(AtomicUsize::new(0));
    let calls2 = Arc::clone(&calls);

    let err: Error = business_result()
        .with_fallible_context(|| {
            calls2.fetch_add(1, Ordering::SeqCst);
            Err::<GoodContext, FactoryError>(FactoryError("no labels left"))
        })
        .unwrap_err();

    // Factory called exactly once.
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "factory must run once on Err"
    );

    // Original business downcast still works.
    let business = err
        .downcast_ref::<BusinessError>()
        .expect("business error must remain downcastable");
    assert_eq!(business.0, "disk on fire");

    // The separately stored factory failure is also downcastable.
    let factory = err
        .downcast_ref::<FactoryError>()
        .expect("factory error must be stashed and downcastable");
    assert_eq!(*factory, FactoryError("no labels left"));

    // The two causes are genuinely different custom types.
    assert_ne!(
        std::any::Any::type_id(business),
        std::any::Any::type_id(factory),
    );

    // Display/root cause remain the business error, not the factory error.
    assert_eq!(err.to_string(), "business exploded: disk on fire");
    let root = err.root_cause();
    assert_eq!(root.to_string(), "business exploded: disk on fire");

    // The business error has no cause of its own, so it is both the sole chain
    // node and the root; the chain never contains the factory error.
    let chain: Vec<String> = err.chain().map(|c| c.to_string()).collect();
    assert_eq!(chain, vec!["business exploded: disk on fire".to_owned()]);
    let in_chain = err
        .chain()
        .any(|cause| cause.downcast_ref::<FactoryError>().is_some());
    assert!(!in_chain, "factory error must not hijack the source chain");

    // The alternate debug repr ({:#?}) delegates to the underlying struct and
    // surfaces the stashed factory failure for diagnostics; the normal debug
    // repr intentionally shows only the business source chain.
    let alt_debug = format!("{:#?}", err);
    assert!(
        alt_debug.contains("context_factory_error")
            && alt_debug.contains("FactoryError")
            && alt_debug.contains("no labels left"),
        "alternate debug repr should expose stashed factory error, got: {alt_debug}"
    );
}

// 1b. When the business error has its own underlying cause, that deeper cause
//     is preserved as the root via source()-forwarding, the business message
//     is not duplicated in the chain, and both business layers plus the stashed
//     factory error remain independently downcastable.
#[test]
fn test_factory_failure_forwards_business_source_chain() {
    #[derive(Debug, Error)]
    #[error("deep io cause")]
    struct DeepError;

    #[derive(Debug, Error)]
    #[error("middleware failure")]
    struct MiddleError(#[from] DeepError);

    let err: Error = Err::<(), MiddleError>(MiddleError(DeepError))
        .with_fallible_context(|| Err::<GoodContext, FactoryError>(FactoryError("ctx boom")))
        .unwrap_err();

    // The original business error downcasts through the anyhow node...
    assert!(err.downcast_ref::<MiddleError>().is_some());
    // ...its deeper std cause is preserved through the source chain...
    assert_eq!(
        err.chain().nth(1).map(|c| c.to_string()),
        Some("deep io cause".to_owned()),
    );
    // ...and the factory failure is still stashed separately.
    assert!(err.downcast_ref::<FactoryError>().is_some());

    // source() forwards to the business error's own cause: root is DeepError.
    assert_eq!(err.root_cause().to_string(), "deep io cause");

    // The business message must appear exactly once (no duplicate "Caused by").
    let chain: Vec<String> = err.chain().map(|c| c.to_string()).collect();
    assert_eq!(
        chain,
        vec!["middleware failure".to_owned(), "deep io cause".to_owned(),],
    );
    assert!(!chain.iter().any(|m| m.contains("ctx boom")));
}

// 2. Factory succeeds: ordinary context attaches, business error remains a
//    reachable cause, factory still runs exactly once.
#[test]
fn test_factory_success_attaches_context() {
    let calls = Arc::new(AtomicUsize::new(0));
    let calls2 = Arc::clone(&calls);

    let err: Error = business_result()
        .with_fallible_context(|| {
            calls2.fetch_add(1, Ordering::SeqCst);
            Ok::<_, FactoryError>(GoodContext("while flushing"))
        })
        .unwrap_err();

    assert_eq!(calls.load(Ordering::SeqCst), 1);

    let ctx = err
        .downcast_ref::<GoodContext>()
        .expect("successful context must be attached and downcastable");
    assert_eq!(ctx.0, "while flushing");

    let business = err
        .downcast_ref::<BusinessError>()
        .expect("business error stays downcastable through attached context");
    assert_eq!(business.0, "disk on fire");

    assert_eq!(err.to_string(), "attached context: while flushing");
    assert_eq!(
        err.root_cause().to_string(),
        "business exploded: disk on fire",
    );
}

// 3. Ok branch: the factory is never invoked (zero calls) and the value passes
//    through unchanged.
#[test]
fn test_ok_does_not_call_factory() {
    thread_local! { static CALLS: Cell<usize> = const { Cell::new(0) }; }

    let result: Result<i32, Error> = Ok::<i32, BusinessError>(42).with_fallible_context(|| {
        CALLS.with(|c| c.set(c.get() + 1));
        Err::<GoodContext, FactoryError>(FactoryError("must not happen"))
    });

    assert_eq!(result.unwrap(), 42);
    CALLS.with(|c| assert_eq!(c.get(), 0, "factory must never run on Ok"));
}

// 4. Real-world-ish fixture using a small temporary directory: reading a
//    missing file fails (io::Error is the business root cause); a factory that
//    tries to stat a second missing path to build richer context also fails,
//    and that factory failure is stashed without clobbering the io::Error.
#[test]
fn test_temp_dir_fixture_io_error_survives_failed_factory() {
    let dir = std::env::temp_dir().join(format!(
        "anyhow-fallible-context-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
    ));
    fs::create_dir_all(&dir).unwrap();

    let result = (|| -> anyhow::Result<()> {
        let missing_data: PathBuf = dir.join("data.bin");
        let missing_meta: PathBuf = dir.join("meta.txt");

        fs::read(&missing_data).with_fallible_context(|| {
            // Building the context itself requires another I/O operation that
            // fails here; report it as a distinct custom error type.
            let meta_len = fs::metadata(&missing_meta).map_err(|io| {
                FactoryError(Box::leak(
                    format!("meta lookup failed: {io}").into_boxed_str(),
                ))
            })?;
            let label: &'static str =
                Box::leak(format!("read data ({} bytes)", meta_len.len()).into_boxed_str());
            Ok::<GoodContext, FactoryError>(GoodContext(label))
        })?;
        Ok(())
    })();

    let err = result.unwrap_err();

    // Original business io::Error preserved and downcastable.
    let io_err = err
        .downcast_ref::<std::io::Error>()
        .expect("business io::Error must remain downcastable");
    assert_eq!(io_err.kind(), std::io::ErrorKind::NotFound);

    // Factory failure stored separately.
    let factory_err = err
        .downcast_ref::<FactoryError>()
        .expect("factory io failure must be stashed");
    assert!(factory_err.0.starts_with("meta lookup failed"));

    // The business io::Error stays the head/root of the chain (its Display is
    // the OS "entity not found" message); the factory's meta.txt failure never
    // enters the chain.
    let chain_nodes = err.chain().count();
    assert!(chain_nodes >= 1);
    let whole_chain = err
        .chain()
        .map(|cause| cause.to_string())
        .collect::<Vec<_>>()
        .join(" | ");
    assert!(!whole_chain.contains("meta.txt"));
    assert_eq!(err.root_cause().to_string(), io_err.to_string());

    // Cleanup: deterministic teardown of the small fixture directory.
    fs::remove_dir_all(&dir).unwrap();
}

// 5. Also works when the business error is already an anyhow::Error (the
//    `Result<_, anyhow::Error>` dispatch path): chain and downcast survive.
#[test]
fn test_anyhow_error_business_via_inherent_fail_context() {
    let business: Error = Error::new(BusinessError("nested".to_owned()));

    let err = business
        .fail_context(|| Err::<GoodContext, FactoryError>(FactoryError("label service down")));

    assert!(err.downcast_ref::<BusinessError>().is_some());
    assert!(err.downcast_ref::<FactoryError>().is_some());
    assert_eq!(err.to_string(), "business exploded: nested");
    assert_eq!(err.root_cause().to_string(), "business exploded: nested");
}

// 6. Downcast by value exercises the custom drop-rest vtable paths: the
//    extracted value is moved out and everything else is dropped exactly once.
#[test]
fn test_downcast_by_value_and_drops() {
    use std::sync::atomic::AtomicBool;

    struct Counted {
        dropped: Arc<AtomicBool>,
        which: &'static str,
    }
    impl Drop for Counted {
        fn drop(&mut self) {
            self.dropped.store(true, Ordering::SeqCst);
        }
    }
    impl std::fmt::Debug for Counted {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_tuple(self.which).finish()
        }
    }
    impl std::fmt::Display for Counted {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str(self.which)
        }
    }
    impl std::error::Error for Counted {}

    #[derive(Debug)]
    struct Biz;
    impl std::fmt::Display for Biz {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("biz")
        }
    }
    impl std::error::Error for Biz {}

    // 6a. Move the business error out by value; stashed factory is dropped with
    //     the rest of the structure.
    let business_dropped = Arc::new(AtomicBool::new(false));
    let factory_dropped = Arc::new(AtomicBool::new(false));
    {
        let err: Error = Err::<(), Counted>(Counted {
            dropped: Arc::clone(&business_dropped),
            which: "business",
        })
        .with_fallible_context(|| {
            Err::<GoodContext, _>(Counted {
                dropped: Arc::clone(&factory_dropped),
                which: "factory",
            })
        })
        .unwrap_err();

        let business = err
            .downcast::<Counted>()
            .ok()
            .expect("downcast business by value");
        assert_eq!(business.which, "business");
        assert!(!business_dropped.load(Ordering::SeqCst));
        assert!(
            factory_dropped.load(Ordering::SeqCst),
            "stashed factory error dropped with the rest"
        );
        drop(business);
        assert!(business_dropped.load(Ordering::SeqCst));
    }

    // 6b. Move the stashed factory error out by value; business (distinct
    //     type) is dropped together with the rest of the structure.
    let factory_dropped2 = Arc::new(AtomicBool::new(false));
    {
        let err: Error = Err::<(), Biz>(Biz)
            .with_fallible_context(|| {
                Err::<GoodContext, _>(Counted {
                    dropped: Arc::clone(&factory_dropped2),
                    which: "factory2",
                })
            })
            .unwrap_err();

        let factory = err
            .downcast::<Counted>()
            .ok()
            .expect("downcast stashed factory by value");
        assert_eq!(factory.which, "factory2");
        assert!(!factory_dropped2.load(Ordering::SeqCst));
        drop(factory);
        assert!(factory_dropped2.load(Ordering::SeqCst));
    }
}
