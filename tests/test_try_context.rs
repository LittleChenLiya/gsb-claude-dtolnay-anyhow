#![allow(clippy::nonstandard_macro_braces)]

//! Acceptance tests for the fallible context factory adapter
//! (`try_context` / `try_with_context`).
//!
//! Fixtures use a tiny temporary directory inside `target/` (D: drive
//! worktree) so nothing escapes the controlled workspace.

use anyhow::{Context, ContextFactoryError, Result};
use std::cell::Cell;
use std::fmt::{self, Display};
use std::fs;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Two distinct custom error types: business root cause vs factory failure.
// ---------------------------------------------------------------------------

/// Business root cause (the error we must always be able to downcast to).
#[derive(Debug)]
struct BizError {
    op: &'static str,
}

impl Display for BizError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "business operation failed: {}", self.op)
    }
}

impl std::error::Error for BizError {}

/// Factory failure reason (a different custom type, stored alongside).
#[derive(Debug)]
struct FactoryError {
    reason: String,
}

impl Display for FactoryError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "context factory failed: {}", self.reason)
    }
}

impl std::error::Error for FactoryError {}

// ---------------------------------------------------------------------------
// Tiny temp-dir fixture kept inside the worktree (target/try-context-tests).
// ---------------------------------------------------------------------------

struct TempDir(PathBuf);

impl TempDir {
    fn new(test: &str) -> Self {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("try-context-tests")
            .join(format!("{}-{}", test, std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        TempDir(dir)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

// ---------------------------------------------------------------------------
// Fallible business operation and fallible context factory.
// ---------------------------------------------------------------------------

/// Business operation: fails with `BizError` when the data file is absent.
fn load_business_data(dir: &Path) -> Result<String, BizError> {
    fs::read_to_string(dir.join("data.txt")).map_err(|_| BizError {
        op: "load business data",
    })
}

/// Context factory: fails with `FactoryError` when the metadata file is
/// absent. Counts every invocation through `calls`.
fn make_context(dir: &Path, calls: &Cell<usize>) -> Result<String, FactoryError> {
    calls.set(calls.get() + 1);
    fs::read_to_string(dir.join("meta.txt"))
        .map(|meta| format!("context: {}", meta.trim()))
        .map_err(|_| FactoryError {
            reason: "metadata file unavailable".to_owned(),
        })
}
// ---------------------------------------------------------------------------
// 1. Factory failure: both causes queryable, original downcast, one call.
// ---------------------------------------------------------------------------

#[test]
fn factory_failure_preserves_both_causes() {
    let tmp = TempDir::new("factory-failure");
    // No data.txt -> business error. No meta.txt -> factory error.
    let calls = Cell::new(0);

    let err = load_business_data(tmp.path())
        .try_with_context(|| make_context(tmp.path(), &calls))
        .unwrap_err();

    // Factory ran exactly once.
    assert_eq!(calls.get(), 1, "factory must be called exactly once");

    // (a) Original business error is still the downcastable root cause.
    let biz = err
        .downcast_ref::<BizError>()
        .expect("business root cause must remain downcastable");
    assert_eq!(biz.op, "load business data");

    // (b) Factory failure reason is stored alongside, queryable separately.
    let note = err
        .downcast_ref::<ContextFactoryError<FactoryError>>()
        .expect("factory failure must be preserved in a context layer");
    assert_eq!(note.factory_error().reason, "metadata file unavailable");

    // (c) Both causes appear in the chain; neither replaces the other.
    let chain: Vec<String> = err.chain().map(ToString::to_string).collect();
    assert_eq!(chain.len(), 2, "chain: [factory note, business cause]");
    assert_eq!(
        chain[0],
        "failed to produce error context: context factory failed: metadata file unavailable"
    );
    assert_eq!(chain[1], "business operation failed: load business data");
}

// ---------------------------------------------------------------------------
// 2. Factory success: context attached normally, root cause downcastable.
// ---------------------------------------------------------------------------

#[test]
fn factory_success_attaches_context() {
    let tmp = TempDir::new("factory-success");
    fs::write(tmp.path().join("meta.txt"), "request id 42\n").unwrap();
    // No data.txt -> business error; meta.txt present -> factory Ok.
    let calls = Cell::new(0);

    let err = load_business_data(tmp.path())
        .try_with_context(|| make_context(tmp.path(), &calls))
        .unwrap_err();

    assert_eq!(calls.get(), 1, "factory must be called exactly once");
    assert_eq!(err.to_string(), "context: request id 42");
    assert!(err.downcast_ref::<BizError>().is_some());
    assert!(err
        .downcast_ref::<ContextFactoryError<FactoryError>>()
        .is_none());
}

// ---------------------------------------------------------------------------
// 3. Ok path: factory is never called (zero calls), value passes through.
// ---------------------------------------------------------------------------

#[test]
fn ok_path_never_calls_factory() {
    let tmp = TempDir::new("ok-path");
    fs::write(tmp.path().join("data.txt"), "payload").unwrap();
    let calls = Cell::new(0);

    let value = load_business_data(tmp.path())
        .try_with_context(|| make_context(tmp.path(), &calls))
        .unwrap();

    assert_eq!(value, "payload");
    assert_eq!(calls.get(), 0, "factory must not run on Ok");
}
// ---------------------------------------------------------------------------
// 4. Eager variant `try_context` mirrors the lazy one.
// ---------------------------------------------------------------------------

#[test]
fn eager_try_context_matches_lazy_behavior() {
    // Factory output Ok -> context attached.
    let err = Result::<(), BizError>::Err(BizError { op: "eager" })
        .try_context(Ok::<_, FactoryError>("eager context"))
        .unwrap_err();
    assert_eq!(err.to_string(), "eager context");
    assert!(err.downcast_ref::<BizError>().is_some());

    // Factory output Err -> dual cause preserved.
    let err = Result::<(), BizError>::Err(BizError { op: "eager" })
        .try_context(Err::<String, _>(FactoryError {
            reason: "eager failure".to_owned(),
        }))
        .unwrap_err();
    assert!(err.downcast_ref::<BizError>().is_some());
    let note = err
        .downcast_ref::<ContextFactoryError<FactoryError>>()
        .unwrap();
    assert_eq!(note.factory_error().reason, "eager failure");

    // Ok input -> factory output ignored.
    let ok = Result::<_, BizError>::Ok(7u32)
        .try_context(Err::<String, _>(FactoryError {
            reason: "unused".to_owned(),
        }))
        .unwrap();
    assert_eq!(ok, 7);
}

// ---------------------------------------------------------------------------
// 5. Option<T> support: None with a failing factory still records the reason.
// ---------------------------------------------------------------------------

#[test]
fn option_none_with_failing_factory() {
    let calls = Cell::new(0);
    let err = None::<u8>
        .try_with_context(|| -> Result<&'static str, FactoryError> {
            calls.set(calls.get() + 1);
            Err(FactoryError {
                reason: "no metadata for missing option".to_owned(),
            })
        })
        .unwrap_err();

    assert_eq!(calls.get(), 1);
    let note = err
        .downcast_ref::<ContextFactoryError<FactoryError>>()
        .unwrap();
    assert_eq!(
        note.factory_error().reason,
        "no metadata for missing option"
    );

    // Some(_) -> factory never called.
    let calls = Cell::new(0);
    let value = Some(9u8)
        .try_with_context(|| -> Result<&'static str, FactoryError> {
            calls.set(calls.get() + 1);
            Ok("unused")
        })
        .unwrap();
    assert_eq!(value, 9);
    assert_eq!(calls.get(), 0);
}

// ---------------------------------------------------------------------------
// 6. New capability not used -> original interface behaves exactly as before.
// ---------------------------------------------------------------------------

#[test]
fn original_interface_unaffected() {
    let err = Result::<(), BizError>::Err(BizError { op: "legacy" })
        .context("legacy context")
        .unwrap_err();
    assert_eq!(err.to_string(), "legacy context");
    assert!(err.downcast_ref::<BizError>().is_some());

    let calls = Cell::new(0);
    let err = Result::<(), BizError>::Err(BizError { op: "legacy" })
        .with_context(|| {
            calls.set(calls.get() + 1);
            "lazy legacy context"
        })
        .unwrap_err();
    assert_eq!(err.to_string(), "lazy legacy context");
    assert_eq!(calls.get(), 1);

    let calls = Cell::new(0);
    let ok = Result::<_, BizError>::Ok(1u8)
        .with_context(|| {
            calls.set(calls.get() + 1);
            "never"
        })
        .unwrap();
    assert_eq!(ok, 1);
    assert_eq!(calls.get(), 0);
}
