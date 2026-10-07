//! Acceptance tests for the small owned multi-error container.
//!
//! Covers, for two errors of *different* types each carrying context:
//! - the reported source is the explicitly selected first error,
//! - original ordering is preserved for borrowed queries,
//! - each error keeps its own context chain and downcast capability,
//! - `into_errors` transfers ownership exactly once,
//! - an empty collection cannot form a failure result.

use anyhow::{EmptyErrors, Error, MultiError};
use std::error::Error as StdError;
use std::fmt::{self, Display};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

#[derive(Debug)]
struct IoError {
    code: i32,
    drops: Arc<AtomicUsize>,
}

impl Display for IoError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "io failure (code {})", self.code)
    }
}

impl StdError for IoError {}

impl Drop for IoError {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}

#[derive(Debug)]
struct ParseError {
    at: usize,
    drops: Arc<AtomicUsize>,
}

impl Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "parse failure at byte {}", self.at)
    }
}

impl StdError for ParseError {}

impl Drop for ParseError {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}

fn two_failures(
    io_drops: &Arc<AtomicUsize>,
    parse_drops: &Arc<AtomicUsize>,
) -> Vec<Error> {
    let io = IoError {
        code: 42,
        drops: Arc::clone(io_drops),
    };
    let parse = ParseError {
        at: 7,
        drops: Arc::clone(parse_drops),
    };
    vec![
        Error::new(io).context("loading config"),
        Error::new(parse).context("reading header"),
    ]
}

#[test]
fn source_is_explicitly_selected_first_error() {
    let multi = MultiError::new(two_failures(
        &Arc::new(AtomicUsize::new(0)),
        &Arc::new(AtomicUsize::new(0)),
    ))
    .unwrap();

    // `source` points at the first error, whose outermost layer is its context.
    let source = <MultiError as StdError>::source(&multi).unwrap();
    assert_eq!(source.to_string(), "loading config");

    // The first error's own chain is intact: context -> underlying io error.
    let first = multi.first();
    let chain: Vec<String> = first.chain().map(|cause| cause.to_string()).collect();
    assert_eq!(chain, vec!["loading config", "io failure (code 42)"]);
}

#[test]
fn order_is_preserved_for_borrowed_queries() {
    let multi = MultiError::new(two_failures(
        &Arc::new(AtomicUsize::new(0)),
        &Arc::new(AtomicUsize::new(0)),
    ))
    .unwrap();

    assert_eq!(multi.len(), 2);
    assert!(!multi.is_empty());

    let all: Vec<String> = multi.iter().map(ToString::to_string).collect();
    assert_eq!(all, vec!["loading config", "reading header"]);

    // `other_errors` borrows the remainder after the selected first error.
    let rest: Vec<String> = multi.other_errors().map(ToString::to_string).collect();
    assert_eq!(rest, vec!["reading header"]);

    assert_eq!(
        multi.errors()[1].to_string(),
        "reading header",
        "slice access keeps original order",
    );

    // Display summarizes the primary error and the count of remaining errors.
    assert_eq!(
        multi.to_string(),
        "loading config (+1 more error)",
    );
}

#[test]
fn each_error_keeps_context_chain_and_downcast() {
    let multi = MultiError::new(two_failures(
        &Arc::new(AtomicUsize::new(0)),
        &Arc::new(AtomicUsize::new(0)),
    ))
    .unwrap();

    let first = &multi.errors()[0];
    let second = &multi.errors()[1];

    // Context values are downcastable on the corresponding entry only ...
    assert_eq!(
        first.downcast_ref::<&str>().copied(),
        Some("loading config"),
    );
    assert!(first.downcast_ref::<ParseError>().is_none());

    assert_eq!(
        second.downcast_ref::<&str>().copied(),
        Some("reading header"),
    );
    assert!(second.downcast_ref::<IoError>().is_none());

    // ... and the underlying typed causes survive through their own chains.
    let first_io = first
        .chain()
        .find_map(|cause| cause.downcast_ref::<IoError>());
    assert_eq!(first_io.unwrap().code, 42);

    let second_parse = second
        .chain()
        .find_map(|cause| cause.downcast_ref::<ParseError>());
    assert_eq!(second_parse.unwrap().at, 7);

    // The reported source is the first entry's inner std error: it exposes
    // that entry's own context chain (context -> io cause).
    let source = <MultiError as StdError>::source(&multi).unwrap();
    assert_eq!(source.to_string(), "loading config");
    let mut source_chain = Vec::new();
    let mut node: Option<&(dyn StdError + 'static)> = Some(source);
    while let Some(cause) = node {
        source_chain.push(cause.to_string());
        node = cause.source();
    }
    assert_eq!(source_chain, vec!["loading config", "io failure (code 42)"]);
    assert!(source.downcast_ref::<IoError>().is_none());
}

#[test]
fn into_errors_moves_ownership_exactly_once() {
    let io_drops = Arc::new(AtomicUsize::new(0));
    let parse_drops = Arc::new(AtomicUsize::new(0));

    let multi = MultiError::new(two_failures(&io_drops, &parse_drops)).unwrap();

    // Nothing has been dropped while the container owns the errors.
    assert_eq!(io_drops.load(Ordering::SeqCst), 0);
    assert_eq!(parse_drops.load(Ordering::SeqCst), 0);

    // Single ownership transfer: same order, same two entries, no cloning.
    let errors = multi.into_errors();
    assert_eq!(errors.len(), 2);
    assert_eq!(errors[0].to_string(), "loading config");
    assert_eq!(errors[1].to_string(), "reading header");

    // The underlying values are still alive after the move (no early drop).
    assert_eq!(io_drops.load(Ordering::SeqCst), 0);
    assert_eq!(parse_drops.load(Ordering::SeqCst), 0);

    // Now that the extracted vector goes out of scope, each underlying
    // error is dropped exactly once.
    drop(errors);
    assert_eq!(io_drops.load(Ordering::SeqCst), 1);
    assert_eq!(parse_drops.load(Ordering::SeqCst), 1);
}

#[test]
fn converts_into_anyhow_and_reports_first_as_source() {
    let multi = MultiError::new(two_failures(
        &Arc::new(AtomicUsize::new(0)),
        &Arc::new(AtomicUsize::new(0)),
    ))
    .unwrap();

    // Works with `?`-style conversion into a single anyhow::Error and can be
    // recovered by downcasting back to the concrete container.
    let unified: Error = multi.into();
    let recovered = unified
        .downcast_ref::<MultiError>()
        .expect("MultiError downcasts back out of anyhow::Error");
    assert_eq!(recovered.len(), 2);

    let source = unified.source().unwrap();
    assert_eq!(source.to_string(), "loading config");
}

#[test]
fn empty_collection_is_rejected() {
    let result = MultiError::new(Vec::new());
    assert_eq!(result.unwrap_err(), EmptyErrors);
    assert_eq!(
        EmptyErrors.to_string(),
        "cannot build a MultiError from zero errors",
    );

    // Empty iterators (not just vectors) are rejected as well.
    assert!(MultiError::new(core::iter::empty()).is_err());

    // A single error is the minimum valid failure result.
    let only_one = MultiError::new(vec![Error::msg("solo")]).unwrap();
    assert_eq!(only_one.len(), 1);
    assert!(only_one.other_errors().next().is_none());
    assert_eq!(only_one.to_string(), "solo");
}
