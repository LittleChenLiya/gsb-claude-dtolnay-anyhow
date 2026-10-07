use anyhow::{anyhow, Error, MultiError};
use std::error::Error as _;
use std::io;
use thiserror::Error as ThisError;

#[derive(ThisError, Debug)]
#[error("disk full")]
struct DiskFull;

fn first_error() -> Error {
    Error::new(DiskFull).context("writing index")
}

fn second_error() -> Error {
    Error::new(io::Error::new(io::ErrorKind::Other, "flush failed")).context("flushing buffer")
}

#[test]
fn test_two_typed_errors_with_context() {
    // Two errors of different underlying types, each carrying context.
    let mut multi = MultiError::new(first_error());
    multi.push(second_error());
    assert_eq!(multi.len(), 2);
    assert!(!multi.is_empty());
    assert_eq!(multi.to_string(), "writing index (and 1 more error)");

    // source() points at the explicitly selected first error, and that
    // error's context chain stays reachable through it.
    let source = multi.source().unwrap();
    assert_eq!(source.to_string(), "writing index");
    let root = source.source().unwrap();
    assert_eq!(root.to_string(), "disk full");
    assert!(root.downcast_ref::<DiskFull>().is_some());
    assert!(root.source().is_none());

    // The rest can be borrowed in their original order.
    let borrowed: Vec<&Error> = multi.iter().collect();
    assert_eq!(borrowed.len(), 2);
    assert_eq!(borrowed[0].to_string(), "writing index");
    assert_eq!(borrowed[1].to_string(), "flushing buffer");
    assert_eq!(multi.first().to_string(), "writing index");

    // Each item keeps its own context chain and downcast ability.
    let chain0: Vec<String> = borrowed[0].chain().map(|e| e.to_string()).collect();
    assert_eq!(chain0, ["writing index", "disk full"]);
    assert!(borrowed[0].is::<DiskFull>());
    assert!(!borrowed[0].is::<io::Error>());
    assert_eq!(
        borrowed[0].downcast_ref::<DiskFull>().unwrap().to_string(),
        "disk full",
    );
    let chain1: Vec<String> = borrowed[1].chain().map(|e| e.to_string()).collect();
    assert_eq!(chain1, ["flushing buffer", "flush failed"]);
    assert!(borrowed[1].is::<io::Error>());
    assert_eq!(
        borrowed[1].downcast_ref::<io::Error>().unwrap().kind(),
        io::ErrorKind::Other,
    );

    // into_errors consumes the container and moves ownership out exactly
    // once; the owned errors can now be downcast by value.
    let owned: Vec<Error> = multi.into_errors();
    assert_eq!(owned.len(), 2);
    let mut owned = owned.into_iter();
    let first: DiskFull = owned.next().unwrap().downcast().unwrap();
    assert_eq!(first.to_string(), "disk full");
    let second: io::Error = owned.next().unwrap().downcast().unwrap();
    assert_eq!(second.to_string(), "flush failed");
    assert!(owned.next().is_none());
}

#[test]
fn test_empty_is_not_a_failure() {
    // An empty set of errors cannot constitute a failure result.
    assert!(MultiError::from_errors(Vec::new()).is_none());

    let one = MultiError::from_errors(vec![anyhow!("boom")]).unwrap();
    assert_eq!(one.len(), 1);
    assert!(!one.is_empty());
    assert_eq!(one.to_string(), "boom");
    assert_eq!(one.source().unwrap().to_string(), "boom");
    assert_eq!(one.into_errors().len(), 1);
}

#[test]
fn test_single_error_regression() {
    // Short regression of the original single-error interface: the same
    // error behaves identically on its own and inside the container.
    let solo = first_error();
    assert_eq!(solo.to_string(), "writing index");
    let solo_chain: Vec<String> = solo.chain().map(|e| e.to_string()).collect();
    assert_eq!(solo_chain, ["writing index", "disk full"]);
    assert!(solo.is::<DiskFull>());
    let recovered: DiskFull = solo.downcast().unwrap();
    assert_eq!(recovered.to_string(), "disk full");

    let multi = MultiError::new(first_error());
    assert_eq!(multi.to_string(), "writing index");
    assert_eq!(multi.source().unwrap().to_string(), "writing index");
    let multi_chain: Vec<String> = multi.first().chain().map(|e| e.to_string()).collect();
    assert_eq!(multi_chain, solo_chain);
    let recovered: DiskFull = multi.into_errors().pop().unwrap().downcast().unwrap();
    assert_eq!(recovered.to_string(), "disk full");
}

#[test]
fn test_tempdir_fixture() {
    // Small temp-dir fixture: two failing reads become a two-element
    // MultiError, then the fixture is cleaned up.
    let dir = std::env::temp_dir().join(format!("anyhow-multierror-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    let mut errors = Vec::new();
    for name in ["a.txt", "b.txt"] {
        if let Err(err) = std::fs::read_to_string(dir.join(name)) {
            errors.push(Error::new(err).context(format!("reading {name}")));
        }
    }

    let multi = MultiError::from_errors(errors).unwrap();
    assert_eq!(multi.len(), 2);
    assert_eq!(multi.to_string(), "reading a.txt (and 1 more error)");
    assert_eq!(multi.source().unwrap().to_string(), "reading a.txt");
    let order: Vec<String> = multi.iter().map(|e| e.to_string()).collect();
    assert_eq!(order, ["reading a.txt", "reading b.txt"]);
    for error in multi.into_errors() {
        let io_error = error.downcast_ref::<io::Error>().unwrap();
        assert_eq!(io_error.kind(), io::ErrorKind::NotFound);
    }

    std::fs::remove_dir_all(&dir).unwrap();
}
