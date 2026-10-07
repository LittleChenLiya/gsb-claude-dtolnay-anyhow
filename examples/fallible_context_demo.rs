// Tiny in-memory acceptance demo for the non-default `fallible-context`
// feature. Uses no files or network; run with:
//
//     cargo run --features fallible-context --example fallible_context_demo

#[cfg(feature = "fallible-context")]
use anyhow::{Context, Error};
#[cfg(feature = "fallible-context")]
use std::sync::atomic::{AtomicUsize, Ordering};
#[cfg(feature = "fallible-context")]
use std::sync::Arc;
#[cfg(feature = "fallible-context")]
use thiserror::Error;

#[cfg(feature = "fallible-context")]
#[derive(Debug, Error)]
#[error("business root cause: {0}")]
struct BusinessError(&'static str);

#[cfg(feature = "fallible-context")]
#[derive(Debug, Error)]
#[error("context factory failure: {0}")]
struct FactoryError(&'static str);

#[cfg(feature = "fallible-context")]
fn main() {
    // --- Case A: the context factory itself fails -------------------------
    let calls = Arc::new(AtomicUsize::new(0));
    let calls2 = Arc::clone(&calls);

    let result: Result<(), BusinessError> = Err(BusinessError("wallet empty"));
    let err: Error = result
        .with_fallible_context(|| {
            calls2.fetch_add(1, Ordering::SeqCst);
            Err::<&'static str, FactoryError>(FactoryError("label printer offline"))
        })
        .unwrap_err();

    println!("== factory failed ==");
    println!("Display      : {}", err);
    println!("Root cause   : {}", err.root_cause());
    println!(
        "downcast business: {} (original downcast preserved)",
        err.downcast_ref::<BusinessError>().is_some(),
    );
    println!(
        "downcast factory : {} (factory reason stashed separately)",
        err.downcast_ref::<FactoryError>().is_some(),
    );
    println!(
        "factory invocations on Err: {} (must be 1)",
        calls.load(Ordering::SeqCst),
    );

    // --- Case B: an Ok value never invokes the factory --------------------
    let ok_calls = Arc::new(AtomicUsize::new(0));
    let ok_calls2 = Arc::clone(&ok_calls);
    let ok: Result<i32, BusinessError> = Ok(7);
    let value = ok
        .with_fallible_context(|| {
            ok_calls2.fetch_add(1, Ordering::SeqCst);
            Err::<&'static str, FactoryError>(FactoryError("never"))
        })
        .unwrap();
    println!("\n== ok value ==");
    println!("value: {} (passed through)", value);
    println!(
        "factory invocations on Ok: {} (must be 0)",
        ok_calls.load(Ordering::SeqCst),
    );
}

// When the feature is disabled the original API is unchanged and this demo has
// nothing to show; keep a `main` so the example still compiles.
#[cfg(not(feature = "fallible-context"))]
fn main() {
    eprintln!(
        "enable the non-default `fallible-context` feature to run this demo:\n\
         cargo run --features fallible-context --example fallible_context_demo",
    );
}
