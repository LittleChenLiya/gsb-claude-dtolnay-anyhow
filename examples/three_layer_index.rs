//! A three-layer error demonstrated through the explicit chain index.
//!
//! Run with:
//!
//! ```console
//! cargo run --example three_layer_index
//! ```
//!
//! Layers, outermost first:
//!   1. `Stage("retry")` -- a machine-readable context layer
//!   2. `Stage("load")`  -- another context layer of the SAME type
//!   3. `IoCause`        -- the concrete bottom cause
//!
//! The index distinguishes the two same-typed contexts (each with its own
//! identity and attach position), keeps the bottom type downcastable, and never
//! renders message text while indexing.

use anyhow::{anyhow, LayerKind};
use std::error::Error as StdError;
use std::fmt;

#[derive(Debug)]
struct Stage(&'static str);

impl fmt::Display for Stage {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "stage {}", self.0)
    }
}

#[derive(Debug)]
struct IoCause;

impl fmt::Display for IoCause {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("disk on fire")
    }
}

impl StdError for IoCause {}

fn main() {
    let error = anyhow!(IoCause)
        .context(Stage("load"))
        .context(Stage("retry"));

    let index = error.chain_index();

    println!("walk status: {:?}", index.status());
    println!("layer count: {}", index.len());
    for layer in index.layers() {
        let label = match layer.kind() {
            LayerKind::Context => "context",
            LayerKind::Cause => "cause  ",
        };
        println!(
            "  depth={} kind={} type={:?} addr={:?}",
            layer.depth(),
            label,
            layer.type_name(),
            layer.identity_addr(),
        );
    }

    // Two same-typed contexts, both retained in attach order.
    let stages = index.contexts::<Stage>();
    println!("contexts of type Stage: {}", stages.len());
    for stage in &stages {
        println!("  - {}", stage.0);
    }
    assert_eq!(stages.len(), 2);
    assert_eq!(stages[0].0, "retry");
    assert_eq!(stages[1].0, "load");

    // Bottom concrete type remains downcastable by real type identity.
    let bottom = index.cause_ref::<IoCause>().expect("IoCause bottom");
    println!("bottom downcast ok: {}", bottom);

    // Indexing did not take the error: it still renders and downcasts.
    println!("error still renders: {error}");
    assert!(error.downcast_ref::<IoCause>().is_some());
}

