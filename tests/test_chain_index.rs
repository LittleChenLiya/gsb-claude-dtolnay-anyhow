//! Acceptance tests for the explicit, read-only chain index.
//!
//! Run with:
//!
//! ```console
//! cargo test --test test_chain_index
//! ```
//!
//! The three-layer example (two context layers over a bottom cause) is in
//! `examples/three_layer_index.rs` and can be run with:
//!
//! ```console
//! cargo run --example three_layer_index
//! ```

use anyhow::{anyhow, ChainIndex, EmptyChain, Error, LayerKind, WalkStatus};
use std::error::Error as StdError;
use std::fmt::{self, Display};
use std::io;
use std::sync::atomic::{AtomicUsize, Ordering};

// A machine-readable context layer type (does not implement StdError).
#[derive(Debug)]
struct Stage(&'static str);

impl Display for Stage {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "stage {}", self.0)
    }
}

// The concrete bottom cause.
#[derive(Debug)]
struct IoCause;

impl Display for IoCause {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("disk on fire")
    }
}

impl StdError for IoCause {}

fn three_layer_error() -> Error {
    anyhow!(IoCause)
        .context(Stage("load"))
        .context(Stage("retry"))
}

#[test]
fn two_same_typed_contexts_each_have_identity_and_order() {
    let error = three_layer_error();
    let index = error.chain_index();

    assert!(index.is_complete());
    assert_eq!(index.len(), 3);

    // Two same-typed contexts are BOTH retained, in attach (outer-first) order.
    let stages = index.contexts::<Stage>();
    assert_eq!(stages.len(), 2);
    assert_eq!(stages[0].0, "retry");
    assert_eq!(stages[1].0, "load");

    // Each has its own structural identity even though the type is identical.
    let context_layers: Vec<_> = index.context_layers().collect();
    assert_eq!(context_layers.len(), 2);
    let addr0 = context_layers[0].identity_addr().unwrap();
    let addr1 = context_layers[1].identity_addr().unwrap();
    assert_ne!(addr0, addr1, "two same-type contexts keep distinct identities");
    assert_eq!(context_layers[0].depth(), 0);
    assert_eq!(context_layers[1].depth(), 1);
}

#[test]
fn context_layers_are_distinguished_from_bottom_cause() {
    let error = three_layer_error();
    let index = error.chain_index();

    let kinds: Vec<LayerKind> = index.layers().map(|layer| layer.kind()).collect();
    assert_eq!(
        kinds,
        vec![LayerKind::Context, LayerKind::Context, LayerKind::Cause],
    );

    // Exactly one underlying cause layer.
    assert_eq!(index.cause_layers().count(), 1);
    assert_eq!(index.context_layers().count(), 2);
}

#[test]
fn bottom_type_is_still_downcastable() {
    let error = three_layer_error();
    let index = error.chain_index();

    // Downcast follows Anyhow's original semantics by real type identity.
    assert!(index.cause_ref::<IoCause>().is_some());
    assert!(index.all::<IoCause>().len() == 1);

    // The same value is reachable through the original error API too.
    assert!(error.downcast_ref::<IoCause>().is_some());
}

#[derive(Debug)]
struct MiddleError {
    source: io::Error,
}

impl Display for MiddleError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("middle")
    }
}

impl StdError for MiddleError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        Some(&self.source)
    }
}

#[test]
fn held_bottom_and_its_native_source_are_both_indexed() {
    let error = anyhow!(MiddleError {
        source: io::Error::new(io::ErrorKind::NotFound, "missing"),
    })
    .context(Stage("outer"));

    let index = error.chain_index();
    assert!(index.is_complete());
    // Context + held MiddleError + io::Error reached via native source().
    assert_eq!(index.len(), 3);

    // The held concrete type is downcastable by real type identity...
    assert!(index.cause_ref::<MiddleError>().is_some());
    // ...and so is a cause reached only through a custom source implementation.
    let io = index.cause_ref::<io::Error>().expect("io cause");
    assert_eq!(io.kind(), io::ErrorKind::NotFound);

    // Root cause is the deepest link (the io::Error).
    assert_eq!(index.root_cause().unwrap().to_string(), "missing");
}

#[test]
fn adhoc_message_bottom_is_downcastable() {    let error = anyhow!("{}", "plain message").context(Stage("wrap"));
    let index = error.chain_index();

    // An interpolated anyhow! message is a String bottom value. It is not a
    // std::error::Error, but is still downcastable by type identity the same
    // way the original Error::downcast supports adhoc message values.
    assert_eq!(index.all::<String>(), vec![&String::from("plain message")]);
    assert_eq!(index.contexts::<Stage>().len(), 1);
    // Original Anyhow downcast agrees.
    assert!(error.downcast_ref::<String>().is_some());
}

// An error whose Display is observable, so tests can prove it is never rendered.
static DISPLAY_CALLS: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug)]
struct LoudContext;

impl Display for LoudContext {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        DISPLAY_CALLS.fetch_add(1, Ordering::SeqCst);
        f.write_str("loud")
    }
}

#[test]
fn indexing_and_queries_do_not_render_display() {
    let error = anyhow!(IoCause).context(LoudContext);
    DISPLAY_CALLS.store(0, Ordering::SeqCst);

    let index = error.chain_index();
    let _stages = index.contexts::<LoudContext>();
    let _ = index.cause_ref::<IoCause>();
    let report = index.report();

    // Build the message-free report and query by type: Display count stays 0.
    assert_eq!(report.context_count_of::<LoudContext>(), 1);
    assert_eq!(report.count_of::<IoCause>(), 1);
    assert_eq!(DISPLAY_CALLS.load(Ordering::SeqCst), 0);
}

#[test]
fn report_does_not_store_message_values() {
    fn assert_static<T: 'static>(_: &T) {}

    let error = three_layer_error();
    let index = error.chain_index();
    let report = index.report();

    // Compile-time proof: the report borrows nothing and lives for 'static, so
    // it cannot be holding any message value borrowed from the error.
    assert_static(&report);

    assert_eq!(report.len(), 3);
    for layer in report.layers() {
        // Only a compile-time type name; no message string is stored.
        assert!(layer.type_name.is_some());
    }
    let stage_name = report
        .layers()
        .find(|l| l.kind == LayerKind::Context)
        .and_then(|l| l.type_name)
        .unwrap();
    assert!(stage_name.contains("Stage"), "got {stage_name}");

    // An owned clone holds only structural metadata.
    let moved = report.clone();
    assert_eq!(moved.context_count_of::<Stage>(), 2);
    assert_eq!(moved.count_of::<IoCause>(), 1);
}

#[test]
fn index_is_read_only_error_not_taken() {
    let error = three_layer_error();

    let index = error.chain_index();
    // Consume several read-only views.
    let _a = index.contexts::<Stage>();
    let _b = index.cause_ref::<IoCause>();
    drop(index);

    // The error is still fully usable after indexing.
    assert_eq!(error.to_string(), "stage retry");
    assert!(error.downcast_ref::<IoCause>().is_some());
}

// A node whose source points at itself: a malicious/hand-rolled self-loop.
struct LoopNode {
    this: core::cell::UnsafeCell<*const LoopNode>,
}

unsafe impl Send for LoopNode {}
unsafe impl Sync for LoopNode {}

impl fmt::Debug for LoopNode {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("LoopNode")
    }
}

impl LoopNode {
    fn self_referential() -> Box<LoopNode> {
        let b = Box::new(LoopNode {
            this: core::cell::UnsafeCell::new(core::ptr::null()),
        });
        let raw = Box::into_raw(b);
        // Point the node's source at its own stable heap address, then reclaim
        // ownership. The Box is never moved/resized afterward, so the pointer
        // stays valid for the life of the returned box.
        unsafe { *(*raw).this.get() = raw };
        unsafe { Box::from_raw(raw) }
    }
}

impl Display for LoopNode {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("loop")
    }
}

impl StdError for LoopNode {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        let ptr = unsafe { *self.this.get() };
        if ptr.is_null() {
            None
        } else {
            // Self-loop: source is this very same object.
            Some(unsafe { &*ptr })
        }
    }
}

#[test]
fn self_loop_does_not_hang() {
    let node = LoopNode::self_referential();
    // Index the raw cause chain directly: the self-loop must be rejected
    // finitely with Cycle, not hang.
    let index = ChainIndex::from_sources(Some(node.as_ref() as &dyn StdError)).unwrap();

    assert_eq!(index.status(), WalkStatus::Cycle);
    // The node itself is indexed once; the loop edge is not followed again.
    assert_eq!(index.len(), 1);
}

#[test]
fn self_loop_below_context_does_not_hang() {
    // Construct a bottom StdError that self-loops, wrap it as Anyhow's held
    // value. The index records the held cause then detects the cycle.
    let node = LoopNode::self_referential();
    let boxed: Box<dyn StdError + Send + Sync> = node;
    let error = Error::from_boxed(boxed).context(Stage("around"));

    let index = error.chain_index();
    assert_eq!(index.status(), WalkStatus::Cycle);
    // Context plus the single visited loop node.
    assert_eq!(index.len(), 2);
    assert_eq!(index.contexts::<Stage>().len(), 1);
}

#[test]
fn empty_chain_is_not_a_success_value() {
    // No head => there is nothing to index; this fails explicitly instead of
    // yielding a silently-empty index.
    let result = ChainIndex::from_sources(None);
    assert_eq!(result.err(), Some(EmptyChain));
}

#[test]
fn index_over_a_real_error_is_never_empty() {
    let error = anyhow!("just one layer");
    let index = error.chain_index();
    assert!(!index.is_empty());
    assert_eq!(index.len(), 1);
    assert_eq!(index.status(), WalkStatus::Complete);
}

// A long, hand-rolled source chain to prove the finite depth bound.
struct Link {
    next: core::cell::UnsafeCell<Option<*const Link>>,
}

impl Link {
    fn new() -> Self {
        Link {
            next: core::cell::UnsafeCell::new(None),
        }
    }
}

impl Display for Link {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("link")
    }
}

impl fmt::Debug for Link {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("Link")
    }
}

impl StdError for Link {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        let next = unsafe { *self.next.get() };
        next.map(|ptr| unsafe { &*ptr } as &(dyn StdError + 'static))
    }
}

#[test]
fn overly_long_chain_is_truncated_finitely() {
    // Build a chain longer than the max depth.
    const N: usize = ChainIndex::MAX_CHAIN_DEPTH + 20;
    let links: Vec<Link> = (0..N).map(|_| Link::new()).collect();
    for i in 0..N - 1 {
        unsafe { *links[i].next.get() = Some(&links[i + 1]) };
    }

    let index = ChainIndex::from_sources(Some(&links[0] as &dyn StdError)).unwrap();
    assert_eq!(index.status(), WalkStatus::Truncated);
    assert_eq!(index.len(), ChainIndex::MAX_CHAIN_DEPTH);
}

