use anyhow::{anyhow, Chain, ChainIndex, Context, Error, LayerKind, MAX_CHAIN_DEPTH};
use std::any::TypeId;
use std::cell::Cell;
use std::error::Error as StdError;
use std::fmt::{self, Display};
use std::io;
use std::ptr;
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Debug)]
struct Ctx(&'static str);

impl Display for Ctx {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(self.0)
    }
}

#[derive(Debug)]
struct RootCause(&'static str);

impl Display for RootCause {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(self.0)
    }
}

impl StdError for RootCause {}

// Three layers: two same-type context layers over a concrete root cause.
fn three_layer() -> Error {
    Error::new(RootCause("disk on fire"))
        .context(Ctx("reading config"))
        .context(Ctx("starting server"))
}

#[test]
fn test_same_type_contexts_each_have_identity() {
    let error = three_layer();
    let index = error.chain_index();

    assert_eq!(index.len(), 3);
    assert!(index.is_complete());

    // Two layers share the type Ctx; both are recorded, in chain order.
    let contexts = index.get_all::<Ctx>();
    assert_eq!(contexts.len(), 2);
    assert_eq!(contexts[0].0, "starting server");
    assert_eq!(contexts[1].0, "reading config");
    // Each context is its own value at its own address.
    assert!(!ptr::eq(contexts[0], contexts[1]));

    // Layer classification and real type identity.
    let layers = index.layers();
    assert_eq!(layers[0].kind(), LayerKind::Context);
    assert_eq!(layers[1].kind(), LayerKind::Context);
    assert_eq!(layers[2].kind(), LayerKind::Cause);
    assert_eq!(layers[0].type_id(), Some(TypeId::of::<Ctx>()));
    assert_eq!(layers[1].type_id(), Some(TypeId::of::<Ctx>()));
    assert_eq!(layers[2].type_id(), Some(TypeId::of::<RootCause>()));
    assert!(layers[0].is::<Ctx>());
    assert!(!layers[0].is::<RootCause>());

    // Context-only and cause-only views.
    assert_eq!(index.contexts().count(), 2);
    assert_eq!(index.causes().count(), 1);
}

#[test]
fn test_downcast_follows_anyhow_semantics() {
    let error = three_layer();
    let index = error.chain_index();

    // The underlying type is still downcastable through the index.
    let root = index.root_cause().unwrap();
    assert!(root.is::<RootCause>());
    assert_eq!(root.downcast_ref::<RootCause>().unwrap().0, "disk on fire");
    assert_eq!(index.get::<RootCause>().unwrap().0, "disk on fire");

    // `get` returns the outermost match, exactly like Error::downcast_ref.
    assert_eq!(index.get::<Ctx>().unwrap().0, "starting server");
    assert_eq!(error.downcast_ref::<Ctx>().unwrap().0, "starting server");

    // The index is read-only: the error was not taken and still works.
    assert_eq!(error.downcast_ref::<RootCause>().unwrap().0, "disk on fire");
    assert_eq!(error.to_string(), "starting server");
    drop(index);
    assert_eq!(error.chain().count(), 3);
}

#[test]
fn test_three_layer_io_error_example() {
    // The canonical three-layer chain: io::Error root cause plus two
    // same-type (&str) context layers attached through Result::context.
    let result: Result<(), io::Error> = Err(io::Error::new(io::ErrorKind::NotFound, "config.json"));
    let error = result
        .context("loading configuration")
        .context("starting service")
        .unwrap_err();

    let index = error.chain_index();
    assert_eq!(index.len(), 3);
    assert!(index.is_complete());
    assert_eq!(
        index.layers().iter().map(|l| l.kind()).collect::<Vec<_>>(),
        [LayerKind::Context, LayerKind::Context, LayerKind::Cause],
    );

    // Both &str contexts keep their own identity and order.
    let contexts = index.get_all::<&str>();
    assert_eq!(contexts, [&"starting service", &"loading configuration"]);

    // The io::Error root cause is still downcastable, through both the
    // index and the original error.
    let root = index.root_cause().unwrap();
    assert!(root.is::<io::Error>());
    assert_eq!(
        root.downcast_ref::<io::Error>().unwrap().kind(),
        io::ErrorKind::NotFound
    );
    assert_eq!(
        index.get::<io::Error>().unwrap().kind(),
        io::ErrorKind::NotFound
    );
    assert_eq!(
        error.downcast_ref::<io::Error>().unwrap().kind(),
        io::ErrorKind::NotFound
    );
}

static DISPLAY_CALLS: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug)]
struct Malicious;

impl Display for Malicious {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        DISPLAY_CALLS.fetch_add(1, Ordering::SeqCst);
        f.write_str("MALICIOUS-MESSAGE")
    }
}

impl StdError for Malicious {}

#[test]
fn test_malicious_display_never_called() {
    let error = Error::new(Malicious).context(Malicious).context(Malicious);
    assert_eq!(DISPLAY_CALLS.load(Ordering::SeqCst), 0);

    let index = error.chain_index();
    let report = index.report();
    let rendered = report.to_string();
    let debugged = format!("{:?}", report);

    // Building the index and the report must not format any error message.
    assert_eq!(DISPLAY_CALLS.load(Ordering::SeqCst), 0);

    // The report holds type identities, not message values.
    assert!(rendered.contains("Malicious")); // type name
    assert!(!rendered.contains("MALICIOUS-MESSAGE"));
    assert!(!debugged.contains("MALICIOUS-MESSAGE"));
    assert_eq!(report.len(), 3);
    assert!(report.is_complete());
}

#[test]
fn test_report_does_not_store_messages() {
    let error = three_layer();
    let report = error.chain_index().report();

    assert_eq!(report.layers().len(), 3);
    assert_eq!(report.layers()[0].depth(), 0);
    assert_eq!(report.layers()[0].kind(), LayerKind::Context);
    assert_eq!(report.layers()[2].kind(), LayerKind::Cause);
    assert_eq!(report.layers()[2].type_id(), Some(TypeId::of::<RootCause>()));
    assert!(report.layers()[2].type_name().unwrap().contains("RootCause"));

    let text = report.to_string();
    assert!(text.contains("context"));
    assert!(text.contains("cause"));
    assert!(!text.contains("disk on fire"));
    assert!(!text.contains("reading config"));
    assert!(!text.contains("starting server"));

    // The report is owned and outlives the error.
    drop(error);
    assert_eq!(report.len(), 3);
}

#[derive(Debug)]
struct SelfLoop;

impl Display for SelfLoop {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("self loop")
    }
}

impl StdError for SelfLoop {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        Some(self)
    }
}

#[test]
fn test_self_cycle_does_not_hang() {
    let error = Error::new(SelfLoop);
    let index = error.chain_index(); // must terminate

    assert_eq!(index.len(), 1);
    assert!(!index.is_complete());
    assert!(index.root_cause().is_none());
    assert!(index.report().to_string().contains("truncated"));
}

#[derive(Debug)]
struct Node {
    next: Cell<Option<&'static (dyn StdError + 'static)>>,
}

// The cycle fixture is built once, leaked, and only read afterwards.
unsafe impl Send for Node {}
unsafe impl Sync for Node {}

impl Display for Node {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("node")
    }
}

impl StdError for Node {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        self.next.get()
    }
}

#[derive(Debug)]
struct EntersCycle(&'static Node);

impl Display for EntersCycle {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("enters cycle")
    }
}

impl StdError for EntersCycle {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        Some(self.0)
    }
}

#[test]
fn test_two_node_cycle_does_not_hang() {
    let a: &'static Node = Box::leak(Box::new(Node {
        next: Cell::new(None),
    }));
    let b: &'static Node = Box::leak(Box::new(Node {
        next: Cell::new(None),
    }));
    a.next.set(Some(b));
    b.next.set(Some(a));

    let error = Error::new(EntersCycle(a));
    let index = error.chain_index(); // must terminate

    // EntersCycle, a, b are recorded; the repeated a is rejected.
    assert_eq!(index.len(), 3);
    assert!(!index.is_complete());
    assert!(index.root_cause().is_none());
}

#[derive(Debug)]
struct Deep {
    next: Option<Box<Deep>>,
}

impl Display for Deep {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("deep")
    }
}

impl StdError for Deep {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        self.next
            .as_deref()
            .map(|next| next as &(dyn StdError + 'static))
    }
}

#[test]
fn test_unbounded_chain_is_finitely_rejected() {
    let mut deep = Deep { next: None };
    for _ in 0..MAX_CHAIN_DEPTH + 100 {
        deep = Deep {
            next: Some(Box::new(deep)),
        };
    }

    let error = Error::new(deep);
    let index = error.chain_index(); // must terminate

    assert_eq!(index.len(), MAX_CHAIN_DEPTH);
    assert!(!index.is_complete());
    assert!(index.root_cause().is_none());
}

#[test]
fn test_empty_chain_is_not_a_success() {
    let result = ChainIndex::from_chain(Chain::default());
    let err = result.unwrap_err();
    assert_eq!(err.to_string(), "cannot index an empty error chain");

    // A non-empty chain indexes fine.
    let error = three_layer();
    let index = ChainIndex::from_chain(error.chain()).unwrap();
    assert_eq!(index.len(), 3);
    assert!(index.is_complete());
    // Layers from a bare chain are opaque but still readable.
    assert_eq!(index.layers()[2].type_id(), None);
    assert_eq!(index.layers()[2].error().to_string(), "disk on fire");
}

#[test]
fn test_adhoc_message_layers() {
    let error = anyhow!("root message").context("outer context");
    let index = error.chain_index();

    assert_eq!(index.len(), 2);
    assert_eq!(index.layers()[0].kind(), LayerKind::Context);
    assert_eq!(index.layers()[1].kind(), LayerKind::Cause);
    // The adhoc message layer has the message's real type identity.
    assert!(index.layers()[1].is::<&str>());
    // Both layers are &str; `get` sees the outermost, exactly like
    // Error::downcast_ref, and `get_all` preserves chain order.
    assert_eq!(index.get::<&str>(), Some(&"outer context"));
    assert_eq!(error.downcast_ref::<&str>(), Some(&"outer context"));
    assert_eq!(
        index.get_all::<&str>(),
        [&"outer context", &"root message"]
    );
    assert_eq!(
        index.root_cause().unwrap().downcast_ref::<&str>(),
        Some(&"root message")
    );
}
