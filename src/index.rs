use crate::chain::ChainState;
use crate::error::ErrorImpl;
use crate::ptr::Ref;
use crate::{Chain, Error, StdError};
use alloc::vec::Vec;
use core::any::TypeId;
use core::fmt::{self, Debug, Display};

/// Maximum number of layers that a [`ChainIndex`] will record.
///
/// Custom [`std::error::Error::source`] implementations can form cycles or
/// unbounded chains. Indexing refuses to follow such chains without bound:
/// once this many layers have been recorded, the walk stops and the index is
/// marked incomplete.
pub const MAX_CHAIN_DEPTH: usize = 1024;

/// Which role a layer plays in an error chain.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LayerKind {
    /// A context layer attached with [`Error::context`] or
    /// [`Context::context`][crate::Context].
    Context,
    /// An underlying cause: the innermost error held by the `anyhow::Error`,
    /// or an error reached through a custom `source()` chain below it.
    Cause,
}

impl Display for LayerKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            LayerKind::Context => f.write_str("context"),
            LayerKind::Cause => f.write_str("cause"),
        }
    }
}

/// An explicit, read-only index over the chain of an [`Error`].
///
/// Unlike [`Error::chain`], which is a streaming iterator over `dyn Error`
/// trait objects, a `ChainIndex` records every layer up front and classifies
/// it as a [`LayerKind::Context`] layer or an underlying [`LayerKind::Cause`].
/// Layers are identified by their real type identity ([`TypeId`]), never by
/// their `Display` message, so two layers with identical messages stay
/// distinct and layers with identical types keep their chain order.
///
/// The index only borrows the error: building it never takes ownership of the
/// `Error`, and downcasting through the index follows the original anyhow
/// semantics, matching what [`Error::downcast_ref`] would find.
///
/// Custom `source()` chains are followed with a finite guard: cycles
/// (including an error that is its own source) and chains longer than
/// [`MAX_CHAIN_DEPTH`] stop the walk instead of hanging, and the resulting
/// index is marked [`is_complete`][ChainIndex::is_complete] `== false`.
///
/// # Example
///
/// A three-layer error chain: two context layers over an `io::Error` root
/// cause.
///
/// ```
/// use anyhow::{anyhow, Context, LayerKind};
/// use std::io;
///
/// let error = anyhow!(io::Error::new(io::ErrorKind::NotFound, "config.json"))
///     .context("loading configuration")
///     .context("starting service");
///
/// let index = error.chain_index();
/// assert_eq!(index.len(), 3);
/// assert_eq!(index.layers()[0].kind(), LayerKind::Context);
/// assert_eq!(index.layers()[1].kind(), LayerKind::Context);
/// assert_eq!(index.layers()[2].kind(), LayerKind::Cause);
///
/// // Both context layers have type `&str`; each keeps its own identity.
/// let contexts = index.get_all::<&str>();
/// assert_eq!(contexts, [&"starting service", &"loading configuration"]);
///
/// // The underlying cause is still downcastable, exactly as
/// // `Error::downcast_ref` would find it.
/// let cause: &io::Error = index.get().unwrap();
/// assert_eq!(cause.kind(), io::ErrorKind::NotFound);
/// ```
pub struct ChainIndex<'a> {
    layers: Vec<Layer<'a>>,
    complete: bool,
}

impl Debug for ChainIndex<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("ChainIndex")
            .field("layers", &self.layers)
            .field("complete", &self.complete)
            .finish()
    }
}

impl<'a> ChainIndex<'a> {
    /// Builds an index over the chain of `error`.
    ///
    /// The index borrows `error`; the error is not moved, consumed, or
    /// formatted. An `anyhow::Error` always contains at least one layer, so
    /// this constructor cannot produce an empty index.
    pub fn new(error: &'a Error) -> Self {
        let mut layers: Vec<Layer<'a>> = Vec::new();
        let pending: Option<&'a (dyn StdError + 'static)>;

        // Walk anyhow's own layers first: context layers and the innermost
        // error, all with real type identity recovered from the vtable.
        unsafe {
            let mut node = error.inner.by_ref();
            loop {
                let info = ErrorImpl::layer(node);
                let layer_error: &'a (dyn StdError + 'static) = ErrorImpl::error(node);
                let inline_source = info.inline_source.map(|inline| {
                    // For ContextError<C, E>, the source of the layer's error
                    // is exactly the E described by `inline`.
                    let error = layer_error.source().unwrap_or(layer_error);
                    (inline, error)
                });
                layers.push(Layer {
                    kind: info.kind,
                    type_id: Some(info.type_id),
                    type_name: Some(info.type_name),
                    payload: Some(info.payload),
                    error: layer_error,
                });
                match (inline_source, info.next) {
                    (Some((inline, error)), _) => {
                        layers.push(Layer {
                            kind: LayerKind::Cause,
                            type_id: Some(inline.type_id),
                            type_name: Some(inline.type_name),
                            payload: Some(inline.payload),
                            error,
                        });
                        pending = error.source();
                        break;
                    }
                    (None, Some(next)) => node = next,
                    (None, None) => {
                        pending = layer_error.source();
                        break;
                    }
                }
            }
        }

        // Then follow custom `source()` chains below anyhow's own layers,
        // with a finite guard against cycles and unbounded chains.
        let complete = Self::walk_sources(&mut layers, pending);
        ChainIndex { layers, complete }
    }

    /// Builds an index from an existing [`Chain`] iterator, for example
    /// [`Error::chain`] or a chain over plain `std::error::Error` trait
    /// objects.
    ///
    /// Layers built this way are opaque: their concrete type identity is not
    /// recoverable on stable Rust, so [`Layer::type_id`] returns `None` and
    /// typed queries do not match them. Prefer [`ChainIndex::new`] when
    /// indexing an `anyhow::Error`.
    ///
    /// An empty chain is **not** a success value: indexing
    /// [`Chain::default`] or any other empty chain fails with
    /// [`EmptyChain`].
    pub fn from_chain(chain: Chain<'a>) -> Result<Self, EmptyChain> {
        match chain.state {
            ChainState::Linked { next } => match next {
                Some(head) => {
                    let mut layers = Vec::new();
                    let complete = Self::walk_sources(&mut layers, Some(head));
                    Ok(ChainIndex { layers, complete })
                }
                None => Err(EmptyChain),
            },
            ChainState::Buffered { rest } => {
                let layers: Vec<Layer<'a>> = rest.map(Layer::opaque).collect();
                if layers.is_empty() {
                    return Err(EmptyChain);
                }
                Ok(ChainIndex {
                    layers,
                    complete: true,
                })
            }
        }
    }

    // Follows a `source()` chain, recording one opaque cause layer per error.
    // Returns false if the walk was cut short by a cycle or the depth limit.
    fn walk_sources(
        layers: &mut Vec<Layer<'a>>,
        mut next: Option<&'a (dyn StdError + 'static)>,
    ) -> bool {
        let mut visited: Vec<*const ()> = layers.iter().map(Layer::error_ptr).collect();
        while let Some(cause) = next {
            let ptr = cause as *const (dyn StdError + 'static) as *const ();
            if visited.contains(&ptr) || layers.len() >= MAX_CHAIN_DEPTH {
                return false;
            }
            visited.push(ptr);
            layers.push(Layer::opaque(cause));
            next = cause.source();
        }
        true
    }

    /// The number of layers recorded in the index. Always at least 1.
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Always `false`: an index never contains zero layers. Empty chains are
    /// rejected by [`ChainIndex::from_chain`] instead.
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// Whether the whole chain was traversed. `false` means the walk stopped
    /// at a cycle or at [`MAX_CHAIN_DEPTH`] and deeper layers may exist.
    pub fn is_complete(&self) -> bool {
        self.complete
    }

    /// All recorded layers, outermost first.
    pub fn layers(&self) -> &[Layer<'a>] {
        &self.layers
    }

    /// The layer at the given depth, where depth 0 is the outermost layer.
    pub fn layer(&self, depth: usize) -> Option<&Layer<'a>> {
        self.layers.get(depth)
    }

    /// Iterates over the context layers only, in chain order.
    pub fn contexts(&self) -> impl Iterator<Item = &Layer<'a>> {
        self.layers
            .iter()
            .filter(|layer| layer.kind == LayerKind::Context)
    }

    /// Iterates over the underlying cause layers only, in chain order.
    pub fn causes(&self) -> impl Iterator<Item = &Layer<'a>> {
        self.layers
            .iter()
            .filter(|layer| layer.kind == LayerKind::Cause)
    }

    /// The lowest layer of the chain: the root cause.
    ///
    /// Returns `None` if the chain was truncated by the cycle or depth guard,
    /// because the true root cause was never reached.
    pub fn root_cause(&self) -> Option<&Layer<'a>> {
        if self.complete {
            self.layers.last()
        } else {
            None
        }
    }

    /// Returns a reference to the first layer value of type `T`, scanning
    /// from the outermost layer downward.
    ///
    /// This follows the original anyhow downcast semantics: for an index
    /// built with [`ChainIndex::new`], the result matches what
    /// [`Error::downcast_ref`] would return for the same error.
    pub fn get<T>(&self) -> Option<&'a T>
    where
        T: 'static,
    {
        self.layers.iter().find_map(Layer::downcast_ref::<T>)
    }

    /// Returns references to every layer value of type `T`, in chain order.
    ///
    /// Multiple layers of the same type (for example several identical
    /// context types) each keep their own identity and their relative order.
    pub fn get_all<T>(&self) -> Vec<&'a T>
    where
        T: 'static,
    {
        self.layers
            .iter()
            .filter_map(Layer::downcast_ref::<T>)
            .collect()
    }

    /// Returns true if any layer has real type identity `T`.
    pub fn contains<T>(&self) -> bool
    where
        T: 'static,
    {
        self.layers.iter().any(Layer::is::<T>)
    }

    /// Produces an owned summary of the index.
    ///
    /// The report records layer kinds, depths, and type identities only. It
    /// never stores, formats, or copies error messages, and it does not
    /// borrow from the indexed error.
    pub fn report(&self) -> ChainReport {
        ChainReport {
            layers: self
                .layers
                .iter()
                .enumerate()
                .map(|(depth, layer)| LayerReport {
                    depth,
                    kind: layer.kind,
                    type_id: layer.type_id,
                    type_name: layer.type_name,
                })
                .collect(),
            complete: self.complete,
        }
    }
}

/// One layer of a [`ChainIndex`]: a single context or cause in the chain.
///
/// A layer is a read-only view; it borrows the indexed error and never takes
/// ownership of it.
pub struct Layer<'a> {
    kind: LayerKind,
    type_id: Option<TypeId>,
    type_name: Option<&'static str>,
    payload: Option<Ref<'a, ()>>,
    error: &'a (dyn StdError + 'static),
}

impl Debug for Layer<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Layer")
            .field("kind", &self.kind)
            .field("type_id", &self.type_id)
            .field("type_name", &self.type_name)
            .finish()
    }
}

impl<'a> Layer<'a> {
    fn opaque(error: &'a (dyn StdError + 'static)) -> Self {
        Layer {
            kind: LayerKind::Cause,
            type_id: None,
            type_name: None,
            payload: None,
            error,
        }
    }

    fn error_ptr(&self) -> *const () {
        self.error as *const (dyn StdError + 'static) as *const ()
    }

    /// Whether this layer is a context layer or an underlying cause.
    pub fn kind(&self) -> LayerKind {
        self.kind
    }

    /// The real type identity of the value this layer contributes to
    /// downcasting: the context type for context layers, the error type for
    /// cause layers.
    ///
    /// `None` for opaque layers reached through custom `source()` chains or
    /// through [`ChainIndex::from_chain`], whose concrete type is not
    /// recoverable on stable Rust.
    pub fn type_id(&self) -> Option<TypeId> {
        self.type_id
    }

    /// The name of the type identified by [`Layer::type_id`], if known.
    pub fn type_name(&self) -> Option<&'static str> {
        self.type_name
    }

    /// Returns true if `T` is the real type identity of this layer.
    pub fn is<T>(&self) -> bool
    where
        T: 'static,
    {
        self.type_id == Some(TypeId::of::<T>())
    }

    /// Attempts to downcast this layer's value to a concrete type, by real
    /// type identity rather than message text.
    pub fn downcast_ref<T>(&self) -> Option<&'a T>
    where
        T: 'static,
    {
        if self.is::<T>() {
            // Safety: the payload pointer was recorded together with
            // `type_id` from the same vtable, so it points at a live T
            // borrowed from the indexed error for 'a.
            Some(unsafe { self.payload?.cast::<T>().deref() })
        } else {
            None
        }
    }

    /// The layer's error as a `std::error::Error` trait object, for
    /// read-only inspection such as formatting or further `source()` walks.
    pub fn error(&self) -> &'a (dyn StdError + 'static) {
        self.error
    }
}

/// An owned summary of a [`ChainIndex`], containing no error messages.
///
/// Produced by [`ChainIndex::report`]. Stores only layer depths, kinds, and
/// type identities; message values are never recorded.
#[derive(Clone, Debug)]
pub struct ChainReport {
    layers: Vec<LayerReport>,
    complete: bool,
}

impl ChainReport {
    /// One entry per recorded layer, outermost first.
    pub fn layers(&self) -> &[LayerReport] {
        &self.layers
    }

    /// The number of layers in the report.
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Always `false`; see [`ChainIndex::is_empty`].
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// Whether the indexed chain was traversed to its root cause.
    pub fn is_complete(&self) -> bool {
        self.complete
    }
}

impl Display for ChainReport {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for layer in &self.layers {
            writeln!(
                f,
                "#{} {} {}",
                layer.depth,
                layer.kind,
                layer.type_name_or_unknown()
            )?;
        }
        if !self.complete {
            writeln!(f, "... chain truncated (cycle or depth limit)")?;
        }
        Ok(())
    }
}

/// One layer of a [`ChainReport`].
#[derive(Clone, Debug)]
pub struct LayerReport {
    depth: usize,
    kind: LayerKind,
    type_id: Option<TypeId>,
    type_name: Option<&'static str>,
}

impl LayerReport {
    /// Position in the chain; 0 is the outermost layer.
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Whether this layer is a context layer or an underlying cause.
    pub fn kind(&self) -> LayerKind {
        self.kind
    }

    /// The real type identity of the layer, if known.
    pub fn type_id(&self) -> Option<TypeId> {
        self.type_id
    }

    /// The name of the layer's type, if known.
    pub fn type_name(&self) -> Option<&'static str> {
        self.type_name
    }

    fn type_name_or_unknown(&self) -> &str {
        self.type_name.unwrap_or("<unknown>")
    }
}

/// Error returned when indexing an empty error chain.
///
/// An empty chain is not a success value: [`ChainIndex::from_chain`] rejects
/// it instead of producing an empty index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EmptyChain;

impl Display for EmptyChain {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("cannot index an empty error chain")
    }
}

impl StdError for EmptyChain {}
