//! Explicit, read-only index over an [`Error`]'s cause chain.
//!
//! Unlike iterating [`Error::chain`] and deciding what something "is" from its
//! human-readable [`Display`][core::fmt::Display] text, this index identifies
//! every layer by its *concrete Rust type* (a [`TypeId`]). It keeps the
//! distinction between context layers that were attached with
//! [`.context(...)`][crate::Context] and the underlying cause chain, while
//! preserving the order in which each layer was attached (outermost first).
//!
//! The index is built by borrowing the error ([`Error::chain_index`]); it never
//! takes ownership of or mutates the error, so the error remains usable after
//! indexing. The index stores type identities and references only; it never
//! renders, copies, or stores [`Display`][core::fmt::Display] message values.
//!
//! Custom `source` implementations are not assumed to be acyclic: walking the
//! cause chain is bounded by [`ChainIndex::MAX_CHAIN_DEPTH`] and by detecting a
//! revisited trait object (matching data address and vtable), so a
//! self-referential source terminates instead of hanging. The resulting
//! [`WalkStatus`] records whether the chain was
//! [`Complete`](WalkStatus::Complete), [`Truncated`](WalkStatus::Truncated), or
//! contained a [`Cycle`](WalkStatus::Cycle).
//!
//! # Example
//!
//! A three-layer error: two typed context layers over a concrete bottom cause.
//!
//! ```
//! use anyhow::{anyhow, Context};
//! use std::fmt;
//!
//! #[derive(Debug)]
//! struct Stage(&'static str);
//! impl fmt::Display for Stage {
//!     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
//!         write!(f, "stage {}", self.0)
//!     }
//! }
//!
//! #[derive(Debug)]
//! struct IoCause;
//! impl fmt::Display for IoCause {
//!     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
//!         f.write_str("disk on fire")
//!     }
//! }
//! impl std::error::Error for IoCause {}
//!
//! let error = anyhow!(IoCause)
//!     .context(Stage("load"))
//!     .context(Stage("retry"));
//!
//! let index = error.chain_index();
//!
//! // Two distinct, same-typed context layers, kept in attach order.
//! let stages: Vec<&Stage> = index.contexts();
//! assert_eq!(stages.len(), 2);
//! assert_eq!(stages[0].0, "retry"); // outermost first
//! assert_eq!(stages[1].0, "load");
//!
//! // The bottom concrete type is still downcastable by real type identity.
//! assert!(index.cause_ref::<IoCause>().is_some());
//! assert_eq!(index.layers().filter(|l| l.is_cause()).count(), 1);
//! ```

use crate::StdError;
use alloc::vec::Vec;
use core::any::{type_name, type_name_of_val, Any, TypeId};
use core::ptr::addr_of;

/// Marker type used solely to type the generic cause walker when a layer's
/// concrete value is not statically nameable to Anyhow (a boxed trait object).
struct Unnameable;

/// Upper bound on how many links of an arbitrary (possibly cyclic) `source`
/// chain are walked when building an index.
///
/// Anyhow's own context structure is an acyclic linked list, but a user-provided
/// [`StdError::source`] is free to report cycles. Walking stops at this depth;
/// the index then reports [`WalkStatus::Truncated`].
impl ChainIndex<'_> {
    pub const MAX_CHAIN_DEPTH: usize = 64;
}

/// Whether a chain layer was attached as context or is an underlying cause.
///
/// This classification is structural, not textual: layers attached with
/// [`.context(...)`][crate::Context] are [`LayerKind::Context`]; layers reached
/// by following [`StdError::source`] into the held error are
/// [`LayerKind::Cause`].
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum LayerKind {
    /// A layer attached via [`Context`][crate::Context].
    Context,
    /// An underlying error reached through the cause (`source`) chain.
    Cause,
}

/// Outcome of walking the cause chain while building a [`ChainIndex`].
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum WalkStatus {
    /// Every link of the chain was visited; no cycle or depth limit was hit.
    Complete,
    /// The chain was longer than [`ChainIndex::MAX_CHAIN_DEPTH`] and walking
    /// stopped to stay finite.
    Truncated,
    /// Following `source` revisited the same trait object (matching data
    /// address and vtable), so the chain is cyclic (including a self-loop);
    /// walking stopped to stay finite.
    Cycle,
}

/// One indexed layer: either a context layer or an underlying cause.
///
/// A layer borrows the original error and exposes no mutable access; it never
/// renders or stores the layer's message text.
pub struct Layer<'a> {
    kind: LayerKind,
    /// Zero-based position of this layer on the full chain, outermost first.
    depth: usize,
    /// Queryable value when its concrete static type is known to Anyhow
    /// (context values, and the directly held value of a plain/adhoc error).
    identity: Option<&'a (dyn Any + Send + Sync)>,
    /// Monomorphized concrete type name captured where the layer is built.
    /// Stored separately because `type_name_of_val` on a `dyn Any` would only
    /// report the trait-object name.
    concrete_name: Option<&'static str>,
    /// The error as a standard trait object, present for cause layers. A cause
    /// reached through `source()` is only guaranteed `dyn StdError + 'static`
    /// (not `Send + Sync`), so this reference carries no marker bounds.
    cause: Option<&'a (dyn StdError + 'static)>,
}

impl<'a> Layer<'a> {
    /// Whether this layer was attached as context.
    pub fn is_context(&self) -> bool {
        self.kind == LayerKind::Context
    }

    /// Whether this layer is an underlying cause (not attached context).
    pub fn is_cause(&self) -> bool {
        self.kind == LayerKind::Cause
    }

    /// The structural kind of this layer.
    pub fn kind(&self) -> LayerKind {
        self.kind
    }

    /// Zero-based position of this layer on the full chain, outermost first.
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Stable object identity of the value backing this layer, or `None` for a
    /// cause whose concrete type Anyhow cannot name (for example one reached
    /// only through a custom `source`).
    ///
    /// Two layers can hold equal values or identical [`Display`][core::fmt::Display]
    /// text yet still have distinct identities. This address, together with
    /// [`Layer::type_id`], distinguishes them without inspecting messages.
    pub fn identity_addr(&self) -> Option<usize> {
        self.identity
            .map(|any| addr_of!(*any) as *const () as usize)
    }

    /// The [`TypeId`] of the value backing this layer when its concrete static
    /// type is known.
    pub fn type_id(&self) -> Option<TypeId> {
        self.identity.map(Any::type_id)
    }

    /// Best-effort concrete type name for diagnostics. Never evaluates
    /// [`Display`][core::fmt::Display]; returns `None` when the type is unknown.
    pub fn type_name(&self) -> Option<&'static str> {
        self.concrete_name
            .or_else(|| self.cause.map(type_name_of_val))
    }

    /// Downcast a layer whose concrete value is known to Anyhow by its type
    /// identity. This covers context values (which need not implement
    /// `StdError`) and any directly held value. Never inspects message text.
    fn identity_downcast<T>(&self) -> Option<&'a T>
    where
        T: Send + Sync + 'static,
    {
        self.identity
            .and_then(|any| (any as &dyn Any).downcast_ref::<T>())
    }

    /// Downcast this layer to `T` by concrete type identity, considering both a
    /// directly held value and a cause reached as `dyn StdError`.
    ///
    /// This never uses message text. It succeeds for a directly held value of
    /// type `T`, including the bottom cause, preserving Anyhow's original
    /// downcast semantics.
    pub fn downcast_ref<T>(&self) -> Option<&'a T>
    where
        T: StdError + 'static,
    {
        if let Some(value) = self
            .identity
            .and_then(|any| (any as &dyn Any).downcast_ref::<T>())
        {
            return Some(value);
        }
        // Cause layers reached only through a custom source may be known solely
        // as `dyn StdError`; std's downcast compares TypeIds, not Display text.
        self.cause.and_then(|error| error.downcast_ref::<T>())
    }
}

/// A read-only index over every layer of an [`Error`][crate::Error]'s chain.
///
/// Construct one with [`Error::chain_index`][crate::Error::chain_index], or
/// index a bare cause chain with [`ChainIndex::from_sources`]. The index
/// borrows the error and never moves or mutates it.
pub struct ChainIndex<'a> {
    layers: Vec<Layer<'a>>,
    status: WalkStatus,
}

/// Error returned by [`ChainIndex::from_sources`] when the chain head is `None`.
///
/// An empty chain is not a success value: there is no error to index, so
/// indexing fails explicitly rather than producing an index that silently
/// contains no layers.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct EmptyChain;

impl<'a> ChainIndex<'a> {
    /// Index an arbitrary cause chain starting at `head`.
    ///
    /// Every layer reached is classified as [`LayerKind::Cause`]. Returns
    /// [`Err(EmptyChain)`](EmptyChain) if `head` is `None`. Walking is bounded
    /// by depth and by cycle detection, so a `source` that forms a self-loop
    /// terminates with [`WalkStatus::Cycle`] instead of hanging.
    pub fn from_sources(
        head: Option<&'a (dyn StdError + 'static)>,
    ) -> Result<ChainIndex<'a>, EmptyChain> {
        let first = match head {
            Some(first) => first,
            None => return Err(EmptyChain),
        };
        let mut walker = Walker::new();
        walker.begin_dyn_cause(Some(first));
        Ok(walker.finish())
    }

    /// Number of indexed layers (context plus causes).
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Always `false` for an index produced from an [`Error`][crate::Error]: a
    /// real error always has at least the head layer. An empty chain cannot be
    /// constructed over an error; use [`ChainIndex::from_sources`], which
    /// returns [`EmptyChain`] for that case.
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// How the walk ended.
    pub fn status(&self) -> WalkStatus {
        self.status
    }

    /// Whether the whole chain was visited without truncation or a cycle.
    pub fn is_complete(&self) -> bool {
        self.status == WalkStatus::Complete
    }

    /// Iterate all layers, outermost (most recently attached) first.
    pub fn layers(&self) -> Layers<'_, 'a> {
        Layers {
            inner: self.layers.iter(),
        }
    }

    /// Iterate only the context layers, in attach order (outermost first).
    pub fn context_layers(&self) -> impl Iterator<Item = &Layer<'a>> {
        self.layers.iter().filter(|layer| layer.is_context())
    }

    /// Iterate only the underlying cause layers, outermost first.
    pub fn cause_layers(&self) -> impl Iterator<Item = &Layer<'a>> {
        self.layers.iter().filter(|layer| layer.is_cause())
    }

    /// Every context value of concrete type `T`, in attach order.
    ///
    /// Multiple contexts of the same type are each retained as a distinct
    /// layer; identity is the concrete type, never the message text.
    pub fn contexts<T>(&self) -> Vec<&'a T>
    where
        T: Send + Sync + 'static,
    {
        self.layers
            .iter()
            .filter(|layer| layer.is_context())
            .filter_map(Layer::identity_downcast::<T>)
            .collect()
    }

    /// Every context layer or directly held value whose concrete type is `T`,
    /// in chain order. Context values need not implement `StdError`.
    pub fn all<T>(&self) -> Vec<&'a T>
    where
        T: Send + Sync + 'static,
    {
        self.layers
            .iter()
            .filter_map(Layer::identity_downcast::<T>)
            .collect()
    }

    /// The first underlying cause of concrete type `T`, in chain order.
    ///
    /// The bottom concrete type remains downcastable here even after context is
    /// attached above it.
    pub fn cause_ref<T>(&self) -> Option<&'a T>
    where
        T: StdError + 'static,
    {
        self.layers
            .iter()
            .filter(|layer| layer.is_cause())
            .find_map(Layer::downcast_ref::<T>)
    }

    /// Every underlying cause of concrete type `T`, in chain order.
    pub fn causes<T>(&self) -> Vec<&'a T>
    where
        T: StdError + 'static,
    {
        self.layers
            .iter()
            .filter(|layer| layer.is_cause())
            .filter_map(Layer::downcast_ref::<T>)
            .collect()
    }

    /// The last underlying cause: the end of the `source` chain.
    ///
    /// If the walk was truncated or cyclic, this is the deepest cause that was
    /// actually visited rather than a true mathematical root.
    pub fn root_cause(&self) -> Option<&'a (dyn StdError + 'static)> {
        self.layers
            .iter()
            .rev()
            .find(|layer| layer.is_cause())
            .and_then(|layer| layer.cause)
    }

    /// Build a message-free structural [`ChainReport`] from this index.
    pub fn report(&self) -> ChainReport {
        ChainReport::from_index(self)
    }
}

/// Iterator over the layers of a [`ChainIndex`].
pub struct Layers<'index, 'error> {
    inner: core::slice::Iter<'index, Layer<'error>>,
}

impl<'index, 'error> Iterator for Layers<'index, 'error> {
    type Item = &'index Layer<'error>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<'index, 'error> DoubleEndedIterator for Layers<'index, 'error> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.next_back()
    }
}

impl<'index, 'error> ExactSizeIterator for Layers<'index, 'error> {
    fn len(&self) -> usize {
        self.inner.len()
    }
}

/// Message-free structural summary of a layer (see [`ChainReport`]).
#[derive(Clone)]
pub struct LayerReport {
    /// Structural position, outermost first.
    pub depth: usize,
    /// Context versus cause.
    pub kind: LayerKind,
    /// Concrete type name when known. This is a compile-time type name, not a
    /// rendered error message.
    pub type_name: Option<&'static str>,
    /// Concrete type identity when known.
    pub type_id: Option<TypeId>,
    /// Distinguishes two layers of the same type without inspecting messages.
    pub identity_addr: Option<usize>,
}

/// Message-free structural report over an indexed error chain.
///
/// A report deliberately holds no [`Display`][core::fmt::Display] strings and
/// no borrowed error values: only layer kinds, type identities/names, and
/// object addresses. It is `'static` and can outlive the error.
#[derive(Clone)]
pub struct ChainReport {
    layers: Vec<LayerReport>,
    status: WalkStatus,
}

impl ChainReport {
    fn from_index(index: &ChainIndex<'_>) -> Self {
        let layers = index
            .layers
            .iter()
            .map(|layer| LayerReport {
                depth: layer.depth,
                kind: layer.kind,
                type_name: layer.type_name(),
                type_id: layer.type_id(),
                identity_addr: layer.identity_addr(),
            })
            .collect();
        ChainReport {
            layers,
            status: index.status,
        }
    }

    /// Number of reported layers.
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Whether there are no reported layers.
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// How the underlying walk ended.
    pub fn status(&self) -> WalkStatus {
        self.status
    }

    /// Iterate the reported layers in chain order.
    pub fn layers(&self) -> core::slice::Iter<'_, LayerReport> {
        self.layers.iter()
    }

    /// Number of layers (context and causes alike) with the concrete type `T`.
    ///
    /// Counting is by [`TypeId`]; message text is never evaluated.
    pub fn count_of<T>(&self) -> usize
    where
        T: 'static,
    {
        let target = TypeId::of::<T>();
        self.layers
            .iter()
            .filter(|layer| layer.type_id == Some(target))
            .count()
    }

    /// Number of context layers with the concrete type `T`.
    pub fn context_count_of<T>(&self) -> usize
    where
        T: 'static,
    {
        let target = TypeId::of::<T>();
        self.layers
            .iter()
            .filter(|layer| layer.kind == LayerKind::Context)
            .filter(|layer| layer.type_id == Some(target))
            .count()
    }
}

/// Identity of a visited cause: its data address and its trait-object vtable.
///
/// Comparing the whole fat pointer is more precise than the data address
/// alone. A struct and its first field can share a data address while being
/// distinct objects with different vtables; a genuine self-loop yields the
/// very same trait object, so data pointer and vtable both match.
#[derive(Copy, Clone, Eq, PartialEq)]
struct ObjectId {
    data: usize,
    vtable: usize,
}

/// Decompose a `dyn StdError` fat pointer into its data and vtable addresses.
fn object_id(error: &(dyn StdError + 'static)) -> ObjectId {
    let fat: *const dyn StdError = error;
    // A `*const dyn Trait` is laid out as the pair (data pointer, vtable
    // pointer). Transmuting to that pair reads both words without touching the
    // pointee.
    let (data, vtable): (*const (), *const ()) = unsafe { core::mem::transmute(fat) };
    ObjectId {
        data: data as usize,
        vtable: vtable as usize,
    }
}

/// Internal builder driven through Anyhow's per-type vtable while descending
/// the structurally known part of the chain, then bounded-walk any custom
/// `source` links.
pub(crate) struct Walker<'a> {
    layers: Vec<Layer<'a>>,
    // Cause trait objects already visited, keyed by full fat-pointer identity.
    visited: Vec<ObjectId>,
    status: WalkStatus,
}

impl<'a> Walker<'a> {
    pub(crate) fn new() -> Self {
        Walker {
            layers: Vec::new(),
            visited: Vec::new(),
            status: WalkStatus::Complete,
        }
    }

    fn next_depth(&self) -> usize {
        self.layers.len()
    }

    /// Record a context layer (a value attached with `.context`).
    pub(crate) fn context<T>(&mut self, value: &'a T)
    where
        T: Send + Sync + 'static,
    {
        let depth = self.next_depth();
        self.layers.push(Layer {
            kind: LayerKind::Context,
            depth,
            identity: Some(value),
            concrete_name: Some(type_name::<T>()),
            cause: None,
        });
    }

    /// Record a cause layer whose concrete value Anyhow cannot name (for
    /// example a boxed trait object), then bounded-walk its `source` chain.
    pub(crate) fn begin_dyn_cause(&mut self, head: Option<&'a (dyn StdError + 'static)>) {
        self.begin_cause::<Unnameable>(None, head);
    }

    /// Record a cause layer and, from it, bounded-walk the `source` chain.
    ///
    /// `identity` is the directly held value when its static type is known to
    /// Anyhow; `concrete_name` is that value's monomorphized type name. `head`
    /// is the same layer viewed as `dyn StdError` and must be present for the
    /// first call of a bottom cause.
    pub(crate) fn begin_cause<T>(
        &mut self,
        identity: Option<&'a T>,
        head: Option<&'a (dyn StdError + 'static)>,
    ) where
        T: Send + Sync + 'static,
    {
        let mut current = head;
        let mut is_head = true;

        while let Some(error) = current {
            if self.layers.len() >= ChainIndex::MAX_CHAIN_DEPTH {
                self.status = WalkStatus::Truncated;
                break;
            }

            // A cycle is the exact same trait object being revisited. Matching
            // on the full fat pointer (data + vtable) avoids flagging a parent
            // struct that merely lives at the same address as its first field.
            let id = object_id(error);
            if self.visited.contains(&id) {
                self.status = WalkStatus::Cycle;
                break;
            }
            self.visited.push(id);

            // Only the head's concrete identity is structurally known; deeper
            // links reached via `source` report their type name through the
            // trait object instead.
            let (layer_identity, concrete_name) = if is_head {
                match identity {
                    Some(value) => (
                        Some(value as &(dyn Any + Send + Sync)),
                        Some(type_name::<T>()),
                    ),
                    None => (None, None),
                }
            } else {
                (None, None)
            };
            let depth = self.next_depth();
            self.layers.push(Layer {
                kind: LayerKind::Cause,
                depth,
                identity: layer_identity,
                concrete_name,
                cause: Some(error),
            });

            // `source()` returns a plain `dyn StdError + 'static`; following it
            // is the only fallible/cyclic part and is what the bound guards.
            current = error.source();
            is_head = false;
        }
    }

    pub(crate) fn finish(self) -> ChainIndex<'a> {
        ChainIndex {
            layers: self.layers,
            status: self.status,
        }
    }
}
