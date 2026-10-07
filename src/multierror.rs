//! A small owned container for multiple independent [`Error`] values.
//!
//! [`MultiError`] owns one or more `anyhow::Error` values. Every contained
//! error keeps its own context chain and downcast capability; the container
//! does not flatten or merge them.
//!
//! The first error is explicitly selected as the error reported through
//! [`std::error::Error::source`]; the remaining errors stay borrowable in
//! their original order via [`MultiError::other_errors`] /
//! [`MultiError::iter`]. Ownership of every contained error is transferred
//! exactly once, by [`MultiError::into_errors`].
//!
//! An empty collection can never produce a `MultiError`.

use crate::Error;
use crate::StdError;
use alloc::vec::Vec;
use core::fmt::{self, Debug, Display, Formatter};
use core::result::Result as CoreResult;

/// Error returned when attempting to build a [`MultiError`] from an empty
/// collection of errors.
///
/// A failure result with zero errors carries no information, so construction
/// rejects the empty case instead of producing an empty `MultiError`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct EmptyErrors;

impl Display for EmptyErrors {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        formatter.write_str("cannot build a MultiError from zero errors")
    }
}

#[cfg(any(feature = "std", not(anyhow_no_core_error)))]
impl StdError for EmptyErrors {}

/// A non-empty collection of one or more independently owned [`Error`] values.
///
/// Each element preserves its own context chain and can be downcast on its
/// own. The element at index 0 is the *primary* error: it is what
/// [`MultiError::source`] reports. The elements at later indices are the
/// *other* errors and remain queryable by reference in their original order.
///
/// # Example
///
/// ```
/// use anyhow::{Context, Error, MultiError};
///
/// # #[derive(Debug)]
/// # struct Io;
/// # impl std::fmt::Display for Io {
/// #     fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
/// #         write!(f, "io error")
/// #     }
/// # }
/// # impl std::error::Error for Io {}
/// #
/// # #[derive(Debug)]
/// # struct Parse;
/// # impl std::fmt::Display for Parse {
/// #     fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
/// #         write!(f, "parse error")
/// #     }
/// # }
/// # impl std::error::Error for Parse {}
/// #
/// # fn main() -> anyhow::Result<()> {
/// let failures = vec![
///     Error::from(Io).context("loading config"),
///     Error::from(Parse).context("reading header"),
/// ];
///
/// let multi = MultiError::new(failures)?;
/// assert_eq!(multi.len(), 2);
///
/// // The explicitly selected first error is the reported source ...
/// let source: &dyn std::error::Error = std::error::Error::source(&multi).unwrap();
/// assert_eq!(source.to_string(), "loading config");
///
/// // ... and the rest are still borrowed in their original order.
/// let rest: Vec<String> = multi.other_errors().map(|error| error.to_string()).collect();
/// assert_eq!(rest, vec!["reading header"]);
/// # Ok(())
/// # }
/// ```
pub struct MultiError {
    errors: Vec<Error>,
}

impl MultiError {
    /// Build a `MultiError` from owned errors.
    ///
    /// The first error becomes the primary error reported by
    /// [`MultiError::source`]. All errors are kept in their original order.
    ///
    /// # Errors
    ///
    /// Returns [`EmptyErrors`] if `errors` is empty: an empty collection can
    /// not represent a failure.
    pub fn new<I>(errors: I) -> CoreResult<Self, EmptyErrors>
    where
        I: IntoIterator<Item = Error>,
    {
        let errors: Vec<Error> = errors.into_iter().collect();
        if errors.is_empty() {
            Err(EmptyErrors)
        } else {
            Ok(MultiError { errors })
        }
    }

    /// The explicitly selected first (primary) error.
    #[must_use]
    pub fn first(&self) -> &Error {
        // Invariant: `errors` is never empty; construction rejects that case.
        &self.errors[0]
    }

    /// The errors following the first one, borrowed in their original order.
    pub fn other_errors(&self) -> impl Iterator<Item = &Error> {
        self.errors[1..].iter()
    }

    /// Every contained error in its original order, starting with the primary
    /// error.
    pub fn iter(&self) -> impl Iterator<Item = &Error> {
        self.errors.iter()
    }

    /// Borrow the full slice of owned errors in their original order.
    #[must_use]
    pub fn errors(&self) -> &[Error] {
        &self.errors
    }

    /// Build a multi-error from a vector of failures collected through the
    /// regular `anyhow::Result`/`?` interface, returning [`EmptyErrors`] when
    /// nothing failed.
    ///
    /// This is a short regression example of how the existing interface and
    /// the container fit together:
    ///
    /// ```
    /// use anyhow::{Context, MultiError};
    ///
    /// fn load_all() -> anyhow::Result<()> {
    ///     # fn fail_io() -> anyhow::Result<()> {
    ///     #     Err(anyhow::Error::from(std::io::Error::new(
    ///     #         std::io::ErrorKind::NotFound,
    ///     #         "missing",
    ///     #     )))
    ///     #     .context("io step")
    ///     # }
    ///     # fn fail_parse() -> anyhow::Result<()> {
    ///     #     Err(anyhow::Error::msg("bad token")).context("parse step")
    ///     # }
    ///     let mut failures = Vec::new();
    ///     if let Err(error) = fail_io() {
    ///         failures.push(error);
    ///     }
    ///     if let Err(error) = fail_parse() {
    ///         failures.push(error);
    ///     }
    ///     match MultiError::from_vec(failures) {
    ///         Ok(multi) => Err(multi.into()),
    ///         Err(_empty) => Ok(()),
    ///     }
    /// }
    ///
    /// let error = load_all().unwrap_err();
    /// assert_eq!(error.to_string(), "io step (+1 more error)");
    ///
    /// let multi = error.downcast_ref::<MultiError>().unwrap();
    /// let messages: Vec<String> = multi.iter().map(ToString::to_string).collect();
    /// assert_eq!(messages, vec!["io step", "parse step"]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`EmptyErrors`] if `errors` is empty.
    pub fn from_vec(errors: Vec<Error>) -> CoreResult<Self, EmptyErrors> {
        Self::new(errors)
    }

    /// Number of contained errors, always at least 1.
    #[must_use]
    pub fn len(&self) -> usize {
        self.errors.len()
    }

    /// Always `false`: a `MultiError` can never be empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Consume the container, transferring ownership of every error exactly
    /// once. The returned vector is guaranteed to be non-empty and to keep the
    /// errors in their original order.
    #[must_use]
    pub fn into_errors(self) -> Vec<Error> {
        self.errors
    }
}

impl Debug for MultiError {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        formatter.debug_list().entries(self.errors.iter()).finish()
    }
}

impl Display for MultiError {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        Display::fmt(self.first(), formatter)?;
        let other_count = self.errors.len() - 1;
        if other_count == 1 {
            formatter.write_str(" (+1 more error)")
        } else if other_count > 1 {
            write!(formatter, " (+{other_count} more errors)")
        } else {
            Ok(())
        }
    }
}

#[cfg(any(feature = "std", not(anyhow_no_core_error)))]
impl StdError for MultiError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        // Explicitly selected first error. `Error` derefs to
        // `dyn StdError + Send + Sync + 'static`, which coerces to
        // `dyn StdError + 'static`.
        Some(&**self.first())
    }
}
