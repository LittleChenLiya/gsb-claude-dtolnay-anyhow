use crate::{Error, MultiError, StdError};
use alloc::vec::Vec;
use core::fmt::{self, Debug, Display};
use core::slice;

impl MultiError {
    /// Create a multi-error container from its explicitly selected first
    /// error. This is the error that [`std::error::Error::source`] will
    /// point at.
    pub fn new(error: Error) -> Self {
        MultiError {
            errors: alloc::vec![error],
        }
    }

    /// Collect owned errors into a container, preserving their original
    /// order. The first element becomes the error that `source()` points at.
    ///
    /// Returns `None` when `errors` is empty: an empty set of errors does
    /// not constitute a failure.
    pub fn from_errors(errors: Vec<Error>) -> Option<Self> {
        if errors.is_empty() {
            None
        } else {
            Some(MultiError { errors })
        }
    }

    /// Append another independent error, keeping the original order.
    pub fn push(&mut self, error: Error) {
        self.errors.push(error);
    }

    /// The number of errors held by the container. Always at least 1.
    pub fn len(&self) -> usize {
        self.errors.len()
    }

    /// Always `false`: an empty `MultiError` cannot be constructed.
    pub fn is_empty(&self) -> bool {
        false
    }

    /// The explicitly selected first error, the same one that `source()`
    /// points at.
    pub fn first(&self) -> &Error {
        &self.errors[0]
    }

    /// Borrow every error in its original order. Each element retains its
    /// own context chain and downcast ability.
    pub fn iter(&self) -> slice::Iter<'_, Error> {
        self.errors.iter()
    }

    /// Consume the container, moving ownership of the errors out exactly
    /// once. The returned vector preserves the original order.
    pub fn into_errors(self) -> Vec<Error> {
        self.errors
    }
}

impl Display for MultiError {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        Display::fmt(&self.errors[0], formatter)?;
        let rest = self.errors.len() - 1;
        if rest > 0 {
            write!(
                formatter,
                " (and {} more error{})",
                rest,
                if rest == 1 { "" } else { "s" },
            )?;
        }
        Ok(())
    }
}

impl Debug for MultiError {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.debug_list().entries(&self.errors).finish()
    }
}

#[cfg(any(feature = "std", not(anyhow_no_core_error)))]
impl StdError for MultiError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        // The explicitly selected first error. Its own `source` chain keeps
        // that error's context chain reachable from here.
        self.errors[0].chain().next()
    }
}

#[cfg(all(not(feature = "std"), anyhow_no_core_error))]
impl StdError for MultiError {}

impl<'a> IntoIterator for &'a MultiError {
    type Item = &'a Error;
    type IntoIter = slice::Iter<'a, Error>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
