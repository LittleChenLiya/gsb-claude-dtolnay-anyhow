use crate::error::ContextError;
use crate::{Context, Error, StdError};
use core::convert::Infallible;
use core::fmt::{self, Debug, Display, Write};

#[cfg(error_generic_member_access)]
use crate::nightly::{self, Request};

mod ext {
    use super::*;

    pub trait StdError {
        fn ext_context<C>(self, context: C) -> Error
        where
            C: Display + Send + Sync + 'static;
    }

    #[cfg(any(feature = "std", not(anyhow_no_core_error)))]
    impl<E> StdError for E
    where
        E: crate::StdError + Send + Sync + 'static,
    {
        fn ext_context<C>(self, context: C) -> Error
        where
            C: Display + Send + Sync + 'static,
        {
            let backtrace = backtrace_if_absent!(&self);
            Error::construct_from_context(context, self, backtrace)
        }
    }

    impl StdError for Error {
        fn ext_context<C>(self, context: C) -> Error
        where
            C: Display + Send + Sync + 'static,
        {
            self.context(context)
        }
    }
}

impl<T, E> Context<T, E> for Result<T, E>
where
    E: ext::StdError + Send + Sync + 'static,
{
    fn context<C>(self, context: C) -> Result<T, Error>
    where
        C: Display + Send + Sync + 'static,
    {
        // Not using map_err to save 2 useless frames off the captured backtrace
        // in ext_context.
        match self {
            Ok(ok) => Ok(ok),
            Err(error) => Err(error.ext_context(context)),
        }
    }

    fn with_context<C, F>(self, context: F) -> Result<T, Error>
    where
        C: Display + Send + Sync + 'static,
        F: FnOnce() -> C,
    {
        match self {
            Ok(ok) => Ok(ok),
            Err(error) => Err(error.ext_context(context())),
        }
    }

    fn try_context<C, FE>(self, context: Result<C, FE>) -> Result<T, Error>
    where
        C: Display + Send + Sync + 'static,
        FE: Display + Send + Sync + 'static,
    {
        // Not using map_err to save 2 useless frames off the captured backtrace
        // in ext_context.
        match self {
            Ok(ok) => Ok(ok),
            Err(error) => Err(match context {
                Ok(context) => error.ext_context(context),
                Err(factory_error) => error.ext_context(ContextFactoryError { factory_error }),
            }),
        }
    }

    fn try_with_context<C, F, FE>(self, context: F) -> Result<T, Error>
    where
        C: Display + Send + Sync + 'static,
        F: FnOnce() -> Result<C, FE>,
        FE: Display + Send + Sync + 'static,
    {
        match self {
            Ok(ok) => Ok(ok),
            Err(error) => Err(match context() {
                Ok(context) => error.ext_context(context),
                Err(factory_error) => error.ext_context(ContextFactoryError { factory_error }),
            }),
        }
    }
}

/// ```
/// # type T = ();
/// #
/// use anyhow::{Context, Result};
///
/// fn maybe_get() -> Option<T> {
///     # const IGNORE: &str = stringify! {
///     ...
///     # };
///     # unimplemented!()
/// }
///
/// fn demo() -> Result<()> {
///     let t = maybe_get().context("there is no T")?;
///     # const IGNORE: &str = stringify! {
///     ...
///     # };
///     # unimplemented!()
/// }
/// ```
impl<T> Context<T, Infallible> for Option<T> {
    fn context<C>(self, context: C) -> Result<T, Error>
    where
        C: Display + Send + Sync + 'static,
    {
        // Not using ok_or_else to save 2 useless frames off the captured
        // backtrace.
        match self {
            Some(ok) => Ok(ok),
            None => Err(Error::construct_from_display(context, backtrace!())),
        }
    }

    fn with_context<C, F>(self, context: F) -> Result<T, Error>
    where
        C: Display + Send + Sync + 'static,
        F: FnOnce() -> C,
    {
        match self {
            Some(ok) => Ok(ok),
            None => Err(Error::construct_from_display(context(), backtrace!())),
        }
    }

    fn try_context<C, FE>(self, context: Result<C, FE>) -> Result<T, Error>
    where
        C: Display + Send + Sync + 'static,
        FE: Display + Send + Sync + 'static,
    {
        // Not using ok_or_else to save 2 useless frames off the captured
        // backtrace.
        match self {
            Some(ok) => Ok(ok),
            None => Err(match context {
                Ok(context) => Error::construct_from_display(context, backtrace!()),
                Err(factory_error) => Error::construct_from_display(
                    ContextFactoryError { factory_error },
                    backtrace!(),
                ),
            }),
        }
    }

    fn try_with_context<C, F, FE>(self, context: F) -> Result<T, Error>
    where
        C: Display + Send + Sync + 'static,
        F: FnOnce() -> Result<C, FE>,
        FE: Display + Send + Sync + 'static,
    {
        match self {
            Some(ok) => Ok(ok),
            None => Err(match context() {
                Ok(context) => Error::construct_from_display(context, backtrace!()),
                Err(factory_error) => Error::construct_from_display(
                    ContextFactoryError { factory_error },
                    backtrace!(),
                ),
            }),
        }
    }
}

impl<C, E> Debug for ContextError<C, E>
where
    C: Display,
    E: Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Error")
            .field("context", &Quoted(&self.context))
            .field("source", &self.error)
            .finish()
    }
}

impl<C, E> Display for ContextError<C, E>
where
    C: Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Display::fmt(&self.context, f)
    }
}

impl<C, E> StdError for ContextError<C, E>
where
    C: Display,
    E: StdError + 'static,
{
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        Some(&self.error)
    }

    #[cfg(error_generic_member_access)]
    fn provide<'a>(&'a self, request: &mut Request<'a>) {
        nightly::provide(&self.error, request);
    }
}

impl<C> StdError for ContextError<C, Error>
where
    C: Display,
{
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        Some(unsafe { crate::ErrorImpl::error(self.error.inner.by_ref()) })
    }

    #[cfg(error_generic_member_access)]
    fn provide<'a>(&'a self, request: &mut Request<'a>) {
        Error::provide(&self.error, request);
    }
}

struct Quoted<C>(C);

impl<C> Debug for Quoted<C>
where
    C: Display,
{
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_char('"')?;
        Quoted(&mut *formatter).write_fmt(format_args!("{}", self.0))?;
        formatter.write_char('"')?;
        Ok(())
    }
}

impl Write for Quoted<&mut fmt::Formatter<'_>> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        Display::fmt(&s.escape_debug(), self.0)
    }
}

pub(crate) mod private {
    use super::*;

    pub trait Sealed {}

    impl<T, E> Sealed for Result<T, E> where E: ext::StdError {}
    impl<T> Sealed for Option<T> {}
}

/// Context attached by [`Context::try_context`] and
/// [`Context::try_with_context`] when the context factory itself fails.
///
/// This is stored as an additional context layer *around* the original
/// error, so the business root cause remains available for
/// [`Error::downcast`](crate::Error::downcast) while the factory's failure
/// reason is preserved separately inside this layer. Both causes can be
/// inspected independently:
///
/// ```
/// # use anyhow::{Context, ContextFactoryError};
/// # use std::io;
/// let result: anyhow::Result<()> = Err(io::Error::new(io::ErrorKind::Other, "boom").into());
/// let err = result
///     .try_with_context(|| -> Result<&'static str, io::Error> {
///         Err(io::Error::new(io::ErrorKind::NotFound, "no metadata"))
///     })
///     .unwrap_err();
///
/// // The original error is still the downcastable root cause.
/// assert!(err.downcast_ref::<io::Error>().is_some());
///
/// // The factory failure is stored alongside, not in place of it.
/// let note = err
///     .downcast_ref::<ContextFactoryError<io::Error>>()
///     .unwrap();
/// assert_eq!(note.factory_error().kind(), io::ErrorKind::NotFound);
/// ```
pub struct ContextFactoryError<F> {
    pub(crate) factory_error: F,
}

impl<F> ContextFactoryError<F> {
    /// The error reported by the fallible context factory, preserved
    /// alongside (never instead of) the original error.
    pub fn factory_error(&self) -> &F {
        &self.factory_error
    }

    /// Unwrap this context layer, returning the factory's failure reason.
    pub fn into_factory_error(self) -> F {
        self.factory_error
    }
}

impl<F> Debug for ContextFactoryError<F>
where
    F: Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("ContextFactoryError")
            .field("factory_error", &Quoted(&self.factory_error))
            .finish()
    }
}

impl<F> Display for ContextFactoryError<F>
where
    F: Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "failed to produce error context: {}", self.factory_error,)
    }
}
