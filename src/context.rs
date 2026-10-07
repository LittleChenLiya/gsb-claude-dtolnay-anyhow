use crate::error::ContextError;
#[cfg(all(
    feature = "fallible-context",
    any(feature = "std", not(anyhow_no_core_error))
))]
use crate::error::{FailContextChainError, FailContextError};
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

        #[cfg(all(
            feature = "fallible-context",
            any(feature = "std", not(anyhow_no_core_error))
        ))]
        fn ext_fail_context<C, F, G>(self, factory: G) -> Error
        where
            C: Display + Send + Sync + 'static,
            F: crate::StdError + Send + Sync + 'static,
            G: FnOnce() -> Result<C, F>;
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

        #[cfg(all(
            feature = "fallible-context",
            any(feature = "std", not(anyhow_no_core_error))
        ))]
        fn ext_fail_context<C, F, G>(self, factory: G) -> Error
        where
            C: Display + Send + Sync + 'static,
            F: crate::StdError + Send + Sync + 'static,
            G: FnOnce() -> Result<C, F>,
        {
            // The factory runs at most once, and only because self is the Err
            // value of a Result. On success the ordinary context path is used;
            // on failure the business error (`self`) is retained as the root.
            match factory() {
                Ok(context) => {
                    let backtrace = backtrace_if_absent!(&self);
                    Error::construct_from_context(context, self, backtrace)
                }
                Err(factory_error) => {
                    let backtrace = backtrace_if_absent!(&self);
                    Error::construct_from_fail_context(self, factory_error, backtrace)
                }
            }
        }
    }

    impl StdError for Error {
        fn ext_context<C>(self, context: C) -> Error
        where
            C: Display + Send + Sync + 'static,
        {
            self.context(context)
        }

        #[cfg(all(
            feature = "fallible-context",
            any(feature = "std", not(anyhow_no_core_error))
        ))]
        fn ext_fail_context<C, F, G>(self, factory: G) -> Error
        where
            C: Display + Send + Sync + 'static,
            F: crate::StdError + Send + Sync + 'static,
            G: FnOnce() -> Result<C, F>,
        {
            self.fail_context(factory)
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

    // Only available with the non-default `fallible-context` feature enabled;
    // the trait object otherwise carries exactly its original two methods.
    #[cfg(all(
        feature = "fallible-context",
        any(feature = "std", not(anyhow_no_core_error))
    ))]
    fn with_fallible_context<C, F, G>(self, factory: G) -> Result<T, Error>
    where
        C: Display + Send + Sync + 'static,
        F: StdError + Send + Sync + 'static,
        G: FnOnce() -> Result<C, F>,
    {
        // Not using map_err to save 2 useless frames off the captured backtrace
        // in ext_fail_context, and to guarantee the factory is never invoked on
        // the Ok branch.
        match self {
            Ok(ok) => Ok(ok),
            Err(error) => Err(error.ext_fail_context(factory)),
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

    #[cfg(all(
        feature = "fallible-context",
        any(feature = "std", not(anyhow_no_core_error))
    ))]
    fn with_fallible_context<C, F, G>(self, factory: G) -> Result<T, Error>
    where
        C: Display + Send + Sync + 'static,
        F: StdError + Send + Sync + 'static,
        G: FnOnce() -> Result<C, F>,
    {
        match self {
            // An Option has no business error, so the factory never runs on
            // Some, and a factory failure on None is itself the whole error.
            Some(ok) => Ok(ok),
            None => match factory() {
                Ok(context) => Err(Error::construct_from_display(context, backtrace!())),
                Err(factory_error) => Err(Error::new(factory_error)),
            },
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

// When context attachment fails, the business error `error` remains the
// displayed error and the head of the source chain; `factory_error` is stored
// beside it purely for downcast inspection and is deliberately not returned by
// source(), so it can never masquerade as the business root cause.
#[cfg(all(
    feature = "fallible-context",
    any(feature = "std", not(anyhow_no_core_error))
))]
impl<E, F> Debug for FailContextError<E, F>
where
    E: Debug,
    F: Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Error")
            .field("error", &self.error)
            .field("context_factory_error", &self.factory_error)
            .finish()
    }
}

#[cfg(all(
    feature = "fallible-context",
    any(feature = "std", not(anyhow_no_core_error))
))]
impl<E, F> Display for FailContextError<E, F>
where
    E: Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Display::fmt(&self.error, f)
    }
}

#[cfg(all(
    feature = "fallible-context",
    any(feature = "std", not(anyhow_no_core_error))
))]
impl<E, F> StdError for FailContextError<E, F>
where
    E: StdError + 'static,
    F: StdError + 'static,
{
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        // Forward straight to the business error's own cause rather than
        // yielding the business error itself: the outer anyhow node already
        // carries it (so it is downcastable and is the root when it has no
        // cause), and this avoids printing the business message twice in the
        // "Caused by:" chain. The factory error is deliberately absent here
        // and is reachable only via downcast.
        self.error.source()
    }

    #[cfg(error_generic_member_access)]
    fn provide<'a>(&'a self, request: &mut Request<'a>) {
        nightly::provide(&self.error, request);
    }
}

#[cfg(all(
    feature = "fallible-context",
    any(feature = "std", not(anyhow_no_core_error))
))]
impl<F> Debug for FailContextChainError<F>
where
    F: Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Error")
            .field("error", &self.error)
            .field("context_factory_error", &self.factory_error)
            .finish()
    }
}

#[cfg(all(
    feature = "fallible-context",
    any(feature = "std", not(anyhow_no_core_error))
))]
impl<F> Display for FailContextChainError<F> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Display::fmt(&self.error, f)
    }
}

#[cfg(all(
    feature = "fallible-context",
    any(feature = "std", not(anyhow_no_core_error))
))]
impl<F> StdError for FailContextChainError<F>
where
    F: StdError + 'static,
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
