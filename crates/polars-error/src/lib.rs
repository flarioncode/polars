#![feature(error_generic_member_access)]
pub mod constants;
mod warning;

use std::backtrace::Backtrace;
use std::borrow::Cow;
use std::collections::TryReserveError;
use std::error::Error;
use std::fmt::{self, Display, Formatter, Write};
use std::ops::Deref;
use std::sync::{Arc, LazyLock};
use std::{backtrace, env, io};

pub use warning::*;

enum ErrorStrategy {
    Panic,
    WithBacktrace,
    Normal,
}

static ERROR_STRATEGY: LazyLock<ErrorStrategy> = LazyLock::new(|| {
    if env::var("POLARS_PANIC_ON_ERR").as_deref() == Ok("1") {
        ErrorStrategy::Panic
    } else if env::var("POLARS_BACKTRACE_IN_ERR").as_deref() == Ok("1") {
        ErrorStrategy::WithBacktrace
    } else {
        ErrorStrategy::Normal
    }
});

#[derive(Debug, Clone)]
pub struct ErrString(Cow<'static, str>);

impl ErrString {
    pub const fn new_static(s: &'static str) -> Self {
        Self(Cow::Borrowed(s))
    }
}

impl<T> From<T> for ErrString
where
    T: Into<Cow<'static, str>>,
{
    fn from(msg: T) -> Self {
        match &*ERROR_STRATEGY {
            ErrorStrategy::Panic => panic!("{}", msg.into()),
            ErrorStrategy::WithBacktrace => ErrString(Cow::Owned(format!(
                "{}\n\nRust backtrace:\n{}",
                msg.into(),
                std::backtrace::Backtrace::force_capture()
            ))),
            ErrorStrategy::Normal => ErrString(msg.into()),
        }
    }
}

impl AsRef<str> for ErrString {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl Deref for ErrString {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Display for ErrString {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, thiserror::Error, Clone)]
pub enum PolarsError {
    #[error("not found: {0}")]
    ColumnNotFound(ErrString, #[backtrace] Arc<Backtrace>),
    #[error("{0}")]
    ComputeError(ErrString, #[backtrace] Arc<Backtrace>),
    #[error("duplicate: {0}")]
    Duplicate(ErrString, #[backtrace] Arc<Backtrace>),
    #[error("{0}")]
    InvalidOperation(ErrString, #[backtrace] Arc<Backtrace>),
    #[error("{}", match msg {
        Some(msg) => format!("{}", msg),
        None => format!("{}", error)
    })]
    IO {
        error: Arc<io::Error>,
        msg: Option<ErrString>,
    },
    #[error("no data: {0}")]
    NoData(ErrString, #[backtrace] Arc<Backtrace>),
    #[error("{0}")]
    OutOfBounds(ErrString, #[backtrace] Arc<Backtrace>),
    #[error("field not found: {0}")]
    SchemaFieldNotFound(ErrString, #[backtrace] Arc<Backtrace>),
    #[error("{0}")]
    SchemaMismatch(ErrString, #[backtrace] Arc<Backtrace>),
    #[error("lengths don't match: {0}")]
    ShapeMismatch(ErrString, #[backtrace] Arc<Backtrace>),
    #[error("{0}")]
    SQLInterface(ErrString, #[backtrace] Arc<Backtrace>),
    #[error("{0}")]
    SQLSyntax(ErrString, #[backtrace] Arc<Backtrace>),
    #[error("string caches don't match: {0}")]
    StringCacheMismatch(ErrString, #[backtrace] Arc<Backtrace>),
    #[error("field not found: {0}")]
    StructFieldNotFound(ErrString, #[backtrace] Arc<Backtrace>),
    #[error("{error}: {msg}")]
    Context {
        error: Box<PolarsError>,
        msg: ErrString,
    },
}

impl From<io::Error> for PolarsError {
    fn from(value: io::Error) -> Self {
        PolarsError::IO {
            error: Arc::new(value),
            msg: None,
        }
    }
}

#[cfg(feature = "regex")]
impl From<regex::Error> for PolarsError {
    fn from(err: regex::Error) -> Self {
        PolarsError::ComputeError(format!("regex error: {err}").into(), Arc::new(Backtrace::capture()))
    }
}

#[cfg(feature = "object_store")]
impl From<object_store::Error> for PolarsError {
    fn from(err: object_store::Error) -> Self {
        std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("object-store error: {err:?}"),
        )
        .into()
    }
}

#[cfg(feature = "avro-schema")]
impl From<avro_schema::error::Error> for PolarsError {
    fn from(value: avro_schema::error::Error) -> Self {
        polars_err!(ComputeError: "avro-error: {}", value)
    }
}

impl From<simdutf8::basic::Utf8Error> for PolarsError {
    fn from(value: simdutf8::basic::Utf8Error) -> Self {
        polars_err!(ComputeError: "invalid utf8: {}", value)
    }
}
#[cfg(feature = "arrow-format")]
impl From<arrow_format::ipc::planus::Error> for PolarsError {
    fn from(err: arrow_format::ipc::planus::Error) -> Self {
        polars_err!(ComputeError: "parquet error: {err:?}")
    }
}

impl From<TryReserveError> for PolarsError {
    fn from(value: TryReserveError) -> Self {
        polars_err!(ComputeError: "OOM: {}", value)
    }
}

pub type PolarsResult<T> = Result<T, PolarsError>;

impl PolarsError {
    pub fn context_trace(self) -> Self {
        use PolarsError::*;
        match self {
            Context { error, msg } => {
                // If context is 1 level deep, just return error.
                if !matches!(&*error, PolarsError::Context { .. }) {
                    return *error;
                }
                let mut current_error = &*error;
                let material_error = error.get_err();

                let mut messages = vec![&msg];

                while let PolarsError::Context { msg, error } = current_error {
                    current_error = error;
                    messages.push(msg)
                }

                let mut bt = String::new();

                let mut count = 0;
                while let Some(msg) = messages.pop() {
                    count += 1;
                    writeln!(&mut bt, "\t[{count}] {}", msg).unwrap();
                }
                material_error.wrap_msg(move |msg| {
                    format!("{msg}\n\nThis error occurred with the following context stack:\n{bt}")
                })
            },
            err => err,
        }
    }

    pub fn wrap_msg<F: FnOnce(&str) -> String>(&self, func: F) -> Self {
        use PolarsError::*;
        match self {
            ColumnNotFound(msg, b) => ColumnNotFound(func(msg).into(), b.clone()),
            ComputeError(msg, b) => ComputeError(func(msg).into(), b.clone()),
            Duplicate(msg, b) => Duplicate(func(msg).into(), b.clone()),
            InvalidOperation(msg, b) => InvalidOperation(func(msg).into(), b.clone()),
            IO { error, msg } => {
                let msg = match msg {
                    Some(msg) => func(msg),
                    None => func(&format!("{}", error)),
                };
                IO {
                    error: error.clone(),
                    msg: Some(msg.into()),
                }
            },
            NoData(msg, b) => NoData(func(msg).into(), b.clone()),
            OutOfBounds(msg, b) => OutOfBounds(func(msg).into(), b.clone()),
            SchemaFieldNotFound(msg, b) => SchemaFieldNotFound(func(msg).into(), b.clone()),
            SchemaMismatch(msg, b) => SchemaMismatch(func(msg).into(), b.clone()),
            ShapeMismatch(msg, b) => ShapeMismatch(func(msg).into(), b.clone()),
            StringCacheMismatch(msg, b) => StringCacheMismatch(func(msg).into(), b.clone()),
            StructFieldNotFound(msg, b) => StructFieldNotFound(func(msg).into(), b.clone()),
            SQLInterface(msg, b) => SQLInterface(func(msg).into(), b.clone()),
            SQLSyntax(msg, b) => SQLSyntax(func(msg).into(), b.clone()),
            _ => unreachable!(),
        }
    }

    fn get_err(&self) -> &Self {
        use PolarsError::*;
        match self {
            Context { error, .. } => error.get_err(),
            err => err,
        }
    }

    pub fn context(self, msg: ErrString) -> Self {
        PolarsError::Context {
            msg,
            error: Box::new(self),
        }
    }

    pub fn backtrace(&self) -> Arc<Backtrace> {
        match self {
            PolarsError::ColumnNotFound(_, arc) => arc.clone(),
            PolarsError::ComputeError(_, arc) => arc.clone(),
            PolarsError::Duplicate(_, arc) => arc.clone(),
            PolarsError::InvalidOperation(_, arc) => arc.clone(),
            PolarsError::IO { error: _, msg: _ } => todo!(),
            PolarsError::NoData(_, arc) => arc.clone(),
            PolarsError::OutOfBounds(_, arc) => arc.clone(),
            PolarsError::SchemaFieldNotFound(_, arc) => arc.clone(),
            PolarsError::SchemaMismatch(_, arc) => arc.clone(),
            PolarsError::ShapeMismatch(_, arc) => arc.clone(),
            PolarsError::SQLInterface(_, arc) => arc.clone(),
            PolarsError::SQLSyntax(_, arc) => arc.clone(),
            PolarsError::StringCacheMismatch(_, arc) => arc.clone(),
            PolarsError::StructFieldNotFound(_, arc) => arc.clone(),
            PolarsError::Context { error, msg } => todo!(),
        }
    }
}

pub fn map_err<E: Error>(error: E) -> PolarsError {
    PolarsError::ComputeError(format!("{error}").into(), Arc::new(Backtrace::capture()))
}

#[macro_export]
macro_rules! polars_err {
    ($variant:ident: $fmt:literal $(, $arg:expr)* $(,)?) => {
        $crate::__private::must_use(
            $crate::PolarsError::$variant(format!($fmt, $($arg),*).into(), std::sync::Arc::new(std::backtrace::Backtrace::capture()))
        )
    };
    ($variant:ident: $err:expr $(,)?) => {
        $crate::__private::must_use(
            $crate::PolarsError::$variant($err.into(), std::sync::Arc::new(std::backtrace::Backtrace::capture()))
        )
    };
    (expr = $expr:expr, $variant:ident: $err:expr $(,)?) => {
        $crate::__private::must_use(
            $crate::PolarsError::$variant(
                format!("{}\n\nError originated in expression: '{:?}'", $err, $expr).into(),
                std::sync::Arc::new(std::backtrace::Backtrace::capture())
            )
        )
    };
    (expr = $expr:expr, $variant:ident: $fmt:literal, $($arg:tt)+) => {
        polars_err!(expr = $expr, $variant: format!($fmt, $($arg)+))
    };
    (op = $op:expr, got = $arg:expr, expected = $expected:expr) => {
        $crate::polars_err!(
            InvalidOperation: "{} operation not supported for dtype `{}` (expected: {})",
            $op, $arg, $expected
        )
    };
    (opq = $op:ident, got = $arg:expr, expected = $expected:expr) => {
        $crate::polars_err!(
            op = concat!("`", stringify!($op), "`"), got = $arg, expected = $expected
        )
    };
    (un_impl = $op:ident) => {
        $crate::polars_err!(
            InvalidOperation: "{} operation is not implemented.", concat!("`", stringify!($op), "`")
        )
    };
    (op = $op:expr, $arg:expr) => {
        $crate::polars_err!(
            InvalidOperation: "{} operation not supported for dtype `{}`", $op, $arg
        )
    };
    (op = $op:expr, $arg:expr, hint = $hint:literal) => {
        $crate::polars_err!(
            InvalidOperation: "{} operation not supported for dtype `{}`\n\nHint: {}", $op, $arg, $hint
        )
    };
    (op = $op:expr, $lhs:expr, $rhs:expr) => {
        $crate::polars_err!(
            InvalidOperation: "{} operation not supported for dtypes `{}` and `{}`", $op, $lhs, $rhs
        )
    };
    (oos = $($tt:tt)+) => {
        $crate::polars_err!(ComputeError: "out-of-spec: {}", $($tt)+)
    };
    (nyi = $($tt:tt)+) => {
        $crate::polars_err!(ComputeError: "not yet implemented: {}", format!($($tt)+) )
    };
    (opq = $op:ident, $arg:expr) => {
        $crate::polars_err!(op = concat!("`", stringify!($op), "`"), $arg)
    };
    (opq = $op:ident, $lhs:expr, $rhs:expr) => {
        $crate::polars_err!(op = stringify!($op), $lhs, $rhs)
    };
    (append) => {
        polars_err!(SchemaMismatch: "cannot append series, data types don't match")
    };
    (extend) => {
        polars_err!(SchemaMismatch: "cannot extend series, data types don't match")
    };
    (unpack) => {
        polars_err!(SchemaMismatch: "cannot unpack series, data types don't match")
    };
    (not_in_enum,value=$value:expr,categories=$categories:expr) =>{
        polars_err!(ComputeError: "value '{}' is not present in Enum: {:?}",$value,$categories)
    };
    (string_cache_mismatch) => {
        polars_err!(StringCacheMismatch: r#"
cannot compare categoricals coming from different sources, consider setting a global StringCache.

Help: if you're using Python, this may look something like:

    with pl.StringCache():
        # Initialize Categoricals.
        df1 = pl.DataFrame({'a': ['1', '2']}, schema={'a': pl.Categorical})
        df2 = pl.DataFrame({'a': ['1', '3']}, schema={'a': pl.Categorical})
    # Your operations go here.
    pl.concat([df1, df2])

Alternatively, if the performance cost is acceptable, you could just set:

    import polars as pl
    pl.enable_string_cache()

on startup."#.trim_start())
    };
    (duplicate = $name:expr) => {
        polars_err!(Duplicate: "column with name '{}' has more than one occurrences", $name)
    };
    (col_not_found = $name:expr) => {
        polars_err!(ColumnNotFound: "{:?} not found", $name)
    };
    (oob = $idx:expr, $len:expr) => {
        polars_err!(OutOfBounds: "index {} is out of bounds for sequence of length {}", $idx, $len)
    };
    (agg_len = $agg_len:expr, $groups_len:expr) => {
        polars_err!(
            ComputeError:
            "returned aggregation is of different length: {} than the groups length: {}",
            $agg_len, $groups_len
        )
    };
    (parse_fmt_idk = $dtype:expr) => {
        polars_err!(
            ComputeError: "could not find an appropriate format to parse {}s, please define a format",
            $dtype,
        )
    };
}

#[macro_export]
macro_rules! polars_bail {
    ($($tt:tt)+) => {
        return Err($crate::polars_err!($($tt)+))
    };
}

#[macro_export]
macro_rules! polars_ensure {
    ($cond:expr, $($tt:tt)+) => {
        if !$cond {
            $crate::polars_bail!($($tt)+);
        }
    };
}

#[inline]
#[cold]
#[must_use]
pub fn to_compute_err(err: impl Display) -> PolarsError {
    PolarsError::ComputeError(err.to_string().into(), Arc::new(Backtrace::capture()))
}

#[macro_export]
macro_rules! feature_gated {
    ($feature:expr, $content:expr) => {{
        #[cfg(feature = $feature)]
        {
            $content
        }
        #[cfg(not(feature = $feature))]
        {
            panic!("activate '{}' feature", $feature)
        }
    }};
}

// Not public, referenced by macros only.
#[doc(hidden)]
pub mod __private {
    #[doc(hidden)]
    #[inline]
    #[cold]
    #[must_use]
    pub fn must_use(error: crate::PolarsError) -> crate::PolarsError {
        error
    }
}
