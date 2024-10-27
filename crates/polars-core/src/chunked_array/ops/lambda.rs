use std::borrow::Cow;
#[cfg(feature = "serde-lazy")]
use serde::{Deserialize, Serialize};
use crate::datatypes::{AnyValue, PolarsNumericType};
use crate::prelude::Array;
use crate::series::Series;

#[derive(Clone, Eq, PartialEq, Debug, Hash)]
#[cfg_attr(feature = "serde-lazy", derive(Serialize, Deserialize))]
pub enum LambdaExpression {
    Null,
    Boolean(bool),
    Int32(i32),
    Int64(i64),
    StaticStr(Cow<'static, str>),
    Variable(usize),
    GreaterThan(Box<Self>, Box<Self>),
    LessThan(Box<Self>, Box<Self>),
    IfThenElse(Box<Self>, Box<Self>, Box<Self>),
    Length(Box<Self>),
    CaseWhen(Vec<(Self, Self)>, Box<Self>)
}

impl LambdaExpression {
    #[inline]
    pub(crate) fn eval_array<'a>(&'a self, args: &'a[&'a dyn Array]) -> AnyValue<'a> {
        match self {
            LambdaExpression::Null => {
                AnyValue::Null
            }
            LambdaExpression::Boolean(v) => {
                AnyValue::Boolean(*v)
            }
            LambdaExpression::Int32(v) => {
                AnyValue::Int32(*v)
            }
            LambdaExpression::Int64(v) => {
                AnyValue::Int64(*v)
            }
            LambdaExpression::StaticStr(v) => {
                AnyValue::String(v)
            }
            LambdaExpression::Variable(idx) => {
                let arr = args[*idx];
                let series = Series::from_arrow("".into(), arr.to_boxed())
                    .expect("could not convert array to series");
                AnyValue::List(series)
            }
            LambdaExpression::GreaterThan(left, right) => {
                left.eval_array(args).gt(&right.eval_array(args)).into()
            }
            LambdaExpression::LessThan(left, right) => {
                left.eval_array(args).lt(&right.eval_array(args)).into()
            }
            LambdaExpression::IfThenElse(cond, truthy, falsy) => {
                if unsafe {
                    match cond.eval_array(args) {
                        AnyValue::Boolean(v) => v,
                        _ => std::hint::unreachable_unchecked(), // tell the compiler it's unreachable
                    }
                } {
                    truthy.eval_array(args)
                } else {
                    falsy.eval_array(args)
                }
            }
            LambdaExpression::Length(expr) => {
                match expr.eval_array(args) {
                    AnyValue::Null => AnyValue::Null,
                    AnyValue::List(arr) => AnyValue::Int32(arr.len() as i32),
                    _ => AnyValue::Int32(1)
                }
            }
            LambdaExpression::CaseWhen(cases, otherwise) => {
                for (cond, value) in cases {
                    if unsafe {
                        match cond.eval_array(args) {
                            AnyValue::Boolean(v) => v,
                            _ => std::hint::unreachable_unchecked(), // tell the compiler it's unreachable
                        }
                    } {
                        return value.eval_array(args)
                    }
                }
                otherwise.eval_array(args)
            }
        }
    }

    #[inline]
    pub(crate) fn eval_numeric<T: PolarsNumericType>(&self, args: &[&T::Native]) -> AnyValue {
        match self {
            LambdaExpression::Null => {
                AnyValue::Null
            }
            LambdaExpression::Boolean(v) => {
                AnyValue::Boolean(*v)
            }
            LambdaExpression::Int32(v) => {
                AnyValue::Int32(*v)
            }
            LambdaExpression::Int64(v) => {
                AnyValue::Int64(*v)
            }
            LambdaExpression::StaticStr(v) => {
                AnyValue::String(v)
            }
            LambdaExpression::Variable(idx) => {
                (*args[*idx]).into()
            }
            LambdaExpression::GreaterThan(left, right) => {
                AnyValue::Boolean(left.eval_numeric::<T>(args) > right.eval_numeric::<T>(args))
            }
            LambdaExpression::LessThan(left, right) => {
                AnyValue::Boolean(left.eval_numeric::<T>(args) < right.eval_numeric::<T>(args))
            }
            LambdaExpression::IfThenElse(cond, truthy, falsy) => {
                if unsafe {
                    match cond.eval_numeric::<T>(args) {
                        AnyValue::Boolean(v) => v,
                        _ => std::hint::unreachable_unchecked(), // tell the compiler it's unreachable
                    }
                } {
                    truthy.eval_numeric::<T>(args)
                } else {
                    falsy.eval_numeric::<T>(args)
                }
            }
            LambdaExpression::Length(expr) => {
                match expr.eval_numeric::<T>(args) {
                    AnyValue::Null => AnyValue::Null,
                    _ => AnyValue::Int32(1)
                }
            }
            LambdaExpression::CaseWhen(cases, otherwise) => {
                for (cond, value) in cases {
                    if unsafe {
                        match cond.eval_numeric::<T>(args) {
                            AnyValue::Boolean(v) => v,
                            _ => std::hint::unreachable_unchecked(), // tell the compiler it's unreachable
                        }
                    } {
                        return value.eval_numeric::<T>(args)
                    }
                }
                otherwise.eval_numeric::<T>(args)
            }
            LambdaExpression::Length(expr) => {
                match expr.eval_numeric::<T>(args) {
                    AnyValue::Null => AnyValue::Null,
                    _ => AnyValue::Int64(1)
                }
            }
        }
    }

    #[inline]
    pub(crate) fn eval_bool(&self, args: &[&bool]) -> AnyValue {
        match self {
            LambdaExpression::Null => {
                AnyValue::Null
            }
            LambdaExpression::Boolean(v) => {
                AnyValue::Boolean(*v)
            }
            LambdaExpression::Int32(v) => {
                AnyValue::Int32(*v)
            }
            LambdaExpression::Int64(v) => {
                AnyValue::Int64(*v)
            }
            LambdaExpression::StaticStr(v) => {
                AnyValue::String(v)
            }
            LambdaExpression::Variable(idx) => {
                (*args[*idx]).into()
            }
            LambdaExpression::GreaterThan(left, right) => {
                AnyValue::Boolean(left.eval_bool(args) > right.eval_bool(args))
            }
            LambdaExpression::LessThan(left, right) => {
                AnyValue::Boolean(left.eval_bool(args) < right.eval_bool(args))
            }
            LambdaExpression::IfThenElse(cond, truthy, falsy) => {
                if unsafe {
                    match cond.eval_bool(args) {
                        AnyValue::Boolean(v) => v,
                        _ => std::hint::unreachable_unchecked(), // tell the compiler it's unreachable
                    }
                } {
                    truthy.eval_bool(args)
                } else {
                    falsy.eval_bool(args)
                }
            }
            LambdaExpression::Length(expr) => {
                match expr.eval_bool(args) {
                    AnyValue::Null => AnyValue::Null,
                    _ => AnyValue::Int32(1)
                }
            }
            LambdaExpression::CaseWhen(cases, otherwise) => {
                for (cond, value) in cases {
                    if unsafe {
                        match cond.eval_bool(args) {
                            AnyValue::Boolean(v) => v,
                            _ => std::hint::unreachable_unchecked(), // tell the compiler it's unreachable
                        }
                    } {
                        return value.eval_bool(args)
                    }
                }
                otherwise.eval_bool(args)
            }
            LambdaExpression::Length(expr) => {
                match expr.eval_bool(args) {
                    AnyValue::Null => AnyValue::Null,
                    _ => AnyValue::Int64(1)
                }
            }
        }
    }

    #[inline]
    pub(crate) fn eval_slice<'a>(&'a self, args: &'a[&'a[u8]]) -> AnyValue<'a> {
        match self {
            LambdaExpression::Null => {
                AnyValue::Null
            }
            LambdaExpression::Boolean(v) => {
                AnyValue::Boolean(*v)
            }
            LambdaExpression::Int32(v) => {
                AnyValue::Int32(*v)
            }
            LambdaExpression::Int64(v) => {
                AnyValue::Int64(*v)
            }
            LambdaExpression::StaticStr(v) => {
                AnyValue::String(v)
            }
            LambdaExpression::Variable(idx) => {
                AnyValue::Binary(args[*idx])
            }
            LambdaExpression::GreaterThan(left, right) => {
                left.eval_slice(args).gt(&right.eval_slice(args)).into()
            }
            LambdaExpression::LessThan(left, right) => {
                left.eval_slice(args).lt(&right.eval_slice(args)).into()
            }
            LambdaExpression::IfThenElse(cond, truthy, falsy) => {
                if unsafe {
                    match cond.eval_slice(args) {
                        AnyValue::Boolean(v) => v,
                        _ => std::hint::unreachable_unchecked(), // tell the compiler it's unreachable
                    }
                } {
                    truthy.eval_slice(args)
                } else {
                    falsy.eval_slice(args)
                }
            }
            LambdaExpression::Length(expr) => {
                match expr.eval_slice(args) {
                    AnyValue::Null => AnyValue::Null,
                    AnyValue::Binary(bytes) => AnyValue::Int32(bytes.len() as i32),
                    AnyValue::String(s) => AnyValue::Int32(s.len() as i32),
                    AnyValue::List(arr) => AnyValue::Int32(arr.len() as i32),
                    _ => AnyValue::Int32(1)
                }
            }
            LambdaExpression::CaseWhen(cases, otherwise) => {
                for (cond, value) in cases {
                    if unsafe {
                        match cond.eval_slice(args) {
                            AnyValue::Boolean(v) => v,
                            _ => std::hint::unreachable_unchecked(), // tell the compiler it's unreachable
                        }
                    } {
                        return value.eval_slice(args)
                    }
                }
                otherwise.eval_slice(args)
            }
        }
    }
}
