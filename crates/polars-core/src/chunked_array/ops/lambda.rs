use std::borrow::Cow;
use std::hash::{Hash, Hasher};
use std::str::from_utf8;

use num_traits::ToBytes;
use polars_error::{PolarsError, PolarsResult};
#[cfg(feature = "serde-lazy")]
use serde::{Deserialize, Serialize};

use super::flarion_funcs::{flarion_get_char_position, flarion_substring};
use super::DataType;
use crate::datatypes::{AnyValue, PolarsNumericType};
use crate::prelude::Array;
use crate::series::Series;
use crate::utils::dtypes_to_supertype;

#[derive(Clone, PartialEq, Debug)]
#[cfg_attr(feature = "serde-lazy", derive(Serialize, Deserialize))]
pub enum LambdaExpression {
    Null,
    Boolean(bool),
    Int8(i8),
    Int16(i16),
    Int32(i32),
    Int64(i64),
    Float32(f32),
    Float64(f64),
    BinaryBlob(Vec<u8>),
    StaticStr(Cow<'static, str>),
    Variable(usize),
    GreaterThan(Box<Self>, Box<Self>),
    LessThan(Box<Self>, Box<Self>),
    IfThenElse(Box<Self>, Box<Self>, Box<Self>),
    Length(Box<Self>),
    CaseWhen(Vec<(Self, Self)>, Box<Self>),
    Substring(Box<Self>, Box<Self>, Box<Self>),
    Instr(Box<Self>, Box<Self>),
    Add(Box<Self>, Box<Self>),
    IsNull(Box<Self>),
    EqualNullSafe(Box<Self>, Box<Self>)
}

impl Eq for LambdaExpression {}

impl Hash for LambdaExpression {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            LambdaExpression::Null => 0.hash(state),
            LambdaExpression::Boolean(v) => v.hash(state),
            LambdaExpression::Int8(v) => v.hash(state),
            LambdaExpression::Int16(v) => v.hash(state),
            LambdaExpression::Int32(v) => v.hash(state),
            LambdaExpression::Int64(v) => v.hash(state),
            LambdaExpression::Float32(v) => v.to_le_bytes().hash(state),
            LambdaExpression::Float64(v) => v.to_le_bytes().hash(state),
            LambdaExpression::BinaryBlob(v) => v.hash(state),
            LambdaExpression::StaticStr(v) => v.hash(state),
            LambdaExpression::Variable(v) => v.hash(state),
            LambdaExpression::GreaterThan(first, second) => {
                first.hash(state);
                second.hash(state);
            },
            LambdaExpression::LessThan(first, second) => {
                first.hash(state);
                second.hash(state);
            },
            LambdaExpression::IfThenElse(first, second, third) => {
                first.hash(state);
                second.hash(state);
                third.hash(state);
            },
            LambdaExpression::Length(v) => v.hash(state),
            LambdaExpression::CaseWhen(first, second) => {
                first.hash(state);
                second.hash(state);
            },
            LambdaExpression::Substring(first, second, third) => {
                first.hash(state);
                second.hash(state);
                third.hash(state);
            },
            LambdaExpression::Instr(first, second) => {
                first.hash(state);
                second.hash(state);
            },
            LambdaExpression::Add(first, second) => {
                first.hash(state);
                second.hash(state);
            },
            LambdaExpression::IsNull(v) => v.hash(state),
            LambdaExpression::EqualNullSafe(first, second) => {
                first.hash(state);
                second.hash(state);
            },
        }
    }
}

pub fn flarion_substring_anyvalue<'a>(
    s: AnyValue<'a>,
    from: AnyValue<'a>,
    len: AnyValue<'a>,
) -> AnyValue<'a> {
    match (s, from, len) {
        (AnyValue::String(s), AnyValue::Int32(from), AnyValue::Int32(len)) => {
            AnyValue::StringOwned(flarion_substring(s, from, len).into())
        },
        _ => AnyValue::Null,
    }
}

impl LambdaExpression {
    #[inline]
    pub(crate) fn eval_array<'a>(&'a self, args: &'a [&'a dyn Array]) -> AnyValue<'a> {
        match self {
            LambdaExpression::Null => AnyValue::Null,
            LambdaExpression::Boolean(v) => AnyValue::Boolean(*v),
            LambdaExpression::Int8(v) => AnyValue::Int8(*v),
            LambdaExpression::Int16(v) => AnyValue::Int16(*v),
            LambdaExpression::Int32(v) => AnyValue::Int32(*v),
            LambdaExpression::Int64(v) => AnyValue::Int64(*v),
            LambdaExpression::Float32(v) => AnyValue::Float32(*v),
            LambdaExpression::Float64(v) => AnyValue::Float64(*v),
            LambdaExpression::BinaryBlob(v) => AnyValue::Binary(v),
            LambdaExpression::StaticStr(v) => AnyValue::String(v),
            LambdaExpression::Variable(idx) => {
                let arr = args[*idx];
                let series = Series::from_arrow("".into(), arr.to_boxed())
                    .expect("could not convert array to series");
                AnyValue::List(series)
            },
            LambdaExpression::GreaterThan(left, right) => {
                // CURRENTLY ONLY SUPPORTS NESTED ARRAYS THAT HAVE A SINGLE MEMBER.
                // WITH MULTIPLE MEMBERS WE CAN DECIDE THE LOGIC IF WE ACTUALLY ENCOUNTER THESE CASES.
                let left_val = left.eval_array(args);
                let right_val = right.eval_array(args);

                match (left_val, right_val) {
                    (AnyValue::List(left_series), AnyValue::List(right_series)) => {
                        // Both are arrays - compare first values
                        if left_series.len() > 0 && right_series.len() > 0 {
                            let left_first = left_series
                                .get(0)
                                .expect("could not get first value from left array in comparison");
                            let right_first = right_series
                                .get(0)
                                .expect("could not get first value from right array in comparison");
                            left_first.gt(&right_first).into()
                        } else {
                            AnyValue::Boolean(false)
                        }
                    },
                    (AnyValue::List(left_series), right) => {
                        // Left is array, right is scalar
                        if left_series.len() > 0 {
                            let left_first = left_series.get(0)
                                .expect("could not get first value from array in left-array-scalar comparison");
                            left_first.gt(&right).into()
                        } else {
                            AnyValue::Boolean(false)
                        }
                    },
                    (left, AnyValue::List(right_series)) => {
                        // Left is scalar, right is an array
                        if right_series.len() > 0 {
                            let right_first = right_series.get(0)
                                .expect("could not get first value from array in scalar-right-array comparison");
                            left.gt(&right_first).into()
                        } else {
                            AnyValue::Boolean(false)
                        }
                    },
                    // Regular scalar comparison
                    (left, right) => left.gt(&right).into(),
                }
            },

            LambdaExpression::LessThan(left, right) => {
                // CURRENTLY ONLY SUPPORTS NESTED ARRAYS THAT HAVE A SINGLE MEMBER.
                // WITH MULTIPLE MEMBERS WE CAN DECIDE THE LOGIC IF WE ACTUALLY ENCOUNTER THESE CASES.
                let left_val = left.eval_array(args);
                let right_val = right.eval_array(args);

                match (left_val, right_val) {
                    (AnyValue::List(left_series), AnyValue::List(right_series)) => {
                        // Both are arrays - compare first values
                        if left_series.len() > 0 && right_series.len() > 0 {
                            let left_first = left_series
                                .get(0)
                                .expect("could not get first value from left array in comparison");
                            let right_first = right_series
                                .get(0)
                                .expect("could not get first value from right array in comparison");
                            left_first.lt(&right_first).into()
                        } else {
                            AnyValue::Boolean(false)
                        }
                    },
                    (AnyValue::List(left_series), right) => {
                        // Left is array, right is scalar
                        if left_series.len() > 0 {
                            let left_first = left_series.get(0)
                                .expect("could not get first value from array in left-array-scalar comparison");
                            left_first.lt(&right).into()
                        } else {
                            AnyValue::Boolean(false)
                        }
                    },
                    (left, AnyValue::List(right_series)) => {
                        // Left is scalar, right is an array
                        if right_series.len() > 0 {
                            let right_first = right_series.get(0)
                                .expect("could not get first value from array in scalar-right-array comparison");
                            left.lt(&right_first).into()
                        } else {
                            AnyValue::Boolean(false)
                        }
                    },
                    // Regular scalar comparison
                    (left, right) => left.lt(&right).into(),
                }
            },
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
            },
            LambdaExpression::Length(expr) => match expr.eval_array(args) {
                AnyValue::Null => AnyValue::Null,
                AnyValue::List(arr) => AnyValue::Int32(arr.len() as i32),
                _ => AnyValue::Int32(1),
            },
            LambdaExpression::CaseWhen(cases, otherwise) => {
                for (cond, value) in cases {
                    if unsafe {
                        match cond.eval_array(args) {
                            AnyValue::Boolean(v) => v,
                            _ => std::hint::unreachable_unchecked(), // tell the compiler it's unreachable
                        }
                    } {
                        return value.eval_array(args);
                    }
                }
                otherwise.eval_array(args)
            },
            LambdaExpression::Substring(s, from, len) => {
                let s = s.eval_array(args);
                let from = from.eval_array(args).cast(&DataType::Int32);
                let len = len.eval_array(args).cast(&DataType::Int32);
                flarion_substring_anyvalue(s, from, len)
            },
            LambdaExpression::Instr(s, pat) => {
                let s = s.eval_array(args);
                let pat = pat.eval_array(args);
                unsafe {
                    match (s, pat) {
                        (AnyValue::String(s), AnyValue::String(pat)) => {
                            AnyValue::Int32(flarion_get_char_position(s, pat))
                        },
                        _ => std::hint::unreachable_unchecked(), // tell the compiler it's unreachable
                    }
                }
            },
            LambdaExpression::Add(left, right) => {
                let left = left.eval_array(args);
                let right = right.eval_array(args);
                left.add(&right.cast(&left.dtype()))
            },
            LambdaExpression::IsNull(expr) => {
                if expr.eval_array(args).is_null() {
                    AnyValue::Boolean(true)
                } else {
                    AnyValue::Boolean(false)
                }
            },
            LambdaExpression::EqualNullSafe(left, right) => {
                let left = left.eval_array(args);
                let right = right.eval_array(args);
                if left.is_null() && right.is_null() {
                    AnyValue::Boolean(true)
                } else if left.is_null() || right.is_null() {
                    AnyValue::Boolean(false)
                } else {
                    AnyValue::Boolean(left.eq(&right))
                }
            }
        }
    }

    #[inline]
    pub(crate) fn eval_numeric<T: PolarsNumericType>(&self, args: &[&T::Native]) -> AnyValue {
        match self {
            LambdaExpression::Null => AnyValue::Null,
            LambdaExpression::Boolean(v) => AnyValue::Boolean(*v),
            LambdaExpression::Int8(v) => AnyValue::Int8(*v),
            LambdaExpression::Int16(v) => AnyValue::Int16(*v),
            LambdaExpression::Int32(v) => AnyValue::Int32(*v),
            LambdaExpression::Int64(v) => AnyValue::Int64(*v),
            LambdaExpression::Float32(v) => AnyValue::Float32(*v),
            LambdaExpression::Float64(v) => AnyValue::Float64(*v),
            LambdaExpression::BinaryBlob(v) => AnyValue::Binary(v),
            LambdaExpression::StaticStr(v) => AnyValue::String(v),
            LambdaExpression::Variable(idx) => (*args[*idx]).into(),
            LambdaExpression::GreaterThan(left, right) => {
                AnyValue::Boolean(left.eval_numeric::<T>(args) > right.eval_numeric::<T>(args))
            },
            LambdaExpression::LessThan(left, right) => {
                AnyValue::Boolean(left.eval_numeric::<T>(args) < right.eval_numeric::<T>(args))
            },
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
            },
            LambdaExpression::Length(expr) => match expr.eval_numeric::<T>(args) {
                AnyValue::Null => AnyValue::Null,
                _ => AnyValue::Int32(1),
            },
            LambdaExpression::CaseWhen(cases, otherwise) => {
                for (cond, value) in cases {
                    if unsafe {
                        match cond.eval_numeric::<T>(args) {
                            AnyValue::Boolean(v) => v,
                            _ => std::hint::unreachable_unchecked(), // tell the compiler it's unreachable
                        }
                    } {
                        return value.eval_numeric::<T>(args);
                    }
                }
                otherwise.eval_numeric::<T>(args)
            },
            LambdaExpression::Substring(s, from, len) => {
                let s = s.eval_numeric::<T>(args);
                let from = from.eval_numeric::<T>(args).cast(&DataType::Int32);
                let len = len.eval_numeric::<T>(args).cast(&DataType::Int32);
                flarion_substring_anyvalue(s, from, len)
            },
            LambdaExpression::Instr(s, pat) => {
                let s = s.eval_numeric::<T>(args);
                let pat = pat.eval_numeric::<T>(args);
                unsafe {
                    match (s, pat) {
                        (AnyValue::String(s), AnyValue::String(pat)) => {
                            AnyValue::Int32(flarion_get_char_position(s, pat))
                        },
                        _ => std::hint::unreachable_unchecked(), // tell the compiler it's unreachable
                    }
                }
            },
            LambdaExpression::Add(left, right) => {
                let left = left.eval_numeric::<T>(args);
                let right = right.eval_numeric::<T>(args);
                left.add(&right.cast(&left.dtype()))
            },
            LambdaExpression::IsNull(expr) => {
                if expr.eval_numeric::<T>(args).is_null() {
                    AnyValue::Boolean(true)
                } else {
                    AnyValue::Boolean(false)
                }
            },
            LambdaExpression::EqualNullSafe(left, right) => {
                let left = left.eval_numeric::<T>(args);
                let right = right.eval_numeric::<T>(args);
                if left.is_null() && right.is_null() {
                    AnyValue::Boolean(true)
                } else if left.is_null() || right.is_null() {
                    AnyValue::Boolean(false)
                } else {
                    AnyValue::Boolean(left.eq(&right))
                }
            }
        }
    }

    #[inline]
    pub(crate) fn eval_bool(&self, args: &[&bool]) -> AnyValue {
        match self {
            LambdaExpression::Null => AnyValue::Null,
            LambdaExpression::Boolean(v) => AnyValue::Boolean(*v),
            LambdaExpression::Int8(v) => AnyValue::Int8(*v),
            LambdaExpression::Int16(v) => AnyValue::Int16(*v),
            LambdaExpression::Int32(v) => AnyValue::Int32(*v),
            LambdaExpression::Int64(v) => AnyValue::Int64(*v),
            LambdaExpression::Float32(v) => AnyValue::Float32(*v),
            LambdaExpression::Float64(v) => AnyValue::Float64(*v),
            LambdaExpression::BinaryBlob(v) => AnyValue::Binary(v),
            LambdaExpression::StaticStr(v) => AnyValue::String(v),
            LambdaExpression::Variable(idx) => (*args[*idx]).into(),
            LambdaExpression::GreaterThan(left, right) => {
                AnyValue::Boolean(left.eval_bool(args) > right.eval_bool(args))
            },
            LambdaExpression::LessThan(left, right) => {
                AnyValue::Boolean(left.eval_bool(args) < right.eval_bool(args))
            },
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
            },
            LambdaExpression::Length(expr) => match expr.eval_bool(args) {
                AnyValue::Null => AnyValue::Null,
                _ => AnyValue::Int32(1),
            },
            LambdaExpression::CaseWhen(cases, otherwise) => {
                for (cond, value) in cases {
                    if unsafe {
                        match cond.eval_bool(args) {
                            AnyValue::Boolean(v) => v,
                            _ => std::hint::unreachable_unchecked(), // tell the compiler it's unreachable
                        }
                    } {
                        return value.eval_bool(args);
                    }
                }
                otherwise.eval_bool(args)
            },
            LambdaExpression::Substring(s, from, len) => {
                let s = s.eval_bool(args);
                let from = from.eval_bool(args).cast(&DataType::Int32);
                let len = len.eval_bool(args).cast(&DataType::Int32);
                flarion_substring_anyvalue(s, from, len)
            },
            LambdaExpression::Instr(s, pat) => {
                let s = s.eval_bool(args);
                let pat = pat.eval_bool(args);
                unsafe {
                    match (s, pat) {
                        (AnyValue::String(s), AnyValue::String(pat)) => {
                            AnyValue::Int32(flarion_get_char_position(s, pat))
                        },
                        _ => std::hint::unreachable_unchecked(), // tell the compiler it's unreachable
                    }
                }
            },
            LambdaExpression::Add(left, right) => {
                let left = left.eval_bool(args);
                let right = right.eval_bool(args);
                left.add(&right.cast(&left.dtype()))
            },
            LambdaExpression::IsNull(expr) => {
                if expr.eval_bool(args).is_null() {
                    AnyValue::Boolean(true)
                } else {
                    AnyValue::Boolean(false)
                }
            },
            LambdaExpression::EqualNullSafe(left, right) => {
                let left = left.eval_bool(args);
                let right = right.eval_bool(args);
                if left.is_null() && right.is_null() {
                    AnyValue::Boolean(true)
                } else if left.is_null() || right.is_null() {
                    AnyValue::Boolean(false)
                } else {
                    AnyValue::Boolean(left.eq(&right))
                }
            }
        }
    }

    #[inline]
    pub(crate) fn eval_slice<'a>(&'a self, args: &'a [&'a [u8]]) -> AnyValue<'a> {
        match self {
            LambdaExpression::Null => AnyValue::Null,
            LambdaExpression::Boolean(v) => AnyValue::Boolean(*v),
            LambdaExpression::Int8(v) => AnyValue::Int8(*v),
            LambdaExpression::Int16(v) => AnyValue::Int16(*v),
            LambdaExpression::Int32(v) => AnyValue::Int32(*v),
            LambdaExpression::Int64(v) => AnyValue::Int64(*v),
            LambdaExpression::Float32(v) => AnyValue::Float32(*v),
            LambdaExpression::Float64(v) => AnyValue::Float64(*v),
            LambdaExpression::BinaryBlob(v) => AnyValue::Binary(v),
            LambdaExpression::StaticStr(v) => AnyValue::String(v),
            LambdaExpression::Variable(idx) => AnyValue::Binary(args[*idx]),
            LambdaExpression::GreaterThan(left, right) => {
                left.eval_slice(args).gt(&right.eval_slice(args)).into()
            },
            LambdaExpression::LessThan(left, right) => {
                left.eval_slice(args).lt(&right.eval_slice(args)).into()
            },
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
            },
            LambdaExpression::Length(expr) => match expr.eval_slice(args) {
                AnyValue::Null => AnyValue::Null,
                AnyValue::Binary(bytes) => {
                    // case we have a binary slice, convert to utf8 as spark expects
                    let new_str = from_utf8(bytes).unwrap();
                    AnyValue::Int32(new_str.chars().count() as i32)
                },
                AnyValue::String(s) => AnyValue::Int32(s.chars().count() as i32),
                AnyValue::List(arr) => AnyValue::Int32(arr.len() as i32),
                _ => AnyValue::Int32(1),
            },
            LambdaExpression::CaseWhen(cases, otherwise) => {
                for (cond, value) in cases {
                    if unsafe {
                        match cond.eval_slice(args) {
                            AnyValue::Boolean(v) => v,
                            _ => std::hint::unreachable_unchecked(), // tell the compiler it's unreachable
                        }
                    } {
                        return value.eval_slice(args);
                    }
                }
                otherwise.eval_slice(args)
            },
            LambdaExpression::Substring(s, from, len) => {
                let s = s.eval_slice(args);
                let from = from.eval_slice(args).cast(&DataType::Int32);
                let len = len.eval_slice(args).cast(&DataType::Int32);
                flarion_substring_anyvalue(s, from, len)
            },
            LambdaExpression::Instr(s, pat) => {
                let s = s.eval_slice(args);
                let pat = pat.eval_slice(args);
                unsafe {
                    match (s, pat) {
                        (AnyValue::String(s), AnyValue::String(pat)) => {
                            AnyValue::Int32(flarion_get_char_position(s, pat))
                        },
                        (AnyValue::Binary(s), AnyValue::String(oat)) => {
                            AnyValue::Int32(flarion_get_char_position(from_utf8(s).unwrap(), oat))
                        },
                        _ => std::hint::unreachable_unchecked(), // tell the compiler it's unreachable
                    }
                }
            },
            LambdaExpression::Add(left, right) => {
                let left = left.eval_slice(args);
                let right = right.eval_slice(args);
                left.add(&right.cast(&left.dtype()))
            },
            LambdaExpression::IsNull(expr) => {
                if expr.eval_slice(args).is_null() {
                    AnyValue::Boolean(true)
                } else {
                    AnyValue::Boolean(false)
                }
            },
            LambdaExpression::EqualNullSafe(left, right) => {
                let left = left.eval_slice(args);
                let right = right.eval_slice(args);
                if left.is_null() && right.is_null() {
                    AnyValue::Boolean(true)
                } else if left.is_null() || right.is_null() {
                    AnyValue::Boolean(false)
                } else {
                    AnyValue::Boolean(left.eq(&right))
                }
            }
        }
    }

    pub(crate) fn eval_any<'a>(&'a self, args: &[AnyValue<'a>]) -> AnyValue<'a> {
        match self {
            LambdaExpression::Null => AnyValue::Null,
            LambdaExpression::Boolean(v) => AnyValue::Boolean(*v),
            LambdaExpression::Int8(v) => AnyValue::Int8(*v),
            LambdaExpression::Int16(v) => AnyValue::Int16(*v),
            LambdaExpression::Int32(v) => AnyValue::Int32(*v),
            LambdaExpression::Int64(v) => AnyValue::Int64(*v),
            LambdaExpression::Float32(v) => AnyValue::Float32(*v),
            LambdaExpression::Float64(v) => AnyValue::Float64(*v),
            LambdaExpression::BinaryBlob(v) => AnyValue::Binary(v),
            LambdaExpression::StaticStr(v) => AnyValue::String(v),
            LambdaExpression::Variable(idx) => args[*idx].clone(),
            LambdaExpression::GreaterThan(left, right) => {
                left.eval_any(args).gt(&right.eval_any(args)).into()
            },
            LambdaExpression::LessThan(left, right) => {
                left.eval_any(args).lt(&right.eval_any(args)).into()
            },
            LambdaExpression::IfThenElse(cond, truthy, falsy) => {
                if unsafe {
                    match cond.eval_any(args) {
                        AnyValue::Boolean(v) => v,
                        _ => std::hint::unreachable_unchecked(), // tell the compiler it's unreachable
                    }
                } {
                    truthy.eval_any(args)
                } else {
                    falsy.eval_any(args)
                }
            },
            LambdaExpression::Length(expr) => match expr.eval_any(args) {
                AnyValue::Null => AnyValue::Null,
                AnyValue::Binary(bytes) => AnyValue::Int32(bytes.len() as i32),
                AnyValue::String(s) => AnyValue::Int32(s.chars().count() as i32),
                AnyValue::List(arr) => AnyValue::Int32(arr.len() as i32),
                _ => AnyValue::Int32(1),
            },
            LambdaExpression::CaseWhen(cases, otherwise) => {
                for (cond, value) in cases {
                    if unsafe {
                        match cond.eval_any(args) {
                            AnyValue::Boolean(v) => v,
                            _ => std::hint::unreachable_unchecked(), // tell the compiler it's unreachable
                        }
                    } {
                        return value.eval_any(args);
                    }
                }
                otherwise.eval_any(args)
            },
            LambdaExpression::Substring(s, from, len) => {
                let s = s.eval_any(args);
                let from = from.eval_any(args).cast(&DataType::Int32);
                let len = len.eval_any(args).cast(&DataType::Int32);
                flarion_substring_anyvalue(s, from, len)
            },
            LambdaExpression::Instr(s, pat) => {
                let s = s.eval_any(args);
                let pat = pat.eval_any(args);
                unsafe {
                    match (s, pat) {
                        (AnyValue::String(s), AnyValue::String(pat)) => {
                            AnyValue::Int32(flarion_get_char_position(s, pat))
                        },
                        _ => std::hint::unreachable_unchecked(), // tell the compiler it's unreachable
                    }
                }
            },
            LambdaExpression::Add(left, right) => {
                let left = left.eval_any(args);
                let right = right.eval_any(args);
                left.add(&right.cast(&left.dtype()))
            },
            LambdaExpression::IsNull(expr) => {
                if expr.eval_any(args).is_null() {
                    AnyValue::Boolean(true)
                } else {
                    AnyValue::Boolean(false)
                }
            },
            LambdaExpression::EqualNullSafe(left, right) => {
                let left = left.eval_any(args);
                let right = right.eval_any(args);
                if left.is_null() && right.is_null() {
                    AnyValue::Boolean(true)
                } else if left.is_null() || right.is_null() {
                    AnyValue::Boolean(false)
                } else {
                    AnyValue::Boolean(left.eq(&right))
                }
            }
        }
    }

    // None encode that type same as input value
    pub fn return_type(&self, input_type: &DataType) -> Result<DataType, PolarsError> {
        // dtypes_to_supertype
        Ok(match self {
            LambdaExpression::Null => DataType::Null,
            LambdaExpression::Boolean(_) => DataType::Boolean,
            LambdaExpression::Int8(_) => DataType::Int8,
            LambdaExpression::Int16(_) => DataType::Int16,
            LambdaExpression::Int32(_) => DataType::Int32,
            LambdaExpression::Int64(_) => DataType::Int64,
            LambdaExpression::Float32(_) => DataType::Float32,
            LambdaExpression::Float64(_) => DataType::Float64,
            LambdaExpression::BinaryBlob(_) => DataType::Binary,
            LambdaExpression::StaticStr(_) => DataType::String,
            LambdaExpression::Variable(_) => input_type.clone(),
            LambdaExpression::GreaterThan(_, _) => DataType::Boolean,
            LambdaExpression::LessThan(_, _) => DataType::Boolean,
            LambdaExpression::IfThenElse(_, then, els) => dtypes_to_supertype([
                &then.return_type(input_type)?,
                &els.return_type(input_type)?,
            ])?,
            LambdaExpression::Length(_) => DataType::Int32,
            LambdaExpression::CaseWhen(cases, otherwise) => {
                let child_types = cases
                    .iter()
                    .map(|pair| &pair.1)
                    .chain([otherwise.as_ref()])
                    .map(|expr| expr.return_type(input_type))
                    .collect::<PolarsResult<Vec<_>>>()?;
                dtypes_to_supertype(&child_types)?
            },
            LambdaExpression::Substring(s, _, _) => s.return_type(input_type)?, // substring is used for string and byte arrays
            LambdaExpression::Instr(_, _) => DataType::Int32,
            LambdaExpression::Add(left, right) => dtypes_to_supertype([
                &left.return_type(input_type)?,
                &right.return_type(input_type)?,
            ])?,
            LambdaExpression::IsNull(_) => DataType::Boolean,
            LambdaExpression::EqualNullSafe(_, _) => DataType::Boolean,
        })
    }
}
