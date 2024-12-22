use std::borrow::Cow;
use std::hash::{Hash, Hasher};
use std::iter;
use std::ops::Add;
use std::str::from_utf8;

use arrow::array::ViewType;
use num_traits::ToBytes;
use polars_error::{PolarsError, PolarsResult};
use polars_utils::itertools::Itertools;
use polars_utils::pl_str::PlSmallStr;
#[cfg(feature = "serde-lazy")]
use serde::{Deserialize, Serialize};

use super::flarion_funcs::{flarion_get_char_position, flarion_substring};
use super::DataType;
use crate::chunked_array::ops::ChunkCompare;
use crate::datatypes::{AnyValue, PolarsNumericType};
use crate::prelude::flarion_funcs::{flarion_instr_helper, flarion_slice_helper};
use crate::prelude::{
    Array, BinaryChunked, BooleanChunked, CastOptions, ChunkFull, IntoSeries, NamedFromOwned,
};
use crate::series::Series;
use crate::utils::dtypes_to_supertype;

#[derive(Clone, PartialEq, Debug)]
#[cfg_attr(feature = "serde-lazy", derive(Serialize, Deserialize))]
pub enum LambdaExpression {
    Null,
    Boolean(bool),
    #[cfg(feature = "dtype-i8")]
    Int8(i8),
    #[cfg(feature = "dtype-i16")]
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
    #[cfg(feature = "zip_with")]
    CaseWhen(Vec<(Self, Self)>, Box<Self>),
    Substring(Box<Self>, Box<Self>, Box<Self>),
    Instr(Box<Self>, Box<Self>),
    Add(Box<Self>, Box<Self>),
    IsNull(Box<Self>),
    EqualNullSafe(Box<Self>, Box<Self>),
    Cast(Box<Self>, DataType),
}

impl Eq for LambdaExpression {}

impl Hash for LambdaExpression {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            LambdaExpression::Null => 0.hash(state),
            LambdaExpression::Boolean(v) => v.hash(state),
            #[cfg(feature = "dtype-i8")]
            LambdaExpression::Int8(v) => v.hash(state),
            #[cfg(feature = "dtype-i16")]
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
            #[cfg(feature = "zip_with")]
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
            LambdaExpression::Cast(v, data_type) => {
                v.hash(state);
                data_type.hash(state);
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
        (AnyValue::Binary(s), AnyValue::Int32(from), AnyValue::Int32(len)) => {
            AnyValue::BinaryOwned(
                flarion_substring(from_utf8(s).unwrap(), from, len)
                    .to_bytes()
                    .into(),
            )
        },
        _ => unreachable!(),
    }
}

impl LambdaExpression {
    pub fn is_literal(&self) -> bool {
        matches!(
            self,
            LambdaExpression::Null
                | LambdaExpression::Boolean(_)
                | LambdaExpression::Int32(_)
                | LambdaExpression::Int64(_)
                | LambdaExpression::Float32(_)
                | LambdaExpression::Float64(_)
                | LambdaExpression::BinaryBlob(_)
                | LambdaExpression::StaticStr(_)
        )
    }

    pub fn eval(&self, s: &Series, is_root: bool) -> PolarsResult<Series> {
        let repeat_count = if is_root { s.len() } else { 1 }; // Handle broadcast lambdas expressions if root
        match self {
            LambdaExpression::Null => Ok(Series::new_null(PlSmallStr::EMPTY, 1)),
            LambdaExpression::Boolean(v) => Ok(Series::from_iter(iter::repeat_n(v, repeat_count))),
            #[cfg(feature = "dtype-i8")]
            LambdaExpression::Int8(v) => Ok(Series::from_iter(iter::repeat_n(v, repeat_count))),
            #[cfg(feature = "dtype-i16")]
            LambdaExpression::Int16(v) => Ok(Series::from_iter(iter::repeat_n(v, repeat_count))),
            LambdaExpression::Int32(v) => Ok(Series::from_iter(iter::repeat_n(v, repeat_count))),
            LambdaExpression::Int64(v) => Ok(Series::from_iter(iter::repeat_n(v, repeat_count))),
            LambdaExpression::Float32(v) => Ok(Series::from_iter(iter::repeat_n(v, repeat_count))),
            LambdaExpression::Float64(v) => Ok(Series::from_iter(iter::repeat_n(v, repeat_count))),
            LambdaExpression::BinaryBlob(v) => Ok(BinaryChunked::from_iter(iter::repeat_n(
                v.as_slice(),
                repeat_count,
            ))
            .into_series()),
            LambdaExpression::StaticStr(v) => {
                Ok(Series::from_iter(iter::repeat_n(v.as_ref(), repeat_count)))
            },
            LambdaExpression::Variable(_) => Ok(s.clone()),
            LambdaExpression::GreaterThan(left, right) => {
                let left = left.eval(s, false)?;
                let right = right.eval(s, false)?;
                Ok(left.gt(&right)?.into_series())
            },
            LambdaExpression::LessThan(left, right) => {
                let left = left.eval(s, false)?;
                let right = right.eval(s, false)?;
                Ok(left.lt(&right)?.into_series())
            },
            LambdaExpression::IfThenElse(pred, value, otherwise) => {
                let pred = pred.eval(s, false)?;
                let value = value.eval(s, false)?;
                let otherwise = otherwise.eval(s, false)?;

                // I think this is the only place where its valid to use AnyValues, since both series can be absolutely anything
                let vals = pred
                    .bool()?
                    .iter()
                    .zip(value.iter().zip(otherwise.iter()))
                    .map(|(pred, (left, right))| {
                        // Predicates are not nullable
                        if pred.unwrap_or_default() {
                            left
                        } else {
                            right
                        }
                    })
                    .collect_vec();
                Series::from_any_values(PlSmallStr::EMPTY, &vals, true)
            },
            LambdaExpression::Length(child) => {
                let s = child.eval(s, false)?;
                Ok(match s.dtype() {
                    DataType::Null => Series::new_null(PlSmallStr::EMPTY, s.len()),
                    DataType::Binary => Series::from_iter(
                        s.binary()?
                            .iter()
                            .map(|opt_v: Option<&[u8]>| opt_v.map(|v| v.len() as i32)),
                    ),
                    DataType::String => Series::from_iter(
                        s.str()?
                            .iter()
                            .map(|opt_v| opt_v.map(|v| v.chars().count() as i32)),
                    ),
                    DataType::List(_) => {
                        let ca = s.as_list();
                        let mut lengths = Vec::with_capacity(ca.len());
                        ca.downcast_iter().for_each(|arr| {
                            let offsets = arr.offsets().as_slice();
                            let mut last = offsets[0];
                            for o in &offsets[1..] {
                                lengths.push((*o - last) as i32);
                                last = *o;
                            }
                        });
                        Series::from_vec(PlSmallStr::EMPTY, lengths)
                    },
                    _ => Series::from_iter(iter::repeat_n(1i32, s.len())),
                })
            },
            #[cfg(feature = "zip_with")]
            LambdaExpression::CaseWhen(branches, otherwise) => {
                let predicates: Vec<Series> = branches
                    .iter()
                    .map(|(pred, _)| pred.eval(s, false))
                    .collect::<PolarsResult<_>>()?;
                let branches: Vec<(&BooleanChunked, Series)> = predicates
                    .iter()
                    .zip(branches)
                    .map(|(predicates, (_, thens))| Ok((predicates.bool()?, thens.eval(s, false)?)))
                    .collect::<PolarsResult<_>>()?;
                let otherwise: Series = otherwise.eval(s, false)?;

                // Start with all false mask of appropriate length
                let mut final_mask =
                    BooleanChunked::full(PlSmallStr::EMPTY, false, otherwise.len());
                let mut result = otherwise;

                // Iterate through branches in order
                for (pred_mask, then_series) in branches {
                    // Only apply values where:
                    // 1. Current predicate is true
                    // 2. No previous predicate was true (not in final_mask)
                    let new_mask = pred_mask & &(!&final_mask);

                    // Where new_mask is true, take values from then_series
                    // Where new_mask is false, keep values from result
                    result = then_series.zip_with(&new_mask, &result)?;

                    // Update final mask to include this predicate
                    final_mask = &final_mask | pred_mask;
                }

                Ok(result)
            },
            LambdaExpression::Substring(child, start, length) => {
                let child = child.eval(s, false)?;
                let start = start.eval(s, false)?;
                let length = length.eval(s, false)?;
                flarion_slice_helper(child.str()?, &start, &length)
            },
            LambdaExpression::Instr(left, right) => {
                let left = left.eval(s, false)?;
                let right = right.eval(s, false)?;

                flarion_instr_helper(left.str()?, &right)
            },
            LambdaExpression::Add(left, right) => {
                let left = left.eval(s, false)?;
                let right = right.eval(s, false)?;
                left.add(right)
            },
            LambdaExpression::IsNull(child) => {
                let child = child.eval(s, false)?;
                Ok(child.is_null().into_series())
            },
            LambdaExpression::EqualNullSafe(left, right) => {
                let left = left.eval(s, false)?;
                let right = right.eval(s, false)?;
                left.equal_missing(&right).map(IntoSeries::into_series)
            },
            LambdaExpression::Cast(child, dtype) => {
                let s = child.eval(s, false)?;
                s.cast_with_options(dtype, CastOptions::Overflowing)
            },
        }
    }
}

impl LambdaExpression {
    #[inline]
    pub(crate) fn eval_array<'a>(&'a self, args: &'a [&'a dyn Array]) -> AnyValue<'a> {
        match self {
            LambdaExpression::Null => AnyValue::Null,
            LambdaExpression::Boolean(v) => AnyValue::Boolean(*v),
            #[cfg(feature = "dtype-i8")]
            LambdaExpression::Int8(v) => AnyValue::Int8(*v),
            #[cfg(feature = "dtype-i16")]
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
            #[cfg(feature = "zip_with")]
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
            },
            LambdaExpression::Cast(expr, data_type) => expr.eval_array(args).cast(data_type),
        }
    }

    #[inline]
    pub(crate) fn eval_numeric<T: PolarsNumericType>(&self, args: &[&T::Native]) -> AnyValue {
        match self {
            LambdaExpression::Null => AnyValue::Null,
            LambdaExpression::Boolean(v) => AnyValue::Boolean(*v),
            #[cfg(feature = "dtype-i8")]
            LambdaExpression::Int8(v) => AnyValue::Int8(*v),
            #[cfg(feature = "dtype-i16")]
            LambdaExpression::Int16(v) => AnyValue::Int16(*v),
            LambdaExpression::Int32(v) => AnyValue::Int32(*v),
            LambdaExpression::Int64(v) => AnyValue::Int64(*v),
            LambdaExpression::Float32(v) => AnyValue::Float32(*v),
            LambdaExpression::Float64(v) => AnyValue::Float64(*v),
            LambdaExpression::BinaryBlob(v) => AnyValue::Binary(v),
            LambdaExpression::StaticStr(v) => AnyValue::String(v),
            LambdaExpression::Variable(idx) => (*args[*idx]).into(),
            LambdaExpression::GreaterThan(left, right) => {
                let left = left.eval_numeric::<T>(args);
                let right = right.eval_numeric::<T>(args);

                // Special case for Int8/Int16 and Int32 when using Array[Byte]/Array[Short]
                // For example: Array[Byte](77) < 5
                match (left, right) {
                    (AnyValue::Int8(left), AnyValue::Int32(right)) => {
                        AnyValue::Boolean((left as i32) > right)
                    },
                    (AnyValue::Int16(left), AnyValue::Int32(right)) => {
                        AnyValue::Boolean((left as i32) > right)
                    },
                    (left, right) => AnyValue::Boolean(left > right),
                }
            },
            LambdaExpression::LessThan(left, right) => {
                let left = left.eval_numeric::<T>(args);
                let right = right.eval_numeric::<T>(args);

                // Special case for Int8/Int16 and Int32 when using Array[Byte]/Array[Short]
                // For example: Array[Byte](77) < 5
                match (&left, &right) {
                    #[cfg(feature = "dtype-i8")]
                    (AnyValue::Int8(left), AnyValue::Int32(right)) => {
                        AnyValue::Boolean((*left as i32) < *right)
                    },
                    #[cfg(feature = "dtype-i16")]
                    (AnyValue::Int16(left), AnyValue::Int32(right)) => {
                        AnyValue::Boolean((*left as i32) < *right)
                    },
                    _ => AnyValue::Boolean(left < right),
                }
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
            #[cfg(feature = "zip_with")]
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
            },
            LambdaExpression::Cast(expr, data_type) => expr.eval_numeric::<T>(args).cast(data_type),
        }
    }

    #[inline]
    pub(crate) fn eval_bool(&self, args: &[&bool]) -> AnyValue {
        match self {
            LambdaExpression::Null => AnyValue::Null,
            LambdaExpression::Boolean(v) => AnyValue::Boolean(*v),
            #[cfg(feature = "dtype-i8")]
            LambdaExpression::Int8(v) => AnyValue::Int8(*v),
            #[cfg(feature = "dtype-i16")]
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
            #[cfg(feature = "zip_with")]
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
            },
            LambdaExpression::Cast(expr, data_type) => expr.eval_bool(args).cast(data_type),
        }
    }

    #[inline]
    pub(crate) fn eval_slice<'a>(&'a self, args: &'a [&'a [u8]]) -> AnyValue<'a> {
        match self {
            LambdaExpression::Null => AnyValue::Null,
            LambdaExpression::Boolean(v) => AnyValue::Boolean(*v),
            #[cfg(feature = "dtype-i8")]
            LambdaExpression::Int8(v) => AnyValue::Int8(*v),
            #[cfg(feature = "dtype-i16")]
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
            LambdaExpression::Length(expr) => {
                match expr.eval_slice(args) {
                    AnyValue::Null => AnyValue::Null,
                    AnyValue::Binary(bytes) => {
                        // case we have a binary slice, try to convert to utf8 as spark expects, else return length of bytes
                        match from_utf8(bytes) {
                            Ok(new_str) => AnyValue::Int32(new_str.chars().count() as i32),
                            Err(_) => AnyValue::Int32(bytes.len() as i32),
                        }
                    },
                    AnyValue::BinaryOwned(bytes) => {
                        // case we have a binary slice, convert to utf8 as spark expects
                        match from_utf8(&bytes) {
                            Ok(utf8_str) => AnyValue::Int32(utf8_str.chars().count() as i32),
                            Err(_) => AnyValue::Int32(bytes.len() as i32),
                        }
                    },
                    AnyValue::String(s) => AnyValue::Int32(s.chars().count() as i32),
                    AnyValue::List(arr) => AnyValue::Int32(arr.len() as i32),
                    _ => AnyValue::Int32(1),
                }
            },
            #[cfg(feature = "zip_with")]
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
                        (AnyValue::BinaryOwned(s), AnyValue::String(oat)) => {
                            AnyValue::Int32(flarion_get_char_position(from_utf8(&s).unwrap(), oat))
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
            },
            LambdaExpression::Cast(expr, data_type) => expr.eval_slice(args).cast(data_type),
        }
    }

    pub(crate) fn eval_any<'a>(&'a self, args: &[AnyValue<'a>]) -> AnyValue<'a> {
        match self {
            LambdaExpression::Null => AnyValue::Null,
            LambdaExpression::Boolean(v) => AnyValue::Boolean(*v),
            #[cfg(feature = "dtype-i8")]
            LambdaExpression::Int8(v) => AnyValue::Int8(*v),
            #[cfg(feature = "dtype-i16")]
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
            #[cfg(feature = "zip_with")]
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
            },
            LambdaExpression::Cast(expr, data_type) => expr.eval_any(args).cast(data_type),
        }
    }

    // None encode that type same as input value
    pub fn return_type(&self, input_type: &DataType) -> Result<DataType, PolarsError> {
        // dtypes_to_supertype
        Ok(match self {
            LambdaExpression::Null => DataType::Null,
            LambdaExpression::Boolean(_) => DataType::Boolean,
            #[cfg(feature = "dtype-i8")]
            LambdaExpression::Int8(_) => DataType::Int8,
            #[cfg(feature = "dtype-i16")]
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
            #[cfg(feature = "zip_with")]
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
            LambdaExpression::Cast(_, data_type) => data_type.clone(),
        })
    }
}

// Was using these to debug but I see no harm in having more unit tests, in fact we should probably have more here
#[cfg(test)]
mod tests {
    use crate::prelude::{LambdaExpression, Series};

    #[test]
    fn test_empty_transform() {
        let start_array = Series::from_iter(vec!["key1==value1", "key2===value2"]);
        let empty_lambda = LambdaExpression::StaticStr("meep".into());

        let res = empty_lambda.eval(&start_array, true).unwrap();
        assert_eq!(res.len(), 2);

        assert_eq!(res.str().unwrap().get(0).unwrap(), "meep");
        assert_eq!(res.str().unwrap().get(1).unwrap(), "meep");
    }

    #[test]
    fn test_lambda_length() {
        let start_array = Series::from_iter(vec!["key1==value1", "key2===value2"]);
        let length_lambda = LambdaExpression::Length(Box::new(LambdaExpression::Variable(0)));

        let res = length_lambda.eval(&start_array, true).unwrap();
        unsafe {
            assert_eq!(res.i32().unwrap().value_unchecked(0), 12);
            assert_eq!(res.i32().unwrap().value_unchecked(1), 13);
        }
    }

    #[test]
    fn test_lambda_substring() {
        let start_array = Series::from_iter(vec!["key1==value1", "key2===value2"]);
        let substring_lambda = LambdaExpression::Substring(
            Box::new(LambdaExpression::Variable(0)),
            Box::new(LambdaExpression::Int32(4)),
            Box::new(LambdaExpression::Int32(2)),
        );

        let res = substring_lambda.eval(&start_array, true).unwrap();
        unsafe {
            assert_eq!(res.str().unwrap().value_unchecked(0), "1=");
            assert_eq!(res.str().unwrap().value_unchecked(1), "2=");
        }
    }

    #[cfg(feature = "zip_with")]
    #[test]
    fn test_lambda_casewhen() {
        let start_array = Series::from_iter(vec![
            "key1==value1",
            "key2===u",
            "key3==value3whichisverylong",
        ]);
        let casewhen_lambda = LambdaExpression::CaseWhen(
            vec![(
                LambdaExpression::GreaterThan(
                    Box::new(LambdaExpression::Length(Box::new(
                        LambdaExpression::Variable(0),
                    ))),
                    Box::new(LambdaExpression::Int32(10)),
                ),
                LambdaExpression::Variable(0),
            )],
            Box::new(LambdaExpression::StaticStr("nope".into())),
        );

        let res = casewhen_lambda.eval(&start_array, true).unwrap();
        unsafe {
            assert_eq!(res.str().unwrap().value_unchecked(0), "key1==value1");
            assert_eq!(res.str().unwrap().value_unchecked(1), "nope");
            assert_eq!(
                res.str().unwrap().value_unchecked(2),
                "key3==value3whichisverylong"
            );
        }
    }
}
