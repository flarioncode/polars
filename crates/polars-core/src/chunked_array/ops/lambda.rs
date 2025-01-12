use std::borrow::Cow;
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use std::ops::Add;
use std::{iter, mem};

use num_traits::ToBytes;
use polars_error::{polars_bail, PolarsError, PolarsResult};
use polars_utils::pl_str::PlSmallStr;
#[cfg(feature = "serde-lazy")]
use serde::{Deserialize, Serialize};

use super::DataType;
use crate::chunked_array::ops::ChunkCompare;
use crate::prelude::flarion_funcs::{
    flarion_get_char_position, flarion_instr_helper, flarion_slice_helper, flarion_substring,
    substring_with_null_length,
};
use crate::prelude::{
    AnyValue, BinaryChunked, BooleanChunked, CastOptions, ChunkFull, IntoSeries, NamedFromOwned,
};
use crate::series::Series;
use crate::utils::dtypes_to_supertype;

impl TryFrom<AnyValue<'_>> for Ordering {
    type Error = PolarsError;

    fn try_from(value: AnyValue) -> Result<Self, Self::Error> {
        match value {
            AnyValue::Int32(0) => Ok(Ordering::Equal),
            AnyValue::Int32(v) if v.is_positive() => Ok(Ordering::Greater),
            AnyValue::Int32(v) if v.is_negative() => Ok(Ordering::Less),
            _ => unreachable!("Expected int32 value"),
        }
    }
}

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
    GreaterThanOrEqual(Box<Self>, Box<Self>),
    LessThan(Box<Self>, Box<Self>),
    LessThanOrEqual(Box<Self>, Box<Self>),
    #[cfg(feature = "zip_with")]
    IfThenElse(Box<Self>, Box<Self>, Box<Self>),
    Length(Box<Self>),
    #[cfg(feature = "zip_with")]
    CaseWhen(Vec<(Self, Self)>, Box<Self>),
    Substring(Box<Self>, Box<Self>, Box<Self>),
    Instr(Box<Self>, Box<Self>),
    Add(Box<Self>, Box<Self>),
    IsNull(Box<Self>),
    IsNotNull(Box<Self>),
    EqualNullSafe(Box<Self>, Box<Self>),
    Cast(Box<Self>, DataType),
}

impl Eq for LambdaExpression {}

impl Hash for LambdaExpression {
    fn hash<H: Hasher>(&self, state: &mut H) {
        mem::discriminant(self).hash(state);
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
            LambdaExpression::GreaterThanOrEqual(first, second) => {
                first.hash(state);
                second.hash(state);
            },
            LambdaExpression::LessThan(first, second) => {
                first.hash(state);
                second.hash(state);
            },
            LambdaExpression::LessThanOrEqual(first, second) => {
                first.hash(state);
                second.hash(state);
            },
            #[cfg(feature = "zip_with")]
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
            LambdaExpression::IsNotNull(v) => v.hash(state),
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

impl LambdaExpression {
    #[cfg(all(feature = "tracy", not(feature = "tracy-no-instrument")))]
    fn fmt_str(&self) -> &'static str {
        match self {
            LambdaExpression::Null => "literal(null)",
            LambdaExpression::Boolean(_) => "literal(boolean)",
            #[cfg(feature = "dtype-i8")]
            LambdaExpression::Int8(_) => "literal(int8)",
            #[cfg(feature = "dtype-i16")]
            LambdaExpression::Int16(_) => "literal(int16)",
            LambdaExpression::Int32(_) => "literal(int32)",
            LambdaExpression::Int64(_) => "literal(int64)",
            LambdaExpression::Float32(_) => "literal(float32)",
            LambdaExpression::Float64(_) => "literal(float64)",
            LambdaExpression::BinaryBlob(_) => "literal(binary)",
            LambdaExpression::StaticStr(_) => "literal(string)",
            LambdaExpression::Variable(_) => "variable({})",
            LambdaExpression::GreaterThan(_, _) => "greater_than",
            LambdaExpression::GreaterThanOrEqual(_, _) => "greater_than_or_equal",
            LambdaExpression::LessThan(_, _) => "less_than",
            LambdaExpression::LessThanOrEqual(_, _) => "less_than_or_equal",
            #[cfg(feature = "zip_with")]
            LambdaExpression::IfThenElse(_, _, _) => "if_then_else",
            LambdaExpression::Length(_) => "length",
            #[cfg(feature = "zip_with")]
            LambdaExpression::CaseWhen(_, _) => "case_when",
            LambdaExpression::Substring(_, _, _) => "substring",
            LambdaExpression::Instr(_, _) => "instr",
            LambdaExpression::Add(_, _) => "add",
            LambdaExpression::IsNull(_) => "is_null",
            LambdaExpression::IsNotNull(_) => "is_not_null",
            LambdaExpression::EqualNullSafe(_, _) => "equal_null_safe",
            LambdaExpression::Cast(_, _) => "cast",
        }
    }

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

    pub fn eval_window<'a>(
        &'a self,
        curr: &AnyValue<'a>,
        next: &AnyValue<'a>,
    ) -> PolarsResult<AnyValue<'a>> {
        #[cfg(all(feature = "tracy", not(feature = "tracy-no-instrument")))]
        tracy_gizmos::zone!(span, "evaluate");
        #[cfg(all(feature = "tracy", not(feature = "tracy-no-instrument")))]
        span.text(self.fmt_str());

        match self {
            LambdaExpression::Null => Ok(AnyValue::Null),
            LambdaExpression::Boolean(v) => Ok(AnyValue::Boolean(*v)),
            #[cfg(feature = "dtype-i8")]
            LambdaExpression::Int8(v) => Ok(AnyValue::Int8(*v)),
            #[cfg(feature = "dtype-i16")]
            LambdaExpression::Int16(v) => Ok(AnyValue::Int16(*v)),
            LambdaExpression::Int32(v) => Ok(AnyValue::Int32(*v)),
            LambdaExpression::Int64(v) => Ok(AnyValue::Int64(*v)),
            LambdaExpression::Float32(v) => Ok(AnyValue::Float32(*v)),
            LambdaExpression::Float64(v) => Ok(AnyValue::Float64(*v)),
            LambdaExpression::BinaryBlob(v) => Ok(AnyValue::Binary(v.as_slice())),
            LambdaExpression::StaticStr(v) => Ok(AnyValue::String(v.as_ref())),
            LambdaExpression::Variable(0) => Ok(curr.to_owned()),
            LambdaExpression::Variable(1) => Ok(next.clone()),
            LambdaExpression::Variable(_) => {
                polars_bail!(InvalidOperation: "No 3rd variable exists for eval_window")
            },
            LambdaExpression::GreaterThan(left, right) => {
                let left = left.eval_window(curr, next)?;
                let right = right.eval_window(curr, next)?;

                // Override the comparison function for array_sort
                match (left.is_null(), right.is_null()) {
                    (true, false) => Ok(AnyValue::Boolean(true)),
                    (false, true) => Ok(AnyValue::Boolean(false)),
                    _ => Ok(AnyValue::Boolean(left.gt(&right))),
                }
            },
            LambdaExpression::GreaterThanOrEqual(left, right) => {
                let left = left.eval_window(curr, next)?;
                let right = right.eval_window(curr, next)?;

                // Override the comparison function for array_sort
                match (left.is_null(), right.is_null()) {
                    (true, false) => Ok(AnyValue::Boolean(true)),
                    (false, true) => Ok(AnyValue::Boolean(false)),
                    _ => Ok(AnyValue::Boolean(left.ge(&right))),
                }
            },
            LambdaExpression::LessThan(left, right) => {
                let left = left.eval_window(curr, next)?;
                let right = right.eval_window(curr, next)?;

                // Override the comparison function for array_sort
                match (left.is_null(), right.is_null()) {
                    (true, false) => Ok(AnyValue::Boolean(false)),
                    (false, true) => Ok(AnyValue::Boolean(true)),
                    _ => Ok(AnyValue::Boolean(left.lt(&right))),
                }
            },
            LambdaExpression::LessThanOrEqual(left, right) => {
                let left = left.eval_window(curr, next)?;
                let right = right.eval_window(curr, next)?;

                // Override the comparison function for array_sort
                match (left.is_null(), right.is_null()) {
                    (true, false) => Ok(AnyValue::Boolean(false)),
                    (false, true) => Ok(AnyValue::Boolean(true)),
                    _ => Ok(AnyValue::Boolean(left.le(&right))),
                }
            },
            #[cfg(feature = "zip_with")]
            LambdaExpression::IfThenElse(pred, value, otherwise) => {
                let pred = match pred.as_ref().eval_window(curr, next)? {
                    AnyValue::Boolean(v) => v,
                    _ => polars_bail!(SchemaMismatch: "Expected boolean value in predicate"),
                };
                if pred {
                    value.as_ref().eval_window(curr, next)
                } else {
                    otherwise.as_ref().eval_window(curr, next)
                }
            },
            LambdaExpression::Length(child) => {
                let s = child.eval_window(curr, next)?;
                Ok(match s {
                    AnyValue::Null => AnyValue::Null,
                    AnyValue::Binary(v) => AnyValue::Int32(v.len() as i32),
                    AnyValue::String(v) => AnyValue::Int32(v.chars().count() as i32),
                    AnyValue::List(inner_list) => AnyValue::Int32(inner_list.len() as i32),
                    _ => AnyValue::Int32(1),
                })
            },
            #[cfg(feature = "zip_with")]
            LambdaExpression::CaseWhen(branches, otherwise) => {
                let mut result = None;
                for (pred, value) in branches {
                    let pred = match pred.eval_window(curr, next)? {
                        AnyValue::Boolean(v) => v,
                        _ => polars_bail!(SchemaMismatch: "Expected boolean value in predicate"),
                    };

                    if pred {
                        result = Some(value.eval_window(curr, next)?);
                        break;
                    }
                }

                if let Some(result) = result {
                    Ok(result)
                } else {
                    otherwise.eval_window(curr, next)
                }
            },
            LambdaExpression::Substring(child, start, length) => {
                let child = child.eval_window(curr, next)?;
                let start = start.eval_window(curr, next)?;
                let length = length.eval_window(curr, next)?;

                match (child, start, length) {
                    (AnyValue::String(s), AnyValue::Int32(start), AnyValue::Null) => {
                        Ok(substring_with_null_length(s, start)
                            .map(AnyValue::String)
                            .unwrap_or(AnyValue::Null))
                    },
                    (AnyValue::String(s), AnyValue::Int32(start), AnyValue::Int32(length)) => {
                        Ok(AnyValue::String(flarion_substring(s, start, length)))
                    },
                    (child, start, length) => {
                        polars_bail!(SchemaMismatch: "Expected (string, int32, int32), Found ({}, {}, {})", child, start, length)
                    },
                }
            },
            LambdaExpression::Instr(left, right) => {
                let left = left.eval_window(curr, next)?;
                let right = right.eval_window(curr, next)?;

                let (str_val, pat) = match (&left, &right) {
                    (AnyValue::StringOwned(str_val), AnyValue::StringOwned(pat)) => {
                        (str_val.as_str(), pat.as_str())
                    },
                    (AnyValue::StringOwned(str_val), AnyValue::String(pat)) => {
                        (str_val.as_str(), *pat)
                    },
                    (AnyValue::String(str_val), AnyValue::StringOwned(pat)) => {
                        (*str_val, pat.as_str())
                    },
                    (AnyValue::String(str_val), AnyValue::String(pat)) => (*str_val, *pat),
                    (left, right) => {
                        polars_bail!(SchemaMismatch: "Expected (string, string), found ({:?}, {:?})", left, right)
                    },
                };

                Ok(AnyValue::Int32(flarion_get_char_position(str_val, pat)))
            },
            LambdaExpression::Add(left, right) => {
                let left = left.eval_window(curr, next)?;
                let right = right.eval_window(curr, next)?;
                Ok(left.add(&right))
            },
            LambdaExpression::IsNull(child) => {
                let child = child.eval_window(curr, next)?;
                Ok(AnyValue::Boolean(child.is_null()))
            },
            LambdaExpression::IsNotNull(child) => {
                let child = child.eval_window(curr, next)?;
                Ok(AnyValue::Boolean(!child.is_null()))
            },
            LambdaExpression::EqualNullSafe(left, right) => {
                let left = left.eval_window(curr, next)?;
                let right = right.eval_window(curr, next)?;
                Ok(AnyValue::Boolean(left == right))
            },
            LambdaExpression::Cast(child, dtype) => {
                let s = child.eval_window(curr, next)?;
                Ok(s.cast(dtype).to_owned())
            },
        }
    }

    pub fn eval(&self, s: &Series, i: Option<&Series>) -> PolarsResult<Series> {
        #[cfg(all(feature = "tracy", not(feature = "tracy-no-instrument")))]
        tracy_gizmos::zone!(span, "eval");
        #[cfg(all(feature = "tracy", not(feature = "tracy-no-instrument")))]
        span.text(self.fmt_str());

        match self {
            LambdaExpression::Null => Ok(Series::new_null(PlSmallStr::EMPTY, 1)),
            LambdaExpression::Boolean(v) => Ok(Series::from_iter(iter::once(v))),
            #[cfg(feature = "dtype-i8")]
            LambdaExpression::Int8(v) => Ok(Series::from_iter(iter::once(v))),
            #[cfg(feature = "dtype-i16")]
            LambdaExpression::Int16(v) => Ok(Series::from_iter(iter::once(v))),
            LambdaExpression::Int32(v) => Ok(Series::from_iter(iter::once(v))),
            LambdaExpression::Int64(v) => Ok(Series::from_iter(iter::once(v))),
            LambdaExpression::Float32(v) => Ok(Series::from_iter(iter::once(v))),
            LambdaExpression::Float64(v) => Ok(Series::from_iter(iter::once(v))),
            LambdaExpression::BinaryBlob(v) => {
                Ok(BinaryChunked::from_iter(iter::once(v.as_slice())).into_series())
            },
            LambdaExpression::StaticStr(v) => Ok(Series::from_iter(iter::once(v.as_ref()))),
            LambdaExpression::Variable(0) => Ok(s.clone()),
            LambdaExpression::Variable(1) => Ok(i.unwrap().clone()), // Will never get called if there are not indices
            LambdaExpression::Variable(_) => {
                polars_bail!(InvalidOperation: "No 3rd variable exists for eval")
            },
            LambdaExpression::GreaterThan(left, right) => {
                let left = left.eval(s, i)?;
                let right = right.eval(s, i)?;
                Ok(left.gt(&right)?.into_series())
            },
            LambdaExpression::GreaterThanOrEqual(left, right) => {
                let left = left.eval(s, i)?;
                let right = right.eval(s, i)?;
                Ok(left.gt_eq(&right)?.into_series())
            },
            LambdaExpression::LessThan(left, right) => {
                let left = left.eval(s, i)?;
                let right = right.eval(s, i)?;
                Ok(left.lt(&right)?.into_series())
            },
            LambdaExpression::LessThanOrEqual(left, right) => {
                let left = left.eval(s, i)?;
                let right = right.eval(s, i)?;
                Ok(left.lt_eq(&right)?.into_series())
            },
            #[cfg(feature = "zip_with")]
            LambdaExpression::IfThenElse(pred, value, otherwise) => {
                let pred = pred.eval(s, i)?;
                let value = value.eval(s, i)?;
                let otherwise = otherwise.eval(s, i)?;

                // I think this is the only place where its valid to use AnyValues, since both series can be absolutely anything
                value
                    .zip_with(pred.bool()?, &otherwise)
                    .map(IntoSeries::into_series)
            },
            LambdaExpression::Length(child) => {
                let s = child.eval(s, i)?;
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
                    .map(|(pred, _)| pred.eval(s, i))
                    .collect::<PolarsResult<_>>()?;
                let branches: Vec<(&BooleanChunked, Series)> = predicates
                    .iter()
                    .zip(branches)
                    .map(|(predicates, (_, thens))| Ok((predicates.bool()?, thens.eval(s, i)?)))
                    .collect::<PolarsResult<_>>()?;
                let otherwise: Series = otherwise.eval(s, i)?;

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
                let child = child.eval(s, i)?;
                let start = start.eval(s, i)?;
                let length = length.eval(s, i)?;

                flarion_slice_helper(child.str()?, &start, &length)
            },
            LambdaExpression::Instr(left, right) => {
                let left = left.eval(s, i)?;
                let right = right.eval(s, i)?;

                flarion_instr_helper(left.str()?, &right)
            },
            LambdaExpression::Add(left, right) => {
                let left = left.eval(s, i)?;
                let right = right.eval(s, i)?;
                left.add(right)
            },
            LambdaExpression::IsNull(child) => {
                let child = child.eval(s, i)?;
                Ok(child.is_null().into_series())
            },
            LambdaExpression::IsNotNull(child) => {
                let child = child.eval(s, i)?;
                Ok(child.is_not_null().into_series())
            },
            LambdaExpression::EqualNullSafe(left, right) => {
                let left = left.eval(s, i)?;
                let right = right.eval(s, i)?;
                left.equal_missing(&right).map(IntoSeries::into_series)
            },
            LambdaExpression::Cast(child, dtype) => {
                let s = child.eval(s, i)?;
                s.cast_with_options(dtype, CastOptions::Overflowing)
            },
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
            LambdaExpression::Variable(1) => DataType::Int32,
            LambdaExpression::Variable(_) => input_type.clone(),
            LambdaExpression::GreaterThan(_, _) => DataType::Boolean,
            LambdaExpression::GreaterThanOrEqual(_, _) => DataType::Boolean,
            LambdaExpression::LessThan(_, _) => DataType::Boolean,
            LambdaExpression::LessThanOrEqual(_, _) => DataType::Boolean,
            #[cfg(feature = "zip_with")]
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
            LambdaExpression::IsNotNull(_) => DataType::Boolean,
            LambdaExpression::EqualNullSafe(_, _) => DataType::Boolean,
            LambdaExpression::Cast(_, data_type) => data_type.clone(),
        })
    }
}
