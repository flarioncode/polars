use std::borrow::Cow;
use std::hash::{Hash, Hasher};
use std::iter;
use std::ops::Add;

use num_traits::ToBytes;
use polars_error::{PolarsError, PolarsResult};
use polars_utils::itertools::Itertools;
use polars_utils::pl_str::PlSmallStr;
#[cfg(feature = "serde-lazy")]
use serde::{Deserialize, Serialize};

use super::DataType;
use crate::chunked_array::ops::ChunkCompare;
use crate::prelude::flarion_funcs::{flarion_instr_helper, flarion_slice_helper};
use crate::prelude::{
    BinaryChunked, BooleanChunked, CastOptions, ChunkFull, IntoSeries, NamedFromOwned,
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
