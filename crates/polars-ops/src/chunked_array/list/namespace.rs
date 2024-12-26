use std::fmt::Write;

use arrow::array::{Array, MutableArray, MutablePrimitiveArray, ValueSize};
use arrow::legacy::kernels::list::{index_is_oob, sublist_get};
use polars_core::chunked_array::builder::get_list_builder;
#[cfg(feature = "list_gather")]
use polars_core::export::num::ToPrimitive;
#[cfg(feature = "list_gather")]
use polars_core::export::num::{NumCast, Signed, Zero};
use polars_core::series::amortized_iter::AmortSeries;
#[cfg(feature = "diff")]
use polars_core::series::ops::NullBehavior;
use polars_core::utils::try_get_supertype;

use super::*;
#[cfg(feature = "list_any_all")]
use crate::chunked_array::list::any_all::*;
use crate::chunked_array::list::min_max::{list_max_function, list_min_function};
use crate::chunked_array::list::sum_mean::sum_with_nulls;
#[cfg(feature = "diff")]
use crate::prelude::diff;
use crate::prelude::list::sum_mean::{mean_list_numerical, sum_list_numerical};
use crate::series::ArgAgg;

pub(super) fn has_inner_nulls(ca: &ListChunked) -> bool {
    for arr in ca.downcast_iter() {
        if arr.values().null_count() > 0 {
            return true;
        }
    }
    false
}

fn cast_rhs(
    other: &mut [Series],
    inner_type: &DataType,
    dtype: &DataType,
    length: usize,
    allow_broadcast: bool,
) -> PolarsResult<()> {
    for s in other.iter_mut() {
        // make sure that inner types match before we coerce into list
        if !matches!(s.dtype(), DataType::List(_)) {
            *s = s.cast(inner_type)?
        }
        if !matches!(s.dtype(), DataType::List(_)) && s.dtype() == inner_type {
            // coerce to list JIT
            *s = s.reshape_list(&[-1, 1]).unwrap();
        }

        if s.dtype() != dtype {
            *s = s.cast(dtype).map_err(|e| {
                polars_err!(
                    SchemaMismatch:
                    "cannot concat `{}` into a list of `{}`: {}",
                    s.dtype(),
                    dtype,
                    e
                )
            })?;
        }

        if s.len() != length {
            polars_ensure!(
                s.len() == 1,
                ShapeMismatch: "series length {} does not match expected length of {}",
                s.len(), length
            );
            if allow_broadcast {
                // broadcast JIT
                *s = s.new_from_index(0, length)
            }
            // else do nothing
        }
    }
    Ok(())
}

pub fn list_flarion_slice_amortized(s: AmortSeries, mut offset: i64, length: i64) -> Series {
    let element_len = s.as_ref().len() as i64;
    if offset < 0 {
        offset += element_len;
    }
    if (0..element_len).contains(&offset) {
        s.as_ref().slice(offset, length as usize)
    } else {
        Series::new_empty(s.as_ref().name().clone(), s.as_ref().dtype())
    }
}

pub trait ListNameSpaceImpl: AsList {
    /// In case the inner dtype [`DataType::String`], the individual items will be joined into a
    /// single string separated by `separator`.
    fn lst_join(
        &self,
        separator: &StringChunked,
        ignore_nulls: bool,
    ) -> PolarsResult<StringChunked> {
        let ca = self.as_list();
        match ca.inner_dtype() {
            DataType::String => match separator.len() {
                1 => match separator.get(0) {
                    Some(separator) => self.join_literal(separator, ignore_nulls),
                    _ => Ok(StringChunked::full_null(ca.name().clone(), ca.len())),
                },
                _ => self.join_many(separator, ignore_nulls),
            },
            dt => polars_bail!(op = "`lst.join`", got = dt, expected = "String"),
        }
    }

    fn join_literal(&self, separator: &str, ignore_nulls: bool) -> PolarsResult<StringChunked> {
        let ca = self.as_list();
        // used to amortize heap allocs
        let mut buf = String::with_capacity(128);
        let mut builder = StringChunkedBuilder::new(ca.name().clone(), ca.len());

        ca.for_each_amortized(|opt_s| {
            let opt_val = opt_s.and_then(|s| {
                // make sure that we don't write values of previous iteration
                buf.clear();
                let ca = s.as_ref().str().unwrap();

                if ca.null_count() != 0 && !ignore_nulls {
                    return None;
                }

                for arr in ca.downcast_iter() {
                    for val in arr.non_null_values_iter() {
                        buf.write_str(val).unwrap();
                        buf.write_str(separator).unwrap();
                    }
                }

                // last value should not have a separator, so slice that off
                // saturating sub because there might have been nothing written.
                Some(&buf[..buf.len().saturating_sub(separator.len())])
            });
            builder.append_option(opt_val)
        });
        Ok(builder.finish())
    }

    fn join_many(
        &self,
        separator: &StringChunked,
        ignore_nulls: bool,
    ) -> PolarsResult<StringChunked> {
        let ca = self.as_list();
        // used to amortize heap allocs
        let mut buf = String::with_capacity(128);
        let mut builder = StringChunkedBuilder::new(ca.name().clone(), ca.len());
        {
            ca.amortized_iter()
                .zip(separator)
                .for_each(|(opt_s, opt_sep)| match opt_sep {
                    Some(separator) => {
                        let opt_val = opt_s.and_then(|s| {
                            // make sure that we don't write values of previous iteration
                            buf.clear();
                            let ca = s.as_ref().str().unwrap();

                            if ca.null_count() != 0 && !ignore_nulls {
                                return None;
                            }

                            for arr in ca.downcast_iter() {
                                for val in arr.non_null_values_iter() {
                                    buf.write_str(val).unwrap();
                                    buf.write_str(separator).unwrap();
                                }
                            }

                            // last value should not have a separator, so slice that off
                            // saturating sub because there might have been nothing written.
                            Some(&buf[..buf.len().saturating_sub(separator.len())])
                        });
                        builder.append_option(opt_val)
                    },
                    _ => builder.append_null(),
                })
        }
        Ok(builder.finish())
    }

    fn lst_max(&self) -> PolarsResult<Series> {
        list_max_function(self.as_list())
    }

    #[cfg(feature = "list_any_all")]
    fn lst_all(&self) -> PolarsResult<Series> {
        let ca = self.as_list();
        list_all(ca)
    }

    #[cfg(feature = "list_any_all")]
    fn lst_any(&self) -> PolarsResult<Series> {
        let ca = self.as_list();
        list_any(ca)
    }

    fn lst_min(&self) -> PolarsResult<Series> {
        list_min_function(self.as_list())
    }

    fn lst_sum(&self) -> PolarsResult<Series> {
        let ca = self.as_list();

        if has_inner_nulls(ca) {
            return sum_with_nulls(ca, ca.inner_dtype());
        };

        match ca.inner_dtype() {
            DataType::Boolean => Ok(count_boolean_bits(ca).into_series()),
            dt if dt.is_numeric() => Ok(sum_list_numerical(ca, dt)),
            dt => sum_with_nulls(ca, dt),
        }
    }

    fn lst_mean(&self) -> Series {
        let ca = self.as_list();

        if has_inner_nulls(ca) {
            return sum_mean::mean_with_nulls(ca);
        };

        match ca.inner_dtype() {
            dt if dt.is_numeric() => mean_list_numerical(ca, dt),
            _ => sum_mean::mean_with_nulls(ca),
        }
    }

    fn lst_median(&self) -> Series {
        let ca = self.as_list();
        dispersion::median_with_nulls(ca)
    }

    fn lst_std(&self, ddof: u8) -> Series {
        let ca = self.as_list();
        dispersion::std_with_nulls(ca, ddof)
    }

    fn lst_var(&self, ddof: u8) -> Series {
        let ca = self.as_list();
        dispersion::var_with_nulls(ca, ddof)
    }

    fn same_type(&self, out: ListChunked) -> ListChunked {
        let ca = self.as_list();
        let dtype = ca.dtype();
        if out.dtype() != dtype {
            out.cast(ca.dtype()).unwrap().list().unwrap().clone()
        } else {
            out
        }
    }

    fn eval_lambda_on_amort(
        s_ref: &Series,
        lambda_expression: Arc<LambdaExpression>,
        index_data: Option<&mut (MutablePrimitiveArray<i32>, AmortSeries)>,
    ) -> PolarsResult<Series> {
        let s_len = s_ref.len() as i32;
        (if let Some((index_arr, index_amort)) = index_data {
            let index_count = index_arr.len() as i32;
            if index_count < s_len {
                index_arr.extend_trusted_len_values(index_count..s_len)
            } else {
                index_arr.truncate(s_len as usize);
            }

            // Create a temporary Box<dyn Array> for this iteration, this is unsafe code but I think should be ok here??
            let mut temp_box: Box<dyn Array> = unsafe { std::mem::transmute(index_arr.as_box()) };

            unsafe {
                index_amort.with_array(&mut temp_box, |arr| {
                    lambda_expression.eval(s_ref, Some(arr.as_ref()))
                })
            }
        } else {
            lambda_expression.eval(s_ref, None)
        }).and_then(|eval_result| {
            let eval_result_len = eval_result.len() as i32;
            if eval_result_len != s_len {
                if eval_result_len == 1 {
                    Ok(match eval_result.dtype() {
                        DataType::Boolean => Series::from_iter(std::iter::repeat_n(eval_result.bool()?.get(0), s_len as usize)),
                        DataType::UInt8 => Series::from_iter(std::iter::repeat_n(eval_result.u8()?.get(0), s_len as usize)),
                        DataType::UInt16 => Series::from_iter(std::iter::repeat_n(eval_result.u16()?.get(0), s_len as usize)),
                        DataType::UInt32 => Series::from_iter(std::iter::repeat_n(eval_result.u32()?.get(0), s_len as usize)),
                        DataType::UInt64 => Series::from_iter(std::iter::repeat_n(eval_result.u64()?.get(0), s_len as usize)),
                        DataType::Int8 => Series::from_iter(std::iter::repeat_n(eval_result.i8()?.get(0), s_len as usize)),
                        DataType::Int16 => Series::from_iter(std::iter::repeat_n(eval_result.i16()?.get(0), s_len as usize)),
                        DataType::Int32 => Series::from_iter(std::iter::repeat_n(eval_result.i32()?.get(0), s_len as usize)),
                        DataType::Int64 => Series::from_iter(std::iter::repeat_n(eval_result.i64()?.get(0), s_len as usize)),
                        DataType::Float32 => Series::from_iter(std::iter::repeat_n(eval_result.f32()?.get(0), s_len as usize)),
                        DataType::Float64 => Series::from_iter(std::iter::repeat_n(eval_result.f64()?.get(0), s_len as usize)),
                        DataType::String => Series::from_iter(std::iter::repeat_n(eval_result.str()?.get(0), s_len as usize)),
                        DataType::Binary => BinaryChunked::from_iter(std::iter::repeat_n(eval_result.binary()?.get(0), s_len as usize)).into_series(),
                        DataType::List(_) => ListChunked::from_iter(std::iter::repeat_n(eval_result.list()?.get_as_series(0), s_len as usize)).into_series(),
                        other => polars_bail!(SchemaMismatch: "invalid dtype for lambda function: {}", other),
                    })
                } else {
                    polars_bail!(ShapeMismatch: "lambda function did not return a series of equal length")
                }
            } else {
                Ok(eval_result)
            }
        })
    }

    #[cfg_attr(
        all(feature = "tracy", not(feature = "tracy-no-instrument")),
        tracy_gizmos::instrument
    )]
    fn lst_filter_by_func(
        &self,
        lambda_expression: Arc<LambdaExpression>,
        with_index: bool,
    ) -> PolarsResult<ListChunked> {
        let mut index_data = with_index.then(|| {
            (
                MutablePrimitiveArray::<i32>::with_capacity(8),
                AmortSeries::new(Series::new_empty(PlSmallStr::EMPTY, &DataType::Int32).into()),
            )
        });

        self.as_list().try_apply_amortized(|s| {
            let s_ref = s.as_ref();

            s_ref.filter(
                Self::eval_lambda_on_amort(s_ref, lambda_expression.clone(), index_data.as_mut())?
                    .bool()?,
            )
        })
    }

    #[cfg_attr(
        all(feature = "tracy", not(feature = "tracy-no-instrument")),
        tracy_gizmos::instrument
    )]
    fn lst_transform(
        &self,
        lambda_expression: Arc<LambdaExpression>,
        with_index: bool,
    ) -> PolarsResult<ListChunked> {
        let mut index_data = with_index.then(|| {
            (
                MutablePrimitiveArray::<i32>::with_capacity(8),
                AmortSeries::new(Series::new_empty(PlSmallStr::EMPTY, &DataType::Int32).into()),
            )
        });

        self.as_list().try_apply_amortized(|s| {
            let s_ref = s.as_ref();
            Self::eval_lambda_on_amort(s_ref, lambda_expression.clone(), index_data.as_mut())
        })
    }

    fn lst_sort(&self, options: SortOptions) -> PolarsResult<ListChunked> {
        let ca = self.as_list();
        let out = ca.try_apply_amortized(|s| s.as_ref().sort_with(options))?;
        Ok(self.same_type(out))
    }

    fn lst_sort_by_func(
        &self,
        _options: SortOptions,
        lambda_expressions: Arc<LambdaExpression>,
    ) -> PolarsResult<ListChunked> {
        let ca = self.as_list();
        ca.try_apply_amortized(|s| {
            let mut vals = s.as_ref().iter().collect::<Vec<_>>();
            vals.sort_by(
                |curr, next| {
                    lambda_expressions
                        .eval_window(curr, next)
                        .expect("Could not run lambda expression")
                        .try_into()
                        .unwrap()
                }, // Conversion already has some error handling
            );

            Series::from_any_values_and_dtype(PlSmallStr::EMPTY, &vals, ca.inner_dtype(), true)
        })
    }

    #[must_use]
    fn lst_reverse(&self) -> ListChunked {
        let ca = self.as_list();
        let out = ca.apply_amortized(|s| s.as_ref().reverse());
        self.same_type(out)
    }

    fn lst_n_unique(&self) -> PolarsResult<IdxCa> {
        let ca = self.as_list();
        ca.try_apply_amortized_generic(|s| {
            let opt_v = s.map(|s| s.as_ref().n_unique()).transpose()?;
            Ok(opt_v.map(|idx| idx as IdxSize))
        })
    }

    fn lst_unique(&self) -> PolarsResult<ListChunked> {
        let ca = self.as_list();
        let out = ca.try_apply_amortized(|s| s.as_ref().unique())?;
        Ok(self.same_type(out))
    }

    fn lst_unique_stable(&self) -> PolarsResult<ListChunked> {
        let ca = self.as_list();
        let out = ca.try_apply_amortized(|s| s.as_ref().unique_stable())?;
        Ok(self.same_type(out))
    }

    fn lst_arg_min(&self) -> IdxCa {
        let ca = self.as_list();
        ca.apply_amortized_generic(|opt_s| {
            opt_s.and_then(|s| s.as_ref().arg_min().map(|idx| idx as IdxSize))
        })
    }

    fn lst_arg_max(&self) -> IdxCa {
        let ca = self.as_list();
        ca.apply_amortized_generic(|opt_s| {
            opt_s.and_then(|s| s.as_ref().arg_max().map(|idx| idx as IdxSize))
        })
    }

    #[cfg(feature = "diff")]
    fn lst_diff(&self, n: i64, null_behavior: NullBehavior) -> PolarsResult<ListChunked> {
        let ca = self.as_list();
        ca.try_apply_amortized(|s| diff(s.as_ref(), n, null_behavior))
    }

    fn lst_shift(&self, periods: &Series) -> PolarsResult<ListChunked> {
        let ca = self.as_list();
        let periods_s = periods.cast(&DataType::Int64)?;
        let periods = periods_s.i64()?;
        let out = match periods.len() {
            1 => {
                if let Some(periods) = periods.get(0) {
                    ca.apply_amortized(|s| s.as_ref().shift(periods))
                } else {
                    ListChunked::full_null_with_dtype(ca.name().clone(), ca.len(), ca.inner_dtype())
                }
            },
            _ => ca.zip_and_apply_amortized(periods, |opt_s, opt_periods| {
                match (opt_s, opt_periods) {
                    (Some(s), Some(periods)) => Some(s.as_ref().shift(periods)),
                    _ => None,
                }
            }),
        };
        Ok(self.same_type(out))
    }

    fn lst_flarion_slice(&self, (offset, length): (i64, i64)) -> ListChunked {
        let ca = self.as_list();
        let out = ca.apply_amortized(|s| list_flarion_slice_amortized(s, offset, length));
        self.same_type(out)
    }

    fn lst_slice(&self, offset: i64, length: usize) -> ListChunked {
        let ca = self.as_list();
        let out = ca.apply_amortized(|s| s.as_ref().slice(offset, length));
        self.same_type(out)
    }

    fn lst_lengths(&self) -> IdxCa {
        let ca = self.as_list();
        let mut lengths = Vec::with_capacity(ca.len());
        ca.downcast_iter().for_each(|arr| {
            let offsets = arr.offsets().as_slice();
            let mut last = offsets[0];
            for o in &offsets[1..] {
                lengths.push((*o - last) as IdxSize);
                last = *o;
            }
        });
        IdxCa::from_vec(ca.name().clone(), lengths)
    }

    /// Get the value by index in the sublists.
    /// So index `0` would return the first item of every sublist
    /// and index `-1` would return the last item of every sublist
    /// if an index is out of bounds, it will return a `None`.
    fn lst_get(&self, idx: i64, null_on_oob: bool) -> PolarsResult<Series> {
        let ca = self.as_list();
        if !null_on_oob && ca.downcast_iter().any(|arr| index_is_oob(arr, idx)) {
            polars_bail!(ComputeError: "get index is out of bounds");
        }

        let chunks = ca
            .downcast_iter()
            .map(|arr| sublist_get(arr, idx))
            .collect::<Vec<_>>();
        // SAFETY: every element in list has dtype equal to its inner type
        unsafe {
            Series::try_from((ca.name().clone(), chunks))
                .unwrap()
                .cast_unchecked(ca.inner_dtype())
        }
    }

    #[cfg(feature = "list_gather")]
    fn lst_gather_every(&self, n: &IdxCa, offset: &IdxCa) -> PolarsResult<Series> {
        let list_ca = self.as_list();
        let out = match (n.len(), offset.len()) {
            (1, 1) => match (n.get(0), offset.get(0)) {
                (Some(n), Some(offset)) => list_ca
                    .apply_amortized(|s| s.as_ref().gather_every(n as usize, offset as usize)),
                _ => ListChunked::full_null_with_dtype(
                    list_ca.name().clone(),
                    list_ca.len(),
                    list_ca.inner_dtype(),
                ),
            },
            (1, len_offset) if len_offset == list_ca.len() => {
                if let Some(n) = n.get(0) {
                    list_ca.zip_and_apply_amortized(offset, |opt_s, opt_offset| {
                        match (opt_s, opt_offset) {
                            (Some(s), Some(offset)) => {
                                Some(s.as_ref().gather_every(n as usize, offset as usize))
                            },
                            _ => None,
                        }
                    })
                } else {
                    ListChunked::full_null_with_dtype(
                        list_ca.name().clone(),
                        list_ca.len(),
                        list_ca.inner_dtype(),
                    )
                }
            },
            (len_n, 1) if len_n == list_ca.len() => {
                if let Some(offset) = offset.get(0) {
                    list_ca.zip_and_apply_amortized(n, |opt_s, opt_n| match (opt_s, opt_n) {
                        (Some(s), Some(n)) => {
                            Some(s.as_ref().gather_every(n as usize, offset as usize))
                        },
                        _ => None,
                    })
                } else {
                    ListChunked::full_null_with_dtype(
                        list_ca.name().clone(),
                        list_ca.len(),
                        list_ca.inner_dtype(),
                    )
                }
            },
            (len_n, len_offset) if len_n == len_offset && len_n == list_ca.len() => list_ca
                .binary_zip_and_apply_amortized(n, offset, |opt_s, opt_n, opt_offset| {
                    match (opt_s, opt_n, opt_offset) {
                        (Some(s), Some(n), Some(offset)) => {
                            Some(s.as_ref().gather_every(n as usize, offset as usize))
                        },
                        _ => None,
                    }
                }),
            _ => {
                polars_bail!(ComputeError: "The lengths of `n` and `offset` should be 1 or equal to the length of list.")
            },
        };
        Ok(out.into_series())
    }

    #[cfg(feature = "list_gather")]
    fn lst_gather(&self, idx: &Series, null_on_oob: bool) -> PolarsResult<Series> {
        let list_ca = self.as_list();

        let index_typed_index = |idx: &Series| {
            let idx = idx.cast(&IDX_DTYPE).unwrap();
            {
                list_ca
                    .amortized_iter()
                    .map(|s| {
                        s.map(|s| {
                            let s = s.as_ref();
                            take_series(s, idx.clone(), null_on_oob)
                        })
                        .transpose()
                    })
                    .collect::<PolarsResult<ListChunked>>()
                    .map(|mut ca| {
                        ca.rename(list_ca.name().clone());
                        ca.into_series()
                    })
            }
        };

        use DataType::*;
        match idx.dtype() {
            List(boxed_dt) if boxed_dt.is_integer() => {
                let idx_ca = idx.list().unwrap();
                let mut out = {
                    list_ca
                        .amortized_iter()
                        .zip(idx_ca)
                        .map(|(opt_s, opt_idx)| {
                            {
                                match (opt_s, opt_idx) {
                                    (Some(s), Some(idx)) => {
                                        Some(take_series(s.as_ref(), idx, null_on_oob))
                                    },
                                    _ => None,
                                }
                            }
                            .transpose()
                        })
                        .collect::<PolarsResult<ListChunked>>()?
                };
                out.rename(list_ca.name().clone());

                Ok(out.into_series())
            },
            UInt32 | UInt64 => index_typed_index(idx),
            dt if dt.is_signed_integer() => {
                if let Some(min) = idx.min::<i64>().unwrap() {
                    if min >= 0 {
                        index_typed_index(idx)
                    } else {
                        let mut out = {
                            list_ca
                                .amortized_iter()
                                .map(|opt_s| {
                                    opt_s
                                        .map(|s| take_series(s.as_ref(), idx.clone(), null_on_oob))
                                        .transpose()
                                })
                                .collect::<PolarsResult<ListChunked>>()?
                        };
                        out.rename(list_ca.name().clone());
                        Ok(out.into_series())
                    }
                } else {
                    polars_bail!(ComputeError: "all indices are null");
                }
            },
            dt => polars_bail!(ComputeError: "cannot use dtype `{}` as an index", dt),
        }
    }

    #[cfg(feature = "list_drop_nulls")]
    fn lst_drop_nulls(&self) -> ListChunked {
        let list_ca = self.as_list();

        list_ca.apply_amortized(|s| s.as_ref().drop_nulls())
    }

    #[cfg(feature = "list_sample")]
    fn lst_sample_n(
        &self,
        n: &Series,
        with_replacement: bool,
        shuffle: bool,
        seed: Option<u64>,
    ) -> PolarsResult<ListChunked> {
        let ca = self.as_list();

        let n_s = n.cast(&IDX_DTYPE)?;
        let n = n_s.idx()?;

        let out = match n.len() {
            1 => {
                if let Some(n) = n.get(0) {
                    ca.try_apply_amortized(|s| {
                        s.as_ref()
                            .sample_n(n as usize, with_replacement, shuffle, seed)
                    })
                } else {
                    Ok(ListChunked::full_null_with_dtype(
                        ca.name().clone(),
                        ca.len(),
                        ca.inner_dtype(),
                    ))
                }
            },
            _ => ca.try_zip_and_apply_amortized(n, |opt_s, opt_n| match (opt_s, opt_n) {
                (Some(s), Some(n)) => s
                    .as_ref()
                    .sample_n(n as usize, with_replacement, shuffle, seed)
                    .map(Some),
                _ => Ok(None),
            }),
        };
        out.map(|ok| self.same_type(ok))
    }

    #[cfg(feature = "list_sample")]
    fn lst_sample_fraction(
        &self,
        fraction: &Series,
        with_replacement: bool,
        shuffle: bool,
        seed: Option<u64>,
    ) -> PolarsResult<ListChunked> {
        let ca = self.as_list();

        let fraction_s = fraction.cast(&DataType::Float64)?;
        let fraction = fraction_s.f64()?;

        let out = match fraction.len() {
            1 => {
                if let Some(fraction) = fraction.get(0) {
                    ca.try_apply_amortized(|s| {
                        let n = (s.as_ref().len() as f64 * fraction) as usize;
                        s.as_ref().sample_n(n, with_replacement, shuffle, seed)
                    })
                } else {
                    Ok(ListChunked::full_null_with_dtype(
                        ca.name().clone(),
                        ca.len(),
                        ca.inner_dtype(),
                    ))
                }
            },
            _ => ca.try_zip_and_apply_amortized(fraction, |opt_s, opt_n| match (opt_s, opt_n) {
                (Some(s), Some(fraction)) => {
                    let n = (s.as_ref().len() as f64 * fraction) as usize;
                    s.as_ref()
                        .sample_n(n, with_replacement, shuffle, seed)
                        .map(Some)
                },
                _ => Ok(None),
            }),
        };
        out.map(|ok| self.same_type(ok))
    }

    fn lst_concat(&self, other: &[Series]) -> PolarsResult<ListChunked> {
        let ca = self.as_list();
        let other_len = other.len();
        let length = ca.len();
        let mut other = other.to_vec();
        let mut inner_super_type = ca.inner_dtype().clone();

        for s in &other {
            match s.dtype() {
                DataType::List(inner_type) => {
                    inner_super_type = try_get_supertype(&inner_super_type, inner_type)?;
                    #[cfg(feature = "dtype-categorical")]
                    if matches!(
                        &inner_super_type,
                        DataType::Categorical(_, _) | DataType::Enum(_, _)
                    ) {
                        inner_super_type = merge_dtypes(&inner_super_type, inner_type)?;
                    }
                },
                dt => {
                    inner_super_type = try_get_supertype(&inner_super_type, dt)?;
                    #[cfg(feature = "dtype-categorical")]
                    if matches!(
                        &inner_super_type,
                        DataType::Categorical(_, _) | DataType::Enum(_, _)
                    ) {
                        inner_super_type = merge_dtypes(&inner_super_type, dt)?;
                    }
                },
            }
        }

        // cast lhs
        let dtype = &DataType::List(Box::new(inner_super_type.clone()));
        let ca = ca.cast(dtype)?;
        let ca = ca.list().unwrap();

        // broadcasting path in case all unit length
        // this path will not expand the series, so saves memory
        let out = if other.iter().all(|s| s.len() == 1) && ca.len() != 1 {
            cast_rhs(&mut other, &inner_super_type, dtype, length, false)?;
            let to_append = other
                .iter()
                .flat_map(|s| {
                    let lst = s.list().unwrap();
                    lst.get_as_series(0)
                })
                .collect::<Vec<_>>();
            // there was a None, so all values will be None
            if to_append.len() != other_len {
                return Ok(ListChunked::full_null_with_dtype(
                    ca.name().clone(),
                    length,
                    &inner_super_type,
                ));
            }

            let vals_size_other = other
                .iter()
                .map(|s| s.list().unwrap().get_values_size())
                .sum::<usize>();

            let mut builder = get_list_builder(
                &inner_super_type,
                ca.get_values_size() + vals_size_other + 1,
                length,
                ca.name().clone(),
            )?;
            ca.into_iter().for_each(|opt_s| {
                let opt_s = opt_s.map(|mut s| {
                    for append in &to_append {
                        s.append(append).unwrap();
                    }
                    match inner_super_type {
                        // structs don't have chunks, so we must first rechunk the underlying series
                        #[cfg(feature = "dtype-struct")]
                        DataType::Struct(_) => s = s.rechunk(),
                        // nothing
                        _ => {},
                    }
                    s
                });
                builder.append_opt_series(opt_s.as_ref()).unwrap();
            });
            builder.finish()
        } else {
            // normal path which may contain same length list or unit length lists
            cast_rhs(&mut other, &inner_super_type, dtype, length, true)?;

            let vals_size_other = other
                .iter()
                .map(|s| s.list().unwrap().get_values_size())
                .sum::<usize>();
            let mut iters = Vec::with_capacity(other_len + 1);

            for s in other.iter_mut() {
                iters.push(s.list()?.amortized_iter())
            }
            let mut first_iter: Box<dyn PolarsIterator<Item = Option<Series>>> = ca.into_iter();
            let mut builder = get_list_builder(
                &inner_super_type,
                ca.get_values_size() + vals_size_other + 1,
                length,
                ca.name().clone(),
            )?;

            for _ in 0..ca.len() {
                let mut acc = match first_iter.next().unwrap() {
                    Some(s) => s,
                    None => {
                        builder.append_null();
                        // make sure that the iterators advance before we continue
                        for it in &mut iters {
                            it.next().unwrap();
                        }
                        continue;
                    },
                };

                let mut has_nulls = false;
                for it in &mut iters {
                    match it.next().unwrap() {
                        Some(s) => {
                            if !has_nulls {
                                acc.append(s.as_ref())?;
                            }
                        },
                        None => {
                            has_nulls = true;
                        },
                    }
                }
                if has_nulls {
                    builder.append_null();
                    continue;
                }

                match inner_super_type {
                    // structs don't have chunks, so we must first rechunk the underlying series
                    #[cfg(feature = "dtype-struct")]
                    DataType::Struct(_) => acc = acc.rechunk(),
                    // nothing
                    _ => {},
                }
                builder.append_series(&acc).unwrap();
            }
            builder.finish()
        };
        Ok(out)
    }
}

impl ListNameSpaceImpl for ListChunked {}

#[cfg(feature = "list_gather")]
fn take_series(s: &Series, idx: Series, null_on_oob: bool) -> PolarsResult<Series> {
    let len = s.len();
    let idx = cast_index(idx, len, null_on_oob)?;
    let idx = idx.idx().unwrap();
    s.take(idx)
}

#[cfg(feature = "list_gather")]
fn cast_signed_index_ca<T: PolarsNumericType>(idx: &ChunkedArray<T>, len: usize) -> Series
where
    T::Native: Copy + PartialOrd + PartialEq + NumCast + Signed + Zero,
{
    idx.iter()
        .map(|opt_idx| opt_idx.and_then(|idx| idx.negative_to_usize(len).map(|idx| idx as IdxSize)))
        .collect::<IdxCa>()
        .into_series()
}

#[cfg(feature = "list_gather")]
fn cast_unsigned_index_ca<T: PolarsNumericType>(idx: &ChunkedArray<T>, len: usize) -> Series
where
    T::Native: Copy + PartialOrd + ToPrimitive,
{
    idx.iter()
        .map(|opt_idx| {
            opt_idx.and_then(|idx| {
                let idx = idx.to_usize().unwrap();
                if idx >= len {
                    None
                } else {
                    Some(idx as IdxSize)
                }
            })
        })
        .collect::<IdxCa>()
        .into_series()
}

#[cfg(feature = "list_gather")]
fn cast_index(idx: Series, len: usize, null_on_oob: bool) -> PolarsResult<Series> {
    let idx_null_count = idx.null_count();
    use DataType::*;
    let out = match idx.dtype() {
        #[cfg(feature = "big_idx")]
        UInt32 => {
            if null_on_oob {
                let a = idx.u32().unwrap();
                cast_unsigned_index_ca(a, len)
            } else {
                idx.cast(&IDX_DTYPE).unwrap()
            }
        },
        #[cfg(feature = "big_idx")]
        UInt64 => {
            if null_on_oob {
                let a = idx.u64().unwrap();
                cast_unsigned_index_ca(a, len)
            } else {
                idx
            }
        },
        #[cfg(not(feature = "big_idx"))]
        UInt64 => {
            if null_on_oob {
                let a = idx.u64().unwrap();
                cast_unsigned_index_ca(a, len)
            } else {
                idx.cast(&IDX_DTYPE).unwrap()
            }
        },
        #[cfg(not(feature = "big_idx"))]
        UInt32 => {
            if null_on_oob {
                let a = idx.u32().unwrap();
                cast_unsigned_index_ca(a, len)
            } else {
                idx
            }
        },
        dt if dt.is_unsigned_integer() => idx.cast(&IDX_DTYPE).unwrap(),
        Int8 => {
            let a = idx.i8().unwrap();
            cast_signed_index_ca(a, len)
        },
        Int16 => {
            let a = idx.i16().unwrap();
            cast_signed_index_ca(a, len)
        },
        Int32 => {
            let a = idx.i32().unwrap();
            cast_signed_index_ca(a, len)
        },
        Int64 => {
            let a = idx.i64().unwrap();
            cast_signed_index_ca(a, len)
        },
        _ => {
            unreachable!()
        },
    };
    polars_ensure!(
        out.null_count() == idx_null_count || null_on_oob,
        OutOfBounds: "gather indices are out of bounds"
    );
    Ok(out)
}

// Was using these to debug but I see no harm in having more unit tests, in fact we should probably have more here
#[cfg(test)]
mod tests {
    use polars_core::prelude::{IntoSeries, LambdaExpression, ListChunked, Series, SortOptions};

    use crate::chunked_array::ListNameSpaceImpl;

    #[test]
    fn test_empty_transform() {
        let start_array =
            ListChunked::from_iter([Series::from_iter(["key1==value1", "key2===value2"])])
                .into_series();
        let empty_lambda = LambdaExpression::StaticStr("meep".into());

        let result = start_array
            .as_list()
            .lst_transform(empty_lambda.into(), false)
            .expect("Could not evaluate lambda");
        let res = result.get_as_series(0).unwrap();

        assert_eq!(res.len(), 2);

        assert_eq!(res.str().unwrap().get(0).unwrap(), "meep");
        assert_eq!(res.str().unwrap().get(1).unwrap(), "meep");
    }

    #[test]
    fn test_lambda_length() {
        let start_array =
            ListChunked::from_iter([Series::from_iter(["key1==value1", "key2===value2"])])
                .into_series();
        let length_lambda = LambdaExpression::Length(Box::new(LambdaExpression::Variable(0)));

        let result = start_array
            .as_list()
            .lst_transform(length_lambda.into(), false)
            .expect("Could not evaluate lambda");
        let res = result.get_as_series(0).unwrap();
        assert_eq!(res.i32().unwrap().get(0).unwrap(), 12);
        assert_eq!(res.i32().unwrap().get(1).unwrap(), 13);
    }

    #[test]
    fn test_lambda_substring() {
        let start_array = Series::from_iter(vec!["key1==value1", "key2===value2"]);
        let substring_lambda = LambdaExpression::Substring(
            Box::new(LambdaExpression::Variable(0)),
            Box::new(LambdaExpression::Int32(4)),
            Box::new(LambdaExpression::Int32(2)),
        );

        let result = start_array
            .as_list()
            .lst_transform(substring_lambda.into(), false)
            .expect("Could not evaluate lambda");
        let res = result.get_as_series(0).unwrap();

        // Substring should start at the index of the element in the array + 2
        assert_eq!(res.str().unwrap().get(0).unwrap(), "1=");
        assert_eq!(res.str().unwrap().get(1).unwrap(), "2=");
    }

    #[test]
    fn test_lambda_with_index() {
        let start_array =
            ListChunked::from_iter([Series::from_iter(vec!["key1==value1", "key2===value2"])])
                .into_series();
        let substring_lambda = LambdaExpression::Substring(
            Box::new(LambdaExpression::Variable(0)),
            Box::new(LambdaExpression::Add(
                Box::new(LambdaExpression::Variable(1)),
                Box::new(LambdaExpression::Int32(2)),
            )),
            Box::new(LambdaExpression::Int32(2)),
        );

        let result = start_array
            .as_list()
            .lst_transform(substring_lambda.into(), true)
            .expect("Could not evaluate lambda");
        let res = result.get_as_series(0).unwrap();

        assert_eq!(res.str().unwrap().get(0).unwrap(), "y1");
        assert_eq!(res.str().unwrap().get(1).unwrap(), "2=");
    }

    #[test]
    fn test_lambda_casewhen() {
        let start_array = ListChunked::from_iter([Series::from_iter(vec![
            "key1==value1",
            "key2===u",
            "key3==value3whichisverylong",
        ])])
        .into_series();

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

        let result = start_array
            .as_list()
            .lst_transform(casewhen_lambda.into(), false)
            .expect("Could not evaluate lambda");
        let res = result.get_as_series(0).unwrap();

        assert_eq!(res.str().unwrap().get(0).unwrap(), "key1==value1");
        assert_eq!(res.str().unwrap().get(1).unwrap(), "nope");
        assert_eq!(
            res.str().unwrap().get(2).unwrap(),
            "key3==value3whichisverylong"
        );
    }

    #[test]
    fn test_array_sort() {
        let start_array = ListChunked::from_iter([Series::from_iter(vec![
            Some(1i32),
            None,
            Some(2),
            Some(3),
            None,
            Some(4),
            Some(5),
        ])])
        .into_series();

        let ascending_lambda = LambdaExpression::CaseWhen(
            vec![
                (
                    LambdaExpression::IsNull(LambdaExpression::Variable(0).into()),
                    LambdaExpression::CaseWhen(
                        vec![(
                            LambdaExpression::IsNull(LambdaExpression::Variable(1).into()),
                            LambdaExpression::Int32(0),
                        )],
                        LambdaExpression::Int32(1).into(),
                    ),
                ),
                (
                    LambdaExpression::IsNull(LambdaExpression::Variable(1).into()),
                    LambdaExpression::Int32(-1),
                ),
                (
                    LambdaExpression::GreaterThan(
                        LambdaExpression::Variable(0).into(),
                        LambdaExpression::Variable(1).into(),
                    ),
                    LambdaExpression::Int32(1),
                ),
                (
                    LambdaExpression::LessThan(
                        LambdaExpression::Variable(0).into(),
                        LambdaExpression::Variable(1).into(),
                    ),
                    LambdaExpression::Int32(-1),
                ),
            ],
            LambdaExpression::Int32(0).into(),
        );

        let result = start_array
            .as_list()
            .lst_sort_by_func(SortOptions::new().with_nulls_last(true), ascending_lambda.into())
            .expect("Could not evaluate lambda");
        let res = result.get_as_series(0).unwrap();

        assert_eq!(
            res,
            Series::from_iter(vec![
                Some(1i32),
                Some(2),
                Some(3),
                Some(4),
                Some(5),
                None,
                None
            ])
        );
    }
}
