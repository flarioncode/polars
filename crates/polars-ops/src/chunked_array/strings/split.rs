use arrow::array::ValueSize;
#[cfg(feature = "dtype-struct")]
use arrow::array::{MutableArray, MutableUtf8Array};
use polars_core::chunked_array::ops::arity::binary_elementwise_for_each;
use polars_core::prelude::split::split_chars;
#[cfg(feature = "dtype-struct")]
use polars_core::prelude::split::splitn_chars;

use super::*;

#[cfg(feature = "dtype-struct")]
pub fn split_to_struct<'a, F, I>(
    ca: &'a StringChunked,
    by: &'a StringChunked,
    n: usize,
    op: F,
    keep_remainder: bool,
) -> PolarsResult<StructChunked>
where
    F: Fn(&'a str, &'a str) -> I,
    I: Iterator<Item = &'a str>,
{
    use polars_utils::format_pl_smallstr;

    let mut arrs = (0..n)
        .map(|_| MutableUtf8Array::<i64>::with_capacity(ca.len()))
        .collect::<Vec<_>>();

    if by.len() == 1 {
        if let Some(by) = by.get(0) {
            if by.is_empty() {
                ca.for_each(|opt_s| match opt_s {
                    None => {
                        for arr in &mut arrs {
                            arr.push_null()
                        }
                    },
                    Some(s) => {
                        let mut arr_iter = arrs.iter_mut();
                        splitn_chars(s, n, keep_remainder)
                            .zip(&mut arr_iter)
                            .for_each(|(splitted, arr)| arr.push(Some(splitted)));
                        // fill the remaining with null
                        for arr in arr_iter {
                            arr.push_null()
                        }
                    },
                });
            } else {
                ca.for_each(|opt_s| match opt_s {
                    None => {
                        for arr in &mut arrs {
                            arr.push_null()
                        }
                    },
                    Some(s) => {
                        let mut arr_iter = arrs.iter_mut();
                        op(s, by)
                            .zip(&mut arr_iter)
                            .for_each(|(splitted, arr)| arr.push(Some(splitted)));
                        // fill the remaining with null
                        for arr in arr_iter {
                            arr.push_null()
                        }
                    },
                });
            }
        } else {
            for arr in &mut arrs {
                arr.push_null()
            }
        }
    } else {
        binary_elementwise_for_each(ca, by, |opt_s, opt_by| match (opt_s, opt_by) {
            (Some(s), Some(by)) => {
                let mut arr_iter = arrs.iter_mut();
                if by.is_empty() {
                    splitn_chars(s, n, keep_remainder)
                        .zip(&mut arr_iter)
                        .for_each(|(splitted, arr)| arr.push(Some(splitted)));
                } else {
                    op(s, by)
                        .zip(&mut arr_iter)
                        .for_each(|(splitted, arr)| arr.push(Some(splitted)));
                };
                // fill the remaining with null
                for arr in arr_iter {
                    arr.push_null()
                }
            },
            _ => {
                for arr in &mut arrs {
                    arr.push_null()
                }
            },
        })
    }

    let fields = arrs
        .into_iter()
        .enumerate()
        .map(|(i, mut arr)| {
            Series::try_from((format_pl_smallstr!("field_{i}"), arr.as_box())).unwrap()
        })
        .collect::<Vec<_>>();

    StructChunked::from_series(ca.name().clone(), &fields)
}

pub fn split_helper<'a, F, I>(ca: &'a StringChunked, by: &'a StringChunked, op: F) -> ListChunked
where
    F: Fn(&'a str, &'a str) -> I,
    I: Iterator<Item = &'a str>,
{
    if by.len() == 1 {
        if let Some(by) = by.get(0) {
            let mut builder =
                ListStringChunkedBuilder::new(ca.name().clone(), ca.len(), ca.get_values_size());

            if by.is_empty() {
                ca.for_each(|opt_s| match opt_s {
                    Some(s) => builder.append_values_iter(split_chars(s)),
                    _ => builder.append_null(),
                });
            } else {
                ca.for_each(|opt_s| match opt_s {
                    Some(s) => builder.append_values_iter(op(s, by)),
                    _ => builder.append_null(),
                });
            }
            builder.finish()
        } else {
            ListChunked::full_null_with_dtype(ca.name().clone(), ca.len(), &DataType::String)
        }
    } else {
        let mut builder =
            ListStringChunkedBuilder::new(ca.name().clone(), ca.len(), ca.get_values_size());

        binary_elementwise_for_each(ca, by, |opt_s, opt_by| match (opt_s, opt_by) {
            (Some(s), Some(by)) => {
                if by.is_empty() {
                    builder.append_values_iter(split_chars(s))
                } else {
                    builder.append_values_iter(op(s, by))
                }
            },
            _ => builder.append_null(),
        });

        builder.finish()
    }
}
