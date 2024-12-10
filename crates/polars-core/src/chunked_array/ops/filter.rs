use polars_compute::filter::filter as filter_fn;

#[cfg(feature = "object")]
use crate::chunked_array::object::builder::ObjectChunkedBuilder;
use crate::prelude::*;

macro_rules! check_filter_len {
    ($self:expr, $filter:expr) => {{
        polars_ensure!(
            $self.len() == $filter.len(),
            ShapeMismatch: "filter's length: {} differs from that of the series: {}",
            $filter.len(), $self.len()
        )
    }};
}

/// Checks if a lambda expression is a length comparison
fn is_length_comparison(lambda: &LambdaExpression) -> Option<(usize, bool)> {
    match lambda {
        LambdaExpression::GreaterThan(left, right) => match (left.as_ref(), right.as_ref()) {
            (LambdaExpression::Length(var), LambdaExpression::Int64(threshold))
                if matches!(&**var, LambdaExpression::Variable(_)) =>
            {
                Some((*threshold as usize, true))
            },
            _ => None,
        },
        LambdaExpression::LessThan(left, right) => match (left.as_ref(), right.as_ref()) {
            (LambdaExpression::Length(var), LambdaExpression::Int64(threshold))
                if matches!(&**var, LambdaExpression::Variable(_)) =>
            {
                Some((*threshold as usize, false))
            },
            _ => None,
        },
        _ => None,
    }
}

impl<T> ChunkFilter<T> for ChunkedArray<T>
where
    T: PolarsDataType<HasViews = FalseT, IsObject = FalseT>,
{
    fn filter(&self, filter: &BooleanChunked) -> PolarsResult<ChunkedArray<T>> {
        // Broadcast.
        if filter.len() == 1 {
            return match filter.get(0) {
                Some(true) => Ok(self.clone()),
                _ => Ok(self.clear()),
            };
        }
        check_filter_len!(self, filter);
        Ok(unsafe {
            arity::binary_unchecked_same_type(
                self,
                filter,
                |left, mask| filter_fn(left, mask),
                true,
                true,
            )
        })
    }

    fn filter_with_func(&self, _lambda: &LambdaExpression) -> PolarsResult<ChunkedArray<T>> {
        polars_bail!(
            ComputeError: "filter_with_func not implemented for type: {:?}", T::get_dtype()
        )
    }
}

impl<T> ChunkedArray<T>
where
    T: PolarsNumericType,
{
    pub fn filter_with_func(&self, lambda: &LambdaExpression) -> PolarsResult<ChunkedArray<T>> {
        let mut keep_nulls= false; // filter out nulls by default

        // in spark, if expression is isnull() then it expects to keep the nulls when it encounters one.
        if let LambdaExpression::IsNull(_) = lambda {
            keep_nulls = true;
        }

        // Evaluate each element using eval_numeric
        let mask = self.iter().map(|opt_val| {
            match opt_val {
                Some(val) => {
                    match lambda.eval_numeric::<T>(&[&val]) {
                        AnyValue::Boolean(b) => b,
                        _ => panic!("Lambda must return boolean values"),
                    }
                },
                None => keep_nulls,
            }
        });

        let bool_mask = BooleanChunked::from_iter_values(self.name().clone(), mask);
        self.filter(&bool_mask)
    }
}

impl ChunkedArray<ListType> {
    pub fn filter_with_func(
        &self,
        lambda: &LambdaExpression,
    ) -> PolarsResult<ChunkedArray<ListType>> {
        if let Some((threshold, is_greater)) = is_length_comparison(lambda) {
            let mask = self.iter().map(|opt_val| match opt_val {
                Some(arr) => {
                    let len = arr.len();
                    if is_greater {
                        len > threshold
                    } else {
                        len < threshold
                    }
                },
                None => false,
            });
            let bool_mask = BooleanChunked::from_iter_values(self.name().clone(), mask);
            return self.filter(&bool_mask);
        }

        // Evaluate each array element using eval_array
        let mask = self.iter().map(|opt_val| match opt_val {
            Some(arr) => match lambda.eval_array(&[arr.as_ref()]) {
                AnyValue::Boolean(b) => b,
                // For list operations, we should get a List back containing booleans
                AnyValue::List(series) => {
                    // Convert the series to a boolean indicating if the condition is true for any/all elements
                    match series.bool() {
                        Ok(bool_arr) => bool_arr.any(),
                        _ => panic!("Lambda expression did not return boolean values"),
                    }
                },
                _ => panic!("Lambda must return boolean values or list of boolean values"),
            },
            None => false,
        });
        let bool_mask = BooleanChunked::from_iter_values(self.name().clone(), mask);
        self.filter(&bool_mask)
    }
}

// impl ChunkFilter<BooleanType> for BooleanChunked {
//     fn filter(&self, filter: &BooleanChunked) -> PolarsResult<ChunkedArray<BooleanType>> {
//         // Broadcast.
//         if filter.len() == 1 {
//             return match filter.get(0) {
//                 Some(true) => Ok(self.clone()),
//                 _ => Ok(self.clear()),
//             };
//         }
//         check_filter_len!(self, filter);
//         Ok(unsafe {
//             arity::binary_unchecked_same_type(
//                 self,
//                 filter,
//                 |left, mask| filter_fn(left, mask),
//                 true,
//                 true,
//             )
//         })
//     }
// }

impl ChunkFilter<StringType> for StringChunked {
    fn filter(&self, filter: &BooleanChunked) -> PolarsResult<ChunkedArray<StringType>> {
        let out = self.as_binary().filter(filter)?;
        unsafe { Ok(out.to_string_unchecked()) }
    }

    fn filter_with_func(
        &self,
        lambda: &LambdaExpression,
    ) -> PolarsResult<ChunkedArray<StringType>> {
        // Check if this is a length comparison
        if let Some((threshold, is_greater)) = is_length_comparison(lambda) {
            // Use optimized length-based filtering
            let mask = self.iter().map(|opt_val| match opt_val {
                Some(val) => {
                    let len = val.len();
                    if is_greater {
                        len > threshold
                    } else {
                        len < threshold
                    }
                },
                None => false,
            });

            let bool_mask = BooleanChunked::from_iter_values(self.name().clone(), mask);
            return self.filter(&bool_mask);
        }

        let mask = self.iter().map(|opt_val| match opt_val {
            Some(val) => match lambda.eval_slice(&[val.as_bytes()]) {
                AnyValue::Boolean(b) => b,
                _ => panic!("Lambda must return boolean values"),
            },
            None => false,
        });

        let bool_mask = BooleanChunked::from_iter_values(self.name().clone(), mask);
        self.filter(&bool_mask)
    }
}

impl ChunkFilter<BinaryType> for BinaryChunked {
    fn filter(&self, filter: &BooleanChunked) -> PolarsResult<ChunkedArray<BinaryType>> {
        // Broadcast.
        if filter.len() == 1 {
            return match filter.get(0) {
                Some(true) => Ok(self.clone()),
                _ => Ok(self.clear()),
            };
        }
        check_filter_len!(self, filter);
        Ok(unsafe {
            arity::binary_unchecked_same_type(
                self,
                filter,
                |left, mask| filter_fn(left, mask),
                true,
                true,
            )
        })
    }

    fn filter_with_func(
        &self,
        lambda: &LambdaExpression,
    ) -> PolarsResult<ChunkedArray<BinaryType>> {
        // Check if this is a length comparison
        if let Some((threshold, is_greater)) = is_length_comparison(lambda) {
            // Use optimized length-based filtering
            let mask = self.iter().map(|opt_val| match opt_val {
                Some(val) => {
                    let len = val.len();
                    if is_greater {
                        len > threshold
                    } else {
                        len < threshold
                    }
                },
                None => false,
            });

            let bool_mask = BooleanChunked::from_iter_values(self.name().clone(), mask);
            return self.filter(&bool_mask);
        }

        let mask = self.iter().map(|opt_val| match opt_val {
            Some(val) => match lambda.eval_slice(&[val]) {
                AnyValue::Boolean(b) => b,
                _ => panic!("Lambda must return boolean values"),
            },
            None => false,
        });

        let bool_mask = BooleanChunked::from_iter_values(self.name().clone(), mask);
        self.filter(&bool_mask)
    }
}

// impl ChunkFilter<BinaryOffsetType> for BinaryOffsetChunked {
//     fn filter(&self, filter: &BooleanChunked) -> PolarsResult<BinaryOffsetChunked> {
//         // Broadcast.
//         if filter.len() == 1 {
//             return match filter.get(0) {
//                 Some(true) => Ok(self.clone()),
//                 _ => Ok(self.clear()),
//             };
//         }
//         check_filter_len!(self, filter);
//         Ok(unsafe {
//             arity::binary_unchecked_same_type(
//                 self,
//                 filter,
//                 |left, mask| filter_fn(left, mask),
//                 true,
//                 true,
//             )
//         })
//     }
// }
//
// impl ChunkFilter<ListType> for ListChunked {
//     fn filter(&self, filter: &BooleanChunked) -> PolarsResult<ListChunked> {
//         // Broadcast.
//         if filter.len() == 1 {
//             return match filter.get(0) {
//                 Some(true) => Ok(self.clone()),
//                 _ => Ok(self.clear()),
//             };
//         }
//         check_filter_len!(self, filter);
//         Ok(unsafe {
//             arity::binary_unchecked_same_type(
//                 self,
//                 filter,
//                 |left, mask| filter_fn(left, mask),
//                 true,
//                 true,
//             )
//         })
//     }
// }
//
// #[cfg(feature = "dtype-struct")]
// impl ChunkFilter<StructType> for StructChunked {
//     fn filter(&self, filter: &BooleanChunked) -> PolarsResult<ChunkedArray<StructType>>
//     where
//         Self: Sized
//     {
//         if filter.len() == 1 {
//             return match filter.get(0) {
//                 Some(true) => Ok(self.clone()),
//                 _ => Ok(self.clear())
//             }
//         }
//     }
// }
//
// #[cfg(feature = "dtype-array")]
// impl ChunkFilter<FixedSizeListType> for ArrayChunked {
//     fn filter(&self, filter: &BooleanChunked) -> PolarsResult<ArrayChunked> {
//         // Broadcast.
//         if filter.len() == 1 {
//             return match filter.get(0) {
//                 Some(true) => Ok(self.clone()),
//                 _ => Ok(ArrayChunked::from_chunk_iter(
//                     self.name(),
//                     [FixedSizeListArray::new_empty(
//                         self.dtype().to_arrow(CompatLevel::newest()),
//                     )],
//                 )),
//             };
//         }
//         check_filter_len!(self, filter);
//         Ok(unsafe {
//             arity::binary_unchecked_same_type(
//                 self,
//                 filter,
//                 |left, mask| filter_fn(left, mask),
//                 true,
//                 true,
//             )
//         })
//     }
// }

#[cfg(feature = "object")]
impl<T> ChunkFilter<ObjectType<T>> for ObjectChunked<T>
where
    T: PolarsObject,
{
    fn filter(&self, filter: &BooleanChunked) -> PolarsResult<ChunkedArray<ObjectType<T>>>
    where
        Self: Sized,
    {
        // Broadcast.
        if filter.len() == 1 {
            return match filter.get(0) {
                Some(true) => Ok(self.clone()),
                _ => Ok(ObjectChunked::new_empty(self.name().clone())),
            };
        }
        check_filter_len!(self, filter);
        let chunks = self.downcast_iter().collect::<Vec<_>>();
        let mut builder = ObjectChunkedBuilder::<T>::new(self.name().clone(), self.len());
        for (idx, mask) in filter.into_iter().enumerate() {
            if mask.unwrap_or(false) {
                let (chunk_idx, idx) = self.index_to_chunked_index(idx);
                unsafe {
                    let arr = chunks.get_unchecked(chunk_idx);
                    match arr.is_null(idx) {
                        true => builder.append_null(),
                        false => {
                            let v = arr.value(idx);
                            builder.append_value(v.clone())
                        },
                    }
                }
            }
        }
        Ok(builder.finish())
    }

    fn filter_with_func(
        &self,
        _lambda: &LambdaExpression,
    ) -> PolarsResult<ChunkedArray<ObjectType<T>>> {
        polars_bail!(ComputeError: "filter_with_func not implemented for Object type")
    }
}
