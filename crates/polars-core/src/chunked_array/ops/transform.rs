use arrow::array::Array;
use polars_utils::pl_str::PlSmallStr;

use super::{
    BinaryChunked, BooleanChunked, ChunkTransform, ChunkedArray, ListChunked, PolarsNumericType,
    SeriesTrait, StringChunked,
};
use crate::prelude::{AnyValue, ChunkCast, DataType};
use crate::series::implementations::SeriesWrap;
use crate::series::{IntoSeries, Series};

macro_rules! impl_common_early_return {
    ($self:ident, $lambda:ident) => {
        if $self.is_empty() {
            if let Ok(out_type) = $lambda.return_type($self.dtype()) {
                if $self.dtype() != &out_type {
                    return $self.cast(&out_type);
                }
            }

            return Ok($self.clone().into_series());
        }
    }
}

impl<T: PolarsNumericType + 'static> ChunkTransform for ChunkedArray<T>
where
    SeriesWrap<ChunkedArray<T>>: SeriesTrait,
{
    fn transform(&self, lambda: &super::LambdaExpression) -> polars_error::PolarsResult<Series>
    where
        Self: Sized,
    {
        impl_common_early_return!(self, lambda);

        let vec: Vec<AnyValue> = self
            .iter()
            .enumerate()
            .map(|(i, value): (usize, Option<T::Native>)| {
                lambda.eval_any(&[value.into(), AnyValue::Int32(i as i32)])
            })
            .collect();

        let return_type = lambda.return_type(self.dtype())?;
        Series::from_any_values_and_dtype(PlSmallStr::EMPTY, &vec, &return_type, true)
    }
}

impl ChunkTransform for StringChunked {
    fn transform(&self, lambda: &super::LambdaExpression) -> polars_error::PolarsResult<Series> {
        impl_common_early_return!(self, lambda);

        let vec: Vec<AnyValue> = self
            .iter()
            .enumerate()
            .map(|(i, value)| lambda.eval_any(&[value.into(), AnyValue::Int32(i as i32)]))
            .collect();

        Series::from_any_values_and_dtype(
            PlSmallStr::EMPTY,
            &vec,
            &lambda.return_type(self.dtype())?,
            true,
        )
    }
}

impl ChunkTransform for BinaryChunked {
    fn transform(
        &self,
        lambda: &crate::chunked_array::LambdaExpression,
    ) -> polars_error::PolarsResult<Series> {
        impl_common_early_return!(self, lambda);

        let vec: Vec<AnyValue> = self
            .iter()
            .enumerate()
            .map(|(i, value)| lambda.eval_any(&[value.into(), AnyValue::Int32(i as i32)]))
            .collect();

        Series::from_any_values_and_dtype(
            PlSmallStr::EMPTY,
            &vec,
            &lambda.return_type(self.dtype())?,
            true,
        )
    }
}

fn array_to_any(value: Option<Box<dyn Array>>) -> AnyValue<'static> {
    match value {
        Some(value) => {
            let series =
                Series::from_arrow("".into(), value).expect("could not convert array to series");
            AnyValue::List(series)
        },
        None => AnyValue::Null,
    }
}

impl ChunkTransform for ListChunked {
    fn transform(
        &self,
        lambda: &crate::chunked_array::LambdaExpression,
    ) -> polars_error::PolarsResult<Series> {
        impl_common_early_return!(self, lambda);

        let vec: Vec<AnyValue> = self
            .iter()
            .enumerate()
            .map(|(i, value)| lambda.eval_any(&[array_to_any(value), AnyValue::Int32(i as i32)]))
            .collect();

        Series::from_any_values_and_dtype(
            PlSmallStr::EMPTY,
            &vec,
            &lambda.return_type(self.dtype())?,
            true,
        )
    }
}

impl ChunkTransform for BooleanChunked {
    fn transform(
        &self,
        lambda: &crate::chunked_array::LambdaExpression,
    ) -> polars_error::PolarsResult<Series> {
        impl_common_early_return!(self, lambda);

        let vec: Vec<AnyValue> = self
            .iter()
            .enumerate()
            .map(|(i, value)| lambda.eval_any(&[value.into(), AnyValue::Int32(i as i32)]))
            .collect();

        Series::from_any_values_and_dtype(
            PlSmallStr::EMPTY,
            &vec,
            &lambda.return_type(self.dtype())?,
            true,
        )
    }
}
