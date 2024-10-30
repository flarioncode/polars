
use arrow::array::Array;

use super::{BinaryChunked, BooleanChunked, ChunkTransform, ChunkedArray, ListChunked, PolarsNumericType, SeriesTrait, StringChunked};
use crate::prelude::AnyValue;
use crate::series::implementations::SeriesWrap;
use crate::series::{IntoSeries, Series};

impl<T: PolarsNumericType + 'static> ChunkTransform for ChunkedArray<T>
where
    SeriesWrap<ChunkedArray<T>>: SeriesTrait,
{
    fn transform(&self, lambda: &super::LambdaExpression) -> polars_error::PolarsResult<Series>
    where
        Self: Sized,
    {
        if self.is_empty() {
            return Ok(self.clone().into_series());
        }

        let vec: Vec<AnyValue> = self
            .iter()
            .enumerate()
            .map(|(i, value): (usize, Option<T::Native>)| lambda.eval_any(&[value.into(), AnyValue::Int32(i as i32)]))
            .collect();

        Series::from_any_values("".into(), &vec, true)
    }
}


impl ChunkTransform for StringChunked {
    fn transform(&self, lambda: &super::LambdaExpression) -> polars_error::PolarsResult<Series> {
        if self.is_empty() {
            return Ok(self.clone().into_series());
        }

        let vec: Vec<AnyValue> = self
            .iter()
            .enumerate()
            .map(|(i, value)| lambda.eval_any(&[value.into(), AnyValue::Int32(i as i32)]))
            .collect();

        Series::from_any_values("".into(), &vec, true)
    }
}

impl ChunkTransform for BinaryChunked {
    fn transform(&self, lambda: &crate::chunked_array::LambdaExpression) -> polars_error::PolarsResult<Series> {
        if self.is_empty() {
            return Ok(self.clone().into_series());
        }

        let vec: Vec<AnyValue> = self
            .iter()
            .enumerate()
            .map(|(i, value)| lambda.eval_any(&[value.into(), AnyValue::Int32(i as i32)]))
            .collect();

        Series::from_any_values("".into(), &vec, true)
    }
}

fn array_to_any(value: Option<Box<dyn Array>>) -> AnyValue<'static> {
    match value {
        Some(value) => {
            let series = Series::from_arrow("".into(), value)
                .expect("could not convert array to series");
            AnyValue::List(series)
        },
        None => AnyValue::Null,
    }
}

impl ChunkTransform for ListChunked {
    fn transform(&self, lambda: &crate::chunked_array::LambdaExpression) -> polars_error::PolarsResult<Series> {
        if self.is_empty() {
            return Ok(self.clone().into_series());
        }

        let vec: Vec<AnyValue> = self
            .iter()
            .enumerate()
            .map(|(i, value)| lambda.eval_any(&[array_to_any(value), AnyValue::Int32(i as i32)]))
            .collect();

        Series::from_any_values("".into(), &vec, true)
    }
}


impl ChunkTransform for BooleanChunked {
    fn transform(&self, lambda: &crate::chunked_array::LambdaExpression) -> polars_error::PolarsResult<Series> {
        if self.is_empty() {
            return Ok(self.clone().into_series());
        }

        let vec: Vec<AnyValue> = self
            .iter()
            .enumerate()
            .map(|(i, value)| lambda.eval_any(&[value.into(), AnyValue::Int32(i as i32)]))
            .collect();

        Series::from_any_values("".into(), &vec, true)
    }
}