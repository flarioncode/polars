
use arrow::array::Array;
use num_traits::NumCast;

use super::{BinaryChunked, BinaryChunkedBuilder, BooleanChunked, BooleanChunkedBuilder, ChunkTransform, ChunkedArray, ChunkedBuilder, Float32Type, Float64Type, Int16Type, Int32Type, Int64Type, Int8Type, ListBinaryChunkedBuilder, ListBooleanChunkedBuilder, ListChunked, ListPrimitiveChunkedBuilder, ListStringChunkedBuilder, PolarsNumericType, PrimitiveChunkedBuilder, SeriesTrait, StringChunked, UInt16Type, UInt32Type, UInt64Type, UInt8Type};
use crate::prelude::{AnyValue, DataType, ListBuilderTrait, StringChunkedBuilder};
use crate::series::implementations::SeriesWrap;
use crate::series::{IntoSeries, Series};

const TRANSFORMED_COLUMN_NAME: &'static str = "Transformed";

fn create_int_series<T: PolarsNumericType>(vec: Vec<AnyValue>, cap: usize) -> Series 
where 
    <T as PolarsNumericType>::Native: NumCast,
    SeriesWrap<ChunkedArray<T>>: SeriesTrait,
{
    let mut builder = PrimitiveChunkedBuilder::<T>::new(TRANSFORMED_COLUMN_NAME.into(), cap);
    for value in vec {
        match value {
            AnyValue::Null => builder.append_null(),
            _ => {
                let value = value.extract().expect("TODO: write message here");
                builder.append_value(value);
            }
        }
    }
    builder.finish().into_series()
}

fn append_primitive_list_to_series<T: PolarsNumericType>(builder: &mut ListPrimitiveChunkedBuilder<T>, series: Series, datatype: &DataType) {
    let array: &ChunkedArray<T> = series.as_ref().as_any().downcast_ref::<ChunkedArray<T>>().unwrap();
    builder.append_iter(array.iter());
}

fn create_primitive_list_series<T: PolarsNumericType>(vec: Vec<AnyValue>, list_type: DataType, cap: usize) -> Series {
    // TODO: figure out value capasity
    let mut builder = ListPrimitiveChunkedBuilder::<T>::new(TRANSFORMED_COLUMN_NAME.into(), cap, 5, list_type.clone());
    for value in vec {
        match value {
            AnyValue::List(lst_series) => {
                append_primitive_list_to_series(&mut builder, lst_series, &list_type)
            }
            AnyValue::Null => builder.append_null(),
            _ => unreachable!()
        }
    }
    builder.finish().into_series()
}

fn create_list_series(vec: Vec<AnyValue>, list_type: DataType, cap: usize) -> Series {
    match list_type {
        DataType::UInt8 => create_primitive_list_series::<UInt8Type>(vec, list_type, cap),
        DataType::UInt16 => create_primitive_list_series::<UInt16Type>(vec, list_type, cap),
        DataType::UInt32 => create_primitive_list_series::<UInt32Type>(vec, list_type, cap),
        DataType::UInt64 => create_primitive_list_series::<UInt64Type>(vec, list_type, cap),
        DataType::Int8 => create_primitive_list_series::<Int8Type>(vec, list_type, cap),
        DataType::Int16 => create_primitive_list_series::<Int16Type>(vec, list_type, cap),
        DataType::Int32 => create_primitive_list_series::<Int32Type>(vec, list_type, cap),
        DataType::Int64 => create_primitive_list_series::<Int64Type>(vec, list_type, cap),
        DataType::Float32 => create_primitive_list_series::<Float32Type>(vec, list_type, cap),
        DataType::Float64 => create_primitive_list_series::<Float64Type>(vec, list_type, cap),
        DataType::Boolean => {
            let mut builder = ListBooleanChunkedBuilder::new(TRANSFORMED_COLUMN_NAME.into(), cap, 5);
            for value in vec {
                match value {
                    AnyValue::List(lst) => {
                        builder.append(lst.bool().unwrap());
                    }
                    AnyValue::Null => {
                        builder.append_null();
                    }
                    _ => unreachable!()
                }
            }
            return builder.finish().into_series()
        },
        DataType::String => {
            let mut builder = ListStringChunkedBuilder::new(TRANSFORMED_COLUMN_NAME.into(), cap, 5);
            for value in vec {
                match value {
                    AnyValue::List(lst) => {
                        builder.append(lst.str().unwrap());
                    }
                    AnyValue::Null => {
                        builder.append_null();
                    }
                    _ => unreachable!()
                }
            }
            return builder.finish().into_series()
        },
        DataType::Binary => {
            let mut builder = ListBinaryChunkedBuilder::new(TRANSFORMED_COLUMN_NAME.into(), cap, 5);
            for value in vec {
                match value {
                    AnyValue::List(lst) => {
                        builder.append(lst.binary().unwrap());
                    }
                    AnyValue::Null => {
                        builder.append_null();
                    }
                    _ => unreachable!()
                }
            }
            return builder.finish().into_series()
        },
        DataType::List(_) => unreachable!(), // TODO: This case means triple nested lists, this theoreticaly could be reached, but there is no practical reason implement it right now.
        _ => unreachable!()
    }
}

// This methods right now don't suppory all possible values. Only that possible as the result of LambdaExpression
fn create_series(vec: Vec<AnyValue>) -> Series {
    let datatype = vec[0].dtype();
    let len = vec.len();
    match datatype {
        DataType::UInt8 | DataType::UInt16 => create_int_series::<UInt32Type>(vec, len), // There is no implemenation Series for UInt8 and UInt16
        DataType::UInt32 => create_int_series::<UInt32Type>(vec, len),
        DataType::UInt64 => create_int_series::<UInt64Type>(vec, len),
        DataType::Int8 | DataType::Int16 => create_int_series::<Int32Type>(vec, len), // There is no implemenation Series for Int8 and Int16
        DataType::Int32 => create_int_series::<Int32Type>(vec, len),
        DataType::Int64 => create_int_series::<Int64Type>(vec, len),
        DataType::Boolean => {
            let mut builder = BooleanChunkedBuilder::new(TRANSFORMED_COLUMN_NAME.into(), len);
            for value in vec {
                match value {
                    AnyValue::Boolean(value) => builder.append_value(value),
                    AnyValue::Null => builder.append_null(),
                    _ => panic!("Expected Boolean but got {:?}", value.dtype())
                };
            }
            builder.finish().into_series()
        },
        DataType::Float32 => {
            let mut builder = PrimitiveChunkedBuilder::<Float32Type>::new(TRANSFORMED_COLUMN_NAME.into(), len);
            for value in vec {
                match value {
                    AnyValue::Float32(value) => builder.append_value(value),
                    AnyValue::Null => builder.append_null(),
                    _ => panic!("Expected Float32 but got {:?}", value.dtype())
                };
                
            }
            builder.finish().into_series()
        },
        DataType::Float64 => {
            let mut builder = PrimitiveChunkedBuilder::<Float64Type>::new(TRANSFORMED_COLUMN_NAME.into(), len);
            for value in vec {
                match value {
                    AnyValue::Float64(value) => builder.append_value(value),
                    AnyValue::Null => builder.append_null(),
                    _ => panic!("Expected Float64 but got {:?}", value.dtype())
                };
            }
            builder.finish().into_series()
        },
        DataType::String => {
            let mut builder = StringChunkedBuilder::new(TRANSFORMED_COLUMN_NAME.into(), len);
            for value in vec {
                match value {
                    AnyValue::String(value) => builder.append_value(value),
                    AnyValue::Null => builder.append_null(),
                    _ => panic!("Expected String but got {:?}", value.dtype())
                };
            }
            builder.finish().into_series()
        },
        DataType::Binary => {
            let mut builder = BinaryChunkedBuilder::new(TRANSFORMED_COLUMN_NAME.into(), len);
            for value in vec {
                match value {
                    AnyValue::Binary(value) => builder.append_value(value),
                    AnyValue::Null => builder.append_null(),
                    _ => panic!("Expected Binary but got {:?}", value.dtype())
                };
            }
            builder.finish().into_series()
        },
        DataType::List(data_type) => {
            create_list_series(vec, *data_type, len)
        },
        // If datatype is null that means that all elements is null
        DataType::Null => create_int_series::<Int32Type>(vec, len),
        _ => unreachable!(),
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
        if self.is_empty() {
            return Ok(self.clone().into_series());
        }

        let vec: Vec<AnyValue> = self
            .iter()
            .enumerate()
            .map(|(i, value): (usize, Option<T::Native>)| lambda.eval_any(&[value.into(), AnyValue::Int32(i as i32)]))
            .collect();

        Ok(create_series(vec))
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

        Ok(create_series(vec))
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

        Ok(create_series(vec))
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

        Ok(create_series(vec))
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

        Ok(create_series(vec))
    }
}