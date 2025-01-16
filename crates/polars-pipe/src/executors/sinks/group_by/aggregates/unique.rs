use std::any::Any;

use arrow::legacy::error::PolarsResult;
use polars_core::datatypes::{AnyValue, BinaryChunked, DataType};
use polars_core::prelude::{IntoSeries, ListChunked, Series};
use polars_core::utils::Wrap;
use polars_expr::prelude::NormalizedFloat;
use polars_utils::aliases::{PlIndexSet};
use polars_utils::pl_str::PlSmallStr;
use polars_utils::unwrap::UnwrapUncheckedRelease;
use polars_utils::IdxSize;

use crate::executors::sinks::group_by::aggregates::AggregateFn;

#[allow(clippy::enum_variant_names)]
#[derive(Default)]
pub(crate) enum AggregateBufferSet {
    #[default]
    NullSet,
    BoolSet(PlIndexSet<bool>),
    ByteSet(PlIndexSet<i8>),
    ShortSet(PlIndexSet<i16>),
    IntSet(PlIndexSet<i32>),
    LongSet(PlIndexSet<i64>),
    FloatSet(PlIndexSet<NormalizedFloat<f32>>),
    DoubleSet(PlIndexSet<NormalizedFloat<f64>>),
    StringSet(PlIndexSet<String>),
    BinarySet(PlIndexSet<Vec<u8>>),
    ArraySet(PlIndexSet<Wrap<Series>>)
}

impl AggregateBufferSet {
    pub(crate) fn from_series(s: &Series, data_type: &DataType) -> PolarsResult<Self> {
        let s = if s.dtype() == data_type {
            s
        } else {
            &s.cast(data_type)?
        };

        Ok(match data_type {
            DataType::Boolean => Self::BoolSet(s.bool()?.into_iter().flatten().collect()),
            DataType::Int8 => Self::ByteSet(s.i8()?.into_iter().flatten().collect()),
            DataType::Int16 => Self::ShortSet(s.i16()?.into_iter().flatten().collect()),
            DataType::Int32 => Self::IntSet(s.i32()?.into_iter().flatten().collect()),
            DataType::Int64 => Self::LongSet(s.i64()?.into_iter().flatten().collect()),
            // Normalizing these floats ensures NaN are the same bitwise, then when the hasher calls to_bits(), same results are guaranteed
            DataType::Float32 => Self::FloatSet(
                s.f32()?
                    .to_canonical()
                    .into_iter()
                    .flatten()
                    .map(NormalizedFloat::new)
                    .collect(),
            ),
            DataType::Float64 => Self::DoubleSet(
                s.f64()?
                    .to_canonical()
                    .into_iter()
                    .flatten()
                    .map(NormalizedFloat::new)
                    .collect(),
            ),
            DataType::String => Self::StringSet(
                s.str()?
                    .into_iter()
                    .flatten()
                    .map(ToString::to_string)
                    .collect(),
            ),
            DataType::Binary => Self::BinarySet(
                s.binary()?
                    .into_iter()
                    .flatten()
                    .map(ToOwned::to_owned)
                    .collect(),
            ),
            DataType::List(_) => Self::ArraySet(
                s.list()?.into_iter().flatten().map(Wrap).collect()
            ),
            _ => unreachable!("Unsupported datatype for aggregate buffer set"),
        })
    }

    pub(crate) fn into_series(self, data_type: &DataType) -> Series {
        if self.is_empty() {
            return Series::new_empty(PlSmallStr::EMPTY, data_type);
        }

        let res_ser = match self {
            Self::NullSet => Series::new_null(PlSmallStr::EMPTY, 1),
            Self::BoolSet(val) => Series::from_iter(val),
            Self::ByteSet(val) => Series::from_iter(val),
            Self::ShortSet(val) => Series::from_iter(val),
            Self::IntSet(val) => Series::from_iter(val),
            Self::LongSet(val) => Series::from_iter(val),
            Self::FloatSet(val) => {
                Series::from_iter(val.into_iter().map(NormalizedFloat::into_inner))
            }
            Self::DoubleSet(val) => {
                Series::from_iter(val.into_iter().map(NormalizedFloat::into_inner))
            }
            Self::StringSet(val) => Series::from_iter(val),
            Self::BinarySet(val) => BinaryChunked::from_iter(val).into_series(),
            Self::ArraySet(val) => {
                ListChunked::from_iter(
                    val.into_iter().map(|x| x.0)
                )
                    .into_series()
            }
        };

        if res_ser.dtype() == data_type {
            res_ser
        } else {
            res_ser.cast(data_type).unwrap()
        }
    }

    fn extend_from(&mut self, other: &Self) {
        match other {
            AggregateBufferSet::NullSet => {}
            AggregateBufferSet::BoolSet(_) => {}
            AggregateBufferSet::ByteSet(_) => {}
            AggregateBufferSet::ShortSet(_) => {}
            AggregateBufferSet::IntSet(_) => {}
            AggregateBufferSet::LongSet(_) => {}
            AggregateBufferSet::FloatSet(_) => {}
            AggregateBufferSet::DoubleSet(_) => {}
            AggregateBufferSet::StringSet(_) => {}
            AggregateBufferSet::BinarySet(_) => {}
            AggregateBufferSet::ArraySet(_) => {}
        }
        match (self, other) {
            (AggregateBufferSet::NullSet, AggregateBufferSet::NullSet) => {}
            (AggregateBufferSet::BoolSet(lhs), AggregateBufferSet::BoolSet(rhs)) => {lhs.extend(rhs.iter())}
            (AggregateBufferSet::ByteSet(lhs), AggregateBufferSet::ByteSet(rhs)) => {lhs.extend(rhs.iter())}
            (AggregateBufferSet::ShortSet(lhs), AggregateBufferSet::ShortSet(rhs)) => {lhs.extend(rhs.iter())}
            (AggregateBufferSet::IntSet(lhs), AggregateBufferSet::IntSet(rhs)) => {lhs.extend(rhs.iter())}
            (AggregateBufferSet::LongSet(lhs), AggregateBufferSet::LongSet(rhs)) => {lhs.extend(rhs.iter())}
            (AggregateBufferSet::FloatSet(lhs), AggregateBufferSet::FloatSet(rhs)) => {lhs.extend(rhs.iter())}
            (AggregateBufferSet::DoubleSet(lhs), AggregateBufferSet::DoubleSet(rhs)) => {lhs.extend(rhs.iter())}
            (AggregateBufferSet::StringSet(lhs), AggregateBufferSet::StringSet(rhs)) => {
                for item in rhs {
                    if !lhs.contains(item) {
                        lhs.insert(item.to_owned());
                    }
                }
            }
            (AggregateBufferSet::BinarySet(lhs), AggregateBufferSet::BinarySet(rhs)) => {
                for item in rhs {
                    if !lhs.contains(item) {
                        lhs.insert(item.to_owned());
                    }
                }
            }
            (AggregateBufferSet::ArraySet(lhs), AggregateBufferSet::ArraySet(rhs)) => {
                for item in rhs {
                    if !lhs.contains(item) {
                        lhs.insert(Wrap(item.0.clone()));
                    }
                }
            }
            _ => unreachable!("Mismatched types in AggregateBufferSet::extend_from"),
        };
    }

    pub(crate) fn is_empty(&self) -> bool {
        match self {
            AggregateBufferSet::NullSet => true,
            AggregateBufferSet::BoolSet(v) => v.is_empty(),
            AggregateBufferSet::ByteSet(v) => v.is_empty(),
            AggregateBufferSet::ShortSet(v) => v.is_empty(),
            AggregateBufferSet::IntSet(v) => v.is_empty(),
            AggregateBufferSet::LongSet(v) => v.is_empty(),
            AggregateBufferSet::FloatSet(v) => v.is_empty(),
            AggregateBufferSet::DoubleSet(v) => v.is_empty(),
            AggregateBufferSet::StringSet(v) => v.is_empty(),
            AggregateBufferSet::BinarySet(v) => v.is_empty(),
            AggregateBufferSet::ArraySet(v) => v.is_empty(),
        }
    }
}

pub(crate) struct UniqueAgg {
    chunk_idx: IdxSize,
    set: AggregateBufferSet,
    pub(crate) dtype: DataType,
}

impl UniqueAgg {
    pub(crate) fn new(dtype: DataType) -> Self {
        Self {
            chunk_idx: 0,
            set: AggregateBufferSet::default(),
            dtype,
        }
    }
}

impl AggregateFn for UniqueAgg {
    fn pre_agg(&mut self, chunk_idx: IdxSize, item: &Series) {
        self.chunk_idx = chunk_idx;
        self.set = unsafe { AggregateBufferSet::from_series(item, &self.dtype).unwrap_unchecked_release() };
    }

    fn pre_agg_ordered(
        &mut self,
        chunk_idx: IdxSize,
        offset: IdxSize,
        length: IdxSize,
        values: &Series,
    ) {
        self.chunk_idx = chunk_idx;
        let take = values.slice_from_offsets(offset, length);
        self.set = unsafe { AggregateBufferSet::from_series(&take, &self.dtype).unwrap_unchecked_release() };
    }

    fn dtype(&self) -> DataType {
        self.dtype.clone()
    }

    fn combine(&mut self, other: &dyn Any) {
        let other = unsafe { other.downcast_ref::<Self>().unwrap_unchecked_release() };
        if !other.set.is_empty() && other.chunk_idx >= self.chunk_idx {
            self.set.extend_from(&other.set);
            self.chunk_idx = other.chunk_idx;
        };
    }

    fn finalize(&mut self) -> AnyValue<'static> {
        let set = std::mem::take(&mut self.set);
        AnyValue::List(set.into_series(&self.dtype))
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}
