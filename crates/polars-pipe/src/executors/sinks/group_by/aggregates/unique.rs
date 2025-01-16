use std::any::Any;

use polars_core::datatypes::{AnyValue, DataType};
use polars_core::prelude::Series;
use polars_utils::pl_str::PlSmallStr;
use polars_utils::unwrap::UnwrapUncheckedRelease;
use polars_utils::IdxSize;

use crate::executors::sinks::group_by::aggregates::AggregateFn;

pub(crate) struct UniqueAgg {
    chunk_idx: IdxSize,
    set: Series,
    pub(crate) dtype: DataType,
}

impl UniqueAgg {
    pub(crate) fn new(dtype: DataType) -> Self {
        Self {
            chunk_idx: 0,
            set: Series::new_empty(PlSmallStr::EMPTY, &dtype),
            dtype,
        }
    }
}

impl AggregateFn for UniqueAgg {
    fn pre_agg(&mut self, chunk_idx: IdxSize, item: &Series) {
        self.chunk_idx = chunk_idx;
        self.set = item.unique().unwrap();
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
        self.set = take.unique().unwrap();
    }

    fn dtype(&self) -> DataType {
        self.dtype.clone()
    }

    fn combine(&mut self, other: &dyn Any) {
        let other = unsafe { other.downcast_ref::<Self>().unwrap_unchecked_release() };
        if !other.set.is_empty() && other.chunk_idx >= self.chunk_idx {
            self.set = self.set.extend(&other.set).unwrap().unique().unwrap();
            self.chunk_idx = other.chunk_idx;
        };
    }

    fn finalize(&mut self) -> AnyValue<'static> {
        let set = std::mem::take(&mut self.set);
        AnyValue::List(set)
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}
