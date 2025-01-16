use std::any::Any;

use polars_core::datatypes::{AnyValue, DataType};
use polars_core::prelude::Series;
use polars_utils::aliases::PlIndexSet;
use polars_utils::pl_str::PlSmallStr;
use polars_utils::unwrap::UnwrapUncheckedRelease;
use polars_utils::IdxSize;

use crate::executors::sinks::group_by::aggregates::AggregateFn;

pub(crate) struct UniqueAgg {
    chunk_idx: IdxSize,
    rowwise_set: PlIndexSet<AnyValue<'static>>,
    columnar_set: Series,
    pub(crate) dtype: DataType,
}

impl UniqueAgg {
    pub(crate) fn new(dtype: DataType) -> Self {
        Self {
            chunk_idx: 0,
            rowwise_set: PlIndexSet::default(),
            columnar_set: Series::new_empty(PlSmallStr::EMPTY, &dtype),
            dtype,
        }
    }
}

impl AggregateFn for UniqueAgg {
    fn pre_agg(&mut self, chunk_idx: IdxSize, item: &mut dyn ExactSizeIterator<Item = AnyValue>) {
        let item = unsafe {
            item.next()
                .unwrap_unchecked_release()
                .into_static()
                .unwrap_unchecked_release()
        };
        self.chunk_idx = chunk_idx;
        self.rowwise_set.insert(item);
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
        let _ = unsafe {
            self.columnar_set
                .extend(&take)
                .unwrap_unchecked_release()
                .unique()
                .unwrap_unchecked_release()
        };
    }

    fn dtype(&self) -> DataType {
        self.dtype.clone()
    }

    fn combine(&mut self, other: &dyn Any) {
        let other = unsafe { other.downcast_ref::<Self>().unwrap_unchecked_release() };

        if !other.rowwise_set.is_empty() {
            self.rowwise_set = self
                .rowwise_set
                .union(&other.rowwise_set)
                .cloned()
                .map(|x| unsafe { x.into_static().unwrap_unchecked_release() })
                .collect();
        }

        if !other.columnar_set.is_empty() {
            self.columnar_set = unsafe {
                self.columnar_set
                    .extend(&other.columnar_set)
                    .unwrap_unchecked_release()
                    .unique()
                    .unwrap_unchecked_release()
            };
        }
    }

    fn finalize(&mut self) -> AnyValue<'static> {
        let rowwise_set = std::mem::take(&mut self.rowwise_set);
        let columnar_set = std::mem::take(&mut self.columnar_set);

        if rowwise_set.is_empty() {
            return AnyValue::List(columnar_set);
        }

        let set_as_slice = rowwise_set.into_iter().collect::<Vec<_>>();
        if columnar_set.is_empty() {
            return AnyValue::List(unsafe {
                Series::from_any_values_and_dtype(
                    PlSmallStr::EMPTY,
                    &set_as_slice,
                    &self.dtype,
                    true,
                )
                .unwrap_unchecked_release()
            });
        }

        AnyValue::List(unsafe {
            Series::from_any_values_and_dtype(PlSmallStr::EMPTY, &set_as_slice, &self.dtype, true)
                .unwrap_unchecked_release()
                .extend(&columnar_set)
                .unwrap_unchecked_release()
                .unique()
                .unwrap_unchecked_release()
        })
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
