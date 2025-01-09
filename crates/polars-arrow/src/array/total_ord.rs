use std::cmp::Ordering;

use polars_utils::min_max::MinMax;
use polars_utils::total_ord::{TotalEq, TotalOrd};

use crate::array::{
    Array, BinaryViewArray, BooleanArray, ListArray, PrimitiveArray, Utf8ViewArray,
};
use crate::datatypes::ArrowDataType;

impl TotalEq for Box<dyn Array> {
    fn tot_eq(&self, other: &Self) -> bool {
        self == other
    }
}

pub(crate) fn lexical_cmp<T: TotalOrd>(
    mut iter1: impl Iterator<Item = Option<T>>,
    mut iter2: impl Iterator<Item = Option<T>>,
) -> Ordering {
    loop {
        match (iter1.next(), iter2.next()) {
            (None, None) => break Ordering::Equal,
            (None, Some(_)) => break Ordering::Less,
            (Some(_), None) => break Ordering::Greater,
            (Some(None), Some(None)) => continue,
            (Some(a), Some(b)) => match a.tot_cmp(&b) {
                Ordering::Equal => continue,
                other => break other,
            },
        }
    }
}

impl TotalOrd for Box<dyn Array> {
    fn tot_cmp(&self, other: &Self) -> Ordering {
        let dtype = self.dtype();
        assert_eq!(
            dtype,
            other.dtype(),
            "Cannot compare arrays of different types ({:?}, {:?})",
            dtype,
            other.dtype()
        );

        macro_rules! numeric_list_lex_compare {
            ($dtype:ident, $a:ident, $b:ident, [$(($variant:tt, $primitive:ident),)*]) => {
                match $dtype {
                    $(ArrowDataType::$variant => {
                        let self_iter = self
                            .as_any()
                            .downcast_ref::<PrimitiveArray<$primitive>>()
                            .unwrap()
                            .iter()
                            .map(|v| v.copied());
                        let other_iter = other.as_any()
                            .downcast_ref::<PrimitiveArray<$primitive>>()
                            .unwrap()
                            .iter()
                            .map(|v| v.copied());
                        Some(lexical_cmp(self_iter, other_iter))
                    })*,
                    _ => None,
                }
            }
        }

        // Fast path for if one of the iterators is empty
        match (self.is_empty(), other.is_empty()) {
            (true, false) => Ordering::Less,
            (false, true) => Ordering::Greater,
            (true, true) => Ordering::Equal,
            (false, false) => {
                if let Some(ordering) = numeric_list_lex_compare!(
                    dtype,
                    self,
                    other,
                    [
                        (UInt8, u8),
                        (UInt16, u16),
                        (UInt32, u32),
                        (UInt64, u64),
                        (Int8, i8),
                        (Int16, i16),
                        (Int32, i32),
                        (Int64, i64),
                        (Float32, f32),
                        (Float64, f64),
                    ]
                ) {
                    return ordering;
                }

                match dtype {
                    ArrowDataType::Boolean => {
                        let self_iter =
                            self.as_any().downcast_ref::<BooleanArray>().unwrap().iter();
                        let other_iter = other
                            .as_any()
                            .downcast_ref::<BooleanArray>()
                            .unwrap()
                            .iter();
                        lexical_cmp(self_iter, other_iter)
                    },
                    ArrowDataType::Utf8View => {
                        let self_binview = self
                            .as_any()
                            .downcast_ref::<Utf8ViewArray>()
                            .unwrap()
                            .to_binview();
                        let self_iter = self_binview.iter();
                        let other_binview = other
                            .as_any()
                            .downcast_ref::<Utf8ViewArray>()
                            .unwrap()
                            .to_binview();
                        let other_iter = other_binview.iter();
                        lexical_cmp(self_iter, other_iter)
                    },
                    ArrowDataType::BinaryView => {
                        let self_iter = self
                            .as_any()
                            .downcast_ref::<BinaryViewArray>()
                            .unwrap()
                            .iter();
                        let other_iter = other
                            .as_any()
                            .downcast_ref::<BinaryViewArray>()
                            .unwrap()
                            .iter();
                        lexical_cmp(self_iter, other_iter)
                    },
                    ArrowDataType::LargeList(_) => {
                        let self_iter = self
                            .as_any()
                            .downcast_ref::<ListArray<i64>>()
                            .unwrap()
                            .iter();
                        let other_iter = other
                            .as_any()
                            .downcast_ref::<ListArray<i64>>()
                            .unwrap()
                            .iter();
                        lexical_cmp(self_iter, other_iter)
                    },
                    _ => panic!("Comparison unsupported for array with dtype: {:?}", dtype),
                }
            },
        }
    }
}

impl MinMax for Box<dyn Array> {
    fn nan_min_lt(&self, other: &Self) -> bool {
        self.tot_cmp(other) == Ordering::Less
    }

    fn nan_max_lt(&self, other: &Self) -> bool {
        self.tot_cmp(other) == Ordering::Less
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitmap::Bitmap;
    use crate::offset::OffsetsBuffer;
    use crate::types::NativeType;

    fn create_primitive_list<T, S1, S2>(arrays: S2) -> Box<dyn Array>
    where
        T: NativeType,
        S1: AsRef<[Option<T>]>,
        S2: AsRef<[Option<S1>]>,
    {
        let offsets_iter = arrays
            .as_ref()
            .iter()
            .filter_map(|x| x.as_ref())
            .map(|a| a.as_ref().len() as i64)
            .scan(0i64, |sum, l| {
                *sum += l;
                Some(*sum)
            });
        let offsets =
            OffsetsBuffer::try_from(std::iter::once(0).chain(offsets_iter).collect::<Vec<_>>())
                .unwrap();
        let values = PrimitiveArray::<T>::from_iter(
            arrays
                .as_ref()
                .iter()
                .filter_map(|x| x.as_ref())
                .flat_map(|a| a.as_ref().iter())
                .copied(),
        );
        let dtype = ListArray::<i64>::default_datatype(values.dtype().clone());
        let validity = Bitmap::from_trusted_len_iter(arrays.as_ref().iter().map(|x| x.is_some()));

        ListArray::<i64>::new(dtype, offsets, Box::new(values), Some(validity)).boxed()
    }

    #[test]
    fn test_total_ord() {
        let prim = PrimitiveArray::<i32>::from_iter([Some(1), Some(2), Some(3)]).boxed();
        assert_eq!(prim.tot_cmp(&prim), Ordering::Equal);

        let prim1 = PrimitiveArray::<i32>::from_iter([Some(1), Some(2), Some(3)]).boxed();
        let prim2 = PrimitiveArray::<i32>::from_iter([Some(1), Some(2), Some(4)]).boxed();
        assert_eq!(prim1.tot_cmp(&prim2), Ordering::Less);
        assert_eq!(prim2.tot_cmp(&prim1), Ordering::Greater);

        let bool = BooleanArray::from_slice([true, false, false]).boxed();
        assert_eq!(bool.tot_cmp(&bool), Ordering::Equal);

        let bool1 = BooleanArray::from_slice([true, false, false]).boxed();
        let bool2 = BooleanArray::from_slice([true, false, true]).boxed();
        assert_eq!(bool1.tot_cmp(&bool2), Ordering::Less);
        assert_eq!(bool2.tot_cmp(&bool1), Ordering::Greater);

        let utf8 = Utf8ViewArray::from_slice([Some("a"), Some("b"), Some("c")]).boxed();
        assert_eq!(utf8.tot_cmp(&utf8), Ordering::Equal);

        let utf8_1 = Utf8ViewArray::from_slice([Some("a"), Some("b"), Some("c")]).boxed();
        let utf8_2 = Utf8ViewArray::from_slice([Some("a"), Some("b"), Some("d")]).boxed();
        assert_eq!(utf8_1.tot_cmp(&utf8_2), Ordering::Less);
        assert_eq!(utf8_2.tot_cmp(&utf8_1), Ordering::Greater);

        let bin = BinaryViewArray::from_slice([Some(b"a"), Some(b"b"), Some(b"c")]).boxed();
        assert_eq!(bin.tot_cmp(&bin), Ordering::Equal);

        let bin_1 = BinaryViewArray::from_slice([Some(b"a"), Some(b"b"), Some(b"c")]).boxed();
        let bin_2 = BinaryViewArray::from_slice([Some(b"a"), Some(b"b"), Some(b"d")]).boxed();
        assert_eq!(bin_1.tot_cmp(&bin_2), Ordering::Less);
        assert_eq!(bin_2.tot_cmp(&bin_1), Ordering::Greater);

        let nested = create_primitive_list([
            Some([Some(1), Some(2), Some(3)]),
            Some([Some(4), Some(5), Some(6)]),
        ]);

        assert_eq!(nested.tot_cmp(&nested), Ordering::Equal);

        let nested1 = create_primitive_list([
            Some([Some(1), Some(2), Some(3)]),
            Some([Some(4), Some(5), Some(6)]),
        ]);
        let nested2 = create_primitive_list([
            Some([Some(1), Some(2), Some(3)]),
            Some([Some(4), Some(5), Some(7)]),
        ]);

        assert_eq!(nested1.tot_cmp(&nested2), Ordering::Less);
        assert_eq!(nested2.tot_cmp(&nested1), Ordering::Greater);

        let null1 = PrimitiveArray::<i32>::from_iter([Some(1), None, Some(3)]).boxed();
        let null2 = PrimitiveArray::<i32>::from_iter([Some(1), Some(2), None]).boxed();
        assert_eq!(null1.tot_cmp(&null2), Ordering::Less);
        assert_eq!(null2.tot_cmp(&null1), Ordering::Greater);

        let len1 = PrimitiveArray::<i32>::from_iter([Some(1), Some(2), Some(3)]).boxed();
        let len2 = PrimitiveArray::<i32>::from_iter([Some(1), Some(3)]).boxed();
        assert_eq!(len1.tot_cmp(&len2), Ordering::Less);
        assert_eq!(len2.tot_cmp(&len1), Ordering::Greater);

        let empty = PrimitiveArray::<i32>::new_empty(ArrowDataType::Int32).boxed();
        let non_empty = PrimitiveArray::<i32>::new_null(ArrowDataType::Int32, 1).boxed();
        assert_eq!(empty.tot_cmp(&non_empty), Ordering::Less);
        assert_eq!(non_empty.tot_cmp(&empty), Ordering::Greater);
    }
}
