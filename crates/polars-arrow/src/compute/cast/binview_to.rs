use chrono::Datelike;
use num_traits::Bounded;
use polars_error::PolarsResult;
use polars_utils::float::IsFloat;

use crate::array::*;
use crate::compute::cast::binary_to::Parse;
use crate::compute::cast::{CastOptionsImpl, SparkAsPrimitive};
#[cfg(feature = "dtype-decimal")]
use crate::compute::decimal::deserialize_decimal;
use crate::datatypes::{ArrowDataType, TimeUnit};
use crate::offset::Offset;
use crate::temporal_conversions::EPOCH_DAYS_FROM_CE;
use crate::types::NativeType;

pub(super) const RFC3339: &str = "%Y-%m-%dT%H:%M:%S%.f%:z";

/// Cast [`BinaryViewArray`] to [`DictionaryArray`], also known as packing.
/// # Errors
/// This function errors if the maximum key is smaller than the number of distinct elements
/// in the array.
pub(super) fn binview_to_dictionary<K: DictionaryKey>(
    from: &BinaryViewArray,
) -> PolarsResult<DictionaryArray<K>> {
    let mut array = MutableDictionaryArray::<K, MutableBinaryViewArray<[u8]>>::new();
    array.reserve(from.len());
    array.try_extend(from.iter())?;

    Ok(array.into())
}

pub(super) fn utf8view_to_dictionary<K: DictionaryKey>(
    from: &Utf8ViewArray,
) -> PolarsResult<DictionaryArray<K>> {
    let mut array = MutableDictionaryArray::<K, MutableBinaryViewArray<str>>::new();
    array.reserve(from.len());
    array.try_extend(from.iter())?;

    Ok(array.into())
}

pub(super) fn view_to_binary<O: Offset>(array: &BinaryViewArray) -> BinaryArray<O> {
    let len: usize = Array::len(array);
    let mut mutable = MutableBinaryValuesArray::<O>::with_capacities(len, array.total_bytes_len());
    for slice in array.values_iter() {
        mutable.push(slice)
    }
    let out: BinaryArray<O> = mutable.into();
    out.with_validity(array.validity().cloned())
}

pub fn utf8view_to_utf8<O: Offset>(array: &Utf8ViewArray) -> Utf8Array<O> {
    let array = array.to_binview();
    let out = view_to_binary::<O>(&array);

    let dtype = Utf8Array::<O>::default_dtype();
    unsafe {
        Utf8Array::new_unchecked(
            dtype,
            out.offsets().clone(),
            out.values().clone(),
            out.validity().cloned(),
        )
    }
}
/// Casts a [`BinaryArray`] to a [`PrimitiveArray`], making any uncastable value a Null.
pub(super) fn binview_to_primitive<T>(
    from: &BinaryViewArray,
    to: &ArrowDataType,
) -> PrimitiveArray<T>
where
    T: NativeType + Parse + IsFloat + Bounded + SparkAsPrimitive<f64>,
    f64: SparkAsPrimitive<T>,
{
    let iter = from.iter().map(|x| {
        x.and_then::<T, _>(|x| {
            // Attempt to interpret input as UTF-8 and trim invalid or whitespace characters
            let x = std::str::from_utf8(x)
                .map(|s| s.trim().as_bytes())
                .unwrap_or_else(|_| x.trim_ascii());

            // Extract and process leading sign
            let (x, is_negative) = match x.first() {
                Some(b'+') => (&x[1..], false),
                Some(b'-') => (&x[1..], true),
                _ => (x, false),
            };

            // Check if the number has a float suffix
            let x = if matches!(x.last(), Some(b'f' | b'F' | b'd' | b'D')) {
                if !T::is_float() {
                    return None;
                }
                &x[..x.len() - 1]
            } else {
                x
            };

            // Handle leading decimal point
            let (x, had_leading_decimal) = if let Some(b'.') = x.first() {
                (&x[1..], true)
            } else {
                (x, false)
            };

            // Only decimal point (or sign and decimal point) equals 0
            if had_leading_decimal && x.is_empty() {
                return if !T::is_float() {
                    Some(T::zeroed())
                } else {
                    None
                };
            }

            // Early exit if the first digits are a scientific notation or another sign
            if matches!(x.first(), Some(b'E' | b'e' | b'+' | b'-')) {
                return None;
            }

            // Handle leading zeros if did not have a leading decimal
            let x = if !had_leading_decimal && matches!(x.first(), Some(b'0')) {
                match x.iter().position(|&c| c != b'0') {
                    Some(pos) if matches!(x.get(pos), Some(b'e' | b'E' | b'.')) => &x[0.max(pos - 1)..],
                    Some(pos) if !x.get(pos).map(u8::is_ascii_digit).unwrap_or(false) => return None,
                    Some(pos) => &x[pos..],
                    None => b"0".as_ref(),
                }
            } else {
                x
            };

            // For non-floats, having a scientific notation is not allowed in general
            if !T::is_float() && x.iter().any(|&c| c == b'e' || c == b'E') {
                return None;
            }

            // Reconstruct with sign and leading decimal if necessary
            let mut reconstructed = Vec::with_capacity(x.len() + 3);
            if is_negative {
                reconstructed.push(b'-');
            }
            if had_leading_decimal {
                reconstructed.extend_from_slice(b"0.");
            }
            reconstructed.extend_from_slice(x);
            if matches!(reconstructed.last(), Some(b'.')) {
                reconstructed.push(b'0');
            }

            // Parse all numbers with decimals as f64 first
            if reconstructed.contains(&b'.') {
                f64::parse(&reconstructed).and_then(|value| {
                    if T::is_float() {
                        Some(value.as_())
                    } else {
                        let min: f64 = SparkAsPrimitive::as_(T::min_value());
                        let max: f64 = SparkAsPrimitive::as_(T::max_value());
                        if (min..=max).contains(&value) {
                            Some(value.as_())
                        } else {
                            None
                        }
                    }
                })
            } else {
                // For integers without decimal points, parse directly to T
                T::parse(&reconstructed)
            }
        })
    });

    PrimitiveArray::<T>::from_trusted_len_iter(iter).to(to.clone())
}

pub(super) fn binview_to_primitive_dyn<T>(
    from: &dyn Array,
    to: &ArrowDataType,
    options: CastOptionsImpl,
) -> PolarsResult<Box<dyn Array>>
where
    T: NativeType + Parse + IsFloat + Bounded + SparkAsPrimitive<f64>,
    f64: SparkAsPrimitive<T>,
{
    let from = from.as_any().downcast_ref().unwrap();
    if options.partial {
        unimplemented!()
    } else {
        Ok(Box::new(binview_to_primitive::<T>(from, to)))
    }
}

#[cfg(feature = "dtype-decimal")]
pub fn binview_to_decimal(
    array: &BinaryViewArray,
    precision: Option<usize>,
    scale: usize,
) -> PrimitiveArray<i128> {
    let precision = precision.map(|p| p as u8);
    array
        .iter()
        .map(|val| val.and_then(|val| deserialize_decimal(val, precision, scale as u8)))
        .collect()
}

pub(super) fn utf8view_to_naive_timestamp_dyn(
    from: &dyn Array,
    time_unit: TimeUnit,
) -> PolarsResult<Box<dyn Array>> {
    let from = from.as_any().downcast_ref().unwrap();
    Ok(Box::new(utf8view_to_naive_timestamp(from, time_unit)))
}

/// [`crate::temporal_conversions::utf8view_to_timestamp`] applied for RFC3339 formatting
pub fn utf8view_to_naive_timestamp(
    from: &Utf8ViewArray,
    time_unit: TimeUnit,
) -> PrimitiveArray<i64> {
    crate::temporal_conversions::utf8view_to_naive_timestamp(from, RFC3339, time_unit)
}

pub(super) fn utf8view_to_date32(from: &Utf8ViewArray) -> PrimitiveArray<i32> {
    let iter = from.iter().map(|x| {
        x.and_then(|x| {
            x.parse::<chrono::NaiveDate>()
                .ok()
                .map(|x| x.num_days_from_ce() - EPOCH_DAYS_FROM_CE)
        })
    });
    PrimitiveArray::<i32>::from_trusted_len_iter(iter).to(ArrowDataType::Date32)
}

pub(super) fn utf8view_to_date32_dyn(from: &dyn Array) -> PolarsResult<Box<dyn Array>> {
    let from = from.as_any().downcast_ref().unwrap();
    Ok(Box::new(utf8view_to_date32(from)))
}
