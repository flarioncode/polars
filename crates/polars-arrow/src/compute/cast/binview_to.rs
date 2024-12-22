use chrono::Datelike;
use num_traits::{AsPrimitive, Bounded};
use polars_error::PolarsResult;
use polars_utils::float::IsFloat;

use crate::array::*;
use crate::compute::cast::binary_to::Parse;
use crate::compute::cast::CastOptionsImpl;
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

#[inline]
pub fn str_to_bool(s: &str) -> Option<bool> {
    // Only matches U+0020, \n, \t, \f (\x0C), \v (\x0B) and \r to be exactly like Spark's behavior regarding whitespaces.
    // Original logic appears in Spark functions isTrueString and isFalseString in StringUtils.scala
    let s = s
        .trim_matches(&[' ', '\n', '\r', '\t', '\x0C', '\x0B'][..])
        .to_lowercase();
    match s.as_str() {
        "t" | "true" | "y" | "yes" | "1" => Some(true),
        "f" | "false" | "n" | "no" | "0" => Some(false),
        _ => None,
    }
}

pub fn binview_to_primitive_impl<T>(x: &[u8]) -> Option<T>
where
    T: NativeType + Parse + IsFloat + Bounded + AsPrimitive<f64>,
    f64: AsPrimitive<T>,
{
    let is_float = T::is_float();

    // Attempt to interpret input as UTF-8 and trim invalid or whitespace characters
    let x = unsafe { std::str::from_utf8_unchecked(x).trim().as_bytes() };

    // Extract and process leading sign
    let (x, is_negative) = match x.first() {
        Some(b'+') => (&x[1..], false),
        Some(b'-') => (&x[1..], true),
        _ => (x, false),
    };

    // Check if the number has a float suffix
    let x = if matches!(x.last(), Some(b'f' | b'F' | b'd' | b'D')) {
        if !is_float {
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
        return if !is_float { Some(T::zeroed()) } else { None };
    }

    // Early exit if the first digits are a scientific notation or another sign
    if matches!(x.first(), Some(b'E' | b'e' | b'+' | b'-')) {
        return None;
    }

    // Handle leading zeros if did not have a leading decimal
    let x = if !had_leading_decimal {
        let mut start = 0;
        while start < x.len() && x[start] == b'0' {
            start += 1;
        }
        if start > 0 {
            match x.get(start) {
                Some(b'e' | b'E' | b'.') => &x[start.saturating_sub(1)..],
                Some(b) if !b.is_ascii_digit() => return None,
                Some(_) => &x[start..],
                None => b"0".as_ref(),
            }
        } else {
            x
        }
    } else {
        x
    };

    // For non-floats, having a scientific notation is not allowed in general
    if !is_float && x.iter().any(|&c| c == b'e' || c == b'E') {
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
            if is_float {
                Some(value.as_())
            } else {
                let min: f64 = T::min_value().as_();
                let max: f64 = T::max_value().as_();
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
}

/// Casts a [`BinaryArray`] to a [`PrimitiveArray`], making any uncastable value a Null.
pub(super) fn binview_to_primitive<T>(
    from: &BinaryViewArray,
    to: &ArrowDataType,
) -> PrimitiveArray<T>
where
    T: NativeType + Parse + IsFloat + Bounded + AsPrimitive<f64>,
    f64: AsPrimitive<T>,
{
    let iter = from
        .iter()
        .map(|x| x.and_then::<T, _>(binview_to_primitive_impl));

    PrimitiveArray::<T>::from_trusted_len_iter(iter).to(to.clone())
}

pub(super) fn binview_to_primitive_dyn<T>(
    from: &dyn Array,
    to: &ArrowDataType,
    options: CastOptionsImpl,
) -> PolarsResult<Box<dyn Array>>
where
    T: NativeType + Parse + IsFloat + Bounded + AsPrimitive<f64>,
    f64: AsPrimitive<T>,
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
