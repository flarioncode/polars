use polars_core::chunked_array::ops::flarion_funcs::flarion_substring;
use polars_core::prelude::{DataType, StringChunked, StringChunkedBuilder};
use polars_core::series::{IntoSeries, Series};
use polars_error::PolarsResult;

pub fn flarion_slice_helper(
    strings: &StringChunked,
    offset: &Series,
    length: &Series,
) -> PolarsResult<Series> {
    let from_iter = offset.i32()?;
    let lens_iter = length.i32()?;

    if from_iter.is_empty() || lens_iter.is_empty() {
        return Ok(Series::full_null(
            "".into(),
            strings.len(),
            &DataType::String,
        ));
    }

    // Helper function to handle substring with null length
    let substring_with_null_length = |s: &str, from: i32| match from {
        f if f <= 0 || f == 1 => Some(s.to_string()),
        f if (f as usize) <= s.len() => Some(s[(f as usize - 1)..].to_string()),
        _ => None,
    };

    let froms_length = from_iter.len();
    let lens_length = lens_iter.len();
    let first_from = from_iter.get(0);
    let first_len = lens_iter.get(0);

    // Fast path: if from_iter is length 1 and it's null, return all nulls
    if froms_length == 1 && first_from.is_none() {
        return Ok(Series::full_null(
            "".into(),
            strings.len(),
            &DataType::String,
        ));
    }

    let result: StringChunked = match (froms_length, lens_length) {
        // Commented out because it is already covered in previous check
        // (0, _) | (_, 0) => None,
        (1, _) => {
            let mut builder = StringChunkedBuilder::new("".into(), strings.len());
            for (s_opt, len_opt) in strings.into_iter().zip(lens_iter) {
                let value = match (s_opt, first_from, len_opt) {
                    (Some(s), Some(from), Some(len)) => Some(flarion_substring(s, from, len)),
                    (Some(s), Some(from), None) => substring_with_null_length(s, from),
                    _ => None,
                };
                builder.append_option(value);
            }
            builder.finish()
        },
        (_, 1) => {
            let mut builder = StringChunkedBuilder::new("".into(), strings.len());
            for (s_opt, from_opt) in strings.into_iter().zip(from_iter) {
                let value = match (s_opt, from_opt, first_len) {
                    (Some(s), Some(from), Some(len)) => Some(flarion_substring(s, from, len)),
                    (Some(s), Some(from), None) => substring_with_null_length(s, from),
                    _ => None,
                };
                builder.append_option(value);
            }
            builder.finish()
        },
        _ => {
            let mut builder = StringChunkedBuilder::new("".into(), strings.len());
            for (s_opt, (from_opt, len_opt)) in strings
                .into_iter()
                .zip(from_iter.into_iter().zip(lens_iter))
            {
                let value = match (s_opt, from_opt, len_opt) {
                    (Some(s), Some(from), Some(len)) => Some(flarion_substring(s, from, len)),
                    (Some(s), Some(from), None) => substring_with_null_length(s, from),
                    _ => None,
                };
                builder.append_option(value);
            }
            builder.finish()
        },
    };

    Ok(result.into_series())
}
