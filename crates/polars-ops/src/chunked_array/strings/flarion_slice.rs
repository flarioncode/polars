use polars_core::prelude::{DataType, StringChunked, StringChunkedBuilder};
use polars_core::series::{IntoSeries, Series};
use polars_error::PolarsResult;

pub fn flarion_slice_helper(
    strings: &StringChunked,
    offset: &Series,
    length: &Series,
) -> PolarsResult<Series> {
    let froms_iter = offset.i32()?;
    let lens_iter = length.i32()?;

    if froms_iter.is_empty() || lens_iter.is_empty() {
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

    let froms_length = froms_iter.len();
    let lens_length = lens_iter.len();
    let first_from = froms_iter.get(0);
    let first_len = lens_iter.get(0);

    // Fast path: if froms_iter is length 1 and it's null, return all nulls
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
                    (Some(s), Some(from), Some(len)) => Some(substring_core(s, from, len)),
                    (Some(s), Some(from), None) => substring_with_null_length(s, from),
                    _ => None,
                };
                builder.append_option(value);
            }
            builder.finish()
        },
        (_, 1) => {
            let mut builder = StringChunkedBuilder::new("".into(), strings.len());
            for (s_opt, from_opt) in strings.into_iter().zip(froms_iter) {
                let value = match (s_opt, from_opt, first_len) {
                    (Some(s), Some(from), Some(len)) => Some(substring_core(s, from, len)),
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
                .zip(froms_iter.into_iter().zip(lens_iter))
            {
                let value = match (s_opt, from_opt, len_opt) {
                    (Some(s), Some(from), Some(len)) => Some(substring_core(s, from, len)),
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

// Core substring function that handles a single string
fn substring_core(s: &str, from: i32, len: i32) -> String {
    if s.is_empty() || len <= 0 {
        return String::new();
    }

    if from >= 0 {
        // This implementation is pretty clean but would not work as well with negative 'from' values due to all sorts of Spark quirks.
        let from_as_usize = match from {
            0 => 0,
            i => i - 1,
        } as usize;

        let mut iter = s.char_indices();

        let start_char = iter.nth(from_as_usize);
        let end_char = iter.nth(len as usize - 1);

        match start_char {
            None => String::new(),
            Some((start_idx, _)) => match end_char {
                None => s[start_idx..].to_string(),
                Some((end_idx, _)) => s[start_idx..end_idx].to_string(),
            },
        }
    } else {
        let mut start_char = 0;
        let mut end_char = 0;
        let mut found_start = false;

        let distance_to_advance = from.unsigned_abs() as usize;
        let end_idx = std::cmp::max(0, distance_to_advance as i32 - len) as usize; // We know len is positive

        for (idx, (byte_pos, ch)) in s.char_indices().rev().enumerate() {
            // We will always reach this code because end_idx is smaller than distance_to_advance, and idx == distance_to_advance breaks the flow.
            if idx == end_idx {
                // The "+ ch.len_utf8()" operation is because we need to see where the char ends.
                end_char = byte_pos + ch.len_utf8();
            }

            if idx == distance_to_advance {
                // The "+ ch.len_utf8()" operation is because we need to see where the char ends.
                start_char = byte_pos + ch.len_utf8();
                found_start = true;
                break;
            }
        }

        match found_start {
            true => s[start_char..end_char].to_string(),
            false => s[..end_char].to_string(),
        }
    }
}

/*
// Function that handles Series operations and calls substring_core
fn substring(params: &mut [Series]) -> PolarsResult<Option<Series>> {
    if params.len() != 3 {
        return Ok(None);
    }


}
    */
