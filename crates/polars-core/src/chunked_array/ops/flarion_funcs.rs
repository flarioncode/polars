use arrow::array::ValueSize;
use polars_error::{PolarsError, PolarsResult};
use polars_utils::pl_str::PlSmallStr;
#[cfg(feature = "regex")]
use regex::RegexBuilder;

use crate::datatypes::{DataType, Int32Chunked, ListChunked, StringChunked};
use crate::prelude::split::split_chars;
#[cfg(feature = "dtype-struct")]
use crate::prelude::split::splitn_chars;
use crate::prelude::{
    IntoSeries, ListBuilderTrait, ListStringChunkedBuilder, Series, StringChunkedBuilder,
};

pub fn flarion_get_char_position(haystack: &str, needle: &str) -> i32 {
    if needle.is_empty() {
        return 1;
    }

    match haystack.find(needle) {
        Some(byte_idx) => {
            // Always add 1 since SQL uses 1-based indexing
            1 + haystack[..byte_idx].chars().count() as i32
        },
        None => 0,
    }
}

// Core substring function that handles a single string
pub fn flarion_substring(s: &str, from: i32, len: i32) -> String {
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

pub fn flarion_instr_helper(
    string_series: &StringChunked,
    pattern: &Series,
) -> PolarsResult<Series> {
    let pattern_series = pattern.str()?;

    let result = if pattern_series.len() == 1 {
        let pattern = pattern_series.get(0).unwrap_or("");
        string_series
            .into_iter()
            .map(|opt_str| opt_str.map(|s| flarion_get_char_position(s, pattern)))
            .collect::<Int32Chunked>()
            .into_series()
    } else if string_series.len() == pattern_series.len() {
        string_series
            .into_iter()
            .zip(pattern_series)
            .map(|(opt_s, opt_p)| match (opt_s, opt_p) {
                (Some(s), Some(p)) => Some(flarion_get_char_position(s, p)),
                _ => None,
            })
            .collect::<Int32Chunked>()
            .into_series()
    } else {
        return Err(PolarsError::ComputeError(
            "Lengths of 'string' and 'pattern' series must be equal or pattern series must have length 1".into(),
        ));
    };

    Ok(result)
}

pub fn flarion_slice_helper(
    strings: &StringChunked,
    offset: &Series,
    length: &Series,
) -> PolarsResult<Series> {
    let from_iter = offset.i32()?;
    let lens_iter = length.i32()?;

    if from_iter.is_empty() || lens_iter.is_empty() {
        return Ok(Series::full_null(
            PlSmallStr::EMPTY,
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

    let from_length = from_iter.len();
    let lens_length = lens_iter.len();
    let first_from = from_iter.get(0);
    let first_len = lens_iter.get(0);

    // Fast path: if from_iter is length 1 and it's null, return all nulls
    if from_length == 1 && first_from.is_none() {
        return Ok(Series::full_null(
            PlSmallStr::EMPTY,
            strings.len(),
            &DataType::String,
        ));
    }

    let result: StringChunked = match (from_length, lens_length) {
        // Commented out because it is already covered in previous check
        // (0, _) | (_, 0) => None,
        (1, 1) => {
            let mut builder = StringChunkedBuilder::new(PlSmallStr::EMPTY, strings.len());
            for s_opt in strings.into_iter() {
                let value = match (s_opt, first_from, first_len) {
                    (Some(s), Some(from), Some(len)) => Some(flarion_substring(s, from, len)),
                    (Some(s), Some(from), None) => substring_with_null_length(s, from),
                    _ => None,
                };
                builder.append_option(value);
            }
            builder.finish()
        },
        (1, _) => {
            let mut builder = StringChunkedBuilder::new(PlSmallStr::EMPTY, strings.len());
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

// I'm hoping the compiler optimizes this out, but it is required for correct outputs
#[inline]
fn fill_empty_iter<'a, I: Iterator<Item = &'a str>>(in_iter: I) -> impl Iterator<Item = &'a str> {
    let chars = in_iter.collect::<Vec<_>>();
    if chars.is_empty() {
        // Can't use iter::once() because both we expect Vec's specific IntoIter
        vec![""].into_iter()
    } else {
        chars.into_iter()
    }
}

#[cfg(all(feature = "strings", feature = "dtype-struct"))]
pub fn flarion_split_helper(
    ca: &StringChunked,
    pattern: &str,
    n: i32,
) -> PolarsResult<ListChunked> {
    let mut builder =
        ListStringChunkedBuilder::new(ca.name().clone(), ca.len(), ca.get_values_size());

    // When the regex pattern is an empty string, we split the string into characters
    if pattern.is_empty() {
        if n > 0 {
            ca.into_iter().fold(&mut builder, |builder, opt_s| {
                match opt_s {
                    Some(s) => builder
                        .append_values_iter(fill_empty_iter(splitn_chars(s, n as usize, false))),
                    None => builder.append_null(),
                }
                builder
            });
        } else {
            ca.into_iter().fold(&mut builder, |builder, opt_s| {
                match opt_s {
                    Some(s) => builder.append_values_iter(fill_empty_iter(split_chars(s))),
                    None => builder.append_null(),
                };
                builder
            });
        };
    } else {
        // Special handling for "." pattern to match Spark behavior
        let pattern = if pattern == "." {
            // Match any character except \r and \n
            "[^\\r\\n]"
        } else {
            pattern
        };

        let re = RegexBuilder::new(pattern)
            .size_limit(30 * 1024 * 1024)
            .build()?;

        // When n is already a literal, we can check it ahead, and optimize for a negative number(no limit)
        if n > 0 {
            ca.into_iter().fold(&mut builder, |builder, opt_s| {
                match opt_s {
                    Some(s) => builder.append_values_iter(re.splitn(s, n as usize)),
                    None => builder.append_null(),
                };
                builder
            });
        } else {
            ca.into_iter().fold(&mut builder, |builder, opt_s| {
                match opt_s {
                    Some(s) => builder.append_values_iter(re.split(s)),
                    None => builder.append_null(),
                };
                builder
            });
        }
    }

    Ok(builder.finish())
}

#[cfg(all(test, all(feature = "strings", feature = "dtype-struct")))]
mod tests {
    use polars_utils::pl_str::PlSmallStr;

    use super::*;
    use crate::prelude::{AnyValue, NamedFrom};

    fn create_string_chunked(name: &str, values: Vec<Option<&str>>) -> StringChunked {
        let mut builder = StringChunkedBuilder::new(PlSmallStr::from_str(name), 10);
        for value in values {
            match value {
                Some(s) => builder.append_value(s),
                None => builder.append_null(),
            }
        }
        builder.finish()
    }

    #[test]
    fn test_split_empty_pattern() {
        let input = create_string_chunked("input", vec![Some("hello"), Some("世界"), None]);

        let result = flarion_split_helper(&input, "", 2).unwrap().into_series();

        assert_eq!(result.len(), 3);
        // When len is larger then limit, Spark returns the remainder as the last element
        assert_eq!(
            result.get(0).unwrap(),
            AnyValue::List(Series::new(PlSmallStr::EMPTY, &["h", "e"]))
        );
        assert_eq!(
            result.get(1).unwrap(),
            AnyValue::List(Series::new(PlSmallStr::EMPTY, &["世", "界"]))
        );
        assert!(result.get(2).unwrap().is_null());
    }

    #[test]
    fn test_split_with_regex() {
        let input = create_string_chunked(
            "input",
            vec![Some("a,b,c"), Some("x;y;z"), Some("1||2||3"), None],
        );

        let result = flarion_split_helper(&input, "[,;|]+", -1)
            .unwrap()
            .into_series();

        assert_eq!(result.len(), 4);
        assert_eq!(
            result.get(0).unwrap(),
            AnyValue::List(Series::new(PlSmallStr::EMPTY, &["a", "b", "c"]))
        );
        assert_eq!(
            result.get(1).unwrap(),
            AnyValue::List(Series::new(PlSmallStr::EMPTY, &["x", "y", "z"]))
        );
        assert_eq!(
            result.get(2).unwrap(),
            AnyValue::List(Series::new(PlSmallStr::EMPTY, &["1", "2", "3"]))
        );
        assert!(result.get(3).unwrap().is_null());
    }

    #[test]
    fn test_split_with_limit() {
        let input = create_string_chunked("input", vec![Some("a,b,c,d")]);

        let result = flarion_split_helper(&input, ",", 2).unwrap().into_series();

        assert_eq!(result.len(), 1);
        assert_eq!(
            result.get(0).unwrap(),
            AnyValue::List(Series::new(PlSmallStr::EMPTY, &["a", "b,c,d"]))
        );
    }

    #[test]
    fn test_invalid_regex_pattern() {
        let input = create_string_chunked("input", vec![Some("test")]);

        let res = flarion_split_helper(&input, "[", -1);
        assert!(res.is_err());
    }

    #[test]
    fn test_unicode_splitting() {
        let input = create_string_chunked("input", vec![Some("你好，世界")]);

        // This is not a regular comma!
        let result = flarion_split_helper(&input, "，", -1)
            .unwrap()
            .into_series();

        assert_eq!(result.len(), 1);
        assert_eq!(
            result.get(0).unwrap(),
            AnyValue::List(Series::new(PlSmallStr::EMPTY, &["你好", "世界"]))
        );
    }

    #[test]
    fn test_empty_input() {
        let input = create_string_chunked("input", vec![Some("")]);

        let result = flarion_split_helper(&input, ",", -1).unwrap().into_series();

        assert_eq!(result.len(), 1);
        assert_eq!(
            result.get(0).unwrap(),
            AnyValue::List(Series::new(PlSmallStr::EMPTY, &[""]))
        );
    }

    #[test]
    fn test_dot_pattern_splitting() {
        let input = create_string_chunked("input", vec![Some("a\rb\rc\nd\te")]);

        let result = flarion_split_helper(&input, ".", -1).unwrap().into_series();

        assert_eq!(result.len(), 1);
        assert_eq!(
            result.get(0).unwrap(),
            AnyValue::List(Series::new(
                PlSmallStr::EMPTY,
                &["", "\r", "\r", "\n", "", "", ""]
            ))
        );
    }
}
