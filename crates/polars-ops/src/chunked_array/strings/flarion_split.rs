use arrow::array::ValueSize;
use polars_utils::cache::FastFixedCache;
use regex::RegexBuilder;

use super::*;

// Helper function for a recurring pattern
// For when no regex pattern is provided, and we only need the first n characters
#[inline]
fn split_charsn(builder: &mut ListStringChunkedBuilder, opt_s: Option<&str>, n: usize) {
    match opt_s {
        Some(s) => builder.append_values_iter(splitn_chars(s, n, false)),
        None => builder.append_null(),
    }
}

// Helper function for a recurring pattern
// For when no regex pattern is provided, and we need all the characters(i.e., n is negative)
#[inline]
fn split_chars_no_limit(builder: &mut ListStringChunkedBuilder, opt_s: Option<&str>) {
    match opt_s {
        Some(s) => builder.append_values_iter(split_chars(s)),
        None => builder.append_null(),
    }
}

// Helper function for a recurring pattern
// When no regex pattern is provided, we split the string into characters(and possibly take the first n characters)
#[inline]
fn split_chars_helper(builder: &mut ListStringChunkedBuilder, opt_s: Option<&str>, n: i32) {
    if n >= 0 {
        split_charsn(builder, opt_s, n as usize);
        return;
    }

    split_chars_no_limit(builder, opt_s)
}

// Helper function for a recurring pattern
// This can be called directly in a loop when n is a zero/positive literal, to optimize performance
#[inline]
fn regex_splitn(
    builder: &mut ListStringChunkedBuilder,
    opt_s: Option<&str>,
    re: &regex::Regex,
    n: usize,
) {
    match opt_s {
        Some(s) => builder.append_values_iter(re.splitn(s, n)),
        None => builder.append_null(),
    }
}

// Helper function for a recurring pattern
// This can be called directly in a loop when n is a negative literal, to optimize performance
#[inline]
fn regex_split_no_limit(
    builder: &mut ListStringChunkedBuilder,
    opt_s: Option<&str>,
    re: &regex::Regex,
) {
    match opt_s {
        Some(s) => builder.append_values_iter(re.split(s)),
        None => builder.append_null(),
    }
}

// Helper function for a recurring pattern
// We already have the compiled regex pattern, and number of splits, and we can split the string accordingly
#[inline]
fn regex_split_helper(
    builder: &mut ListStringChunkedBuilder,
    opt_s: Option<&str>,
    re: &regex::Regex,
    n: i32,
) {
    if n >= 0 {
        regex_splitn(builder, opt_s, re, n as usize);
        return;
    }

    regex_split_no_limit(builder, opt_s, re)
}

// In the case where both the regex pattern, and the number of splits are literals, we can check them ahead,
// and optimize both for an empty pattern, and for a negative number of splits(no limit)
fn handle_all_literals(
    ca: &StringChunked,
    opt_by: Option<&str>,
    opt_n: Option<i32>,
) -> PolarsResult<ListChunked> {
    let n = opt_n.ok_or(polars_err!(ComputeError: "n must be provided for split operation"))?;

    if let Some(by) = opt_by {
        let mut builder =
            ListStringChunkedBuilder::new(ca.name().clone(), ca.len(), ca.get_values_size());

        // When the regex pattern is an empty string, we split the string into characters
        if by.is_empty() {
            if n >= 0 {
                ca.into_iter().fold(&mut builder, |builder, opt_s| {
                    split_charsn(builder, opt_s, n as usize);
                    builder
                });
            } else {
                ca.into_iter().fold(&mut builder, |builder, opt_s| {
                    split_chars_no_limit(builder, opt_s);
                    builder
                });
            };
        } else {
            let re = RegexBuilder::new(by).size_limit(31457280).build().unwrap();

            // When n is already a literal, we can check it ahead, and optimize for a negative number(no limit)
            if n >= 0 {
                ca.into_iter().fold(&mut builder, |builder, opt_s| {
                    regex_splitn(builder, opt_s, &re, n as usize);
                    builder
                });
            } else {
                ca.into_iter().fold(&mut builder, |builder, opt_s| {
                    regex_split_no_limit(builder, opt_s, &re);
                    builder
                });
            }
        }

        return Ok(builder.finish());
    }

    Ok(ListChunked::full_null_with_dtype(
        ca.name().clone(),
        ca.len(),
        &DataType::String,
    ))
}

// If the regex pattern is a literal, we can check it ahead, and optimize for an empty literal
fn handle_literal_by(
    ca: &StringChunked,
    by: Option<&str>,
    n: &Int32Chunked,
) -> PolarsResult<ListChunked> {
    let mut builder =
        ListStringChunkedBuilder::new(ca.name().clone(), ca.len(), ca.get_values_size());
    if let Some(by) = by {
        if by.is_empty() {
            // Not a regex pattern, but an empty string, we split the string into characters
            for (opt_s, opt_n) in ca.into_iter().zip(n) {
                let n = opt_n
                    .ok_or(polars_err!(ComputeError: "n must be provided for split operation"))?;
                split_chars_helper(&mut builder, opt_s, n);
            }
        } else {
            let re = RegexBuilder::new(by).size_limit(31457280).build().unwrap();

            // Regex is provided, we compile it, and split every string accordingly
            for (opt_s, opt_n) in ca.into_iter().zip(n) {
                let n = opt_n
                    .ok_or(polars_err!(ComputeError: "n must be provided for split operation"))?;
                regex_split_helper(&mut builder, opt_s, &re, n)
            }
        }
        return Ok(builder.finish());
    }

    // No regex pattern provided at all, return a full null column/list
    Ok(ListChunked::full_null_with_dtype(
        ca.name().clone(),
        ca.len(),
        &DataType::String,
    ))
}

// When n is a literal, we can optimize ahead of time to select a better loop
// This is the second least ideal case, where we have a literal n, but not a literal regex pattern
// We need to compile a regex for each row
fn handle_literal_n(
    ca: &StringChunked,
    by: &StringChunked,
    n: Option<i32>,
) -> PolarsResult<ListChunked> {
    let n = n.ok_or(polars_err!(ComputeError: "n must be provided for split operation"))?;

    let mut builder =
        ListStringChunkedBuilder::new(ca.name().clone(), ca.len(), ca.get_values_size());

    // Since the regex is not a literal, we want to use a cache so we don't compile the same regex multiple times
    let mut reg_cache = FastFixedCache::new((ca.len() as f64).sqrt() as usize);
    if n >= 0 {
        ca.into_iter()
            .zip(by)
            .fold(&mut builder, |builder, (opt_s, opt_by)| {
                if let Some(by) = opt_by {
                    if by.is_empty() {
                        split_charsn(builder, opt_s, n as usize);
                    } else {
                        let re = reg_cache.get_or_insert_with(by, |by| {
                            RegexBuilder::new(by).size_limit(31457280).build().unwrap()
                        });
                        regex_splitn(builder, opt_s, re, n as usize);
                    }
                } else {
                    builder.append_null();
                }
                builder
            });
    } else {
        ca.into_iter()
            .zip(by)
            .fold(&mut builder, |builder, (opt_s, opt_by)| {
                if let Some(by) = opt_by {
                    if by.is_empty() {
                        split_chars_no_limit(builder, opt_s)
                    } else {
                        let re = reg_cache.get_or_insert_with(by, |by| {
                            RegexBuilder::new(by).size_limit(31457280).build().unwrap()
                        });
                        regex_split_no_limit(builder, opt_s, re)
                    }
                } else {
                    builder.append_null()
                }

                builder
            });
    }

    Ok(builder.finish())
}

// Least ideal case, where we none of the expressions are literals, and we need to check every row, without optimization
#[cold] // This function is not expected to be called often, if at all, so is marked cold
fn handle_regular_case(
    ca: &StringChunked,
    by: &StringChunked,
    n: &Int32Chunked,
) -> PolarsResult<ListChunked> {
    let mut builder =
        ListStringChunkedBuilder::new(ca.name().clone(), ca.len(), ca.get_values_size());

    // Since the regex is not a literal, we want to use a cache so we don't compile the same regex multiple times
    let mut reg_cache = FastFixedCache::new((ca.len() as f64).sqrt() as usize);
    for ((opt_s, opt_by), opt_n) in ca.into_iter().zip(by).zip(n) {
        let n = opt_n.ok_or(polars_err!(ComputeError: "n must be provided for split operation"))?;
        if n >= 0 {
            if let Some(by) = opt_by {
                if by.is_empty() {
                    split_charsn(&mut builder, opt_s, n as usize);
                    continue;
                }

                let re = reg_cache.get_or_insert_with(by, |by| {
                    RegexBuilder::new(by).size_limit(31457280).build().unwrap()
                });
                regex_splitn(&mut builder, opt_s, re, n as usize);
                continue;
            }

            builder.append_null();
            continue;
        }

        if let Some(by) = opt_by {
            if by.is_empty() {
                split_chars_no_limit(&mut builder, opt_s);
                continue;
            }

            let re = reg_cache.get_or_insert_with(by, |by| {
                RegexBuilder::new(by).size_limit(31457280).build().unwrap()
            });
            regex_split_no_limit(&mut builder, opt_s, re);
            continue;
        }
        builder.append_null();
    }
    Ok(builder.finish())
}

pub fn flarion_split_helper(
    ca: &StringChunked,
    by: &StringChunked,
    n: &Int32Chunked,
) -> PolarsResult<ListChunked> {
    match (by.len(), n.len()) {
        (1, 1) => handle_all_literals(ca, by.get(0), n.get(0)),
        (1, _) => handle_literal_by(ca, by.get(0), n),
        (_, 1) => handle_literal_n(ca, by, n.get(0)),
        (_, _) => handle_regular_case(ca, by, n),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn create_int32_chunked(name: &str, values: Vec<Option<i32>>) -> Int32Chunked {
        Int32Chunked::from_iter_options(PlSmallStr::from_str(name), values.into_iter())
    }

    #[test]
    fn test_split_empty_pattern() {
        let input = create_string_chunked("input", vec![Some("hello"), Some("世界"), None]);
        let by = create_string_chunked("by", vec![Some("")]);
        let n = create_int32_chunked("n", vec![Some(2)]);

        let result = flarion_split_helper(&input, &by, &n).unwrap().into_series();

        assert_eq!(result.len(), 3);
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
        let by = create_string_chunked("by", vec![Some("[,;|]+")]);
        let n = create_int32_chunked("n", vec![Some(-1)]);

        let result = flarion_split_helper(&input, &by, &n).unwrap().into_series();

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
        let by = create_string_chunked("by", vec![Some(",")]);
        let n = create_int32_chunked("n", vec![Some(2)]);

        let result = flarion_split_helper(&input, &by, &n).unwrap().into_series();

        assert_eq!(result.len(), 1);
        assert_eq!(
            result.get(0).unwrap(),
            AnyValue::List(Series::new(PlSmallStr::EMPTY, &["a", "b,c,d"]))
        );
    }

    #[test]
    fn test_split_with_variable_n() {
        let input = create_string_chunked("input", vec![Some("a,b,c"), Some("d,e,f")]);
        let by = create_string_chunked("by", vec![Some(",")]);
        let n = create_int32_chunked("n", vec![Some(2), Some(3)]);

        let result = flarion_split_helper(&input, &by, &n).unwrap().into_series();

        assert_eq!(result.len(), 2);
        assert_eq!(
            result.get(0).unwrap(),
            AnyValue::List(Series::new(PlSmallStr::EMPTY, &["a", "b,c"]))
        );
        assert_eq!(
            result.get(1).unwrap(),
            AnyValue::List(Series::new(PlSmallStr::EMPTY, &["d", "e", "f"]))
        );
    }

    #[test]
    fn test_split_with_variable_pattern() {
        let input = create_string_chunked("input", vec![Some("a,b,c"), Some("d|e|f")]);
        let by = create_string_chunked("by", vec![Some(","), Some(r"\|")]);
        let n = create_int32_chunked("n", vec![Some(-1), Some(-1)]);

        let result = flarion_split_helper(&input, &by, &n).unwrap().into_series();

        assert_eq!(result.len(), 2);
        assert_eq!(
            result.get(0).unwrap(),
            AnyValue::List(Series::new(PlSmallStr::EMPTY, &["a", "b", "c"]))
        );
        assert_eq!(
            result.get(1).unwrap(),
            AnyValue::List(Series::new(PlSmallStr::EMPTY, &["d", "e", "f"]))
        );
    }

    #[test]
    fn test_invalid_regex_pattern() {
        let input = create_string_chunked("input", vec![Some("test")]);
        let by = create_string_chunked("by", vec![Some("[")]); // Invalid regex pattern
        let n = create_int32_chunked("n", vec![Some(-1)]);

        let result = flarion_split_helper(&input, &by, &n);
        assert!(result.is_err());
    }

    #[test]
    fn test_null_n_value() {
        let input = create_string_chunked("input", vec![Some("test")]);
        let by = create_string_chunked("by", vec![Some(",")]);
        let n = create_int32_chunked("n", vec![None]);

        let result = flarion_split_helper(&input, &by, &n);
        assert!(result.is_err());
    }

    #[test]
    fn test_null_pattern() {
        let input = create_string_chunked("input", vec![Some("test")]);
        let by = create_string_chunked("by", vec![None]);
        let n = create_int32_chunked("n", vec![Some(-1)]);

        let result = flarion_split_helper(&input, &by, &n).unwrap().into_series();
        assert_eq!(result.len(), 1);
        assert!(result.get(0).unwrap().is_null());
    }

    #[test]
    fn test_unicode_splitting() {
        let input = create_string_chunked("input", vec![Some("你好，世界")]);
        let by = create_string_chunked("by", vec![Some("，")]);
        let n = create_int32_chunked("n", vec![Some(-1)]);

        let result = flarion_split_helper(&input, &by, &n).unwrap().into_series();

        assert_eq!(result.len(), 1);
        assert_eq!(
            result.get(0).unwrap(),
            AnyValue::List(Series::new(PlSmallStr::EMPTY, &["你好", "世界"]))
        );
    }

    #[test]
    fn test_empty_input() {
        let input = create_string_chunked("input", vec![Some("")]);
        let by = create_string_chunked("by", vec![Some(",")]);
        let n = create_int32_chunked("n", vec![Some(-1)]);

        let result = flarion_split_helper(&input, &by, &n).unwrap().into_series();

        assert_eq!(result.len(), 1);
        assert_eq!(
            result.get(0).unwrap(),
            AnyValue::List(Series::new(PlSmallStr::EMPTY, &[""]))
        );
    }

    #[test]
    fn test_mixed_scenarios() {
        let input =
            create_string_chunked("input", vec![Some("a,b"), Some("c|d"), None, Some("e;f")]);
        let by = create_string_chunked("by", vec![Some(","), Some(r"\|"), Some(";"), Some(";")]);
        let n = create_int32_chunked("n", vec![Some(2), Some(-1), Some(1), Some(2)]);

        let result = flarion_split_helper(&input, &by, &n).unwrap().into_series();

        assert_eq!(result.len(), 4);
        assert_eq!(
            result.get(0).unwrap(),
            AnyValue::List(Series::new(PlSmallStr::EMPTY, &["a", "b"]))
        );
        assert_eq!(
            result.get(1).unwrap(),
            AnyValue::List(Series::new(PlSmallStr::EMPTY, &["c", "d"]))
        );
        assert!(result.get(2).unwrap().is_null());
        assert_eq!(
            result.get(3).unwrap(),
            AnyValue::List(Series::new(PlSmallStr::EMPTY, &["e", "f"]))
        );
    }

    // Helper function tests
    #[test]
    fn test_split_chars_helper() {
        let mut builder = ListStringChunkedBuilder::new(PlSmallStr::EMPTY, 2, 10);

        // Test with positive n
        split_chars_helper(&mut builder, Some("hello"), 2);
        // Test with negative n
        split_chars_helper(&mut builder, Some("world"), -1);

        let result = builder.finish().into_series();
        assert_eq!(
            result.get(0).unwrap(),
            AnyValue::List(Series::new(PlSmallStr::EMPTY, &["h", "e"]))
        );
        assert_eq!(
            result.get(1).unwrap(),
            AnyValue::List(Series::new(PlSmallStr::EMPTY, &["w", "o", "r", "l", "d"]))
        );
    }

    #[test]
    fn test_regex_split_helper() {
        let mut builder = ListStringChunkedBuilder::new(PlSmallStr::EMPTY, 2, 10);
        let re = RegexBuilder::new(",").size_limit(31457280).build().unwrap();

        // Test with positive n
        regex_split_helper(&mut builder, Some("a,b,c"), &re, 2);
        // Test with negative n
        regex_split_helper(&mut builder, Some("d,e,f"), &re, -1);

        let result = builder.finish().into_series();
        assert_eq!(
            result.get(0).unwrap(),
            AnyValue::List(Series::new(PlSmallStr::EMPTY, &["a", "b,c"]))
        );
        assert_eq!(
            result.get(1).unwrap(),
            AnyValue::List(Series::new(PlSmallStr::EMPTY, &["d", "e", "f"]))
        );
    }
}
