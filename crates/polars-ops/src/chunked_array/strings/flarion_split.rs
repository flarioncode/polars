use arrow::array::ValueSize;
use regex::RegexBuilder;

use super::*;

pub fn flarion_split_helper(
    ca: &StringChunked,
    pattern: &str,
    n: i32,
) -> PolarsResult<ListChunked> {
    let mut builder =
        ListStringChunkedBuilder::new(ca.name().clone(), ca.len(), ca.get_values_size());

    // When the regex pattern is an empty string, we split the string into characters
    if pattern.is_empty() {
        if n >= 0 {
            ca.into_iter().fold(&mut builder, |builder, opt_s| {
                match opt_s {
                    Some(s) => builder.append_values_iter(splitn_chars(s, n as usize, false)),
                    None => builder.append_null(),
                }
                builder
            });
        } else {
            ca.into_iter().fold(&mut builder, |builder, opt_s| {
                match opt_s {
                    Some(s) => builder.append_values_iter(split_chars(s)),
                    None => builder.append_null(),
                };
                builder
            });
        };
    } else {
        let re = RegexBuilder::new(pattern).size_limit(31457280).build()?;

        // When n is already a literal, we can check it ahead, and optimize for a negative number(no limit)
        if n >= 0 {
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

    #[test]
    fn test_split_empty_pattern() {
        let input = create_string_chunked("input", vec![Some("hello"), Some("世界"), None]);

        let result = flarion_split_helper(&input, "", 2).unwrap().into_series();

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

        let result = flarion_split_helper(&input, ",", -1).unwrap().into_series();

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
}
