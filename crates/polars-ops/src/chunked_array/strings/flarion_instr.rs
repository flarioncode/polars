use polars_core::prelude::flarion_funcs::flarion_get_char_position;
use polars_core::prelude::{Int32Chunked, StringChunked};
use polars_core::series::{IntoSeries, Series};
use polars_error::{polars_bail, PolarsError, PolarsResult};

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
        polars_bail!(
            ComputeError:
            "Lengths of 'string' and 'pattern' series must be equal or pattern series must have length 1",
        );
    };

    Ok(result)
}
