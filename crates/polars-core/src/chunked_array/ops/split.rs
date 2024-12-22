pub struct SplitNChars<'a> {
    s: &'a str,
    n: usize,
    keep_remainder: bool,
}

impl<'a> Iterator for SplitNChars<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        let single_char_limit = if self.keep_remainder { 2 } else { 1 };
        if self.n >= single_char_limit {
            self.n -= 1;
            let ch = self.s.chars().next()?;
            let first;
            (first, self.s) = self.s.split_at(ch.len_utf8());
            Some(first)
        } else if self.n == 1 && !self.s.is_empty() {
            self.n -= 1;
            Some(self.s)
        } else {
            None
        }
    }
}

/// Splits a string into substrings consisting of single characters.
///
/// Returns at most n strings, where the last string is the entire remainder
/// of the string if keep_remainder is True, and just the nth character otherwise.
#[cfg(feature = "dtype-struct")]
pub fn splitn_chars(s: &str, n: usize, keep_remainder: bool) -> SplitNChars<'_> {
    SplitNChars {
        s,
        n,
        keep_remainder,
    }
}

/// Splits a string into substrings consisting of single characters.
pub fn split_chars(s: &str) -> SplitNChars<'_> {
    SplitNChars {
        s,
        n: usize::MAX,
        keep_remainder: false,
    }
}
