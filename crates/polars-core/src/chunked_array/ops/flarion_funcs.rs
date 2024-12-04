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

        let end_idx = (distance_to_advance as i32).wrapping_sub(len) as usize; // We know len is positive

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
