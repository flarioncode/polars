use crate::compute::cast::primitive_to::SerPrimitive;

/// Spark as primitive
/// This is a copy of the `AsPrimitive` trait from the `num-traits` crate but how Spark does it.
///
/// we require this trait to implement num_traits::AsPrimitive to make sure we are aligned
pub trait SparkAsPrimitive<T>: 'static + Copy + num_traits::AsPrimitive<T>
where
    T: 'static + Copy,
{
    /// Convert a value to another, using the `as` operator.
    fn as_(self) -> T;
}

// Implementation inspired by num-traits
macro_rules! impl_x_as_t {
    (@ $(#[$cfg:meta])* $X: ty => $T: ty ) => {
        $(#[$cfg])*
        impl SparkAsPrimitive<$T> for $X {
            #[inline] fn as_(self) -> $T { self as $T }
        }
    };
    ({ $( $X: ty ),* } => $T: ty) => {$(
        impl_x_as_t!(@ $X => $T);
    )*};
}

macro_rules! impl_x_as_int_as_t {
    (@ $(#[$cfg:meta])* $X: ty => $T: ty ) => {
        $(#[$cfg])*
        impl SparkAsPrimitive<$T> for $X {
            #[inline] fn as_(self) -> $T { self as i32 as $T }
        }
    };
    ({ $( $X: ty ),* } => $T: ty) => {$(
        impl_x_as_int_as_t!(@ $X => $T);
    )*};
}

macro_rules! impl_unsigned_as_t {
    ( $T: ty ) => {
        impl_x_as_t!({ u8, u16, u32, u64, u128, usize } => $T);
    };
}
macro_rules! impl_signed_as_t {
    ( $T: ty ) => {
        impl_x_as_t!({ i8, i16, i32, i64, i128, isize } => $T);
    };
}

macro_rules! impl_all_and_more_as_t {
    ({ $( $X: ty ),* } => $T: ty ) => {
        impl_x_as_t!({ $( $X ),* } => $T);
        impl_unsigned_as_t!($T);
        impl_signed_as_t!($T);
    };
}

// X as i8
impl_unsigned_as_t!(i8);

// convert to int first
// Spark code: https://github.com/apache/spark/blob/6729992c76fc59ab07f63f97a9858691274447d0/sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/expressions/Cast.scala#L1047-L1048
impl_x_as_int_as_t!({
    i8, i16, i32, i64, i128, isize, // i8 should be optimized away
    f32, f64
} => i8);
impl_x_as_t!({char, bool} => i8);

// X as i16
impl_unsigned_as_t!(i16);
// convert to int first
// Spark code: https://github.com/apache/spark/blob/6729992c76fc59ab07f63f97a9858691274447d0/sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/expressions/Cast.scala#L1047-L1048
impl_x_as_int_as_t!({
    i8, i16, i32, i64, i128, isize, // i8 and i16 should be optimized away
    f32, f64
} => i16);
impl_x_as_t!({char, bool} => i16);

impl_all_and_more_as_t!({ f32, f64, char, bool } => i32);
impl_all_and_more_as_t!({ f32, f64, char, bool } => i64);
impl_all_and_more_as_t!({ f32, f64, char, bool } => i128);
impl_all_and_more_as_t!({ f32, f64, char, bool } => isize);
impl_all_and_more_as_t!({ f32, f64, char, bool } => u8);
impl_all_and_more_as_t!({ f32, f64, char, bool } => u16);
impl_all_and_more_as_t!({ f32, f64, char, bool } => u32);
impl_all_and_more_as_t!({ f32, f64, char, bool } => u64);
impl_all_and_more_as_t!({ f32, f64, char, bool } => u128);
impl_all_and_more_as_t!({ f32, f64, char, bool } => usize);
impl_all_and_more_as_t!({ f32, f64 } => f32);
impl_all_and_more_as_t!({ f32, f64 } => f64);
impl_x_as_t!({ u8 } => char);

// ----------------------------------------------------------------------------
// Cast from float/double to string

// If the absolute number is less than 10,000,000 and greater or equal than 0.001, the
// result is expressed without scientific notation with at least one digit on either side of
// the decimal point. Otherwise, Spark uses a mantissa followed by E and an
// exponent. The mantissa has an optional leading minus sign followed by one digit to the
// left of the decimal point, and the minimal number of digits greater than zero to the
// right. The exponent has and optional leading minus sign.
// source: https://docs.databricks.com/en/sql/language-manual/functions/cast.html

const LOWER_SCIENTIFIC_BOUND_F32: f32 = 0.001;
const UPPER_SCIENTIFIC_BOUND_F32: f32 = 10000000.0;
const LOWER_SCIENTIFIC_BOUND_F64: f64 = 0.001;
const UPPER_SCIENTIFIC_BOUND_F64: f64 = 10000000.0;

macro_rules! cast_float_to_string {
    ($type:ty, $lower_bound: expr, $upper_bound: expr) => {

    impl SerPrimitive for $type {
        fn write(f: &mut Vec<u8>, val: Self) -> usize
        where
            Self: Sized,
        {
            if val == <$type>::INFINITY {
                f.extend_from_slice(b"Infinity");
                return 8;
            }

            if val == <$type>::NEG_INFINITY {
                f.extend_from_slice(b"-Infinity");
                return 9;
            }

            if (val.abs() < $upper_bound && val.abs() >= $lower_bound) || val.abs() == 0.0 {
                let trailing_zero = if val.fract() == 0.0 { ".0" } else { "" };

                let value = format!("{val}{trailing_zero}");
                f.extend_from_slice(value.as_bytes());
                return value.len();
            }

            if val.abs() >= $upper_bound || val.abs() < $lower_bound {
                let formatted = format!("{val:E}");

                if formatted.contains(".") {
                    f.extend_from_slice(formatted.as_bytes());

                    formatted.len()
                } else {
                    // `formatted` is already in scientific notation and can be split up by E
                    // in order to add the missing trailing 0 which gets removed for numbers with a fraction of 0.0
                    let prepare_number: Vec<&str> = formatted.split("E").collect();

                    let coefficient = prepare_number[0];

                    let exponent = prepare_number[1];

                    let str = format!("{coefficient}.0E{exponent}");
                    f.extend_from_slice(str.as_bytes());

                    str.len()
                }
            } else {
                // Fallback
                let mut buffer = ryu::Buffer::new();
                let value = buffer.format(val);
                f.extend_from_slice(value.as_bytes());
                value.len()
            }
        }
    }
};
    }


cast_float_to_string!(f32, LOWER_SCIENTIFIC_BOUND_F32, UPPER_SCIENTIFIC_BOUND_F32);
cast_float_to_string!(f64, LOWER_SCIENTIFIC_BOUND_F64, UPPER_SCIENTIFIC_BOUND_F64);
