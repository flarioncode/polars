/// Spark as primitive
/// This is a copy of the `AsPrimitive` trait from the `num-traits` crate but how Spark does it.
///
/// we require this trait to implement num_traits::AsPrimitive to make sure we are aligned
pub trait SparkAsPrimitive<T>: 'static + Copy + num_traits::AsPrimitive::<T>
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
