use std::borrow::Cow;
use std::hash::{Hash, Hasher};

use num_traits::ToBytes;
#[cfg(feature = "serde-lazy")]
use serde::{Deserialize, Serialize};

use super::DataType;
use crate::datatypes::{AnyValue, PolarsNumericType};
use crate::prelude::Array;
use crate::series::Series;

#[derive(Clone, PartialEq, Debug)]
#[cfg_attr(feature = "serde-lazy", derive(Serialize, Deserialize))]
pub enum LambdaExpression {
    Null,
    Boolean(bool),
    Int8(i8),
    Int16(i16),
    Int32(i32),
    Int64(i64),
    Float32(f32),
    Float64(f64),
    BinaryBlob(Vec<u8>),
    StaticStr(Cow<'static, str>),
    Variable(usize),
    GreaterThan(Box<Self>, Box<Self>),
    LessThan(Box<Self>, Box<Self>),
    IfThenElse(Box<Self>, Box<Self>, Box<Self>),
    Length(Box<Self>),
    CaseWhen(Vec<(Self, Self)>, Box<Self>),
    Substring(Box<Self>, Box<Self>, Box<Self>),
    Instr(Box<Self>, Box<Self>),
    Add(Box<Self>, Box<Self>),
}

impl Eq for LambdaExpression {}

impl Hash for LambdaExpression {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            LambdaExpression::Null => 0.hash(state),
            LambdaExpression::Boolean(v) => v.hash(state),
            LambdaExpression::Int8(v) => v.hash(state),
            LambdaExpression::Int16(v) => v.hash(state),
            LambdaExpression::Int32(v) => v.hash(state),
            LambdaExpression::Int64(v) => v.hash(state),
            LambdaExpression::Float32(v) => v.to_le_bytes().hash(state),
            LambdaExpression::Float64(v) => v.to_le_bytes().hash(state),
            LambdaExpression::BinaryBlob(v) => v.hash(state),
            LambdaExpression::StaticStr(v) => v.hash(state),
            LambdaExpression::Variable(v) => v.hash(state),
            LambdaExpression::GreaterThan(first, second) => {
                first.hash(state);
                second.hash(state);
            },
            LambdaExpression::LessThan(first, second) => {
                first.hash(state);
                second.hash(state);
            },
            LambdaExpression::IfThenElse(first, second, third) => {
                first.hash(state);
                second.hash(state);
                third.hash(state);
            },
            LambdaExpression::Length(v) => v.hash(state),
            LambdaExpression::CaseWhen(first, second) => {
                first.hash(state);
                second.hash(state);
            },
            LambdaExpression::Substring(first, second, third) => {
                first.hash(state);
                second.hash(state);
                third.hash(state);
            },
            LambdaExpression::Instr(first, second) => {
                first.hash(state);
                second.hash(state);
            },
            LambdaExpression::Add(first, second) => {
                first.hash(state);
                second.hash(state);
            },
        }
    }
}

fn substring<'a>(s: AnyValue<'a>, from: AnyValue<'a>, len: AnyValue<'a>) -> AnyValue<'a> {
    fn convert_indexes(from: i32, len: i32, s_len: usize) -> (usize, usize) {
        let mut from = from as i64;
        let len = len as i64; // this can be int32_max
        let s_len = s_len as i64;
        if from == 0 {
            panic!("From can't be zero in 1 based indexation")
        }
        if from > 0 {
            from -= 1;
        }
        if from < 0 {
            from = from + s_len;
        }
        let to = (from + len).min(s_len);
        (from.max(0) as usize, to.max(0) as usize)
    }
    unsafe {
        match (s, from, len) {
            (AnyValue::String(s), AnyValue::Int32(from), AnyValue::Int32(len)) => {
                let (to, from) = convert_indexes(from, len, s.len());
                let result = &s[from.max(0) as usize..to.max(0) as usize];
                AnyValue::String(result)
            },
            (AnyValue::Binary(bin), AnyValue::Int32(from), AnyValue::Int32(len)) => {
                let (to, from) = convert_indexes(from, len, bin.len());
                let result = &bin[from.max(0) as usize..to.max(0) as usize];
                AnyValue::Binary(result)
            },
            _ => std::hint::unreachable_unchecked(), // tell the compiler it's unreachable
        }
    }
}

impl LambdaExpression {
    #[inline]
    pub(crate) fn eval_array<'a>(&'a self, args: &'a [&'a dyn Array]) -> AnyValue<'a> {
        match self {
            LambdaExpression::Null => AnyValue::Null,
            LambdaExpression::Boolean(v) => AnyValue::Boolean(*v),
            LambdaExpression::Int8(v) => AnyValue::Int8(*v),
            LambdaExpression::Int16(v) => AnyValue::Int16(*v),
            LambdaExpression::Int32(v) => AnyValue::Int32(*v),
            LambdaExpression::Int64(v) => AnyValue::Int64(*v),
            LambdaExpression::Float32(v) => AnyValue::Float32(*v),
            LambdaExpression::Float64(v) => AnyValue::Float64(*v),
            LambdaExpression::BinaryBlob(v) => AnyValue::Binary(v),
            LambdaExpression::StaticStr(v) => AnyValue::String(v),
            LambdaExpression::Variable(idx) => {
                let arr = args[*idx];
                let series = Series::from_arrow("".into(), arr.to_boxed())
                    .expect("could not convert array to series");
                AnyValue::List(series)
            },
            LambdaExpression::GreaterThan(left, right) => {
                left.eval_array(args).gt(&right.eval_array(args)).into()
            },
            LambdaExpression::LessThan(left, right) => {
                left.eval_array(args).lt(&right.eval_array(args)).into()
            },
            LambdaExpression::IfThenElse(cond, truthy, falsy) => {
                if unsafe {
                    match cond.eval_array(args) {
                        AnyValue::Boolean(v) => v,
                        _ => std::hint::unreachable_unchecked(), // tell the compiler it's unreachable
                    }
                } {
                    truthy.eval_array(args)
                } else {
                    falsy.eval_array(args)
                }
            },
            LambdaExpression::Length(expr) => match expr.eval_array(args) {
                AnyValue::Null => AnyValue::Null,
                AnyValue::List(arr) => AnyValue::Int32(arr.len() as i32),
                _ => AnyValue::Int32(1),
            },
            LambdaExpression::CaseWhen(cases, otherwise) => {
                for (cond, value) in cases {
                    if unsafe {
                        match cond.eval_array(args) {
                            AnyValue::Boolean(v) => v,
                            _ => std::hint::unreachable_unchecked(), // tell the compiler it's unreachable
                        }
                    } {
                        return value.eval_array(args);
                    }
                }
                otherwise.eval_array(args)
            },
            LambdaExpression::Substring(s, from, len) => {
                let s = s.eval_array(args);
                let from = from.eval_array(args).cast(&DataType::Int32);
                let len = len.eval_array(args).cast(&DataType::Int32);
                substring(s, from, len)
            },
            LambdaExpression::Instr(s, pat) => {
                let s = s.eval_array(args);
                let pat = pat.eval_array(args);
                unsafe {
                    match (s, pat) {
                        (AnyValue::String(s), AnyValue::String(pat)) => {
                            AnyValue::Int32(s.find(pat).map(|x| x + 1).unwrap_or(0) as i32)
                        },
                        _ => std::hint::unreachable_unchecked(), // tell the compiler it's unreachable
                    }
                }
            },
            LambdaExpression::Add(left, right) => {
                let left = left.eval_array(args);
                let right = right.eval_array(args);
                left.add(&right.cast(&left.dtype()))
            },
        }
    }

    #[inline]
    pub(crate) fn eval_numeric<T: PolarsNumericType>(&self, args: &[&T::Native]) -> AnyValue {
        match self {
            LambdaExpression::Null => AnyValue::Null,
            LambdaExpression::Boolean(v) => AnyValue::Boolean(*v),
            LambdaExpression::Int8(v) => AnyValue::Int8(*v),
            LambdaExpression::Int16(v) => AnyValue::Int16(*v),
            LambdaExpression::Int32(v) => AnyValue::Int32(*v),
            LambdaExpression::Int64(v) => AnyValue::Int64(*v),
            LambdaExpression::Float32(v) => AnyValue::Float32(*v),
            LambdaExpression::Float64(v) => AnyValue::Float64(*v),
            LambdaExpression::BinaryBlob(v) => AnyValue::Binary(v),
            LambdaExpression::StaticStr(v) => AnyValue::String(v),
            LambdaExpression::Variable(idx) => (*args[*idx]).into(),
            LambdaExpression::GreaterThan(left, right) => {
                AnyValue::Boolean(left.eval_numeric::<T>(args) > right.eval_numeric::<T>(args))
            },
            LambdaExpression::LessThan(left, right) => {
                AnyValue::Boolean(left.eval_numeric::<T>(args) < right.eval_numeric::<T>(args))
            },
            LambdaExpression::IfThenElse(cond, truthy, falsy) => {
                if unsafe {
                    match cond.eval_numeric::<T>(args) {
                        AnyValue::Boolean(v) => v,
                        _ => std::hint::unreachable_unchecked(), // tell the compiler it's unreachable
                    }
                } {
                    truthy.eval_numeric::<T>(args)
                } else {
                    falsy.eval_numeric::<T>(args)
                }
            },
            LambdaExpression::Length(expr) => match expr.eval_numeric::<T>(args) {
                AnyValue::Null => AnyValue::Null,
                _ => AnyValue::Int32(1),
            },
            LambdaExpression::CaseWhen(cases, otherwise) => {
                for (cond, value) in cases {
                    if unsafe {
                        match cond.eval_numeric::<T>(args) {
                            AnyValue::Boolean(v) => v,
                            _ => std::hint::unreachable_unchecked(), // tell the compiler it's unreachable
                        }
                    } {
                        return value.eval_numeric::<T>(args);
                    }
                }
                otherwise.eval_numeric::<T>(args)
            },
            LambdaExpression::Substring(s, from, len) => {
                let s = s.eval_numeric::<T>(args);
                let from = from.eval_numeric::<T>(args).cast(&DataType::Int32);
                let len = len.eval_numeric::<T>(args).cast(&DataType::Int32);
                substring(s, from, len)
            },
            LambdaExpression::Instr(s, pat) => {
                let s = s.eval_numeric::<T>(args);
                let pat = pat.eval_numeric::<T>(args);
                unsafe {
                    match (s, pat) {
                        (AnyValue::String(s), AnyValue::String(pat)) => {
                            AnyValue::Int32(s.find(pat).map(|x| x + 1).unwrap_or(0) as i32)
                        },
                        _ => std::hint::unreachable_unchecked(), // tell the compiler it's unreachable
                    }
                }
            },
            LambdaExpression::Add(left, right) => {
                let left = left.eval_numeric::<T>(args);
                let right = right.eval_numeric::<T>(args);
                left.add(&right.cast(&left.dtype()))
            },
        }
    }

    #[inline]
    pub(crate) fn eval_bool(&self, args: &[&bool]) -> AnyValue {
        match self {
            LambdaExpression::Null => AnyValue::Null,
            LambdaExpression::Boolean(v) => AnyValue::Boolean(*v),
            LambdaExpression::Int8(v) => AnyValue::Int8(*v),
            LambdaExpression::Int16(v) => AnyValue::Int16(*v),
            LambdaExpression::Int32(v) => AnyValue::Int32(*v),
            LambdaExpression::Int64(v) => AnyValue::Int64(*v),
            LambdaExpression::Float32(v) => AnyValue::Float32(*v),
            LambdaExpression::Float64(v) => AnyValue::Float64(*v),
            LambdaExpression::BinaryBlob(v) => AnyValue::Binary(v),
            LambdaExpression::StaticStr(v) => AnyValue::String(v),
            LambdaExpression::Variable(idx) => (*args[*idx]).into(),
            LambdaExpression::GreaterThan(left, right) => {
                AnyValue::Boolean(left.eval_bool(args) > right.eval_bool(args))
            },
            LambdaExpression::LessThan(left, right) => {
                AnyValue::Boolean(left.eval_bool(args) < right.eval_bool(args))
            },
            LambdaExpression::IfThenElse(cond, truthy, falsy) => {
                if unsafe {
                    match cond.eval_bool(args) {
                        AnyValue::Boolean(v) => v,
                        _ => std::hint::unreachable_unchecked(), // tell the compiler it's unreachable
                    }
                } {
                    truthy.eval_bool(args)
                } else {
                    falsy.eval_bool(args)
                }
            },
            LambdaExpression::Length(expr) => match expr.eval_bool(args) {
                AnyValue::Null => AnyValue::Null,
                _ => AnyValue::Int32(1),
            },
            LambdaExpression::CaseWhen(cases, otherwise) => {
                for (cond, value) in cases {
                    if unsafe {
                        match cond.eval_bool(args) {
                            AnyValue::Boolean(v) => v,
                            _ => std::hint::unreachable_unchecked(), // tell the compiler it's unreachable
                        }
                    } {
                        return value.eval_bool(args);
                    }
                }
                otherwise.eval_bool(args)
            },
            LambdaExpression::Substring(s, from, len) => {
                let s = s.eval_bool(args);
                let from = from.eval_bool(args).cast(&DataType::Int32);
                let len = len.eval_bool(args).cast(&DataType::Int32);
                substring(s, from, len)
            },
            LambdaExpression::Instr(s, pat) => {
                let s = s.eval_bool(args);
                let pat = pat.eval_bool(args);
                unsafe {
                    match (s, pat) {
                        (AnyValue::String(s), AnyValue::String(pat)) => {
                            AnyValue::Int32(s.find(pat).map(|x| x + 1).unwrap_or(0) as i32)
                        },
                        _ => std::hint::unreachable_unchecked(), // tell the compiler it's unreachable
                    }
                }
            },
            LambdaExpression::Add(left, right) => {
                let left = left.eval_bool(args);
                let right = right.eval_bool(args);
                left.add(&right.cast(&left.dtype()))
            },
        }
    }

    #[inline]
    pub(crate) fn eval_slice<'a>(&'a self, args: &'a [&'a [u8]]) -> AnyValue<'a> {
        match self {
            LambdaExpression::Null => AnyValue::Null,
            LambdaExpression::Boolean(v) => AnyValue::Boolean(*v),
            LambdaExpression::Int8(v) => AnyValue::Int8(*v),
            LambdaExpression::Int16(v) => AnyValue::Int16(*v),
            LambdaExpression::Int32(v) => AnyValue::Int32(*v),
            LambdaExpression::Int64(v) => AnyValue::Int64(*v),
            LambdaExpression::Float32(v) => AnyValue::Float32(*v),
            LambdaExpression::Float64(v) => AnyValue::Float64(*v),
            LambdaExpression::BinaryBlob(v) => AnyValue::Binary(v),
            LambdaExpression::StaticStr(v) => AnyValue::String(v),
            LambdaExpression::Variable(idx) => AnyValue::Binary(args[*idx]),
            LambdaExpression::GreaterThan(left, right) => {
                left.eval_slice(args).gt(&right.eval_slice(args)).into()
            },
            LambdaExpression::LessThan(left, right) => {
                left.eval_slice(args).lt(&right.eval_slice(args)).into()
            },
            LambdaExpression::IfThenElse(cond, truthy, falsy) => {
                if unsafe {
                    match cond.eval_slice(args) {
                        AnyValue::Boolean(v) => v,
                        _ => std::hint::unreachable_unchecked(), // tell the compiler it's unreachable
                    }
                } {
                    truthy.eval_slice(args)
                } else {
                    falsy.eval_slice(args)
                }
            },
            LambdaExpression::Length(expr) => match expr.eval_slice(args) {
                AnyValue::Null => AnyValue::Null,
                AnyValue::Binary(bytes) => AnyValue::Int32(bytes.len() as i32),
                AnyValue::String(s) => AnyValue::Int32(s.len() as i32),
                AnyValue::List(arr) => AnyValue::Int32(arr.len() as i32),
                _ => AnyValue::Int32(1),
            },
            LambdaExpression::CaseWhen(cases, otherwise) => {
                for (cond, value) in cases {
                    if unsafe {
                        match cond.eval_slice(args) {
                            AnyValue::Boolean(v) => v,
                            _ => std::hint::unreachable_unchecked(), // tell the compiler it's unreachable
                        }
                    } {
                        return value.eval_slice(args);
                    }
                }
                otherwise.eval_slice(args)
            },
            LambdaExpression::Substring(s, from, len) => {
                let s = s.eval_slice(args);
                let from = from.eval_slice(args).cast(&DataType::Int32);
                let len = len.eval_slice(args).cast(&DataType::Int32);
                substring(s, from, len)
            },
            LambdaExpression::Instr(s, pat) => {
                let s = s.eval_slice(args);
                let pat = pat.eval_slice(args);
                unsafe {
                    match (s, pat) {
                        (AnyValue::String(s), AnyValue::String(pat)) => {
                            AnyValue::Int32(s.find(pat).map(|x| x + 1).unwrap_or(0) as i32)
                        },
                        _ => std::hint::unreachable_unchecked(), // tell the compiler it's unreachable
                    }
                }
            },
            LambdaExpression::Add(left, right) => {
                let left = left.eval_slice(args);
                let right = right.eval_slice(args);
                left.add(&right.cast(&left.dtype()))
            },
        }
    }

    pub(crate) fn eval_any<'a>(&'a self, args: &[AnyValue<'a>]) -> AnyValue<'a> {
        match self {
            LambdaExpression::Null => AnyValue::Null,
            LambdaExpression::Boolean(v) => AnyValue::Boolean(*v),
            LambdaExpression::Int8(v) => AnyValue::Int8(*v),
            LambdaExpression::Int16(v) => AnyValue::Int16(*v),
            LambdaExpression::Int32(v) => AnyValue::Int32(*v),
            LambdaExpression::Int64(v) => AnyValue::Int64(*v),
            LambdaExpression::Float32(v) => AnyValue::Float32(*v),
            LambdaExpression::Float64(v) => AnyValue::Float64(*v),
            LambdaExpression::BinaryBlob(v) => AnyValue::Binary(v),
            LambdaExpression::StaticStr(v) => AnyValue::String(v),
            LambdaExpression::Variable(idx) => args[*idx].clone(),
            LambdaExpression::GreaterThan(left, right) => {
                left.eval_any(args).gt(&right.eval_any(args)).into()
            },
            LambdaExpression::LessThan(left, right) => {
                left.eval_any(args).lt(&right.eval_any(args)).into()
            },
            LambdaExpression::IfThenElse(cond, truthy, falsy) => {
                if unsafe {
                    match cond.eval_any(args) {
                        AnyValue::Boolean(v) => v,
                        _ => std::hint::unreachable_unchecked(), // tell the compiler it's unreachable
                    }
                } {
                    truthy.eval_any(args)
                } else {
                    falsy.eval_any(args)
                }
            },
            LambdaExpression::Length(expr) => match expr.eval_any(args) {
                AnyValue::Null => AnyValue::Null,
                AnyValue::Binary(bytes) => AnyValue::Int32(bytes.len() as i32),
                AnyValue::String(s) => AnyValue::Int32(s.len() as i32),
                AnyValue::List(arr) => AnyValue::Int32(arr.len() as i32),
                _ => AnyValue::Int32(1),
            },
            LambdaExpression::CaseWhen(cases, otherwise) => {
                for (cond, value) in cases {
                    if unsafe {
                        match cond.eval_any(args) {
                            AnyValue::Boolean(v) => v,
                            _ => std::hint::unreachable_unchecked(), // tell the compiler it's unreachable
                        }
                    } {
                        return value.eval_any(args);
                    }
                }
                otherwise.eval_any(args)
            },
            LambdaExpression::Substring(s, from, len) => {
                let s = s.eval_any(args);
                let from = from.eval_any(args).cast(&DataType::Int32);
                let len = len.eval_any(args).cast(&DataType::Int32);
                substring(s, from, len)
            },
            LambdaExpression::Instr(s, pat) => {
                let s = s.eval_any(args);
                let pat = pat.eval_any(args);
                unsafe {
                    match (s, pat) {
                        (AnyValue::String(s), AnyValue::String(pat)) => {
                            AnyValue::Int32(s.find(pat).map(|x| x + 1).unwrap_or(0) as i32)
                        },
                        _ => std::hint::unreachable_unchecked(), // tell the compiler it's unreachable
                    }
                }
            },
            LambdaExpression::Add(left, right) => {
                let left = left.eval_any(args);
                let right = right.eval_any(args);
                left.add(&right.cast(&left.dtype()))
            },
        }
    }

    // None encode that type same as input value
    pub fn return_type(&self) -> Option<DataType> {
        match self {
            LambdaExpression::Null => Some(DataType::Null),
            LambdaExpression::Boolean(_) => Some(DataType::Boolean),
            LambdaExpression::Int8(_) => Some(DataType::Int8),
            LambdaExpression::Int16(_) => Some(DataType::Int16),
            LambdaExpression::Int32(_) => Some(DataType::Int32),
            LambdaExpression::Int64(_) => Some(DataType::Int64),
            LambdaExpression::Float32(_) => Some(DataType::Float32),
            LambdaExpression::Float64(_) => Some(DataType::Float64),
            LambdaExpression::BinaryBlob(_) => Some(DataType::Binary),
            LambdaExpression::StaticStr(_) => Some(DataType::String),
            LambdaExpression::Variable(_) => None,
            LambdaExpression::GreaterThan(_, _) => Some(DataType::Boolean),
            LambdaExpression::LessThan(_, _) => Some(DataType::Boolean),
            LambdaExpression::IfThenElse(_, then, _) => then.return_type(),
            LambdaExpression::Length(_) => Some(DataType::Int32),
            LambdaExpression::CaseWhen(cases, _) => cases
                .iter()
                .find_map(|pair| {
                    if let Some(dtype) = pair.1.return_type() {
                        dtype.is_null().then_some(dtype).map(Some)
                    } else {
                        Some(None)
                    }
                })
                .unwrap_or(Some(DataType::Null)),
            LambdaExpression::Substring(s, _, _) => s.return_type(), // substring is used for string and byte arrays
            LambdaExpression::Instr(_, _) => Some(DataType::Int32),
            LambdaExpression::Add(left, _) => left.return_type(),
        }
    }
}
