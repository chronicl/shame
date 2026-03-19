use super::{mat::mat, scalar_type::ScalarTypeFp, AsAny};
use crate::{frontend::rust_types::len::*, impl_ops, frontend::rust_types::vec::vec};
use std::ops::*;

impl_ops! {
    <T: ScalarTypeFp, C: Len2, R: Len2> Add !op: add(l: mat<T, C, R>, r: mat<T, C, R>) -> mat<T, C, R>: {op(l.as_any(), r.as_any()).into()};
    <T: ScalarTypeFp, C: Len2, R: Len2> Sub !op: sub(l: mat<T, C, R>, r: mat<T, C, R>) -> mat<T, C, R>: {op(l.as_any(), r.as_any()).into()};

    <T: ScalarTypeFp, C: Len2, R: Len2> Mul !op: mul(l: mat<T, C, R>, r: vec<T, x1>) -> mat<T, C, R>: {op(l.as_any(), r.as_any()).into()};
    <T: ScalarTypeFp, C: Len2, R: Len2> Mul !op: mul(l: vec<T, x1>, r: mat<T, C, R>) -> mat<T, C, R>: {op(l.as_any(), r.as_any()).into()};

    <T: ScalarTypeFp, C: Len2, R: Len2> Mul !op: mul(l: mat<T, C, R>, r: vec<T, C>) -> vec<T, R>: {op(l.as_any(), r.as_any()).into()};
    <T: ScalarTypeFp, C: Len2, R: Len2> Mul !op: mul(l: vec<T, R>, r: mat<T, C, R>) -> vec<T, C>: {op(l.as_any(), r.as_any()).into()};

    <C: Len2, R: Len2, K: Len2, T: ScalarTypeFp>
    Mul !op: mul(l: mat<T, K, R>, r: mat<T, C, K>) -> mat<T, C, R>: {op(l.as_any(), r.as_any()).into()};
}
