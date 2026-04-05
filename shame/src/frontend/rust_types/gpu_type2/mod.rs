#![allow(missing_docs)]

use std::{cell::RefCell, num::NonZeroU32, sync::Arc};

use crate::{
    U32PowerOf2,
    ir::{self, Atomic, Matrix, Repr, Vector},
};

pub mod align_size;

pub(crate) const PACKED_ALIGN: U32PowerOf2 = U32PowerOf2::_1;

pub trait GpuType2 {
    const TYPE: LayoutType<'static>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LayoutType<'a> {
    Sized(SizedType<'a>),
    UnsizedStruct(UnsizedStruct<'a>),
    RuntimeSizedArray(RuntimeSizedArray<'a>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SizedType<'a> {
    Vector(Vector),
    Matrix(Matrix),
    Atomic(Atomic),
    // &'a required to avoid recursion
    Array(SizedArray<'a>),
    Struct(SizedStruct<'a>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SizedArray<'a> {
    pub element: &'a SizedType<'a>,
    pub len: NonZeroU32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RuntimeSizedArray<'a> {
    pub element: SizedType<'a>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SizedStruct<'a> {
    pub name: &'a str,
    pub fields: &'a [SizedField<'a>],
    pub repr: Repr,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnsizedStruct<'a> {
    pub name: &'a str,
    pub sized_fields: &'a [SizedField<'a>],
    pub last_unsized: RuntimeSizedArrayField<'a>,
    pub repr: Repr,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SizedField<'a> {
    pub name: &'a str,
    pub ty: SizedType<'a>,
    pub custom_min_size: Option<u64>,
    pub custom_min_align: Option<U32PowerOf2>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RuntimeSizedArrayField<'a> {
    pub name: &'a str,
    pub array: RuntimeSizedArray<'a>,
    pub custom_min_align: Option<U32PowerOf2>,
}

// some conversions

impl From<Vector> for SizedType<'_> {
    fn from(value: Vector) -> Self { SizedType::Vector(value) }
}
impl From<Matrix> for SizedType<'_> {
    fn from(value: Matrix) -> Self { SizedType::Matrix(value) }
}
impl From<Atomic> for SizedType<'_> {
    fn from(value: Atomic) -> Self { SizedType::Atomic(value) }
}
impl<'a> From<SizedStruct<'a>> for SizedType<'a> {
    fn from(value: SizedStruct<'a>) -> Self { SizedType::Struct(value) }
}
impl<'a> From<SizedArray<'a>> for SizedType<'a> {
    fn from(value: SizedArray<'a>) -> Self { SizedType::Array(value) }
}
impl<'a> From<RuntimeSizedArray<'a>> for LayoutType<'a> {
    fn from(value: RuntimeSizedArray<'a>) -> Self { LayoutType::RuntimeSizedArray(value) }
}
impl<'a> From<UnsizedStruct<'a>> for LayoutType<'a> {
    fn from(value: UnsizedStruct<'a>) -> Self { LayoutType::UnsizedStruct(value) }
}
impl<'a, T: Into<SizedType<'a>>> From<T> for LayoutType<'a> {
    fn from(value: T) -> Self { LayoutType::Sized(value.into()) }
}

// const conversions

impl Matrix {
    pub const fn to_sized_type(self) -> SizedType<'static> { SizedType::Matrix(self) }
    pub const fn to_layout_type(self) -> LayoutType<'static> { LayoutType::Sized(self.to_sized_type()) }
}
impl Vector {
    pub const fn to_sized_type(self) -> SizedType<'static> { SizedType::Vector(self) }
    pub const fn to_layout_type(self) -> LayoutType<'static> { LayoutType::Sized(self.to_sized_type()) }
}
impl Atomic {
    pub const fn to_sized_type(self) -> SizedType<'static> { SizedType::Atomic(self) }
    pub const fn to_layout_type(self) -> LayoutType<'static> { LayoutType::Sized(self.to_sized_type()) }
}
impl<'a> SizedStruct<'a> {
    pub const fn to_sized_type(self) -> SizedType<'a> { SizedType::Struct(self) }
    pub const fn to_layout_type(self) -> LayoutType<'a> { LayoutType::Sized(self.to_sized_type()) }
}
impl<'a> SizedArray<'a> {
    pub const fn to_sized_type(self) -> SizedType<'a> { SizedType::Array(self) }
    pub const fn to_layout_type(self) -> LayoutType<'a> { LayoutType::Sized(self.to_sized_type()) }
}
impl<'a> RuntimeSizedArray<'a> {
    pub const fn to_layout_type(self) -> LayoutType<'a> { LayoutType::RuntimeSizedArray(self) }
}
impl<'a> UnsizedStruct<'a> {
    pub const fn to_layout_type(self) -> LayoutType<'a> { LayoutType::UnsizedStruct(self) }
}

// some helper constructors

impl<'a> SizedField<'a> {
    pub const fn new(name: &'a str, ty: SizedType<'a>) -> Self {
        Self {
            name,
            ty,
            custom_min_size: None,
            custom_min_align: None,
        }
    }
}

impl<'a> RuntimeSizedArrayField<'a> {
    pub const fn new(name: &'a str, array_element: SizedType<'a>, custom_min_align: Option<U32PowerOf2>) -> Self {
        Self {
            name,
            array: RuntimeSizedArray::new(array_element),
            custom_min_align,
        }
    }
}

impl<'a> RuntimeSizedArray<'a> {
    pub const fn new(element: SizedType<'a>) -> Self { Self { element } }
}

impl<'a> SizedStruct<'a> {
    pub const fn new(name: &'a str, fields: &'a [SizedField<'a>], repr: Repr) -> Self { Self { name, fields, repr } }
}

// borrowing for ir types

impl ir::LayoutType {
    pub fn borrow_with<R, F: FnOnce(LayoutType<'_>) -> R>(&self, f: F) -> R {
        let bump = bumpalo::Bump::new();
        f(self.borrow(&bump))
    }

    pub fn borrow<'a>(&'a self, bump: &'a bumpalo::Bump) -> LayoutType<'a> {
        match self {
            ir::LayoutType::Sized(s) => LayoutType::Sized(s.borrow(bump)),
            ir::LayoutType::UnsizedStruct(s) => LayoutType::UnsizedStruct(s.borrow(bump)),
            ir::LayoutType::RuntimeSizedArray(a) => LayoutType::RuntimeSizedArray(a.borrow(bump)),
        }
    }
}

impl ir::SizedType {
    pub fn borrow_with<R, F: FnOnce(SizedType<'_>) -> R>(&self, f: F) -> R {
        let bump = bumpalo::Bump::new();
        f(self.borrow(&bump))
    }

    pub fn borrow<'a>(&'a self, bump: &'a bumpalo::Bump) -> SizedType<'a> {
        match self {
            ir::SizedType::Vector(v) => SizedType::Vector(*v),
            ir::SizedType::Matrix(m) => SizedType::Matrix(*m),
            ir::SizedType::Atomic(a) => SizedType::Atomic(*a),
            ir::SizedType::Array(a) => SizedType::Array(a.borrow(bump)),
            ir::SizedType::Struct(s) => SizedType::Struct(s.borrow(bump)),
        }
    }
}

impl ir::SizedArray {
    pub fn borrow_with<R, F: FnOnce(SizedArray<'_>) -> R>(&self, f: F) -> R {
        let bump = bumpalo::Bump::new();
        f(self.borrow(&bump))
    }

    pub fn borrow<'a>(&'a self, bump: &'a bumpalo::Bump) -> SizedArray<'a> {
        SizedArray {
            element: bump.alloc(self.element.borrow(bump)),
            len: self.len,
        }
    }
}

impl ir::RuntimeSizedArray {
    pub fn borrow_with<R, F: FnOnce(RuntimeSizedArray<'_>) -> R>(&self, f: F) -> R {
        let bump = bumpalo::Bump::new();
        f(self.borrow(&bump))
    }

    pub fn borrow<'a>(&'a self, bump: &'a bumpalo::Bump) -> RuntimeSizedArray<'a> {
        RuntimeSizedArray {
            element: self.element.borrow(bump),
        }
    }
}

impl ir::SizedField {
    pub fn borrow_with<R, F: FnOnce(SizedField<'_>) -> R>(&self, f: F) -> R {
        let bump = bumpalo::Bump::new();
        f(self.borrow(&bump))
    }

    pub fn borrow<'a>(&'a self, bump: &'a bumpalo::Bump) -> SizedField<'a> {
        SizedField {
            name: &self.name,
            custom_min_size: self.custom_min_size,
            custom_min_align: self.custom_min_align,
            ty: self.ty.borrow(bump),
        }
    }
}

impl ir::RuntimeSizedArrayField {
    pub fn borrow_with<R, F: FnOnce(RuntimeSizedArrayField<'_>) -> R>(&self, f: F) -> R {
        let bump = bumpalo::Bump::new();
        f(self.borrow(&bump))
    }

    pub fn borrow<'a>(&'a self, bump: &'a bumpalo::Bump) -> RuntimeSizedArrayField<'a> {
        RuntimeSizedArrayField {
            name: &self.name,
            custom_min_align: self.custom_min_align,
            array: self.array.borrow(bump),
        }
    }
}

impl ir::SizedStruct {
    pub fn borrow_with<R, F: FnOnce(SizedStruct<'_>) -> R>(&self, f: F) -> R {
        let bump = bumpalo::Bump::new();
        f(self.borrow(&bump))
    }

    pub fn borrow<'a>(&'a self, bump: &'a bumpalo::Bump) -> SizedStruct<'a> {
        let fields =
            bumpalo::collections::Vec::from_iter_in(self.fields.iter().map(|f| f.borrow(bump)), bump).into_bump_slice();
        SizedStruct {
            name: &self.name,
            fields,
            repr: self.repr,
        }
    }
}

impl ir::UnsizedStruct {
    pub fn borrow_with<R, F: FnOnce(UnsizedStruct<'_>) -> R>(&self, f: F) -> R {
        let bump = bumpalo::Bump::new();
        f(self.borrow(&bump))
    }

    pub fn borrow<'a>(&'a self, bump: &'a bumpalo::Bump) -> UnsizedStruct<'a> {
        let sized_fields =
            bumpalo::collections::Vec::from_iter_in(self.sized_fields.iter().map(|f| f.borrow(bump)), bump)
                .into_bump_slice();
        UnsizedStruct {
            name: &self.name,
            sized_fields,
            last_unsized: self.last_unsized.borrow(bump),
            repr: self.repr,
        }
    }
}

// No bools and no atomics checks

impl LayoutType<'_> {
    pub const fn contains_bools(&self) -> bool {
        match self {
            LayoutType::Sized(s) => s.contains_bools(),
            LayoutType::UnsizedStruct(s) => s.contains_bools(),
            LayoutType::RuntimeSizedArray(a) => a.contains_bools(),
        }
    }

    pub const fn contains_atomics(&self) -> bool {
        match self {
            LayoutType::Sized(s) => s.contains_atomics(),
            LayoutType::UnsizedStruct(s) => s.contains_atomics(),
            LayoutType::RuntimeSizedArray(a) => a.contains_atomics(),
        }
    }
}

impl SizedType<'_> {
    pub const fn contains_bools(&self) -> bool {
        match self {
            SizedType::Vector(v) => matches!(v.scalar, ir::ScalarType::Bool),
            SizedType::Matrix(_) => false,
            SizedType::Atomic(_) => false,
            SizedType::Array(a) => a.element.contains_bools(),
            SizedType::Struct(s) => s.contains_bools(),
        }
    }

    pub const fn contains_atomics(&self) -> bool {
        match self {
            SizedType::Vector(_) => false,
            SizedType::Matrix(_) => false,
            SizedType::Atomic(_) => true,
            SizedType::Array(a) => a.element.contains_atomics(),
            SizedType::Struct(s) => s.contains_atomics(),
        }
    }
}

impl SizedStruct<'_> {
    pub const fn contains_bools(&self) -> bool {
        let mut i = 0;
        while i < self.fields.len() {
            if self.fields[i].ty.contains_bools() {
                return true;
            }
            i += 1;
        }
        false
    }

    pub const fn contains_atomics(&self) -> bool {
        let mut i = 0;
        while i < self.fields.len() {
            if self.fields[i].ty.contains_atomics() {
                return true;
            }
            i += 1;
        }
        false
    }
}

impl UnsizedStruct<'_> {
    pub const fn contains_bools(&self) -> bool {
        let mut i = 0;
        while i < self.sized_fields.len() {
            if self.sized_fields[i].ty.contains_bools() {
                return true;
            }
            i += 1;
        }
        self.last_unsized.array.contains_bools()
    }

    pub const fn contains_atomics(&self) -> bool {
        let mut i = 0;
        while i < self.sized_fields.len() {
            if self.sized_fields[i].ty.contains_atomics() {
                return true;
            }
            i += 1;
        }
        self.last_unsized.array.contains_atomics()
    }
}

impl RuntimeSizedArray<'_> {
    pub const fn contains_bools(&self) -> bool { self.element.contains_bools() }

    pub const fn contains_atomics(&self) -> bool { self.element.contains_atomics() }
}
