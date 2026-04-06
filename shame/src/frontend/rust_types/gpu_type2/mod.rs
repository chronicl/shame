#![allow(missing_docs)]

use std::{cell::RefCell, num::NonZeroU32, rc::Rc, sync::Arc};

use crate::{
    U32PowerOf2,
    ir::{self, Atomic, Len, Matrix, Repr, Vector},
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

// Some dummys required by the GpuLayout derive

impl SizedType<'static> {
    /// Used by the GpuLayout derive.
    pub const DUMMY: Self = SizedType::Vector(Vector {
        scalar: ir::ScalarType::U32,
        len: Len::X1,
    });
}
impl RuntimeSizedArray<'static> {
    /// Used by the GpuLayout derive.
    pub const DUMMY: Self = Self::new(SizedType::DUMMY);
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
impl<'a> SizedType<'a> {
    pub const fn to_layout_type(self) -> LayoutType<'a> { LayoutType::Sized(self) }
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

impl<'a> SizedArray<'a> {
    pub const fn new(element: &'a SizedType<'a>, len: NonZeroU32) -> Self { Self { element, len } }
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

// to ir conversions

impl From<LayoutType<'_>> for ir::LayoutType {
    fn from(value: LayoutType<'_>) -> Self {
        match value {
            LayoutType::Sized(s) => ir::LayoutType::Sized(s.into()),
            LayoutType::UnsizedStruct(s) => ir::LayoutType::UnsizedStruct(s.into()),
            LayoutType::RuntimeSizedArray(a) => ir::LayoutType::RuntimeSizedArray(a.into()),
        }
    }
}

impl From<SizedType<'_>> for ir::SizedType {
    fn from(value: SizedType<'_>) -> Self {
        match value {
            SizedType::Vector(v) => ir::SizedType::Vector(v),
            SizedType::Matrix(m) => ir::SizedType::Matrix(m),
            SizedType::Atomic(a) => ir::SizedType::Atomic(a),
            SizedType::Array(a) => ir::SizedType::Array(a.into()),
            SizedType::Struct(s) => ir::SizedType::Struct(s.into()),
        }
    }
}

impl From<SizedArray<'_>> for ir::SizedArray {
    fn from(value: SizedArray<'_>) -> Self {
        ir::SizedArray {
            element: Rc::new((*value.element).into()),
            len: value.len,
        }
    }
}

impl From<RuntimeSizedArray<'_>> for ir::RuntimeSizedArray {
    fn from(value: RuntimeSizedArray<'_>) -> Self {
        ir::RuntimeSizedArray {
            element: value.element.into(),
        }
    }
}

impl From<SizedField<'_>> for ir::SizedField {
    fn from(value: SizedField<'_>) -> Self {
        ir::SizedField {
            name: value.name.to_owned().into(),
            ty: value.ty.into(),
            custom_min_size: value.custom_min_size,
            custom_min_align: value.custom_min_align,
        }
    }
}

impl From<RuntimeSizedArrayField<'_>> for ir::RuntimeSizedArrayField {
    fn from(value: RuntimeSizedArrayField<'_>) -> Self {
        ir::RuntimeSizedArrayField {
            name: value.name.to_owned().into(),
            array: value.array.into(),
            custom_min_align: value.custom_min_align,
        }
    }
}

impl From<SizedStruct<'_>> for ir::SizedStruct {
    fn from(value: SizedStruct<'_>) -> Self {
        ir::SizedStruct {
            name: value.name.to_owned().into(),
            fields: value.fields.iter().map(|f| (*f).into()).collect(),
            repr: value.repr,
        }
    }
}

impl From<UnsizedStruct<'_>> for ir::UnsizedStruct {
    fn from(value: UnsizedStruct<'_>) -> Self {
        ir::UnsizedStruct {
            name: value.name.to_owned().into(),
            sized_fields: value.sized_fields.iter().map(|f| (*f).into()).collect(),
            last_unsized: value.last_unsized.into(),
            repr: value.repr,
        }
    }
}

// No bools and no atomics checks

impl LayoutType<'_> {
    pub const fn is_sized(&self) -> bool {
        match self {
            LayoutType::Sized(s) => true,
            LayoutType::UnsizedStruct(_) | LayoutType::RuntimeSizedArray(_) => false,
        }
    }

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
