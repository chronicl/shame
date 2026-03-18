//! This module defines types that can be laid out in memory.

use std::{fmt::Formatter, num::NonZeroU32, rc::Rc};

use crate::{
    GpuSized,
    any::{U32PowerOf2, layout::Repr},
    call_info,
    common::prettify::set_color,
    ir::{self, StructureFieldNamesMustBeUnique, recording::Context},
};

pub use crate::ir::{Len, Len2, PackedVector, ScalarTypeFp, ScalarTypeInteger, ScalarType, ir_type::CanonName};

pub(crate) mod align_size;
pub(crate) mod builder;

pub use align_size::{FieldOffsets, MatrixMajor, StructLayoutCalculator, array_size, array_stride, array_align};
pub use builder::{FieldOptions};

/// `TypeLayoutRecipe` describes how a type should be laid out in memory.
///
/// It does not contain any layout information itself, but can be converted to a `TypeLayout`
/// using the `TypeLayoutRecipe::layout` method.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LayoutType {
    /// A type with a known size.
    Sized(SizedType),
    /// A struct with a runtime sized array as it's last field.
    UnsizedStruct(UnsizedStruct),
    /// An array whose size is determined at runtime.
    RuntimeSizedArray(RuntimeSizedArray),
}

/// Types that have a size which is known at shader creation time.
#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SizedType {
    Vector(Vector),
    Matrix(Matrix),
    Array(SizedArray),
    Atomic(Atomic),
    Struct(SizedStruct),
}

#[allow(missing_docs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Vector {
    pub scalar: ScalarType,
    pub len: Len,
}

#[allow(missing_docs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Matrix {
    pub scalar: ScalarTypeFp,
    pub columns: Len2,
    pub rows: Len2,
}

#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SizedArray {
    pub element: Rc<SizedType>,
    pub len: NonZeroU32,
}

impl SizedArray {
    /// Creates a new `SizedArray` from it's element type and length.
    pub fn new(element_ty: Rc<SizedType>, len: NonZeroU32) -> Self {
        Self {
            element: element_ty,
            len,
        }
    }
}

#[allow(missing_docs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Atomic {
    pub scalar: ScalarTypeInteger,
}

impl Atomic {
    pub fn new(scalar: ScalarTypeInteger) -> Self { Self { scalar } }
}

#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RuntimeSizedArray {
    pub element: SizedType,
}

impl RuntimeSizedArray {
    /// Creates a new `RuntimeSizedArray` from it's element type.
    pub fn new(element_ty: impl Into<SizedType>) -> Self {
        RuntimeSizedArray {
            element: element_ty.into(),
        }
    }
}

/// A struct with a known fixed size.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SizedStruct {
    /// The canonical name of the struct.
    pub name: CanonName,
    /// The fields of the sized struct. May be empty, however using an empty sized struct
    /// for wgsl code generation will lead to an encoding error.
    pub fields: Vec<SizedField>,
    /// The representation/layout rules for this struct. See [`Repr`] for more details.
    pub repr: Repr,
}

/// A struct whose size is not known before shader runtime.
///
/// This struct has a runtime sized array as it's last field.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct UnsizedStruct {
    /// The canonical name of the struct.
    pub name: CanonName,
    /// Fixed-size fields that come before the unsized field
    pub sized_fields: Vec<SizedField>,
    /// Last runtime sized array field of the struct.
    pub last_unsized: RuntimeSizedArrayField,
    /// The representation/layout rules for this struct. See [`Repr`] for more details.
    pub repr: Repr,
}

#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SizedField {
    pub name: CanonName,
    pub custom_min_size: Option<u64>,
    pub custom_min_align: Option<U32PowerOf2>,
    pub ty: SizedType,
}

impl SizedField {
    /// Creates a new `SizedField`.
    pub fn new(options: impl Into<FieldOptions>, ty: impl Into<SizedType>) -> Self {
        let options = options.into();
        Self {
            name: options.name,
            custom_min_size: options.custom_min_size,
            custom_min_align: options.custom_min_align,
            ty: ty.into(),
        }
    }
}

#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RuntimeSizedArrayField {
    pub name: CanonName,
    pub custom_min_align: Option<U32PowerOf2>,
    pub array: RuntimeSizedArray,
}

impl RuntimeSizedArrayField {
    /// Creates a new `RuntimeSizedArrayField` given it's field name,
    /// an optional custom minimum align and it's element type.
    pub fn new(
        name: impl Into<CanonName>,
        custom_min_align: Option<U32PowerOf2>,
        element_ty: impl Into<SizedType>,
    ) -> Self {
        Self {
            name: name.into(),
            custom_min_align,
            array: RuntimeSizedArray {
                element: element_ty.into(),
            },
        }
    }

    pub fn element_ty(&self) -> &SizedType { &self.array.element }
}

//   Conversions to ScalarType, SizedType and TypeLayoutRecipe   //

macro_rules! impl_into_sized_type {
    ($($ty:ident -> $variant:path),*) => {
       $(
           impl From<$ty> for SizedType {
               fn from(v: $ty) -> Self { $variant(v) }
           }
       )*
    };
}

impl_into_sized_type!(
    Vector       -> SizedType::Vector,
    Matrix       -> SizedType::Matrix,
    SizedArray   -> SizedType::Array,
    Atomic       -> SizedType::Atomic,
    SizedStruct  -> SizedType::Struct
);

impl<T> From<T> for LayoutType
where
    SizedType: From<T>,
{
    fn from(value: T) -> Self { LayoutType::Sized(SizedType::from(value)) }
}

impl From<UnsizedStruct> for LayoutType {
    fn from(s: UnsizedStruct) -> Self { LayoutType::UnsizedStruct(s) }
}
impl From<RuntimeSizedArray> for LayoutType {
    fn from(a: RuntimeSizedArray) -> Self { LayoutType::RuntimeSizedArray(a) }
}

// Display impls

impl std::fmt::Display for LayoutType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            LayoutType::Sized(s) => s.fmt(f),
            LayoutType::RuntimeSizedArray(a) => a.fmt(f),
            LayoutType::UnsizedStruct(s) => s.fmt(f),
        }
    }
}

impl std::fmt::Display for SizedType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            SizedType::Vector(v) => v.fmt(f),
            SizedType::Matrix(m) => m.fmt(f),
            SizedType::Array(a) => a.fmt(f),
            SizedType::Atomic(a) => a.fmt(f),
            SizedType::Struct(s) => s.fmt(f),
        }
    }
}

impl std::fmt::Display for Vector {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result { write!(f, "{}{}", self.scalar, self.len) }
}

impl std::fmt::Display for Matrix {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "mat<{}, {}, {}>",
            self.scalar,
            Len::from(self.columns),
            Len::from(self.rows)
        )
    }
}

impl std::fmt::Display for SizedArray {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result { write!(f, "array<{}, {}>", &*self.element, self.len) }
}

impl std::fmt::Display for Atomic {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result { write!(f, "atomic<{}>", ScalarType::from(self.scalar)) }
}

impl std::fmt::Display for SizedStruct {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result { write!(f, "{}", self.name) }
}

impl std::fmt::Display for UnsizedStruct {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result { write!(f, "{}", self.name) }
}

impl std::fmt::Display for RuntimeSizedArray {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result { write!(f, "array<{}>", self.element) }
}
