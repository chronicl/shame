#![allow(missing_docs)]
use std::{num::NonZeroU32, rc::Rc};

use crate::{
    any::U32PowerOf2,
    ir::{self, ir_type::BufferBlockDefinitionError, StructureFieldNamesMustBeUnique},
    GpuSized, GpuType,
};

// TODO(chronicl)
// - Consider moving this module into ir_type?
// - We borrow these types from `StoreType` currently. Maybe it would be better the other
//   way around - `StoreType` should borrow from `HostShareableType`.
pub use crate::ir::{Len, Len2, ScalarTypeFp, ScalarTypeInteger, ir_type::CanonName};

mod builder;
mod layout;

pub use layout::*;
pub use builder::*;

use super::FieldOptions;

/// This reprsents a wgsl spec compliant host-shareable type with the addition
/// that f64 is a supported scalar type.
///
/// https://www.w3.org/TR/WGSL/#host-shareable-types
#[derive(Debug, Clone)]
pub enum HostShareableType {
    Sized(SizedType),
    UnsizedStruct(UnsizedStruct),
    RuntimeSizedArray(RuntimeSizedArray),
}

#[derive(Debug, Clone)]
pub enum SizedType {
    Vector(Vector),
    Matrix(Matrix),
    Array(SizedArray),
    Atomic(Atomic),
    Struct(SizedStruct),
}

#[derive(Debug, Clone)]
pub struct Vector {
    pub scalar: ScalarType,
    pub len: Len,
}

#[derive(Debug, Clone)]
pub struct Matrix {
    pub scalar: ScalarTypeFp,
    pub columns: Len2,
    pub rows: Len2,
}

#[derive(Debug, Clone)]
pub struct SizedArray {
    pub element: Rc<SizedType>,
    pub len: NonZeroU32,
}

#[derive(Debug, Clone, Copy)]
pub struct Atomic {
    pub scalar: ScalarTypeInteger,
}

#[derive(Debug, Clone)]
pub struct RuntimeSizedArray {
    pub element: SizedType,
}

/// Same as `ir::ScalarType`, but without `ScalarType::Bool`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScalarType {
    F16,
    F32,
    U32,
    I32,
    F64,
}

#[derive(Debug, Clone)]
pub struct SizedStruct {
    pub name: CanonName,
    // This is private to ensure a `SizedStruct` always has at least one field.
    fields: Vec<SizedField>,
}

#[derive(Debug, Clone)]
pub struct UnsizedStruct {
    pub name: CanonName,
    pub sized_fields: Vec<SizedField>,
    pub last_unsized: RuntimeSizedArrayField,
}

#[derive(Debug, Clone)]
pub struct SizedField {
    pub name: CanonName,
    pub custom_min_size: Option<u64>,
    pub custom_min_align: Option<U32PowerOf2>,
    pub ty: SizedType,
}

#[derive(Debug, Clone)]
pub struct RuntimeSizedArrayField {
    pub name: CanonName,
    pub custom_min_align: Option<U32PowerOf2>,
    pub array: RuntimeSizedArray,
}

impl SizedField {
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

impl RuntimeSizedArrayField {
    pub fn new(name: impl Into<CanonName>, custom_min_align: Option<U32PowerOf2>, ty: impl Into<SizedType>) -> Self {
        Self {
            name: name.into(),
            custom_min_align,
            array: RuntimeSizedArray { element: ty.into() },
        }
    }
}

/// Used in the `GpuLayout` derive macro to ensure that the last field of a struct is
/// not an `UnsizedStruct`. Auto implemented for all `GpuSized` types and manually implemented
/// for runtime sized arrays.
pub trait HostshareableSizedOrArray {
    fn sized_or_array() -> SizedOrArray;
}

pub enum SizedOrArray {
    Sized(SizedType),
    RuntimeSizedArray(SizedType),
}

impl<T: GpuSized + GpuType> HostshareableSizedOrArray for T {
    fn sized_or_array() -> SizedOrArray { SizedOrArray::Sized(Self::host_shareable_sized()) }
}

impl std::fmt::Display for ScalarType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            ScalarType::F16 => "f16",
            ScalarType::F32 => "f32",
            ScalarType::F64 => "f64",
            ScalarType::U32 => "u32",
            ScalarType::I32 => "i32",
        })
    }
}

//   Conversions to ScalarType, SizedType and HostshareableType   //

macro_rules! impl_into_sized_type {
    ($($ty:ident -> $variant:path),*) => {
       $(
           impl $ty {
               /// Const conversion to [`SizedType`]
               pub const fn into_sized_type(self) -> SizedType { $variant(self) }
               /// Const conversion to [`HostshareableType`]
               pub const fn into_hostshareable(self) -> HostShareableType {
                   HostShareableType::Sized(self.into_sized_type())
               }
           }

           impl From<$ty> for SizedType {
               fn from(v: $ty) -> Self { v.into_sized_type() }
           }
       )*
    };
}
impl_into_sized_type!(
    Vector      -> SizedType::Vector,
    Matrix      -> SizedType::Matrix,
    SizedArray  -> SizedType::Array,
    Atomic      -> SizedType::Atomic,
    SizedStruct -> SizedType::Struct
);

impl<T> From<T> for HostShareableType
where
    SizedType: From<T>,
{
    fn from(value: T) -> Self { HostShareableType::Sized(SizedType::from(value)) }
}

impl From<UnsizedStruct> for HostShareableType {
    fn from(s: UnsizedStruct) -> Self { HostShareableType::UnsizedStruct(s) }
}
impl From<RuntimeSizedArray> for HostShareableType {
    fn from(a: RuntimeSizedArray) -> Self { HostShareableType::RuntimeSizedArray(a) }
}

impl ScalarTypeInteger {
    pub const fn as_scalar_type(self) -> ScalarType {
        match self {
            ScalarTypeInteger::I32 => ScalarType::I32,
            ScalarTypeInteger::U32 => ScalarType::U32,
        }
    }
}
impl From<ScalarTypeInteger> for ScalarType {
    fn from(int: ScalarTypeInteger) -> Self { int.as_scalar_type() }
}
impl ScalarTypeFp {
    pub const fn as_scalar_type(self) -> ScalarType {
        match self {
            ScalarTypeFp::F16 => ScalarType::F16,
            ScalarTypeFp::F32 => ScalarType::F32,
            ScalarTypeFp::F64 => ScalarType::F64,
        }
    }
}
impl From<ScalarTypeFp> for ScalarType {
    fn from(int: ScalarTypeFp) -> Self { int.as_scalar_type() }
}


//     Conversions from/to ir types     //
impl ir::ScalarType {
    pub fn as_host_shareable_unchecked(self) -> ScalarType {
        match self {
            Self::F16 => ScalarType::F16,
            Self::F32 => ScalarType::F32,
            Self::F64 => ScalarType::F64,
            Self::U32 => ScalarType::U32,
            Self::I32 => ScalarType::I32,
            Self::Bool => panic!("ScalarType was bool"),
        }
    }
}

impl From<HostShareableType> for ir::StoreType {
    fn from(host: HostShareableType) -> Self {
        match host {
            HostShareableType::Sized(s) => ir::StoreType::Sized(s.into()),
            HostShareableType::RuntimeSizedArray(s) => ir::StoreType::RuntimeSizedArray(s.element.into()),
            HostShareableType::UnsizedStruct(s) => ir::StoreType::BufferBlock(s.into()),
        }
    }
}

impl From<SizedType> for ir::SizedType {
    fn from(host: SizedType) -> Self {
        match host {
            SizedType::Vector(v) => ir::SizedType::Vector(v.len, v.scalar.into()),
            SizedType::Matrix(m) => ir::SizedType::Matrix(m.columns, m.rows, m.scalar),
            SizedType::Array(a) => ir::SizedType::Array(Rc::new(Rc::unwrap_or_clone(a.element).into()), a.len),
            SizedType::Atomic(i) => ir::SizedType::Atomic(i.scalar),
            SizedType::Struct(s) => ir::SizedType::Structure(s.into()),
        }
    }
}

impl From<ScalarType> for ir::ScalarType {
    fn from(scalar_type: ScalarType) -> Self {
        match scalar_type {
            ScalarType::F16 => ir::ScalarType::F16,
            ScalarType::F32 => ir::ScalarType::F32,
            ScalarType::F64 => ir::ScalarType::F64,
            ScalarType::U32 => ir::ScalarType::U32,
            ScalarType::I32 => ir::ScalarType::I32,
        }
    }
}

impl From<SizedStruct> for ir::ir_type::SizedStruct {
    fn from(host: SizedStruct) -> Self {
        let mut fields: Vec<ir::ir_type::SizedField> = host.fields.into_iter().map(Into::into).collect();
        // has at least one field
        let last_field = fields.pop().unwrap();

        // Note: This might throw an error in real usage if the fields aren't valid
        // We're assuming they're valid in the conversion
        match ir::ir_type::SizedStruct::new_nonempty(host.name, fields, last_field) {
            Ok(s) => s,
            Err(StructureFieldNamesMustBeUnique) => {
                unreachable!("field names are unique for `hostshareable::UnsizedStruct`")
            }
        }
    }
}

impl From<UnsizedStruct> for ir::ir_type::BufferBlock {
    fn from(host: UnsizedStruct) -> Self {
        let sized_fields: Vec<ir::ir_type::SizedField> = host.sized_fields.into_iter().map(Into::into).collect();

        let last_unsized = host.last_unsized.into();

        // Note: This might throw an error in real usage if the struct isn't valid
        // We're assuming it's valid in the conversion
        match ir::ir_type::BufferBlock::new(host.name, sized_fields, Some(last_unsized)) {
            Ok(b) => b,
            Err(BufferBlockDefinitionError::FieldNamesMustBeUnique) => {
                unreachable!("`hostshareable::UnsizedStruct` field names are unique.")
            }
            Err(BufferBlockDefinitionError::MustHaveAtLeastOneField) => {
                unreachable!("`hostshareable::UnsizedStruct` has at least one field.")
            }
        }
    }
}

impl From<SizedField> for ir::ir_type::SizedField {
    fn from(host: SizedField) -> Self {
        ir::ir_type::SizedField::new(host.name, host.custom_min_size, host.custom_min_align, host.ty.into())
    }
}

impl From<RuntimeSizedArrayField> for ir::ir_type::RuntimeSizedArrayField {
    fn from(host: RuntimeSizedArrayField) -> Self {
        ir::ir_type::RuntimeSizedArrayField::new(host.name, host.custom_min_align, host.array.element.into())
    }
}
