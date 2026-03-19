//! This module defines types that can be laid out in memory.

use std::{
    fmt::{Display, Formatter},
    num::NonZeroU32,
    rc::Rc,
};

use crate::{
    any::U32PowerOf2,
    ir::{CallInfo, recording::Context},
};

pub(crate) mod align_size;
pub(crate) mod builder;
pub(crate) mod type_layout;

mod canon_name;
mod tensor;

pub use tensor::*;
pub use canon_name::*;

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
    /// Creates a new atomic with the provided scalar type.
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

    /// [`SizedType`] of the elements of the array
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

impl From<ScalarType> for SizedType {
    fn from(value: ScalarType) -> Self { SizedType::Vector(Vector::new(value, Len::X1)) }
}

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

// Struct helpers

/// Enum of sized or unsized struct
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum StructKind {
    /// Sized struct
    Sized(SizedStruct),
    /// Unsized struct
    Unsized(UnsizedStruct),
}

impl StructKind {
    /// New Self
    pub fn new(
        name: impl Into<CanonName>,
        sized_fields: Vec<SizedField>,
        last_unsized: Option<RuntimeSizedArrayField>,
        repr: Repr,
    ) -> Self {
        match last_unsized {
            Some(last_unsized) => UnsizedStruct::new(name, sized_fields, last_unsized, repr).into(),
            None => SizedStruct::new(name, sized_fields, repr).into(),
        }
    }

    /// Makes Self an enum over a reference of a sized or a reference of an unsized strucct
    pub fn as_ref(&self) -> StructKindRef<'_> {
        match self {
            StructKind::Sized(s) => StructKindRef::Sized(s),
            StructKind::Unsized(s) => StructKindRef::Unsized(s),
        }
    }
}

impl std::fmt::Display for StructKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StructKind::Sized(s) => s.fmt(f),
            StructKind::Unsized(s) => s.fmt(f),
        }
    }
}

impl From<StructKind> for LayoutType {
    fn from(s: StructKind) -> Self {
        match s {
            StructKind::Sized(s) => LayoutType::Sized(SizedType::Struct(s)),
            StructKind::Unsized(s) => LayoutType::UnsizedStruct(s),
        }
    }
}
impl From<SizedStruct> for StructKind {
    fn from(value: SizedStruct) -> Self { StructKind::Sized(value) }
}
impl From<UnsizedStruct> for StructKind {
    fn from(value: UnsizedStruct) -> Self { StructKind::Unsized(value) }
}

/// Ref equivalent of [`StructKind`]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StructKindRef<'a> {
    /// Sized struct
    Sized(&'a SizedStruct),
    /// Unsized struct
    Unsized(&'a UnsizedStruct),
}

impl<'a> From<&'a SizedStruct> for StructKindRef<'a> {
    fn from(s: &'a SizedStruct) -> Self { StructKindRef::Sized(s) }
}

impl<'a> From<&'a UnsizedStruct> for StructKindRef<'a> {
    fn from(s: &'a UnsizedStruct) -> Self { StructKindRef::Unsized(s) }
}

impl StructKindRef<'_> {
    /// Clones the inner reference and returns a [`StructKind`]
    pub fn to_owned(&self) -> StructKind {
        match self {
            StructKindRef::Sized(s) => StructKind::Sized((*s).clone()),
            StructKindRef::Unsized(s) => StructKind::Unsized((*s).clone()),
        }
    }

    /// (no documentation yet)
    pub fn name(&self) -> &CanonName {
        match self {
            StructKindRef::Sized(s) => &s.name,
            StructKindRef::Unsized(s) => &s.name,
        }
    }

    /// (no documentation yet)
    pub fn sized_fields(&self) -> &[SizedField] {
        match self {
            StructKindRef::Sized(s) => &s.fields,
            StructKindRef::Unsized(s) => &s.sized_fields,
        }
    }

    /// (no documentation yet)
    pub fn last_unsized(&self) -> Option<&RuntimeSizedArrayField> {
        match self {
            StructKindRef::Sized(_) => None,
            StructKindRef::Unsized(s) => Some(&s.last_unsized),
        }
    }

    /// (no documentation yet)
    pub fn repr(&self) -> Repr {
        match self {
            StructKindRef::Sized(s) => s.repr,
            StructKindRef::Unsized(s) => s.repr,
        }
    }

    /// Copy enum of the struct kind
    pub fn kind(&self) -> StructKindVariant {
        match self {
            StructKindRef::Sized(_) => StructKindVariant::Sized,
            StructKindRef::Unsized(_) => StructKindVariant::Unsized,
        }
    }

    /// (no documentation yet)
    pub fn find_field(&self, name: &CanonName) -> Option<LayoutType> {
        self.sized_fields()
            .iter()
            .find(|f| &f.name == name)
            .map(|f| LayoutType::Sized(f.ty.clone()))
            .or_else(|| {
                self.last_unsized()
                    .filter(|f| &f.name == name)
                    .map(|f| LayoutType::RuntimeSizedArray(f.array.clone()))
            })
    }

    /// (no documentation yet)
    pub fn is_empty(&self) -> bool { self.sized_fields().is_empty() && self.last_unsized().is_none() }

    /// (no documentation yet)
    pub fn field_names(&self) -> impl Iterator<Item = &CanonName> {
        self.sized_fields()
            .iter()
            .map(|f| &f.name)
            .chain(self.last_unsized().as_ref().map(|f| &f.name))
    }
}

impl std::fmt::Display for StructKindRef<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StructKindRef::Sized(s) => s.fmt(f),
            StructKindRef::Unsized(s) => s.fmt(f),
        }
    }
}

/// try register `struct_` if we're currently in a pipeline encoding,
/// otherwise the registration will happen later with a less useful `call_info`
fn try_register_struct(call_info: CallInfo, struct_: StructKindRef<'_>) {
    Context::try_with(call_info, |ctx| {
        ctx.struct_registry_mut().register_mentioned_structs_recursively(
            struct_,
            &mut ctx.pool_mut(),
            ctx.latest_user_caller(),
        );
    });
}

#[doc(hidden)] // internal
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum StructKindVariant {
    Sized,
    Unsized,
}

impl Display for StructKindVariant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            StructKindVariant::Sized => "struct",
            StructKindVariant::Unsized => "unsized struct",
        })
    }
}

/// Enum of layout algorithms.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Repr {
    /// WGSL's layout algorithm
    /// https://www.w3.org/TR/WGSL/#alignment-and-size
    #[default]
    Wgsl,
    /// Modified layout algorithm based on [`Repr::Wgsl`], but with different type
    /// alignments and array strides that make the resulting Layout match wgsl's
    /// uniform address space requirements.
    ///
    /// https://www.w3.org/TR/WGSL/#address-space-layout-constraints
    ///
    /// (matrix strides remain unchanged however, which makes this different from the std140 layout for mat2x2)
    ///
    /// Internally used for checking whether a type can be used in the wgsl's
    /// uniform address space
    WgslUniform,
    /// byte-alignment of everything is 1. Custom alignment attributes
    /// in [`LayoutType`] are unsupported.
    Packed,
}

impl std::fmt::Display for Repr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Repr::Wgsl => write!(f, "wgsl"),
            Repr::WgslUniform => write!(f, "wgsl uniform"),
            Repr::Packed => write!(f, "packed"),
        }
    }
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
