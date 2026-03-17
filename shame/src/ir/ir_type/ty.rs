use self::struct_::SizedStruct;
use crate::{any::layout::Repr, frontend::any::shared_io, ir::recording::MemoryRegion};

use super::*;
use std::{
    fmt::Display,
    num::{NonZeroU32, NonZeroU64},
    rc::Rc,
};

#[derive(Clone, PartialEq, Eq)]
// Types according to the WebGPU type system,
// slightly modified to include
// - `StoreType::BufferBlock` for potential OpenGL/GLSL compatibility
// - `Rc<Allocation>` for tracking memory cell accesses (useful for the stage solver)
#[doc(hidden)] // runtime api
pub enum Type {
    Unit,
    Ptr(Rc<MemoryRegion>, StoreType, AccessMode),
    Ref(Rc<MemoryRegion>, StoreType, AccessMode),
    Store(StoreType),
}

/// types that pointers/reference can point to.
#[doc(hidden)] // runtime api
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum StoreType {
    /// WGSL "creation-fixed-footprint"
    LayoutType(LayoutType),
    Handle(HandleType),
    BindingArray(Rc<StoreType>, Option<NonZeroU32>),
}

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
/// WGSL "creation-fixed-footprint"
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

#[allow(missing_docs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Atomic {
    pub scalar: ScalarTypeInteger,
}

//   Conversions to ScalarType, SizedType and LayoutType   //
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

impl ScalarTypeInteger {
    pub const fn as_scalar_type(self) -> ScalarType {
        match self {
            ScalarTypeInteger::I32 => ScalarType::I32,
            ScalarTypeInteger::U32 => ScalarType::U32,
        }
    }
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

// END conversions


impl SizedArray {
    /// Creates a new `SizedArray` from it's element type and length.
    pub fn new(element_ty: Rc<SizedType>, len: NonZeroU32) -> Self {
        Self {
            element: element_ty,
            len,
        }
    }
}

impl RuntimeSizedArray {
    /// Creates a new `RuntimeSizedArray` from it's element type.
    pub fn new(element_ty: impl Into<SizedType>) -> Self {
        RuntimeSizedArray {
            element: element_ty.into(),
        }
    }
}

/// types that represent handles to resources (Textures and Samplers).
#[doc(hidden)] // runtime api
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum HandleType {
    SampledTexture(TextureShape, TextureSampleUsageType, SamplesPerPixel),
    StorageTexture(TextureShape, TextureFormatWrapper, AccessMode),
    Sampler(shared_io::SamplingMethod),
}

// TODO(chronicl) these are somewhat random here
impl From<SizedType> for StoreType {
    fn from(value: SizedType) -> Self { StoreType::LayoutType(LayoutType::Sized(value)) }
}
impl From<SizedType> for Type {
    fn from(value: SizedType) -> Self { Type::Store(value.into()) }
}
impl From<ScalarType> for Type {
    fn from(value: ScalarType) -> Self {
        Type::Store(StoreType::LayoutType(LayoutType::Sized(SizedType::Vector(Vector {
            scalar: value,
            len: Len::X1,
        }))))
    }
}

impl Type {
    pub fn is_ref(&self) -> bool { matches!(self, Type::Ref { .. }) }
}

impl std::fmt::Debug for Type {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        /// custom debug implementation to hide details of the allocation, and instead print refs and ptrs
        /// as expected from wgsl
        match self {
            Self::Unit => write!(f, "Unit"),
            Self::Ptr(arg0, arg1, arg2) => f.debug_tuple("Ptr").field(&arg0.address_space).field(arg1).field(arg2).finish(),
            Self::Ref(arg0, arg1, arg2) => f.debug_tuple("Ref").field(&arg0.address_space).field(arg1).field(arg2).finish(),
            Self::Store(arg0) => f.debug_tuple("Store").field(arg0).finish(),
        }
    }
}

impl Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mem_view_str = match self {
            Type::Ptr(..) => "Ptr",
            Type::Ref(..) => "Ref",
            _ => "",
        };
        match self {
            Type::Unit => f.write_str("()"),
            Type::Ptr(a, s, am) | Type::Ref(a, s, am) => write!(f, "{mem_view_str}<{s}, {}, {am}>", a.address_space),
            Type::Store(s) => write!(f, "{s}"),
        }
    }
}

impl Display for StoreType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StoreType::LayoutType(layout_type) => match layout_type {
                LayoutType::Sized(s) => write!(f, "{s}"),
                LayoutType::UnsizedStruct(s) => write!(f, "{}", s.name),
                LayoutType::RuntimeSizedArray(a) => write!(f, "Array<{}>", a.element),
            },
            StoreType::Handle(x) => write!(f, "{x}"),
            StoreType::BindingArray(x, _) => write!(f, "BindingArray<{}>", x),
        }
    }
}

impl Display for SizedType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SizedType::Vector(v) => write!(f, "{}{}", v.scalar, v.len),
            SizedType::Matrix(m) => {
                write!(
                    f,
                    "mat<{}, {}, {}>",
                    ScalarType::from(m.scalar),
                    Len::from(m.columns),
                    Len::from(m.rows)
                )
            }
            SizedType::Array(a) => write!(f, "array<{}, {}>", a.element, a.len),
            SizedType::Atomic(a) => write!(f, "atomic<{}>", ScalarType::from(a.scalar)),
            SizedType::Struct(s) => write!(f, "{}", s.name),
        }
    }
}

impl Display for HandleType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HandleType::SampledTexture(shape, fmt, spp) => write!(f, "Texture<{fmt}, {shape}, {spp:?}>"),
            HandleType::StorageTexture(shape, fmt, access) => write!(f, "StorageTexture<{fmt:?}, {shape}, {access}>"),
            HandleType::Sampler(s) => write!(f, "Sampler<{s}>"),
        }
    }
}

#[doc(hidden)] // runtime api
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AlignedType {
    Sized(SizedType),
    RuntimeSizedArray(SizedType),
}

impl AlignedType {
    pub fn align(&self) -> u64 {
        match self {
            // TODO(chronicl) repr
            AlignedType::Sized(sized) => sized.align(Repr::Wgsl).as_u64(),
            AlignedType::RuntimeSizedArray(sized) => sized.align(Repr::Wgsl).as_u64(),
        }
    }
}
